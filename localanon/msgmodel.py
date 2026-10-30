"""msgmodel: the ONNX model of the message filter on onnxruntime +
tokenizers.

No torch and no transformers. The model directory (``models/msgfilter``):
``model.onnx`` (int8 or fp32 — by ``precision_cpu`` of the index.json entry, without it — by
``cpu.precision`` in ``export_report.json``; with ``fp32`` a quantized file is not used, see ``cpu_file``),
``tokenizer.json``, ``labels.json`` (labels by output index), ``meta.json`` (max_len, threshold,
metrics). Empty ``model_dir`` — ``models/msgfilter`` in the repository root (unpacked there from the
models archive by run.sh / run.ps1), then ``localanon/models/msgfilter``.

The input format is exactly as in training: a pair (text, context), the context is the previous
messages, nearest first, joined with " | ", empty — "(без контекста)"; longest_first truncation to
max_len. Runs in batches.

Hardware: ``predict(..., plan=hw_plan.stages["msgmodel"])`` or ``device="cuda:0"``.
The provider follows the device: CUDA (``device_id``), ROCm/MIGraphX, DirectML, otherwise CPU; on a
GPU only the GPU file of the tier is used (``files.gpu``, ``model.fp16.onnx`` by default): without it
the tier runs on CPU (int8 is not put on a GPU). The provider failed — CPU with ``ctx.note``. More
than one slot and at least ``POOL_MIN`` texts — through ``localanon.devpool`` (a process per slot,
batches in order — the same labels as one process). Tiers — by ``index.json`` (``tiers[].dir``,
``files.cpu``/``files.gpu``); without it — the directory itself. Without the new arguments the path
is the old one: CPU, ``model.onnx``, batches of ``BATCH``.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

MODEL_FILE = "model.onnx"
FP16_FILE = "model.fp16.onnx"
FP32_FILE = "model_fp32.onnx"          # export_onnx --keep-fp32
REPORT_FILE = "export_report.json"
# op_type of exported int8 graphs: dynamic (MatMulInteger, DynamicQuantizeLinear), QDQ, QLinear*
QUANT_OPS = (b"MatMulInteger", b"ConvInteger", b"QuantizeLinear", b"QLinear", b"DynamicQuantize")
INDEX_FILE = "index.json"
TOKENIZER_FILE = "tokenizer.json"
LABELS_FILE = "labels.json"
META_FILE = "meta.json"
DEFAULT_MAX_LEN = 256
BATCH = 32
POOL_MIN = 4096            # texts from which several processes pay off the model loading
CTX_SEP = " | "
NO_CTX = "(без контекста)"
_WS = re.compile(r"\s+")
_SESSIONS: dict = {}


class ProviderUnavailable(RuntimeError):
    """The device provider failed to start (no onnxruntime-gpu, driver, GPU)."""


def pair(text: str, context: list[str] | None) -> tuple[str, str]:
    """Mirror of the training input pair (train/common.py::pair) — change only together."""
    t = _WS.sub(" ", (text or "")).strip()
    parts = [_WS.sub(" ", c).strip() for c in reversed(context or []) if c and c.strip()]
    return t, (CTX_SEP.join(parts) if parts else NO_CTX)


# Repository root: run.sh / run.ps1 install the package in editable mode, so this is the clone itself and
# its models/ (unpacked from the models archive) is found without copying.
REPO_ROOT = Path(__file__).resolve().parents[1]


def default_dirs() -> list[Path]:
    return [REPO_ROOT / "models" / "msgfilter", Path(__file__).resolve().parent / "models" / "msgfilter"]


def _index(root: Path) -> dict | None:
    p = root / INDEX_FILE
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return data if isinstance(data, dict) and data.get("tiers") else None


def _root(model_dir: str | Path | None) -> Path:
    if model_dir:
        return Path(model_dir)
    for d in default_dirs():
        if (d / MODEL_FILE).is_file() or (d / INDEX_FILE).is_file():
            return d
    return default_dirs()[0]


def _tier_entry(model_dir, tier: str | None) -> tuple[Path, dict]:
    """→ (tier directory, its index.json entry; without index.json — {}). The tier is not named (or
    "default") and the root has its own model.onnx — the root itself; otherwise — a tier from index.json (not named —
    ``default``)."""
    root = _root(model_dir)
    if tier in (None, "", "default") and (root / MODEL_FILE).is_file():
        return root, {}
    idx = _index(root)
    if idx is None:
        return root, {}
    name = tier if tier not in (None, "", "default") else (idx.get("default") or "")
    for t in idx["tiers"]:
        if t.get("name") == name:
            return root / str(t.get("dir") or t["name"]), dict(t)
    if tier not in (None, "", "default"):
        raise FileNotFoundError(f"tier \"{tier}\" is not in {INDEX_FILE}")
    t = idx["tiers"][0]
    return root / str(t.get("dir") or t.get("name", "")), dict(t)


def _tier(model_dir, tier: str | None) -> tuple[Path, dict]:
    """→ (tier directory, its files)."""
    d, entry = _tier_entry(model_dir, tier)
    return d, dict(entry.get("files") or {})


def cpu_precision(d: Path, entry: dict) -> str:
    """Precision of the tier's CPU file by the export manifest: ``precision_cpu`` of the index.json entry,
    without it — ``cpu.precision`` from ``export_report.json`` in the tier directory; unknown — ""."""
    p = entry.get("precision_cpu")
    if not p:
        try:
            p = json.loads((d / REPORT_FILE).read_text(encoding="utf-8"))["cpu"]["precision"]
        except (OSError, ValueError, KeyError, TypeError):
            p = ""
    return str(p or "")


def _quantized(path: Path) -> bool:
    """The graph has int8 operations (dynamic or QDQ quantization) — by op_type names in the file bytes
    (the onnx package may be missing). External data (``*.data``) holds only weights, it is not read."""
    tail = max(len(x) for x in QUANT_OPS)
    prev = b""
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            buf = prev + chunk
            if any(op in buf for op in QUANT_OPS):
                return True
            prev = buf[-tail:]
    return False


def cpu_file(model_dir: str | Path | None, tier: str | None = None) -> tuple[Path | None, str]:
    """→ (CPU file of the tier, precision by the manifest). Usually ``files.cpu`` (``model.onnx`` by default).
    The manifest says ``fp32`` — int8 is not used: of ``files.cpu``, ``model_fp32.onnx`` (export --keep-fp32)
    and ``model.onnx`` — the first existing one without int8 operations; none — (None, "fp32")."""
    d, entry = _tier_entry(model_dir, tier)
    prec = cpu_precision(d, entry)
    first = d / str((entry.get("files") or {}).get("cpu") or MODEL_FILE)
    if prec != "fp32":
        return first, prec
    for p in dict.fromkeys([first, d / FP32_FILE, d / MODEL_FILE]):
        if p.is_file() and not _quantized(p):
            return p, prec
    return None, prec


def resolve_dir(model_dir: str | Path | None, tier: str | None = None) -> Path:
    return _tier(model_dir, tier)[0]


def gpu_file(model_dir: str | Path | None, tier: str | None = None) -> Path | None:
    """GPU file of the tier (``files.gpu``, ``model.fp16.onnx`` by default) if it exists.
    None otherwise: the CPU file (int8) is not put on a GPU, the tier runs on CPU."""
    try:
        d, files = _tier(model_dir, tier)
    except FileNotFoundError:
        return None
    p = d / str(files.get("gpu") or FP16_FILE)
    return p if p.is_file() else None


def available(model_dir: str, tier: str | None = None) -> bool:
    try:
        d, _ = _tier(model_dir, tier)
        cpu, _ = cpu_file(model_dir, tier)
    except FileNotFoundError:
        return False
    if cpu is None or not all(p.is_file() for p in (cpu, d / TOKENIZER_FILE, d / LABELS_FILE)):
        return False
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


_PROVIDERS = {"cuda": ["CUDAExecutionProvider"], "rocm": ["ROCMExecutionProvider", "MIGraphXExecutionProvider"],
              "dml": ["DmlExecutionProvider"]}


def providers_for(device: str, available_providers: list[str], gpu_mem_limit_mb: int = 0) -> list:
    """onnxruntime providers for a device (``cpu``, ``cuda:N``, ``rocm:N``, ``dml:N``);
    the build has no such provider — []."""
    if device == "cpu":
        return ["CPUExecutionProvider"]
    kind, _, num = device.partition(":")
    dev_id = int(num or 0)
    for name in _PROVIDERS.get(kind, []):
        if name in available_providers:
            opts = {"device_id": dev_id}
            if name == "CUDAExecutionProvider":
                opts["arena_extend_strategy"] = "kSameAsRequested"
                if gpu_mem_limit_mb:
                    opts["gpu_mem_limit"] = int(gpu_mem_limit_mb) << 20
            return [(name, opts), "CPUExecutionProvider"]
    return []


_PRELOADED: list[bool] = []


def _preload_gpu_libs(ort) -> None:
    """onnxruntime-gpu from pip looks for CUDA/cuDNN in the system; libraries from the pip packages
    ``nvidia-*`` are loaded only by ``preload_dlls`` (ORT ≥ 1.21). Once per process, a failure is harmless."""
    if _PRELOADED:
        return
    _PRELOADED.append(True)
    fn = getattr(ort, "preload_dlls", None)
    if fn is not None:
        try:
            fn()
        except Exception:  # noqa: BLE001 — no nvidia-* packages — so CUDA is the system one
            pass


class _Loaded:
    def __init__(self, d: Path, device: str = "cpu", threads: int = 0, gpu_mem_limit_mb: int = 0,
                 files: dict | None = None, batch: int = BATCH, cpu_model: Path | None = None,
                 precision: str = ""):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.labels = json.loads((d / LABELS_FILE).read_text(encoding="utf-8"))
        meta = {}
        if (d / META_FILE).is_file():
            meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        self.max_len = int(meta.get("max_len", DEFAULT_MAX_LEN))
        self.tok = Tokenizer.from_file(str(d / TOKENIZER_FILE))
        self.tok.no_padding()
        self.tok.enable_truncation(self.max_len, strategy="longest_first")
        pad = meta.get("pad_token", "[PAD]")
        self.pad_id = self.tok.token_to_id(pad)
        if self.pad_id is None:
            self.pad_id = 0
        so = ort.SessionOptions()
        so.log_severity_level = 3
        files = files or {}
        self.device = device
        self.batch = max(1, int(batch or BATCH))
        model = cpu_model or d / files.get("cpu", MODEL_FILE)
        self.precision = precision if device == "cpu" else "fp16"
        if device == "cpu":
            if threads:
                so.intra_op_num_threads = int(threads)
                so.inter_op_num_threads = 1
            providers = ["CPUExecutionProvider"]
        else:
            gpu = d / str(files.get("gpu") or FP16_FILE)
            if not gpu.is_file():
                raise ProviderUnavailable(f"no GPU file ({gpu.name}) — the int8 model is not put on a GPU")
            model = gpu
            if device.startswith("cuda"):
                _preload_gpu_libs(ort)
            providers = providers_for(device, ort.get_available_providers(), gpu_mem_limit_mb)
            if not providers:
                raise ProviderUnavailable(f"onnxruntime has no provider for {device.split(':')[0]}")
            if device.startswith("dml"):
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.model_file = model.name
        try:
            self.sess = ort.InferenceSession(str(model), so, providers=providers)
        except Exception as e:  # noqa: BLE001 — on a GPU: the provider failed
            if device == "cpu":
                raise
            raise ProviderUnavailable(f"session on {device} not created ({type(e).__name__})") from e
        if device != "cpu":
            got = self.sess.get_providers()
            want = providers[0][0]
            if not got or got[0] != want:
                raise ProviderUnavailable(f"{want} failed to start")
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def logits(self, texts: list[str], contexts: list[list[str]]) -> list[list[float]]:
        """Raw logits per message, in the order of ``self.labels``."""
        import numpy as np

        out: list[list[float]] = []
        b = self.batch
        for s in range(0, len(texts), b):
            chunk = [pair(t, c) for t, c in zip(texts[s:s + b], contexts[s:s + b])]
            encs = self.tok.encode_batch(chunk)
            n = max(len(e.ids) for e in encs)
            ids = np.full((len(encs), n), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(encs), n), dtype=np.int64)
            tt = np.zeros((len(encs), n), dtype=np.int64)
            for i, e in enumerate(encs):
                k = len(e.ids)
                ids[i, :k] = e.ids
                mask[i, :k] = 1
                tt[i, :k] = e.type_ids
            feed = {}
            for name in self.inputs:
                if name == "input_ids":
                    feed[name] = ids
                elif name == "attention_mask":
                    feed[name] = mask
                elif name == "token_type_ids":
                    feed[name] = tt
            logits = self.sess.run(None, feed)[0]
            out.extend([float(x) for x in row] for row in logits)
        return out

    def run(self, texts: list[str], contexts: list[list[str]]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for row in self.logits(texts, contexts):
            probs = softmax(row)
            j = max(range(len(row)), key=lambda q: row[q])
            out.append((self.labels[j], probs[j]))
        return out


def softmax(row, temperature: float = 1.0) -> list[float]:
    """softmax(row / temperature) in plain Python (row — one message)."""
    t = float(temperature) if temperature and temperature > 0 else 1.0
    z = [float(x) / t for x in row]
    m = max(z)
    exps = [math.exp(x - m) for x in z]
    s = sum(exps)
    return [e / s for e in exps]


def _load(model_dir: str, tier: str | None = None, device: str = "cpu", threads: int = 0,
          gpu_mem_limit_mb: int = 0, batch: int = BATCH) -> _Loaded:
    d, files = _tier(model_dir, tier)
    d = d.resolve()
    key = (str(d), device, int(threads or 0), int(gpu_mem_limit_mb or 0), int(batch or BATCH))
    if key not in _SESSIONS:
        cpu, prec = cpu_file(model_dir, tier) if device == "cpu" else (None, "")
        if device == "cpu" and cpu is None:
            raise FileNotFoundError(_FP32_MISSING)
        _SESSIONS[key] = _Loaded(d, device, threads, gpu_mem_limit_mb, files, batch, cpu, prec)
    return _SESSIONS[key]


_FP32_MISSING = ("the tier is declared fp32 (precision_cpu in the export manifest), but the CPU file is "
                 "quantized (int8) — int8 is not used; export the tier again")


def _pool_init(model_dir: str, tier, slot):
    """devpool worker initializer: the model loads once per slot (a failure on a GPU —
    devpool moves the slot to CPU)."""
    return _load(model_dir, tier, slot.device, slot.threads if slot.device == "cpu" else 0,
                 slot.gpu_mem_limit_mb, slot.batch_size or BATCH)


def _pair_len(item) -> int:
    """Length of a (message, context) pool item — the weight that groups messages of one length."""
    text, context = item
    return len(text or "") + sum(len(c or "") for c in context)


def _pool_run(state: _Loaded, batch: list, out: str = "label") -> list:
    texts, contexts = [t for t, _ in batch], [c for _, c in batch]
    return state.logits(texts, contexts) if out == "logits" else state.run(texts, contexts)


def _note(ctx, msg: str) -> None:
    if ctx is not None:
        ctx.note("msgmodel: " + msg)


def predict(model_dir: str, texts: list[str], contexts: list[list[str]], *, plan=None, device: str | None = None,
            tier: str | None = None, threads: int = 0, batch_size: int | None = None, gpu_mem_limit_mb: int = 0,
            ctx=None) -> list[tuple[str, float]]:
    """ONNX filter model (onnxruntime + tokenizers): → (category, probability) for every message.

    Optional: ``plan`` — ``hw.StagePlan`` of the msgmodel stage (slots and tier); ``device`` —
    one device (``cpu``, ``cuda:N``, ``rocm:N``, ``dml:N``); ``tier`` — a tier from index.json;
    ``threads``/``batch_size``/``gpu_mem_limit_mb`` — for one device."""
    return _infer(model_dir, texts, contexts, "label", plan=plan, device=device, tier=tier, threads=threads,
                  batch_size=batch_size, gpu_mem_limit_mb=gpu_mem_limit_mb, ctx=ctx)


def predict_proba(model_dir: str, texts: list[str], contexts: list[list[str]], *, temperature: float = 1.0,
                  plan=None, device: str | None = None, tier: str | None = None, threads: int = 0,
                  batch_size: int | None = None, gpu_mem_limit_mb: int = 0, ctx=None) -> list[dict[str, float]]:
    """Probabilities of all categories per message: ``softmax(logits / temperature)`` as
    {label: probability}. The temperature of a tier is in its calibration (``calibration``). Other
    arguments — as in ``predict``."""
    rows = _infer(model_dir, texts, contexts, "logits", plan=plan, device=device, tier=tier, threads=threads,
                  batch_size=batch_size, gpu_mem_limit_mb=gpu_mem_limit_mb, ctx=ctx)
    if not rows:
        return []
    labels = _labels(model_dir, _plan_tier(plan, tier))
    return [dict(zip(labels, softmax(r, temperature))) for r in rows]


def calibration(model_dir: str | Path | None, tier: str | None = None) -> dict:
    """Calibration of the tier from index.json (``calibration``: ``temperature``, ``threshold``,
    optionally ``thresholds`` per category; written by train/oppoint.py). No index.json or no
    entry — {}."""
    try:
        _, entry = _tier_entry(model_dir, tier)
    except FileNotFoundError:
        return {}
    cal = entry.get("calibration")
    return dict(cal) if isinstance(cal, dict) else {}


def _plan_tier(plan, tier: str | None) -> str | None:
    if plan is not None and tier is None and plan.model and plan.model != "default":
        return plan.model
    return tier


def _labels(model_dir, tier: str | None) -> list[str]:
    d, _ = _tier(model_dir, tier)
    return json.loads((d / LABELS_FILE).read_text(encoding="utf-8"))


def _infer(model_dir: str, texts: list[str], contexts: list[list[str]], out: str, *, plan=None,
           device: str | None = None, tier: str | None = None, threads: int = 0, batch_size: int | None = None,
           gpu_mem_limit_mb: int = 0, ctx=None) -> list:
    """predict / predict_proba: ``out`` — "label" ((label, probability)) or "logits" (raw rows)."""
    if not texts:
        return []
    if len(contexts) != len(texts):
        raise ValueError("contexts and texts differ in length")
    tier = _plan_tier(plan, tier)
    if not available(model_dir, tier):
        try:
            fp32_missing = cpu_file(model_dir, tier) == (None, "fp32")
        except FileNotFoundError:
            fp32_missing = False
        if fp32_missing:
            raise FileNotFoundError(_FP32_MISSING)
        raise FileNotFoundError(f"filter model not found: {_root(model_dir)}"
                                + (f" (tier {tier})" if tier else ""))
    texts = list(texts)
    contexts = [list(c or []) for c in contexts]
    if plan is None and device is None and not threads and batch_size is None and not gpu_mem_limit_mb:
        return _run(_load(model_dir, tier), texts, contexts, out)     # the old path, byte for byte
    from .hw import Slot
    if plan is not None and plan.slots:
        slots = list(plan.slots)
    else:
        slots = [Slot(device or "cpu", 0, threads, batch_size or BATCH, "", gpu_mem_limit_mb)]
    if batch_size:
        slots = [Slot(s.device, s.instance, s.threads, batch_size, s.compute_type, s.gpu_mem_limit_mb) for s in slots]
    if len(slots) > 1 and len(texts) >= POOL_MIN:
        import functools
        from . import devpool
        items = list(zip(texts, contexts))
        # longest messages first: a batch holds messages of one length — less padding (the order of
        # the results is the order of the messages)
        return devpool.run(functools.partial(_pool_run, out=out), items, slots,
                           init=functools.partial(_pool_init, str(model_dir or ""), tier),
                           batch_size=slots[0].batch_size or BATCH, weight=_pair_len, ctx=ctx,
                           inprocess=False)
    s0 = slots[0]
    try:
        m = _load(model_dir, tier, s0.device, s0.threads if s0.device == "cpu" else 0, s0.gpu_mem_limit_mb,
                  s0.batch_size or BATCH)
    except ProviderUnavailable as e:
        from .hw import scrub_note
        _note(ctx, f"{s0.device}: {scrub_note(str(e))} — running on CPU")   # exception text may hold a path
        m = _load(model_dir, tier, "cpu", 0, 0, BATCH)
    return _run(m, texts, contexts, out)


def _run(m: _Loaded, texts: list[str], contexts: list[list[str]], out: str) -> list:
    return m.logits(texts, contexts) if out == "logits" else m.run(texts, contexts)
