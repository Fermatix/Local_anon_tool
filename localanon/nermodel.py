"""nermodel: name search with a fine-tuned model (BIO: O, B-PER, I-PER; a joint model also
B-ORG, I-ORG — the classes come from ``labels.json``) on onnxruntime + tokenizers.

No torch and no transformers. The models directory ``models/ner`` has the same contract as the
message filter (``models/msgfilter``): ``index.json`` with tiers ``tiers[].dir``, ``files.cpu``
(int8 or fp32 — by ``precision_cpu``), ``files.gpu`` (fp16); the tier directory holds
``tokenizer.json``, ``labels.json``, ``meta.json`` (max_len, pad_token). On top of the contract the
index has ``default_gpu`` — the default tier on a GPU (``default`` — on CPU) and ``org_tier`` — the tier
of a joint PER+ORG model that finds organizations when the chosen tier finds names only (``org_tier``).
Empty ``model_dir`` — ``models/ner`` in the repository root (unpacked there from the models archive by
run.sh / run.ps1, the package is installed editable), then ``localanon/models/ner``.

A line longer than the window (``max_len`` tokens including special ones) is cut into windows
overlapping by ``OVERLAP``; a token's label comes from the window where it is farthest from the edge.
Consecutive tokens of one class (only spaces between them; ``B-…`` at a word start begins a new span) →
a span by tokenizer offsets, widened to word boundaries. Windows run in batches sorted by length; one
pass gives the spans of all requested classes (``predict_classes``).

Hardware — as in msgmodel: ``plan=hw_plan.stages["ner_model"]`` (slots and tier) or ``device``; on a
GPU only the fp16 file of the tier is used; if it is missing or the provider fails — CPU with ``ctx.note``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import msgmodel as _mm

INDEX_FILE = "index.json"
TOKENIZER_FILE = "tokenizer.json"
LABELS_FILE = "labels.json"
META_FILE = "meta.json"
DEFAULT_MAX_LEN = 256
OVERLAP = 64
BATCH = 32
POOL_MIN = 2048            # lines from which several processes pay off the model loading
_SESSIONS: dict = {}

ProviderUnavailable = _mm.ProviderUnavailable


def default_dirs() -> list[Path]:
    return [_mm.REPO_ROOT / "models" / "ner", Path(__file__).resolve().parent / "models" / "ner"]


def root(model_dir: str | Path | None = None) -> Path:
    if model_dir:
        return Path(model_dir)
    for d in default_dirs():
        if (d / INDEX_FILE).is_file() or (d / _mm.MODEL_FILE).is_file():
            return d
    return default_dirs()[0]


def index(model_dir: str | Path | None = None) -> dict | None:
    return _mm._index(root(model_dir))


def default_tier(model_dir: str | Path | None = None, gpu: bool = False) -> str:
    """Default tier: on a GPU — ``default_gpu`` of the index (none — ``default``), on CPU — ``default``."""
    idx = index(model_dir) or {}
    names = {t.get("name") for t in idx.get("tiers", [])}
    if gpu and idx.get("default_gpu") in names:
        return str(idx["default_gpu"])
    return str(idx.get("default") or "")


def tier_dir(model_dir, tier: str | None = None) -> Path:
    return _mm.resolve_dir(root(model_dir), tier)


def cpu_file(model_dir, tier: str | None = None):
    return _mm.cpu_file(root(model_dir), tier)


def gpu_file(model_dir, tier: str | None = None) -> Path | None:
    return _mm.gpu_file(root(model_dir), tier)


def available(model_dir: str | Path | None = None, tier: str | None = None) -> bool:
    return _mm.available(str(root(model_dir)), tier)


def missing_reason(model_dir: str | Path | None = None, tier: str | None = None) -> str:
    """Why the model will not start (for ``ctx.note``); empty — everything is in place."""
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return "no onnxruntime or tokenizers (run.sh / run.ps1 installs them)"
    idx = index(model_dir)
    if tier and idx is not None and tier not in {t.get("name") for t in idx["tiers"]}:
        return f"tier \"{tier}\" is not in index.json"
    if not available(model_dir, tier):
        return ("name model not found in models/ner (put the models archive into the repository folder and "
                "start run.sh / run.ps1 again)")
    return ""


def _is_word(ch: str) -> bool:
    return ch.isalpha() or ch.isdigit()


def widen(text: str, s: int, e: int) -> tuple[int, int]:
    """Span without spaces at the edges, widened to word boundaries (letters and digits)."""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if s >= e:
        return s, s
    while s > 0 and _is_word(text[s - 1]) and _is_word(text[s]):
        s -= 1
    while e < len(text) and _is_word(text[e]) and _is_word(text[e - 1]):
        e += 1
    return s, e


def bio_spans(text: str, offsets, labels: list[str], cls: str = "PER") -> list[tuple[int, int]]:
    """Token labels → spans of class ``cls`` (PER, ORG). Consecutive tokens of the class with only
    spaces between them are one span; ``B-…`` at the start of a new word begins a new one. Edges go to
    word boundaries, overlaps merge."""
    raw: list[list[int]] = []
    cur = None
    for (a, b), lab in zip(offsets, labels):
        if a >= b:
            continue
        lab = lab.upper()
        if not lab.endswith("-" + cls) and lab != cls:
            cur = None
            continue
        if cur is not None:
            gap = text[cur[1]:a]
            # continuation: adjacent (part of a word) or across spaces, unless it starts a new name
            if not gap.strip() and (gap == "" or not lab.startswith("B")):
                cur[1] = max(cur[1], b)
                continue
        cur = [a, b]
        raw.append(cur)
    out: list[tuple[int, int]] = []
    for a, b in raw:
        s, e = widen(text, a, b)
        if s >= e:
            continue
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def label_classes(labels) -> frozenset:
    """Entity classes of a BIO label list: ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"] → {PER, ORG}."""
    return frozenset(str(x).upper().split("-", 1)[1] for x in labels if "-" in str(x))


def classes_of(model_dir: str | Path | None = None, tier: str | None = None) -> frozenset:
    """Classes the tier's model finds (from ``labels.json``); no model or no file — empty."""
    try:
        d = tier_dir(model_dir, tier)
        return label_classes(json.loads((d / LABELS_FILE).read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return frozenset()


def org_tier(model_dir: str | Path | None = None, tier: str | None = None) -> str | None:
    """A separate tier for organizations when the model of ``tier`` has no ORG class: ``org_tier`` of the
    index (a joint PER+ORG model) if it is among the tiers and its model has ORG; otherwise None."""
    if "ORG" in classes_of(model_dir, tier):
        return None
    idx = index(model_dir) or {}
    name = str(idx.get("org_tier") or "")
    names = {t.get("name") for t in idx.get("tiers", [])}
    return name if name in names and "ORG" in classes_of(model_dir, name) else None


def windows(n: int, size: int, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    """Windows [a, b) over ``n`` content tokens: ``size`` long, step ``size - overlap``."""
    if n <= size:
        return [(0, n)]
    step = max(1, size - max(0, min(overlap, size // 2)))
    out, a = [], 0
    while True:
        b = min(n, a + size)
        out.append((a, b))
        if b >= n:
            return out
        a += step


class _Loaded:
    def __init__(self, d: Path, device: str = "cpu", threads: int = 0, gpu_mem_limit_mb: int = 0,
                 files: dict | None = None, batch: int = BATCH, cpu_model: Path | None = None,
                 precision: str = ""):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.labels = [str(x) for x in json.loads((d / LABELS_FILE).read_text(encoding="utf-8"))]
        self.classes = label_classes(self.labels)
        meta = {}
        if (d / META_FILE).is_file():
            meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        self.max_len = int(meta.get("max_len", DEFAULT_MAX_LEN))
        self.tok = Tokenizer.from_file(str(d / TOKENIZER_FILE))
        self.tok.no_padding()
        self.tok.no_truncation()
        pad = meta.get("pad_token", "[PAD]")
        self.pad_id = self.tok.token_to_id(pad)
        if self.pad_id is None:
            self.pad_id = 0
        # special tokens around the content ([CLS] … [SEP], <s> … </s>) — from a probe string
        probe = self.tok.encode("а")
        sm = list(probe.special_tokens_mask)
        k = sm.index(0) if 0 in sm else len(sm)
        tail = len(sm) - (max(i for i, x in enumerate(sm) if x == 0) + 1) if 0 in sm else 0
        self.prefix = list(probe.ids[:k])
        self.suffix = list(probe.ids[len(probe.ids) - tail:]) if tail else []
        self.content = max(8, self.max_len - len(self.prefix) - len(self.suffix))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        files = files or {}
        self.device = device
        self.batch = max(1, int(batch or BATCH))
        model = cpu_model or d / files.get("cpu", _mm.MODEL_FILE)
        self.precision = precision if device == "cpu" else "fp16"
        if device == "cpu":
            if threads:
                so.intra_op_num_threads = int(threads)
                so.inter_op_num_threads = 1
            providers = ["CPUExecutionProvider"]
        else:
            gpu = d / str(files.get("gpu") or _mm.FP16_FILE)
            if not gpu.is_file():
                raise ProviderUnavailable(f"no GPU file ({gpu.name}) — the int8 model is not put on a GPU")
            model = gpu
            if device.startswith("cuda"):
                _mm._preload_gpu_libs(ort)
            providers = _mm.providers_for(device, ort.get_available_providers(), gpu_mem_limit_mb)
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

    def _logits(self, rows: list[list[int]]):
        import numpy as np

        n = max(len(r) for r in rows)
        ids = np.full((len(rows), n), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), n), dtype=np.int64)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = r
            mask[i, :len(r)] = 1
        feed = {}
        for name in self.inputs:
            if name == "input_ids":
                feed[name] = ids
            elif name == "attention_mask":
                feed[name] = mask
            elif name == "token_type_ids":
                feed[name] = np.zeros_like(ids)
        return self.sess.run(None, feed)[0]

    def spans(self, texts: list[str]) -> list[list[tuple[int, int]]]:
        """→ PER spans (start, end) for every line."""
        return [d["PER"] for d in self.class_spans(texts, ("PER",))]

    def class_spans(self, texts: list[str], classes=("PER",)) -> list[dict[str, list[tuple[int, int]]]]:
        """→ for every line {class: spans} — one model pass for all ``classes`` (PER, ORG); a class
        the model does not have gets an empty list."""
        encs = self.tok.encode_batch([t or "" for t in texts], add_special_tokens=False)
        # Encoding.ids builds a new list on every read: once per line, not per window (a line of
        # 900 thousand tokens re-read for each of its 4 700 windows is 5 minutes of one worker)
        ids = [e.ids for e in encs]
        # (line, window start, window end) — all windows of all lines
        jobs = []
        for i, x in enumerate(ids):
            if not x:
                continue
            for a, b in windows(len(x), self.content):
                jobs.append((i, a, b))
        jobs.sort(key=lambda j: j[2] - j[1])
        best: dict[int, list] = {i: [None] * len(x) for i, x in enumerate(ids) if x}
        for s in range(0, len(jobs), self.batch):
            part = jobs[s:s + self.batch]
            rows = [self.prefix + ids[i][a:b] + self.suffix for i, a, b in part]
            pred = self._logits(rows).argmax(-1)
            k0 = len(self.prefix)
            for (i, a, b), row in zip(part, pred):
                slot = best[i]
                for t in range(a, b):
                    score = min(t - a, b - 1 - t)
                    if slot[t] is None or score > slot[t][0]:
                        slot[t] = (score, int(row[k0 + t - a]))
        out = []
        for i, (text, e) in enumerate(zip(texts, encs)):
            if not ids[i]:
                out.append({c: [] for c in classes})
                continue
            labs = [self.labels[x[1]] if x is not None and x[1] < len(self.labels) else "O" for x in best[i]]
            offsets = e.offsets
            out.append({c: bio_spans(text or "", offsets, labs, c) if c in self.classes else []
                        for c in classes})
        return out


def _load(model_dir, tier: str | None = None, device: str = "cpu", threads: int = 0,
          gpu_mem_limit_mb: int = 0, batch: int = BATCH) -> _Loaded:
    r = root(model_dir)
    d, files = _mm._tier(r, tier)
    d = d.resolve()
    key = (str(d), device, int(threads or 0), int(gpu_mem_limit_mb or 0), int(batch or BATCH))
    if key not in _SESSIONS:
        cpu, prec = _mm.cpu_file(r, tier) if device == "cpu" else (None, "")
        if device == "cpu" and cpu is None:
            raise FileNotFoundError(_mm._FP32_MISSING)
        _SESSIONS[key] = _Loaded(d, device, threads, gpu_mem_limit_mb, files, batch, cpu, prec)
    return _SESSIONS[key]


def _pool_init(model_dir: str, tier, slot):
    return _load(model_dir, tier, slot.device, slot.threads if slot.device == "cpu" else 0,
                 slot.gpu_mem_limit_mb, slot.batch_size or BATCH)


def _pool_run(classes, state: _Loaded, batch: list) -> list:
    return state.class_spans(batch, classes)


def _note(ctx, msg: str) -> None:
    if ctx is not None:
        ctx.note("ner_model: " + msg)


def predict_spans(model_dir, texts: list[str], **kw) -> list[list[tuple[int, int]]]:
    """PER spans per line (``predict_classes`` with the PER class only)."""
    return [d["PER"] for d in predict_classes(model_dir, texts, classes=("PER",), **kw)]


def predict_classes(model_dir, texts: list[str], *, classes=("PER",), plan=None, device: str | None = None,
                    tier: str | None = None, threads: int = 0, batch_size: int | None = None,
                    gpu_mem_limit_mb: int = 0, ctx=None,
                    info: dict | None = None) -> list[dict[str, list[tuple[int, int]]]]:
    """{class: spans} per line, one model pass for all ``classes``. ``plan`` — ``hw.StagePlan`` of the
    ner_model stage (slots and tier); without it — one device ``device`` (CPU by default). ``info``
    (if given) receives the tier, device, precision and loading time."""
    classes = tuple(classes)
    if not texts:
        return []
    if plan is not None and tier is None and plan.model:
        tier = plan.model
    if not available(model_dir, tier):
        raise FileNotFoundError(f"name model not found: {root(model_dir)}"
                                + (f" (tier {tier})" if tier else ""))
    from .hw import Slot
    if plan is not None and plan.slots:
        slots = list(plan.slots)
    else:
        slots = [Slot(device or "cpu", 0, threads, batch_size or BATCH, "", gpu_mem_limit_mb)]
    if batch_size:
        slots = [Slot(s.device, s.instance, s.threads, batch_size, s.compute_type, s.gpu_mem_limit_mb) for s in slots]
    texts = list(texts)
    if info is not None:
        info.update(tier=tier or default_tier(model_dir), device=slots[0].device, workers=len(slots))
    if len(slots) > 1 and len(texts) >= POOL_MIN:
        import functools
        from . import devpool
        # longest lines first: a pool batch holds lines of one length — less padding in a model batch
        # (×1.3 on real chats; the order of the results is the order of the lines)
        return devpool.run(functools.partial(_pool_run, classes), texts, slots,
                           init=functools.partial(_pool_init, str(root(model_dir)), tier),
                           batch_size=256, weight=len, ctx=ctx, inprocess=False)
    s0 = slots[0]
    t0 = time.perf_counter()
    try:
        m = _load(model_dir, tier, s0.device, s0.threads if s0.device == "cpu" else 0, s0.gpu_mem_limit_mb,
                  s0.batch_size or BATCH)
    except ProviderUnavailable as e:
        from .hw import scrub_note
        _note(ctx, f"{s0.device}: {scrub_note(str(e))} — running on CPU")
        m = _load(model_dir, tier, "cpu", 0, 0, BATCH)
    if info is not None:
        info.update(device=m.device, precision=m.precision, load_s=round(time.perf_counter() - t0, 2))
    return m.class_spans(texts, classes)
