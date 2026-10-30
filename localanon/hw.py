"""hw: what compute the machine has and how to lay the stages out over it.

    res = hw.detect(ctx)                       # CPU, RAM, GPU, onnxruntime providers — no paths
    pl = hw.plan(res, ctx.config, hw.Workload(n_messages=…, audio_seconds=…))
    pl.stages["msgmodel"].slots                # [Slot(device="cuda:0", instance=0, threads=1, …), …]

Detection never fails: a failure of any part is a record in ``notes`` (and ``ctx.note``) and a
fallback. Without a GPU the plan is CPU. CUDA in onnxruntime is checked by the presence of cuDNN 9
and a probe session on a tiny Conv + MatMul graph in a subprocess (a provider in the list does not
yet mean that CUDA and cuDNN will load, and without cuDNN onnxruntime aborts); the install trap
(CPU onnxruntime over onnxruntime-gpu) is recognized by package metadata. Cards for ASR (ctranslate2)
are counted only if cuBLAS 12 and cuDNN 9 loaded (``preload_ct2_cuda_libs`` — the ASR worker calls
it too before ``WhisperModel``). Any choice is overridden by ``[compute]``,
``[media].asr_device``/``asr_model`` and ``[ner].device``.

Devices are written as strings: ``cpu``, ``cuda:N``, ``rocm:N``, ``dml:N``. ``N`` is the index in
the process numbering: after ``CUDA_VISIBLE_DEVICES`` (``HIP_VISIBLE_DEVICES``) and in PCI bus
order, as in nvidia-smi: ``detect`` (before the torch/onnxruntime/ctranslate2 probes) and the devpool
workers set ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` if it is not set.

``Resources``/``Plan`` hold no paths, user names or host names: only numbers, device model names
and providers. Error messages carry only the exception type (the text may contain paths).

Several machines: ``shard_of``/``select_shard`` split the input by a stable hash of the relative
path, ``merge_outputs`` assembles the shard outputs into one (``localanon run --shard k/N`` and
``localanon merge``). Engine tokens do not depend on the shard: ``anonymizer/engine/mint.py`` mints
them from the salt and the value without state.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = ["asr", "msgmodel", "ner_model", "natasha", "documents", "anonymize"]
SMI_TIMEOUT = 15
NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
                    "--format=csv,noheader,nounits"]
NVIDIA_SMI_UUID = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]


# ── resources ──────────────────────────────────────────────────────────────────────────────────

@dataclass
class CPU:
    logical: int = 1
    physical: int = 1
    affinity: int | None = None          # processors in the process mask (Linux, psutil)
    cgroup_quota: float | None = None    # cgroup quota in cores (cpu.max / cfs_quota_us)
    processor_groups: int = 1            # Windows: processor groups (more than 64 logical)
    group_size: int = 0                  # logical in the largest group (0 — unknown)
    numa_nodes: int = 1
    usable: int = 1                      # logical that can actually be used
    usable_physical: int = 1             # the same in physical cores


@dataclass
class GPU:
    index: int                           # index in the process numbering (cuda:N, rocm:N, dml:N)
    vendor: str                          # nvidia | amd | dml
    name: str = ""
    mem_total_mb: int = 0                # 0 — unknown
    mem_free_mb: int = 0
    compute_cap: str = ""
    source: str = ""                     # nvml | nvidia-smi | rocm-smi | torch | onnxruntime

    @property
    def device(self) -> str:
        kind = {"nvidia": "cuda", "amd": "rocm"}.get(self.vendor, self.vendor)
        return f"{kind}:{self.index}"


@dataclass
class Resources:
    os: str = ""                         # linux | windows | darwin
    python: str = ""
    cpu: CPU = field(default_factory=CPU)
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    gpus: list[GPU] = field(default_factory=list)
    ort_providers: list[str] = field(default_factory=list)
    ort_packages: dict = field(default_factory=dict)  # {"onnxruntime": "1.20.1", "onnxruntime-gpu": …}
    cuda_probe: str = ""                 # CUDA probe session: "" — none, ok | failed
    torch: dict = field(default_factory=dict)       # {"version", "cuda", "hip", "devices"}
    ct2_cuda_devices: int | None = None             # ctranslate2 (faster-whisper); None — not installed
    cpu_probe: dict = field(default_factory=dict)   # probe_cpu_speed: {"tier", "msgs_per_s", …}; {} — no probe
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "Resources":
        d = dict(d)
        d["cpu"] = CPU(**d.get("cpu", {}))
        d["gpus"] = [GPU(**g) for g in d.get("gpus", [])]
        return cls(**d)

    def gpu(self, device: str) -> GPU | None:
        for g in self.gpus:
            if g.device == device:
                return g
        return None


class _Notes:
    def __init__(self, ctx):
        self.ctx = ctx
        self.items: list[str] = []

    def __call__(self, msg: str) -> None:
        # the note template is not scrubbed (a login "nvidia" or "cpu" would spoil the fix command) — the caller
        # passes external text (stderr, exception text) through scrub_note before substitution
        self.items.append(msg)
        if self.ctx is not None:
            self.ctx.note("hw: " + msg)


def _err(e: BaseException) -> str:
    return type(e).__name__


def _run(cmd: list[str], timeout: float = SMI_TIMEOUT) -> str | None:
    """Command stdout or None (no tool, error, timeout). Argument list only, no shell."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


# CPU ─────────────────────────────────────────────────────────────────────────────────────────

def parse_cgroup_v2_max(text: str) -> float | None:
    """"400000 100000" → 4.0 cores; "max 100000" → None (no quota)."""
    parts = (text or "").split()
    if not parts or parts[0] == "max":
        return None
    try:
        quota = int(parts[0])
        period = int(parts[1]) if len(parts) > 1 else 100000
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def cgroup_cpu_quota(root: Path = Path("/sys/fs/cgroup"), self_cgroup: str | None = None) -> float | None:
    """CPU quota in cores (the smallest along the process cgroup path), None — no quota. v2 and v1."""
    if self_cgroup is None:
        self_cgroup = _read(Path("/proc/self/cgroup")) or ""
    best: float | None = None

    def take(q):
        nonlocal best
        if q is not None and (best is None or q < best):
            best = q

    # v2: "0::/path"
    rel = ""
    v1_rel = ""
    for line in self_cgroup.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and parts[1] == "":
            rel = parts[2]
        elif "cpu" in parts[1].split(","):
            v1_rel = parts[2]
    if (root / "cgroup.controllers").exists() or (root / "cpu.max").exists() or rel:
        d = root / rel.strip("/") if rel.strip("/") else root
        for p in [d, *d.parents]:
            take(parse_cgroup_v2_max(_read(p / "cpu.max") or ""))
            if p == root:
                break
    # v1: cpu,cpuacct/cpu.cfs_quota_us and cpu.cfs_period_us
    for name in ("cpu,cpuacct", "cpu", "cpuacct,cpu"):
        base = root / name
        if not base.is_dir():
            continue
        cands = []
        if v1_rel.strip("/"):
            cands.append(base / v1_rel.strip("/"))
        cands.append(base)
        for d in cands:
            q, per = _read(d / "cpu.cfs_quota_us"), _read(d / "cpu.cfs_period_us")
            try:
                qi, pi = int((q or "").strip()), int((per or "").strip())
            except ValueError:
                continue
            if qi > 0 and pi > 0:
                take(qi / pi)
        break
    return best


def cgroup_mem_limit_mb(root: Path = Path("/sys/fs/cgroup"), self_cgroup: str | None = None) -> int | None:
    """cgroup memory limit in MB (v2 memory.max, v1 memory.limit_in_bytes), None — no limit."""
    if self_cgroup is None:
        self_cgroup = _read(Path("/proc/self/cgroup")) or ""
    rel = ""
    for line in self_cgroup.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            rel = parts[2].strip("/")
    vals = []
    d = root / rel if rel else root
    for p in [d, *d.parents]:
        t = (_read(p / "memory.max") or "").strip()
        if t.isdigit():
            vals.append(int(t))
        if p == root:
            break
    t = (_read(root / "memory" / "memory.limit_in_bytes") or "").strip()
    if t.isdigit() and int(t) < 1 << 60:          # "no limit" in v1 is a huge number
        vals.append(int(t))
    return min(vals) // (1 << 20) if vals else None


def _linux_physical(sysfs: Path) -> int | None:
    cores = set()
    for d in sysfs.glob("cpu[0-9]*/topology"):
        core = _read(d / "core_id")
        pkg = _read(d / "physical_package_id")
        if core is None:
            continue
        cores.add(((pkg or "0").strip(), core.strip()))
    return len(cores) or None


def _windows_groups() -> tuple[int, int] | None:
    """(number of active groups, logical in the largest one) via kernel32."""
    try:
        import ctypes
        k = ctypes.WinDLL("kernel32")
        n = k.GetActiveProcessorGroupCount()
        sizes = [k.GetActiveProcessorCount(g) for g in range(n)]
        return n, max(sizes) if sizes else 0
    except Exception:  # noqa: BLE001 — not Windows or an old system
        return None


def usable_cpus(cpu: CPU) -> CPU:
    """How many logical and physical cores are actually available (affinity, quota, Windows groups)."""
    logical = max(1, cpu.logical)
    physical = max(1, min(cpu.physical or logical, logical))
    cap = logical
    if cpu.affinity:
        aff = cpu.affinity
        # Windows before 11: the process mask is one group (≤ 64); this is not a user limit,
        # the OS spreads worker processes over the groups — count the whole machine.
        if cpu.processor_groups > 1 and cpu.group_size and aff >= cpu.group_size:
            aff = logical
        cap = min(cap, aff)
    phys = max(1, round(cap * physical / logical))
    if cpu.cgroup_quota:
        q = max(1, math.floor(cpu.cgroup_quota + 1e-9))
        cap = min(cap, q)
        phys = min(phys, q)          # the quota is CPU time: it does not double on SMT siblings
    cpu.usable = max(1, cap)
    cpu.usable_physical = max(1, min(phys, cpu.usable))
    return cpu


def detect_cpu(note, *, sysfs_root: Path = Path("/sys"), cgroup_root: Path = Path("/sys/fs/cgroup"),
               os_name: str | None = None) -> CPU:
    os_name = os_name or sys.platform
    cpu = CPU()
    logical = None
    try:
        logical = os.cpu_count()
    except Exception as e:  # noqa: BLE001
        note(f"os.cpu_count: {_err(e)}")
    psutil = None
    try:
        import psutil  # type: ignore
    except ImportError:
        pass
    physical = None
    if psutil is not None:
        try:
            logical = psutil.cpu_count(logical=True) or logical
            physical = psutil.cpu_count(logical=False)
        except Exception as e:  # noqa: BLE001
            note(f"psutil.cpu_count: {_err(e)}")
    if physical is None and os_name.startswith("linux"):
        physical = _linux_physical(sysfs_root / "devices" / "system" / "cpu")
    if physical is None:
        note("number of physical cores unknown — counting logical ones")
    cpu.logical = max(1, logical or 1)
    cpu.physical = max(1, physical or cpu.logical)
    # affinity
    aff = None
    if hasattr(os, "process_cpu_count"):                 # Python 3.13+
        try:
            aff = os.process_cpu_count()
        except Exception:  # noqa: BLE001
            aff = None
    if aff is None and hasattr(os, "sched_getaffinity"):
        try:
            aff = len(os.sched_getaffinity(0))
        except Exception:  # noqa: BLE001
            aff = None
    if aff is None and psutil is not None:
        try:
            aff = len(psutil.Process().cpu_affinity())
        except Exception:  # noqa: BLE001 — macOS: no cpu_affinity
            aff = None
    cpu.affinity = aff
    if os_name.startswith("linux"):
        try:
            cpu.cgroup_quota = cgroup_cpu_quota(cgroup_root)
        except Exception as e:  # noqa: BLE001
            note(f"cgroup quota not read: {_err(e)}")
        try:
            nodes = list((sysfs_root / "devices" / "system" / "node").glob("node[0-9]*"))
            cpu.numa_nodes = max(1, len(nodes))
        except Exception:  # noqa: BLE001
            cpu.numa_nodes = 1
    if os_name.startswith("win"):
        g = _windows_groups()
        if g:
            cpu.processor_groups, cpu.group_size = g
    return usable_cpus(cpu)


# RAM ─────────────────────────────────────────────────────────────────────────────────────────

def parse_meminfo(text: str) -> tuple[int, int]:
    """/proc/meminfo → (total MB, available MB)."""
    vals = {}
    for line in (text or "").splitlines():
        m = re.match(r"^(\w+):\s+(\d+)\s*kB", line)
        if m:
            vals[m.group(1)] = int(m.group(2)) // 1024
    total = vals.get("MemTotal", 0)
    avail = vals.get("MemAvailable", vals.get("MemFree", 0) + vals.get("Cached", 0))
    return total, avail


def _windows_mem() -> tuple[int, int] | None:
    try:
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMSTAT()
        st.dwLength = ctypes.sizeof(MEMSTAT)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullTotalPhys // (1 << 20), st.ullAvailPhys // (1 << 20)
    except Exception:  # noqa: BLE001
        return None


def detect_ram(note, *, os_name: str | None = None, cgroup_root: Path = Path("/sys/fs/cgroup"),
               meminfo: Path = Path("/proc/meminfo")) -> tuple[int, int]:
    os_name = os_name or sys.platform
    total = avail = 0
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        total, avail = vm.total // (1 << 20), vm.available // (1 << 20)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        note(f"psutil.virtual_memory: {_err(e)}")
    if not total and os_name.startswith("linux"):
        total, avail = parse_meminfo(_read(meminfo) or "")
    if not total and os_name.startswith("win"):
        total, avail = _windows_mem() or (0, 0)
    if not total:
        note("RAM size unknown — assuming 4 GB")
        total, avail = 4096, 2048
    if os_name.startswith("linux"):
        try:
            lim = cgroup_mem_limit_mb(cgroup_root)
        except Exception:  # noqa: BLE001
            lim = None
        if lim and lim < total:
            total, avail = lim, min(avail, lim)
    return total, avail


# GPU ─────────────────────────────────────────────────────────────────────────────────────────

def _int(s: str) -> int:
    try:
        return int(float(s.strip()))
    except (ValueError, AttributeError):
        return 0


def parse_nvidia_smi(text: str | None) -> list[GPU]:
    """Output of ``--query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader,
    nounits`` → cards. Empty, garbage, "No devices were found" — []."""
    out = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        cc = parts[4] if len(parts) > 4 and re.match(r"^\d+\.\d+$", parts[4]) else ""
        out.append(GPU(index=int(parts[0]), vendor="nvidia", name=parts[1],
                       mem_total_mb=_int(parts[2]), mem_free_mb=_int(parts[3]),
                       compute_cap=cc, source="nvidia-smi"))
    return out


def parse_uuid_map(text: str | None) -> dict[str, int]:
    out = {}
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            out[parts[1]] = int(parts[0])
    return out


def apply_visible(gpus: list[GPU], spec: str | None, uuids: dict[str, int] | None = None,
                  note=None) -> list[GPU]:
    """``CUDA_VISIBLE_DEVICES``: selection and renumbering in the string order. None — all cards;
    "" or "-1" — none. An item is an index or a UUID (prefix "GPU-…"); the list stops at the
    first invalid item — as in CUDA."""
    if spec is None:
        return gpus
    by_phys = {g.index: g for g in gpus}
    out: list[GPU] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            break
        phys = None
        if re.match(r"^-?\d+$", item):
            phys = int(item)
        elif uuids:
            for u, i in uuids.items():
                if u == item or u.startswith(item) or u.removeprefix("GPU-").startswith(item.removeprefix("GPU-")):
                    phys = i
                    break
        if phys is None or phys not in by_phys:
            if phys is None and note is not None:
                note("CUDA_VISIBLE_DEVICES: an item does not match a card — further cards are not visible")
            break
        g = by_phys[phys]
        if any(x is g for x in out):
            break
        out.append(g)
    return [GPU(**{**asdict(g), "index": i}) for i, g in enumerate(out)]


def _nvml(note, nvml=None) -> list[GPU] | None:
    """Cards via NVML (pynvml/nvidia-ml-py), None — NVML is unavailable."""
    if nvml is None:
        try:
            import pynvml as nvml  # type: ignore
        except ImportError:
            return None
    try:
        nvml.nvmlInit()
    except Exception:  # noqa: BLE001 — no driver
        return None
    try:
        out = []
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            try:
                cc = "%d.%d" % tuple(nvml.nvmlDeviceGetCudaComputeCapability(h))
            except Exception:  # noqa: BLE001
                cc = ""
            try:
                uuid = nvml.nvmlDeviceGetUUID(h)
                uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            except Exception:  # noqa: BLE001
                uuid = ""
            g = GPU(index=i, vendor="nvidia", name=str(name), mem_total_mb=int(mem.total) >> 20,
                    mem_free_mb=int(mem.free) >> 20, compute_cap=cc, source="nvml")
            g._uuid = uuid  # type: ignore[attr-defined]  # only for CUDA_VISIBLE_DEVICES
            out.append(g)
        return out
    except Exception as e:  # noqa: BLE001
        note(f"NVML: {_err(e)} — trying nvidia-smi")
        return None
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def detect_nvidia(note, *, env=None, runner=_run, nvml=None) -> list[GPU]:
    env = os.environ if env is None else env
    gpus = _nvml(note, nvml)
    uuids: dict[str, int] = {}
    if gpus is not None:
        uuids = {getattr(g, "_uuid", ""): g.index for g in gpus if getattr(g, "_uuid", "")}
    else:
        gpus = parse_nvidia_smi(runner(NVIDIA_SMI_QUERY))
        cvd = env.get("CUDA_VISIBLE_DEVICES")
        if gpus and cvd and not re.match(r"^[\d,\s-]*$", cvd):
            uuids = parse_uuid_map(runner(NVIDIA_SMI_UUID))
    return apply_visible(gpus, env.get("CUDA_VISIBLE_DEVICES"), uuids, note)


def parse_rocm_smi(text: str | None) -> list[GPU]:
    """``rocm-smi --showproductname --showmeminfo vram --json`` → AMD cards."""
    try:
        data = json.loads(text or "")
    except ValueError:
        return []
    out = []
    for key, val in data.items() if isinstance(data, dict) else []:
        m = re.match(r"^card(\d+)$", str(key))
        if not m or not isinstance(val, dict):
            continue
        name = ""
        for k in ("Card series", "Card SKU", "Card model", "Device Name"):
            if val.get(k):
                name = str(val[k])
                break
        total = used = 0
        for k, v in val.items():
            kl = k.lower()
            if "vram total memory" in kl:
                total = _int(str(v)) >> 20
            elif "vram total used memory" in kl:
                used = _int(str(v)) >> 20
        out.append(GPU(index=int(m.group(1)), vendor="amd", name=name, mem_total_mb=total,
                       mem_free_mb=max(0, total - used), source="rocm-smi"))
    out.sort(key=lambda g: g.index)
    return out


def detect_amd(note, *, env=None, runner=_run, os_name: str | None = None) -> list[GPU]:
    env = os.environ if env is None else env
    os_name = os_name or sys.platform
    if not os_name.startswith("linux"):
        return []
    gpus = parse_rocm_smi(runner(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"]))
    spec = env.get("HIP_VISIBLE_DEVICES", env.get("ROCR_VISIBLE_DEVICES"))
    return apply_visible(gpus, spec, None, note)


def detect_ort_providers(note) -> list[str]:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return []
    except Exception as e:  # noqa: BLE001 — broken install
        note(f"onnxruntime cannot be imported: {_err(e)}")
        return []
    try:
        return [p for p in ort.get_available_providers() if p != "AzureExecutionProvider"]
    except Exception as e:  # noqa: BLE001
        note(f"onnxruntime.get_available_providers: {_err(e)}")
        return []


# CUDA probe graph: Conv 1×1 (on CUDA it goes through cuDNN) → MatMul (cuBLAS), float[1,2,4,4] → float[1,2,4,4],
# opset 13, weights are small integers (CPU and CUDA give the same). Embedded as bytes: the onnx package may be absent.
PROBE_ONNX = bytes.fromhex(
    "080712096c6f63616c616e6f6e3ad3010a0f0a01780a01771201632204436f6e760a110a01630a016d12017922064d61"
    "744d756c120570726f62652a1f080208020801080110014201774a100000803f00000040000080bf0000803f2a4b0804"
    "0804100142016d4a400000803f000000000000004000000000000000000000803f00000000000080bf0000803f000080"
    "3f000000000000000000000000000000000000803f0000803f5a1b0a017812160a14080112100a0208010a0208020a02"
    "08040a020804621b0a017912160a14080112100a0208010a0208020a0208040a02080442040a00100d")
CUDA_DEPENDENT = ("CUDAExecutionProvider", "TensorrtExecutionProvider")
PROBE_TIMEOUT = 30
# Probe subprocess: like the filter worker (preload_dlls, CUDA session), plus a run and a check against CPU. The model
# is in stdin, the card number is argv[1]; stdout is the first session provider. Without cuDNN 9 onnxruntime takes a
# system cuDNN of another version and aborts instead of raising, so it is not done in the detect process.
PROBE_CHILD = """\
import sys
import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(4)
try:
    ort.preload_dlls()
except Exception:
    pass
model = sys.stdin.buffer.read()
so = ort.SessionOptions()
so.log_severity_level = 4
x = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4) / 2
ref = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
sess = ort.InferenceSession(model, so, providers=[
    ("CUDAExecutionProvider", {"device_id": int(sys.argv[1]), "gpu_mem_limit": 64 << 20,
                               "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"])
got = sess.get_providers()
print(got[0] if got else "-", flush=True)
if got and got[0] == "CUDAExecutionProvider" and not np.allclose(sess.run(None, {"x": x})[0], ref, atol=1e-3):
    sys.exit("CUDA computes differently from CPU")
"""
# cuDNN 9 for onnxruntime-gpu: (directory in site-packages, pattern, name to load from the system)
ORT_CUDNN_LINUX = ("nvidia/cudnn/lib", "libcudnn.so.9*", "libcudnn.so.9")
ORT_CUDNN_WINDOWS = ("nvidia/cudnn/bin", "cudnn*64_9.dll", "cudnn64_9.dll")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# quoted paths — whole; unquoted — from a drive letter, a UNC root, "~login" or a separator after a boundary
# (a space, "=", ":", "(", "…" — an absolute path; "://" — an address, not a path); a word with a separator — only at
# the line start (the tail of a path broken by a line wrap): "CUDA/ROCm", "12.x/13.x" in the middle of a line — not a path.
# Greedy up to the last separator in the line: a directory may have a space ("C:\Users\Иван Петров"), the file name
# stays; with one separator ("D:\Иван Петров", "/Ivan Petrov: …", "~ivan") — up to ":" or the line end.
_QUOTED_PATH = re.compile(r"\"[^\"\n]*[\\/][^\"\n]*\"|'[^'\n]*[\\/][^'\n]*'")
_SEP = r"\\/／＼"
_PATH = re.compile(
    rf"(?:\b[A-Za-z]:[{_SEP}]|\\\\[^\\\s]+\\|(?:^|(?<=[\s=(\[,;\"'«]))…?[{_SEP}]|(?<=:)(?![{_SEP}]{{2}})[{_SEP}]"
    rf"|(?:^|(?<=[\s=:(\[,;\"'«]))~(?=[^\W\d]|[{_SEP}])|^[ \t]*[^\s{_SEP}=:(\[,;\"'«]+[{_SEP}])"
    rf"(?:.*[{_SEP}]…?|[^\n:{_SEP}]*)", re.MULTILINE)


def ort_cudnn_found(*, site_dirs: list[Path] | None = None, loader=None, os_name: str | None = None) -> bool:
    """Whether cuDNN 9 for onnxruntime-gpu is present: a file in ``site-packages/nvidia/cudnn/lib`` (Windows — ``bin``)
    or the library loads by name from the system. The ``nvidia-cudnn-cu12`` and ``-cu13`` wheels share this directory:
    removing one of them erases the other's files, although the package is listed as installed."""
    os_name = os_name or sys.platform
    sub, pat, name = ORT_CUDNN_WINDOWS if os_name in ("win32", "windows") else ORT_CUDNN_LINUX
    dirs = _site_dirs() if site_dirs is None else [Path(d) for d in site_dirs]
    if any(f.is_file() for d in dirs for f in (d / sub).glob(pat)):
        return True
    if loader is None:
        import ctypes
        loader = ctypes.WinDLL if os_name in ("win32", "windows") else ctypes.CDLL
    try:
        loader(name)
        return True
    except OSError:
        return False


def _probe_subprocess(dev: int, timeout: float = PROBE_TIMEOUT) -> tuple[int | None, str, str]:
    """CUDA probe session in a separate process. → (return code, None — timeout; stdout; stderr)."""
    try:
        p = subprocess.run([sys.executable, "-c", PROBE_CHILD, str(dev)], input=PROBE_ONNX, capture_output=True,
                           timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return None, "", ""
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def _home_values() -> tuple[list[str], str]:
    """Home directories (longest first) and the account name — they must not appear in notes.
    The values are not printed."""
    vals = []
    try:
        vals.append(str(Path.home()))
    except Exception:  # noqa: BLE001 — no HOME and no passwd entry
        pass
    vals += [os.environ.get(k) or "" for k in ("USERPROFILE", "HOME", "HOMEPATH")]
    try:
        login = getpass.getuser().strip()
    except Exception:  # noqa: BLE001
        login = ""
    vals = {unicodedata.normalize("NFC", v.strip().rstrip("\\/")) for v in vals}
    return sorted((v for v in vals if len(v) >= 2), key=len, reverse=True), unicodedata.normalize("NFC", login)


def scrub_note(text: str) -> str:
    """External text (stderr, exception text) for substitution into a note (``ctx.note``, ``notes``) without the
    user name: the home directory (case-insensitive, any slashes) → "…", the account — only as a path segment,
    a quoted path → '"…"', an unquoted path → "…name". Note templates are not passed here."""
    s = unicodedata.normalize("NFC", text or "")   # NFD Cyrillic (macOS) would not match HOME otherwise
    homes, login = _home_values()
    for v in homes:
        pat = "".join(r"[\\/]" if c in "\\/" else re.escape(c) for c in v)   # as a whole word: "/home/ab" does not touch "/home/abc"
        s = re.sub(rf"(?<!\w){pat}(?!\w)", "…", s, flags=re.IGNORECASE)
    if login:   # not as a word anywhere in the text: a login "nvidia" or "cuda" would spoil "nvidia-smi", "on cuda:0"
        s = re.sub(rf"(?<=[\\/]){re.escape(login)}(?=[\\/]|$)", "…", s, flags=re.IGNORECASE | re.MULTILINE)
    s = _QUOTED_PATH.sub(lambda m: m.group(0)[0] + "…" + m.group(0)[0], s)
    return _PATH.sub("…", s)   # "." does not match a newline: each line separately


def stderr_tail(text: str) -> str:
    """The last non-empty stderr line without color codes and paths (paths contain the user name)."""
    lines = [s.strip() for s in _ANSI.sub("", text or "").splitlines() if s.strip()]
    return scrub_note(lines[-1])[:200] if lines else ""


def probe_cuda(note, res: "Resources", *, run=None, cudnn_found=None) -> bool:
    """A probe session with CUDAExecutionProvider on the NVIDIA card with the most free memory.
    A provider in the list does not yet mean that libcublas/cuDNN will load. First a cheap cuDNN 9 check
    (``ort_cudnn_found``), then a Conv + MatMul graph in a subprocess (``PROBE_CHILD``, timeout ``PROBE_TIMEOUT``):
    without cuDNN onnxruntime aborts and would kill detect itself. If it did not start, CUDA (and TensorRT) are
    removed from ``res.ort_providers`` with a note, the plan stays on CPU. ``run``/``cudnn_found`` — for tests.
    → whether it started."""
    nv = [g for g in res.gpus if g.vendor == "nvidia"]
    if "CUDAExecutionProvider" not in res.ort_providers or not nv:
        return False
    res.cuda_probe = "failed"
    dev = max(nv, key=lambda g: g.mem_free_mb).index
    try:
        has_cudnn = (cudnn_found or ort_cudnn_found)()
    except Exception:  # noqa: BLE001 — the check does not crash detect; the probe decides
        has_cudnn = True
    if not has_cudnn:
        sub, _, name = ORT_CUDNN_WINDOWS if res.os == "windows" else ORT_CUDNN_LINUX
        ver = res.ort_packages.get("onnxruntime-gpu", "")
        res.ort_providers = [p for p in res.ort_providers if p not in CUDA_DEPENDENT]
        note(f"onnxruntime: CUDAExecutionProvider is present, but cuDNN 9 not found ({name} is neither in {sub} nor in "
             f'the system) — message filter on CPU. Fix: python -m pip install --force-reinstall "onnxruntime-gpu[cuda,cudnn]'
             f'{"==" + ver if ver else ""}" (README.md, Quick start)')
        return False
    try:
        rc, out, err = (run or _probe_subprocess)(dev)
    except Exception as e:  # noqa: BLE001 — the interpreter did not start etc.
        rc, out, err = -1, "", _err(e)
    first = (out.strip().splitlines() or [""])[-1]
    if rc == 0 and first == "CUDAExecutionProvider":
        res.cuda_probe = "ok"
        return True
    if rc is None:
        why = f"no answer in {PROBE_TIMEOUT} s"
    elif rc == 0:
        why = "the session went to " + (first or "—")
    else:
        why = f"code {rc}"
        if rc < 0:
            try:
                why = f"signal {signal.Signals(-rc).name}"
            except ValueError:
                pass
        tail = stderr_tail(err)
        if tail:
            why += f": {tail}"
    res.ort_providers = [p for p in res.ort_providers if p not in CUDA_DEPENDENT]
    note(f"onnxruntime: CUDAExecutionProvider is present, but the CUDA probe session did not start ({why}) — "
         "message filter on CPU; check CUDA and cuDNN for this onnxruntime-gpu version (README.md, Quick start)")
    return False


ORT_PACKAGES = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-rocm",
                "onnxruntime-openvino")


def detect_ort_packages(note, metadata=None) -> dict:
    """Installed onnxruntime distributions → versions. The packages share one ``onnxruntime`` module:
    a CPU build installed after the GPU build (``faster-whisper`` and the ``msgmodel`` extra pull it in)
    silently overrides CUDA — this is visible only in the metadata."""
    if metadata is None:
        import importlib.metadata as metadata
    out = {}
    for name in ORT_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            note(f"metadata {name}: {_err(e)}")
    return out


INSTALL_SCRIPTS = "delete .venv and start run.sh (Linux, macOS) or run.ps1 (Windows) again"


def ort_fix_command(pkgs: dict) -> str:
    """Fix command when onnxruntime has no CUDA while an NVIDIA card is visible. The GPU build is installed —
    remove the CPU build and reinstall the GPU build of the same version over it (``--no-deps`` so that pip
    does not bring the CPU build back). Only the CPU build is installed — remove it and install the GPU build
    with the CUDA and cuDNN libraries (``[cuda,cudnn]``: without them the CUDA probe gives failed); the version
    is that of the CPU build, PyPI may not have it — then the install script (it takes the latest)."""
    if "onnxruntime" in pkgs and "onnxruntime-gpu" not in pkgs:
        return ('python -m pip uninstall -y onnxruntime && '
                f'python -m pip install "onnxruntime-gpu[cuda,cudnn]=={pkgs["onnxruntime"]}" '
                f"(or {INSTALL_SCRIPTS})")
    ver = pkgs.get("onnxruntime-gpu") or ""
    pin = f"=={ver}" if ver else ""
    cmd = ("python -m pip uninstall -y onnxruntime && "
           f"python -m pip install --force-reinstall --no-deps onnxruntime-gpu{pin}")
    return cmd if ver else f"{cmd} (or {INSTALL_SCRIPTS})"


def ort_trap_note(res: "Resources") -> str:
    """Diagnosis of the install trap (an NVIDIA card is present, no CUDA EP) or ""."""
    if not any(g.vendor == "nvidia" for g in res.gpus) or "CUDAExecutionProvider" in res.ort_providers:
        return ""
    if res.cuda_probe == "failed":
        return ""                        # the GPU build is in place, the CUDA libraries do not load — a different problem
    pkgs = res.ort_packages
    if not pkgs:
        return ""
    if "onnxruntime" in pkgs and "onnxruntime-gpu" in pkgs:
        what = (f"both onnxruntime {pkgs['onnxruntime']} and onnxruntime-gpu {pkgs['onnxruntime-gpu']} are installed — "
                "the CPU build overrode the GPU build")
    elif "onnxruntime-gpu" in pkgs:
        what = f"onnxruntime-gpu {pkgs['onnxruntime-gpu']} is installed, but without CUDAExecutionProvider (files overwritten)"
    else:
        what = "only the CPU build of onnxruntime is installed: " + ", ".join(f"{k} {v}" for k, v in pkgs.items())
    return f"onnxruntime: an NVIDIA card is present, no CUDA: {what}. Fix: {ort_fix_command(pkgs)}"


def detect_torch(note) -> dict:
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        return {}
    try:
        import torch  # type: ignore
        info = {"version": str(torch.__version__), "cuda": bool(torch.cuda.is_available()),
                "hip": str(getattr(torch.version, "hip", "") or ""), "devices": []}
        if info["cuda"]:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({"name": p.name, "mem_total_mb": int(p.total_memory) >> 20})
        return info
    except Exception as e:  # noqa: BLE001
        note(f"torch: {_err(e)}")
        return {}


# ctranslate2 (faster-whisper) is built for CUDA 12: cudart is built in, and cuBLAS 12 and cuDNN 9 are loaded by
# soname lazily, on the first run on the card. The nvidia-*-cu12 packages put them in site-packages, where
# the loader does not look, and onnxruntime-gpu[cuda,cudnn] installs CUDA 13 libraries — no good for ctranslate2.
CT2_CUDA_FIX = ('python -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" '
                f"(or {INSTALL_SCRIPTS})")
# (what, directory in site-packages, file pattern, required sonames)
CT2_LIBS_LINUX = (("cuBLAS 12", "nvidia/cublas/lib", "libcublas*.so.12*", ("libcublas.so.12", "libcublasLt.so.12")),
                  ("cuDNN 9", "nvidia/cudnn/lib", "libcudnn*.so.9*", ("libcudnn.so.9",)))
# Windows: (what, pattern, main library); directories — nvidia/*/bin and the ctranslate2 package itself
CT2_LIBS_WINDOWS = (("cuBLAS 12", "cublas64_12.dll", "cublas64_12.dll"),
                    ("cuDNN 9", "cudnn*64_9.dll", "cudnn64_9.dll"))
_CT2_LIBS: dict = {}
_DLL_DIRS: list = []                     # os.add_dll_directory handles live until the process ends


def _site_dirs() -> list[Path]:
    """site-packages directories of the current environment (venv, system, user)."""
    import site
    import sysconfig
    cands = [sysconfig.get_paths().get(k) for k in ("purelib", "platlib")]
    for fn in (site.getsitepackages, lambda: [site.getusersitepackages()]):
        try:
            cands += list(fn())
        except Exception:  # noqa: BLE001 — site is trimmed in some environments
            pass
    cands += _nvidia_roots()
    out: list[Path] = []
    for c in cands:
        if c and Path(c).is_dir() and Path(c) not in out:
            out.append(Path(c))
    return out


def _nvidia_roots() -> list[str]:
    """Directories where import finds the ``nvidia`` package (nvidia-*-cu12 wheels): the parent of each
    ``nvidia/``. Covers what is not among site-packages: ``PYTHONPATH``, ``.pth``, ``pip --target``."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("nvidia")
        locs = list(spec.submodule_search_locations or []) if spec else []
    except (ImportError, ValueError):
        return []
    return [str(Path(p).parent) for p in locs]


def _load_files(files: list[Path], loader) -> list[str]:
    """Load the files; those that failed because of dependency order — once more. → names of the loaded ones."""
    seen, todo = set(), []
    for f in files:
        key = os.path.realpath(f)
        if key not in seen:
            seen.add(key)
            todo.append(f)
    loaded: list[str] = []
    while todo:
        left = []
        for f in todo:
            try:
                loader(str(f))
                loaded.append(f.name)
            except OSError:
                left.append(f)
        if len(left) == len(todo):
            break
        todo = left
    return loaded


def _preload_linux(dirs: list[Path], loader) -> dict:
    loaded, missing, source = [], [], set()
    for what, sub, pat, need in CT2_LIBS_LINUX:
        files = [f for d in dirs for f in sorted((d / sub).glob(pat)) if f.is_file()]
        got = _load_files(files, loader)
        if got:
            source.add("pip")
        lack = [n for n in need if not any(g.startswith(n) for g in got)]
        for n in list(lack):             # not in the nvidia-* packages — system CUDA (ldconfig, LD_LIBRARY_PATH)
            try:
                loader(n)
                got.append(n)
                lack.remove(n)
                source.add("system")
            except OSError:
                pass
        loaded += got
        if lack:
            missing.append(f"{what} ({', '.join(lack)})")
    return {"loaded": loaded, "missing": missing, "source": "+".join(sorted(source))}


def _preload_windows(dirs: list[Path], loader, add_dir) -> dict:
    bins = [b for d in dirs for b in [*sorted(d.glob("nvidia/*/bin")), d / "ctranslate2"] if b.is_dir()]
    for b in bins:
        try:
            _DLL_DIRS.append(add_dir(str(b)))
        except OSError:
            pass
    if bins:                             # LoadLibrary by name (without search flags) looks at PATH
        os.environ["PATH"] = os.pathsep.join([*map(str, bins), os.environ.get("PATH", "")])
    loaded, missing = [], []
    for what, pat, main in CT2_LIBS_WINDOWS:
        files = [f for b in bins for f in sorted(b.glob(pat)) if f.is_file()]
        first = [f for f in files if f.name.lower() == main] or files[:1]
        got = _load_files(first, loader) if first else []
        if got:
            loaded += got
        else:
            missing.append(f"{what} ({pat}{'' if files else ' not found'})")
    return {"loaded": loaded, "missing": missing, "source": "pip" if loaded else ""}


def preload_ct2_cuda_libs(*, site_dirs: list[Path] | None = None, loader=None, os_name: str | None = None,
                          add_dll_directory=None) -> dict:
    """Load cuBLAS 12 and cuDNN 9 for ctranslate2 (faster-whisper) before the first run on the card.

    Linux: ``nvidia/cublas/lib/libcublas*.so.12*`` and ``nvidia/cudnn/lib/libcudnn*.so.9*`` from
    site-packages — by full path with ``RTLD_GLOBAL`` (then the lazy dlopen by soname finds them already
    loaded); what the packages lack — by soname from the system. Windows: ``os.add_dll_directory`` for
    ``nvidia/*/bin`` (and the ctranslate2 package directory), a check of ``cublas64_12.dll`` and ``cudnn*64_9.dll``.
    Call it in every process that runs ctranslate2 on the card (the ASR worker), before ``WhisperModel``.

    → ``{"ok": bool, "loaded": [file names], "source": "pip"|"system"|…, "reason": "" | what is missing}``
    without paths. A repeated call without arguments does nothing and returns the previous result; the
    arguments are for tests (substituting search and loading)."""
    default = site_dirs is None and loader is None and os_name is None and add_dll_directory is None
    if default and _CT2_LIBS:
        return dict(_CT2_LIBS)
    os_name = os_name or sys.platform
    try:
        import ctypes
        dirs = _site_dirs() if site_dirs is None else [Path(d) for d in site_dirs]
        if os_name in ("win32", "windows"):
            r = _preload_windows(dirs, loader or (lambda p: ctypes.WinDLL(p)),
                                 add_dll_directory or getattr(os, "add_dll_directory", lambda p: None))
        else:
            r = _preload_linux(dirs, loader or (lambda p: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)))
        out = {"ok": not r["missing"], "loaded": r["loaded"], "source": r["source"],
               "reason": "" if not r["missing"] else "cannot load " + ", ".join(r["missing"])}
    except Exception as e:  # noqa: BLE001 — preloading does not crash detection
        out = {"ok": False, "loaded": [], "source": "", "reason": f"preload: {_err(e)}"}
    if default:
        _CT2_LIBS.update(out)
    return dict(out)


def detect_ct2(note) -> int | None:
    """CUDA cards on which ctranslate2 will actually compute: the ctranslate2 card count, but 0 if cuBLAS 12 and
    cuDNN 9 did not load (``preload_ct2_cuda_libs``) — then ASR is on CPU with a note with the install command.
    None — ctranslate2 is not installed."""
    import importlib.util
    if importlib.util.find_spec("ctranslate2") is None:
        return None
    try:
        import ctranslate2  # type: ignore
        n = int(ctranslate2.get_cuda_device_count())
    except Exception as e:  # noqa: BLE001
        note(f"ctranslate2: {_err(e)}")
        return 0
    if n > 0:
        libs = preload_ct2_cuda_libs()
        if not libs["ok"]:
            note(f"ctranslate2: {n} CUDA cards, but {libs['reason']} — transcribing on CPU. "
                 f"Fix: {CT2_CUDA_FIX}")
            return 0
    return n


# Subprocess of the ctranslate2 compute probe: preloading as in the ASR worker, a tiny whisper with random
# weights (encoder convolutions — cuDNN, matrices — cuBLAS), encode on cpu and on card argv[1], a comparison.
# argv[2] — the directory holding the localanon package (running from sources without install).
CT2_PROBE_CHILD = """\
import sys, tempfile
sys.path.insert(0, sys.argv[2])
import numpy as np
import ctranslate2
from ctranslate2 import specs
from localanon import hw
libs = hw.preload_ct2_cuda_libs()
D, MELS = 8, 80
rng = np.random.default_rng(0)
arr = lambda *s: (rng.standard_normal(s) * 0.1).astype(np.float32)
def ln(x):
    x.gamma, x.beta = np.ones(D, np.float32), np.zeros(D, np.float32)
def lin(x, o, i):
    x.weight, x.bias = arr(o, i), arr(o)
spec = specs.WhisperSpec(1, 1, 1, 1)
toks = ["<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|translate|>", "<|transcribe|>",
        "<|startoflm|>", "<|startofprev|>", "<|nocaptions|>", "<|notimestamps|>"]
e, d = spec.encoder, spec.decoder
e.conv1.weight, e.conv1.bias = arr(D, MELS, 3), arr(D)
e.conv2.weight, e.conv2.bias = arr(D, D, 3), arr(D)
e.position_encodings.encodings = arr(1500, D)
d.embeddings.weight = arr(len(toks), D)
d.position_encodings.encodings = arr(448, D)
d.projection.weight = d.embeddings.weight
ln(e.layer_norm)
ln(d.layer_norm)
for lay in list(e.layer) + list(d.layer):
    ln(lay.self_attention.layer_norm)
    lin(lay.self_attention.linear[0], 3 * D, D)
    lin(lay.self_attention.linear[1], D, D)
    ln(lay.ffn.layer_norm)
    lin(lay.ffn.linear_0, D, D)
    lin(lay.ffn.linear_1, D, D)
for lay in d.layer:
    ln(lay.attention.layer_norm)
    lin(lay.attention.linear[0], D, D)
    lin(lay.attention.linear[1], 2 * D, D)
    lin(lay.attention.linear[2], D, D)
spec.register_vocabulary(toks)
spec.config.suppress_ids, spec.config.suppress_ids_begin, spec.config.lang_ids = [], [], [2]
spec.config.alignment_heads = [(0, 0)]
spec.validate()
with tempfile.TemporaryDirectory() as t:
    spec.save(t)
    x = ctranslate2.StorageView.from_array(arr(1, MELS, 3000))
    out = [np.array(ctranslate2.models.Whisper(t, device=dev, device_index=i, compute_type="float32")
                    .encode(x, to_cpu=True)) for dev, i in (("cpu", 0), ("cuda", int(sys.argv[1])))]
if not np.allclose(out[0], out[1], atol=1e-2):
    sys.exit("CUDA считает не так, как CPU")
print("ok" if libs["ok"] else "ok-system", flush=True)
"""
CT2_PROBE_TIMEOUT = 120


def _ct2_probe_subprocess(dev: int, timeout: float = CT2_PROBE_TIMEOUT) -> tuple[int | None, str, str]:
    try:
        p = subprocess.run([sys.executable, "-c", CT2_PROBE_CHILD, str(dev), str(Path(__file__).resolve().parents[1])],
                           capture_output=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return None, "", ""
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def probe_ct2_cuda(note, res: "Resources", *, run=None) -> bool | None:
    """ctranslate2 (faster-whisper) compute probe on the NVIDIA card with the most free memory: in a
    subprocess (``CT2_PROBE_CHILD``) — preloading cuBLAS 12 / cuDNN 9 and the encoder of a tiny whisper on cuda,
    compared with cpu. ``detect_ct2`` checks only that the libraries load; this probe checks that they actually
    compute (the install scripts call it via ``localanon hw --probe-cuda`` after ``nvidia-cudnn-cu12``).
    → True — computes; False — no (a note with the reason and the fix); None — no ctranslate2 or no NVIDIA cards."""
    import importlib.util
    nv = [g for g in res.gpus if g.vendor == "nvidia"]
    if not nv or importlib.util.find_spec("ctranslate2") is None:
        return None
    dev = max(nv, key=lambda g: g.mem_free_mb).index
    try:
        rc, out, err = (run or _ct2_probe_subprocess)(dev)
    except Exception as e:  # noqa: BLE001
        rc, out, err = -1, "", _err(e)
    last = (out.strip().splitlines() or [""])[-1]
    if rc == 0 and last in ("ok", "ok-system"):
        return True
    if rc is None:
        why = f"no answer in {CT2_PROBE_TIMEOUT} s"
    else:
        why = f"code {rc}"
        if rc < 0:
            try:
                why = f"signal {signal.Signals(-rc).name}"
            except ValueError:
                pass
        tail = stderr_tail(err)
        if tail:
            why += f": {tail}"
    note(f"ctranslate2: computing on the card failed ({why}) — transcribing on CPU. Fix: {CT2_CUDA_FIX}")
    return False


def install_probes(res: "Resources", note, *, ct2_run=None) -> dict:
    """Two probes after install (``localanon hw --probe-cuda``): onnxruntime CUDA (message filter) and
    ctranslate2 CUDA (transcription). ``res`` is the ``detect`` result: the onnxruntime probe already ran in it,
    the failure reason is in ``res.notes`` (the probe or the CPU build trap). → ``{"onnxruntime": "ok"|"failed"|"skip",
    "ctranslate2": …}``; skip — no NVIDIA card or no package."""
    nv = any(g.vendor == "nvidia" for g in res.gpus)
    ort = "skip" if not nv or not res.ort_packages else "ok" if res.cuda_probe == "ok" else "failed"
    c = probe_ct2_cuda(note, res, run=ct2_run)
    return {"onnxruntime": ort, "ctranslate2": {True: "ok", False: "failed", None: "skip"}[c]}


def detect(ctx=None, *, env=None, runner=_run, nvml=None, os_name: str | None = None,
           sysfs_root: Path = Path("/sys"), cgroup_root: Path = Path("/sys/fs/cgroup"),
           probe_libs: bool = True, speed_probe=None) -> Resources:
    """Find the machine's compute. Never fails: a failed part → ``notes`` (and ``ctx.note``), a fallback.

    ``env``, ``runner``, ``nvml``, ``os_name``, the sysfs/cgroup roots — for tests (substituting the
    environment, nvidia-smi, NVML and the file system). ``probe_libs=False`` — do not touch
    onnxruntime/torch/ctranslate2. ``speed_probe(note, res) -> dict`` replaces ``probe_cpu_speed``
    (tests)."""
    # before the first CUDA init (torch, ctranslate2, onnxruntime below): cuda:N numbers follow the PCI
    # order, as in nvidia-smi/NVML, not "fastest first" (the CUDA default order)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    note = _Notes(ctx)
    os_name = os_name or sys.platform
    res = Resources(os={"win32": "windows"}.get(os_name, "linux" if os_name.startswith("linux") else os_name),
                    python=platform.python_version())
    steps = [
        ("CPU", lambda: setattr(res, "cpu", detect_cpu(note, sysfs_root=sysfs_root, cgroup_root=cgroup_root,
                                                     os_name=os_name))),
        ("RAM", lambda: _set_ram(res, detect_ram(note, os_name=os_name, cgroup_root=cgroup_root))),
        ("NVIDIA", lambda: res.gpus.extend(detect_nvidia(note, env=env, runner=runner, nvml=nvml))),
    ]
    if probe_libs:
        steps += [("onnxruntime", lambda: setattr(res, "ort_providers", detect_ort_providers(note))),
                  ("onnxruntime metadata", lambda: setattr(res, "ort_packages", detect_ort_packages(note))),
                  ("torch", lambda: setattr(res, "torch", detect_torch(note))),
                  ("ctranslate2", lambda: setattr(res, "ct2_cuda_devices", detect_ct2(note)))]
    for what, fn in steps:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — detection never crashes the run
            note(f"{what}: {_err(e)} — fallback")
    try:
        if not any(g.vendor == "nvidia" for g in res.gpus):
            amd = detect_amd(note, env=env, runner=runner, os_name=os_name)
            if not amd and res.torch.get("hip") and res.torch.get("devices"):
                amd = [GPU(index=i, vendor="amd", name=d["name"], mem_total_mb=d["mem_total_mb"],
                           mem_free_mb=d["mem_total_mb"], source="torch")
                       for i, d in enumerate(res.torch["devices"])]
            res.gpus.extend(amd)
    except Exception as e:  # noqa: BLE001
        note(f"AMD: {_err(e)}")
    if not res.gpus and res.os == "windows" and "DmlExecutionProvider" in res.ort_providers:
        # A video card without CUDA (AMD, Intel) on Windows: DirectML, memory unknown.
        res.gpus.append(GPU(index=0, vendor="dml", name="DirectML", source="onnxruntime"))
    if probe_libs:
        try:
            probe_cuda(note, res)
        except Exception as e:  # noqa: BLE001
            note(f"CUDA probe session: {_err(e)}")
        if res.ort_packages:
            try:
                res.cpu_probe = (speed_probe or probe_cpu_speed)(note, res) or {}
            except Exception as e:  # noqa: BLE001
                note(f"CPU speed probe: {_err(e)} — default tiers")
    trap = ort_trap_note(res)
    if trap:
        note(trap)
    res.notes = note.items
    return res


# ── CPU speed probe ──────────────────────────────────────────────────────────────────────────
# The tier on CPU depends on the processor several times over: the base int8 name model gives 35–39
# msgs/s on all cores of a Threadripper 1920X (Zen 1, no VNNI) and ~260 on an Apple M4 Pro (docs/MODELS.md
# §2.8). Below SLOW_CPU_MSGS_PER_S the base tier needs more than ~1.4 h for 100k messages for the model
# alone, so a slow CPU takes the lightest tier (tiny2: ~12x faster). The Zen 1 server on 4 threads (21)
# is just above the line, so any machine at least as fast as it keeps base.
SLOW_CPU_MSGS_PER_S = 20.0
PROBE_TIER = "base"
PROBE_LINES = 32                         # synthetic messages, ~30 tokens each, as in §2.8
PROBE_BATCH = 8
PROBE_WARMUP = 4
PROBE_SECONDS = 2.5                      # deadline of one round (after loading and warm-up)
# a single round is noisy (another process, frequency ramp-up): up to PROBE_ROUNDS rounds while the
# next one, as long as the last, fits in PROBE_TOTAL_SECONDS; the fastest round counts
PROBE_ROUNDS = 3
PROBE_TOTAL_SECONDS = 3.0
PROBE_TEXTS = [
    "Иван Петров прислал макет главной страницы, посмотрите до пятницы и оставьте комментарии в задаче",
    "Мария Иванова: поправила вёрстку карточки товара, на мобильных кнопка больше не уезжает вниз",
    "Алексей, добавь, пожалуйста, валидацию телефона в форму заявки и обнови тесты для ООО Ромашка",
    "Сергей Смирнов выложил сборку 1.4.2 на тестовый стенд, авторизация через example.com работает",
    "Коллеги, созвон по API переносим на завтра на 11:00, Ольга Кузнецова подготовит список вопросов",
    "Проверил отчёт по заказам, фильтр по датам считает неверно, если выбрать последний день месяца",
    "Анна Соколова согласовала тексты для лендинга Acme, осталось заменить картинки в первом блоке",
    "Дмитрий Волков: миграция базы прошла, индексы пересобраны, время ответа списка упало до 200 мс",
]


def probe_cpu_speed(note, res: Resources, *, load=None, clock=time.perf_counter) -> dict:
    """Messages per second of the ``PROBE_TIER`` name model (int8) on all usable physical cores →
    ``{"tier", "msgs_per_s", "lines", "seconds", "threads", "rounds"}``: the fastest of up to
    ``PROBE_ROUNDS`` rounds within ``PROBE_TOTAL_SECONDS`` (``lines`` and ``seconds`` — of that
    round); ``{}`` — no model or onnxruntime. ``load(root, tier, threads)`` — an object with
    ``spans(texts)`` (tests)."""
    from . import nermodel
    root = nermodel.root(None)
    names = {t.get("name") for t in (nermodel.index(root) or {}).get("tiers", [])}
    tier = PROBE_TIER if PROBE_TIER in names else nermodel.default_tier(root) or None
    if load is None:
        if nermodel.missing_reason(root, tier):
            return {}

        def load(r, t, threads):
            return nermodel._load(r, t, "cpu", threads)
    threads = max(1, res.cpu.usable_physical)
    model = load(root, tier, threads)
    texts = [PROBE_TEXTS[i % len(PROBE_TEXTS)] for i in range(PROBE_LINES)]
    model.spans(texts[:PROBE_WARMUP])
    start, best, rounds, secs = clock(), None, 0, 0.0
    while rounds < PROBE_ROUNDS and (not rounds or clock() - start + secs <= PROBE_TOTAL_SECONDS):
        done, t0 = 0, clock()
        while done < PROBE_LINES:
            model.spans(texts[done:done + PROBE_BATCH])
            done += PROBE_BATCH
            if clock() - t0 >= PROBE_SECONDS:
                break
        secs = max(clock() - t0, 1e-6)
        rounds += 1
        if best is None or done / secs > best[0] / best[1]:
            best = (done, secs)
    done, secs = best
    return {"tier": tier or "", "msgs_per_s": round(done / secs, 1), "lines": done, "seconds": round(secs, 2),
            "threads": threads, "rounds": rounds}


def slow_cpu(res: Resources) -> bool:
    """The probe ran and the CPU is slower than ``SLOW_CPU_MSGS_PER_S``."""
    r = (res.cpu_probe or {}).get("msgs_per_s")
    return bool(r) and float(r) < SLOW_CPU_MSGS_PER_S


def _lightest(tiers: list) -> "Tier":
    return min(tiers, key=lambda t: (t.params_m or math.inf, t.ram_mb_per_worker))


def _set_ram(res: Resources, tr: tuple[int, int]) -> None:
    res.ram_total_mb, res.ram_available_mb = tr


# ── plan ─────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Workload:
    """Amount of work, if known (0 — unknown): limits the number of workers and the budget."""
    n_messages: int = 0
    audio_seconds: float = 0.0
    n_audio: int = 0
    n_documents: int = 0
    n_texts: int = 0


@dataclass
class Slot:
    device: str                          # cpu | cuda:N | rocm:N | dml:N
    instance: int = 0                    # model instance number on the device
    threads: int = 1                     # threads inside the worker (intra_op, cpu_threads, set_num_threads)
    batch_size: int = 1
    compute_type: str = ""               # ASR: float16 | int8_float16 | int8
    gpu_mem_limit_mb: int = 0            # card memory limit per instance (onnxruntime gpu_mem_limit)


@dataclass
class Tier:
    name: str
    dir: str = ""
    params_m: float = 0.0
    files: dict = field(default_factory=dict)
    max_len: int = 256
    min_gpu_mem_gb: float = 1.0
    ram_mb_per_worker: int = 500
    bench: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def quality(self) -> tuple:
        m = self.metrics or {}
        return (float(m.get("macro_f1", 0) or 0), float(m.get("delete_f1", 0) or 0), float(self.params_m or 0))


@dataclass
class StagePlan:
    stage: str
    slots: list[Slot] = field(default_factory=list)
    model: str = ""                      # ASR: model name; msgmodel: tier
    provider: str = ""                   # msgmodel: onnxruntime provider
    reason: str = ""

    def devices(self) -> list[str]:
        out = []
        for s in self.slots:
            if s.device not in out:
                out.append(s.device)
        return out

    def workers_per_device(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.slots:
            out[s.device] = out.get(s.device, 0) + 1
        return out

    @property
    def on_gpu(self) -> bool:
        return any(s.device != "cpu" for s in self.slots)

    def asr_kwargs(self) -> dict:
        """``faster_whisper.WhisperModel`` arguments for one process on all cards
        (``device_index=[…]``, ``num_workers`` — parallel transcriptions)."""
        if not self.slots:
            return {}
        s0 = self.slots[0]
        if s0.device == "cpu":
            return {"model_size_or_path": self.model, "device": "cpu", "compute_type": s0.compute_type or "int8",
                    "cpu_threads": s0.threads, "num_workers": len(self.slots)}
        idx = [int(d.split(":")[1]) for d in self.devices()]
        types = {s.compute_type for s in self.slots}
        ct = "float16" if types == {"float16"} else "int8_float16"
        return {"model_size_or_path": self.model, "device": "cuda", "device_index": idx, "compute_type": ct,
                "cpu_threads": s0.threads, "num_workers": max(1, len(self.slots) // max(1, len(idx)))}


@dataclass
class Plan:
    stages: dict[str, StagePlan] = field(default_factory=dict)
    cpu_budget: int = 1                  # physical cores for workers (minus the reserve)
    ram_budget_mb: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)

    def manifest_summary(self, res: Resources | None = None) -> dict:
        """For local_manifest.json: only numbers (and the tier name) — no paths and no card names."""
        out = {"gpus": len(res.gpus) if res else len({d for sp in self.stages.values() for d in sp.devices()
                                                      if d != "cpu"}),
               "cpu_budget": self.cpu_budget}
        for name, sp in self.stages.items():
            out[f"{name}_workers"] = len(sp.slots)
            out[f"{name}_gpu_workers"] = sum(1 for s in sp.slots if s.device != "cpu")
        mm = self.stages.get("msgmodel")
        if mm and mm.model:
            out["msgmodel_tier"] = mm.model
        nm = self.stages.get("ner_model")
        if nm and nm.model:
            out["ner_model_tier"] = nm.model
        asr = self.stages.get("asr")
        if asr and asr.model:
            out["asr_model"] = asr.model
        return out


# Worker memory estimates, MB: (CPU RAM, card memory per instance). Deliberately generous.
MEM = {
    "asr": {"tiny": (400, 1000), "base": (500, 1200), "small": (900, 2000), "medium": (1800, 3500),
            "large-v2": (3500, 5000), "large-v3": (3500, 5000), "turbo": (2500, 3500),
            "large-v3-turbo": (2500, 3500), "distil-large-v3": (2500, 3500)},
    "ner_model": (900, 1500),            # without an index; with a tier — its ram_mb_per_worker, min_gpu_mem_gb
    "natasha": (600, 0),
    "documents": (400, 0),
    "anonymize": (900, 0),
}
ASR_INT8_GPU_FACTOR = 0.6               # int8_float16 on the card — about 60 % of float16
PREF_THREADS = {"asr": 4, "msgmodel": 2, "ner_model": 4, "natasha": 1, "documents": 1, "anonymize": 1}
GPU_CAP = {"asr": 4, "msgmodel": 2, "ner_model": 2}     # instances per card, at most
BATCH = {"asr": (1, 16), "msgmodel": (32, 128), "ner_model": (32, 64)}  # (CPU, GPU)
CPU_ONLY = ("natasha", "documents", "anonymize")


def _tier_from(d: dict) -> Tier:
    known = {f for f in Tier.__dataclass_fields__}
    return Tier(**{k: v for k, v in d.items() if k in known})


def load_tiers(model_dir: str | Path | None, dirs=None) -> tuple[list[Tier], str, Path | None]:
    """→ (tiers, default tier name, root). The root is the directory with index.json; without it —
    the directory with model.onnx (one tier "default", dir=""); nothing — ([], "", None). ``dirs`` —
    where to look without ``model_dir`` (by default — the message filter directories)."""
    cands = [Path(model_dir)] if model_dir else []
    if not model_dir:
        from . import msgmodel
        cands = list(dirs) if dirs is not None else msgmodel.default_dirs()
    for d in cands:
        idx = d / "index.json"
        if idx.is_file():
            try:
                data = json.loads(idx.read_text(encoding="utf-8"))
                tiers = [_tier_from(t) for t in data.get("tiers", []) if t.get("name")]
                if tiers:
                    default = data.get("default") or tiers[0].name
                    return tiers, default, d
            except (ValueError, TypeError, AttributeError):
                pass
        if (d / "model.onnx").is_file():
            return [Tier(name="default", dir="")], "default", d
    return [], "", None


def _reserve(cfg_reserve: int, phys: int) -> int:
    if cfg_reserve >= 0:
        return cfg_reserve
    return 0 if phys <= 2 else (1 if phys <= 8 else 2)


def _pick_devices(spec: str, res: Resources, note) -> list[str]:
    """"auto" → all cards; "cpu" → []; an explicit list → as given (cards that are not visible —
    with a warning: devpool moves them to CPU itself if they do not start)."""
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        return [g.device for g in res.gpus]
    out = []
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        if item == "cpu":
            out.append("cpu")
            continue
        if item in ("cuda", "rocm", "dml"):
            out += [g.device for g in res.gpus if g.device.startswith(item + ":")] or [item + ":0"]
            continue
        if res.gpu(item) is None:
            note(f"{item} is set in the config but not found — will try, falling back to CPU on failure")
        out.append(item)
    return out


def _cpu_slots(stage: str, budget: int, ram_mb: int, mem_mb: int, comp, workload_cap: int,
               pref_threads: int, batch: int, group_size: int = 0, compute_type: str = "") -> list[Slot]:
    t_pref = comp.threads_per_worker or pref_threads
    if group_size:
        t_pref = min(t_pref, group_size)
    if comp.cpu_workers:
        n = comp.cpu_workers
    else:
        n = max(1, budget // max(1, t_pref))
        if mem_mb:
            n = min(n, max(1, int(ram_mb // mem_mb)))
        if workload_cap:
            n = min(n, max(1, workload_cap))
    n = max(1, n)
    threads = comp.threads_per_worker or max(1, budget // n)
    if group_size:
        threads = min(threads, group_size)
    return [Slot("cpu", i, threads, batch, compute_type) for i in range(n)]


def _gpu_slots(stage: str, devices: list[str], res: Resources, comp, need_mb: float, batch: int,
               feeders: int, note, threads: int = 1, type_for=None) -> list[Slot]:
    """Instances by the free memory of each card; in total — no more than the "feeder" cores."""
    slots: list[Slot] = []
    for dev in devices:
        if dev == "cpu":
            continue
        g = res.gpu(dev)
        need = need_mb(g) if callable(need_mb) else need_mb
        if g is None or not g.mem_free_mb:
            n = comp.gpu_workers_per_device or 1            # memory unknown (DirectML, set explicitly)
            per = 0
        else:
            avail = g.mem_free_mb * comp.max_gpu_mem_frac
            n = comp.gpu_workers_per_device or min(GPU_CAP.get(stage, 1), int(avail // max(1.0, need)))
            if n < 1:
                note(f"{stage}: too little free memory on {dev} ({g.mem_free_mb} MB) — the card is not used")
                continue
            per = int(avail // n)
        ct = type_for(g) if type_for else ""
        slots += [Slot(dev, i, threads, batch, ct, per) for i in range(n)]
    if len(slots) > feeders > 0 and not comp.gpu_workers_per_device:
        # round-robin over the cards so that the cards do not idle when cores are short
        order = sorted(slots, key=lambda s: (s.instance, devices.index(s.device)))
        keep = set(id(s) for s in order[:feeders])
        slots = [s for s in slots if id(s) in keep]
    return slots


def _stage_devices(stage: str, cfg, res: Resources, note) -> list[str]:
    comp = cfg.compute
    if stage == "asr":
        d = (cfg.media.asr_device or "auto").strip().lower()
        if d != "auto":
            return _pick_devices(d, res, note)
    if stage == "ner_model":
        d = (cfg.ner.device or "auto").strip().lower()
        if d != "auto":
            return _pick_devices(d, res, note)
    return _pick_devices(comp.devices, res, note)


def _asr_model(cfg, on_gpu: bool, budget: int) -> str:
    m = (cfg.media.asr_model or "").strip()
    if m and m.lower() != "auto" and not (cfg.compute.asr_auto and m == "medium"):
        return m
    if on_gpu:
        return "large-v3"
    return "medium" if budget >= 8 else "small"


def _ort_gpu(res: Resources, gpu_devs: list[str], note, prefix: str, fallback: str) -> tuple[str, list[str]]:
    """onnxruntime provider for cards of one kind → (provider, cards); none — ("", []) and a note."""
    if not gpu_devs:
        return "", []
    kind = gpu_devs[0].split(":")[0]
    want = {"cuda": ["CUDAExecutionProvider"], "rocm": ["ROCMExecutionProvider", "MIGraphXExecutionProvider"],
            "dml": ["DmlExecutionProvider"]}.get(kind, [])
    provider = next((p for p in want if p in res.ort_providers), "")
    if not provider:
        fix = ""
        if kind == "cuda" and res.cuda_probe == "failed":
            fix = "; the CUDA probe session did not start or there is no cuDNN 9 — see the onnxruntime note"
        elif kind == "cuda":
            fix = ("; the fix is in the onnxruntime note" if ort_trap_note(res)
                   else f"; fix: {ort_fix_command(res.ort_packages)}")
        note(f"{prefix}onnxruntime without {want[0] if want else kind} — {fallback}{fix}")
        return "", []
    return provider, [d for d in gpu_devs if d.split(":")[0] == kind]


def plan_msgmodel(res: Resources, cfg, workload: Workload, budget: int, ram_mb: int,
                  gpu_devs: list[str], explicit_cpu: bool, note, group_size: int = 0) -> StagePlan:
    comp = cfg.compute
    tiers, default, root = load_tiers(getattr(cfg.messages, "model_dir", ""))
    sp = StagePlan("msgmodel")
    provider, gpu_devs = _ort_gpu(res, gpu_devs, note, "msgmodel: ", "message filter on CPU")
    by_name = {t.name: t for t in tiers}
    chosen: Tier | None = None
    want_name = (comp.msgmodel_tier or "").strip()
    if want_name:
        chosen = by_name.get(want_name)
        if chosen is None and tiers:
            note(f"msgmodel: no tier \"{want_name}\" in index.json — choosing automatically")

    def has_gpu_file(t: Tier) -> bool:
        from . import msgmodel
        return root is not None and msgmodel.gpu_file(root, t.name) is not None

    def gpu_fit(t: Tier) -> list[str]:
        if not has_gpu_file(t):
            return []                      # int8 (files.cpu) is not put on a card — CPU only
        out = []
        for d in gpu_devs:
            g = res.gpu(d)
            if g is None or not g.mem_free_mb or g.mem_free_mb * comp.max_gpu_mem_frac >= t.min_gpu_mem_gb * 1024:
                out.append(d)
        return out

    def rate(t: Tier, devs: list[str], cpu_threads: int) -> float:
        b = t.bench or {}
        if devs:
            known = b.get("gpu_msgs_per_s") or {}
            total = 0.0
            for d in devs:
                g = res.gpu(d)
                v = known.get(g.name) if g else None
                if v is None and known:
                    v = min(float(x) for x in known.values())         # the card was not measured: be careful
                total += float(v or 0)
            return total
        return float(b.get("cpu_msgs_per_s_per_core") or 0) * cpu_threads

    cpu_slots = _cpu_slots("msgmodel", budget, ram_mb, 0, comp, 0, PREF_THREADS["msgmodel"],
                           comp.batch_size or BATCH["msgmodel"][0], group_size)
    cpu_threads = sum(s.threads for s in cpu_slots)
    if chosen is None and tiers:
        fits = []
        for t in tiers:
            devs = gpu_fit(t)
            if gpu_devs and not devs and has_gpu_file(t):
                continue                   # there is a card file, but it does not fit: CPU is not considered
            if not devs and t.ram_mb_per_worker > ram_mb:
                continue
            fits.append((t, devs))
        if not fits:
            fits = [(min(tiers, key=lambda t: t.ram_mb_per_worker), [])]
            note("msgmodel: no tier fits in memory — taking the lightest")
        best = sorted(fits, key=lambda x: x[0].quality(), reverse=True)
        budget_s = float(comp.time_budget_hours or 0) * 3600
        if budget_s > 0 and workload.n_messages > 0:
            ok = []
            for t, devs in best:
                r = rate(t, devs, cpu_threads)
                if r <= 0 or workload.n_messages / r <= budget_s:
                    ok.append((t, devs))
            if ok:
                chosen = ok[0][0]
            else:
                chosen = max(best, key=lambda x: rate(x[0], x[1], cpu_threads))[0]
                note("msgmodel: no tier fits the time budget — taking the fastest")
        elif gpu_devs and any(devs for _, devs in best):
            chosen = next(t for t, devs in best if devs)
        elif slow_cpu(res):
            chosen = _lightest([t for t, _ in fits])
            note(f"msgmodel: slow CPU ({res.cpu_probe['msgs_per_s']} msgs/s on the probe, "
                 f"< {SLOW_CPU_MSGS_PER_S:g}) — tier {chosen.name}")
        else:
            dflt = [t for t, devs in fits if t.name == default]
            chosen = dflt[0] if dflt else best[0][0]
    if chosen is not None:
        sp.model = chosen.name
    devs = gpu_fit(chosen) if chosen else []
    if gpu_devs and not devs and (chosen is None or not has_gpu_file(chosen)):
        note(f"msgmodel: tier {chosen.name} has no card file (files.gpu / model.fp16.onnx) — "
             "message filter on CPU" if chosen else "msgmodel: no filter model — plan on CPU")
    if devs:
        need = (chosen.min_gpu_mem_gb if chosen else 1.0) * 1024
        sp.slots = _gpu_slots("msgmodel", devs, res, comp, need, comp.batch_size or BATCH["msgmodel"][1],
                              budget, note, threads=1)
        sp.provider = provider
    if not sp.slots or explicit_cpu:
        mem = chosen.ram_mb_per_worker if chosen else 500
        sp.slots += _cpu_slots("msgmodel", budget if not sp.slots else max(1, budget - len(sp.slots)), ram_mb,
                               mem, comp, _cap_items(workload.n_messages, 2048), PREF_THREADS["msgmodel"],
                               comp.batch_size or BATCH["msgmodel"][0], group_size)
        if not sp.provider:
            sp.provider = "CPUExecutionProvider"
    sp.reason = f"tier {sp.model or '—'}"
    return sp


def plan_nermodel(res: Resources, cfg, workload: Workload, budget: int, ram_mb: int,
                  gpu_devs: list[str], explicit_cpu: bool, note, group_size: int = 0) -> StagePlan:
    """Tier and slots of the name model by ``[ner].model``: ``off`` — a stage without slots; ``auto`` —
    the index ``default_gpu`` on a card (no card file or it does not fit — ``default``), on
    CPU — ``default`` (slow CPU by ``probe_cpu_speed`` — the lightest tier); a tier name — that tier (on a card —
    only with an fp16 file). The index ``org_tier`` (organizations only, a second pass) is not chosen as the
    names tier by ``auto``."""
    from . import nermodel
    comp = cfg.compute
    sp = StagePlan("ner_model")
    want = (cfg.ner.model or "auto").strip()
    if want == "off":
        sp.reason = "off ([ner].model = off)"
        return sp
    tiers, default, root = load_tiers("", dirs=nermodel.default_dirs())
    provider, gpu_devs = _ort_gpu(res, gpu_devs, note, "ner_model: ", "name model on CPU")
    by_name = {t.name: t for t in tiers}
    org = (nermodel.index(root) or {}).get("org_tier") if root is not None else None
    names_tiers = [t for t in tiers if t.name != org] or tiers

    def gpu_fit(t: Tier) -> list[str]:
        if root is None or nermodel.gpu_file(root, t.name) is None:
            return []                      # int8 is not put on a card
        return [d for d in gpu_devs if (g := res.gpu(d)) is None or not g.mem_free_mb
                or g.mem_free_mb * comp.max_gpu_mem_frac >= t.min_gpu_mem_gb * 1024]

    chosen = by_name.get(want) if want != "auto" else None
    if want != "auto" and chosen is None and tiers:
        note(f"ner_model: no tier \"{want}\" in index.json — taking the default tier")
    devs: list[str] = []
    if chosen is None and tiers:
        cands = []
        if gpu_devs:
            cands.append(by_name.get(nermodel.default_tier(root, gpu=True)))
        cands.append(by_name.get(default))
        for t in cands:
            if t is not None and gpu_fit(t):
                chosen = t
                break
        if chosen is None and slow_cpu(res):
            chosen = _lightest(names_tiers)
            note(f"ner_model: slow CPU ({res.cpu_probe['msgs_per_s']} msgs/s on the probe, "
                 f"< {SLOW_CPU_MSGS_PER_S:g}) — tier {chosen.name}")
        if chosen is None:
            chosen = by_name.get(default) or names_tiers[0]
    if chosen is not None:
        sp.model = chosen.name
        devs = gpu_fit(chosen)
        if gpu_devs and not devs:
            note(f"ner_model: tier {chosen.name} cannot go on a card (no fp16 file or too little memory) — on CPU")
    if devs:
        sp.slots = _gpu_slots("ner_model", devs, res, comp, chosen.min_gpu_mem_gb * 1024 or MEM["ner_model"][1],
                              comp.batch_size or BATCH["ner_model"][1], budget, note, threads=1)
        sp.provider = provider
    if not sp.slots or explicit_cpu:
        mem = (chosen.ram_mb_per_worker if chosen else 0) or MEM["ner_model"][0]
        sp.slots += _cpu_slots("ner_model", budget if not sp.slots else max(1, budget - len(sp.slots)), ram_mb,
                               mem, comp, _cap_items(workload.n_texts, 2048), PREF_THREADS["ner_model"],
                               comp.batch_size or BATCH["ner_model"][0], group_size)
        if not sp.provider:
            sp.provider = "CPUExecutionProvider"
    sp.reason = f"tier {sp.model}" if sp.model else "model not installed"
    return sp


def _cap_items(n: int, per_worker: int) -> int:
    """No more workers than work batches (0 — the amount is unknown)."""
    return math.ceil(n / per_worker) if n else 0


def plan(resources: Resources, config, workload: Workload | None = None, ctx=None) -> Plan:
    """Layout of the stages over the found compute (see the module header). No side effects
    except ``ctx.note`` about anything unusual."""
    res = resources
    workload = workload or Workload()
    comp = config.compute
    note = _Notes(ctx)
    cpu = res.cpu
    reserve = _reserve(comp.reserve_cores, cpu.usable_physical)
    budget = max(1, cpu.usable_physical - reserve)
    ram_mb = int(max(256, res.ram_available_mb or res.ram_total_mb // 2) * comp.max_ram_frac)
    gsize = cpu.group_size if cpu.processor_groups > 1 else 0
    pl = Plan(cpu_budget=budget, ram_budget_mb=ram_mb)

    # ASR (faster-whisper / ctranslate2): CUDA only on NVIDIA
    devs = _stage_devices("asr", config, res, note)
    gpu = [d for d in devs if d.startswith("cuda:")]
    if [d for d in devs if d not in gpu and d != "cpu"]:
        note("asr: faster-whisper works only with CUDA — other cards are not used")
    if gpu and not res.ct2_cuda_devices:
        note("asr: faster-whisper is not installed — transcribing on CPU" if res.ct2_cuda_devices is None
             else "asr: ctranslate2 without CUDA — transcribing on CPU")
        gpu = []
    model = _asr_model(config, bool(gpu), budget)
    sp = StagePlan("asr", model=model)
    if gpu:
        fp16 = MEM["asr"].get(model, MEM["asr"]["large-v3"])[1]

        def need(g):
            return fp16 * (ASR_INT8_GPU_FACTOR if g is not None and g.mem_total_mb and g.mem_total_mb < 8192 else 1)

        def ctype(g):
            return "int8_float16" if g is not None and g.mem_total_mb and g.mem_total_mb < 8192 else "float16"
        sp.slots = _gpu_slots("asr", gpu, res, comp, need, comp.batch_size or BATCH["asr"][1],
                              budget, note, threads=max(1, min(4, budget // max(1, len(gpu)))), type_for=ctype)
        if not sp.slots:
            model = sp.model = _asr_model(config, False, budget)
    if not sp.slots or "cpu" in devs and gpu:
        mem = MEM["asr"].get(model, MEM["asr"]["medium"])[0]
        sp.slots += _cpu_slots("asr", budget if not sp.slots else max(1, budget - len(sp.slots)), ram_mb, mem, comp,
                               workload.n_audio, PREF_THREADS["asr"], comp.batch_size or BATCH["asr"][0], gsize,
                               "int8")
    sp.reason = "GPU" if sp.on_gpu else "CPU"
    pl.stages["asr"] = sp

    # message filter
    devs = _pick_devices(comp.devices, res, note)
    pl.stages["msgmodel"] = plan_msgmodel(res, config, workload, budget, ram_mb,
                                          [d for d in devs if d != "cpu"], "cpu" in devs and len(devs) > 1,
                                          note, gsize)

    # name model (onnxruntime, models/ner)
    devs = _stage_devices("ner_model", config, res, note)
    pl.stages["ner_model"] = plan_nermodel(res, config, workload, budget, ram_mb,
                                           [d for d in devs if d != "cpu"], "cpu" in devs and len(devs) > 1,
                                           note, gsize)

    # CPU only: Natasha, documents (office/pdf), engine
    for name, cap in (("natasha", _cap_items(workload.n_texts, 64)),
                      ("documents", workload.n_documents),
                      ("anonymize", _cap_items(workload.n_texts, 64))):
        sp = StagePlan(name)
        sp.slots = _cpu_slots(name, budget, ram_mb, MEM[name][0], comp, cap, PREF_THREADS[name],
                              comp.batch_size or 1, gsize)
        sp.reason = "CPU"
        pl.stages[name] = sp
    if res.cpu.processor_groups > 1:
        note(f"Windows: {res.cpu.processor_groups} processor groups — worker processes are spread over the groups, "
             "threads per worker do not exceed the group size")
    pl.notes = note.items
    return pl


# ── stage hookup ───────────────────────────────────────────────────────────────────────

DETECT_TIMEOUT = 240
# Detection subprocess: the same detect, the result is JSON in the last stdout line. argv[1] — the directory
# holding the localanon package (running from sources without install).
DETECT_CHILD = """\
import json, sys
sys.path.insert(0, sys.argv[1])
from localanon import hw
print(json.dumps(hw.detect().to_dict(), ensure_ascii=False))
"""


def _detect_subprocess(timeout: float = DETECT_TIMEOUT) -> tuple[int | None, str, str]:
    try:
        p = subprocess.run([sys.executable, "-c", DETECT_CHILD, str(Path(__file__).resolve().parents[1])],
                           capture_output=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return None, "", ""
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def detect_isolated(ctx=None, *, run=None) -> Resources:
    """``detect`` in a subprocess: the main run process does not import ctranslate2, torch and onnxruntime
    for detection (a native crash in a probe does not crash the run; the ASR invariant — the model and CUDA
    only in workers). If it fails — ``detect`` without library probes (cards are visible, ASR and the filter
    on CPU) and a note. ``run`` — for tests."""
    # as in detect: workers (ASR, devpool) inherit the environment, and their cuda:N are the same numbers as in the plan
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    why = ""
    try:
        rc, out, err = (run or _detect_subprocess)()
        if rc == 0:
            res = Resources.from_dict(json.loads((out.strip().splitlines() or ["{}"])[-1]))
            for n in res.notes:
                if ctx is not None:
                    ctx.note("hw: " + n)
            return res
        why = f"no answer in {DETECT_TIMEOUT} s" if rc is None else f"code {rc}"
        tail = stderr_tail(err)
        if tail:
            why += f": {tail}"
    except Exception as e:  # noqa: BLE001 — detection never crashes the run
        why = _err(e)
    note = _Notes(ctx)
    note(f"compute detection in a subprocess failed ({why}) — without library probes: transcription "
         "and the message filter on CPU")
    res = detect(ctx, probe_libs=False)
    res.notes = note.items + res.notes
    return res


def resources(ctx) -> Resources:
    """Compute found once per run (``ctx.data["hw_resources"]``) — in a subprocess
    (``detect_isolated``)."""
    res = ctx.data.get("hw_resources")
    if res is None:
        res = ctx.data["hw_resources"] = detect_isolated(ctx)
    return res


def _dist_version(name: str) -> str:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:  # noqa: BLE001 — not installed or broken metadata
        return ""


def compute_summary(res: Resources, pl: Plan) -> dict:
    """Start of ``ctx.data["compute"]`` (→ ``local_manifest.json``): cards, CPU budget, versions of
    onnxruntime (each installed distribution) and ctranslate2. Numbers and package versions only —
    no paths, no device names of the machine."""
    out = {"gpus": len(res.gpus), "cpu_budget": pl.cpu_budget,
           "onnxruntime": dict(sorted((res.ort_packages or {}).items()))}
    ct2 = _dist_version("ctranslate2") if res.ct2_cuda_devices is not None else ""
    if ct2:
        out["ctranslate2"] = ct2
    return out


def stage_plan(ctx, stage: str, workload: Workload | None = None) -> StagePlan:
    """Plan of one stage given its amount of work. The stage calls this at the start of ``run(ctx)``:

        sp = hw.stage_plan(ctx, "asr", hw.Workload(n_audio=len(files), audio_seconds=…))

    The manifest summary (only numbers and the tier name) accumulates in ``ctx.data["compute"]``."""
    res = resources(ctx)
    pl = plan(res, ctx.config, workload, ctx=None)
    sp = pl.stages[stage]
    seen = ctx.data.setdefault("_hw_noted", set())
    for n in pl.notes:
        if n.startswith(stage + ":") and n not in seen:
            seen.add(n)
            ctx.note("hw: " + n)
    comp = ctx.data.get("compute")
    if comp is None:
        comp = ctx.data["compute"] = compute_summary(res, pl)
    comp[f"{stage}_workers"] = len(sp.slots)
    comp[f"{stage}_gpu_workers"] = sum(1 for s in sp.slots if s.device != "cpu")
    comp[f"{stage}_devices"] = sorted(sp.workers_per_device())
    if stage in ("msgmodel", "ner_model") and sp.model:
        comp[f"{stage}_tier"] = sp.model
        probe = res.cpu_probe or {}
        if probe.get("msgs_per_s"):
            comp["cpu_probe_msgs_per_s"] = probe["msgs_per_s"]
            comp["cpu_probe_rounds"] = probe.get("rounds", 1)
        msg = f"{stage}: tier {sp.model} on {', '.join(sp.devices()) or 'cpu'}" + (
            f"; CPU probe {probe['msgs_per_s']} msgs/s ({probe.get('tier')} int8, {probe.get('threads')} threads, "
            f"slow below {SLOW_CPU_MSGS_PER_S:g})" if probe.get("msgs_per_s") else "; no CPU probe")
        if msg not in seen:
            seen.add(msg)
            ctx.note("hw: " + msg)
    if stage == "asr" and sp.model:
        comp["asr_model"] = sp.model
    return sp


def resolve_device(ctx, stage: str) -> str:
    """One device for a stage that cannot use several (``auto`` → the first card of the plan
    or ``cpu``): ``cuda:0`` etc."""
    sp = stage_plan(ctx, stage)
    return sp.slots[0].device if sp.slots else "cpu"


# ── human-readable output ───────────────────────────────────────────────────────────────────────

def describe(res: Resources, pl: Plan) -> str:
    c = res.cpu
    lines = [f"OS: {res.os}, Python {res.python}",
             f"CPU: logical {c.logical}, physical {c.physical}, available {c.usable} logical / "
             f"{c.usable_physical} physical"
             + (f", affinity {c.affinity}" if c.affinity else "")
             + (f", cgroup quota {c.cgroup_quota:g}" if c.cgroup_quota else "")
             + (f", processor groups {c.processor_groups}" if c.processor_groups > 1 else "")
             + (f", NUMA nodes {c.numa_nodes}" if c.numa_nodes > 1 else ""),
             f"RAM: total {res.ram_total_mb} MB, available {res.ram_available_mb} MB"]
    if res.gpus:
        for g in res.gpus:
            mem = f"{g.mem_total_mb} MB, free {g.mem_free_mb} MB" if g.mem_total_mb else "memory unknown"
            lines.append(f"GPU {g.device}: {g.name} — {mem}" + (f", CC {g.compute_cap}" if g.compute_cap else ""))
    else:
        lines.append("GPU: none found — everything on CPU")
    lines.append("onnxruntime: " + (", ".join(res.ort_providers) if res.ort_providers else "not installed"))
    if res.torch:
        lines.append(f"torch {res.torch.get('version')}: CUDA {'yes' if res.torch.get('cuda') else 'no'}")
    if res.ct2_cuda_devices is not None:
        lines.append(f"ctranslate2 (faster-whisper): CUDA cards {res.ct2_cuda_devices}")
    if res.cpu_probe:
        p = res.cpu_probe
        lines.append(f"CPU probe: {p.get('msgs_per_s')} msgs/s ({p.get('tier')} int8, {p.get('threads')} threads, "
                     f"best of {p.get('rounds', 1)}; "
                     f"slow below {SLOW_CPU_MSGS_PER_S:g})" + (" — slow CPU, lightest tiers" if slow_cpu(res) else ""))
    lines.append(f"Plan (cores for workers {pl.cpu_budget}, RAM for workers {pl.ram_budget_mb} MB):")
    for name in STAGES:
        sp = pl.stages.get(name)
        if not sp:
            continue
        per = ", ".join(f"{d}×{n}" for d, n in sp.workers_per_device().items())
        s0 = sp.slots[0] if sp.slots else Slot("cpu")
        extra = f", model {sp.model}" if sp.model else ""
        if s0.compute_type and name == "asr" and s0.device != "cpu":
            extra += f", {sp.asr_kwargs()['compute_type']}"      # one type for all cards of the process
        elif s0.compute_type:
            extra += f", {s0.compute_type}"
        extra += f", {sp.provider}" if sp.provider else ""
        lines.append(f"  {name:<11} {per}; threads {s0.threads}, batch {s0.batch_size}{extra}")
    for n in res.notes + pl.notes:
        lines.append(f"  ! {n}")
    return "\n".join(lines)


# ── several machines: shards and merge ──────────────────────────────────────────────────────────

def parse_shard(spec: str) -> tuple[int, int]:
    """"k/N" → (k, N), 1 ≤ k ≤ N."""
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", spec or "")
    if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
        raise ValueError("--shard: expected k/N, 1 ≤ k ≤ N (for example, 1/2)")
    return int(m.group(1)), int(m.group(2))


def shard_of(rel: str, n: int) -> int:
    """Shard number 1…N by a stable hash of the relative path ('/' as the separator, NFC)."""
    import unicodedata
    key = unicodedata.normalize("NFC", rel.replace("\\", "/").strip("/"))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % n + 1


EXPORT_MARKERS = {"result.json": "Telegram"}   # a level file → the export refers to neighbouring folders
OS_JUNK = {"desktop.ini", "thumbs.db", ".ds_store", "__macosx"}


def is_os_junk(name: str) -> bool:
    """An OS or Office service file: desktop.ini, Thumbs.db (Explorer), .DS_Store, __MACOSX, ._* (macOS),
    ~$* (Office lock). Copies of one input on different machines differ by them — sharding and
    file counting do not see them."""
    n = name.lower()
    return n in OS_JUNK or n.startswith(("._", "~$"))


def _shard_level(work: Path) -> tuple[list[tuple[str, list[Path]]], str, list[Path]]:
    """→ (units, level kind, OS service files on the passed levels — wrappers and the level itself)."""
    work = Path(work)
    root = work
    junk: list[Path] = []
    while True:
        kids = []
        for c in sorted(root.iterdir()):
            (junk if is_os_junk(c.name) else kids).append(c)
        if len(kids) == 1 and kids[0].is_dir():
            root = kids[0]
            continue
        break
    key = (lambda p: p.relative_to(work).as_posix() if p != work else "")
    if any(c.is_file() and c.name.lower() in EXPORT_MARKERS for c in kids):
        return [(key(root), kids)], "export", junk
    dirs = [c for c in kids if c.is_dir()]
    files = [c for c in kids if not c.is_dir()]
    if not dirs:
        return [(key(f), [f]) for f in files], "files", junk
    return [(key(d), [d]) for d in dirs] + ([(key(root), files)] if files else []), "dirs", junk


def shard_units(work: Path) -> tuple[list[tuple[str, list[Path]]], str]:
    """Shard units → ([(hash key, paths)], level kind). Single-directory wrappers are skipped;
    OS service files (``is_os_junk``) count neither when looking for a wrapper nor as units.
    Kind "export": the level holds an export (Telegram ``result.json``) — it refers to the neighbouring
    photos/, files/, the exports stage must see them together: the level is one indivisible unit.
    Kind "dirs": each directory (a project, a chat) is a unit, the top-level files are one more shared
    unit (the key is the path of the level itself, it does not match the directories), even if there are CSV/JSON among them.
    Kind "files": the level has only files — each file is its own unit (this is how a single
    project or an export of standalone files is split)."""
    units, kind, _ = _shard_level(work)
    return units, kind


def _count_files(u: Path) -> int:
    """Files of a unit without OS service files (and anything inside __MACOSX)."""
    if not u.is_dir():
        return 1
    return sum(1 for p in u.rglob("*")
               if p.is_file() and not any(is_os_junk(s) for s in p.relative_to(u).parts))


def select_shard(work: Path, k: int, n: int, note=None) -> dict:
    """Keep only shard k of N in the working copy (the removal is not counted as removed: this is not
    cleaning but splitting the work). → {"units", "kept", "dropped"} — file and unit counts; OS service
    files are not counted (``merge`` checks ``units`` and ``kept + dropped`` of the shards), those on the
    passed levels stay only in shard 1 (otherwise merge sees one path in several shards).
    ``note(msg)`` receives warnings: the export is indivisible, fewer units than N, the shard is empty."""
    import shutil
    work = Path(work)
    units, kind, junk = _shard_level(work)
    kept = dropped = 0
    for rel, paths in units:
        nfiles = sum(_count_files(u) for u in paths)
        if shard_of(rel, n) == k:
            kept += nfiles
            continue
        dropped += nfiles
        for u in paths:
            shutil.rmtree(u) if u.is_dir() else u.unlink()
    if k != 1:
        for u in junk:
            shutil.rmtree(u) if u.is_dir() else u.unlink()
    warn = note or (lambda m: None)
    if n > 1 and kind == "export":
        what = next(EXPORT_MARKERS[c.name.lower()] for c in units[0][1] if c.name.lower() in EXPORT_MARKERS)
        warn(f"shards: the input is one {what} export, it cannot be split by files (attachments are linked to it) — "
             f"it is whole in one shard, the other {n - 1} are empty")
    elif len(units) < n:
        warn(f"shards: {len(units)} split units is fewer than {n} shards — some shards are empty")
    if kept == 0:
        warn(f"shards: shard {k}/{n} is empty — nothing to process on this machine")
    return {"units": len(units), "kept": kept, "dropped": dropped}


MANIFEST = "local_manifest.json"
_SUM_KEYS = ("counters",)


def _merge_manifests(ms: list[dict]) -> dict:
    """Counters are added up; other fields must match (version, salt fingerprint), otherwise an error."""
    out: dict = {}
    for m in ms:
        for k, v in m.items():
            if k == "shard":
                continue
            if isinstance(v, dict) and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                           for x in v.values()) and k in ("counters", "removed", "masked"):
                acc = out.setdefault(k, {})
                for ck, cv in v.items():
                    acc[ck] = acc.get(ck, 0) + cv
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and k.startswith(("n_", "count")):
                out[k] = out.get(k, 0) + v
            elif k == "compute":
                out.setdefault("compute_shards", []).append(v)
            elif k in out and out[k] != v:
                raise ValueError(f"merge: manifest field \"{k}\" differs between shards (different versions or salt)")
            else:
                out.setdefault(k, v)
    out["shards"] = len(ms)
    return out


def _check_shard_set(mans: list[dict]) -> list[int]:
    """The outputs are a full shard set of one split: all have ``shard`` {k, n}, the same n and exactly
    one output for each k in 1…n (a missing shard — an incomplete archive, a repeated one — doubled
    counters) and one input: all have the same ``units`` and ``kept + dropped`` (otherwise the shards
    split different copies of the input and files are lost or doubled). → k numbers in output order; else ValueError."""
    ks, ns, inputs = [], set(), []
    for i, m in enumerate(mans, 1):
        s = m.get("shard")
        s = s if isinstance(s, dict) else {}
        k, n = s.get("k"), s.get("n")
        if not (isinstance(k, int) and isinstance(n, int) and 1 <= k <= n):
            raise ValueError(f"merge: output {i} is not a shard (no shard k/N in the manifest; a run --shard is needed)")
        ks.append(k)
        ns.add(n)
        inputs.append(tuple(s.get(f) for f in ("units", "kept", "dropped")))
    if len(ns) != 1:
        raise ValueError(f"merge: shards of different splits (N = {', '.join(map(str, sorted(ns)))}) — "
                         "outputs of one k/N split are needed")
    n = ns.pop()
    dup = sorted({k for k in ks if ks.count(k) > 1})
    if dup:
        raise ValueError(f"merge: shard {', '.join(f'{k}/{n}' for k in dup)} given more than once")
    missing = [k for k in range(1, n + 1) if k not in ks]
    if missing:
        raise ValueError(f"merge: missing shards {', '.join(f'{k}/{n}' for k in missing)} — "
                         f"{len(ks)} of {n} given")
    bad = [i for i, t in enumerate(inputs, 1) if not all(isinstance(x, int) for x in t)]
    if bad:
        raise ValueError(f"merge: the manifest of output {', '.join(map(str, bad))} has no shard unit and file counts "
                         "(units, kept, dropped) — the input cannot be checked; rebuild the shard with this version")
    sizes = [(u, kept + dropped) for u, kept, dropped in inputs]
    if len(set(sizes)) != 1:
        raise ValueError("merge: shard inputs differ (split units: "
                         f"{', '.join(str(u) for u, _ in sizes)}; files: {', '.join(str(f) for _, f in sizes)}) — "
                         "the shards ran on different copies of the input; rebuild all shards from one copy")
    return ks


def merge_outputs(dirs: list[Path], out: Path) -> dict:
    """Shard outputs (directories with ``*_clean.zip``, ``local_manifest.json``, ``report.html``) → one:
    the archives are merged (a matching path with different content is an error), manifest counters
    are added up, the reports are put alongside as ``report.shard-k.html``. The set of outputs must be
    a full shard set of one split (``_check_shard_set``). All checks (the set, the manifests, the
    content) happen before files in ``out`` are replaced: the archive is built in a temporary file, on
    an error the previous output in ``out`` is unchanged. A file or link ``*_clean.zip.part`` in ``out``
    (the rest of a killed merge) is removed at the start; directories and other names (including another
    case) are not touched, if removal fails — a note in ``notes``. → a summary (numbers and notes without paths)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    notes = []
    for stale in out.glob("*_clean.zip.part"):     # .part of a killed merge (any archive name)
        # Windows glob is case-insensitive; a directory with this name is not ours
        if not stale.name.endswith("_clean.zip.part") or not (stale.is_symlink() or stale.is_file()):
            continue
        try:
            stale.unlink(missing_ok=True)
        except OSError as e:
            notes.append(f"merge: the rest of an interrupted merge in --out was not removed ({_err(e)})")
    zips, mans = [], []
    for d in dirs:
        d = Path(d)
        z = sorted(d.glob("*_clean.zip"))
        if len(z) != 1:
            raise ValueError(f"merge: shard output {len(zips) + 1} must have one *_clean.zip, found {len(z)}")
        zips.append(z[0])
        mp = d / MANIFEST
        man = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
        if not isinstance(man, dict):
            raise ValueError("merge: shard manifest cannot be read")
        mans.append(man)
    ks = _check_shard_set(mans)
    merged = _merge_manifests(mans)
    names = {z.name for z in zips}
    name = names.pop() if len(names) == 1 else zips[0].name
    seen: dict[str, str] = {}
    target = out / name
    tmp = target.with_name(target.name + ".part")
    tmp_man = out / (MANIFEST + ".part")
    try:                                 # any error (a conflict, a broken shard archive) — no .part left in out
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for z in zips:
                with zipfile.ZipFile(z) as zin:
                    for info in zin.infolist():
                        data = zin.read(info) if not info.is_dir() else b""
                        h = hashlib.sha256(data).hexdigest()
                        if info.filename in seen:
                            if seen[info.filename] != h:
                                raise ValueError("merge: the same path in two shards with different content")
                            continue
                        seen[info.filename] = h
                        zout.writestr(info, data)
        tmp_man.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)
        tmp_man.replace(out / MANIFEST)
    finally:
        tmp.unlink(missing_ok=True)
        tmp_man.unlink(missing_ok=True)
    for k, d in zip(ks, dirs):
        rep = Path(d) / "report.html"
        if rep.is_file():
            (out / f"report.shard-{k}.html").write_bytes(rep.read_bytes())
    return {"shards": len(dirs), "files": sum(1 for k in seen if not k.endswith("/")), "notes": notes}
