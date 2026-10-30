"""Model download at installation (the only place where network is allowed).

    python -m localanon.fetch --asr medium|auto [--gliner] [--dir <directory>]
    python -m localanon.fetch --ensure-models [<repository root>]    # offline: models/ in place, else unpack the archive
    python -m localanon.fetch --asr auto --check                     # offline: is whisper for this machine cached

Models go to the project cache directory in the Hugging Face layout (``models--<org>--<name>/…``):
by default ``~/.localanon/models`` on Linux and ``%LOCALAPPDATA%\\localanon\\models`` on Windows;
the ``LOCALANON_MODELS`` variable overrides the path (for runs from a non-standard directory).
While the pipeline runs, models are taken only from here (``local_files_only``), without network.

gliner-x-large (``knowledgator/gliner-x-large``, ``[ner].gliner``) goes to a plain directory
``<models>/gliner-x-large`` (not the HF cache): its ``gliner_config.json`` gets the "whitespace" word
splitter — the native stanza downloads its own models from the network on first start.
``gliner_path()`` returns the directory or None — without network.

The name model and the message filter are not downloaded: they come as a separate archive
(``localanon-models-<date>.zip``, tools/release.py) that is unpacked into ``models/`` of the client
repository; the package is installed from the clone in editable mode (run.sh / run.ps1).
``models.lock.json`` in the repository root lists every file of the archive with its size and sha256.
``--ensure-models`` checks ``models/`` against it (sizes only, fast) and, if files are missing, unpacks
the archive lying in the repository folder, checking sha256 — without network. ``--asr auto`` picks the
whisper size the run plan will use on this machine
(localanon/hw.py: large-v3 on a CUDA card, medium or small on CPU); ``--asr auto --check`` tells offline
whether that size is in the cache — run.sh / run.ps1 call it on every start and download the model when
the hardware changed after the installation.

Heavy libraries (huggingface_hub, faster_whisper) are imported only inside functions: the module is
also imported by stages that need only the directory path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

ENV_MODELS_DIR = "LOCALANON_MODELS"
GLINER_REPO = "knowledgator/gliner-x-large"
GLINER_DIR = "gliner-x-large"
LOCK_FILE = "models.lock.json"           # in the repository root: the files of the models archive
LOCK_STAMP = ".lock.sha256"              # in models/: sha256 of the lock the files were last checked against
ARCHIVE_GLOB = "localanon-models*.zip"
ASR_MISSING = 3                         # exit code of ``--asr … --check``: the model is not in the cache
# Copy of the faster_whisper.utils._MODELS table for the sizes we need: it finds the model in the cache
# without loading faster-whisper (available() is called on every run).
WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def default_models_dir() -> Path:
    """Models directory, ignoring the environment variable."""
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "localanon" / "models"
    return Path.home() / ".localanon" / "models"


def models_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Models directory: explicit argument → ``LOCALANON_MODELS`` → the default directory."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(ENV_MODELS_DIR, "").strip()
    if env:
        return Path(env).expanduser()
    return default_models_dir()


def whisper_repo(model: str) -> str:
    """Hugging Face repository name for a model size ("medium") or an id like "org/name"."""
    if "/" in model:
        return model
    repo = WHISPER_REPOS.get(model)
    if repo is None:
        raise ValueError(f"unknown whisper model {model!r}; expected one of: "
                         + ", ".join(sorted(WHISPER_REPOS)))
    return repo


def _snapshot_dir(root: Path, repo: str, required: tuple[str, ...],
                  any_of: tuple[str, ...] = ()) -> Path | None:
    """Snapshot directory of a model in the Hugging Face cache layout that has all ``required``
    files and at least one of ``any_of`` (not broken symlinks); several matches — the newest.
    Without network and without libraries."""
    base = root / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not base.is_dir():
        return None
    best, best_mtime = None, -1.0
    for snap in base.iterdir():
        if not snap.is_dir():
            continue
        if not all((snap / name).is_file() for name in required):
            continue
        if any_of and not any((snap / name).is_file() for name in any_of):
            continue
        try:
            mtime = snap.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best, best_mtime = snap, mtime
    return best


WHISPER_FILES = ("config.json", "model.bin", "tokenizer.json")
GLINER_CONFIG = "gliner_config.json"
GLINER_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def whisper_path(model: str = "medium", root: str | os.PathLike | None = None) -> Path | None:
    """Directory of the whisper model in the cache or None. ``model`` may also be a ready directory
    with ``model.bin``."""
    p = Path(model).expanduser()
    if (p / "model.bin").is_file():
        return p
    try:
        repo = whisper_repo(model)
    except ValueError:
        return None
    return _snapshot_dir(models_dir(root), repo, WHISPER_FILES)


def gliner_path(root: str | os.PathLike | None = None) -> Path | None:
    """gliner-x-large directory (whitespace word splitter) or None if not downloaded."""
    d = models_dir(root) / GLINER_DIR
    if not (d / GLINER_CONFIG).is_file() or not any((d / w).is_file() for w in GLINER_WEIGHTS):
        return None
    try:
        cfg = json.loads((d / GLINER_CONFIG).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if cfg.get("words_splitter_type") == "whitespace" else None


def gliner_whitespace(d: Path) -> None:
    """gliner word splitter → whitespace (no stanza and its models from the network)."""
    p = Path(d) / GLINER_CONFIG
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["words_splitter_type"] = "whitespace"
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class ModelsError(Exception):
    """The models archive cannot be used (wrong archive, broken file, no space)."""


def _safe_rel(rel: str) -> bool:
    """A lock entry is a relative path inside ``models/`` (no ``..``, no absolute paths)."""
    p = PurePosixPath(rel)
    return (not p.is_absolute() and p.parts[:1] == ("models",) and ".." not in p.parts
            and "\\" not in rel and ":" not in rel)


def read_lock(root: str | os.PathLike) -> dict | None:
    """``models.lock.json`` of the repository: ``{"archive": name, "files": {rel: {"size", "sha256"}}}``
    or None (no lock — a development checkout, models are not shipped)."""
    p = Path(root) / LOCK_FILE
    if not p.is_file():
        return None
    lock = json.loads(p.read_text(encoding="utf-8"))
    files = lock.get("files") or {}
    bad = [rel for rel in files if not _safe_rel(rel)]
    if bad:
        raise ModelsError(f"{p}: paths outside models/: {', '.join(bad[:3])}")
    return lock


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_digest(root: str | os.PathLike) -> str:
    return hashlib.sha256((Path(root) / LOCK_FILE).read_bytes()).hexdigest()


def stamp_ok(root: str | os.PathLike) -> bool:
    """``models/`` was checked against this very lock (no hashing needed on every start)."""
    try:
        return (Path(root) / "models" / LOCK_STAMP).read_text(encoding="utf-8").strip() == lock_digest(root)
    except OSError:
        return False


def write_stamp(root: str | os.PathLike) -> None:
    (Path(root) / "models").mkdir(exist_ok=True)
    (Path(root) / "models" / LOCK_STAMP).write_text(lock_digest(root) + "\n", encoding="utf-8")


def missing_models(root: str | os.PathLike, lock: dict, deep: bool = False) -> list[str]:
    """Files of the lock that are absent from ``models/`` or have another size; ``deep`` — also those whose
    sha256 differs (after an update the lock may name a new file of the same size)."""
    out = []
    for rel, meta in sorted((lock.get("files") or {}).items()):
        p = Path(root) / rel
        try:
            if p.stat().st_size != int(meta["size"]) or (deep and _sha256(p) != meta["sha256"]):
                out.append(rel)
        except OSError:
            out.append(rel)
    return out


def find_archive(root: str | os.PathLike, lock: dict) -> Path | None:
    """The models archive in the repository folder: the name from the lock, else the newest
    ``localanon-models*.zip``."""
    root = Path(root)
    named = root / str(lock.get("archive") or "")
    if lock.get("archive") and named.is_file():
        return named
    found = sorted((p for p in root.glob(ARCHIVE_GLOB) if p.is_file()), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def unpack_models(archive: str | os.PathLike, root: str | os.PathLike, lock: dict,
                  todo: list[str] | None = None) -> int:
    """Extracts the files of the lock (``todo``, default: the missing ones) from the archive into ``root``,
    checking size and sha256 of each one; other members of the archive are ignored. A file is written next
    to its place and renamed only after the check, so an interrupted unpack leaves no broken model.
    Returns the number of files written."""
    root = Path(root)
    files = lock.get("files") or {}
    todo = missing_models(root, lock) if todo is None else todo
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        absent = [rel for rel in todo if rel not in names]
        if absent:
            raise ModelsError(f"{Path(archive).name} is not the models archive for this version of the code "
                              f"(no {absent[0]}); the right one is {lock.get('archive')}")
        need = sum(int(files[rel]["size"]) for rel in todo)
        free = shutil.disk_usage(root).free
        if need > free:
            raise ModelsError(f"not enough disk space: the models need {need / 2**30:.1f} GB, "
                              f"free {free / 2**30:.1f} GB")
        for rel in todo:
            meta = files[rel]
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            h, size = hashlib.sha256(), 0
            with z.open(rel) as src, open(part, "wb") as out:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    h.update(chunk)
                    size += len(chunk)
                    out.write(chunk)
            if size != int(meta["size"]) or h.hexdigest() != meta["sha256"]:
                part.unlink(missing_ok=True)
                raise ModelsError(f"{rel} in {Path(archive).name} does not match {LOCK_FILE} — the archive is "
                                  f"damaged or belongs to another version (the right one is {lock.get('archive')})")
            os.replace(part, dest)
    return len(todo)


def ensure_models(root: str | os.PathLike | None = None) -> int:
    """0 — the models of ``models.lock.json`` are in place (unpacked now from the archive in the repository
    folder if they were not), 1 — they are missing and there is no usable archive. Without a lock (a
    development checkout) — 0, whatever is in ``models/``."""
    root = Path(root) if root else repo_root()
    try:
        lock = read_lock(root)
        if lock is None:
            if not (root / "models").is_dir():
                print(f"fetch: no models/ in {root} — the name model and the message filter will be off",
                      file=sys.stderr)
            return 0
        checked = stamp_ok(root)
        missing = missing_models(root, lock, deep=not checked)
        if not missing:
            if not checked:
                write_stamp(root)
            return 0
        archive = find_archive(root, lock)
        if archive is None:
            print(f"fetch: the models are not in {root / 'models'} ({len(missing)} of {len(lock['files'])} files "
                  f"missing).\nThey come as a separate archive, {lock.get('archive')}: put it into {root} "
                  f"(do not unpack it) and start again.", file=sys.stderr)
            return 1
        total = sum(int(lock["files"][rel]["size"]) for rel in missing)
        print(f"unpacking the models from {archive.name}: {len(missing)} files, {total / 2**30:.1f} GB …", flush=True)
        n = unpack_models(archive, root, lock, missing)
        left = missing_models(root, lock)
        if left:
            raise ModelsError(f"{len(left)} files are still missing after unpacking, e.g. {left[0]}")
        write_stamp(root)
        print(f"  done: {n} files in {root / 'models'} (the archive is no longer needed and can be deleted)")
        return 0
    except (ModelsError, zipfile.BadZipFile, OSError, ValueError, KeyError) as exc:
        print(f"fetch: models: {exc}", file=sys.stderr)
        return 1


def auto_asr_model() -> str:
    """Whisper size the run plan picks on this machine with the default config. The CPU speed probe is
    skipped: the size depends on the card and the core count only."""
    from . import hw
    from .config import Config
    return hw.plan(hw.detect(speed_probe=lambda note, res: {}), Config()).stages["asr"].model


def check_asr(model: str, root: str | os.PathLike | None = None) -> int:
    """Offline: 0 — the whisper model is in the cache, ``ASR_MISSING`` — it is not (run.sh / run.ps1 then
    download it: the hardware changed after the installation or the cache was removed)."""
    if model == "auto":
        model = auto_asr_model()
    if model in ("", "-") or whisper_path(model, root) is not None:
        return 0
    print(f"fetch: whisper {model} for this machine is not in {models_dir(root)}", file=sys.stderr)
    return ASR_MISSING


# ── download (network) ────────────────────────────────────────────────────────

def _online_hub():
    """huggingface_hub with offline mode off: fetch is the only command with network, while the run
    environment may have HF_HUB_OFFLINE=1 (asr.py sets it itself)."""
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(var, None)
    if "huggingface_hub" in sys.modules:
        import huggingface_hub
        import huggingface_hub.constants as c
        c.HF_HUB_OFFLINE = False
        return huggingface_hub
    try:
        import huggingface_hub
    except ImportError as exc:
        raise RuntimeError("no huggingface_hub package (it comes with faster-whisper): delete .venv and "
                           "start run.sh / run.ps1 again") from exc
    return huggingface_hub


def fetch_whisper(model: str, root: str | os.PathLike | None = None) -> Path:
    """Download a whisper model (CTranslate2 conversion by Systran) to the cache directory."""
    hub = _online_hub()
    repo = whisper_repo(model)
    dest = models_dir(root)
    dest.mkdir(parents=True, exist_ok=True)
    # Contents of Systran/faster-whisper-*: model.bin, config.json, tokenizer.json, vocabulary.txt
    # (some also have preprocessor_config.json, vocabulary.json).
    path = hub.snapshot_download(
        repo, cache_dir=str(dest),
        allow_patterns=["config.json", "preprocessor_config.json", "model.bin",
                        "tokenizer.json", "vocabulary.*"])
    return Path(path)


def fetch_gliner(root: str | os.PathLike | None = None) -> Path:
    """Download gliner-x-large to ``<models>/gliner-x-large`` (without onnx/ — a second copy of the
    weights), word splitter → whitespace, then load the model once without network: everything it
    needs is already in place (otherwise the error shows here, not during a run)."""
    hub = _online_hub()
    dest = models_dir(root) / GLINER_DIR
    dest.mkdir(parents=True, exist_ok=True)
    hub.snapshot_download(GLINER_REPO, local_dir=str(dest), ignore_patterns=["onnx/*", "*.onnx"])
    gliner_whitespace(dest)
    from gliner import GLiNER
    GLiNER.from_pretrained(str(dest), local_files_only=True)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m localanon.fetch",
        description="Download models for LocalAnonPipeline (the only step with network)")
    ap.add_argument("--asr", metavar="MODEL", default=None,
                    help="whisper model: tiny | base | small | medium (default) | large-v3 | "
                         "org/name | auto (size by this machine); \"-\" — do not download")
    ap.add_argument("--gliner", action="store_true",
                    help="download gliner-x-large for [ner].gliner (needs the gliner extra: "
                         ".venv/bin/python -m pip install -e \".[gliner]\")")
    ap.add_argument("--ensure-models", metavar="ROOT", nargs="?", const="", default=None,
                    help=f"offline: check models/ of the repository against {LOCK_FILE}; if files are missing, "
                         f"unpack the models archive ({ARCHIVE_GLOB}) lying in the repository folder")
    ap.add_argument("--check", action="store_true",
                    help=f"with --asr: offline check that the model is in the cache (exit code {ASR_MISSING} "
                         "if not), nothing is downloaded")
    ap.add_argument("--dir", default=None,
                    help=f"models directory (default {default_models_dir()} or "
                         f"${ENV_MODELS_DIR})")
    ns = ap.parse_args(argv)
    if ns.ensure_models is not None:
        return ensure_models(ns.ensure_models or None)
    asr = ns.asr
    if ns.check:
        return check_asr(asr or "medium", ns.dir)
    if asr is None and not ns.gliner:
        asr = "medium"
    rc = 0
    try:
        if asr == "auto":
            asr = auto_asr_model()
            print(f"whisper size for this machine: {asr}")
        if asr and asr != "-":
            print(f"whisper {asr} → {models_dir(ns.dir)} …", flush=True)
            p = fetch_whisper(asr, ns.dir)
            # Check after the download: the same lookup as asr.available() during a run
            if whisper_path(asr, ns.dir) is None:
                print(f"fetch: whisper model {asr} is not downloaded completely ({p}): no "
                      f"{' / '.join(WHISPER_FILES)} — run the command again", file=sys.stderr)
                rc = 1
            else:
                print(f"  done: {p}")
        if ns.gliner:
            print(f"gliner-x-large → {models_dir(ns.dir) / GLINER_DIR} …", flush=True)
            p = fetch_gliner(ns.dir)
            if gliner_path(ns.dir) is None:
                print(f"fetch: gliner-x-large is not downloaded completely ({p}): no {GLINER_CONFIG} or weights "
                      f"({' / '.join(GLINER_WEIGHTS)}) — run the command again", file=sys.stderr)
                rc = 1
            else:
                print(f"  done: {p}")
    except Exception as exc:  # noqa: BLE001 — one clear line instead of a traceback
        print(f"fetch: failed: {exc.__class__.__name__}: {str(exc)[:300]}",
              file=sys.stderr)
        rc = 1
    if ns.dir and rc == 0:
        print(f"non-standard directory: set {ENV_MODELS_DIR}={ns.dir} before running")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
