"""Document views of one run shared by the stages and their worker processes.

A view (PDF extraction with OCR, OOXML view lines) depends only on the file bytes, so the key is
their sha256 plus the kind and options: a document that a later stage or another worker process
reads again unchanged is not parsed (and recognized) again; a rewritten document has another key.

The directory is set by the run (``enable`` in ``cli``) for the parent and the worker processes it
starts (environment ``LOCALANON_VIEW_CACHE``); without it nothing is stored and every read parses
the file, as before. The directory lies in the run's temporary directory next to the working copy
and goes away with it. Entries are written atomically: two processes may write the same key.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path

ENV = "LOCALANON_VIEW_CACHE"
VERSION = 1


def enable(directory) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    os.environ[ENV] = str(d)
    return d


def disable() -> None:
    os.environ.pop(ENV, None)


def directory() -> Path | None:
    d = os.environ.get(ENV)
    return Path(d) if d else None


def key(kind: str, data: bytes, *opts) -> str:
    h = hashlib.sha256(f"{VERSION}\0{kind}\0{opts!r}\0".encode("utf-8"))
    h.update(data)
    return h.hexdigest()


def get(k: str):
    """The stored value or None (no cache, no entry, an unreadable entry)."""
    d = directory()
    if d is None:
        return None
    try:
        with open(d / k, "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — a broken entry is read as missing: the view is parsed again
        return None


def put(k: str, value) -> None:
    d = directory()
    if d is None:
        return
    try:
        fd, tmp = tempfile.mkstemp(prefix=".view.", dir=str(d))
    except OSError:
        return
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, d / k)
    except OSError:
        pass
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
