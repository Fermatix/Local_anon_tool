"""Units of the computation store — for incremental recomputation.

A round computed with a store (Pipeline.run(capture=…)) puts, next to the finished
outputs, the report state of EVERY processing unit — a whole file or a part of
a large structured file — into ``<store>/units/<name>.json.gz``.
The next computation of the same tree with a pack whose rules were only
REMOVED (keep verdicts after the gate block) takes a unit from the store if
none of the removed rules fired in it (no record, no error) —
its output and report are exactly the same; the rest is recomputed (Pipeline,
_incremental_check). Reports hold text excerpts (examples) — files are
0600, like the registry cache. The sha256 of every unit is in the metadata (unit_sha):
the next computation checks them before the first task; damage → full recomputation.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import zlib

from .report import Report
from .walk import _mk_tmp

UNITS = "units"
# what reading a damaged unit raises (truncated/broken gzip, not JSON,
# wrong report schema)
READ_ERRORS = (OSError, EOFError, zlib.error, ValueError, KeyError, TypeError,
               AttributeError)


def unit_path(store: str, name: str) -> str:
    return os.path.join(store, UNITS, name + ".json.gz")


def file_sha(path: str):
    """sha256 of the file (hex) or None — no file or it cannot be read."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def write_state(path: str, data) -> None:
    """json.gz atomically (tmp + os.replace), 0600; ensure_ascii — lone
    surrogates in excerpts survive the round trip intact."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = _mk_tmp(path)       # short name, not derived from the unit name
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3,
                               mtime=0) as gz:
                gz.write(json.dumps(data, ensure_ascii=True,
                                    separators=(",", ":")).encode("ascii"))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_state(path: str):
    with gzip.open(path, "rb") as gz:
        return json.loads(gz.read().decode("ascii"))


def write_unit(store: str, name: str, report: Report) -> None:
    write_state(unit_path(store, name), report.to_state())


def read_unit(store: str, name: str) -> Report:
    return Report.from_state(read_state(unit_path(store, name)))


def fired(report: Report) -> list:
    """Rules that left a trace in the unit: a record (counter), an error or
    the shadow of a discarded pass (Report.trace). A rule without a trace does
    not change the unit when removed (same text at every step of the chain, same
    rollback decisions — see Pipeline._incremental_check)."""
    return sorted(report.trace())
