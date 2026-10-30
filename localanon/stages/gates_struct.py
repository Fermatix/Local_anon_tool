"""Stage "gates_struct".

Structural checks of the output before packing: each ``.json`` is parsed whole,
``.jsonl``/``.ndjson`` line by line; docx/xlsx/pptx (and odt/ods/odp) are opened as zip and all
XML parts are parsed; a PDF is opened (if pymupdf is installed, otherwise a note); text files
(the text and structured kinds from the inventory extension table) are read as strict UTF-8.
Violations → ``ctx.data['gate_failures']``: ``{'rel', 'check', 'detail'}`` without file content
(detail holds only the error type and position). They do not stop the run: the archive is built
anyway, the violations go to the report and their count to the manifest (cli). A ``.json`` holding
one JSON value per line (JSON Lines under a ``.json`` name) is intact.

A ``.json``/``.jsonl``/``.ndjson`` that does not parse already in the input gets the
``input_broken:<check>`` mark from the inventory stage (``mark_input_broken``, before any
processing): it is masked as text like any file, this stage does not check it (not our defect),
counter ``gate:skipped_input_broken``.

A file with an extension outside the table (``.DS_Store``, ``data.dat``, no extension): if it reads
as UTF-8, it is text and stays (text with a BOM or in cp1251/cp866/koi8-r has already been
re-encoded by the inventory stage); otherwise it cannot be checked for personal data — it is not
removed (data owner's decision of 27.10), it stays with the ``unprocessed:unknown_binary`` mark
("not anonymized" mark, the control gate does not check its content); this is not a violation. A
path from ``[documents].restore`` (as in the report, ``/`` or ``\\``) stays without a mark — a
note "kept by restore: N". The stage goes last, after all other stages.

A file with the ``unprocessed`` mark ("not anonymized": a broken or unreadable document, an archive
not unpacked) is not checked: it was left as it was in the input, its structure is not our
processing result; counter ``gate:skipped_unprocessed``.

The files are read and parsed in forked processes over the CPU slots of the hw plan
(engine_adapter.fork_map); marks, violations and notes are made here in file order — the result is
that of one pass.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from ..context import UNPROCESSED
from .inventory import kind_of, known_ext

NAME = "gates_struct"
UNKNOWN_CATEGORY = "unprocessed:unknown_binary"
INPUT_BROKEN = "input_broken"          # mark category key: the file came broken in the input

OOXML_EXT = {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}
JSON_EXT = {".json"}
JSONL_EXT = {".jsonl", ".ndjson"}
_UTF8_CHUNK = 1 << 20


def _fail(ctx, rel: str, check: str, detail: str) -> None:
    ctx.data.setdefault("gate_failures", []).append(
        {"rel": rel, "check": check, "detail": detail[:200]})
    ctx.count("gate:" + check)


def _utf8_ok(path: Path) -> str | None:
    """None — the file reads as UTF-8 (a BOM is allowed), otherwise a short description of the place."""
    import codecs
    dec = codecs.getincrementaldecoder("utf-8")()
    pos = 0
    try:
        with open(path, "rb") as fh:
            first = True
            while True:
                chunk = fh.read(_UTF8_CHUNK)
                if first:
                    first = False
                    if chunk.startswith(codecs.BOM_UTF8):
                        chunk = chunk[len(codecs.BOM_UTF8):]
                        pos = len(codecs.BOM_UTF8)
                try:
                    if not chunk:
                        dec.decode(b"", final=True)
                        return None
                    dec.decode(chunk)
                except UnicodeDecodeError as e:
                    return f"not UTF-8: byte {pos + e.start}"
                pos += len(chunk)
    except OSError as e:
        return f"not readable ({type(e).__name__})"


def _json_ok(path: Path) -> str | None:
    try:
        with open(path, encoding="utf-8-sig") as fh:
            json.load(fh)
        return None
    except json.JSONDecodeError as e:
        # "Extra data": the first value parsed whole and more follows — JSON Lines under a .json name
        if e.msg == "Extra data" and _jsonl_ok(path) is None:
            return None
        return f"broken JSON: line {e.lineno}, column {e.colno}"
    except UnicodeDecodeError as e:
        return f"not UTF-8: byte {e.start}"
    except (OSError, RecursionError, ValueError) as e:
        return f"not parsed ({type(e).__name__})"


def _jsonl_ok(path: Path) -> str | None:
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for n, line in enumerate(fh, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    json.loads(s)
                except json.JSONDecodeError as e:
                    return f"broken line {n}: column {e.colno}"
        return None
    except UnicodeDecodeError as e:
        return f"not UTF-8: byte {e.start}"
    except (OSError, RecursionError, ValueError) as e:
        return f"not parsed ({type(e).__name__})"


def _ooxml_ok(path: Path) -> str | None:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError, ValueError) as e:
        return f"container does not open ({type(e).__name__})"
    with zf:
        bad = zf.testzip()
        if bad is not None:
            return "broken container member"
        names = zf.namelist()
        if not names:
            return "empty container"
        for name in names:
            low = name.lower()
            if not (low.endswith(".xml") or low.endswith(".rels")):
                continue
            try:
                with zf.open(name) as fh:
                    for _ev in ET.iterparse(fh, events=("end",)):
                        pass
            except ET.ParseError as e:
                return f"broken XML in part {names.index(name) + 1} of {len(names)}: {e.position}"
            except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as e:
                return f"part not parsed ({type(e).__name__})"
    return None


_NO_PYMUPDF = "pdf_unchecked"


def _pdf_why(path: Path) -> str | None:
    """None — the PDF opens; _NO_PYMUPDF — pymupdf is not installed; otherwise the reason."""
    try:
        import fitz  # pymupdf
    except ImportError:
        return _NO_PYMUPDF
    try:
        doc = fitz.open(str(path))
    except Exception as e:  # noqa: BLE001 — pymupdf raises its own types
        return f"PDF does not open ({type(e).__name__})"
    try:
        if doc.is_encrypted and doc.needs_pass:
            return "PDF is password-protected"
        _ = doc.page_count
    except Exception as e:  # noqa: BLE001
        return f"PDF not readable ({type(e).__name__})"
    finally:
        doc.close()
    return None


def _pdf_unchecked(ctx) -> None:
    if "pdf_unchecked" not in ctx.data:
        ctx.data["pdf_unchecked"] = True
        ctx.note("pymupdf is not installed — PDF structure not checked")
    ctx.count("gate:pdf_unchecked")


def _check(path: Path):
    """(check, detail) on a violation, None — the file is intact, _NO_PYMUPDF — a PDF not checked."""
    ext = path.suffix.lower()
    if ext in JSON_EXT:
        why = _json_ok(path)
        return ("json", why) if why else None
    if ext in JSONL_EXT:
        why = _jsonl_ok(path)
        return ("jsonl", why) if why else None
    if ext in OOXML_EXT:
        why = _ooxml_ok(path)
        return ("office", why) if why else None
    if ext == ".pdf":
        why = _pdf_why(path)
        if why == _NO_PYMUPDF:
            return _NO_PYMUPDF
        return ("pdf", why) if why else None
    if not known_ext(path.name):
        return None                      # unknown kind — decided in run(), not a violation
    if kind_of(path.name) in ("text", "structured"):
        why = _utf8_ok(path)
        return ("utf8", why) if why else None
    return None


def check_file(ctx, path: Path) -> tuple[str, str] | None:
    """(check, detail) on a violation, None — the file is intact."""
    got = _check(path)
    if got == _NO_PYMUPDF:
        _pdf_unchecked(ctx)
        return None
    return got


def unknown_binary(path: Path) -> bool:
    """A file outside the kind table that does not read as UTF-8: not text, cannot be checked."""
    return not known_ext(path.name) and _utf8_ok(path) is not None


def restore_key(rel: str) -> str:
    """A path from ``[documents].restore`` or ``ctx.rel`` in comparable form: ``/`` instead of
    ``\\``, without leading ``./`` and ``/``."""
    s = str(rel).strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s.lstrip("/")


_CHUNK_BYTES = 8 << 20      # files per pool task: contiguous, about this many bytes
_CHUNK_FILES = 300


def _probe_chunk(_state, chunk) -> list:
    """Checks of one slice of files (in a worker of ``fork_map`` or here): per file "unknown" (an unknown
    kind, not text) or the result of _check."""
    return ["unknown" if unknown_binary(p) else _check(p) for p in chunk[0]]


def _chunks(files: list) -> list:
    out, cur, size = [], [], 0
    for p in files:
        try:
            n = p.stat().st_size
        except OSError:
            n = 0
        if cur and (size + n > _CHUNK_BYTES or len(cur) >= _CHUNK_FILES):
            out.append((cur, size))
            cur, size = [], 0
        cur.append(p)
        size += n
    if cur:
        out.append((cur, size))
    return out


def _json_chunk(_state, chunk) -> list:
    """JSON/JSONL checks of one slice of files: per file None or (check, detail)."""
    return [_check(p) for p in chunk[0]]


def mark_input_broken(ctx) -> int:
    """Inventory, before any processing: each ``.json``/``.jsonl``/``.ndjson`` that does not parse
    gets the ``input_broken:<check>`` mark (the path and the error position go to the owner's
    report, the count to the manifest). The file is processed like any other — masked as text;
    the structural check at the end skips it. Returns the number of marked files."""
    from .. import engine_adapter as EA
    from .anonymize import _engine_workers
    files = [p for p in ctx.files()
             if not p.is_symlink() and p.suffix.lower() in JSON_EXT | JSONL_EXT]
    if not files:
        return 0
    chunks = _chunks(files)
    got = EA.fork_map(_json_chunk, chunks, _engine_workers(ctx), weight=lambda c: c[1],
                      label="input_json")
    n = 0
    for (paths, _size), results in zip(chunks, got):
        for p, res in zip(paths, results):
            if res is None:
                continue
            ctx.mark(p, f"{INPUT_BROKEN}:{res[0]}",
                     f"came broken in the input ({res[1]}) — masked as text, structure kept as it came")
            n += 1
    if n:
        ctx.note(f"JSON files broken already in the input: {n} — masked as text, see the "
                 "\"Kept with a mark\" section")
    return n


def run(ctx) -> None:
    from .. import engine_adapter as EA
    from .anonymize import _engine_workers
    checked = 0
    skipped = 0
    came_broken = 0
    unknown = 0
    restored = 0
    restore = {restore_key(r) for r in ctx.config.documents.restore}
    todo = []
    for p in ctx.files():
        if p.is_symlink():
            continue
        if ctx.is_marked(p, UNPROCESSED):
            skipped += 1                 # left as it was in the input — not our defect
            continue
        if ctx.is_marked(p, INPUT_BROKEN):
            came_broken += 1             # broken already in the input — not our defect
            continue
        todo.append(p)
    # the files are read and parsed in forked processes (slices in file order, the biggest first);
    # marks, violations and notes — here, in file order, as from one pass
    chunks = _chunks(todo)
    got = EA.fork_map(_probe_chunk, chunks, _engine_workers(ctx), weight=lambda c: c[1],
                      label="gates_struct")
    for (paths, _size), results in zip(chunks, got):
        for p, res in zip(paths, results):
            if res == "unknown":
                rel = restore_key(ctx.rel(p))
                if rel in restore:
                    restored += 1
                    ctx.count("gate:restored")
                    continue
                ctx.mark(p, UNKNOWN_CATEGORY, "not anonymized: unknown file kind, not text — "
                         "cannot be checked for personal data")
                unknown += 1
                continue
            checked += 1
            if res == _NO_PYMUPDF:
                _pdf_unchecked(ctx)
            elif res is not None:
                _fail(ctx, ctx.rel(p), res[0], res[1])
    if unknown:
        ctx.note(f"unknown-kind files (not text) kept with a mark: {unknown} — see "
                 "the \"Kept with a mark\" section")
    if restored:
        ctx.note(f"unknown-kind files kept by [documents].restore: {restored}")
    ctx.count("gate:checked", checked)
    if skipped:
        ctx.count("gate:skipped_unprocessed", skipped)
    if came_broken:
        ctx.count("gate:skipped_input_broken", came_broken)
    ctx.data["gate_struct_checked"] = checked
