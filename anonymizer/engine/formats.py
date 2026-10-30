"""Format-aware value masking (the megafix2/3 core — the single most reusable
engine piece).

Free-text regex detectors run on the raw file. But VALUE-level detectors (the
entropy 'is this a credential' gate, CRM column masking) must only see whole
*values* — a JSON string leaf or a CSV cell — never free prose, or they wreck
ordinary sentences. This module extracts those values, lets a callback rewrite
them, and re-serialises with structure intact.

Serialisation discipline (the two invariants every walker keeps):
  * UNCHANGED input comes back byte-identical — re-dumping normalises numbers
    (1e3 -> 1000.0), spacing and quoting, and collapses duplicate JSON keys
    (data loss), so we only re-serialise when value_fn actually changed a value.
  * The output must survive ``write_text`` (utf-8 + surrogateescape): a lone
    surrogate that json.loads materialised from a ``\\udXXX`` escape is
    re-escaped before the line leaves this module (a raw surrogate either
    crashes the encoder or lands on disk as an invalid byte).
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys

from .walk import max_file_bytes

# json.loads turns an ESCAPED lone surrogate ("\\udc80") into a real code
# point; dumps(ensure_ascii=False) would emit it raw. Re-escape on the way out.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _escape_surrogates(s: str) -> str:
    return _SURROGATE_RE.sub(lambda m: "\\u%04x" % ord(m.group(0)), s)


def detect_format(path: str, text: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return "jsonl"
    if ext in (".json",):
        return "json"
    if ext == ".tsv":
        return "tsv"
    if ext == ".csv":
        return "csv"
    return "text"


def _walk_json(node, value_fn, key=None):
    if isinstance(node, str):
        return value_fn(node, key)
    if isinstance(node, list):
        return [_walk_json(x, value_fn, key) for x in node]
    if isinstance(node, dict):
        return {k: _walk_json(v, value_fn, k) for k, v in node.items()}
    return node


# A JSONL line where the value pass changed a value used to be rebuilt with
# json.dumps — a duplicate key vanished, an escaped \uXXXX got unescaped (a name the
# rules had already passed came out open, a repeated dry run found it → BLOCK idempotency),
# numbers and separators were normalized. Now — substitution by spans, as in
# mask_json_document; spans diverged from leaves — the line is re-dumped (_subst_leaves → None).


def _subst_leaves(text: str, value_fn):
    """value_fn over the string leaves of document ``text`` (duplicate keys — every leaf, in document
    order) and substitution of changed values by spans. None — not JSON or nothing
    changed; spans diverged from leaves — the old re-dump (the mask is not lost)."""
    try:
        obj = json.loads(text, object_pairs_hook=_Pairs)
    except Exception:
        return None
    leaves = list(_iter_leaves(obj))
    new = [value_fn(v, k) for v, k in leaves]
    if all(n == v for n, (v, _k) in zip(new, leaves)):
        return None
    try:
        spans = _string_value_spans(text)
        if len(spans) != len(leaves):
            raise ValueError(f"{len(spans)} spans for {len(leaves)} leaves")
    except Exception as e:  # noqa: BLE001
        print(f"formats: WARN jsonl: cannot collect value spans ({e}) — re-dumping the line",
              flush=True)
        return _escape_surrogates(json.dumps(
            _walk_json(json.loads(text), lambda v, k: value_fn(v, k)), ensure_ascii=False))
    out, pos = [], 0
    for (st, en), (v, _k), n in zip(spans, leaves, new):
        if n == v:
            continue
        out.append(text[pos:st])
        out.append(_escape_surrogates(json.dumps(n, ensure_ascii=False)))
        pos = en
    out.append(text[pos:])
    return "".join(out)


def _mask_json_line(line: str, value_fn) -> str:
    """Value pass over ONE JSONL line (without EOL) — the shared brick of the in-memory
    branch (``mask_json_string_values``) and the streaming branch of Pipeline:
    both paths must give byte-for-byte identical results, so the logic
    exists in exactly one place. An empty / unparsable line and a line
    where value_fn changed nothing are returned AS IS (see the serialization
    discipline in the module docstring)."""
    s = line.strip()
    if not s:
        return line
    # A CRLF file in the in-memory branch (read_text with newline='') is split on "\n" and
    # leaves a trailing "\r" IN the line; json.loads tolerates it, but a re-dump
    # of a changed line lost it — the stream branch (EOL split off before the call)
    # keeps "\r\n", and the stream==memory byte oracle broke. The trailing
    # "\r" is removed before parsing and put back on the changed line.
    body, cr = (line[:-1], "\r") if line.endswith("\r") else (line, "")
    out = _subst_leaves(body, value_fn)
    return line if out is None else out + cr


def mask_json_string_values(text: str, value_fn) -> str:
    """Apply value_fn to every JSON string leaf, per JSONL line. On parse
    failure for a line, leave it untouched (the raw-regex pass still sees it,
    though label-free entropy secrets are only caught when the line parses).
    Lines where value_fn changed nothing keep their ORIGINAL text."""
    return "\n".join(_mask_json_line(line, value_fn) for line in text.split("\n"))


def _detect_indent(text: str):
    """Indent unit of a pretty-printed document: int (spaces) or a tab string;
    None when the document is single-line/compact."""
    for line in text.split("\n")[1:]:
        stripped = line.lstrip(" \t")
        if stripped and stripped != line:
            ws = line[: len(line) - len(stripped)]
            return ws if "\t" in ws else len(ws)
    return None


class _Pairs(list):
    """A JSON object as a list of pairs (object_pairs_hook): the order and DUPLICATES of
    document keys are kept — leaves are walked exactly in document order."""


def _iter_leaves(node, key=None):
    """String leaves in document order: (value, key) — the key is the same
    as _walk_json passes (the key of the nearest ancestor object)."""
    if isinstance(node, str):
        yield node, key
    elif isinstance(node, _Pairs):
        for k, v in node:
            yield from _iter_leaves(v, k)
    elif isinstance(node, list):
        for x in node:
            yield from _iter_leaves(x, key)


def _string_value_spans(text: str) -> list:
    """Spans [start, end) of string VALUES (not keys) of the document in document
    order — a pure-Python json scanner with a parse_string hook (the C scanner does not
    call the hook; object keys are read by the module-level scanstring — they do not get into spans).
    Several times slower than C, so it is called ONLY when value_fn changed something."""
    import json.scanner as _scanner
    from json.decoder import scanstring as _scanstring
    spans: list = []

    def _ps(s, end, strict=True):
        v, e = _scanstring(s, end, strict)
        spans.append((end - 1, e))
        return v, e

    dec = json.JSONDecoder(object_pairs_hook=_Pairs)
    dec.parse_string = _ps
    dec.scan_once = _scanner.py_make_scanner(dec)
    dec.decode(text)
    return spans


def mask_json_document(text: str, value_fn) -> str:
    """Apply value_fn to every string leaf of a WHOLE .json document. Pretty-
    printed (multi-line) files never parse per-line, so the JSONL walker would
    silently skip them — parse the full text instead. A document that doesn't
    parse falls back to the per-line walker (a .json that is really JSONL);
    truly broken files come back untouched either way.

    Changed values are substituted BY SPANS into the source text, not
    by re-dumping the whole document (one replacement in a vault file used to rewrite
    every line — 1e3→1000.0, duplicate keys collapsed, indents/quotes
    were normalized → a false entropy_secret magnitude and a giant diff).
    Everything outside the replaced values is the source byte for byte."""
    try:
        obj = json.loads(text, object_pairs_hook=_Pairs)
    except Exception:
        return mask_json_string_values(text, value_fn)
    leaves = list(_iter_leaves(obj))
    new = [value_fn(v, k) for v, k in leaves]
    if all(n == v for n, (v, _k) in zip(new, leaves)):
        return text
    try:
        spans = _string_value_spans(text)
        if len(spans) != len(leaves):
            raise ValueError(f"{len(spans)} spans for {len(leaves)} leaves")
    except Exception as e:  # noqa: BLE001 — the scanner diverged from json.loads
        # the old path (re-dump) — worse for the diff, but does not lose the mask
        print(f"formats: WARN json: cannot collect value spans ({e}) — "
              f"re-dumping the document", flush=True)
        masked = _walk_json(json.loads(text), lambda v, k: value_fn(v, k))
        out = _escape_surrogates(
            json.dumps(masked, ensure_ascii=False, indent=_detect_indent(text))
        )
        if text.endswith("\n") and not out.endswith("\n"):
            out += "\n"
        return out
    out, pos = [], 0
    for (st, en), (v, _k), n in zip(spans, leaves, new):
        if n == v:
            continue
        out.append(text[pos:st])
        out.append(_escape_surrogates(json.dumps(n, ensure_ascii=False)))
        pos = en
    out.append(text[pos:])
    return "".join(out)


def raise_csv_field_limit() -> int:
    """Field limit of csv.reader (process-global state) — not below the file cap
    (ANON_MAX_FILE_BYTES: a cell is not bigger than the file); only upwards, someone else's limit is not cut.
    A single point for ALL CSV readers (mask, registry profile, mine/NER blanking,
    column profile): before, only value passes raised the limit,
    and the "registry or not" decision on a file with a cell over 128 KB depended
    on whether this process had already masked some CSV (a round worker — yes, a fresh
    mine/NER — no), and blanking such a registry crashed mine and ner-rubert with csv.Error."""
    cap = min(max_file_bytes(), sys.maxsize)
    if csv.field_size_limit() < cap:
        csv.field_size_limit(cap)
    return cap


def sniff_delimiter(text: str) -> str:
    """CSV/TSV delimiter by the header line: tab / «;» / «,» — the one that occurs
    STRICTLY more often in the header wins; a tie → «,» (the old behaviour:
    tab only when tabs > commas). «;» — Russian Excel CRM exports:
    without it a ;-CSV was parsed as one column per line, and neither the column
    pass nor the mine column profile saw anything. A single sniff for the profile
    (tools/csv_profile) and the mask (mask_csv_cells): the profile must see the same
    columns that are masked later."""
    head = text.split("\n", 1)[0]
    delim = ","
    if head.count(";") > head.count(delim):
        delim = ";"
    if head.count("\t") > head.count(delim):
        delim = "\t"
    return delim


class CsvStructureError(csv.Error):
    """The CSV/TSV structure diverged from what the value pass expects (dialect not
    recognized / named columns are not in the header). A subclass of csv.Error, so that
    the caller (pipeline) catches it in one except together with parser errors and
    skips the pass LOUDLY: WARN + report.rule_errors, not silence."""


# "foreign" delimiters in the header without the sniffed one: the file is not single-column
_FOREIGN_DELIMS = re.compile(r"[,;|\t]")


def csv_delimiter(path: str, text: str) -> str:
    """Delimiter of value passes by extension: .tsv is
    always tab (sniffing by character counts in the header lost to commas in a header
    «notes, remarks» — the pack columns were not found, neither masks nor a WARN);
    .csv — sniff_delimiter (a single sniff with the profile: «;» is the default of Russian
    Excel; mask_values used to take «,» rigidly — a ;-CSV with login/password in
    the header was parsed as one column, key = the whole header → vault → EVERY line
    whole into SECRET_REDACTED, id/login destroyed). No delimiter in the header
    → the file is single-column (cell key = column name, as before); but
    if the header has ANOTHER delimiter («|», or «,»/«;» in a .tsv)
    — the dialect is not recognized, parsing as one column would repeat the same damage
    → CsvStructureError, the pass is skipped loudly."""
    head = text.split("\n", 1)[0]
    if os.path.splitext(path)[1].lower() == ".tsv":
        delim = "\t"
    else:
        delim = sniff_delimiter(text)
    if delim not in head:
        other = sorted(set(_FOREIGN_DELIMS.findall(head)) - {delim})
        if other:
            raise CsvStructureError(
                f"delimiter {delim!r} not found in the header, it has "
                f"{other!r} — dialect not recognized")
    return delim


def mask_csv_cells(text: str, value_fn, columns=None, delimiter=None) -> str:
    """Apply value_fn to CSV/TSV cells. If ``columns`` (set of header names or
    int indices) is given, only those columns are touched; else every cell.
    ``delimiter=None`` sniffs the header row (``sniff_delimiter``) — callers
    that know the format pass it explicitly (pipeline: ``csv_delimiter`` by
    extension). Returns the original text when no cell changed (no
    quoting/EOL normalisation).
    A column name is matched after strip and by ALL occurrences in the header (duplicate
    headers are both columns of one entity; index() used to take the first).
    csv.Error (CR in an unquoted field, a cell over the limit) is NOT swallowed
    (the old ``except: return text`` silently skipped the column and
    entropy passes over the whole file — without WARN and rule_errors); the field limit
    is raised to the file cap (ANON_MAX_FILE_BYTES: a cell is not bigger than the file,
    while the csv default of 128 KB cut CRM exports with correspondence in one cell).
    Named ``columns`` missing from the header — CsvStructureError: a pack rule
    did not find its column = a broken expectation, not silence."""
    if delimiter is None:
        delimiter = sniff_delimiter(text)
    raise_csv_field_limit()
    raws: list = []
    tracker = _LineTracker(io.StringIO(text))
    rows = []
    for row in csv.reader(tracker, delimiter=delimiter):
        rows.append(row)
        raws.append(tracker.take())
    if not rows:
        return text
    header = rows[0]
    idxs = None
    if columns:
        idxs = set()
        for c in columns:
            if isinstance(c, int):
                idxs.add(c)
            else:
                want = str(c).strip()
                idxs.update(i for i, h in enumerate(header)
                            if h.strip() == want)
        if not idxs:
            raise CsvStructureError(
                f"columns {sorted(map(str, columns))!r} not found in the header "
                f"{header[:8]!r} (delimiter {delimiter!r})")
    changed = False
    out_rows = [header]
    for row in rows[1:]:
        new_row = []
        for i, cell in enumerate(row):
            new = cell
            if idxs is None or i in idxs:
                new = value_fn(cell, header[i] if i < len(header) else None)
                if new != cell:
                    changed = True
            new_row.append(new)
        out_rows.append(new_row)
    if not changed:
        return text
    if "".join(raws) == text:
        return _csv_subst_rows(rows, out_rows, raws, delimiter)
    # csv.reader lines did not match the text — rebuild
    print("formats: WARN csv: raw records did not match the text — rebuilding the file with csv.writer",
          flush=True)
    buf = io.StringIO()
    eol = _csv_eol(text)
    w = csv.writer(buf, delimiter=delimiter, lineterminator=eol)
    w.writerows(out_rows)
    out = buf.getvalue()
    if not text.endswith(("\n", "\r")) and out.endswith(eol):
        out = out[:-len(eol)]                # there was no trailing line break — do not add one
    return out


# The value pass (INN columns, entropy secret, pack and CRM columns)
# used to rebuild the whole CSV through csv.writer — in a file where all cells are quoted
# (Excel/1C), quotes vanished in EVERY record, including the header and unchanged records (the round diff
# was the whole file, release saw a "rewritten" file). Now changed cells are substituted by
# position into the raw record (like mask_json_document by spans): the header and untouched records are byte for
# byte, a changed cell keeps its quoting (was quoted — stays quoted), neighbouring cells and
# the record's line break are intact. A raw record = the lines csv.reader took from the iterator for
# this record (_LineTracker: the reader reads a line only when it needs it). Cell spans of a record
# are checked against the csv parse by value; a record the scanner could not parse (a quote not at the edge
# of a cell, unclosed) is rebuilt by csv.writer alone — with its own line break.


class _LineTracker:
    """Line iterator for csv.reader that remembers the lines handed out: take() — the raw text
    of the record assembled from them (csv.reader does not read ahead: a line is requested only when
    a quoted cell continues on the next one)."""

    def __init__(self, src):
        self._it = iter(src)
        self._buf: list = []

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._it)
        self._buf.append(line)
        return line

    def take(self) -> str:
        raw = "".join(self._buf)
        self._buf.clear()
        return raw


def _csv_record_eol(raw: str) -> str:
    if raw.endswith("\r\n"):
        return "\r\n"
    if raw.endswith(("\n", "\r")):
        return raw[-1]
    return ""


def _csv_record_cells(raw: str, delimiter: str):
    """Cell spans of a raw record [(start, end, quoted)] (without the record EOL) by the rules
    of the default csv dialect (quote «"», doubled inside). None — the record is beyond the scanner
    (an unclosed quote, characters between the closing quote and the delimiter)."""
    body = len(raw) - len(_csv_record_eol(raw))
    spans: list = []
    i = 0
    while True:
        if i < body and raw[i] == '"':
            j = i + 1
            while True:
                k = raw.find('"', j, body)
                if k == -1:
                    return None
                if k + 1 < body and raw[k + 1] == '"':
                    j = k + 2
                    continue
                break
            end = k + 1
            if end < body and raw[end] != delimiter:
                return None
            spans.append((i, end, True))
        else:
            k = raw.find(delimiter, i, body)
            end = body if k == -1 else k
            spans.append((i, end, False))
        if end < body and raw[end] == delimiter:
            i = end + 1
            continue
        return spans


def _csv_cell_text(raw: str, span) -> str:
    s, e, quoted = span
    return raw[s + 1:e - 1].replace('""', '"') if quoted else raw[s:e]


def _csv_quote(value: str, delimiter: str, force: bool) -> str:
    if force or any(ch in value for ch in (delimiter, '"', "\r", "\n")):
        return '"' + value.replace('"', '""') + '"'
    return value


def _csv_subst_rows(rows: list, out_rows: list, raws: list, delimiter: str) -> str:
    """File from raw records; in changed ones — substitution of changed cells by spans."""
    out: list = []
    warned = False
    for n, (row, new_row, raw) in enumerate(zip(rows, out_rows, raws)):
        if new_row == row:
            out.append(raw)
            continue
        spans = _csv_record_cells(raw, delimiter)
        if spans is not None and len(spans) == len(row) \
                and all(_csv_cell_text(raw, sp) == c for sp, c in zip(spans, row)):
            piece, pos = [], 0
            for sp, old, new in zip(spans, row, new_row):
                if new == old:
                    continue
                piece.append(raw[pos:sp[0]])
                # an empty single cell — quoted, as csv.writer does (otherwise an empty record)
                piece.append(_csv_quote(new, delimiter, sp[2] or (len(spans) == 1 and not new)))
                pos = sp[1]
            piece.append(raw[pos:])
            out.append("".join(piece))
            continue
        if not warned:
            print(f"formats: WARN csv: record {n + 1} not parsed by the cell scanner — "
                  "rebuilding the record with csv.writer", flush=True)
            warned = True
        eol = _csv_record_eol(raw)
        buf = io.StringIO()
        csv.writer(buf, delimiter=delimiter, lineterminator=eol or "\n").writerow(new_row)
        out.append(buf.getvalue() if eol else buf.getvalue()[:-1])
    return "".join(out)


# The value pass rebuilt CSV with lineterminator="\n" — in a CRLF file records
# became LF, while inside a quoted cell «\r\n» remained (a mixed style; release checks
# EOL only for re-encoded files and only prints a WARN), and a file without a trailing line
# break got one. Now the line break is as in the first record of the source, the tail is as it was.


_CSV_EOL_OR_QUOTE = re.compile(r'[\r\n"]')


def _first_eol(text: str, start: int = 0) -> str:
    """First line break from ``start``: «\r\n», «\r» or «\n» (none — «\n»)."""
    i = text.find("\n", start)
    j = text.find("\r", start)
    if j != -1 and (i == -1 or j < i):
        return "\r\n" if text[j + 1:j + 2] == "\n" else "\r"
    return "\n"


def _csv_eol(text: str) -> str:
    """Line break of the first record OUTSIDE quotes: «\r\n», «\r» or «\n» (none — «\n»).
    Before, the first «\n»/«\r» of the text was taken, and a quoted
    multi-line cell in the header («"a\nb",password\r\n…») gave LF — a CRLF file was rebuilt
    with LF, so the damage stayed. Quotes are counted by parity («""» inside a cell is
    two), an unclosed quote — the old first line break."""
    i = 0
    while True:
        m = _CSV_EOL_OR_QUOTE.search(text, i)
        if m is None:
            return "\n"
        c = m.group(0)
        if c == '"':
            j = text.find('"', m.end())
            if j == -1:
                return _first_eol(text)
            i = j + 1
            continue
        if c == "\r":
            return "\r\n" if text[m.end():m.end() + 1] == "\n" else "\r"
        return "\n"


def csv_record_chunks(text: str, target: int) -> list:
    """Boundaries of CSV/TSV record batches — [0, b1, …, len(text)]; a batch is at least
    ``target`` characters (the last one is the rest) and ends with a line break OUTSIDE quotes (an even
    number of «"» from the start of the text; «""» inside a cell is two), so a multi-line quoted
    cell is not cut. The boundary is right after «\\n» («\\r\\n» wholly in the batch). An unclosed
    quote (a broken export) does not keep the rest of the file as one batch: no even line
    break found within 8 × ``target`` characters — the boundary is at the nearest line break (batches are a unit
    of rule scheduling, not of CSV parsing; column passes go over the whole text)."""
    n = len(text)
    bounds = [0]
    if target <= 0 or n <= target:
        return [0, n]
    a = 0                    # start of the current batch
    q = 0                     # quotes in [a, pos)
    pos = 0
    while True:
        want = a + target
        if want >= n:
            break
        q += text.count('"', pos, want)
        pos = want
        hard = a + 8 * target
        cut = -1
        while True:
            j = text.find("\n", pos)
            if j < 0:
                break
            q += text.count('"', pos, j)
            pos = j + 1
            if q % 2 == 0 or pos >= hard:
                cut = pos
                break
        if cut < 0 or cut >= n:
            break
        bounds.append(cut)
        a, q = cut, 0
    bounds.append(n)
    return bounds


def iter_csv_cells(text: str, delimiter: str):
    """(row, col, start, end) — the range of EVERY cell in the text (a quoted
    cell — together with the quotes) with csv.reader semantics (excel dialect:
    quotechar «"», doublequote, no escapechar, skipinitialspace=False,
    strict=False — text after the closing quote up to the delimiter is the tail
    of the same cell). Column indexes match csv.reader (the profile sees the same
    columns; tools/csv_profile, structured/tabular); a record ends at \\r\\n | \\n | \\r outside quotes. An empty line
    gives one empty cell (csv.reader gives []; for blanking there is no difference).
    Raises nothing: an unclosed quote stretches to the end of the text."""
    stop = re.compile("[" + re.escape(delimiter) + "\r\n]")
    n = len(text)
    i = row = 0
    while i < n:
        col = 0
        while True:
            start = i
            if i < n and text[i] == '"':
                i += 1
                while True:
                    q = text.find('"', i)
                    if q < 0:
                        i = n                     # unclosed quote
                        break
                    if text.startswith('""', q):
                        i = q + 2                 # escaped quote
                        continue
                    i = q + 1
                    break
            m = stop.search(text, i)
            i = m.start() if m else n
            yield row, col, start, i
            col += 1
            if i < n and text[i] == delimiter:
                i += 1
            else:
                break
        if i < n and text[i] == "\r":
            i += 1
        if i < n and text[i] == "\n":
            i += 1
        row += 1


def mask_values(path: str, text: str, value_fn, csv_columns=None) -> str:
    """Format-dispatch a value-level pass. `text` fallback returns unchanged
    (free text has no discrete 'values' — use regex rules there)."""
    fmt = detect_format(path, text)
    if fmt == "json":
        return mask_json_document(text, value_fn)
    if fmt == "jsonl":
        return mask_json_string_values(text, value_fn)
    if fmt in ("csv", "tsv"):
        # delimiter by extension/header (see csv_delimiter), not «,»
        return mask_csv_cells(text, value_fn, csv_columns,
                              delimiter=csv_delimiter(path, text))
    return text
