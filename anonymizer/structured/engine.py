"""Structured profile engine: a JSON export is processed FIELD BY FIELD.

Profile = format signature + document "spine" + a "path → action" table
(see telegram.py). The engine is shared and has nothing format-specific:

  * parsing is STREAMED over the lines of a pretty-printed "one element per
    line" layout (``iter_line_units``): the file is never held in memory as a
    tree. Spine containers (root, lists of chats and messages) are walked
    through; everything outside the spine (a message, a contacts block) is a
    UNIT: its lines are collected, parsed with ``json.loads`` and handed to the
    profile. Scalars of a spine object (chat name/type/id) go to the profile
    as a "header" group;
  * the profile returns a ``Plan``: point edits ``{path: value}`` (identity,
    ids, typed entities — no rules and no LLM) and free-text SLOTS. Slot texts
    of a batch are joined into a projection and pass the rule chain in ONE
    call (``_run_slots``) instead of the whole file: only message text goes
    to the detectors;
  * writing substitutes INTO THE SOURCE LINES (``apply_edits``): only the
    scalar value on its own line changes, everything else (key order,
    indentation, empty arrays, escapes of untouched values) stays byte for
    byte;
  * a layout that is not line based (minified JSON) takes the fallback path
    through the whole tree (``_mask_dom``): same profile logic, output is
    json.dumps with the source indentation.

The same units and slots give the text projection of the file for mining and
NER (``projection``) and a line projection for the L0 audit (``text_lines``:
file line numbers kept, non-text lines blanked).

JSONL profile (``profile.jsonl``, YouTrack export): a unit is a file LINE (a
record), there is no spine. A record is parsed with json.loads (duplicate key —
leaf mode: every string value is free text), edits are substituted by value
spans in the line (``apply_record``: everything outside the replaced values
stays byte for byte), a non-JSON line is free text as a whole. The rule batch
size is per profile (``batch_units``); there is no fallback through the tree —
no entry point reads the file into memory as a whole.
"""
from __future__ import annotations

import difflib
import hashlib
import io
import json
import os
import re
from collections import Counter
from json.decoder import scanstring

from ..engine import placeholders as P
from ..engine import walk
from ..engine.boundary import TOKEN_RE, is_own_token
from ..engine.formats import (_detect_indent, _escape_surrogates, _iter_leaves,
                              _Pairs, _string_value_spans)
from ..engine.report import Report
from .entities import VERSION as _REG_VERSION
from .entities import Registry

HEAD_BYTES = 65536
# slot separator in the rule projection: its own line of Private Use chars —
# not a letter/digit (L/R/CL/CR boundaries pass), not whitespace (\s+ does not
# stretch over it), the newlines around it cut line patterns
SEP_CHAR = "\ue000"
SEP = "\n" + SEP_CHAR * 2 + "\n"
# separator of ISOLATED slots: wider than any rule context window
# (Rule.window, default 120) — a neighbour does not serve as context
ISO_SEP = "\n" + SEP_CHAR * 200 + "\n"
# A batch is exactly BATCH_UNITS document units (messages), NOT by text size:
# batch composition must be the same on the first and the repeated pass (text
# length changes after masking, the number of units does not), otherwise the
# neighbour context across a batch border would change and a repeated dry run
# would give replacements.
BATCH_UNITS = 2000
# the post-rules step of a slot (Slot.post: file name detector) and rules over
# its output run to a fixed point, at most this many rounds (1–2 in practice);
# rounds exhausted with replacements — WARN in the stage log (_run_slots), the
# idempotency oracle catches the rest
POST_ROUNDS = 4

RAW, UNIT, HEADER = 0, 1, 2

# JSONL profile: file extensions and the record-unit pattern
JSONL_EXT = (".jsonl", ".ndjson")
REC = ("*",)

_KEY_RX = re.compile(r'[ \t]*"((?:[^"\\]|\\.)*)"[ \t]*:[ \t]*')
_WS = " \t"
# word / whitespace / single char: joined findall == the source string
_WORD_RX = re.compile(r"\w+|\s+|[^\w\s]")
_ALIGN_CAP = 4_000_000       # words×words in difflib; above it — proportional

# process diagnostics (before/after measurements): how much went to the rules
STATS: Counter = Counter()


class LayoutError(Exception):
    """The layout is not "one element per line" — the fallback (DOM) is needed."""


def is_token(value) -> bool:
    """The value is already masked (a repeated pass is idempotent): our token
    as a whole (boundary.is_own_token — a nick "PETROV_77" is not a token)."""
    return is_own_token(value)


def enc(value) -> str:
    """JSON literal of the new value (lone surrogates escaped as in
    engine.formats: the file stays valid UTF-8)."""
    return _escape_surrogates(json.dumps(value, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# slots and the unit plan
# --------------------------------------------------------------------------- #
class Slot:
    """Free text of one field: ``text`` goes to the rules, the result
    (``out``) is written to the paths ``paths``. ``orig`` is the value in the
    file (for a file name, text has already passed the name detector),
    ``prefix`` is the untouched part of the value before the text (the
    directory of a path), ``ner`` — natural language text (also goes to NER;
    URLs/file names — rules and mining only).

    ``iso`` — an ISOLATED slot (channel/group name in an identity field): it
    passes the rules apart from the message stream, without neighbour context,
    once per unique value in the file. Such a slot exists only while the value
    has not become a token as a whole (after the rules "channel Ivan Petrov" is
    already a token, and on the repeated pass there is no slot); in the common
    stream this would change the projection between passes and shift the
    context of neighbour messages — a repeated dry run would give replacements.

    ``whole`` — (report rule, token) or None: the value of an IDENTITY FIELD
    that must come out as a token as a whole (group/channel/bot name). Rules
    run as usual (the pack may have approved the name itself — then its token
    is shared with the message text); if after the rules the value is not a
    token as a whole, it is replaced with ``whole`` (see _run_slots)."""
    __slots__ = ("text", "out", "orig", "paths", "key", "ner", "prefix",
                 "iso", "whole", "html", "post")

    def __init__(self, text, paths, key, ner=True, orig=None, prefix="",
                 iso=False, whole=None, html=False, post=None):
        self.text = text
        self.out = text
        self.orig = text if orig is None else orig
        self.paths = paths
        self.key = key
        self.ner = ner
        self.prefix = prefix
        self.iso = iso
        self.whole = whole
        # html — text in an HTML record (YouTrack wikifiedDescription/textPreview):
        # rules over the HTML form (chain.html: &#NNN;, &nbsp;), a separate call
        self.html = html
        # post(out, report) -> out — the post-rules step (file name: name detector)
        self.post = post

    def raw(self) -> str:
        return self.orig

    def commit(self, edits: dict) -> None:
        if self.out != self.orig:
            for p in self.paths:
                edits[p] = self.prefix + self.out


class RichSlot:
    """Message text made of entity parts (``text_entities``): the rules get
    the NATURAL text (parts in a row, typed ones already as tokens), after the
    rules it is distributed back over the parts (``distribute``); the ``text``
    mirror (a string or a list) is rebuilt from the same parts."""
    __slots__ = ("text", "out", "parts", "raws", "paths", "key", "ner",
                 "mirror", "mirror_path", "mirror_kind", "raw_text")
    iso = False
    html = False
    post = None

    def __init__(self, parts, raws, paths, key, mirror=None, mirror_path=None,
                 mirror_kind=None):
        self.parts = parts
        self.raws = raws
        self.paths = paths
        self.text = "".join(parts)
        self.out = self.text
        self.raw_text = "".join(raws)
        self.key = key
        self.ner = True
        self.mirror = mirror
        self.mirror_path = mirror_path
        self.mirror_kind = mirror_kind

    def raw(self) -> str:
        return self.raw_text

    def commit(self, edits: dict) -> None:
        new = self.parts if self.out == self.text \
            else distribute(self.parts, self.out)
        for i, part in enumerate(new):
            if part != self.raws[i]:
                edits[self.paths[i]] = part
        if self.mirror_kind == "str":
            joined = "".join(new)
            if joined != self.mirror:
                edits[self.mirror_path] = joined
        elif self.mirror_kind == "list":
            for i, part in enumerate(new):
                if part != self.raws[i]:
                    item = self.mirror[i]
                    edits[self.mirror_path + ((i,) if isinstance(item, str)
                                              else (i, "text"))] = part


class Plan:
    __slots__ = ("edits", "slots")

    def __init__(self):
        self.edits: dict = {}
        self.slots: list = []

    def slot(self, text, path, key, ner=True, orig=None, prefix="",
             iso=False, whole=None, html=False) -> None:
        self.slots.append(Slot(text, [path], key, ner, orig, prefix, iso,
                               whole, html))


def distribute(parts: list, masked: str) -> list:
    """Distribute ``masked`` (the natural text ``"".join(parts)`` after the
    rules) back over the parts. Source and result are aligned BY WORDS
    (difflib over lists of words/whitespace/chars; common head and tail are
    cut off first): a part border in an unchanged stretch moves exactly, a
    border inside a changed stretch goes to its end (the whole replacement
    goes to the left part). A token is one word, so it is never cut.
    Invariant: ``"".join(result) == masked``."""
    if len(parts) == 1:
        return [masked]
    orig = "".join(parts)
    bounds, pos = [], 0
    for p in parts[:-1]:
        pos += len(p)
        bounds.append(pos)
    a = _WORD_RX.findall(orig)
    b = _WORD_RX.findall(masked)
    head = 0
    top = min(len(a), len(b))
    while head < top and a[head] == b[head]:
        head += 1
    tail = 0
    top -= head
    while tail < top and a[-1 - tail] == b[-1 - tail]:
        tail += 1
    ea, eb = len(a) - tail, len(b) - tail
    if (ea - head) * (eb - head) > _ALIGN_CAP:
        STATS["align_fallback"] += 1
        cuts = _proportional(bounds, len(orig), masked,
                             list(TOKEN_RE.finditer(masked)))
    else:
        ao = [0]
        for w in a:
            ao.append(ao[-1] + len(w))
        bo = [0]
        for w in b:
            bo.append(bo[-1] + len(w))
        ops = []                    # (o_a, o_b, m_a, m_b, equal stretch?)
        if head:
            ops.append((0, ao[head], 0, bo[head], True))
        sm = difflib.SequenceMatcher(None, a[head:ea], b[head:eb],
                                     autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            ops.append((ao[head + i1], ao[head + i2], bo[head + j1],
                        bo[head + j2], tag == "equal"))
        if tail:
            ops.append((ao[ea], ao[-1], bo[eb], bo[-1], True))
        cuts, k = [], 0
        for bnd in bounds:
            while ops[k][1] < bnd:      # first operation ending at >= bnd
                k += 1
            oa, ob, ma, mb, eq = ops[k]
            if bnd == ob or not eq:
                cuts.append(mb)         # stretch joint / inside a changed one
            else:
                cuts.append(ma + (bnd - oa))
    out, prev = [], 0
    for c in cuts:
        c = max(prev, min(c, len(masked)))
        out.append(masked[prev:c])
        prev = c
    out.append(masked[prev:])
    return out


_WORD_CH = re.compile(r"\w")


def _proportional(bounds: list, olen: int, masked: str, toks: list) -> list:
    """Fallback without alignment (text too large for difflib): a part border
    is placed proportionally to length. The border cuts neither a token nor a
    word — it moves to its end (the whole goes to the left part, as with
    alignment). A cut word made a link piece start with a fragment like
    "abc=<TOKEN>", and a per-part check (the round control gate) saw the
    fragment as a separate word: a false BLOCK, and entities made of fragments
    on the client side."""
    spans = [(t.start(), t.end()) for t in toks]
    n = len(masked)
    cuts = []
    for b in bounds:
        c = round(b * n / olen) if olen else 0
        for a, e in spans:
            if a < c < e:
                c = e
                break
        while 0 < c < n and _WORD_CH.match(masked[c - 1]) \
                and _WORD_CH.match(masked[c]):
            c += 1
        cuts.append(c)
    return cuts


# --------------------------------------------------------------------------- #
# line parsing of the "one element per line" layout
# --------------------------------------------------------------------------- #
def is_line_layout(head: str) -> bool:
    """Pretty-print with indentation: the first meaningful line is "{", the
    second is an indented key. Minified/one-line JSON → False (DOM path)."""
    seen = []
    for line in head.split("\n"):
        s = line.strip()
        if s:
            seen.append((line, s))
            if len(seen) == 2:
                break
    if len(seen) < 2:
        return False
    return seen[0][1] == "{" and seen[1][1][:1] == '"' \
        and seen[1][0][:1] in _WS


def iter_line_units(lines, spine, stack=None):
    """Stream of (RAW, None, line) | (UNIT | HEADER, pattern, [lines]).

    Pattern is the container path from the root, array indices replaced with
    "*". A container from ``spine`` is walked through (its brackets are RAW,
    object scalars are collected into HEADER up to the first child
    container/closing); any other container is collected whole into a UNIT.

    ``stack`` — the spine stack at the start: a file part starts at a batch
    border in the middle of the document (None — from the start, empty
    stack). The list is changed IN PLACE — the caller takes a copy of it at
    the yielded item: after RAW/UNIT the line is fully processed, while HEADER
    is yielded BEFORE processing the line that closed it (the stack does not
    have it yet; see plan_parts — the part border is then at the start of that
    line)."""
    if stack is None:
        stack = []              # [(object?, pattern)]
    header = []
    it = iter(lines)
    for line in it:
        body = line.rstrip("\r\n")
        s = body.lstrip(_WS)
        if not s:
            if header:
                header.append(line)
            else:
                yield RAW, None, line
            continue
        c0 = s[0]
        if c0 in "}]":
            if header:
                yield HEADER, stack[-1][1], header
                header = []
            if not stack:
                raise LayoutError("extra closing bracket")
            stack.pop()
            yield RAW, None, line
            continue
        tail = body.rstrip(_WS)
        last = tail[-1]
        if last in "{[":
            if c0 == '"':
                m = _KEY_RX.match(body)
                if m is None or m.end() != len(tail) - 1:
                    raise LayoutError("opening line not parsed")
                if not stack or not stack[-1][0]:
                    raise LayoutError("key outside an object")
                key = m.group(1)
                if "\\" in key:
                    key = json.loads('"' + key + '"')
                pat = stack[-1][1] + (key,)
            elif len(s) == 1:
                if stack:
                    if stack[-1][0]:
                        raise LayoutError("container without a key in an object")
                    pat = stack[-1][1] + ("*",)
                else:
                    pat = ()
            else:
                raise LayoutError("opening line not parsed")
            if header:
                yield HEADER, stack[-1][1], header
                header = []
            if pat in spine:
                stack.append((last == "{", pat))
                yield RAW, None, line
                continue
            unit = [line]
            depth = 1
            for nxt in it:
                unit.append(nxt)
                b = nxt.rstrip("\r\n")
                t = b.lstrip(_WS)
                if not t:
                    continue
                if t[0] in "}]":
                    depth -= 1
                    if depth == 0:
                        break
                elif b.rstrip(_WS)[-1] in "{[":
                    depth += 1
            if depth:
                raise LayoutError("unit not closed")
            yield UNIT, pat, unit
            continue
        if not stack:
            raise LayoutError("scalar outside a container")
        if stack[-1][0] and c0 == '"':
            header.append(line)             # scalar of a spine object
        else:
            yield RAW, None, line           # scalar element of a spine array
    if header or stack:
        raise LayoutError("document not closed")


def parse_unit(lines: list, header: bool):
    """Unit lines → object. HEADER — scalars of a spine object (wrapped in
    "{…}"), UNIT — a container from the first line (with or without a key)."""
    if header:
        src = "{" + "".join(lines).rstrip().rstrip(",") + "}"
    else:
        first = lines[0].rstrip("\r\n" + _WS)
        src = first[-1] + "".join(lines[1:]).rstrip()
        if src.endswith(","):
            src = src[:-1]
    try:
        return json.loads(src)
    except ValueError as exc:
        raise LayoutError(f"unit not parsed: {exc}")


def walk_scalars(lines: list, header: bool, prefixes=None):
    """(line index, path, value start, body without EOL, EOL) for every scalar
    line of a unit; path — keys and INDICES from the unit root. ``prefixes`` —
    the set of container paths worth looking into (None = all): a subtree
    without edits is skipped."""
    stack = [None]              # None — object, int — next array index
    path = []
    n = len(lines)
    i = 0
    if not header:
        first = lines[0].rstrip("\r\n" + _WS)
        stack = [0 if first[-1] == "[" else None]
        i = 1
    while i < n:
        line = lines[i]
        body = line.rstrip("\r\n")
        s = body.lstrip(_WS)
        if not s:
            i += 1
            continue
        if s[0] in "}]":
            stack.pop()
            if path:
                path.pop()
            i += 1
            continue
        if not stack:
            raise LayoutError("line after the unit is closed")
        cur = stack[-1]
        if cur is None:
            m = _KEY_RX.match(body)
            if m is None:
                raise LayoutError("key expected")
            elem = m.group(1)
            if "\\" in elem:
                elem = json.loads('"' + elem + '"')
            vstart = m.end()
        else:
            elem = cur
            stack[-1] = cur + 1
            vstart = len(body) - len(s)
        tail = body.rstrip(_WS)
        if tail[-1] in "{[" and vstart == len(tail) - 1:
            path.append(elem)
            if prefixes is not None and tuple(path) not in prefixes:
                depth = 1                   # no edits inside — skip it
                i += 1
                while i < n and depth:
                    b = lines[i].rstrip("\r\n")
                    t = b.lstrip(_WS)
                    if t:
                        if t[0] in "}]":
                            depth -= 1
                        elif b.rstrip(_WS)[-1] in "{[":
                            depth += 1
                    i += 1
                path.pop()
                continue
            stack.append(0 if tail[-1] == "[" else None)
            i += 1
            continue
        yield i, tuple(path) + (elem,), vstart, body, line[len(body):]
        i += 1


def apply_edits(lines: list, edits: dict, header: bool) -> list:
    """Substitute edits ``{path: value}`` into the unit lines. Only the value
    on its own line changes; key, indentation, comma and EOL stay as they
    were. An edit without its own scalar line (a container value on one line)
    → LayoutError: the file goes the fallback path."""
    prefixes = set()
    for p in edits:
        for k in range(1, len(p)):
            prefixes.add(p[:k])
    out = list(lines)
    done = 0
    for i, path, vstart, body, eol in walk_scalars(lines, header, prefixes):
        if path in edits:
            tail = body.rstrip(_WS)
            comma = "," if tail.endswith(",") else ""
            out[i] = body[:vstart] + enc(edits[path]) + comma + eol
            done += 1
    if done != len(edits):
        raise LayoutError(f"edits {len(edits)}, applied {done}")
    return out


# --------------------------------------------------------------------------- #
# JSONL profile: a file line = a record (unit)
# --------------------------------------------------------------------------- #
def is_jsonl(profile) -> bool:
    return bool(getattr(profile, "jsonl", False))


def batch_of(profile) -> int:
    """Units per rule batch: the profile's own (JSONL records are large),
    otherwise BATCH_UNITS."""
    return int(getattr(profile, "batch_units", 0) or BATCH_UNITS)


def iter_record_units(lines):
    """Stream of (RAW, None, line) | (UNIT, REC, [line]): a non-empty line is
    a record, an empty/whitespace one goes as is."""
    for line in lines:
        if line.strip():
            yield UNIT, REC, [line]
        else:
            yield RAW, None, line


def iter_units(lines, profile, stack=None):
    """File units by the profile layout (see iter_line_units)."""
    if is_jsonl(profile):
        return iter_record_units(lines)
    return iter_line_units(lines, profile.spine, stack)


class _DupKey(ValueError):
    pass


def _no_dups(pairs):
    d = dict(pairs)
    if len(d) != len(pairs):
        raise _DupKey("duplicate key")
    return d


class BadRecord:
    """A JSONL record the profile cannot parse by fields. ``kind``:
    ``line`` — the line is not JSON (free text as a whole, path ``()``);
    ``leaves`` — JSON with a duplicate key (json.loads would lose the first
    value): every string value is free text, path ``("#", leaf number)``."""
    __slots__ = ("kind", "body", "leaves")

    def __init__(self, kind, body, leaves=None):
        self.kind = kind
        self.body = body
        self.leaves = leaves


def _split_body(line: str) -> tuple:
    body = line.rstrip("\r\n")
    return body, line[len(body):]


def _bom_off(body: str) -> int:
    return 1 if body[:1] == "\ufeff" else 0


def parse_record(ulines: list):
    """Record line → object (dict/list) | BadRecord. A top-level scalar
    ("string", number) and a record deeper than the recursion limit — the
    whole line as text, like a flat path (the value is not lost)."""
    body, _eol = _split_body(ulines[0])
    src = body[_bom_off(body):]
    try:
        obj = json.loads(src, object_pairs_hook=_no_dups)
    except _DupKey:
        STATS["record_dupkey"] += 1
        try:
            obj = json.loads(src, object_pairs_hook=_Pairs)
        except (ValueError, RecursionError):
            STATS["record_bad"] += 1
            return BadRecord("line", body)
        return BadRecord("leaves", body, list(_iter_leaves(obj)))
    except (ValueError, RecursionError):
        STATS["record_bad"] += 1
        return BadRecord("line", body)
    if not isinstance(obj, (dict, list)):
        STATS["record_scalar"] += 1
        return BadRecord("line", body)
    return obj


def bad_plan(rec: BadRecord):
    """Plan of a record the profile cannot parse (see BadRecord)."""
    plan = Plan()
    if rec.kind == "line":
        if rec.body.strip():
            plan.slot(rec.body, (), None)
    else:
        for i, (v, k) in enumerate(rec.leaves):
            if v:
                plan.slot(v, ("#", i), k)
    return plan if plan.slots else None


def unit_plan(sess, profile, pat, ulines, header):
    """Unit plan (parsing by the profile layout)."""
    if is_jsonl(profile):
        obj = parse_record(ulines)
        if isinstance(obj, BadRecord):
            return bad_plan(obj)
        return record_plan(sess, obj, ulines)
    return sess.plan(pat, parse_unit(ulines, header), header)


def record_plan(sess, obj, ulines):
    """Record plan; a walk deeper than the recursion limit — the whole line as text."""
    try:
        return sess.plan(REC, obj, False)
    except RecursionError:
        STATS["record_bad"] += 1
        return bad_plan(BadRecord("line", _split_body(ulines[0])[0]))


_SCAN = json.JSONDecoder().scan_once          # C scanner: (value, end)
_WS_RX = re.compile(r"[ \t\n\r]*")


def record_spans(s: str, targets, start: int = 0) -> dict:
    """{path: (start, end)} of the ``targets`` values in the JSON text ``s``;
    path — keys and indices from the root. A subtree without edits is skipped
    by the json C scanner (scan_once) — Python walks only the paths to the
    edits. A duplicate key on the path or a mismatch with parsing —
    LayoutError."""
    prefixes = set()
    for p in targets:
        for k in range(len(p)):
            prefixes.add(p[:k])
    out: dict = {}
    ws = _WS_RX.match
    scan = _SCAN

    def value(i, path):
        if path in targets:
            if path in out:
                raise LayoutError("duplicate edit path")
            end = scan(s, i)[1]
            out[path] = (i, end)
            return end
        if path not in prefixes:
            return scan(s, i)[1]
        c = s[i]
        if c == "{":
            i = ws(s, i + 1).end()
            if s[i] == "}":
                return i + 1
            seen = set()
            while True:
                if s[i] != '"':
                    raise LayoutError("key expected")
                key, i = scanstring(s, i + 1)
                if key in seen:
                    raise LayoutError("duplicate key")
                seen.add(key)
                i = ws(s, i).end()
                if s[i] != ":":
                    raise LayoutError("expected ':'")
                i = value(ws(s, i + 1).end(), path + (key,))
                i = ws(s, i).end()
                if s[i] == ",":
                    i = ws(s, i + 1).end()
                    continue
                if s[i] == "}":
                    return i + 1
                raise LayoutError("object not closed")
        if c == "[":
            i = ws(s, i + 1).end()
            if s[i] == "]":
                return i + 1
            n = 0
            while True:
                i = value(i, path + (n,))
                n += 1
                i = ws(s, i).end()
                if s[i] == ",":
                    i = ws(s, i + 1).end()
                    continue
                if s[i] == "]":
                    return i + 1
                raise LayoutError("array not closed")
        return scan(s, i)[1]

    try:
        value(ws(s, start).end(), ())
    except (IndexError, StopIteration, ValueError, RecursionError) as exc:
        raise LayoutError(f"record not parsed: {exc!r}")
    return out


def _subst(body: str, spans: list) -> str:
    out, pos = [], 0
    for (a, b), new in sorted(spans, key=lambda t: t[0][0]):
        out.append(body[pos:a])
        out.append(enc(new))
        pos = b
    out.append(body[pos:])
    return "".join(out)


def apply_record(ulines: list, edits: dict) -> list:
    """Substitute edits into a record line: only the value at its span
    changes, the rest stays byte for byte (key order and duplicates,
    whitespace, escapes of untouched strings). Path ``()`` — the whole line
    (not JSON), ``("#", i)`` — the i-th string leaf (a record with a duplicate
    key). Spans do not match parsing — the record is rewritten with json.dumps
    (valid JSON with the same edits; record format — the source
    separators)."""
    body, eol = _split_body(ulines[0])
    if () in edits:
        return [edits[()] + eol]
    if any(p[:1] == ("#",) for p in edits):
        spans = _string_value_spans(body[_bom_off(body):])
        off = _bom_off(body)
        got = []
        for p, new in edits.items():
            i = p[1]
            if i >= len(spans):
                raise LayoutError("record leaf outside the spans")
            a, b = spans[i]
            got.append(((a + off, b + off), new))
        return [_subst(body, got) + eol]
    try:
        spans = record_spans(body, set(edits), _bom_off(body))
        if len(spans) != len(edits):
            raise LayoutError(f"edits {len(edits)}, spans {len(spans)}")
    except LayoutError:
        STATS["record_redump"] += 1
        off = _bom_off(body)
        obj = json.loads(body[off:])
        for p, v in edits.items():
            _set_path(obj, p, v)
        seps = (", ", ": ") if '": ' in body[:4096] else (",", ":")
        return [body[:off] + _escape_surrogates(json.dumps(
            obj, ensure_ascii=False, separators=seps)) + eol]
    return [_subst(body, [(spans[p], v) for p, v in edits.items()]) + eol]


def apply_unit(profile, ulines, edits, header) -> list:
    if is_jsonl(profile):
        return apply_record(ulines, edits)
    return apply_edits(ulines, edits, header)


# str.splitlines line separators that json.dumps(ensure_ascii=False) does not
# escape: in a projection line they would shift the numbering (residual/verify
# cut the splitlines view — a finding's line number would drift from the file line)
_LINE_SEPS = {0x2028: "\\u2028", 0x2029: "\\u2029", 0x85: "\\u0085"}


def record_text_line(plan) -> str:
    """Projection line of a record for the L0 audit and residual checks:
    values of free-text slots as JSON literals (escapes as in the file, no
    newlines, U+2028/U+2029/U+0085 escaped), space separated; a record
    without text — an empty line."""
    if plan is None:
        return ""
    out = []
    for s in plan.slots:
        raw = s.raw()
        if raw and not raw.isspace():
            out.append(enc(raw).translate(_LINE_SEPS))
    return " ".join(out)


# --------------------------------------------------------------------------- #
# walk over the whole tree (fallback path and a common source of units)
# --------------------------------------------------------------------------- #
def iter_dom_units(doc, spine, pat=()):
    """(HEADER | UNIT, pattern, unit object) over the tree — the same order
    and the same units as the line parser. HEADER yields the spine object
    ITSELF (the profile looks only at its scalars)."""
    if isinstance(doc, dict):
        if any(not isinstance(v, (dict, list)) for v in doc.values()):
            yield HEADER, pat, doc
        for k, v in doc.items():
            if isinstance(v, (dict, list)):
                child = pat + (k,)
                if child in spine:
                    yield from iter_dom_units(v, spine, child)
                else:
                    yield UNIT, child, v
    elif isinstance(doc, list):
        child = pat + ("*",)
        for v in doc:
            if isinstance(v, (dict, list)):
                if child in spine:
                    yield from iter_dom_units(v, spine, child)
                else:
                    yield UNIT, child, v


def _set_path(node, path, value) -> None:
    for elem in path[:-1]:
        node = node[elem]
    node[path[-1]] = value


def _header_view(obj: dict) -> dict:
    return {k: v for k, v in obj.items() if not isinstance(v, (dict, list))}


def iter_objects_lines(lines, profile):
    """(kind, pattern, object) — document units STREAMED by lines, without
    writing. A layout that is not line based → LayoutError (the caller drops
    what it collected and goes through iter_objects_dom). JSONL: a record the
    profile cannot parse is yielded as BadRecord."""
    if is_jsonl(profile):
        for kind, pat, ulines in iter_record_units(lines):
            if kind != RAW:
                yield kind, pat, parse_record(ulines)
        return
    for kind, pat, ulines in iter_line_units(lines, profile.spine):
        if kind != RAW:
            yield kind, pat, parse_unit(ulines, kind == HEADER)


def iter_objects_dom(text: str, profile):
    """The same over the whole tree (minified JSON); a broken document has
    no units."""
    try:
        doc = json.loads(text.lstrip("\ufeff"))
    except ValueError:
        return
    for kind, pat, obj in iter_dom_units(doc, profile.spine):
        yield kind, pat, _header_view(obj) if kind == HEADER else obj


def _each_object(text: str, profile, fn) -> None:
    """Call ``fn(kind, pattern, object)`` on every unit of the text; on a
    layout fallback ``fn`` gets ``None`` — the signal to start over."""
    if is_jsonl(profile):
        for item in iter_objects_lines(io.StringIO(text, newline=""), profile):
            fn(*item)
        return
    if is_line_layout(text[:HEAD_BYTES].lstrip("\ufeff")):
        try:
            for item in iter_objects_lines(io.StringIO(text, newline=""),
                                           profile):
                fn(*item)
            return
        except LayoutError:
            STATS["layout_fallback"] += 1
            fn(None, None, None)
    for item in iter_objects_dom(text, profile):
        fn(*item)


# --------------------------------------------------------------------------- #
# profile detection
# --------------------------------------------------------------------------- #
def top_keys(head: str) -> dict:
    """{root object key: start of its value (up to 40 chars)} from the file
    head — tolerant to layout (pretty/minified), without json.loads."""
    keys: dict = {}
    depth = 0
    i, n = 0, len(head)
    while i < n:
        c = head[i]
        if c == '"':
            j = i + 1
            while True:
                j = head.find('"', j)
                if j < 0:
                    return keys
                k = j - 1
                while k > i and head[k] == "\\":
                    k -= 1
                if (j - 1 - k) % 2 == 0:
                    break
                j += 1
            if depth == 1:
                k = j + 1
                while k < n and head[k] in " \t\r\n":
                    k += 1
                if k < n and head[k] == ":":
                    v = k + 1
                    while v < n and head[v] in " \t\r\n":
                        v += 1
                    keys.setdefault(head[i + 1:j], head[v:v + 40])
            i = j + 1
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
        i += 1
    return keys


def _profiles():
    from . import PROFILES
    return PROFILES


def _ext_kind(name: str):
    low = name.lower()
    if low.endswith(".json"):
        return "json"
    if low.endswith(JSONL_EXT):
        return "jsonl"
    return None


def detect_text(rel: str, head: str):
    """Profile by extension and text head, or None (the usual path).
    .json — document signatures (sniff), .jsonl/.ndjson — record signatures
    (sniff_jsonl; a profile without it does not accept JSONL)."""
    kind = _ext_kind(rel)
    if kind is None:
        return None
    head = head[:HEAD_BYTES].lstrip("\ufeff")
    for prof in _profiles():
        if kind == "json":
            hit = prof.sniff(head)
        else:
            fn = getattr(prof, "sniff_jsonl", None)
            hit = fn(head) if fn is not None else None
        if hit is not None:
            return hit
    return None


def detect_layout(path: str, rel: str | None = None) -> tuple:
    """(profile | None, line layout?) from the file head. The second one means
    "the file will be streamed": memory O(batch), not O(file)."""
    kind = _ext_kind(rel or path)
    if kind is None:
        return None, False
    try:
        with open(path, encoding="utf-8", errors="surrogateescape",
                  newline="") as fh:
            head = fh.read(HEAD_BYTES)
    except OSError:
        return None, False
    prof = detect_text(rel or path, head)
    if prof is not None and is_jsonl(prof):
        return prof, True                # record = line: always streamed
    if kind != "json":
        return prof, False
    if prof is None and any(k in head for k in _TG_SIGNS) and (rel or path) not in _UNRECOGNIZED:
        # Looks like a Telegram export, but no profile recognized it — do not stay
        # silent. A single "messages" key is not enough: a Discord export and any
        # other JSON with "messages" gave a WARN on every file and drowned the real
        # WARNs in a corpus of thousands of files; the signs are keys that do not
        # occur outside Telegram (from_id, date_unixtime, text_entities)
        _UNRECOGNIZED.add(rel or path)
        print(f"structured: WARN {rel or path}: the head has Telegram export keys "
              f"({'/'.join(k.strip(chr(34)) for k in _TG_SIGNS if k in head)}), no profile recognized it "
              "— the file goes as plain text", flush=True)
    return prof, prof is not None and is_line_layout(head.lstrip("\ufeff"))


_TG_SIGNS = ('"from_id"', '"date_unixtime"', '"text_entities"')
_UNRECOGNIZED: set = set()


def detect(path: str, rel: str | None = None):
    """File profile by content signature (only the head is read)."""
    return detect_layout(path, rel)[0]


# --------------------------------------------------------------------------- #
# rules over the slot projection
# --------------------------------------------------------------------------- #
def _todo_slots(slots: list) -> list:
    """Slots of a batch that go to the rules (empty/whitespace ones do not)."""
    return [s for s in slots if s.text and not s.text.isspace()]


def _iso_compute(iso: list, chain, report, iso_cache: dict) -> None:
    """Isolated slots of a batch (Slot.iso): values not yet in ``iso_cache``
    (the file cache) — in one chain call through the wide separator, once per
    unique value in the file; a rule ate the separator — fall back to a call
    per value. The same code for the sequential pass (_run_slots) and the
    preparation of file parts (plan_parts): there the same calls run in the
    same batch order, the batch report is collected separately."""
    need = []
    for s in iso:
        if s.text not in iso_cache and s.text not in need:
            need.append(s.text)
    plain = [t for t in need if SEP_CHAR not in t]
    if plain:
        joined = ISO_SEP.join(plain)
        STATS["rule_chars"] += len(joined)
        STATS["rule_calls"] += 1
        sub = Report(max_examples=report.max_examples,
                     want_why=report.want_why)
        parts = chain(joined, sub).split(ISO_SEP)
        if len(parts) == len(plain):
            report.merge(sub)
            iso_cache.update(zip(plain, parts))
        else:
            report.shadow_of(sub)        # the trace of the fallback goes to the shadow
    for t in need:
        if t not in iso_cache:
            iso_cache[t] = chain(t, report)


def _run_bulk(todo: list, chain, report) -> None:
    """Slot texts → the rule chain in one call through SEP; a rule ate the
    separator — slot by slot (the batch report goes to the shadow)."""
    bulk = [s for s in todo if SEP_CHAR not in s.text]
    solo = [s for s in todo if SEP_CHAR in s.text]
    if bulk:
        joined = SEP.join(s.text for s in bulk)
        STATS["rule_chars"] += len(joined)
        STATS["rule_calls"] += 1
        sub = Report(max_examples=report.max_examples,
                     want_why=report.want_why)
        parts = chain(joined, sub).split(SEP)
        if len(parts) == len(bulk):
            report.merge(sub)
            for s, new in zip(bulk, parts):
                s.out = new
        else:
            STATS["sep_fallback"] += 1
            # a rule that ate the separator may not fire at all in the
            # slot-by-slot pass — its trace stays in the report shadow
            report.shadow_of(sub)
            solo = todo
    for s in solo:
        STATS["rule_chars"] += len(s.text)
        STATS["rule_calls"] += 1
        s.out = chain(s.text, report)


def _run_slots(slots: list, chain, report, secret_fn, iso_cache,
               pre=None) -> None:
    """Slot texts of a batch → the rule chain in ONE call over the projection
    (texts joined by SEP). A rule ate the separator (part count mismatch) —
    fall back to a slot-by-slot pass; the batch report is merged only on
    success. Isolated slots (Slot.iso) go in a separate call through the wide
    separator, once per unique value in the file (``iso_cache``).

    ``pre`` — isolated slots already computed by the part preparation:
    (outputs of all file values, report of THIS batch | None) — merged into
    the report exactly where the sequential pass would call _iso_compute."""
    todo = _todo_slots(slots)
    if pre is not None and chain is not None \
            and (pre[1] is not None) != any(s.iso for s in todo):
        raise SystemExit(
            "structured: isolated slots of the batch differ from the part "
            "preparation (non-deterministic profile) — the part cannot repeat "
            "the sequential pass")
    if not todo or chain is None:
        return
    iso = [s for s in todo if s.iso]
    if iso:
        todo = [s for s in todo if not s.iso]
        if pre is None:
            _iso_compute(iso, chain, report, iso_cache)
        else:
            iso_cache, iso_rep = pre
            lost = [s.text for s in iso if s.text not in iso_cache]
            if lost:
                raise SystemExit(
                    f"structured: a file part sees {len(lost)} isolated "
                    "values missing from the part preparation — the batch plan "
                    "differs from the preparation (non-deterministic profile)")
            report.merge(iso_rep)
        for s in iso:
            s.out = iso_cache[s.text]
            # identity field: the rules did not make the value a token as a
            # whole → the field token (deterministic, one per name)
            if s.whole is not None and not is_token(s.out):
                report.record(s.whole[0], s.text, s.whole[1])
                s.out = s.whole[1]
    # HTML slots (Slot.html) — their own chain call over the HTML form
    # (chain.html; without it — the common chain); without them — one call
    html_chain = getattr(chain, "html", None) or chain
    plain = [s for s in todo if not s.iso and not s.html]
    html = [s for s in todo if not s.iso and s.html]
    for group, ch in ((plain, chain), (html, html_chain)):
        if group:
            _run_bulk(group, ch, report)
    # post-rules step (file name: name detector); if it changed the text — one
    # more rule pass over its output, and so on to a fixed point (≤ POST_ROUNDS
    # rounds; the rules minted DOC_ID from a number, the detector picks up the
    # id tail next to the token, and a repeated dry run over the output would
    # otherwise replace again): the fixed point is reached in the same pass
    redo = [s for s in todo if s.post is not None]
    for _ in range(POST_ROUNDS):
        again = []
        for s in redo:
            new = s.post(s.out, report)
            if new != s.out:
                s.out = new
                again.append(s)
        if not again:
            break
        for s in again:
            STATS["rule_chars"] += len(s.out)
            STATS["rule_calls"] += 1
            s.out = (html_chain if s.html else chain)(s.out, report)
        redo = again
    else:
        # rounds exhausted, the last one still replaced: no fixed point — a
        # repeated dry run over the output may replace again (the round
        # idempotency oracle catches it); WARN to the stage log, no values
        keys = Counter(s.key for s in redo)
        print(f"structured: WARN no fixed point of detector→rules in "
              f"{POST_ROUNDS} rounds (Slot.post): "
              + ", ".join(f"{k} ×{n}" for k, n in sorted(keys.items())))
    if secret_fn is not None:
        for s in todo:                   # isolated slots are names, not values
            new = secret_fn(s.out, s.key)
            if new != s.out:
                report.record("entropy_secret", "<value>", P.SECRET)
                s.out = new


# --------------------------------------------------------------------------- #
# file masking
# --------------------------------------------------------------------------- #
def mask_file(path: str, rel: str, profile, registry, chain, report,
              apply: bool, secret_fn=None, dest=None) -> bool:
    """Mask a structured file by its profile. ``chain(text, report) -> text``
    is the pipeline rule chain. Returns whether the file changed. Writing is
    atomic (StreamWriter), a dry run creates nothing. The file report is
    collected separately and merged into ``report`` only when the path
    succeeds.

    ``dest`` — where to write the result with ``apply`` (None — in place of
    ``path``): the round computation saves the finished output to the store,
    and writing then substitutes it without running the rules again."""
    try:
        with open(path, encoding="utf-8", errors="surrogateescape",
                  newline="") as fh:
            head = fh.read(HEAD_BYTES)
    except OSError:
        return False                     # mirrors read_text: unreadable file
    if is_jsonl(profile):
        # record = line: no fallback through the tree (the file is not read
        # into memory as a whole); a record the profile cannot parse goes on
        # its own (BadRecord), a substitution failure — json.dumps of the record
        sub = Report(max_examples=report.max_examples, want_why=report.want_why)
        changed = _mask_lines(path, rel, profile, registry, chain, sub,
                              apply, secret_fn, dest)
        report.merge(sub)
        return changed
    if is_line_layout(head.lstrip("\ufeff")):
        sub = Report(max_examples=report.max_examples, want_why=report.want_why)
        try:
            changed = _mask_lines(path, rel, profile, registry, chain, sub,
                                  apply, secret_fn, dest)
        except LayoutError:
            STATS["layout_fallback"] += 1
            # a layout fallback may come from a rule edit (apply_edits) —
            # the trace of the dropped line pass goes to the shadow
            report.shadow_of(sub)
        else:
            report.merge(sub)
            return changed
    sub = Report(max_examples=report.max_examples, want_why=report.want_why)
    changed = _mask_dom(path, rel, profile, registry, chain, sub, apply,
                        secret_fn, dest)
    report.merge(sub)
    return changed


def _mask_units(items, sess, chain, report, secret_fn, write=None,
                iso=None, max_units=None) -> bool:
    """The common loop of the line layout: units are collected in batches of
    BATCH_UNITS, a batch is one rule pass over the projection of its slots
    (_run_slots), then the edits are substituted into the unit lines.
    ``items`` — the iter_line_units stream; ``write(str)`` — the output sink
    (None — counting only); returns whether anything changed.

    File part: ``iso`` — (outputs of the file's isolated values, {batch
    number in the PART: report}) from the preparation (plan_parts) instead of
    a local cache; ``max_units`` — stop right after the batch at which this
    many units were reached (border of the next part; None — to the end of
    the file). The sequential pass over a file is the same code with
    iso=None, max_units=None."""
    changed = False
    batch: list = []
    nunits = 0
    done = 0
    bidx = 0
    iso_cache: dict = {}
    prof = sess.p
    bsize = batch_of(prof)

    def flush():
        nonlocal changed
        slots = []
        for it in batch:
            if type(it) is not str:
                slots.extend(it[2].slots)
        pre = None
        if iso is not None:
            pre = (iso[0], iso[1].get(bidx))
        _run_slots(slots, chain, report, secret_fn, iso_cache, pre)
        for it in batch:
            if type(it) is str:
                if write is not None:
                    write(it)
                continue
            ulines, header, plan = it
            for s in plan.slots:
                s.commit(plan.edits)
            if plan.edits:
                STATS["edits"] += len(plan.edits)
                new = apply_unit(prof, ulines, plan.edits, header)
                if new != ulines:
                    changed = True
                ulines = new
            if write is not None:
                write("".join(ulines))
        batch.clear()

    for kind, pat, ulines in items:
        if kind == RAW:
            batch.append(ulines)
            continue
        STATS["units"] += 1
        header = kind == HEADER
        plan = unit_plan(sess, prof, pat, ulines, header)
        if plan is None:
            batch.append("".join(ulines))
        else:
            batch.append((ulines, header, plan))
        nunits += 1                      # every unit, with or without a plan
        if nunits >= bsize:
            flush()
            nunits = 0
            bidx += 1
            done += bsize
            if max_units is not None and done >= max_units:
                return changed           # part border: the next part goes on
    flush()
    return changed


def _mask_lines(path, rel, profile, registry, chain, report, apply,
                secret_fn, dest=None) -> bool:
    sess = profile.session(registry, report)
    lines = walk.iter_lines(path)
    writer = walk.StreamWriter(dest or path) if apply else None
    try:
        changed = _mask_units(iter_units(lines, profile), sess,
                              chain, report, secret_fn,
                              writer.write if writer is not None else None)
    except BaseException:
        # a half-written tmp is not published; the error goes up (LayoutError →
        # fallback path in mask_file, anything else — loudly)
        if writer is not None:
            writer.abort()
        raise
    if writer is not None:
        writer.commit(changed)
    return changed


def _mask_dom(path, rel, profile, registry, chain, report, apply,
              secret_fn, dest=None) -> bool:
    text = walk.read_text(path)
    if text is None:
        return False
    try:
        doc = json.loads(text.lstrip("\ufeff"))
    except ValueError:
        # the signature in the head matched but the document is broken — rules
        # over the text, as for a plain file (nothing is lost silently)
        new = chain(text, report) if chain is not None else text
        if new != text and apply:
            walk.write_text(dest or path, new)
        return new != text
    sess = profile.session(registry, report)
    units = []
    for kind, pat, obj in iter_dom_units(doc, profile.spine):
        STATS["units"] += 1
        header = kind == HEADER
        plan = sess.plan(pat, _header_view(obj) if header else obj, header)
        if plan is not None:
            units.append((obj, plan))
    slots = [s for _o, plan in units for s in plan.slots]
    _run_slots(slots, chain, report, secret_fn, {})
    changed = False
    for obj, plan in units:
        for s in plan.slots:
            s.commit(plan.edits)
        for p, v in plan.edits.items():
            _set_path(obj, p, v)
            changed = True
    if not changed:
        return False
    if apply:
        indent = _detect_indent(text)
        seps = None if indent is not None or '": ' in text[:HEAD_BYTES] \
            else (",", ":")
        out = _escape_surrogates(json.dumps(doc, ensure_ascii=False,
                                            indent=indent, separators=seps))
        if text.endswith("\n"):
            out += "\r\n" if text.endswith("\r\n") else "\n"
        if text.startswith("\ufeff"):
            out = "\ufeff" + out
        walk.write_text(dest or path, out)
    return True


# --------------------------------------------------------------------------- #
# a file in parts — batches of one file on different workers
# --------------------------------------------------------------------------- #
class _ByteLines:
    """Lines of a text stream (newline='', surrogateescape) counting source
    bytes: ``pos`` — the offset right after the last yielded line, ``last`` —
    its length. utf-8 + surrogateescape round-trips byte for byte, so the sum
    of lengths is the exact file offset (BOM is the U+FEFF char of the first
    line, the same 3 bytes)."""
    __slots__ = ("fh", "pos", "last")

    def __init__(self, fh, pos: int = 0):
        self.fh = fh
        self.pos = pos
        self.last = 0

    def __iter__(self):
        for line in self.fh:
            n = len(line.encode("utf-8", "surrogateescape"))
            self.last = n
            self.pos += n
            yield line


def _open_text(path: str, offset: int = 0):
    """Text stream of the file from byte offset ``offset`` (a line start) —
    the same decoding and line splitting as walk.iter_lines."""
    fb = open(path, "rb")
    try:
        if offset:
            fb.seek(offset)
        return io.TextIOWrapper(fb, encoding="utf-8",
                                errors="surrogateescape", newline="")
    except BaseException:
        fb.close()
        raise


class PartPlan:
    """Preparation of a file for processing in parts (plan_parts).

    ``starts[b]`` — (byte offset of the start of batch b, the spine stack
    there) for every batch with units; ``iso_map`` — outputs of ALL isolated
    values of the file; ``iso_reps[b]`` — the report of the isolated values of
    batch b (Report.part), if they were computed in that batch;
    ``units``/``size`` — counts; ``batch`` — units per batch (batch_of of the
    profile)."""
    __slots__ = ("starts", "iso_map", "iso_reps", "units", "size", "batch")

    def __init__(self, starts, iso_map, iso_reps, units, size,
                 batch=BATCH_UNITS):
        self.starts = starts
        self.iso_map = iso_map
        self.iso_reps = iso_reps
        self.units = units
        self.size = size
        self.batch = batch

    def parts(self, per_part: int = 1, min_bytes: int = 0) -> list:
        """Parts of ``per_part`` batches (and not shorter than ``min_bytes``
        bytes, except the last — JSONL: batches are small, a part is a few MB):
        [(start offset, end offset, stack, max units | None for the last,
        {batch number in the part: report})]. The last part reads to the end
        of the file (the spine tail)."""
        per_part = max(1, int(per_part))
        nb = len(self.starts)
        out = []
        b0 = 0
        while b0 < nb:
            b1 = min(b0 + per_part, nb)
            while b1 < nb and self.starts[b1][0] - self.starts[b0][0] \
                    < min_bytes:
                b1 += 1
            last = b1 >= nb
            out.append((self.starts[b0][0],
                        self.size if last else self.starts[b1][0],
                        self.starts[b0][1],
                        None if last else (b1 - b0) * self.batch,
                        {b - b0: self.iso_reps[b]
                         for b in range(b0, b1) if b in self.iso_reps}))
            b0 = b1
        return out


def plan_parts(path: str, profile, registry, chain, max_examples: int = 30,
               want_why: bool = True):
    """Prepare processing of a file IN PARTS on different workers.

    The sequential pass over the line layout (_mask_units) cuts the file into
    batches of BATCH_UNITS units, and the rules see only the projection of
    their batch — batches are independent except for the cache of isolated
    values (channel/group names in identity fields are computed once per
    FILE, a batch report depends on what came before). One stream over the
    file:
      * batch borders — the byte offset and the spine stack: a part starts
        with a seek into the middle of the file and parses exactly the same
        units. A batch closed by a HEADER ends BEFORE the line at which it was
        yielded (the line is already read but belongs to the next batch);
      * isolated values — EXACTLY as in the sequential pass: a plan of every
        unit (same profile and registry), new values of a batch →
        _iso_compute in batch order; the batch report is collected separately
        (Report.part) and merged by the part at its place (_run_slots pre).
    Returns PartPlan; None — the layout is not line based or broken
    (LayoutError): the file goes whole the old way (with the same DOM
    fallback there). Plan records (identity field masks) go to a separate
    report and are thrown away: the parts write them to the file report
    themselves."""
    size = os.path.getsize(path)
    sess = profile.session(registry, Report(want_why=False))
    bsize = batch_of(profile)
    starts = [(0, ())]
    iso_map: dict = {}
    iso_reps: dict = {}
    pending: list = []                  # isolated slots of the current batch
    units = nunits = bidx = 0
    stack: list = []

    def close_batch():
        if pending and chain is not None:
            rep = Report.part(max_examples, want_why)
            _iso_compute(pending, chain, rep, iso_map)
            iso_reps[bidx] = rep
        pending.clear()

    fh = _open_text(path)
    try:
        lines = _ByteLines(fh)
        for kind, pat, ulines in iter_units(lines, profile, stack):
            if kind == RAW:
                continue
            header = kind == HEADER
            plan = unit_plan(sess, profile, pat, ulines, header)
            if plan is not None:
                pending.extend(s for s in _todo_slots(plan.slots) if s.iso)
            units += 1
            nunits += 1
            if nunits >= bsize:
                close_batch()
                nunits = 0
                bidx += 1
                off = lines.pos if kind == UNIT else lines.pos - lines.last
                starts.append((off, tuple(stack)))
    except LayoutError:
        STATS["layout_fallback"] += 1
        return None
    finally:
        fh.close()
    close_batch()
    if units and units % bsize == 0:
        starts.pop()                    # after the last full batch — only the tail
    return PartPlan(starts, iso_map, iso_reps, units, size, bsize)


def mask_part(path: str, profile, registry, chain, report, secret_fn, part,
              want_text: bool = False) -> tuple:
    """Process a PART of a file (see plan_parts/PartPlan.parts) — the same
    batch loop (_mask_units) as the sequential pass, with isolated values
    ready. ``report`` — Report.part(); returns (whether the part changed, its
    output | None). The part must read exactly its byte range — otherwise the
    border differs from the preparation (SystemExit: a silently glued file is
    corruption)."""
    off0, off1, stack, max_units, reps = part[:5]
    iso_map = part[5]
    sess = profile.session(registry, report)
    out = [] if want_text else None
    last = [None]
    fh = _open_text(path, off0)
    try:
        lines = _ByteLines(fh, off0)

        def items():
            for it in iter_units(lines, profile, list(stack)):
                last[0] = it[0]
                yield it

        changed = _mask_units(items(), sess, chain, report, secret_fn,
                              out.append if out is not None else None,
                              iso=(iso_map, reps), max_units=max_units)
        end = lines.pos
        if max_units is not None and last[0] == HEADER:
            end -= lines.last           # the HEADER line belongs to the next part
    finally:
        fh.close()
    if end != off1:
        raise SystemExit(f"structured: part [{off0}, {off1}) of file "
                         f"{os.path.basename(path)!r} read up to {end} — "
                         "the batch border differs from the part preparation")
    return changed, ("".join(out) if out is not None else None)


# --------------------------------------------------------------------------- #
# projections: mining / NER / audit
# --------------------------------------------------------------------------- #
def projection(text: str, profile, ner: bool = False, registry=None) -> str:
    """Text projection of a file: one line block per free-text field in
    document order; field names, ids, dates, sizes, paths and identity fields
    do not get here. ``ner=True`` — natural language only (no URLs/file
    names/button data). Values are the ORIGINAL ones (before the rules)."""
    out: list = []

    _each_object(text, profile, _proj_fn(profile.session(registry, None),
                                         ner, out, set()))
    return "\n".join(out) + ("\n" if out else "")


def _proj_fn(sess, ner: bool, out: list, seen_iso: set):
    def fn(kind, pat, obj):
        if kind is None:                 # layout fallback: start over
            out.clear()
            seen_iso.clear()
            return
        try:
            plan = bad_plan(obj) if isinstance(obj, BadRecord) \
                else sess.plan(pat, obj, kind == HEADER)
        except RecursionError:
            STATS["record_bad"] += 1     # record deeper than the limit: masking
            return                       # takes the whole line (record_plan)
        if plan is None:
            return
        for s in plan.slots:
            if ner and not s.ner:
                continue
            raw = s.raw()
            if raw and not raw.isspace():
                if s.iso:                # a channel name — once per file, not
                    if raw in seen_iso:  # once per each of its messages
                        continue
                    seen_iso.add(raw)
                out.append(raw)
    return fn


def jsonl_profile_of(rel: str, head: str):
    """JSONL profile by name and text head (an archive member) or None."""
    if not profiles_enabled() or _ext_kind(rel) != "jsonl":
        return None
    prof = detect_text(rel, head)
    return prof if prof is not None and is_jsonl(prof) else None


def projection_file(path: str, profile, ner: bool = False,
                    registry=None) -> str:
    """``projection`` of a JSONL profile file STREAMED by lines: only the
    projection (free text) is in memory, not the whole file (a YouTrack export
    is hundreds of MB with a small share of text). Not a JSONL profile — read
    whole and projection, as before."""
    if not is_jsonl(profile):
        with open(path, encoding="utf-8", errors="replace") as fh:
            return projection(fh.read(), profile, ner, registry)
    out: list = []
    fn = _proj_fn(profile.session(registry, None), ner, out, set())
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for item in iter_objects_lines(fh, profile):
            fn(*item)
    return "\n".join(out) + ("\n" if out else "")


def text_lines(path: str, profile=None):
    """File lines STREAMED with numbering kept: a line with a free-text value
    is yielded as is, any other one (key brackets, ids, dates, paths, identity
    fields, the ``text`` duplicate with ``text_entities``) — empty. Contract of
    tools/audit_zones._iter_lines: items of ``text.split("\\n")``, utf-8
    errors=replace, universal newlines. Profile unknown or the layout is not
    line based → lines unchanged."""
    if profile is None:
        profile = detect(path)

    def raw_lines():
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            rest = ""
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    yield rest
                    return
                parts = (rest + chunk).split("\n")
                rest = parts.pop()
                yield from parts

    if profile is None:
        yield from raw_lines()
        return
    if is_jsonl(profile):
        # record = line: the projection line is the record's free-text values
        # as JSON literals (record_text_line), the same line numbering
        sess = profile.session(None, None)
        last = ""
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                last = line
                if line.strip():
                    yield record_text_line(
                        unit_plan(sess, profile, REC, [line], False))
                else:
                    yield ""
        if last == "" or last.endswith("\n"):
            yield ""      # the item after the final "\n" (split("\n") contract)
        return
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(HEAD_BYTES)
    if not is_line_layout(head.lstrip("\ufeff")):
        yield from raw_lines()
        return
    sess = profile.session(None, None)
    emitted = 0
    last = [""]

    def tracked(fh):
        for line in fh:
            last[0] = line
            yield line

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for kind, pat, ulines in iter_line_units(tracked(fh),
                                                     profile.spine):
                if kind == RAW:
                    emitted += 1
                    yield ""
                    continue
                header = kind == HEADER
                plan = sess.plan(pat, parse_unit(ulines, header), header)
                keep = set()
                if plan is not None:
                    # slot paths; the text mirror (text_entities duplicate) is
                    # not among them — the zone is not doubled
                    for s in plan.slots:
                        keep.update(s.paths)
                show = [""] * len(ulines)
                if keep:
                    prefixes = {p[:k] for p in keep for k in range(1, len(p))}
                    for i, p, _v, body, _e in walk_scalars(ulines, header,
                                                           prefixes):
                        if p in keep:
                            show[i] = body
                emitted += len(show)
                yield from show
    except LayoutError:
        # the layout broke in the middle of the file: the rest goes as is, line
        # numbering is kept (lines already yielded are not recounted)
        for n, line in enumerate(raw_lines()):
            if n >= emitted:
                yield line
        return
    if last[0] == "" or last[0].endswith("\n"):
        yield ""      # the item after the final "\n" (split("\n") contract)


def profiles_enabled() -> bool:
    """Profile switch for readers outside the engine (gates, residual checks,
    audit zones): ANON_STRUCTURED_PROFILES=0 — the file as plain text."""
    return os.environ.get("ANON_STRUCTURED_PROFILES", "1").strip() != "0"


def gate_text(path, rel=None, text=None):
    """Text of a structured profile file AS THE RESIDUAL CHECKS SEE IT (the
    round control gate, residual/verify, the release archive gate): only the
    values of free-text fields — what the profile hands to the rules. ``None``
    — the file is not structured (or profiles are off): read as usual.

    Why: the check "an approved literal did not stay in the output" reads the
    file as plain text, while the profile deliberately leaves service fields
    alone (type/action enums, ids, dates, the exporter's template ``about``).
    A pack literal equal to the value of such a field gave a deterministic
    control gate BLOCK (e.g. "Telegram" in ``about``) that could not be fixed:
    the rules do not go into that field. Identity fields are not part of the
    projection — they are replaced with a whole token by the field table.

    ``path`` — a file on disk: a line layout is returned with THE SAME line
    numbering (text_lines), a minified one — as a projection. ``text`` — the
    content already in memory (a zip archive member): a projection."""
    if not profiles_enabled():
        return None
    if text is None:
        prof, by_lines = detect_layout(path, rel)
        if prof is None:
            return None
        if by_lines:
            return "\n".join(text_lines(path, prof))
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        return projection(text, prof)
    prof = detect_text(rel or path or "", text)
    if prof is None:
        return None
    return projection(text, prof)


GATE_CHUNK = 4 << 20          # raw text chars per gate chunk


def is_jsonl_file(path, rel=None) -> bool:
    """A JSONL profile file (YouTrack export) with profiles enabled."""
    if not profiles_enabled():
        return False
    prof = detect(path, rel)
    return prof is not None and is_jsonl(prof)


def raw_chunks(lines, size: int | None = None):
    """RAW text of a line stream in chunks of ≈ ``size`` chars, the border is
    after a newline (joined chunks == the text). The gates (control, release
    archive, group-sync) read a JSONL profile raw — as before the profile:
    the value of any field, a service one too, does not slip past them; in
    chunks — a file of hundreds of MB is not read into memory whole."""
    if size is None:
        size = GATE_CHUNK
    buf, n = [], 0
    for line in lines:
        buf.append(line)
        n += len(line)
        if n >= size:
            yield "".join(buf)
            buf, n = [], 0
    if buf:
        yield "".join(buf)


# --------------------------------------------------------------------------- #
# entity registry of the tree
# --------------------------------------------------------------------------- #
_MEMO: dict = {}             # source root -> (fingerprint, Registry)
CACHE_NAME = "structured_registry.json"


def _run_layout(root: str):
    """(source tree, state directory | None). Run convention — as in the
    structured gate: <run>/work is processed, the originals are in <run>/raw
    (work/ is already masked after the first round — no names in it). Outside
    a run the source is the tree itself, no cache."""
    root = os.path.abspath(root)
    run_dir = os.path.dirname(root)
    state = os.path.join(run_dir, "state")
    if os.path.basename(root) in ("work", "raw") and os.path.isdir(state):
        raw = os.path.join(run_dir, "raw")
        if os.path.isdir(raw) and any(os.scandir(raw)):
            return raw, state
        return root, state
    return root, None


def _json_files(root: str) -> list:
    """Registry candidate files (and its cache key): all .json and
    .jsonl/.ndjson of a STRUCTURED profile (signature by head: plain JSONL —
    Discord, logs — are not in the key, editing them does not reset the
    registry cache)."""
    out = []
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            kind = _ext_kind(fn)
            if kind is None:
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            if kind == "jsonl" and detect(p, rel) is None:
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            out.append((p, rel, st.st_size, st.st_mtime_ns))
    return out


_REG_POOL_MIN = 4            # structured files: fewer — collected in this process


def _collect_task(task) -> tuple:
    """Registry of one file in a pool worker (``build_registry``) → (Registry, STATS increments)."""
    before = Counter(STATS)
    part = build_registry([task])
    return part, STATS - before


def build_registry(files, workers: int = 1) -> Registry:
    """Registry from a list of (path, rel): structured files only, streamed. With ``workers`` > 1 the
    files are collected in forked processes, each into its own registry, and merged here in file order:
    collecting only adds (counters, the first occurrence of a name) — the registry is that of one pass."""
    files = list(files)
    if workers > 1 and len(files) >= _REG_POOL_MIN:
        from ..engine.pipeline import fork_context
        ctx = fork_context()
        if ctx is not None:
            from ..engine.pool import run_pool
            done, _dead = run_pool(_collect_task, files, min(workers, len(files)), ctx,
                                   label="structured:registry", name=lambda _t: "a structured file")
            reg = Registry()
            for i, task in enumerate(files):
                part, stats = done[i] if i in done else _collect_task(task)
                reg.merge(part)
                STATS.update(stats)
            return reg
    reg = Registry()
    for path, rel in files:
        prof = detect(path, rel)
        if prof is None:
            continue
        try:
            with open(path, encoding="utf-8", errors="surrogateescape",
                      newline="") as fh:
                if is_jsonl(prof):
                    part = Registry()
                    for kind, pat, obj in iter_objects_lines(fh, prof):
                        if not isinstance(obj, BadRecord):
                            try:
                                prof.collect(pat, obj, False, part, rel)
                            except RecursionError:
                                STATS["record_bad"] += 1
                    reg.merge(part)
                    continue
                head = fh.read(HEAD_BYTES)
                fh.seek(0)
                if is_line_layout(head.lstrip("\ufeff")):
                    part = Registry()
                    try:
                        for kind, pat, obj in iter_objects_lines(fh, prof):
                            prof.collect(pat, obj, kind == HEADER, part, rel)
                        reg.merge(part)
                        continue
                    except LayoutError:
                        STATS["layout_fallback"] += 1
                        fh.seek(0)
                text = fh.read()
        except OSError:
            continue
        for kind, pat, obj in iter_objects_dom(text, prof):
            prof.collect(pat, obj, kind == HEADER, reg, rel)
    return reg


# sources the registry collect counters depend on — part of the cache fingerprint
_COLLECT_SRC = ("entities.py", "telegram.py", "youtrack.py")
_COLLECT_SHA: dict = {}


def _collect_code_sha256() -> str:
    """sha256 of the collect sources (_COLLECT_SRC next to the module); memo per process."""
    d = os.path.dirname(os.path.abspath(__file__))
    key = tuple(_COLLECT_SRC)
    got = _COLLECT_SHA.get(key)
    if got is None:
        h = hashlib.sha256()
        for name in _COLLECT_SRC:
            with open(os.path.join(d, name), "rb") as fh:
                h.update(name.encode("utf-8") + b"\0" + hashlib.sha256(fh.read()).digest())
        got = _COLLECT_SHA[key] = h.hexdigest()
    return got


def registry_for_tree(root: str, workers: int = 1) -> Registry:
    """Entity registry of the structured files of a tree: from the run cache
    (state/structured_registry.json, key — fingerprint of the source .json
    files) or built from the source (``build_registry`` over ``workers``). In-process — memo by root."""
    src, state = _run_layout(root)
    files = _json_files(src)
    h = hashlib.sha256(f"v{_REG_VERSION}\n".encode())
    # the registry of the strict "our token" mode (a nick "PETROV_77" is a name, not
    # a token) — its own fingerprint line (the switch is gone, the line stays so
    # that run caches with the same key are reused)
    h.update(b"own_token_strict\n")
    # the cache holds raw collect counters — an edit of the collect code
    # (telegram.py / youtrack.py / Registry.add_* in entities.py) on a repeated batch
    # with the same raw/ took the old registry (no new people/logins, their tokens are
    # not minted). The canon and minting code is not in the cache — it runs after loading
    h.update(_collect_code_sha256().encode("ascii") + b"\n")
    for _p, rel, size, mtime in files:
        h.update(f"{rel}\t{size}\t{mtime}\n".encode("utf-8", "surrogateescape"))
    fp = h.hexdigest()
    memo = _MEMO.get(src)
    if memo is not None and memo[0] == fp:
        return memo[1]
    cache = os.path.join(state, CACHE_NAME) if state else None
    reg = None
    if cache and os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as fh:
                doc = json.load(fh)
            if doc.get("fingerprint") == fp and doc.get("version") == _REG_VERSION:
                reg = Registry.from_json(doc)
        except (OSError, ValueError):
            reg = None
    if reg is None:
        reg = build_registry([(p, rel) for p, rel, _s, _m in files], workers)
        if cache:
            doc = reg.to_json()
            doc["fingerprint"] = fp
            tmp = cache + f".{os.getpid()}.tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(doc, fh, ensure_ascii=False)
                os.chmod(tmp, 0o600)
                os.replace(tmp, cache)
            except OSError:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    _MEMO[src] = (fp, reg)
    return reg
