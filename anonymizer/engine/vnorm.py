"""A single value normalization: the view and the key of a value.

The project has six normalizers (ru_trace.norm, fix_engine._trace_norm, gold_store.norm_form
and case_key, residual._cf_with_map, control.yo_fold) and about 60 comparison places with different semantics.
This module is a single point. Three levels, one function each; they are wired in step by step, here
the module is not wired anywhere and does not change the conveyor output.

1. ``transport_view(rel, text, *, esc_ws=False) -> (view, spans)`` — strip the in-file encoding.
   ``.json``/``.jsonl``/``.ndjson``: ``\\uXXXX`` (with surrogate pairs) and ``\\/`` are decoded;
   the parity of ``\\`` is respected (``\\\\u0417`` is an escaped backslash and letters, not expanded).
   Other escapes (``\\n``, ``\\"``, ``\\\\``) — as is: for boundary.L that is a boundary; ``\\uXXXX`` with
   «"», «\\», a control character, a line separator (U+0085, U+2028, U+2029 — ``splitlines``
   treats them as line breaks, the line numbering of checks does not shift) or a lone surrogate also
   stays. ``esc_ws=True`` — an escape of
   whitespace (``\\n`` ``\\t`` ``\\r`` ``\\b`` ``\\f`` ``\\v``) becomes a space: the mine view.
   ``.html``/``.htm``/``.xml``: ``&#NNNN;``, ``&#xHHHH;`` and ``&nbsp;`` are decoded; markup
   characters (``<>&"'``) and control characters stay escaped. Numeric entities
   ``&#NNNN;``/``&#xHHHH;`` (not
   ``&nbsp;``) are decoded in ``.json``/``.jsonl``/``.ndjson`` too (in one pass with JSON escapes —
   one map) and in ``.txt``; «\\» (``&#92;``) stays escaped there, like markup characters:
   otherwise the JSON view would get a new escape. Other files — as is.
2. ``fold_text(s, *, case) -> (folded, back)`` — per-character folding with a position map (a generalization
   of residual._cf_with_map and control.yo_fold): case (``case=True`` — casefold), ё→е (by the
   ``ANON_YO_FOLD`` switch, like residual/control/archive), horizontal whitespace (NBSP, narrow, tab,
   a run of spaces) → one space, dashes U+2010–U+2015 and U+2212 → «-», invisibles (soft hyphen, zero
   width, U+2060, BOM) removed, NFC for a letter with a combining mark, stress U+0301/U+0300 removed,
   Latin twins in a mixed-script token (all "foreign" letters are twins) → Cyrillic (and
   vice versa if the twins are Cyrillic). A line break is not folded (decision orgs.py:62-69,
   test_packgen_ws): «Ромашкин\\nТрейд» is not «Ромашкин Трейд».
   Invisibles and removed stress marks are transparent: a whitespace run extends through them
   («Ромашкин \\u200b Трейд» → «ромашкин трейд»), a combining mark after them attaches to the letter
   before them («Фе» + soft hyphen + diaeresis + «доров» → «федоров», «и» + stress + breve → «й»),
   with ё→е the base «ё» is composed as «е». Folding is idempotent (a second pass changes nothing), except
   three boundaries outside the Russian corpus: a character that casefold expands into several (ß → ss, ŉ → ʼn, ΐ) is itself
   not normalized; a character outside the _COMB blocks (Arabic) is not composed; twins are folded after
   NFC, and a Cyrillic twin with an unattached mark in a token with Latin («jbт» + dot above)
   is composed as Latin on the second pass.
3. ``value_key(s, *, case=True) -> str`` — the comparison key: ``fold_text`` (ё→е always, like
   ru_trace.norm/_trace_norm/case_key) plus trimming of edge punctuation, like ru_trace.norm. The transport
   level is not part of the key: keys are built from already decoded values.

The position map (``spans``/``back``) is a ``PosMap`` or ``None`` (the view equals the source by positions:
the fast path). ``PosMap.span(a, b)`` — the source segment under the view segment ``[a, b)``;
``back_span(a, b, *maps)`` — the same through a chain of maps (folding over the transport view).
The map stores only "uneven" segments (where the view and the source differ in length), memory O(their number).

Fast path: a string without folding characters is returned as is, without a map and without a copy (with
``case=True`` there is one copy — casefold). The "is folding needed" decision is a search with one regex class,
a check for «  » and a search for Latin next to Cyrillic; each such pass costs about as much as casefold itself
(stdlib CPython: ~2–3 ns per character), so a one-dimensional ``translate`` is not used (4–5 times more expensive than
``replace``), and the slow path builds the view by segments between occurrences, not per character.
Cost and properties — tests/test_vnorm.py (a bench of ~3.6 million characters: ``--bench-only``; the map property
by fuzzing; idempotence; value_key ⊇ ru_trace.norm on dictionary forms). The bench (laptop, nice 15, process
time): the whole view with casefold — ×4.2–4.4 of bare casefold and 2.7 times cheaper than today's
preparation in residual._hits_ac (casefold + two yo_fold translates); one pass of the class regex without
an occurrence — ×1.07 casefold, so the goal "no more than +10 % to casefold" is unreachable in CPython: without
at least one pass the "is folding needed" decision cannot be made, and the fast path on text with ё and without
folding characters costs casefold + four passes (×5.7). Hard time limits — only under
``--bench-only``; in a suite run the limits have a ×10 margin.
"""
from __future__ import annotations

import heapq
import os
import re
import unicodedata
from array import array
from bisect import bisect_right

from .vocab import HOMOGLYPH

__all__ = ["PosMap", "ChainMap", "back_span", "fold_text", "fold_view", "view_pos", "homoglyph_view",
           "transport_view", "value_key", "canon_literal", "YO_FOLD_ENV", "yo_fold_on",
           "transport_rel_on", "EDGE"]

# The same switch as anonymizer.gates.control.YO_FOLD_ENV: without importing gates —
# the engine does not pull in the gates.
YO_FOLD_ENV = "ANON_YO_FOLD"
# Owner's decisions (variant B): engine rules, the round control gate, residual and the
# archive gate work on the transport view of .json/.jsonl/.ndjson/.html/.htm/.xml (transport_view); the output
# keeps the original encoding (viewmask). The control gate — by its own switch ANON_GATE_JSON_UNESCAPE
# (not set → by the view). Numeric entities &#NNNN;/&#xHHHH;
# are decoded in .json/.jsonl/.ndjson too (in one pass with JSON escapes) and in .txt — the engine,
# residual, the control gate and the archive gate use the same view (transport_view; control — _view_rel by
# transport_rel_on). The TRANSPORT_VIEW and TRANSPORT_ENTITIES switches are removed (the view is always on).


def yo_fold_on() -> bool:
    return os.environ.get(YO_FOLD_ENV, "1").strip() != "0"


def transport_rel_on(rel: str) -> bool:
    """File ``rel`` may get a transport view (by extension): JSON, HTML and ``.txt``
    (numeric entities). For choosing the view in the streaming control gate."""
    ext = os.path.splitext(rel or "")[1].lower()
    return ext in _JSON_EXT or ext in _HTML_EXT or ext in _ENT_EXT


# --------------------------------------------------------------------------- position map
class PosMap:
    """View positions → source segments. ``start(i)``/``end(i)`` — the source segment under the i-th view character
    (``start(len(view))`` — the source length without trailing removed characters), ``span(a, b)`` — the source
    segment under ``view[a:b]`` (an empty segment is a point). Only uneven segments are stored
    ``[v0, v1) → [s0, s1)``: the view and the source differ in length (a run of spaces → a space, ``\\uXXXX`` → a character,
    a removed invisible — an empty view segment, ß → ss) ; between them — identity with a shift.

    Memory: five ``array('q')`` arrays (8 bytes per number, no int objects),
    and neighbouring view characters of the same source width are one "run" ``w > 0`` (``\\u0417\\u0435\\u043b``
    → three characters of 6 each: ``start(i) = s0 + (i − v0)·w``), not a segment per escape: for an
    ``ensure_ascii`` file the map is O(words), not O(letters). ``w == 0`` — a block: the whole source segment
    under every view character (ß → ss, removed characters, a composed character)."""
    __slots__ = ("vlen", "slen", "_v0", "_v1", "_s0", "_s1", "_w")

    def __init__(self, vlen: int, slen: int, v0, v1, s0, s1, w=None):
        self.vlen, self.slen = vlen, slen
        self._v0, self._v1, self._s0, self._s1 = v0, v1, s0, s1
        self._w = w if w is not None else array("q", bytes(8 * len(v0)))

    def __len__(self) -> int:
        return self.vlen

    def start(self, i: int) -> int:
        j = bisect_right(self._v0, i) - 1
        if j < 0:
            return i
        v1 = self._v1[j]
        if i < v1:
            return self._s0[j] + (i - self._v0[j]) * self._w[j]
        return i + self._s1[j] - v1

    def end(self, i: int) -> int:
        j = bisect_right(self._v0, i) - 1
        if j < 0:
            return i + 1
        v1 = self._v1[j]
        if i < v1:
            w = self._w[j]
            return self._s0[j] + (i - self._v0[j] + 1) * w if w else self._s1[j]
        return i + 1 + self._s1[j] - v1

    def span(self, a: int, b: int) -> tuple[int, int]:
        if a >= b:
            p = self.start(a)
            return p, p
        return self.start(a), self.end(b - 1)

    def __getitem__(self, i: int) -> tuple[int, int]:
        if not 0 <= i < self.vlen:
            raise IndexError(i)
        return self.start(i), self.end(i)

    def segments(self) -> list:
        """Uneven segments [(v0, v1, s0, s1)] (a run — as one segment) — for tests and debugging."""
        return list(zip(self._v0, self._v1, self._s0, self._s1))

    def nbytes(self) -> int:
        """Map memory in bytes (arrays)."""
        return sum(a.buffer_info()[1] * a.itemsize
                   for a in (self._v0, self._v1, self._s0, self._s1, self._w)
                   if isinstance(a, array))

    def nseg(self) -> int:
        """Number of uneven segments (a report counter: how many places were folded)."""
        return len(self._v0)


def view_pos(pmap, s_pos: int) -> int:
    """View position for source position ``s_pos`` (the control gate stream): the smallest view
    position ``i`` with ``start(i) >= s_pos`` — the first view character lying entirely not before ``s_pos``.
    Inside a whitespace run or a composed character covering ``s_pos`` — the position AFTER it (a character
    started before the seam was already emitted by the previous chunk); for a removed invisible — the view position at its place.
    ``pmap`` None — identity."""
    if pmap is None:
        return s_pos
    j = bisect_right(pmap._s0, s_pos) - 1
    if j < 0:
        return s_pos
    s0, s1, v0, v1, w = pmap._s0[j], pmap._s1[j], pmap._v0[j], pmap._v1[j], pmap._w[j]
    if s_pos >= s1:
        return s_pos - s1 + v1
    if w:
        return v0 + (s_pos - s0 + w - 1) // w
    return v0 if s_pos == s0 else v1


class ChainMap:
    """Composition of maps: ``span(a, b)`` of the outer view → a source segment through ``outer``
    (the map of the outer view, fold_view) and then ``inner`` (the map below it, transport_view). Only
    ``span`` — what viewmask.EditLog.render and json_key_edits call."""
    __slots__ = ("outer", "inner")

    def __init__(self, outer, inner):
        self.outer, self.inner = outer, inner

    def span(self, a: int, b: int) -> tuple[int, int]:
        return self.inner.span(*self.outer.span(a, b))


def back_span(a: int, b: int, *maps) -> tuple[int, int]:
    """Source segment under ``[a, b)`` of the outermost view through a chain of maps: first — the map
    of the outer view (fold_text), last — the transport one (transport_view). ``None`` — identity."""
    for m in maps:
        if m is not None:
            a, b = m.span(a, b)
    return a, b


class _Builder:
    __slots__ = ("v0", "v1", "s0", "s1", "w")

    def __init__(self):
        self.v0, self.v1, self.s0, self.s1, self.w = (array("q") for _ in range(5))

    def add(self, va: int, vb: int, sa: int, sb: int, w=None) -> None:
        """View segment ``[va, vb)`` → source ``[sa, sb)``. ``w`` — source width per view character
        (a run); None — a segment of one view character is a run of one (``w = sb − sa``), others are
        a block (0). A run of the same width right after the last run extends it instead of duplicating."""
        if w is None:
            w = sb - sa if vb - va == 1 else 0
        if w and self.v0 and self.w[-1] == w and self.v1[-1] == va and self.s1[-1] == sa:
            self.v1[-1] = vb
            self.s1[-1] = sb
            return
        self.v0.append(va)
        self.v1.append(vb)
        self.s0.append(sa)
        self.s1.append(sb)
        self.w.append(w)

    def drop(self, v: int, sa: int, sb: int) -> None:
        """A removed source segment ``[sa, sb)`` before view position ``v`` (an empty view segment);
        a neighbouring removed segment at the same position is extended instead of duplicated."""
        if self.v0 and self.v0[-1] == v and self.v1[-1] == v and self.s1[-1] == sa:
            self.s1[-1] = sb
        else:
            self.add(v, v, sa, sb)

    def pop_from(self, s_from: int) -> None:
        """Drop trailing segments starting in the source not before ``s_from`` (segments of the base and
        of removed characters after it — before covering them with one segment of a composed character); a run
        covering ``s_from`` is shortened to it (a view character boundary)."""
        while self.v0 and self.s0[-1] >= s_from:
            for a in (self.v0, self.v1, self.s0, self.s1, self.w):
                a.pop()
        if self.v0 and self.s1[-1] > s_from:
            w = self.w[-1]
            k, r = divmod(s_from - self.s0[-1], w) if w else (0, 1)
            assert w and not r, (s_from, self.v0[-1], self.v1[-1], self.s0[-1], self.s1[-1], w)
            self.v1[-1] = self.v0[-1] + k
            self.s1[-1] = s_from

    def map(self, vlen: int, slen: int) -> PosMap:
        return PosMap(vlen, slen, self.v0, self.v1, self.s0, self.s1, self.w)


# --------------------------------------------------------------------------- transport level
_JSON_EXT = (".json", ".jsonl", ".ndjson")
_HTML_EXT = (".html", ".htm", ".xml")
# «\\» + any character (DOTALL): «\\\\» is eaten as a pair, and u0417 after it is letters, not an escape (like _JSON_ESC
# and _JSON_VIEW_RX of the mine view)
_JSON_ESC = re.compile(r"\\(?:u([0-9a-fA-F]{4})|(.))", re.S)
_JSON_LOW = re.compile(r"\\u([dD][c-fC-F][0-9a-fA-F]{2})")
_JSON_WS = frozenset("ntrbfv")
# a run of «\uXXXX» in a row (with ensure_ascii — a whole word) as one occurrence, another escape — «\» + a character
# (the same parity of «\» as _JSON_ESC: «\\\\» is eaten as a pair); a run is decoded as a chunk via UTF-16
# (a surrogate pair joins by itself), and only a run with a character that is not expanded in the view
# (_JSON_RUN_BAD: a control character, «"», «\», a line separator, a lone surrogate), or with a pair
# (the length does not equal the number of escapes) goes escape by escape, as before — memory O(words), not O(letters)
_JSON_RUN = re.compile(r"(?:\\u[0-9a-fA-F]{4})+|\\(.)", re.S)
_JSON_RUN_BAD = re.compile("[\x00-\x1f\"\\\\\x7f\x85  \ud800-\udfff]")
_HTML_ENT = re.compile(r"&(?:#(?:([0-9]{1,7})|[xX]([0-9a-fA-F]{1,6}))|(nbsp));")
_HTML_KEEP = frozenset("<>&\"'")
# Numeric entities (TRANSPORT_ENTITIES): a numeric entity (without &nbsp;) in .txt and — in one pass with
# JSON escapes (the same parity of «\», the same \uXXXX runs) — in .json/.jsonl/.ndjson; «\» does not appear
# in the view (&#92; stays an entity, like markup characters)
_ENT_EXT = (".txt",)
_ENT_NUM = re.compile(r"&#(?:([0-9]{1,7})|[xX]([0-9a-fA-F]{1,6}));")
_JSON_RUN_ENT = re.compile(r"(?:\\u[0-9a-fA-F]{4})+|\\(.)|&#(?:[0-9]{1,7}|[xX][0-9a-fA-F]{1,6});",
                           re.S)
_ENT_KEEP_PLAIN = _HTML_KEEP | {"\\"}


def _plain_cp(cp: int) -> bool:
    """A code that may be expanded in the view: not a control character, not a surrogate, within Unicode and not
    a line separator (NEL U+0085, U+2028, U+2029: ``str.splitlines`` treats them as line breaks —
    an expanded escape would shift the line numbering of residual and audit zones; a line break does not
    appear in the view, just like ``\\n``)."""
    return (0x20 <= cp <= 0x10FFFF and cp != 0x7F and not 0xD800 <= cp <= 0xDFFF
            and cp != 0x85 and cp != 0x2028 and cp != 0x2029)


def _ent_cp(m) -> int | None:
    """The code of a numeric entity from a _ENT_NUM/_JSON_RUN_ENT match (group(0) — «&#…;»); None — not
    expanded in the view (a control character, a surrogate, a line separator, a markup character or «\\»)."""
    s = m.group(0)
    cp = int(s[3:-1], 16) if s[2] in "xX" else int(s[2:-1])
    if not _plain_cp(cp) or chr(cp) in _ENT_KEEP_PLAIN:
        return None
    return cp


def _json_view(text: str, esc_ws: bool, entities: bool = False):
    out, mb, pos, vlen = [], _Builder(), 0, 0
    for m in (_JSON_RUN_ENT if entities else _JSON_RUN).finditer(text):
        a, end, ch = m.start(), m.end(), m.group(1)
        if ch is None and text[a] == "&":   # numeric entity — one character
            cp = _ent_cp(m)
            if cp is None:
                continue
            if a > pos:
                out.append(text[pos:a])
                vlen += a - pos
            out.append(chr(cp))
            mb.add(vlen, vlen + 1, a, end)
            vlen += 1
            pos = end
            continue
        if ch is None:                   # a \uXXXX run — as a chunk (see _JSON_RUN)
            n = (end - a) // 6
            dec = bytes.fromhex(m.group(0).replace("\\u", "")).decode("utf-16-be", "surrogatepass")
            if len(dec) == n and _JSON_RUN_BAD.search(dec) is None:
                if a > pos:
                    out.append(text[pos:a])
                    vlen += a - pos
                out.append(dec)
                mb.add(vlen, vlen + n, a, end, 6)
                vlen += n
                pos = end
                continue
            for m2 in _JSON_ESC.finditer(text, a, end):      # escape by escape, as before
                a2 = m2.start()
                if a2 < pos:             # the tail of a surrogate pair — already taken
                    continue
                cp, e2 = int(m2.group(1), 16), m2.end()
                if 0xD800 <= cp <= 0xDBFF:
                    m3 = _JSON_LOW.match(text, e2)
                    if m3 is None:
                        continue
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (int(m3.group(1), 16) - 0xDC00)
                    e2 = m3.end()
                elif not _plain_cp(cp) or cp == 0x22 or cp == 0x5C:
                    continue             # «"», «\», a control character, a lone surrogate — as is
                if a2 > pos:
                    out.append(text[pos:a2])
                    vlen += a2 - pos
                out.append(chr(cp))
                mb.add(vlen, vlen + 1, a2, e2)
                vlen += 1
                pos = e2
            continue
        if ch == "/":
            rep = "/"
        elif esc_ws and ch in _JSON_WS:
            rep = " "
        else:
            continue
        if a > pos:
            out.append(text[pos:a])
            vlen += a - pos
        out.append(rep)
        mb.add(vlen, vlen + 1, a, end)
        vlen += 1
        pos = end
    if pos == 0:
        return text, None
    out.append(text[pos:])
    view = "".join(out)
    return view, mb.map(len(view), len(text))


def _html_view(text: str):
    out, mb, pos, vlen = [], _Builder(), 0, 0
    for m in _HTML_ENT.finditer(text):
        dec, hexd, _nb = m.groups()
        if dec is not None:
            cp = int(dec)
        elif hexd is not None:
            cp = int(hexd, 16)
        else:
            cp = 0xA0
        if not _plain_cp(cp):
            continue
        rep = chr(cp)
        if rep in _HTML_KEEP:
            continue
        a = m.start()
        if a > pos:
            out.append(text[pos:a])
            vlen += a - pos
        out.append(rep)
        mb.add(vlen, vlen + 1, a, m.end())
        vlen += 1
        pos = m.end()
    if pos == 0:
        return text, None
    out.append(text[pos:])
    view = "".join(out)
    return view, mb.map(len(view), len(text))


def _ent_view(text: str):
    """The .txt view — only numeric entities (_ENT_NUM, _ent_cp), the map by them."""
    out, mb, pos, vlen = [], _Builder(), 0, 0
    for m in _ENT_NUM.finditer(text):
        cp = _ent_cp(m)
        if cp is None:
            continue
        a = m.start()
        if a > pos:
            out.append(text[pos:a])
            vlen += a - pos
        out.append(chr(cp))
        mb.add(vlen, vlen + 1, a, m.end())
        vlen += 1
        pos = m.end()
    if pos == 0:
        return text, None
    out.append(text[pos:])
    view = "".join(out)
    return view, mb.map(len(view), len(text))


def transport_view(rel: str | None, text: str, *, esc_ws: bool = False):
    """(view, spans): the in-file encoding is stripped by the extension of ``rel`` (see the module docstring). ``spans`` —
    ``PosMap`` or ``None`` (nothing had to be decoded: ``view is text``)."""
    ext = os.path.splitext(rel or "")[1].lower()
    if ext in _JSON_EXT:
        ent = "&#" in text                                  # numeric entities
        if "\\" not in text:
            return _json_view(text, esc_ws, True) if ent else (text, None)
        if not ent and not esc_ws and "\\u" not in text and "\\/" not in text:
            return text, None
        return _json_view(text, esc_ws, ent)
    if ext in _HTML_EXT:
        if "&#" not in text and "&nbsp;" not in text:
            return text, None
        return _html_view(text)
    if ext in _ENT_EXT and "&#" in text:
        return _ent_view(text)
    return text, None


# --------------------------------------------------------------------------- folding
# horizontal whitespace except the ordinary space: everything str.isspace() minus line breaks
# (\n \r U+0085 U+2028 U+2029) — what the pack pair class [^\S\n] takes (ANON_PACKGEN_WS_CLASS)
_HWS = "\t\x0b\x0c\x1c-\x1f\xa0  -   　"
_INVIS = "\xad​-‍⁠﻿"
_INVIS_SET = frozenset("\xad​‌‍⁠﻿")
_COMB = "̀-ͯ᪰-᫿᷀-᷿⃐-⃿︠-︯"
_DASH = "‐-―−"
_DASHES = "‐‑‒–—―−"      # as ru_trace._DASHES
_STRESS = frozenset("̀́")
_ASCII_WS = "\t\x0b\x0c\x1c-\x1f"
# edge punctuation of a key — as ru_trace._EDGE
EDGE = " \t\r\n\"'«»“”„()[]{}<>.,;:!?"

# one class — "the string has something to fold" (ё and dashes are 1:1, the rest changes the length)
_TRIG_YO = re.compile("[ёЁ" + _HWS + _INVIS + _COMB + _DASH + "]")
_TRIG = re.compile("[" + _HWS + _INVIS + _COMB + _DASH + "]")
_TRIG_ASCII = re.compile("[" + _ASCII_WS + "]")
_LEN_TRIG = re.compile("[" + _HWS + _INVIS + _COMB + "]")
_DASH_RX = re.compile("[" + _DASH + "]")
# characters changing the length: non-space whitespace, invisible, combining — in ONE class (an alternation
# of classes in sre is 3–6 times more expensive: every branch is tried at every position), parsing is per character in
# _fold_slow; a whitespace run is extended to the maximum there too. A pair of ordinary spaces is a separate
# literal regex (a class with a space would stumble on every space of the text).
_SEG_BASE = "[" + _HWS + _INVIS + _COMB + "]"
_SEG = re.compile(_SEG_BASE)
# fold_view: only invisibles and combining marks (whitespace — the pack, by class)
_SEG_VIEW = re.compile("[" + _INVIS + _COMB + "]")
_VIEW_TRIG = _SEG_VIEW
_SEG_CASE = None            # + characters whose casefold is longer (ß, ﬁ, İ) — lazily, one BMP scan
_DBL_SPACE = re.compile("  ")
_WS_RUN = frozenset(" \t\x0b\x0c\x1c\x1d\x1e\x1f\xa0   　"
                    + "".join(chr(c) for c in range(0x2000, 0x200B)))
# a whitespace run extends through invisibles («Ромашкин \u200b Трейд» — one space), a run of only
# invisibles is removed whole; an invisible and a removed stress mark do not break the NFC base
_RUN_SET = _WS_RUN | _INVIS_SET
# a Latin letter whose neighbour is Cyrillic (the prefix class [a-z] is a fast scan; not called on pure
# ASCII text: isascii)
_MIXED_CF = re.compile(r"[a-z](?:(?<=[Ѐ-ӿ][a-z])|(?=[Ѐ-ӿ]))")
_MIXED = re.compile(r"[A-Za-z](?:(?<=[Ѐ-ӿ][A-Za-z])|(?=[Ѐ-ӿ]))")
# the head of a JSON escape before a token: «\» (an escape letter follows) or «\u» with 0–3 hex characters
# (the other hex letters are an escape too, not the token); search(v, a-5, a) — \Z at position a
_ESC_HEAD = re.compile(r"\\(?:u[0-9a-fA-F]{0,3})?\Z")
_HEX_LETTERS = frozenset("abcdefABCDEF")

# twins: vocab.HOMOGLYPH (Cyrillic → Latin, lower case) both ways and in both cases —
# the same set as names._cyr_word and orgs._homoglyph_canon
_CYR2LAT = dict(HOMOGLYPH)
_CYR2LAT.update({c.upper(): l.upper() for c, l in HOMOGLYPH.items()})
_LAT2CYR = {l: c for c, l in _CYR2LAT.items()}
_CYR2LAT_TR = str.maketrans(_CYR2LAT)
_LAT2CYR_TR = str.maketrans(_LAT2CYR)


def _seg_rx_case():
    global _SEG_CASE
    if _SEG_CASE is None:
        exp = [chr(c) for c in range(0x10000) if len(chr(c).casefold()) != 1]
        _SEG_CASE = re.compile(_SEG_BASE + "|[" + "".join(re.escape(c) for c in exp) + "]")
    return _SEG_CASE


def _is_cyr(ch: str) -> bool:
    return "Ѐ" <= ch <= "ӿ"


_YO_BASE = {"ё": "е", "Ё": "Е"}


def _yo_base(ch: str, yo: bool) -> str:
    """The NFC base with ё→е: «ё» + breve is composed as «е» + breve → «ӗ» (otherwise a second pass over
    the view «е + breve» would compose further — folding would not be idempotent)."""
    return _YO_BASE.get(ch, ch) if yo else ch


def _compose(base: str, mark: str, case: bool):
    comp = unicodedata.normalize("NFC", base + mark)
    if len(comp) != 1:
        return None
    if case and len(comp.casefold()) != 1:
        return None
    return comp


# an ambiguous token of the pack Latin rule form: a Latin head + a Cyrillic tail («comе» in
# «example.comе», «xyом», «xyОМ», «Coк»); «Сoк»/«Кoт» (a Cyrillic head) is not this form
_LAT_HEAD = re.compile(r"[A-Za-z]+[Ѐ-ӿ]+\Z")


def _twin_fold(tok: str, both: bool = True, raw_ambiguous: bool = False) -> tuple:
    """→ (replacement, ambiguous with a Latin head, a Latin word with Cyrillic twins).
    An ambiguous token — ALL letters of both scripts are twins; the flag is True only for the form of the pack Latin
    rule (_LAT_HEAD). The third — a token like «Romаshka» (folded only with ``both``;
    with both=False — a sign that the second engine stage will fold it)."""
    has_lat = has_cyr = False
    lat_twin = cyr_twin = True
    for ch in tok:
        if "a" <= ch <= "z" or "A" <= ch <= "Z":
            has_lat = True
            if ch not in _LAT2CYR:
                lat_twin = False
        elif "Ѐ" <= ch <= "ӿ":
            has_cyr = True
            if ch not in _CYR2LAT:
                cyr_twin = False
        else:
            return tok, False, False     # a letter of a third script — leave it
    if not (has_lat and has_cyr):
        return tok, False, False
    if lat_twin:                         # a Cyrillic word with Latin twins (and the disputed
        amb = cyr_twin and _LAT_HEAD.match(tok) is not None    # case «Coк»: the corpus is Russian —
        if amb and raw_ambiguous:                               # Cyrillic)
            # the raw-view stage (pipeline._masked, HOMOGLYPH_AMBIGUOUS_STAGE2): an ambiguous
            # token with a Latin head, not masked after folding — as written, so that
            # the pack Latin rule with the Cyrillic tail «[а-яё]*» takes it, as before folding
            return tok, True, False
        return tok.translate(_LAT2CYR_TR), amb, False
    if cyr_twin:                         # a Latin word with Cyrillic twins («Romаshka»)
        if both:
            return tok.translate(_CYR2LAT_TR), False, False
        return tok, False, True
    return tok, False, False


def _fold_mixed(v: str, rx, count=None, both: bool = True, raw_ambiguous: bool = False,
                rev=None) -> str:
    """Mixed-script tokens → one script (1:1). ``rx`` — _MIXED_CF or _MIXED.
    ``count`` — a list: a triple is added to it (folded tokens — a report counter; of them
    a Latin word → Latin; ambiguous tokens with a Latin head — folded or left
    raw). ``both`` False — only a Cyrillic word with Latin twins (owner's
    decision), a Latin word with Cyrillic twins («Romаshka», «Petrоva») stays
    mixed — it is the structural trigger of names.translit_leet_name (the verify canary of the
    homoglyph class), the engine folds it in the second stage (pipeline._masked). ``raw_ambiguous``
    (HOMOGLYPH_AMBIGUOUS_STAGE2) — an ambiguous token with a Latin head
    (ALL letters of both scripts are twins, the form of the pack Latin rule: «comе», «xyом», «Coк»; not
    «Сoк») stays as written — the raw-view stage after folding, where the pack Latin rule with
    a Cyrillic tail takes it, as before folding («example.comе», «xyом» — merge leaks).
    JSON escape letters before a token — ``\\t`` in «\\tПаспорт», trailing hex letters of ``\\u000b`` —
    are not part of the token (_ESC_HEAD): otherwise the escape in the view broke («\\тПаспорт»), and a rule with the
    boundary._ESC boundary after it stayed silent. ``rev`` (HOMOGLYPH_LEAN) — a list: with
    ``both=False`` the number of Latin words with Cyrillic twins is added to it
    («Romаshka») that the second stage (``both=True``) would fold — the engine decides whether it is needed
    without a second pass over the view."""
    out, pos, n, last, n_rev, n_amb, n_rev2 = [], 0, len(v), 0, 0, 0, 0
    for m in rx.finditer(v):
        i = m.start()
        if i < last:
            continue
        a = i
        while a > 0 and v[a - 1].isalpha():
            a -= 1
        b = i + 1
        while b < n and v[b].isalpha():
            b += 1
        last = b
        e = _ESC_HEAD.search(v, max(0, a - 5), a)
        if e is not None:
            if e.group(0) == "\\":
                a += 1                                    # escape letter: \t \n \r \b \f \v \u
            else:
                k = 4 - (len(e.group(0)) - 2)             # missing hex characters of \uXXXX
                while k > 0 and a < b and v[a] in _HEX_LETTERS:
                    a += 1
                    k -= 1
            if a >= b:
                continue
        tok = v[a:b]
        rep, amb, rv = _twin_fold(tok, both, raw_ambiguous)
        if amb:
            n_amb += 1
        if rv:
            n_rev2 += 1
        if rep != tok:
            out.append(v[pos:a])
            out.append(rep)
            pos = b
            if rep.isascii():
                n_rev += 1
    if count is not None and (out or n_amb):
        # (all folded, of them Latin word → Latin, ambiguous with a Latin head)
        count.append((len(out) // 2, n_rev, n_amb))
    if rev is not None and n_rev2:
        rev.append(n_rev2)
    if not out:
        return v
    out.append(v[pos:])
    return "".join(out)


def homoglyph_view(s: str, count=None, *, both: bool = True, raw_ambiguous: bool = False,
                   rev=None) -> str:
    """The homoglyph view for the engine (owner's decision): mixed-script tokens where all letters of one
    of the scripts are twins (vocab.HOMOGLYPH, both cases) are brought to the other script; case and length
    are the same (no map needed), when ambiguous — Cyrillic («Ивaнову» → «Иванову», «Acmе» →
    «Acme»; «Windows-сервер», «iPhone», «Яan» are not touched: «n», «i», «l» are not twins). The view
    is only for rules: an unmasked token («cчет» → «счет» in the view) stays in the output as in the
    source. Without mixing — the same string (``view is s``: the fast path). ``count`` — a list for
    the number of folded tokens. ``both`` False — only Latin twins in a Cyrillic word
    (the first engine stage: a Latin word with Cyrillic twins stays mixed — the trigger of
    translit_leet_name), see _fold_mixed. ``raw_ambiguous`` — an ambiguous token with a Latin head
    (all letters of both scripts are twins, the form of the pack Latin rule: «comе», «xyом», «Coк») stays
    as written — the raw-view stage in pipeline._masked (HOMOGLYPH_AMBIGUOUS_STAGE2);
    without it it is folded into Cyrillic. ``count`` — triples, the third
    number — such tokens (folded or raw): by it the engine decides whether that stage is needed.
    ``rev`` — a list for the number of Latin words with Cyrillic twins with ``both=False``
    (see _fold_mixed)."""
    if s.isascii() or _MIXED.search(s) is None:
        return s
    return _fold_mixed(s, _MIXED, count, both, raw_ambiguous, rev)


def _one_to_one(v: str, case: bool, yo: bool, homoglyph: bool, dash: bool = True) -> str:
    """1:1 folds (the map does not change): ё→е, dashes → «-» (``dash``), twins."""
    if yo:
        if "ё" in v:
            v = v.replace("ё", "е")
        if not case and "Ё" in v:
            v = v.replace("Ё", "Е")
    if dash and _DASH_RX.search(v) is not None:
        for d in _DASHES:
            if d in v:
                v = v.replace(d, "-")
    if homoglyph and not v.isascii():
        rx = _MIXED_CF if case else _MIXED
        if rx.search(v) is not None:
            v = _fold_mixed(v, rx)
    return v


def _fold_slow(s: str, v, case: bool, yo: bool, homoglyph: bool, ws: bool = True,
               dash: bool = True):
    """Slow path: the view by segments between _SEG occurrences, the map — only by them. ``v`` —
    casefold(s) of the same length (view pieces are taken from it, no second casefold needed) or None
    (case=False, or casefold changes the length — then casefold at the end, expanding characters are expanded
    segment by segment). ``ws`` False — whitespace is not folded (fold_view: a run — only
    invisibles); ``dash`` False — dashes are not folded.

    Invisibles are transparent: a whitespace run extends through them, and a combining mark
    after an invisible attaches to the letter before it; a removed stress mark is transparent too («и» + stress +
    breve → «й»). The NFC base is the last view character (``base_ch``, in the original case) and the start of its
    segment in the source (``base_src``); a composed character itself becomes the base («a» + diaeresis + macron →
    «ǟ»), its segment covers the base, the removed characters between them and the character itself. A space, a character
    expanded by casefold into several (ß → ss), and an unattached mark are never a base."""
    src = v if v is not None else s
    if not ws:
        rx = _SEG_VIEW
        run_set = _INVIS_SET
    else:
        rx = _seg_rx_case() if (case and v is None) else _SEG
        run_set = _RUN_SET
    n = len(s)
    out, mb, pos, vlen = [], _Builder(), 0, 0
    base_ch, base_src = "", -1
    hits = ((m.start(), m.end()) for m in rx.finditer(s))
    if ws and "  " in s:
        hits = heapq.merge(hits, ((m.start(), m.end()) for m in _DBL_SPACE.finditer(s)))
    for a, b in hits:
        if a < pos:                                        # inside an already taken run
            continue
        ch = s[a]
        eff_base = _yo_base(s[a - 1], yo) if a > pos else base_ch
        if ch in run_set or (ch in _STRESS and eff_base in ("", " ")):
            # a run of whitespace and invisibles; a stress mark without a base letter (at the start, after a space, a run or
            # an unattached mark) is removed and does not break the run — otherwise NBSP + stress + NBSP would give
            # two spaces, and a second pass would fold further
            has_ws = ch in _WS_RUN
            while a > pos and s[a - 1] in run_set:
                a -= 1
                has_ws = has_ws or s[a] in _WS_RUN
            no_base = eff_base in ("", " ")
            while b < n:                                   # forward: whitespace, invisibles and a stress mark
                c = s[b]                                   # if it has no base (a run with a space or
                if c in run_set:                           # without a letter before it); «o» + ZWJ + stress
                    has_ws = has_ws or c in _WS_RUN        # — the stress stays with the letter: «ó»
                elif c in _STRESS and (has_ws or no_base):
                    pass
                else:
                    break
                b += 1
            if a > pos:
                out.append(src[pos:a])
                vlen += a - pos
                base_ch, base_src = _yo_base(s[a - 1], yo), a - 1
            if has_ws:                                     # → one space
                out.append(" ")
                mb.add(vlen, vlen + 1, a, b)
                vlen += 1
                base_ch, base_src = "", -1
            else:                                          # only invisibles — remove, the base stays
                mb.drop(vlen, a, b)
            pos = b
            continue
        if a > pos:
            out.append(src[pos:a])
            vlen += a - pos
            base_ch, base_src = _yo_base(s[a - 1], yo), a - 1
        if unicodedata.category(ch)[0] == "M":             # a combining mark
            if base_ch and ch in _STRESS and _is_cyr(base_ch):
                mb.drop(vlen, a, b)                        # stress over Cyrillic — remove
            else:
                comp = _compose(base_ch, ch, case) if base_ch else None
                if comp is not None:                       # NFC: letter + mark → a composed character
                    out[-1] = out[-1][:-1]
                    out.append(comp.casefold() if case else comp)
                    mb.pop_from(base_src)
                    mb.add(vlen - 1, vlen, base_src, b)
                    base_ch = _yo_base(comp, yo)
                elif ch in _STRESS:                        # stress without a composed form — remove
                    mb.drop(vlen, a, b)
                else:                                      # another mark — as is, 1:1
                    out.append(ch)
                    vlen += 1
                    base_ch, base_src = "", -1
        else:                                              # casefold is longer than the character (ß → ss)
            cf = ch.casefold()
            out.append(cf)
            mb.add(vlen, vlen + len(cf), a, b)
            vlen += len(cf)
            base_ch, base_src = "", -1
        pos = b
    out.append(src[pos:])
    view = "".join(out)
    if case and v is None:
        view = view.casefold()                             # 1:1: expanding characters are already expanded
    view = _one_to_one(view, case, yo, homoglyph, dash)
    return view, mb.map(len(view), n)


def fold_view(s: str):
    """(view, back) — the view layer for the engine: only what neither the pack
    nor the homoglyph layer strips — invisibles (_INVIS_SET: soft hyphen, ZWSP/ZWNJ/ZWJ, U+2060, BOM)
    removed, a letter + combining mark composed (NFC, as in fold_text), stress U+0300/U+0301 over
    Cyrillic removed. Case, ё, whitespace, dashes, twins and line breaks — as in the source
    (the pack sees them via the classes [её]/[^\\S\\n]/dashes and homoglyph_view). ``back`` — PosMap or None
    (the fast path: neither invisibles nor combining marks — ``view is s``)."""
    if _VIEW_TRIG.search(s) is None:
        return s, None
    return _fold_slow(s, None, False, False, False, ws=False, dash=False)


def fold_text(s: str, *, case: bool, yo: bool | None = None, homoglyph: bool = True):
    """(folded, back): folding with a position map (see the module docstring). ``case`` — casefold; ``yo`` — ё→е (None
    — by the ANON_YO_FOLD switch); ``homoglyph`` — Latin↔Cyrillic twins. ``back`` — PosMap or
    None (positions 1:1 with ``s``)."""
    if yo is None:
        yo = yo_fold_on()
    v = s.casefold() if case else s
    if len(v) != len(s):                                   # ß/ﬁ/İ — a map is needed
        return _fold_slow(s, None, case, yo, homoglyph)
    cf = v if case else None
    if v.isascii():                                        # no Cyrillic, no ё, no dashes, no NBSP
        if _TRIG_ASCII.search(v) is not None or "  " in v:
            return _fold_slow(s, cf, case, yo, homoglyph)
        return v, None
    trig = (_TRIG_YO if yo else _TRIG).search(v)
    if trig is not None and _LEN_TRIG.search(v) is not None:
        return _fold_slow(s, cf, case, yo, homoglyph)
    if "  " in v:
        return _fold_slow(s, cf, case, yo, homoglyph)
    if trig is not None:                                   # only ё and/or dashes — 1:1
        v = _one_to_one(v, case, yo, homoglyph)
    elif homoglyph:
        rx = _MIXED_CF if case else _MIXED
        if rx.search(v) is not None:
            v = _fold_mixed(v, rx)
    return v, None


def value_key(s: str, *, case: bool = True, yo: bool = True) -> str:
    """Value comparison key: fold_text (ё→е, twins, spaces, dashes, NFC, invisibles)
    and trimming of the EDGE edge punctuation — as ru_trace.norm. ``case=False`` — for mine and control;
    ``yo=False`` — ё kept apart (the fix-layer veto comparators; NBSP/NFD/twins are still folded)."""
    return fold_text(str(s), case=case, yo=yo)[0].strip(EDGE)


def canon_literal(s: str) -> str:
    """Spelling for storage: NFC, whitespace → space, dashes → «-», invisibles and
    stress marks removed, twins folded («Фeдоров» with a Latin e → Cyrillic); case and ё
    kept. Edge punctuation is NOT trimmed (unlike value_key: «ООО «Ромашка»» with strip(EDGE)
    would lose the paired quote) — only edge whitespace. A pack literal, a mine candidate and the first
    spelling in candidates.tsv — in this form; the comparison key for it is value_key."""
    return fold_text(str(s), case=False, yo=False)[0].strip()
