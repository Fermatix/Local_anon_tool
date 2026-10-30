"""View → rules → substitution into the source.

Engine rules see not the raw file text but its "view": the transport view (``\\uXXXX``/``\\/`` in
JSON, ``&#NNNN;``/``&nbsp;`` in HTML decoded — vnorm.transport_view with a PosMap) and on top of it
the homoglyph layer (vnorm.homoglyph_view, 1:1 in length, no map needed). The rule fixpoint and
fusion splitting run on the view exactly as they run on text; every replacement is written to a log
(``EditLog``) in coordinates of the ORIGINAL view, and at the end one pass puts the ``out`` of every replacement
onto the source span under its preimage (``PosMap.span``), while everything between replacements is taken from the source
byte for byte. So the escaped form of the file (``ensure_ascii`` form, ``\\/``) and unmasked
mixed-script tokens («cчет») do not change in the output, and what was masked is covered by a token.

Log: spans ``(o0, o1, out)`` in coordinates of the original view, sorted, non-overlapping
(an empty preimage is an insertion of "-" by fusion splitting; an empty ``out`` is a deletion). A batch of replacements of one
rule comes in coordinates of the text version BEFORE that rule (as ``pattern.sub`` and
``_local_matches`` return them: ascending, non-overlapping); the preimage of a new replacement is found from arrays
of current positions of earlier replacements (recomputed once per batch — O(k), k = number of replacements in the file). A new
replacement that fully covers earlier ones (glue rules) merges them into one with the new ``out``; a boundary of a new
replacement INSIDE an earlier ``out`` (a rule ate half a token — «ORG_89…» + «-2024» as a phone)
is represented exactly: the preimage widens to the earlier replacement, and the untouched edge of its ``out``
remains — the output is the same as on the old text path, so there is no rollback to raw text (the log
represents any sequence of replacements). Invariants (oracles in tests/test_engine_homoglyph.py):
I1 — outside preimages the output equals the source byte for byte; I2 — the view of the output equals the output of the view; I3 —
a repeated dry run on the output is empty; I4 — switches =0 give the old output; I5 — stream = memory.

View rounds: the "-" inserted by fusion splitting changes token boundaries («PомaшкаORG_7» did not fold —
«ORG» has no homoglyphs; «Pомaшка-ORG_7» folds into «Ромашка» after splitting and is masked by the pack),
so the view built before the rules may diverge from the view of the output. ``apply_view`` after a round
builds the view of the output again (``refold``) and, if it differs from the output of the view, repeats the round on it
(capped at _MAX_ROUNDS, like fusion splitting): I2 and I3 hold by construction, not by luck.
"""
from __future__ import annotations

import re

__all__ = ["EditLog", "apply_view", "json_key_edits"]

_MAX_ROUNDS = 3
# rest of a JSON string from the end of the preimage to the closing quote and ":" — the replacement is in an object KEY
_KEY_TAIL = re.compile(r'(?:[^"\\]|\\.)*"\s*:', re.S)
_KEY_SCAN = 2048


class EditLog:
    """Log of replacements in coordinates of the original view (see the module header)."""
    __slots__ = ("o0", "o1", "out", "_pos")

    def __init__(self):
        self.o0: list = []
        self.o1: list = []
        self.out: list = []
        self._pos = None             # (c0, c1, cum) — current positions of earlier replacements, lazily

    def __len__(self) -> int:
        return len(self.o0)

    def _positions(self):
        if self._pos is None:
            c0, c1, cum, d = [], [], [], 0
            for a, b, o in zip(self.o0, self.o1, self.out):
                c0.append(a + d)
                d += len(o) - (b - a)
                c1.append(b + d)
                cum.append(d)
            self._pos = (c0, c1, cum)
        return self._pos

    def subs(self, batch) -> None:
        """A batch of replacements ``[(a, b, out)]`` in coordinates of the CURRENT text version: ascending,
        non-overlapping; ``a == b`` is an insertion. Moved into the log in one merge pass with the old
        spans: an old span before ``a`` is rewritten as is, one fully covered by ``[a, b)`` is
        merged, one cut by a boundary gives the covered part to the new replacement, and the rest of ``out``
        stays as an insertion in its place (``carry``)."""
        if not batch:
            return
        c0s, c1s, cums = self._positions()
        O0, O1, OUT = self.o0, self.o1, self.out
        n = len(O0)
        r0, r1, rout = [], [], []
        i = 0
        carry = None                 # (src, text, ca, cb): rest of a cut out — insertion at src

        def cur():
            if carry is not None:
                return carry[0], carry[0], carry[1], carry[2], carry[3]
            if i < n:
                return O0[i], O1[i], OUT[i], c0s[i], c1s[i]
            return None

        def delta():                 # shift of untouched text after the last taken span
            return cums[i - 1] if i > 0 else 0

        for a, b, rep in batch:
            # 1. old spans fully before a — as is
            while True:
                e = cur()
                if e is None or e[4] > a:
                    break
                r0.append(e[0])
                r1.append(e[1])
                rout.append(e[2])
                if carry is not None:
                    carry = None
                else:
                    i += 1
            # 2. left boundary
            e = cur()
            if e is not None and e[3] < a:          # a strictly inside an old out — cut
                sa, pre = e[0], e[2][: a - e[3]]
                rest = (e[1], e[2][a - e[3]:], a, e[4])
                if carry is None:
                    i += 1
                carry = rest
            elif e is not None and e[3] == a:
                sa, pre = e[0], ""
            else:
                sa, pre = a - delta(), ""
            # 3. fully covered (c1 <= b) — merged
            while True:
                e = cur()
                if e is None or e[4] > b:
                    break
                if carry is not None:
                    carry = None
                else:
                    i += 1
            e = cur()
            if e is not None and e[3] < b:          # b strictly inside an old out — cut
                sb, suf = e[1], ""
                rest = (e[1], e[2][b - e[3]:], b, e[4])
                if carry is None:
                    i += 1
                carry = rest
            elif e is not None and e[3] == b:
                sb, suf = e[0], ""
            else:
                sb, suf = b - delta(), ""
            r0.append(sa)
            r1.append(sb)
            rout.append(pre + rep + suf)
        while True:                                  # tail of old spans
            e = cur()
            if e is None:
                break
            r0.append(e[0])
            r1.append(e[1])
            rout.append(e[2])
            if carry is not None:
                carry = None
            else:
                i += 1
        self.o0, self.o1, self.out = r0, r1, rout
        self._pos = None

    def inserts(self, positions, text: str) -> None:
        """Insertions of ``text`` at positions of the current version (fusion splitting: "-")."""
        if positions:
            self.subs([(p, p, text) for p in positions])

    def render(self, src: str, pmap=None) -> str:
        """Substitution: the ``out`` of every replacement goes onto the source span under its preimage
        (``pmap.span``; ``None`` — the view and the source match by position), between replacements —
        the source byte for byte."""
        parts, pos = [], 0
        for a, b, out in zip(self.o0, self.o1, self.out):
            sa, sb = (a, b) if pmap is None else pmap.span(a, b)
            parts.append(src[pos:sa])
            parts.append(out)
            pos = sb
        parts.append(src[pos:])
        return "".join(parts)


def json_key_edits(src: str, log: EditLog, pmap=None) -> int:
    """Number of log replacements whose preimage lies in a JSON object KEY (after the end of the preimage —
    the rest of the string, the closing quote and ":"). A decoded full-name key is masked the same way as in
    a UTF-8 file — a counter for WARN, not a ban. The rest of the string
    is looked at no further than _KEY_SCAN characters (a .json file without quotes is not O(replacements × file))."""
    n = 0
    for a, b in zip(log.o0, log.o1):
        _sa, sb = (a, b) if pmap is None else pmap.span(a, b)
        if _KEY_TAIL.match(src, sb, sb + _KEY_SCAN):
            n += 1
    return n


def apply_view(text: str, view: str, pmap, run, refold=None, audit=None, on_cap=None):
    """``run(view, edits) -> new_view`` — a rule fixpoint on the view that appends to ``edits``
    batches ``("sub", [(a, b, out)])`` (apply_rules) and ``("ins", [positions])`` (split_fusions) in
    coordinates of the view version before each batch. ``refold(text) -> (view, pmap)`` — build the view
    of a text again (the same way as ``view``): the round repeats on the output until the view of the output
    equals the output of the view (see the module header); None — one round. ``audit(log, text, pmap)`` is called with
    the round log before substitution (counting replacements in JSON keys). Returns ``(new_text, new_view,
    n_edits)``: the output in the original form, the output of the view (stream context tail, I2) and the number of log
    spans over all rounds. ``on_cap(rounds)`` is called when
    the round cap is exhausted and the view of the output still differs from the output of the view: I2/I3 are not
    proven on this file (the returned new_view is the output of the view of the previous round, not the view of the returned text);
    without the callback the outcome was indistinguishable from a clean one — pipeline counts it in fixpoint_capped and WARNs."""
    total = 0
    for _ in range(_MAX_ROUNDS):
        edits: list = []
        new_view = run(view, edits)
        log = EditLog()
        for kind, batch in edits:
            if kind == "sub":
                log.subs(batch)
            else:
                log.inserts(batch, "-")
        if not log.o0:
            return text, new_view, total
        if log.render(view) != new_view:
            # fail loudly: the log does not reproduce the output of the view (a rule or a substitute apply_rules
            # changes the text bypassing edits) — the source would be silently corrupted (fusion-splitting insertions
            # in the wrong places)
            raise SystemExit("viewmask: the replacement log does not reproduce the output of the view — text replaced "
                             "bypassing apply_rules(edits=…)/split_fusions(positions=…)")
        if audit is not None:
            audit(log, text, pmap)
        text = log.render(text, pmap)
        total += len(log.o0)
        if refold is None:
            return text, new_view, total
        pmap = log = None              # the round map and log are not needed while the output view is built
        view, pmap = refold(text)
        if view == new_view:
            return text, new_view, total
    if on_cap is not None:
        on_cap(_MAX_ROUNDS)
    return text, new_view, total
