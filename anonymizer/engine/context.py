"""Context-window markers.

Weak / common values (a bare 6-digit ZIP, a 20-digit account, a decimal coord,
a street name, a common first name) must only fire when a matching label sits
nearby — otherwise they shred ordinary text. Each detector declares
``requires_context = <one of these regexes>`` and the engine checks a left
window (default 120 chars) before allowing the match.
"""
from __future__ import annotations

import re

from .boundary import _ESC

try:                                   # 3.11+: sre_parse moved to re._parser
    from re import _parser as _sre
except ImportError:                    # pragma: no cover
    import sre_parse as _sre           # type: ignore

# THE LEFT EDGE OF THE WINDOW DOES NOT CUT THE TEXT. The gate used to look for ctx in the SLICE
# text[start-window:start]: a lookbehind of the ctx regex at the left edge of the slice did not
# see what stood before the window. A window starting inside «PERSON_NAME_…» gave
# «NAME_…» without «person_» on the left — (?<!person_)name of NAME_CTX fired on a
# fragment of a token; a window starting inside «методом »/«кубик»/«platform» gave
# «дом »/«бик»/«lat» without the letter on the left — the _B boundary passed. But the rule's needs
# (the same ctx over the whole text) does not see the fragment: the gate was not a necessary
# condition, the prefilter let the rule through, VERIFY failed, and in production masking
# a surname depended on a context word thousands of characters away (it turns needs on —
# and a token fragment in the window served as context: an extra mask).
# Now the window is the same (the match starts in [start-window, start) and ends
# no further right than start: $ and lookahead see the end of the window, as before), but the search runs
# in the FULL text with pos/endpos: lookbehind and \b at the left edge see the text
# before the window. ^ at the left edge of the window no longer fires (the window edge is not
# a line start; there are no ctx regexes with ^ now). The stream draws the left window
# with the tail of previous lines — the tail is longer than the window by ctx_reach (pipeline._ctx_width).
# The right edge of the right window (ctx_right) works the same way: the slice «счет» of
# «счетчик» passed сч[её]т(?!чик); now lookahead sees the line beyond the
# window (right_hit). The rollback switch CTX_EDGE_LOOKBEHIND=0 (old slices) is removed.

# A short stem without a left boundary is caught INSIDE another
# word — «инн» in «тиннитус», «бик» in «кубик», «счет» in «подсчет», «моб» in
# «автомобиль», «дом » in «методом », «lat» in «platform», «ключ» in «включить».
# _B is the left "not a letter" boundary: the stem must START a word. There is no right boundary
# on purpose: stems catch inflections (счёта/счётом, ключами, контакты). Stems that
# legitimately stand inside a word do NOT get the boundary: банк/bank (Сбербанк,
# Sberbank), город (Новгород, пригород), звон (позвонить), локац (геолокац),
# auth (oauth), birth (childbirth). The boundary is JSONL-aware, like boundary.L:
# the letter of an escape ``\n`` / ``\t`` / ``\uXXXX`` left of a label in raw .jsonl
# («…\nИНН 7707083893») is a line end, not a neighbouring word.
_B = r"(?:" + _ESC + r"|(?<![а-яёa-z]))"

VAULT_CTX = re.compile(
    r"(?i)(парол|password|passwd|pwd|passcode|секрет|secret|токен|token|"
    r"api[_-]?key|access[_-]?key|client[_-]?secret|" + _B + r"ключ|доступ|"
    r"логин|login|credential|auth|bearer)"
)
BANK_CTX = re.compile(
    # сч[её]т(?!чик): «счетчик/счётчик» is not «счёт» (the only exception on the
    # right edge; the inflections счёта/счётом/счетов are alive)
    r"(?i)(" + _B + r"(?:бик|bik|bic|инн|кпп|огрн|okpo|окпо|iban|сч[её]т(?!чик)|"
    r"кор(?:р|сч))|swift|account|р/с|к/с|банк|bank|реквизит)"
)
ADDRESS_CTX = re.compile(
    r"(?i)(адрес|прожива|регистрац|индекс|город|улиц|проспект|шоссе|переулок|"
    r"набережн|микрорайон|кварт|" + _B + r"(?:ул\.|наб\.|мкр|дом |корп|строен))"
)
GEO_CTX = re.compile(
    r"(?i)(координат|широт|долгот|latitude|longitude|адрес|местополож|локац|"
    + _B + r"(?:lat|lon|geo|maps|point|gps))"
)
BIRTH_CTX = re.compile(
    # д.р. / г.р. need the abbreviation dot + letter boundaries, otherwise the
    # branch degrades to a bare 'др'/'гр' bigram inside ordinary words
    # (другой, адрес, группа) and the DOB gate fires on almost any text.
    r"(?i)(дата\s*рожд|(?<![а-яёa-z])д\.\s*р\.?(?![а-яёa-z])|"
    r"(?<![а-яёa-z])г\.\s*р\.?(?![а-яёa-z])|год\s*рожд|родил|birth|"
    + _B + r"born|дата\s*выдач|выдан)"
)
PASSPORT_CTX = re.compile(r"(?i)(паспорт|passport|серия|номер|код\s*подраздел)")
PHONE_CTX = re.compile(
    # «тел» is an abbreviation without inflections: only as a whole word (тел. / тел:),
    # «телеф» covers the inflections; «конт» → «контакт» (контент/контекст/
    # контракт are not a phone context)
    r"(?i)(" + _B + r"(?:тел(?![а-яёa-z])|телеф|моб)|phone|контакт|звон|"
    r"whatsapp|telegram|номер)"
)

_EOL = re.compile(r"[\r\n]")


def right_segment(text: str, end: int, right: int) -> str:
    """Slice [end, end+right) WITHIN the line (up to the first \\r or \\n) —
    the right context window. The stream branch (pipeline._run_stream) sees only
    its own line, so the in-memory path must also look exactly to the end
    of the line — otherwise the stream==memory byte oracle breaks."""
    seg = text[end:end + right]
    cut = _EOL.search(seg)
    return seg[:cut.start()] if cut else seg


def has_context(text: str, start: int, ctx: re.Pattern, window: int = 120,
                right: int = 0, end: int | None = None) -> bool:
    """True if `ctx` appears in the `window` chars immediately left of `start`.

    ``right`` > 0 — additionally searches the right window ``[end, end+right)``
    (``end`` — end of the value; default ``start``) for label-anchored
    records «7707083893 (ИНН)», «+7 999 … (тел.)». The right window is limited
    by the line (see right_segment); the left one, as before, is not limited by the line —
    the stream draws it with the tail of previous lines (rule.apply_rules).
    Both windows are searched in the full text (see left_hit, right_hit)."""
    if ctx.search(text, max(0, start - window), start):
        return True
    if right <= 0:
        return False
    return right_hit(text, start if end is None else end, ctx, right) is not None


def left_hit(text: str, start: int, ctx: re.Pattern, window: int):
    """First match of ``ctx`` in the left window ``[start-window, start)``:
    (word, offset of the word start from ``start`` — negative) or None.
    Search in the full text (pos/endpos: lookbehind sees the text before the window)."""
    a = max(0, start - window)
    mm = ctx.search(text, a, start)
    return None if mm is None else (mm.group(0), mm.start() - start)


def right_hit(text: str, end: int, ctx: re.Pattern, right: int):
    """First match of ``ctx`` in the right window ``[end, end+right)`` within
    the line: (word, offset of the word start from ``end``) or None.
    Search in the full text: the match, as
    in a slice, lies fully in the window, but lookahead, \\b and $ at the right edge of the window
    see the text of the line beyond the window (ctx_ahead characters), lookbehind at the left edge
    sees the value. A slice gave «счет» of «счетчик» — сч[её]т(?!чик) of BANK_CTX
    fired on the fragment."""
    lim = end + right
    stop = min(len(text), lim + ctx_ahead(ctx))
    eol = _EOL.search(text, end, stop)
    if eol is not None:
        stop = eol.start()
    if stop <= lim:                    # the window hits the end of the line — as a slice
        mm = ctx.search(text, end, stop)
    else:                              # the match ends no further right than lim
        mm = _within(ctx, stop - lim).search(text, end, stop)
    return None if mm is None else (mm.group(0), mm.start() - end)


_WITHIN: dict = {}      # (pattern, flags, k) -> ctx + "k characters before endpos"
_FLAGS_HEAD = re.compile(r"\A(?:\(\?[aiLmsux]+\))+")


def _within(ctx: re.Pattern, k: int) -> re.Pattern:
    """``ctx`` whose match ends at least ``k`` characters before
    endpos: with endpos = window edge + k the match lies in the window, and its
    assertions read k characters beyond the edge. Branches are tried the same way as in a slice
    (a branch going beyond the edge backtracks to the next one). Global flags
    stand at the start of the pattern — they are moved into flags. Per-process cache."""
    key = (ctx.pattern, ctx.flags, k)
    rx = _WITHIN.get(key)
    if rx is None:
        body = _FLAGS_HEAD.sub("", ctx.pattern, count=1)
        nl = "\n" if ctx.flags & re.VERBOSE else ""     # a comment at the end of the pattern
        rx = _WITHIN[key] = re.compile(
            "(?:" + body + nl + ")(?=[\\s\\S]{%d})" % k, ctx.flags)
    return rx


_REACH: dict = {}       # (pattern, flags, back) -> ctx_reach / ctx_ahead
# ^ and \A without MULTILINE also read "to the left": they do not fire at pos > 0, and a stream
# tail equal to the window would start the window at position 0
_AT_BACK = {getattr(_sre, n) for n in (
    "AT_BOUNDARY", "AT_NON_BOUNDARY", "AT_BEGINNING", "AT_BEGINNING_LINE",
    "AT_BEGINNING_STRING", "AT_LOC_BOUNDARY", "AT_LOC_NON_BOUNDARY",
    "AT_UNI_BOUNDARY", "AT_UNI_NON_BOUNDARY") if hasattr(_sre, n)}
_AT_AHEAD = {getattr(_sre, n) for n in (
    "AT_BOUNDARY", "AT_NON_BOUNDARY", "AT_END", "AT_END_LINE", "AT_END_STRING",
    "AT_LOC_BOUNDARY", "AT_LOC_NON_BOUNDARY", "AT_UNI_BOUNDARY",
    "AT_UNI_NON_BOUNDARY") if hasattr(_sre, n)}
_REPEATS = {getattr(_sre, n) for n in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT")
            if hasattr(_sre, n)}
_LEAVES = {_sre.LITERAL, _sre.NOT_LITERAL, _sre.ANY, _sre.IN, _sre.GROUPREF}


def _reach(seq, back: bool = True) -> int:
    out = 0
    for op, av in seq:
        if op is _sre.ASSERT or op is _sre.ASSERT_NOT:
            direction, sub = av
            r = _reach(sub, back)
            if (direction < 0) == back:     # a look in the same direction: width + nested
                w = sub.getwidth()[1]
                if w >= _sre.MAXREPEAT:
                    raise SystemExit(
                        "context: a lookaround without a length limit in a ctx regex — "
                        "the window edge will not bound it (update context._reach)")
                r += w
            out = max(out, r)
        elif op is _sre.AT:
            if av in (_AT_BACK if back else _AT_AHEAD):   # \b, \B, ^/\A, $/\Z — a character beyond the edge
                out = max(out, 1)
        elif op is _sre.BRANCH:
            out = max([out] + [_reach(a, back) for a in av[1]])
        elif op is _sre.SUBPATTERN:
            out = max(out, _reach(av[-1], back))
        elif op in _REPEATS:
            out = max(out, _reach(av[2], back))
        elif op is getattr(_sre, "ATOMIC_GROUP", None):
            out = max(out, _reach(av, back))
        elif op is _sre.GROUPREF_EXISTS:
            _g, yes, no = av
            out = max(out, _reach(yes, back), _reach(no, back) if no else 0)
        elif op not in _LEAVES:
            raise SystemExit(
                f"context: node {op} in a ctx regex is not handled — the look "
                f"beyond the window edge is unknown (update context._reach)")
    return out


def _reach_of(ctx: re.Pattern, back: bool) -> int:
    key = (ctx.pattern, ctx.flags, back)
    hit = _REACH.get(key)
    if hit is None:
        hit = _REACH[key] = _reach(_sre.parse(ctx.pattern, ctx.flags), back)
    return hit


def ctx_reach(ctx: re.Pattern) -> int:
    """How many characters LEFT of the match start ``ctx`` may read
    (lookbehind with nested ones, \\b/\\B, ^ and \\A) — an upper bound from the
    parsed regex. The stream keeps a tail longer than the window by this amount:
    a lookbehind at the left edge of the window sees the same as in the whole text.
    Per-process cache."""
    return _reach_of(ctx, True)


def ctx_ahead(ctx: re.Pattern) -> int:
    """How many characters RIGHT of the match end ``ctx`` may read
    (lookahead with nested ones, \\b/\\B, $ and \\Z) — an upper bound; right_hit sees that many
    characters beyond the right edge of the right window. Per-process cache."""
    return _reach_of(ctx, False)
