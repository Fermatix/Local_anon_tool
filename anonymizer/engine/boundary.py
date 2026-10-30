"""Boundary building blocks shared by every detector.

The single most repeated bug in the original 214-script project was a token
fusing onto a neighbouring letter/digit, or a rule re-eating an already-placed
token. These helpers centralise the fixes:

* JSONL newline is the two characters ``\\`` + ``n`` — so the character to the
  left of a value that starts a line is the letter ``n``. A naive
  ``(?<![A-Za-z0-9])`` left boundary therefore MISSES such values. Use ``L``.
* A minted token ending in a digit (``ORG_1234``) must never touch an adjacent
  digit (would read as ``ORG_12340``) or letter. Use ``R`` / ``ND``.
* Already-placed tokens (``PERSON_NAME_12``) must never be re-processed. Use
  ``TOKEN_RE`` / ``is_placeholder``.
"""
from __future__ import annotations

import re

from . import placeholders as _P

# ---- left / right boundaries -------------------------------------------------
# Left boundary, JSONL-aware (matches at start-of-string, after a real
# newline, after an escaped ``\n``, or after any non-alphanumeric). The fallback
# also excludes a preceding backslash: without it a match could start ON the
# ``n`` of the two-char escape (prev char ``\`` is "non-alphanumeric"),
# swallowing the escape letter and corrupting the JSON string (``\E…``).
#
# ONE alphanumeric boundary for BOTH alphabets (found on a batch of
# call transcripts, structural gate). Before, L/R saw only Latin letters and
# digits, CL/CR only Cyrillic, and mixed ASR garbage of transcripts
# was masked HALFWAY, welding the token to a neighbouring letter of the other alphabet:
# «Яan» → «ЯORG_6478», «заalie» → «заPERSON_NAME_…», «Акмеoysко» →
# «ORG_3992oysко». A letter of any alphabet (or a digit) right next to a literal means it is
# NOT a separate word; such a match is forbidden on both sides. A Latin brand with
# a Russian case ending («Slackом») is covered by brand_variation itself (the ending
# is part of the match and goes under the token as a whole), not by the boundary.
_ALNUM = r"A-Za-zА-Яа-яЁё0-9"
# JSONL escapes left of a value: not only ``\n`` — the exporter escapes
# ``\t \r \b \f \v`` and control characters as ``\uXXXX`` (one corpus had 185 places of
# ``\t``+capital letter, 101 of ``\u000b``+Cyrillic). The letter
# of an escape (t/r/b/u/hex) is a line boundary, not a neighbouring word.
_ESC = r"(?<=\\[ntrbfv])|(?<=\\u[0-9A-Fa-f]{4})"
L = r"(?:" + _ESC + r"|(?<![" + _ALNUM + r"\\]))"
R = r"(?![" + _ALNUM + r"])"    # right: not followed by letter (any script)/digit
ND = r"(?![0-9])"               # right: not followed by a digit (token safety)

# Cyrillic boundaries are the same single alphanumeric brick (pack files and
# detectors import CL/CR by name; L+CL / R+CR in names.py is a harmless
# tautology). CL differs from L only in that ``\\`` is not excluded on the left: a Latin
# literal could start ON an escape letter (``\E…``), a Cyrillic one cannot,
# and paths like ``C:\Users\Иванов`` and JSONL ``C:\\Users\\Иванов`` would otherwise lose
# the mask.
CL = r"(?:" + _ESC + r"|(?<![" + _ALNUM + r"]))"
CR = R

# nsjson (JSONL lines after the raw pass): the value class of a rule in
# RAW JSON text does not eat structure — an unescaped «"» (the closing quote of a string) and «\»
# (the head of an escape: ``\n``, ``\"``, ``\\``, ``\uXXXX`` — a replacement does not tear an escape and does not drag
# a tail through ``\n``). The precedent is the password_context class
# (tests/test_secret_prose.py::test_secret_value_stops_at_json_structural). Read at
# detector import (meeting_passcode, env_secret, signed_url, tme_start_param, passport,
# passport_short_series; os_user_path pairs its separator, see contacts).
# tail of a negated value class: ``[^\s…`` + JSON_SAFE + ``]``
JSON_SAFE = r"\"\\"

# Numeric no-fuse (for financial/registration ids: not adjacent to another
# digit, so we don't match a fragment of a longer number — but a trailing
# comma/period is a delimiter and is allowed to end the match).
# ``_`` is glue too: digits welded by an underscore to a
# neighbour are the tail of a compound id (minted placeholders USER_ID_912345678916 /
# ORG_12 / PERSON_NAME_7, order_1700000008, snake_case keys), not a bare
# number. Before, ``(?<!\d)`` let ``_`` through, and the 12-digit tail of USER_ID
# matched the INN checksum in ~1% of cases → «USER_ID_INN_REDACTED»: ~1%
# of users of a Discord corpus merged into ONE token (false attribution of
# reply chains), no gate saw it (TOKEN_RE passes).
NUM_L = r"(?<![\d_])"
NUM_R = r"(?![\d_])"

# ---- space and dash classes inside a multi-word form ------------------------
# Unicode dashes used to write a hyphen in a name (U+2010 ‐, U+2011 ‑, U+2012 ‒,
# U+2013 –, U+2014 —, U+2015 ―, U+2212 −) — the same set as ru_trace._DASHES and
# vnorm._DASHES. DASH_CLASS is «-» or any of them; HWS_CLASS is a run of whitespace
# without a line break (NBSP/NNBSP from Word and PDF, thin space, double space,
# tab) — the one the pack helper _alt writes as a literal (ANON_PACKGEN_WS_CLASS).
# brand_variation puts them in place of «\ » and «\-» from re.escape by the
# specification keys ws_class / dash_class (packgen writes the keys behind its switches).
DASHES = "‐‑‒–—―−"
DASH_CLASS = "[-" + DASHES + "]"
HWS_CLASS = r"[^\S\n]+"

# ---- boundaries of a brand glued to a digit/Latin letters -------------------------------
# The rule ``brand:<seed>:glue`` (orgs.brand_variation, specification key ``glue`` — written by
# packgen behind ANON_PACKGEN_GLUE) takes «Ромашка2024», «РомашкаBot», «2024Ромашка»,
# «Ромашкой_2024» whole as one ORG token of the entity: left and right of the brand there is a tail
# ``[0-9A-Za-z_]{0,24}``, and these boundaries stand OUTSIDE the tails. Unlike L/R, «_» is also
# forbidden on the left (the tail eats it itself: «x_Ромашка» is glued as a whole, not «x_» + token), a JSONL escape
# on the left (``\n`` — the letter ``n``) is still a line boundary, not a glue tail (otherwise the match would eat
# the escape letter and break JSON). Cyrillic right next to it («ДоговорсРомашка») is not glue
# — it is another word, residual warns (ANON_RESIDUAL_GLUE_WARN). After
# an already minted token joined by «_» («PERSON_NAME_7_Ромашка2024», the token tail is digits or
# REDACTED) there is a boundary: otherwise the glue started only at the token head, the "already a token" guard
# vetoed it, and the brand with a year stayed open; the token itself is not part of the tail (GLUE_TAIL below).
GLUE_L = r"(?:" + _ESC + r"|(?<![" + _ALNUM + r"_\\])|(?<=\d_)|(?<=REDACTED_))"
GLUE_R = R

# ---- token protection --------------------------------------------------------
# Any already-minted mask token (PREFIX_123, PREFIX_123_456, PERSON_NAME_133100,
# DOC_ID_12345, *_REDACTED). The prefix may itself be underscore-joined
# (``PERSON_NAME``), so match a run of UPPER/digit segments glued by ``_``,
# ending in a numeric tail (``_133100`` / ``_20000_1``) or ``_REDACTED``.
# Quantifiers are BOUNDED and the pattern starts with (?<![A-ZА-Я0-9]) — a performance guard:
# unbounded ``*`` tails on a long UPPER/underscore run WITHOUT a numeric
# tail made the scan O(n²) (a 100 KB run = tens of seconds of search, 1 MB = a hang
# of every TOKEN_RE.search in guards). The bounds are generous (segment ≤32, ≤9 segments,
# digit tail ≤8 groups of ≤32) — all minted shapes are covered with a margin; the look-
# behind only forbids a match start INSIDE an UPPER/digit run: the leftmost match
# covered such a start with a longer shape anyway, and «9ORG_12» (a digit right next
# to the left) is a glue defect caught by GLUE_RE, not a valid token.
# Identical to ``engine.identity._TOKEN`` — the two MUST stay in sync, or a
# re-run fails to recognise its own minted tokens and rewrites them.
TOKEN_RE = re.compile(
    r"(?:(?<![A-ZА-Я0-9])[A-ZА-Я][A-ZА-Я0-9]{0,31}(?:_[A-ZА-Я0-9]{1,32}){0,8}"
    r"_(?:\d{1,32}(?:_\d{1,32}){0,7}|REDACTED))"
)
# ---- single source of minted token prefixes ----------------------------------
# All placeholder families (``*_REDACTED`` from placeholders.ALL) + minted
# PREFIX_NNNN prefixes (mint.MOD keys + pname/tag/crm_org: PERSON_NAME, ORG,
# HANDLE, USER, USER_ID, RESOURCE, DOC_ID). PERSON and CRM are protective: a bare
# «PERSON» remains when NER cuts «PERSON_NAME_144500» at a word boundary, and
# CRM is a legacy label fragment. Before, three hand-written lists (_GLUE_PREFIXES,
# MINT_FRAG_RE, the dead TOKEN_AHEAD) lived their own lives and diverged: the glue gate
# was blind to 15 families (ACCOUNT/CARD/SNILS/IBAN/… — their welded tokens
# passed the structural gate unnoticed), and MINT_FRAG_RE did not know the financial
# prefixes. Now one source → alt longest-first (PERSON_NAME before
# PERSON, AWS_ACCOUNT before ACCOUNT — the same discipline as alt_longest_first).
_MINT_PREFIXES = frozenset(
    {"PERSON_NAME", "PERSON", "ORG", "CRM", "HANDLE", "USER_ID", "USER",
     "RESOURCE", "DOC_ID"}
    | {v[: -len("_REDACTED")] for v in _P.ALL}
)
_PREFIX_ALT = "|".join(sorted(_MINT_PREFIXES, key=lambda s: (-len(s), s)))
# A character of a brand glue tail (rule brand:<seed>:glue): a digit, a Latin letter,
# «_», but not the head of a minted token («PERSON_NAME_7_», «_ORG_12»): otherwise the greedy tail ate
# the neighbouring token, the "already a token" guard vetoed the whole match (sub does not try further), and
# «Иванов_Ромашка2024» stayed «PERSON_NAME_7_Ромашка2024». Checked only at a capital
# Latin letter (lower case and digits — without lookahead). Usage: GLUE_TAIL + "{0,24}".
GLUE_TAIL = r"(?:[0-9a-z_]|(?!(?:" + _PREFIX_ALT + r")_(?:\d|REDACTED))[A-Z])"

# A fragment of a minted token WITHOUT a numeric tail: an NER span/harvester cuts
# «PERSON_NAME_144500» at a word boundary — a bare UPPER prefix word remains
# («ванова PERSON», caught by the owner). TOKEN_RE does not catch it (the tail
# is required). Case is strict: ordinary words (Person/Email in prose) survive.
# The alternation comes from the single _PREFIX_ALT (longest-first).
MINT_FRAG_RE = re.compile(
    r"(?<![A-Za-zА-Яа-я0-9_])(?:" + _PREFIX_ALT + r")(?![A-Za-z0-9_])")

# Our token as a PART of a longer value — for guards "the value already holds a token". Only
# known prefix families: TOKEN_RE does not fit here, it recognizes a "token" in any
# «UPPER_digits», and a random base64url key with a piece «xF_1» was considered "ours" (with
# TOKEN_RE.search the guards left real keys, ids of
# shared links and nicknames like «MAX_777» open on raw tg sources).
OWN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:" + _PREFIX_ALT + r")_(?:\d{1,32}(?:_\d{1,32}){0,7}|REDACTED)")

# ---- single "this is our token" test -----------------------------------------
# Decisions "the value is already masked" (Telegram fields, the name registry, rule
# guards) used to take the general TOKEN_RE (or their own copies): any «UPPER_digits»
# was considered a token, and nicknames/names like «PETROV_77», «@KOZLOV_2001», «MEGA_BOT_1»
# went out unmasked — in lower case the same values were masked. Now
# one function on OWN_TOKEN_RE (known minting prefixes). The broad TOKEN_RE
# stays where it is about boundaries (restore, cut points).
# The rollback switch ANON_OWN_TOKEN_STRICT=0 (each place decided with the old regex) is removed.

# Our token as a SEPARATE WORD of a value — for nicknames, slugs and name fields: not
# glued to a letter, digit or «_» on either side. OWN_TOKEN_RE does not exclude «_» on the left
# (a file name «report_PERSON_NAME_12.pdf» — our token inside), and in
# someone's nickname «SUPER_USER_2001» / «MY_ORG_77» it found «USER_2001» / «ORG_77» —
# the nickname stayed open. Neighbours «.» / «-» / «@» / space are
# a word boundary: «@HANDLE_1234567.», «ivan-HANDLE_1234567» — our token.
OWN_WORD_RE = re.compile(
    r"(?<![" + _ALNUM + r"_])(?:" + _PREFIX_ALT + r")_(?:\d{1,32}(?:_\d{1,32}){0,7}|REDACTED)"
    r"(?![" + _ALNUM + r"_])")


def own_token_re(*, word=False):
    """The "our token" regex: OWN_TOKEN_RE (``word=True`` — OWN_WORD_RE, as a separate
    word of the value)."""
    return OWN_WORD_RE if word else OWN_TOKEN_RE


def is_own_token(value, *, part=False, word=False) -> bool:
    """The value is our token as a whole (``part=True`` — contains our token;
    ``word=True`` — contains our token as a separate word: a nickname, slug, name field)."""
    if not isinstance(value, str):
        return False
    rx = own_token_re(word=word)
    return (rx.search(value) if part or word else rx.fullmatch(value)) is not None

# A token welded onto a letter/digit — the "glue" defect the structural gate
# detects. Anchored on the KNOWN minted prefixes: a generic UPPER-run pattern
# self-matches inside every valid token (``EMAIL_REDACTED`` reads as ``E`` +
# ``MAIL_REDACTED``, ``ORG_1234`` as ``O`` + ``RG_1``), which fails the gate on
# perfectly clean output. Ordered longest-variant-first so ``PERSON`` cannot
# shadow ``PERSON_NAME`` (same discipline as ``alt_longest_first``).
_GLUE_PREFIXES = _PREFIX_ALT
# Left glue: an alnum char sitting immediately before a complete known token
# (lookahead, so bare tokens never self-match). The ``(?<!\\n)`` guard is the
# ``L`` idiom inverted: in JSONL the char left of a line-starting token is the
# ``n`` of an escaped ``\n`` — clean output, not glue. Right glue: a FULL
# known token (prefix + ``_digits``/``_REDACTED`` tail) immediately followed
# by a letter (a glued trailing digit is indistinguishable from a longer
# tail — that fusion is prevented at mint time by ``R``/``ND``, not detected
# here). The right branch is anchored on known prefixes, like the left one: a bare
# ``_digits(?=letter)`` fired falsely on raw base64/attachment ids
# (``…NL7_68mGz…``, ``…212402__62e8a…`` — the structural gate failed
# on UNtouched source discord text).
# On the left the SAME JSONL escapes as in L/CL (_ESC) are excluded: the escape letter
# ``\t \r \b \f \v`` and the hex tail ``\uXXXX`` before a token are a line boundary,
# not welding (otherwise the gate would fail a legitimate mask after ``\t``).
GLUE_RE = re.compile(
    r"(?:[A-Za-zА-Яа-яЁё0-9](?<!\\[ntrbfv])(?<!\\u[0-9A-Fa-f]{4})"
    r"(?=(?:" + _GLUE_PREFIXES + r")_(?:\d|REDACTED))"
    r"|(?:" + _GLUE_PREFIXES + r")_(?:\d+|REDACTED)(?=[A-Za-zА-Яа-яЁё]))"
)


def is_placeholder(value: str) -> bool:
    """True if the value is already an anonymised token (skip-me oracle)."""
    v = value.strip()
    if not v:
        return True
    if TOKEN_RE.fullmatch(v):
        return True
    # url/template/code fragments behave like placeholders for our purposes
    return v.startswith(("http", "www.", "./", "/", "{{", "${", "<"))


def alt_longest_first(literals) -> str:
    """Regex alternation of literals, longest first (so a short form can't
    pre-empt a longer superstring — e.g. `acme` must not shadow `megaacme`)."""
    return "|".join(re.escape(x) for x in sorted(set(literals), key=len, reverse=True))


# A cheap necessary condition of glue: the text has a token tail («_digit»
# or «_REDACTED»). A literal anchor is orders of magnitude faster than GLUE_RE (measured:
# 2.4 billion vs 9 million characters/s), lines without tokens are not scanned at all.
_TOKEN_HINT = re.compile(r"_(?:\d|REDACTED)")


def split_fusions(text: str, positions=None) -> tuple:
    """Split token fusions: a «-» at every GLUE_RE point. Returns
    (new, n). The same GLUE_RE as the structural gate's (a single source of
    truth), so after splitting the gate has nothing to fire on; no information is
    lost (the token and the neighbour are intact), idempotent — a repeated call gives n=0.
    Called by the engine after the rules (a rule with an incomplete boundary guard
    welded a token to a letter — «EMAIL_REDACTEDпривет», an INN inside a link —
    and the round stopped at the structural gate) and by fix_engine --deglue.
    ``positions`` (viewmask) — a list: positions of the inserted
    «-» in INPUT coordinates (ascending) are appended."""
    if not _TOKEN_HINT.search(text):
        return text, 0
    out, i, n = [], 0, 0
    for m in GLUE_RE.finditer(text):
        out.append(text[i:m.end()])
        out.append("-")
        i = m.end()
        n += 1
        if positions is not None:
            positions.append(i)
    if not n:
        return text, 0
    out.append(text[i:])
    return "".join(out), n
