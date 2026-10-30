"""Contractor from [executor]: masking rules and forms for the check.

Names, legal entities (with and without the legal form, in «» and "" quotes), brands and
``extra_terms`` → ``ORG_…``; domains and email domains → ``RESOURCE_…`` (the seed is the lowercase
host, as in the engine's ``domain`` rule: the site "https://acme.ru/about" gets the same host
token, the path stays); INN/OGRN/KPP — engine constants by length
(``INN_REDACTED``/``OGRN_REDACTED``/``KPP_REDACTED``: the same class the financial detector gives by
checksum — so the main pipeline does not see two different tokens for one registration number).

Forms: Cyrillic — the stem without the ending + a closed set of case endings (like the engine's
``client_pack``: an open ending ``[а-яё]*`` ate "Новая версия" for the client "Нова"),
case-insensitive; Latin — the whole word, case-insensitive, with a Russian case ending ("в Slackом"
— like ``brand_variation``); a Cyrillic name also gets its transliteration (ru_trace.translit — the
same as the pack). Latin/Cyrillic look-alikes inside a word are folded by the engine view layer
(vnorm) before the rules, no separate forms are needed.

The seed of the contractor ORG token is the first name from ``names`` (or the core of the first
legal entity): the same seed the engine's ``opf_name`` mints for "ООО «Ромашка»" (the name inside
the quotes), so names without the legal form, legal entities and brands with the same core are one
token. A brand with another core gets its own token by the brand.

A name or brand that is a common word (the engine's common word dictionary: engine/lexicon
``word_kind == "plain"`` or a common-noun homonym ``name_dicts.is_homonym`` — "Мир", "Старт",
"Проект", "Acme") is masked only Capitalized ("Проект готов"), in CAPS, in quotes («проект») or
next to a legal form ("ООО проект"); lowercase in text ("проектом довольны") is left alone —
otherwise masking would be massively wrong. The check forms of such a term are case-sensitive. The
exception is a host, a file or directory name, a code identifier ("ромашка_тз.docx",
"import acme.settings", "acme-corp", "docs/ромашка": the ``glued`` predicate): there any case
counts; the ``…:glued`` rule of any name sits in the pack slot after the other domains, the gate
judges with the same ``glued``.

Domains with IDN ("ромашка.рф", "shop.ромашка.рус"): both spellings go to the rule and the check
(unicode and punycode "xn--…"), the token seed is the unicode one (one value — one token).

The contractor INN/OGRN/KPP — digits with optional spaces between them ("7707 083893",
"1 027700132195"); the rule is first in the chain (``Pack.first``), before the financial detector,
otherwise it would mint part of the digits as DOC_ID. The check uses the same regex
(gates_control), not the automaton.
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field

from anonymizer.engine import mint
from anonymizer.engine import name_dicts
from anonymizer.engine import placeholders as P
from anonymizer.engine.boundary import (HWS_CLASS, L, NUM_L, NUM_R, R, alt_longest_first,
                                        is_own_token)
from anonymizer.engine.rule import Rule
from anonymizer.engine.vocab import LEGAL_FORMS, TRANSLIT
from anonymizer.known_values.client_pack import _CASE_TAIL

from . import rxdefer
from .roster import in_path

_QUOTES = "«»„“”\"'"
_OPF = sorted({f for f in LEGAL_FORMS}, key=len, reverse=True)
_OPF_ALT = "|".join(re.escape(f) for f in _OPF)
_OPF_RX = re.compile(r"^(?:" + _OPF_ALT + r")\.?\s+", re.IGNORECASE)
_ID_SP = r"[  ]?"
_HAS_CYR = re.compile(r"[А-Яа-яЁё]")
_CYR_VOWELS = "аяоеёиыуюь"
# host label — letters of any alphabet (IDN: "ромашка.рф", "shop.ромашка.рус"), digits, hyphen
_LABEL = r"[^\W_](?:(?:[^\W_]|-)*[^\W_])?"
_HOST_RX = re.compile(r"^" + _LABEL + r"(?:\." + _LABEL + r")+$")
_TAILS = [t for t in _CASE_TAIL[3:-2].split("|") if t]        # client_pack endings as a list


def translit(word: str) -> str:
    return "".join(TRANSLIT.get(ch, ch) for ch in word.lower())


def legal_core(name: str) -> str:
    """Legal entity core without the legal form and quotes: "ООО «Ромашка»" → "Ромашка"."""
    s = " ".join(name.split()).strip(_QUOTES + " ")
    s = _OPF_RX.sub("", s).strip(_QUOTES + " ")
    return s


def cyr_stem(word: str) -> str:
    """Stem of a Cyrillic word: final vowels/ь cut (not shorter than three letters)."""
    w = word
    while len(w) > 3 and w[-1].lower() in _CYR_VOWELS:
        w = w[:-1]
    return w


def word_forms(word: str) -> list:
    """Explicit case forms of a word (for the check): stem + client_pack endings."""
    if not _HAS_CYR.search(word):
        return [word]
    stem = cyr_stem(word)
    out = [word]
    for t in _TAILS:
        f = stem + t
        if f.lower() != word.lower() and f not in out:
            out.append(f)
    if word.lower() != stem.lower() and stem not in out:
        out.append(stem)
    return out


def term_pattern(term: str) -> str:
    """Regex of one term: a Cyrillic word — stem + case ending, a Latin one — as is with a Russian
    ending; spaces — a run of whitespace without a newline, hyphens — dashes."""
    words = term.split()
    parts = []
    for w in words:
        if _HAS_CYR.search(w):
            sub = []
            for piece in w.split("-"):
                sub.append(re.escape(cyr_stem(piece)) + _CASE_TAIL if len(piece) > 3
                           else re.escape(piece))
            parts.append(r"[-‐‑‒–—―−]".join(sub))
        else:
            parts.append(re.escape(w) + r"[а-яё]*")
    return HWS_CLASS.join(parts)


@dataclass
class Pack:
    rules: list = field(default_factory=list)         # before the orgs detector
    first: list = field(default_factory=list)         # before all detectors (registration numbers)
    late: list = field(default_factory=list)          # in the pack slot after the other domains (glued)
    # (form, kind, case_sensitive: True | False | GLUED — any case, host/path only |
    # GLUED_CASED — case-sensitive, host/path only | SQL_CASED, SQL_ANY — case-sensitive and not,
    # except the CAPS form in an SQL position, sql_position | SHORT_UPPER — CAPS of a short term
    # outside a host/path | QUOTED, OPF — any case in quotes, after a legal form)
    controls: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)


# the term is not a host label: "romashka.ru", "acme-corp.ru" left of the dot are not touched, the
# whole domain is taken by the executor:domain rule (own domain) or the engine's domain (foreign) —
# otherwise a stub ".ru" or "ORG_…-corp.ru" would remain
_NOT_HOST = r"(?![\w-]*\.[^\W_])"
_LEFT = r"(?:(?<=\\[ntrbfv])|(?<=\\u[0-9A-Fa-f]{4})|(?<![A-Za-zА-Яа-яЁё0-9\\.@-]))"


def common_word(term: str) -> bool:
    """The term is a common word by the engine dictionaries: every word is plain in engine/lexicon
    (pymorphy/SCOWL; without natasha Cyrillic is not checked) or a common-noun homonym
    (name_dicts.is_homonym: "Мир", "Сокол", "Ромашка")."""
    words = term.split()
    if not words:
        return False
    for w in words:
        w = w.strip(_QUOTES + "-")
        if not w:
            continue
        if _HAS_CYR.search(w) and name_dicts.is_homonym(w):
            continue
        if _lexicon_plain(w):
            continue
        return False
    return True


def _lexicon_plain(word: str) -> bool:
    if _HAS_CYR.search(word) and importlib.util.find_spec("natasha") is None:
        return False
    try:
        from anonymizer.engine import lexicon
        return lexicon.word_kind(word) == "plain"
    except (SystemExit, ImportError, OSError):     # no dictionaries/natasha — not common
        return False


def _org_rule(name: str, terms: list, seed: str) -> Rule:
    pats = [term_pattern(t) for t in sorted(set(terms), key=len, reverse=True)]
    # compiled on first use or by the engine in a batch over its workers (rxdefer)
    rx = rxdefer.compile(_LEFT + r"(" + "|".join(pats) + r")" + R + _NOT_HOST, re.IGNORECASE)
    tok = mint.tag("ORG", seed)

    def mask(m, t=tok):
        return t

    return Rule(name, rx, mask=mask, value_group=1, guard=_guard, confidence="high")


def _guard(m, group: int = 1) -> bool:
    """Do not mask: our own token or the CAPS form of an SQL keyword in a query (sql_position)."""
    return is_own_token(m.group(group), part=True) or sql_position(m.string, m.start(group),
                                                                   m.end(group))


def _cased_pattern(term: str) -> str:
    """A common-word term: only Capitalized or CAPS (endings of any case)."""
    words = term.split()
    cap = HWS_CLASS.join(_term_word(w, w[:1].upper() + w[1:]) for w in words)
    up = HWS_CLASS.join(_term_word(w, w.upper()) for w in words)
    return r"(?:" + cap + r"|" + up + r")"


def _term_word(orig: str, cased: str) -> str:
    if _HAS_CYR.search(orig):
        sub = []
        for piece in cased.split("-"):
            sub.append(re.escape(cyr_stem(piece)) + r"(?i:" + _CASE_TAIL + r")" if len(piece) > 3
                       else re.escape(piece))
        return r"[-‐‑‒–—―−]".join(sub)
    return re.escape(cased) + r"(?i:[а-яё]*)"


def _common_rules(name: str, terms: list, seed: str) -> list:
    """Rules of a common-word term: (1) Capitalized/CAPS in text; (2) any case in quotes or after a
    legal form ("ООО проект", «"проект"»)."""
    tok = mint.tag("ORG", seed)
    ordered = sorted(set(terms), key=len, reverse=True)
    cased = re.compile(_LEFT + r"(" + "|".join(_cased_pattern(t) for t in ordered) + r")" + R
                       + _NOT_HOST)
    return [Rule(name, cased, mask=lambda m, t=tok: t, value_group=1, guard=_guard,
                 confidence="high")] + _quoted_opf_rules(name, ordered, tok)


def quoted_rx(alt: str, flags=re.IGNORECASE):
    """Regex of a quoted term: the ``alt`` alternative right between quotes — the :quoted rule and
    the QUOTED check form (quoted_hit decides what counts)."""
    return re.compile(r"(?<=[" + _QUOTES + r"])(" + alt + r")(?=[" + _QUOTES + r"]|\\[\"'])",
                      flags)


# typographic quotes are always prose; ASCII «"» and «'» are string syntax in code, YAML, XML…
# JSON is not in the list: rules see both the raw file and the decoded field values (a Telegram
# export is prose «Работаем с "data"» with the same rel), and in raw JSON any quoted string is a
# key or a value by the grammar (quoted_literal)
_TYPO_QUOTES = "«»„“”"
CODE_EXT = frozenset("""
.yaml .yml .toml .xml .xsd .xsl .plist .ini .cfg .conf .env
.properties .lock .py .pyi .js .mjs .cjs .jsx .ts .tsx .vue .svelte .java .kt .kts .scala
.groovy .gradle .go .rb .php .cs .fs .vb .c .h .cc .cpp .cxx .hpp .hh .m .mm .swift .rs .dart .lua
.pl .pm .r .jl .ex .exs .erl .hs .clj .sql .sh .bash .zsh .fish .ps1 .psm1 .bat .cmd .tf .hcl
.proto .graphql .gql .dockerfile .mk .cmake
""".split())
CODE_NAMES = frozenset("dockerfile makefile gemfile rakefile procfile jenkinsfile vagrantfile".split())


def code_file(rel: str) -> bool:
    """A code or configuration file by name (CODE_EXT, CODE_NAMES, ".env*"): ASCII quotes in it are
    string syntax, the :quoted rule ignores them (quoted_hit)."""
    name = rel.replace("\\", "/").rsplit("/", 1)[-1].lower()
    dot = name.rfind(".")
    return (name in CODE_NAMES or name.startswith(".env")
            or (dot > 0 and name[dot:] in CODE_EXT) or (dot == 0 and name in CODE_EXT))


def quoted_hit(text: str, start: int, end: int, *, code: bool) -> bool:
    """The term ``text[start:end]`` right inside quotes (quoted_rx) is a quoted name: a typographic
    quote next to it — always; both ASCII — only outside a code file (``code``) and not a literal
    key or value (quoted_literal: ``"data":``, ``: "data"``, ``["group"]``, ``('order')``,
    ``= "team"``); escaped on both sides (``\\"data\\"`` inside a JSON or code string) —
    always: this is not syntax. A predicate shared by the :quoted rules and the gate."""
    if text[start - 1] in _TYPO_QUOTES or text[end] in _TYPO_QUOTES:
        return True
    if text[end] == "\\":
        return start >= 2 and text[start - 2] == "\\"
    return not code and not quoted_literal(text, start - 1, end + 1)


def quoted_literal(text: str, qs: int, qe: int) -> bool:
    """The ASCII-quoted string ``text[qs:qe]`` is a literal key (followed by ":", preceded by "{",
    ",", "[" or the line start) or a value (preceded, across spaces and newlines, by ":", "=", "[",
    "(", "{", ","; followed on the same line by ",", "]", ")", "}", ";" or the line end)."""
    i = qs - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    line_start = i < 0 or text[i] in "\n\r"
    while i >= 0 and text[i] in " \t\n\r":
        i -= 1
    before = text[i] if i >= 0 else ""
    j = qe
    while j < len(text) and text[j] in " \t":
        j += 1
    after = text[j] if j < len(text) else "\n"
    if after == ":" and (line_start or before in ("{", ",", "[")):
        return True
    return before != "" and before in ":=[({," and after in ",]});\n\r"


class _ProseFiles:
    """scope of the ASCII-quote rule (the engine's Rule.scope: rel ∈ scope): the file is not code (code_file)."""

    def __contains__(self, rel) -> bool:
        return not code_file(rel)

    def __iter__(self):            # the rule fingerprint (rule_digest) sorts scope
        return iter(("<не code_file>",))


PROSE_FILES = _ProseFiles()


def opf_rx(alt: str, flags=re.IGNORECASE):
    """Regex of a term after a legal form ("ООО проект"): group 1 — the legal form with spaces, 2 —
    the term; the :opf rule and the OPF check form. Without IGNORECASE — the legal form lowercase,
    for the gate's case-folded text (so re filters positions by the first letter: several times
    faster on a 100 MB log)."""
    opf = _OPF_ALT if flags & re.IGNORECASE else _OPF_ALT.lower()
    return re.compile(r"((?:" + opf + r")\.?" + HWS_CLASS + r")(" + alt + r")" + R
                      + _NOT_HOST, flags)


def _quoted_opf_rules(name: str, ordered: list, tok: str) -> list:
    """Any case in quotes or after a legal form — for a common word and a short term. Quotes — two
    rules (quoted_hit): with a typographic one — in any file; ASCII on both sides — only outside
    code files (scope PROSE_FILES) and not a literal key or value."""
    anyc = "|".join(term_pattern(t) for t in ordered)
    rx = quoted_rx(anyc)

    def quoted_guard(code):
        return lambda m: is_own_token(m.group(1), part=True) \
            or not quoted_hit(m.string, m.start(1), m.end(1), code=code)

    return [
        Rule(name + ":quoted", rx, mask=lambda m, t=tok: t, value_group=1,
             guard=quoted_guard(True), confidence="high"),
        Rule(name + ":quoted_ascii", rx, mask=lambda m, t=tok: t, value_group=1,
             guard=quoted_guard(False), scope=PROSE_FILES, confidence="high"),
        Rule(name + ":opf", opf_rx(anyc), mask=lambda m, t=tok: m.group(1) + t, value_group=2,
             guard=lambda m: is_own_token(m.group(2), part=True), confidence="high"),
    ]


GLUED = "glued"     # third field of a check form: any case, a hit only in a host/path (glued)
GLUED_CASED = "glued_cased"     # the same, case-sensitive (Capitalized/CAPS of a dev word, dev_word)
SQL_CASED = "sql_cased"         # case-sensitive, except the CAPS form in an SQL position (sql_position)
SQL_ANY = "sql_any"             # the same, case-insensitive
SHORT_UPPER = "short_upper"     # CAPS of a short term: a hit outside a host/path and SQL (short_upper_hit)
QUOTED = "quoted"               # any case, right between quotes (quoted_rx)
OPF = "opf"                     # any case, after a legal form (opf_rx)
_GLUE_BEFORE = "_-./\\"
_GLUE_AFTER = "_-/"
# "ромашка-то", "ромашка-де" — a particle, prose (not "ромашка-тошка", not "ромашка-то.docx")
_PARTICLE_RX = re.compile(r"-(?:то|ка|либо|нибудь|таки|де)(?![\w-]|\.\w)", re.IGNORECASE)
# the contractor name is a dev word: the lowercase form inside a glued word or path is not masked
# (the directory "core/" renamed but "from core import" not: the code would break);
# Capitalized/CAPS and domain — as for a common word
DEV_WORDS = frozenset("""
core app apps api web www data main common utils util lib libs src test tests docs doc config
configs conf server client backend frontend mobile admin shared build dist scripts script bin
static assets public vendor internal pkg cmd tools tool models model views view services service
components component modules module packages package plugins plugin examples example
team teams group groups
""".split())
# SQL keywords: a name that is such a word itself ("Group", "Select") in the CAPS form in a query
# ("GROUP BY", "SELECT … FROM") is masked neither by the rule nor by the check (sql_position)
SQL_KEYWORDS = frozenset("""
SELECT FROM WHERE GROUP ORDER BY JOIN LEFT RIGHT INNER OUTER FULL CROSS ON AS AND OR NOT IN IS
NULL LIMIT OFFSET HAVING UNION ALL INSERT INTO VALUES UPDATE SET DELETE CREATE TABLE INDEX DROP
ALTER PRIMARY KEY FOREIGN REFERENCES DISTINCT CASE WHEN THEN ELSE END WITH VIEW EXISTS BETWEEN
LIKE ASC DESC DEFAULT CONSTRAINT UNIQUE CHECK RETURNING TRUNCATE GRANT REVOKE COUNT DATA USER
""".split())
# a query sign in a statement is a CAPS pair: a start (_SQL_START) and a second part (_SQL_PAIR); BY,
# FROM, ORDER, GROUP next to the name are not a sign by themselves ("A PRODUCT BY GROUP", "NEWS FROM
# ORDER")
_SQL_START = frozenset("SELECT INSERT UPDATE DELETE WITH CREATE ALTER MERGE".split())
_SQL_PAIR = frozenset("FROM WHERE INTO SET JOIN VALUES TABLE".split())
_UPPER_WORD_RX = re.compile(r"(?<![\w])[A-Z]+(?![\w])")
# a statement boundary is ";" or a blank line; no further than _STMT_MAX chars each way
# (a log without blank lines — no quadratic pass on every hit)
_BLANK_LINE_RX = re.compile(r"\n[^\S\n]*\n")
_STMT_MAX = 4096


def sql_keyword(term: str) -> bool:
    """The whole term is an SQL keyword (SQL_KEYWORDS, case-insensitive)."""
    return term.strip().upper() in SQL_KEYWORDS


def sql_position(text: str, start: int, end: int) -> bool:
    """The word ``text[start:end]`` is an SQL keyword in CAPS in a query: the same statement (from
    ";" or a blank line to ";" or a blank line, across the lines of a multi-line query) has other
    CAPS words — a query start (SELECT, INSERT, UPDATE, DELETE, WITH, CREATE, ALTER, MERGE) before
    the word and a second part (FROM, WHERE, INTO, SET, JOIN, VALUES, TABLE), and the word's line
    (from its start or from the query start to the end) has no Cyrillic outside literals '…', "…"
    and comments -- and /* */. A predicate shared by the name rules and the gate."""
    if text[start:end] not in SQL_KEYWORDS:
        return False
    # ";" — str.rfind/find, a blank line — by regex only after it (a hit on every log line)
    lo = max(0, start - _STMT_MAX)
    lo = max(lo, text.rfind(";", lo, start) + 1)
    for m in _BLANK_LINE_RX.finditer(text, lo, start):
        lo = m.end()
    hi = min(len(text), end + _STMT_MAX)
    semi = text.find(";", end, hi)
    hi = hi if semi < 0 else semi
    m = _BLANK_LINE_RX.search(text, end, hi)
    hi = m.start() if m else hi
    first = -1
    pair = False
    for m in _UPPER_WORD_RX.finditer(text, lo, hi):
        if m.start() != start:
            if first < 0 and m.group() in _SQL_START and m.start() < start:
                first = m.start()
            pair = pair or m.group() in _SQL_PAIR
            if first >= 0 and pair:
                break
    else:
        return False
    # the name's line (from its start or from the query start) — no Cyrillic outside literals and
    # SQL comments: "Коллеги из DATA просят: SELECT …", a neighbouring chat log line is prose
    bol = max(first, text.rfind("\n", lo, start) + 1)
    eol = text.find("\n", end, hi)
    seg = text[bol:start] + " " + text[end:hi if eol < 0 else eol]
    seg = _SQL_NOISE_RX.sub(" ", seg)
    return not _CYR_RX.search(seg)


_SQL_NOISE_RX = re.compile(r"'(?:[^'\n]|'')*'|\"[^\"\n]*\"|--[^\n]*|/\*.*?\*/", re.DOTALL)
_CYR_RX = re.compile(r"[А-Яа-яЁё]")


def dev_word(term: str) -> bool:
    """The whole term is a dev word (DEV_WORDS, case-insensitive)."""
    return term.strip().lower() in DEV_WORDS


def glued(text: str, start: int, end: int, *, path: bool | None = None) -> bool:
    """The word ``text[start:end]`` is part of a host, a file or directory name, a code identifier:
    right next to "_", "-", "." (after the word — before a letter/digit), "/", "\\" or a digit
    ("ромашка_тз.docx", "import acme.settings", "acme-corp", "docs/ромашка"); a whole file or
    directory name (``path``; None — by roster.path_context). A predicate shared by the lowercase
    name rule and the gate. A sentence end ("ромашка."), a JSON newline ("\\nромашка"), a particle
    after a hyphen ("ромашка-то") and a closing tag name ("</pro>", like "<pro>") — no."""
    if path is None:
        path = in_path()
    if path:
        return True
    if start > 0:
        b = text[start - 1]
        esc = b in "ntrbfv" and start >= 2 and text[start - 2] == "\\"
        tag = b == "/" and start >= 2 and text[start - 2] == "<"     # closing tag "</pro>"
        if not esc and not tag and (b in _GLUE_BEFORE or b.isdigit()):
            return True
    if end < len(text):
        a = text[end]
        nxt = text[end + 1] if end + 1 < len(text) else ""
        if a == "-" and _PARTICLE_RX.match(text, end):
            return False
        if a in _GLUE_AFTER or a.isdigit():
            return True
        if a == "." and nxt.isalnum():
            return True
        if a == "\\" and nxt and nxt not in "ntrbfvu\"":
            return True
    return False


def _glued_rule(name: str, terms: list, seed: str, *, cased: bool = False) -> Rule:
    """The contractor name in any case (``cased`` — only Capitalized/CAPS: a dev word, dev_word) in a
    host/path/identifier (glued) — in the pack slot after the other domains: a foreign host
    ("acme-corp.ru") is a whole RESOURCE by then."""
    ordered = sorted(set(terms), key=len, reverse=True)
    pats = "|".join((_cased_pattern if cased else term_pattern)(t) for t in ordered)
    rx = re.compile(r"(?:(?<=\\[ntrbfv])|(?<![^\W\d_]))(" + pats + r")(?![^\W\d_])",
                    0 if cased else re.IGNORECASE)
    tok = mint.tag("ORG", seed)

    def guard(m):
        return is_own_token(m.group(1), part=True) or not glued(m.string, m.start(1), m.end(1))

    return Rule(name, rx, mask=lambda m, t=tok: t, value_group=1, guard=guard,
                confidence="high")


def _short_rules(name: str, terms: list, seed: str) -> list:
    """Short (≤3 chars) Latin terms: as written (like the pack's exact_case); the CAPS form — outside
    a host/path/identifier (short_upper_hit: "КОМПАНИЯ DEV", not "DEV_MODE"); any case in quotes
    and after a legal form — as for a common word."""
    rx = re.compile(_LEFT + r"(" + alt_longest_first(terms) + r")" + R + _NOT_HOST)
    tok = mint.tag("ORG", seed)
    rules = [Rule(name, rx, mask=lambda m, t=tok: t, value_group=1, guard=_guard,
                  confidence="high")]
    up = sorted({t.upper() for t in terms} - set(terms), key=len, reverse=True)
    if up:
        urx = re.compile(_LEFT + r"(" + alt_longest_first(up) + r")" + R + _NOT_HOST)
        rules.append(Rule(name + ":upper", urx, mask=lambda m, t=tok: t, value_group=1,
                          guard=lambda m: is_own_token(m.group(1), part=True)
                          or not short_upper_hit(m.string, m.start(1), m.end(1)),
                          confidence="high"))
    return rules + _quoted_opf_rules(name, sorted(set(terms), key=len, reverse=True), tok)


def short_upper_hit(text: str, start: int, end: int) -> bool:
    """The CAPS form of a short term ``text[start:end]`` is to be masked: not glued to an identifier,
    host or path (glued without path context: "DEV.docx" as a file name is masked) and not an SQL
    keyword in a query (sql_position). A predicate shared by the rule and the gate."""
    return not glued(text, start, end, path=False) and not sql_position(text, start, end)


def idn_forms(host: str) -> list:
    """A host in both IDN spellings: [unicode, punycode] ("ромашка.рф" ↔ "xn--80aa3agjl3d.xn--p1ai");
    an ASCII host without xn-- — one. Invalid for IDNA — as is."""
    host = host.lower()
    try:
        if host.isascii():
            uni = host.encode("ascii").decode("idna") if "xn--" in host else host
        else:
            uni = host
        puny = uni.encode("idna").decode("ascii")
    except UnicodeError:
        return [host]
    return [uni] if uni == puny else [uni, puny]


def host_seed(host: str) -> str:
    """Seed of a host RESOURCE token — the lowercase unicode spelling: both IDN spellings — one token."""
    return idn_forms(host)[0]


def _domain_rule(hosts: list) -> Rule:
    alt = alt_longest_first([h.lower() for h in hosts])
    rx = re.compile(r"(?:(?<=\\n)|(?<![\w.@\\-]))((?:(?:[^\W_]|-)+\.)*(?:" + alt + r"))(?![\w-])",
                    re.IGNORECASE)

    def mask(m):
        return mint.tag("RESOURCE", host_seed(m.group(1)))

    return Rule("executor:domain", rx, mask=mask, value_group=1,
                guard=lambda m: is_own_token(m.group(1), part=True), confidence="high")


def _id_placeholder(digits: str) -> str:
    n = len(digits)
    if n in (13, 15):
        return P.OGRN
    if n == 9:
        return P.KPP
    return P.INN


def id_pattern(digits: str) -> str:
    """Regex of a registration number: digits with an optional space between them ("7707 083893")."""
    return _ID_SP.join(re.escape(d) for d in digits)


def id_rx(ids: list):
    alts = sorted((id_pattern(d) for d in ids), key=len, reverse=True)
    return re.compile(NUM_L + r"(" + "|".join(alts) + r")" + NUM_R)


def _id_rule(ids: list) -> Rule:
    return Rule("executor:inn", id_rx(ids),
                mask=lambda m: _id_placeholder(re.sub(r"\D", "", m.group(1))), value_group=1,
                confidence="high")


def build(ex) -> Pack:
    """Rules and check forms from the [executor] config section."""
    pack = Pack()
    names = [n.strip() for n in ex.names if n and n.strip()]
    legal = [n.strip() for n in ex.legal_names if n and n.strip()]
    cores = [legal_core(n) for n in legal]
    cores = [c for c in cores if c]
    seed = names[0] if names else (cores[0] if cores else "")
    org_terms = []
    # a whole legal entity does not go into the terms: the core is masked ("ООО «Ромашка Групп»" →
    # "ООО «ORG_…»", the legal form and quotes stay — like the engine's opf_name), the core is the
    # same token as the name
    for t in names + cores:
        if t and t not in org_terms:
            org_terms.append(t)
    for t in list(org_terms):
        if _HAS_CYR.search(t) and len(t) >= 4:
            tr = translit(t)
            if tr not in [x.lower() for x in org_terms]:
                org_terms.append(tr)
    hosts = []
    for h in list(ex.domains) + list(ex.email_domains):
        h = h.strip().lower().rstrip("/")
        h = re.sub(r"^[a-z]+://", "", h)
        if not (h and _HOST_RX.match(h)):
            continue
        for f in idn_forms(h):
            if f not in hosts:
                hosts.append(f)
    # domains go to first (before the detectors): a host label must not end up under ORG, and
    # punycode "xn--…" under ENCODED of the encoding detector; the rule leaves emails alone ("@" on
    # the left is not allowed), the engine's contacts takes them
    if hosts:
        pack.first.append(_domain_rule(hosts))
        for h in hosts:
            pack.controls.append((h, "executor.domain", False))
    n_common = 0
    if org_terms and seed:
        n_common += _term_rules(pack, "executor:org", org_terms, seed, "executor.name")
        for lg in legal:
            pack.controls.append((lg, "executor.legal", False))
    for kind, terms in (("brand", ex.brands), ("term", ex.extra_terms)):
        for i, t in enumerate(t.strip() for t in terms if t and t.strip()):
            forms = [t]
            if _HAS_CYR.search(t) and len(t) >= 4:
                forms.append(translit(t))
            n_common += _term_rules(pack, f"executor:{kind}:{i}", forms, t, f"executor.{kind}")
    ids = []
    for v in ex.inn:
        d = re.sub(r"\D", "", str(v))
        if 9 <= len(d) <= 15 and d not in ids:
            ids.append(d)
    if ids:
        pack.first.append(_id_rule(ids))
        for d in ids:
            pack.controls.append((d, KIND_INN, False))
    pack.counts = {"org_terms": len(org_terms), "brands": len(ex.brands), "domains": len(hosts),
                   "inn": len(ids), "extra_terms": len(ex.extra_terms),
                   "common_word_terms": n_common}
    return pack


KIND_INN = "executor.inn"           # checked by the id_rx regex, not the automaton

# organizations the name model found: rules of the contractor kind, one ORG token per name
ORG_MIN_FINDS = 2           # a common-word or short Latin name: found at least this many times
ORG_MIN_SHARE = 0.2         # finds / occurrences of the name in the lines: below — not a name (a tool, a word)
_ORG_MAX_WORDS = 5
_ORG_EDGE = _QUOTES + " .,;:!?()[]{}<>*_#/\\|—–-"
# generic words the model may tag alone — not a name
_ORG_STOP = frozenset("""
компания компании организация организации фирма фирмы клиент клиенты заказчик исполнитель подрядчик
поставщик партнёр партнер банк офис сервис команда группа отдел холдинг агентство студия магазин
company companies client clients customer vendor partner bank office service team group studio agency
inc llc ltd
""".split()) | frozenset(f.casefold() for f in LEGAL_FORMS)


def _org_core(surface: str) -> str:
    """A model mention → the name without a legal form, quotes and edge punctuation; '' — not a name."""
    core = legal_core(" ".join(surface.split())).strip(_ORG_EDGE)
    if len(core) < 2 or not any(ch.isalpha() for ch in core) or is_own_token(core, part=True):
        return ""
    if len(core.split()) > _ORG_MAX_WORDS or core.casefold() in _ORG_STOP:
        return ""
    return core


# terms the model tags as an organization in logs, CSV and code: protocols and standards, client metrics (librdkafka
# statistics), utility billing (services, meters, generic organization types). Not the customer's data; LocalAnon's
# own list for the model's names only (the engine lexicon is shared code). A word — in any name made of such words,
# a phrase — as a whole; by _org_key.
_ORG_PUBLIC_WORDS = """
pdu apdu hdlc snrm aarq aare dlms cosem obis modbus mbus m-bus zigbee lorawan nb-iot mqtt coap amqp ntp snmp dhcp icmp
arp vlan pci enum eof ttl crc ack nack mtu uart rs232 rs485 rs-485 gsm gprs lte imei imsi iccid msisdn apn ota rx tx
m2m м2м nan null tarif tariff rdkafka librdkafka postcss rxbytes txbytes rxmsgs txmsgs hdrsize stddev outofrange acked
жкх жку мкд тко тбо жбо хвс гвс одн одпу ипу опу пу вру аскуэ аиис куэ вдго вкго гкал квт тэо бп ээ тэ тп ктп рп
фл юл лс расч сгв свк свкм св гос
ук уо рсо тсж тсн жск жэк ерц эк эс
электроэнергия теплоэнергия электроснабжение теплоснабжение газоснабжение водоснабжение водоотведение отопление
газораспределение домофон взнос капремонт вывоз пошлина экология норма норматив колл-центр
""".split()
_ORG_PUBLIC_PHRASES = (
    "тепловая энергия", "электрическая энергия", "холодное водоснабжение", "горячее водоснабжение",
    "капитальный ремонт", "межэтажный щит", "вывоз мусора", "вывоз тко", "обращение с тко")
_public_keys: tuple | None = None
# a code identifier: CamelCase ("RetryCount", "XMLParser"), split into parts — CAPS runs, words, digits
_CAMEL = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[a-z][A-Z]|[A-Z]{2}[a-z])[A-Za-z0-9]*")
_CAMEL_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
# an id: hex with a digit ("5F3A9C01BE"), mostly digits ("0045Z"); a byte of a hex dump ("FF", "0F") is code-shaped
_HEX_ID = re.compile(r"(?=[0-9A-Fa-f]*\d)[0-9A-Fa-f]{4,}")
_HEX_BYTE = re.compile(r"[0-9A-F]{2}")
_ALNUM = re.compile(r"[A-Za-z0-9]{3,}")
_LATIN_WORD = re.compile(r"[A-Za-z]+")
# a line of prose: three Cyrillic words in a row and no marks of structured data (JSON, CSV, a tag, a tab, a log
# timestamp) — a code-shaped name the model found there may be a name (_code_shaped)
_PROSE = re.compile(r"(?<!\w)[А-Яа-яЁё]{2,}(?:[ ,.!?()«»\"—–-]+[А-Яа-яЁё]{2,}){2}(?!\w)")
_STRUCTURED = re.compile(r'\{"|":|";"|\t|<[A-Za-z/]|^\s*\d{4}-\d\d-\d\d[T ]\d|;[^;]*;[^;]*;')
CODE_PROSE_SHARE = 0.5      # a code-shaped name with less of its occurrences in prose lines is code, not a name
# occurrences in text views (chats, documents, prose files) a name needs: one that lives only in CSV cells, log fields
# and data JSON is a code or a category value of the export, not a name (Л-23)
ORG_MIN_TEXT = 1
# a model or size after a public abbreviation: "СВК-20", "СГВ-20Г"
_MODEL_TAIL = re.compile(r"-\d+[A-Za-zА-Яа-яЁё]?$")


def _lexicon():
    """The engine dictionaries (anonymizer.engine.lexicon); None — not installed."""
    try:
        from anonymizer.engine import lexicon
        return lexicon
    except (SystemExit, ImportError, OSError):
        return None


def _dict_check(fn, *args) -> bool:
    """A dictionary predicate; dictionaries missing (lexicon stops with SystemExit) — False."""
    try:
        return bool(fn(*args))
    except (SystemExit, OSError):
        return False


def _english_word(lexicon, part: str) -> bool:
    """An English dictionary word, a proper-noun homonym too ("Reading", "Normal")."""
    return part.casefold() in lexicon._en("words") or lexicon.word_kind(part) == "plain"


def _code_identifier(lexicon, word: str) -> bool:
    """CamelCase of English words — a field, class or method name ("RetryCount", "XMLParser",
    "HttpClientFactory"): every part longer than two letters that is not CAPS is an English word.
    A part outside the dictionary ("RomashkaPay", "KvantiksLab") — a brand, not code."""
    if not _CAMEL.fullmatch(word):
        return False
    words = 0
    for part in _CAMEL_PART.findall(word):
        if part.isdigit() or part.isupper() or len(part) <= 2:
            continue
        if not _english_word(lexicon, part):
            return False
        words += 1
    return words > 0


def _id_like(word: str) -> bool:
    """A hex id or a mostly-digit token — not a name."""
    return bool(_HEX_ID.fullmatch(word)) or (
        bool(_ALNUM.fullmatch(word)) and 2 * sum(ch.isdigit() for ch in word) >= len(word))


def _public_word_keys() -> tuple:
    global _public_keys
    if _public_keys is None:
        _public_keys = (frozenset(_org_key(w) for w in _ORG_PUBLIC_WORDS),
                        frozenset(_org_key(p) for p in _ORG_PUBLIC_PHRASES))
    return _public_keys


def _org_public(core: str) -> bool:
    """Not the customer's data: a well-known international tool/platform (the owner's list,
    lexicon.is_owner_tool), a phrase of _ORG_PUBLIC_PHRASES, or every word is a tech term (lexicon.TECH_TERMS),
    a word of _ORG_PUBLIC_WORDS (with a model tail: "СВК-20") or an id."""
    words, phrases = _public_word_keys()
    if _org_key(core) in phrases:
        return True
    lexicon = _lexicon()
    if lexicon is not None and _dict_check(lexicon.is_owner_tool, core):
        return True
    tech = lexicon.TECH_TERMS if lexicon is not None else frozenset()

    def public(w: str) -> bool:
        if w.casefold() in tech or _id_like(w):
            return True
        return _org_key(w) in words or _org_key(_MODEL_TAIL.sub("", w)) in words
    return all(public(w) for w in core.split())


def _code_shaped(core: str) -> bool:
    """A word of code and logs: one English dictionary word that is not a proper name ("Structure",
    "resolved"; lexicon.word_kind), or every word is a code identifier (_code_identifier) or a byte of a hex
    dump. In logs and CSV it is not a name; in prose a brand or an event may look so ("на HighLoad",
    "Talent"): model_org_pack drops it by where it lives (_prose_share)."""
    lexicon = _lexicon()
    if lexicon is None:
        return False
    if _LATIN_WORD.fullmatch(core) and _dict_check(lambda: lexicon.word_kind(core) == "plain"):
        return True
    return all(_HEX_BYTE.fullmatch(w) or _dict_check(_code_identifier, lexicon, w) for w in core.split())


def _org_key(core: str) -> str:
    """Grouping key of a name: case-insensitive, Cyrillic words by stem, a Russian ending after Latin
    dropped ("Ромашке" = "Ромашка", "Slackом" = "Slack")."""
    out = []
    for w in core.casefold().split():
        if re.search(r"[a-z]", w):
            w = re.sub(r"[а-яё]+$", "", w)
        elif _HAS_CYR.search(w) and len(w) > 3:
            w = cyr_stem(w)
        out.append(w)
    return " ".join(out)


_SUBWORD = re.compile(r"[^\W_]+")
# field separators of CSV rows, log lines, key=value pairs and JSON: the words between them are words of the
# text (the model reads them, the rules mask them there); a link, a path, a login stay one compound token
_FIELD_SEP = re.compile(r"[,;=|:\"«»“”„()\[\]{}<>]+")


def _word_key(w: str) -> tuple:
    """(word key, keys of the parts of a compound token) of one space-split word (_word_keys)."""
    core = w.strip(_ORG_EDGE)
    key = _org_key(core) if core else ""
    parts = _SUBWORD.findall(core)
    sub = ({_org_key(p) for p in parts} - {key, ""}) if len(parts) > 1 else set()
    return key, tuple(sorted(sub))


def _word_key_batch(_state, task) -> tuple[dict, dict, dict]:
    """Counts of one batch of distinct words ``[(word, count, count in prose lines)]`` by word key (``need`` —
    only these keys; None — all) → (as a word, inside a compound token, as a word in prose lines)."""
    pairs, need = task
    words: dict = {}
    inner: dict = {}
    prose: dict = {}
    for w, c, pc in pairs:
        key, sub = _word_key(w)
        if key and (need is None or key in need):
            words[key] = words.get(key, 0) + c
            if pc:
                prose[key] = prose.get(key, 0) + pc
        for k in sub:
            if need is None or k in need:
                inner[k] = inner.get(k, 0) + c
    return words, inner, prose


_KEY_BATCH = 20000           # distinct words per pool task


def _word_keys(lines, workers: int = 1, need=None) -> tuple[dict, dict, dict]:
    """Occurrences over the lines: ({word key: as a word}, {word key: inside a compound token}, {word key: as
    a word in prose lines, _PROSE}) — words split by spaces and field separators (``_FIELD_SEP``: a CSV cell,
    a log field, a JSON value is a word), edge punctuation stripped like the core of a name (_org_key of one
    word); the parts of a compound token (a link, a path, a login, ``a_b``, ``a.b``) are its letter-digit
    runs. ``need`` — count only these keys (the words of the found names). The words are counted first
    (Counter); the keys of the distinct words — in up to ``workers`` forked processes (a corpus has a million
    distinct words: 20 s of one core)."""
    from collections import Counter
    cnt: Counter = Counter()
    pcnt: Counter = Counter()
    for line in lines:
        ws = _FIELD_SEP.sub(" ", line).split()
        cnt.update(ws)
        if _PROSE.search(line) and not _STRUCTURED.search(line):
            pcnt.update(ws)
    pairs = [(w, c, pcnt.get(w, 0)) for w, c in cnt.items()]
    need = None if need is None else frozenset(need)
    batches = [(pairs[i:i + _KEY_BATCH], need) for i in range(0, len(pairs), _KEY_BATCH)]
    words: dict = {}
    inner: dict = {}
    prose: dict = {}
    from . import engine_adapter as EA
    got = EA.fork_map(_word_key_batch, batches, workers, label="org keys")
    for parts in got:
        for acc, part in zip((words, inner, prose), parts):
            for k, c in part.items():
                acc[k] = acc.get(k, 0) + c
    return words, inner, prose


def _occurrences(occ: tuple, key: str) -> int:
    """Occurrences of a name for the share check — of its rarest word, as a word and inside links, paths,
    logins and code identifiers: the rules mask it there too, and the model hardly ever tags a word inside a
    link — a name that lives mostly there (a hash fragment, a path part) is not one."""
    words, inner, _prose = occ
    return min(words.get(w, 0) + inner.get(w, 0) for w in key.split())


def _prose_share(occ: tuple, key: str) -> float:
    """The share of a name's occurrences (as a word and inside compound tokens) that are words of prose lines."""
    words, inner, prose = occ
    total = sum(words.get(w, 0) + inner.get(w, 0) for w in key.split())
    return sum(prose.get(w, 0) for w in key.split()) / total if total else 0.0


def model_org_pack(mentions, persons=(), lines=None, min_share: float | None = None,
                   prefix: str = "org:model", workers: int = 1, text_lines=None) -> Pack:
    """Organization names the name model found {surface: frequency} → rules of the contractor kind
    (``_term_rules``: Cyrillic case forms, Latin with a Russian ending, a common word only
    Capitalized/CAPS or in quotes/after a legal form, any case in a host/path). One name — one ORG
    token, the seed is the most frequent spelling of the name. A common-word or short Latin name needs
    ``ORG_MIN_FINDS`` finds: one find of «Мир» at a sentence start must not mask every «Мир». Dropped:
    a well-known tool, a tech or billing term, an id (_org_public), a name the model also found as a
    person (``persons`` — person mentions; the whole mention or one of its words: the person rules mask it
    with the person's token) and a code-shaped name (_code_shaped) that lives outside prose — less than
    ``CODE_PROSE_SHARE`` of its occurrences in prose lines of ``lines`` (no lines — always). With ``lines``
    (the lines the model read) a name is also dropped when the model tagged it in less than ``min_share``
    (default ``ORG_MIN_SHARE``) of its occurrences (with those inside links and paths, _occurrences): a word
    seen hundreds of times and tagged a few times is a service or a common word, not a name. With
    ``text_lines`` (those of ``lines`` that come from text views: chats, documents, prose files) a name
    that occurs in ``lines`` but has fewer than ``ORG_MIN_TEXT`` occurrences there is dropped too — found
    only in CSV cells, log fields and data JSON, it is a code or a category value; a name that also
    occurs in text is masked everywhere, in the data files too. The check forms are dropped: the model's names are not the
    contractor's details. ``workers`` — processes for the word keys of the share check (_word_keys)."""
    groups: dict = {}
    stats = {"mentions": 0, "dropped": 0, "names": 0, "rare_common": 0, "public": 0, "persons": 0,
             "code": 0, "data_only": 0, "low_share": 0}
    share = ORG_MIN_SHARE if min_share is None else min_share
    person_keys = {_org_key(p) for p in persons} | {_org_key(w) for p in persons for w in p.split()}
    for surface, n in sorted(mentions.items()):
        stats["mentions"] += n
        core = _org_core(surface)
        if not core:
            stats["dropped"] += n
            continue
        if _org_public(core):
            stats["public"] += n
            continue
        if _org_key(core) in person_keys:
            stats["persons"] += n
            continue
        groups.setdefault(_org_key(core), {}).setdefault(core, 0)
        groups[_org_key(core)][core] += n
    code = {key for key, spell in groups.items() if all(_code_shaped(t) for t in spell)}
    occ = text_occ = None
    need = {w for key in groups for w in key.split()}
    if lines is not None and (share > 0 or code or text_lines is not None):
        occ = _word_keys(lines, workers, need=need)
    if lines is not None and text_lines is not None:
        text_occ = _word_keys([ln for ln in lines if ln in text_lines], workers, need=need)
    pack = Pack()
    for i, (_key, spell) in enumerate(sorted(groups.items())):
        total = sum(spell.values())
        terms = sorted(spell)
        if total < ORG_MIN_FINDS and any(term_policy(t) != LONG for t in terms):
            stats["rare_common"] += 1
            continue
        if _key in code and (occ is None or _prose_share(occ, _key) < CODE_PROSE_SHARE):
            stats["code"] += 1
            continue
        # no occurrence counted at all (found inside a glued word): nothing to judge by, like the share check
        if text_occ is not None and _occurrences(occ, _key) and _occurrences(text_occ, _key) < ORG_MIN_TEXT:
            stats["data_only"] += 1
            continue
        if occ is not None and share > 0 and total < share * _occurrences(occ, _key):
            stats["low_share"] += 1
            continue
        seed = max(terms, key=lambda t: (spell[t], -len(t), t))
        _term_rules(pack, f"{prefix}:{i}", terms, seed, "ner.org", controls=False)
        stats["names"] += 1
    pack.counts = stats
    return pack


COMMON, DEV, LONG, SHORT = "common", "dev", "long", "short"


def term_policy(term: str) -> str:
    """How a term is masked — one policy for the rule and the check form (_term_rules):
    COMMON — a common word (Capitalized/CAPS, lowercase — only in a host/path); DEV — a long dev
    word (Capitalized/CAPS, in a host/path — also only those); LONG — a long or Cyrillic term, any
    case; SHORT — a short (≤3) Latin one: as written, CAPS — outside a host/path. All but LONG also
    get any case in quotes and after a legal form."""
    if common_word(term) or (dev_word(term) and len(term) > 3):
        return DEV if dev_word(term) else COMMON
    if len(term) > 3 or _HAS_CYR.search(term):
        return LONG
    return SHORT


def control_forms(term: str, policy: str) -> list:
    """Check forms of a term [(form, mode)] — the same case as the rule of the ``policy``:
    the gate finds exactly what the rule had to mask."""
    sql = sql_keyword(term)
    out = []
    for f in word_forms(term):
        if policy in (COMMON, DEV):
            for c in (f[:1].upper() + f[1:], f.upper()):
                out.append((c, SQL_CASED if sql else True))
                if policy == DEV:
                    out.append((c, GLUED_CASED))
            if policy == COMMON:
                out.append((f, GLUED))
        elif policy == SHORT:
            out.append((f, SQL_CASED if sql else True))
            if f.upper() != f:
                out.append((f.upper(), SHORT_UPPER))
        else:
            out.append((f, SQL_ANY if sql else False))
        if policy != LONG:
            out += [(f, QUOTED), (f, OPF)]
    return out


def _term_rules(pack: Pack, name: str, terms: list, seed: str, kind: str, controls: bool = True) -> int:
    """Rules and check forms of a group of terms with one token. A common-word term (common_word)
    gets separate case-sensitive rules and Capitalized/CAPS check forms; returns their number.
    A long dev word (dev_word) — like a common word, but the lowercase form is not glued: in a
    host/path — only Capitalized/CAPS (check forms GLUED_CASED, no GLUED). ``controls=False`` —
    rules only (the model's names have no check forms)."""
    policy = {t: term_policy(t) for t in terms}
    common = [t for t in terms if policy[t] in (COMMON, DEV)]
    dev = [t for t in terms if policy[t] == DEV]
    long_terms = [t for t in terms if policy[t] == LONG]
    short = [t for t in terms if policy[t] == SHORT]
    glue = [t for t in long_terms + common if t not in dev]
    if long_terms:
        pack.rules.append(_org_rule(name, long_terms, seed))
    if short:
        pack.rules.extend(_short_rules(name + ":short", short, seed))
    if common:
        pack.rules.extend(_common_rules(name + ":word", common, seed))
    if glue:
        pack.late.append(_glued_rule(name + ":glued", glue, seed))
    if dev:
        pack.late.append(_glued_rule(name + ":glued_cased", dev, seed, cased=True))
    if not controls:
        return len(common)
    for t in terms:
        for f, mode in control_forms(t, policy[t]):
            if (f, kind, mode) not in pack.controls:
                pack.controls.append((f, kind, mode))
    return len(common)
