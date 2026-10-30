"""Organisation / brand / client detectors.

Token = ``mint.tag("ORG", name)`` (deterministic → one entity collapses to one
token corpus-wide, so every spelling / inflection / duplicate canonicalises for
free). Domains mint into the ``RESOURCE`` range instead.

Confidence discipline (the whole point of the pack):
  * ``high``    — self-evident structure: a legal form / trailing latin legal suffix /
                  a multi-label domain. The legal form or the TLD IS the
                  evidence, so no nearby label is needed.
  * ``context`` — a «quoted name» or a #hashtag that only reads as an org when an
                  org-context word sits in the left window, plus the inverted
                  "ambiguous brand unless a lowercase common-noun form is near"
                  gate, plus every parameterised brand-variation pack.
  * ``low``     — the mixed-script stylization sweep (very FP-prone) — off by
                  default and column-scoped.

No corpus-specific literals live here (no client names, no brands). The
``brand_variation`` factory, ``ambiguous_name_negctx`` factory and
``company_column_mask`` helper are GENERAL mechanisms; concrete brand stems /
column names come from an optional ``known_values`` pack, never from this module.
"""
from __future__ import annotations

import re

from ..engine import mint, placeholders as P  # noqa: F401  (P kept for parity)
from ..engine.boundary import (L, R, ND, CL, CR, TOKEN_RE, _PREFIX_ALT, alt_longest_first,
                               is_own_token, DASHES, DASH_CLASS, HWS_CLASS, GLUE_L, GLUE_R,
                               GLUE_TAIL)
# "already a token" guards — boundary.is_own_token: only our token, not any CAPITALS_digits
from ..engine.rule import Rule, no_guard
from ..engine.vocab import (
    LEGAL_FORMS,
    LEGAL_SUFFIX_LATIN,
    GENERIC_DESCRIPTORS,
    ORG_CONTEXT,
    UI_LABEL_STOP,
    TECH_TERMS,
    COMMON_EN,
    HOMOGLYPH,
    HOUSING_FORMS,
)

# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
# Legal-form alternation, longest-first so "АО" can't pre-empt "ЗАО"/"НАО". We keep the
# case-sensitivity of the source (legal forms are all-caps RU or Capitalised latin).
# + housing association forms (vocab.HOUSING_FORMS): «ТСЖ «Наш дом»» in text
# and in a registry cell — one ORG by the name without a legal form
_OPF = alt_longest_first(LEGAL_FORMS + HOUSING_FORMS)

# LEGAL_SUFFIX_LATIN mixes REAL legal suffixes (Inc/Ltd/GmbH — evidence of an organisation
# by themselves) and GENERIC words (Tech/Digital/Software/Systems/Labs/
# Studio/Agency — parts of ordinary English phrases: "Big Tech", "Open Source
# Software"). Generic ones as a high suffix gave a false ORG on an everyday phrase.
# Split: legal suffixes stay high, generic ones go to
# a separate rule confidence='context' with requires_context=ORG_CTX_RE.
_GENERIC_SUFFIX_WORDS = {"Tech", "Digital", "Software", "Systems", "Labs", "Lab",
                         "Studio", "Studios", "Agency", "Group"}
_TRUE_LEGAL_SUFFIX = [s for s in LEGAL_SUFFIX_LATIN if s not in _GENERIC_SUFFIX_WORDS]
_GENERIC_SUFFIX = [s for s in LEGAL_SUFFIX_LATIN if s in _GENERIC_SUFFIX_WORDS]
# The first word of a generic phrase that is NOT a brand: "Big Tech", "Open
# Source", "New Digital"… — extends the stop guard of the generic rule.
_GENERIC_FIRST_STOP = {"big", "open", "new", "best", "high", "low", "old", "hard",
                       "soft", "deep", "core", "full", "next", "smart", "source"}
# ИП + Surname is a PERSON, not an ORG — route it away from opf_name.
_IP_RE = re.compile(r"^(?:ИП|IP)$")

# Horizontal whitespace between the words of an organisation name:
# `\s` also matched a line break — «Acme\nLtd», «ООО Ромашка\nИванов» glued
# adjacent lines/CSV records into one ORG token, the content between them
# disappeared. [^\S\r\n] — any whitespace EXCEPT \r/\n (unlike [ \t]
# it keeps NBSP/thin spaces of Word/Confluence exports «ООО\xa0«Ромашка»»).
_HS = r"[^\S\r\n]"
# a Capitalised name token (RU or latin caps head + word tail) OR a short number.
_CAPTOK = r"(?:[А-ЯЁA-Z][\w-]{1,24}|\d{1,4})"
# small link-words allowed *between* capitalised words inside an org name.
_LINK = r"(?:для|и|на|с|по|плюс|the|of|and|&)"
# the org-name body: 1..4 capitalised tokens joined by spaces / link-words
# (never across a newline — see _HS).
_NAME = _CAPTOK + r"(?:" + _HS + r"+(?:" + _LINK + _HS + r"+)?" + _CAPTOK + r"){0,3}"

# names/descriptors that must never *be* the masked org (they're structure).
_NAME_STOP = {w.upper() for w in GENERIC_DESCRIPTORS} | {
    "ФИРМА", "КОМПАНИЯ", "ПРИМЕР", "ТЕСТ", "ID", "ИМЕНИ", "EXAMPLE", "TEST",
    "SAMPLE", "DEMO", "COMPANY", "NAME",
    # «ГК РФ» is the Civil Code, not «a group of companies РФ»: after a legal form
    # the name «РФ»/«РСФСР» is never an organisation name (law abbreviations
    # ГК/УК/НК/ТК/КоАП РФ; of LEGAL_FORMS only «ГК» actually collides
    # with this tail — «согласно ст. 10 ГК РФ» → «ГК ORG_…»).
    "РФ", "РСФСР",
}

# right-of-brand-word that is already a domain-owner token: `.ORG_1234`
_ORG_TAIL = r"(?=\.ORG_\d)"
# roots of package/domain names (Java/Kotlin/Android: com.yandex.mapkit, org.jetbrains…)
_PKG_ROOTS = frozenset({"org", "com", "ru", "net", "io", "edu", "gov", "info", "biz", "me",
                        "co", "dev", "app", "www"})

ORG_CTX_RE = re.compile(ORG_CONTEXT)

# multi-label domain (a real TLD is the evidence). The registrable label run
# must contain a letter so "1.2.3.io" numeric noise doesn't slip through
# (the letter check is in _domain_guard: «1.2.3.dev» is a version in a tech log, not a domain).
# Compound zones (co.uk, com.tr…) stand BEFORE short TLDs, otherwise «example.co.uk»
# would match «example.co» (co as the TLD) and leave a «.uk» stub.
_TLD_COMPOUND = (r"co\.uk|org\.uk|gov\.uk|ac\.uk|com\.tr|com\.br|com\.au|co\.jp|"
                 r"co\.in|com\.ua|com\.ru|net\.ru|org\.ru")
# CIS + EU ccTLDs (Presidio research). Collisions with file extensions:
# .md (Markdown: README.md in dev chats all the time) — a domain only with URL
# evidence; .pl/.am (Perl, Automake — rare; domains sklep.pl / firma.am
# matter more) — domains; .py/.sh — not ccTLDs from the list, not added. Locales in
# dotted names (strings.de.json) and a path (docs/x.de) — in _domain_guard.
_TLD_CC = (r"by|kz|am|ua|ge|uz|az|md|kg|de|fr|pl|it|es|nl|cz|sk|lt|lv|ee|fi|se|"
           r"no|dk|tr|il|ae")
_TLD = (r"(?:" + _TLD_COMPOUND + r"|tech|dev|ai|io|ru|com|net|org|app|cloud|info|"
        r"biz|store|online|site|pro|xyz|me|co|su|team|studio|agency|digital|"
        r"software|cc|" + _TLD_CC + r")")
_CC_TLD = frozenset(_TLD_CC.split("|"))
# ccTLD = file extension: a domain only with URL evidence — a scheme "://"
# on the left or a subdomain (≥3 labels: www.example.md, mail.gov.md); a bare «name.md»
# is a file.
_EXT_TLD = frozenset({"md"})
_DOMAIN = r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+" + _TLD

# A Cyrillic Runet domain (ромашка.рф / сайт.рус / москва.москва): the ASCII branch
# of _DOMAIN does not see it. The registrable label is Cyrillic.
_TLD_CYR = r"(?:рф|рус|дети|москва|онлайн|сайт)"
_DOMAIN_CYR = r"(?:[а-яё0-9](?:[а-яё0-9-]*[а-яё0-9])?\.)+" + _TLD_CYR

# "Does the domain end with a compound zone" — for _domain_guard (how many trailing
# labels count as the TLD when looking for the registrable label).
_COMPOUND_ZONES = re.compile(r".*\.(?:" + _TLD_COMPOUND + r")$")


def _homoglyph_canon(s: str) -> str:
    """Fold Cyrillic homoglyphs to latin so a stylized brand seeds one token."""
    return "".join(HOMOGLYPH.get(c, HOMOGLYPH.get(c.lower(), c)) for c in s).lower()


# The right Cyrillic branch of the brand glue rule — a CamelCase tail with
# a Capital made only of twin letters (vocab.HOMOGLYPH, both cases: the engine's twin layer
# shows the rules «РомашкаBot» as «РомашкаВот», and the branch must take nothing else in Cyrillic
# on the right): «АльфаБанк» with approved «Альфа» — «Б» is not a twin, the branch stays silent, the second brand
# gets its own token. A tail of twins only («Маркет», «Авто», «Тех») is indistinguishable from Latin
# by look — such an approved neighbour (packgen.expand_org, glue_camel_twin) takes the glue key away from the head.
_GLUE_TWIN_LC = "".join(sorted(HOMOGLYPH))
_GLUE_TWIN_UC = _GLUE_TWIN_LC.upper()
_GLUE_CAMEL_TWIN = (r"[" + _GLUE_TWIN_UC + r"](?:[" + _GLUE_TWIN_LC + _GLUE_TWIN_UC + r"]|"
                    + GLUE_TAIL + r"){0,23}")
_GLUE_CAMEL_TWIN_RE = re.compile(_GLUE_CAMEL_TWIN)


def glue_camel_twin(tail: str) -> bool:
    """The tail ``tail`` (the rest of an approved neighbour after the brand stem/form) the right branch of the glue
    rule would take whole: CamelCase of twin letters only (+ digits/Latin/"_")."""
    return bool(tail) and _GLUE_CAMEL_TWIN_RE.fullmatch(tail) is not None


# --------------------------------------------------------------------------- #
# client's own placeholders — «Клиент_3», «Заказчик 2», «Исполнитель 17»
# --------------------------------------------------------------------------- #
# The client "anonymized" the export with role designations itself (owner's decision 19.10.2026, a client
# export: CSV, task texts, JSON, docx edit/comment author attributes). The pipeline
# replaces them with our ORG token — a general mechanism for all clients, not a script for one.
# Token: mint.tag("ORG", "<canon>_<N>"), canon — the word lowercase, a typo → the word, N — digits as in
# the text (leading zeros intact): «Заказчик 3», «ЗАКАЗЧИК_3» and «заказчик_3» — one token. Exactly these
# tokens were issued manually on 19.10 — they must not change.
# Forms: (word)_N — any case, N — any digits; (Word|WORD) N (one space: ordinary,
# non-breaking U+00A0 or narrow U+202F) — only Capitalised or in CAPS and N of 1–3 digits: lowercase
# «заказчик 2 раза» is an ordinary phrase, «Исполнитель 4713» is a task number. Shield of the space form:
# right after the number a numeral ending via a hyphen — 1–3 lowercase
# Cyrillic letters up to a non-letter («Исполнитель 12-го», «Клиент 3-й», «Заказчик 12-ти», «Клиент
# 2-ух», «Клиент 1-ый») — or the word «раз»/«раза» («Клиент 3 раза написал») — a number, not a designation;
# «Клиент 3-город» (letters further on) — a designation; the "_" form has no shield.
# Boundaries (Unicode classes): on the left and right not a letter/digit/"_" of any script
# («Клиент_3ї», «Клиент_3é», «Клиент_3ß» intact; a raw JSON escape `\n`/`\t`/`\uXXXX` on the left is
# a boundary; a word entirely in `\uXXXX` is visible through the transport view of .json/.jsonl). «№», a hyphen,
# inflections, Latin homoglyphs and `\uXXXX` in .txt are not touched (owner's rule).
# The word list is in one place: the next case («Подрядчик_N», «Контрагент_N») is a line in
# CLIENT_ROLE_WORDS (+ the typo canon in CLIENT_ROLE_CANON if needed).
# The same seed for all paths that mint ORG by a literal themselves (one designation — one token):
# a file/directory name (filenames.mask_name — there "_" and a space are name separators, boundaries without
# "_"; the approved rename-pass map and group assembly — round._name_token, designation forms in a name
# by the same boundaries: round._mask_name_forms), a cell of a tabular registry column
# (structured.tabular.cell_value — designations with their own token, the rest of the cell — as a value of the column
# kind: client_roles_split), the pack column rule and the CRM column (column_cell_fn/
# company_cell_fn — before the column minter, each designation with its own token), people fields of structured
# profiles (youtrack.login_token, entities.Registry.user_token/name_token — the whole value or
# a designation inside the value: «Заказчик 2 (ЛПР)» → ORG token + the rest via the field's usual path),
# the fix layer and the edit during group assembly (tools/llm_hotfix._mint → group_fix.engine_token,
# round._name_text_token), packgen (an approved literal of such a form gets no rule — it is
# covered by this rule; the pack's ENGINE_COVERED counter, the ru_trace dictionary has nothing to do with it).
CLIENT_ROLE_WORDS = ("клиент", "заказчик", "исполнитель", "испольнитель")
CLIENT_ROLE_CANON = {"испольнитель": "исполнитель"}
_CR_ALT_CI = "|".join(sorted(CLIENT_ROLE_WORDS, key=len, reverse=True))
_CR_ALT_CAP = "|".join(f"{w.capitalize()}|{w.upper()}"
                       for w in sorted(CLIENT_ROLE_WORDS, key=len, reverse=True))
_CR_SP = "[ \u00A0\u202F]"
# shield of the space form: «12-го», «3-й», «12-ти», «2-ух», «1-ый», «3 раза» — a number (a numeral
# ending via a hyphen: 1–3 lowercase Cyrillic letters up to a non-letter)
_CR_SHIELD = (r"(?!-[а-яё]{1,3}(?!\w))(?!" + _CR_SP + r"раза?(?!\w))")
_CR_BODY = (r"(?:(?P<w1>(?i:" + _CR_ALT_CI + r"))_(?P<n1>[0-9]+)"
            r"|(?P<w2>" + _CR_ALT_CAP + r")" + _CR_SP + r"(?P<n2>[0-9]{1,3})" + _CR_SHIELD + r")")
# on the left "\\" is also a boundary (an escaped "\\" of raw JSON before the word): a match cannot start
# on an escape letter — the body starts with a list word. \w — a letter/digit/"_" of any script
_CR_TEXT_L = r"(?:(?<=\\[ntrbfv])|(?<=\\u[0-9A-Fa-f]{4})|(?<!\w))"
_CR_TEXT_R = r"(?!\w)"
CLIENT_ROLE_RE = re.compile(_CR_TEXT_L + _CR_BODY + _CR_TEXT_R)
# a file/directory name: "_" is a name segment separator («Отчёт_Исполнитель_12_v2.docx»);
# [^\W_] — a letter/digit of any script without "_"
CLIENT_ROLE_NAME_RE = re.compile(r"(?<![^\W_])" + _CR_BODY + r"(?![^\W_])")
_CR_FULL_RE = re.compile(_CR_BODY)
# a list cell: designations joined by separators without other letters/digits
_CR_REST_RE = re.compile(r"[^\W_]")
CLIENT_ROLE_NEEDS = re.compile(r"(?i)(?:" + _CR_ALT_CI + r")[_ \u00A0\u202F][0-9]")


def client_role_seed_of(word: str, n: str) -> str:
    """ORG token seed: the canonical word lowercase + "_" + digits as in the text."""
    w = word.lower()
    return CLIENT_ROLE_CANON.get(w, w) + "_" + n


def client_role_token(m) -> str:
    return mint.tag("ORG", client_role_seed_of(m.group("w1") or m.group("w2"),
                                              m.group("n1") or m.group("n2")))


def client_role_seed(value: str):
    """ORG token seed if ``value`` (without edge whitespace) is entirely a client role
    designation; otherwise None. For paths that mint a token by a literal (fix layer, group assembly,
    nsname2, packgen)."""
    m = _CR_FULL_RE.fullmatch((value or "").strip())
    if m is None:
        return None
    return client_role_seed_of(m.group("w1") or m.group("w2"), m.group("n1") or m.group("n2"))


def sub_client_roles(text: str, names: bool = False) -> str:
    """All client role designations in ``text`` → ORG tokens (``names`` — file/directory
    name boundaries: "_" is a separator)."""
    rx = CLIENT_ROLE_NAME_RE if names else CLIENT_ROLE_RE
    return rx.sub(client_role_token, text)


def client_roles_cell(value: str):
    """A cell/value made only of client role designations (one or a list joined by
    separators: «Исполнитель 1, Исполнитель 4») → the same with tokens; otherwise None."""
    s = (value or "").strip()
    if not s or CLIENT_ROLE_NEEDS.search(s) is None:
        return None
    rest = CLIENT_ROLE_RE.sub("", s)
    if rest == s or _CR_REST_RE.search(rest) is not None:
        return None
    return CLIENT_ROLE_RE.sub(client_role_token, s)


# the core of a rest between designations: from the first to the last letter/digit (edge separators,
# brackets and spaces — outside, verbatim)
_CR_CORE_RE = re.compile(r"[^\W_](?:.*[^\W_])?", re.S)


def client_roles_split(value: str, rest_fn):
    """A value with client role designations INSIDE («Заказчик 2 (ЛПР)», «Исполнитель 1, Иванов
    Иван») → designations with their own ORG token (as in text), each rest with a letter/digit —
    ``rest_fn(core)`` (core — from the first to the last letter/digit of the rest; separators, brackets and
    spaces around — verbatim). None — there are no designations (the value goes the usual path whole).
    Used for people fields of profiles and cells of a tabular registry."""
    s = value or ""
    if not s or CLIENT_ROLE_NEEDS.search(s) is None:
        return None
    out, pos, found = [], 0, False
    for m in CLIENT_ROLE_RE.finditer(s):
        found = True
        out.append(_client_roles_rest(s[pos:m.start()], rest_fn))
        out.append(client_role_token(m))
        pos = m.end()
    if not found:
        return None
    out.append(_client_roles_rest(s[pos:], rest_fn))
    return "".join(out)


def client_roles_rest(value: str):
    """Cores of the value's rests between client role designations (see client_roles_split):
    a list of strings (empty — the value is only designations); None — there are no designations."""
    cores: list = []
    if client_roles_split(value, lambda core: cores.append(core) or core) is None:
        return None
    return cores


def _client_roles_rest(piece: str, rest_fn) -> str:
    m = _CR_CORE_RE.search(piece)
    if m is None:
        return piece
    return piece[:m.start()] + rest_fn(m.group()) + piece[m.end():]


# --------------------------------------------------------------------------- #
# mask helpers (each keeps its structural groups, masks only the name)
# --------------------------------------------------------------------------- #
def _split_leading_descriptors(name: str):
    """Split the name after a legal form into (leading descriptors verbatim, the rest of the name).
    «Компания Рога и Копыта» -> («Компания », «Рога и Копыта»). A descriptor
    (Компания/Группа/Фирма/…) is structure, kept verbatim; the REST is masked,
    otherwise the whole brand leaked as plain text."""
    parts = re.split(r"(\s+)", name)  # words and separators alternate
    kept, i = [], 0
    while i < len(parts):
        word = parts[i]
        if word and word.upper() in _NAME_STOP:
            kept.append(word)
            if i + 1 < len(parts):
                kept.append(parts[i + 1])  # the separator after the descriptor
            i += 2
        else:
            break
    rest = "".join(parts[i:])
    return "".join(kept), rest


def _opf_mask(m):
    """Keep the legal-form word + optional guillemets, mask only the trailing name.
    Leading descriptors (Компания/Группа/…) stay verbatim, only the brand rest
    is minted."""
    kept, rest = _split_leading_descriptors(m.group("name").strip())
    return (m.group("opf") + m.group("sp") + (m.group("oq") or "")
            + kept + mint.tag("ORG", rest.strip()) + m.group("cq"))


def _opf_guard(m):
    """True => leave untouched (skip)."""
    opf, name = m.group("opf"), m.group("name")
    if _IP_RE.match(opf):
        return True  # ИП <Surname> is a PERSON, handled by names.py
    if is_own_token(name, part=True):
        return True  # already contains an ORG_/PERSON_ token
    # Skip ONLY when ALL words of the name are descriptors (nothing to mask):
    # «ООО Компания» is structure entirely; «ООО Компания Рога и Копыта» — the brand
    # «Рога и Копыта» must be masked (before, head∈_NAME_STOP skipped the WHOLE
    # span, and the brand leaked).
    _, rest = _split_leading_descriptors(name.strip())
    return not rest.strip()


def _latin_suffix_guard(m):
    span = m.group(0)
    if is_own_token(span, part=True):
        return True
    first = m.group(1).split()[0].lower()
    return first in TECH_TERMS or first in COMMON_EN


def _generic_suffix_guard(m):
    """A generic "suffix" (Tech/Digital/…): the same first-word check, plus
    an extended stop of generic modifiers ("Big"/"Open"/"New"/…), so that "Big
    Tech"/"Open Source Software" do not become an ORG."""
    span = m.group(0)
    if is_own_token(span, part=True):
        return True
    first = m.group(1).split()[0].lower()
    return (first in TECH_TERMS or first in COMMON_EN
            or first in _GENERIC_FIRST_STOP)


def _quoted_mask(m):
    return m.group(1) + mint.tag("ORG", m.group(2).strip()) + m.group(3)


def _quoted_guard(m):
    name = m.group(2).strip()
    if not name or not (name[0].isupper() or name[0].isdigit()):
        return True
    low = name.lower()
    if low in UI_LABEL_STOP or low in TECH_TERMS or low in COMMON_EN:
        return True
    if is_own_token(name, part=True):
        return True
    if re.search(r"[?!;:]|\.\s", name):  # sentence punctuation => running prose
        return True
    return len(re.split(r"\s+", name)) > 3  # >3 words => not an org name


# (see contacts.ENV_PCT_URL): domain does not start right after "%" — «2Fvk.com» in
# «%2Fvk.com» is taken by domain_pct whole, without the escape digits


def _domain_guard_pct(m):
    i = m.start(1)
    if i > 0 and m.string[i - 1] == "%":
        return True
    return _domain_guard(m)


def _domain_guard(m):
    d = m.group(1).lower()
    if is_own_token(d, part=True):
        return True
    labels = d.split(".")
    if labels[-1] in _CC_TLD:
        s, i, e = m.string, m.start(1), m.end(1)
        scheme = s[max(0, i - 3):i] == "://"
        # «strings.de.json» — a locale in a dotted file name (one more dotted label
        # after the "TLD"); «docs/x.de» — a file name position in a path (not a URL)
        if (s[e:e + 1] == "." and s[e + 1:e + 2].isalnum()) \
                or (i > 0 and s[i - 1] == "/" and not scheme):
            return True
        if labels[-1] in _EXT_TLD and len(labels) < 3 and not scheme:
            return True          # README.md / CHANGELOG.md — a file
    # The TLD may be compound (co.uk) — the last 2 labels; otherwise 1. The registrable
    # label stands to the left of the TLD.
    tld_labels = 2 if _COMPOUND_ZONES.match(d) else 1
    idx = len(labels) - tld_labels - 1
    registrable = labels[idx] if idx >= 0 else labels[0]
    # the registrable label must contain a LETTER: «1.2.3.dev»/«1.2.io» are
    # versions in tech logs, not domains.
    if not re.search(r"[a-zа-яё]", registrable):
        return True
    # registrable name (label immediately left of the TLD) is a dev tool → keep.
    return registrable in TECH_TERMS or d in TECH_TERMS


# (owner 2026-10-06: a rule for all clients): a bare domain with a two-letter zone outside the list
# _TLD («xxxxxxxxxxxx.gg», «x9xxxx.us») was masked by nothing. Zones are ISO 3166 country codes (+uk/eu/ac),
# except those already known to the domain rule (its behaviour does not change) and those that coincide with a file extension
# or a code attribute: in id py sh so rs mk mo pm ml mm tf ps pk pt ax la bz cl sv re cr gp ss cv cf
# ci cd; and the English words do is to as my. Lowercase only (a capital after a dot without a space is
# the end of a sentence), the registrable label of 3+ chars with a letter (not «т.е.»/«e.g.»/«3.14»/«v1.2»).
_CC_ISO = ("ad ae af ag ai al am ao aq ar as at au aw ax az ba bb bd be bf bg bh bi bj bm bn bo br bs bt "
           "bw by bz ca cd cf cg ch ci ck cl cm cn co cr cu cv cw cx cy cz de dj dk dm do dz ec ee eg er "
           "es et eu fi fj fk fm fo fr ga gd ge gf gg gh gi gl gm gn gp gq gr gs gt gu gw gy hk hm hn hr "
           "ht hu id ie il im in io iq ir is it je jm jo jp ke kg kh ki km kn kp kr kw ky kz la lb lc li "
           "lk lr ls lt lu lv ly ma mc md me mg mh mk ml mm mn mo mp mq mr ms mt mu mv mw mx my mz na nc "
           "ne nf ng ni nl no np nr nu nz om pa pe pf pg ph pk pl pm pn pr ps pt pw py qa re ro rs ru rw "
           "sa sb sc sd se sg sh si sk sl sm sn so sr ss st sv sx sy sz tc td tf tg th tj tk tl tm tn to "
           "tr tt tv tw tz ua ug uk us uy uz va vc ve vg vi vn vu wf ws ye yt za zm zw ac").split()
_CC_COLLIDE = frozenset("in id py sh so rs mk mo pm ml mm tf ps pk pt ax la bz cl sv re cr gp ss cv cf "
                        "ci cd do is to as my "
                        # file extensions — autoconf, CUDA, netCDF, Apps Script, GDScript,
                        # TableGen, Smalltalk/templates, BitBake, Nextflow, Maya/numpy.ma, Scala worksheet
                        "ac cu nc gs gd td st bb nf ma sc".split())
_TLD_KNOWN = frozenset(t for t in (_TLD[3:-1]).split("|") if "\\" not in t)
_CC_BARE = tuple(z for z in _CC_ISO if z not in _TLD_KNOWN and z not in _CC_COLLIDE)
# a code identifier as the first label without a scheme — an attribute/module, not a domain («self.ws», «args.lr»,
# «self.conn.ws», «import numpy.ma»)
_CODE_IDENT = frozenset("self this cls args kwargs argv opts cfg conf config params props attrs ctx env "
                        "utils util numpy scipy pandas torch sklearn".split())


def _dict_host_part(labels) -> bool:
    """A host label or its hyphen-separated part is a form of the Russian trace dictionary that it
    masks itself in any case (an unambiguous entry or hom=nctx; a case-sensitive one — only
    the listed spelling): «tilda», «yerevan», «moscow». Such a host is handled by the dictionary and
    domain_display_* (a geo token and a brand token), as before."""
    from ..known_values import ru_trace as _rt
    if not _rt.enabled():
        return False
    for lab in labels:
        for part in [lab] + lab.split("-"):
            e = _rt.lookup(part)
            if e is None or e["flags"].get("hom") not in (None, "nctx"):
                continue
            if e["exact"] and part not in e["forms"]:
                continue
            return True
    return False


def _bare_cc_guard(m):
    """True => do not mask: our token, a registrable label shorter than
    3 chars / without a letter / an IT term; without a scheme — a code identifier as the first label and, for two labels without
    "www.", a dictionary English word (a file name/attribute: «kernel.cu», «self.ws»); a part of the host is
    a form of the Russian trace dictionary (the dictionary masks it)."""
    d = m.group(1)
    if TOKEN_RE.search(d):
        return True
    labels = d.split(".")
    reg = labels[-2]
    if len(reg) < 3 or not re.search(r"[a-z]", reg) or reg in TECH_TERMS or d in TECH_TERMS:
        return True
    i = m.start(1)
    if m.string[max(0, i - 3):i] != "://":
        if labels[0] in _CODE_IDENT:
            return True
        if len(labels) == 2:
            from ..engine import lexicon
            if reg in lexicon._en("words"):
                return True
    return _dict_host_part(labels[:-1])


def _hashtag_guard(m):
    body = m.group(1)
    if is_own_token(body, part=True):
        return True
    return body.lower() in TECH_TERMS or body.lower() in COMMON_EN


def _homoglyph_guard(m):
    tok = m.group(1)
    # must genuinely mix scripts inside one token (deliberate stylization).
    if not (re.search(r"[A-Za-z]", tok) and re.search(r"[А-Яа-яЁё]", tok)):
        return True
    return is_own_token(tok, part=True)


def _display_stem(m) -> tuple:
    """(the tail of our constant, the label). A match that started with "REDACTED" right after "_" stands on the tail
    of a constant («GEO_REDACTED.ORG_N», «GEO_REDACTED-tours.ORG_N» — part of the subdomain label is already covered
    by the dictionary): the tail is not a label, only what follows the hyphen remains the label. Before, the tail
    was re-minted into ORG together with the rest, and the geo token was lost."""
    stem = m.group(1)
    if stem.startswith("REDACTED") and stem[8:9] in ("", "-") \
            and m.string[max(0, m.start(1) - 1):m.start(1)] == "_":
        return stem[:9], stem[9:]
    return "", stem


def _display_guard(m) -> bool:
    """True => do not mask. A label before an owner domain («acme.ORG_N»): a service word
    and a package root are not a brand; an empty label (just a constant tail) — nothing to mask."""
    low = _display_stem(m)[1].lower()
    return not low or low in TECH_TERMS or low in COMMON_EN or low in _PKG_ROOTS


def _display_mask(m) -> str:
    keep, stem = _display_stem(m)
    return keep + mint.tag("ORG", stem)


# --------------------------------------------------------------------------- #
# RULES  (ordered most-specific / longest-first)
# --------------------------------------------------------------------------- #
RULES: list[Rule] = [
    # 0) a client's own placeholder («Клиент_3», «Заказчик 2», «Исполнитель 17») → our
    #    ORG token (see the CLIENT_ROLE_WORDS block). First in orgs: before legal form/brand/names, so that
    #    no other detector cuts the designation (the part after "_" is not an identifier).
    Rule(
        "client_role_placeholder", CLIENT_ROLE_RE,
        mask=client_role_token, confidence="high", needs=CLIENT_ROLE_NEEDS,
    ),

    # 1) legal form + Capitalised name — keep legal form+«», mask the name. High: the legal form is
    #    the evidence. The _NAME body only takes Capitalised words, so a
    #    lowercase running-prose tail ("ООО ТАВ висит товар") naturally stops.
    Rule(
        "opf_name",
        re.compile(
            # JSONL-aware left edge (the ``L`` idiom for this rule's own
            # exclusion set): a legal form right after an escaped \n still matches,
            # and the backslash exclusion keeps a match off the escape itself.
            r"(?:(?<=\\n)|(?<![А-Яа-яA-Za-z\\]))(?P<opf>" + _OPF + r")(?P<sp>" + _HS + r"+)"
            # a quoted name («Наш дом», «Рога и копыта») — up to the closing quote
            # whole, with lowercase words (up to 4 words after the Capitalised body); without quotes —
            # the old body of Capitalised words. The ORG seed is the whole name: the same token as a
            # registry cell (structured.tabular.org_token)
            # the quote group is optional OUTSIDE (`(…)?`, not `(…?)`): an empty match in Python
            # also "matched", and without a quote the branch `(?(oq)…)` ate up to 4 words up to the closing
            # quote of a raw JSON value (test_opf_quote_branch)
            r"(?P<oq>[«\"'])?(?P<name>" + _NAME
            + r"(?(oq)(?:(?:" + _HS + r"+[^\s«»\"']+){0,4}(?=[»\"']))?))(?P<cq>[»\"']?)" + ND
        ),
        mask=_opf_mask, guard=_opf_guard, confidence="high",
    ),

    # 2) latin org: 1..3 Capitalised words + a TRAILING TRUE legal suffix
    #    (Inc/Ltd/GmbH/LLC/…). High: the suffix IS the evidence; whole span → ORG.
    Rule(
        "legal_suffix_latin",
        re.compile(
            L + r"([A-Z][A-Za-z&.'-]{1,18}(?:" + _HS + r"+[A-Z][A-Za-z&.'-]{1,18}){0,2})"
            + _HS + r"+(?:" + alt_longest_first(_TRUE_LEGAL_SUFFIX) + r")\b\.?" + ND
        ),
        mask=lambda m: mint.tag("ORG", m.group(0)),
        guard=_latin_suffix_guard, confidence="high",
    ),

    # 2b) latin org with a GENERIC suffix (Tech/Digital/Software/Systems/…):
    #     a generic word is weak evidence (part of an ordinary English phrase), so
    #     'context' + an org context is required in the left window, and the first modifier word
    #     ("Big"/"Open"/…) is stopped. "Big Tech"/"Open Source Software" are no longer ORG.
    Rule(
        "generic_suffix_latin",
        re.compile(
            L + r"([A-Z][A-Za-z&.'-]{1,18}(?:" + _HS + r"+[A-Z][A-Za-z&.'-]{1,18}){0,2})"
            + _HS + r"+(?:" + alt_longest_first(_GENERIC_SUFFIX) + r")\b\.?" + ND
        ),
        mask=lambda m: mint.tag("ORG", m.group(0)),
        guard=_generic_suffix_guard,
        requires_context=ORG_CTX_RE, window=80, confidence="context",
    ),

    # 3) brand word glued to an already-tokenised owner domain: acme.ORG_1234
    #    → mask only the readable stem, keep the `.ORG_####` tail. High.
    Rule(
        "domain_display_vs_owner",
        re.compile(r"(?:(?<=\\n)|(?<![A-Za-z0-9\\]))"    # JSONL-escape-aware left boundary: do not start on the letter of \n
                   r"([A-Za-z][A-Za-z0-9-]{1,30})" + _ORG_TAIL),
        mask=_display_mask,
        # (owner's policy on identifiers): a package/domain root
        # before the token («com.ORG_1.mapkit», «org.ORG_2…») is not a brand: only
        # the trace itself is masked (yandex/jetbrains), «org»/«com»/«ru»/«net» — never
        guard=_display_guard,
        confidence="high",
        needs=".ORG_",           # the lookahead tail of the match (a probe cannot be derived from a lookahead)
    ),

    # 3a) part of a subdomain label is already covered by a dictionary token: «acme-yerevan.<site builder>» →
    #     «acme-GEO_REDACTED.ORG_N» (geo is in the dictionary). A whole label without geo is masked by
    #     rule 3, while here its rest stayed open — the client's site name (raw
    #     tg sources: one address, 20 occurrences; main masked the label whole).
    #     The rest is masked the same way as a whole label; the token in the label is left alone.
    Rule(
        "domain_display_part",
        re.compile(r"(?:(?<=\\n)|(?<![A-Za-z0-9_\\-]))"
                   r"([A-Za-z][A-Za-z0-9-]{1,30})"
                   r"(?=-(?:" + _PREFIX_ALT + r")_(?:\d{1,32}|REDACTED)"
                   r"(?:-[A-Za-z0-9_-]{1,40})?\.ORG_\d)"),
        mask=lambda m: mint.tag("ORG", m.group(1)),
        guard=_display_guard,
        confidence="high",
        needs=".ORG_",
    ),

    # 4) multi-label domain → RESOURCE token. High: a real TLD is the evidence.
    #    (email addresses are already masked upstream in contacts.py.)
    #    Latin and Cyrillic (Runet: .рф/.рус) branches; compound zones
    #    (co.uk) inside _DOMAIN stand before short TLDs (see _TLD).
    # 4-) a host right after a %XX escape («https%3A%2F%2Fvk.com%2F…») — whole, the escape intact
    #     (domain below started on «2F» and broke the escape: «%RESOURCE_…»).
    Rule(
        "domain_pct",
        re.compile(r"(?<=%[0-9A-Fa-f]{2})((?:" + _DOMAIN + r")|(?:" + _DOMAIN_CYR + r"))(?![\w-])",
                   re.IGNORECASE),
        mask=lambda m: mint.tag("RESOURCE", m.group(1).lower()),
        value_group=1, guard=_domain_guard,
        confidence="high", needs=re.compile(r"%[0-9A-Fa-f]{2}[\w-]+\.[\w.-]")),
    Rule(
        "domain",
        # JSONL-aware left edge; the backslash exclusion stops the match from
        # starting ON the 'n' of an escaped \n (which would corrupt the escape
        # and mint a wrong 'nexample.com' RESOURCE seed).
        re.compile(r"(?:(?<=\\n)|(?<![\w.@\\-]))((?:" + _DOMAIN + r")|(?:"
                   + _DOMAIN_CYR + r"))(?![\w-])", re.IGNORECASE),
        mask=lambda m: mint.tag("RESOURCE", m.group(1).lower()),
        value_group=1, guard=_domain_guard_pct, confidence="high",
        # the last label + TLD — the match must contain «.<tld>»
        # (the prefilter does not glue the dot from a repeat with the TLD branching); IGNORECASE
        # → comparison by casefold
        needs=tuple("." + t.replace(r"\.", ".")
                    for t in (_TLD[3:-1] + "|" + _TLD_CYR[3:-1]).split("|")),
    ),

    # 4a) a bare domain with a two-letter zone outside _TLD («xxxxxxxxxxxx.gg», «x9xxxx.us», and after
    #     «://») → RESOURCE with the same seed as domain. Not a domain: a file name in a path without a scheme
    #     («docs/x.gg»), a continuation with a dotted name («a.gg.json»), a code call/index («x.ch(»),
    #     file-extension zones (_CC_COLLIDE), a short/numeric label.
    Rule(
        "domain_bare_cc",
        re.compile(r"(?:(?<=://)|(?<=\\n)|(?<![\w.@\\/-]))((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
                   r"(?:" + "|".join(_CC_BARE) + r"))(?![\w(\[-]|\.[A-Za-z0-9])"),
        mask=lambda m: mint.tag("RESOURCE", m.group(1).lower()),
        value_group=1, guard=_bare_cc_guard, confidence="high",
        needs=re.compile(r"[a-z0-9]\.[a-z]{2}(?![a-z0-9])"),
    ),

    # 5) «quoted name» gated by org-context in the left window; keep «».
    #    context: too weak to fire on the guillemets alone (millions in JSON).
    #    Plain "..." quotes are deliberately NOT matched.
    Rule(
        "quoted_name_orgctx",
        re.compile(r"(«)([^«»\n]{2,45})(»)"),
        mask=_quoted_mask, guard=_quoted_guard,
        requires_context=ORG_CTX_RE, window=60, confidence="context",
        needs="«",               # the opening quote of the match
    ),

    # 6) #hashtag brand → keep the '#', mask the body. context.
    Rule(
        "hashtag_brand",
        re.compile(r"(?:(?<=\\n)|(?<![\w#\\]))#([A-Za-zА-Яа-яЁё][\w-]{2,30})" + R),
        mask=lambda m: "#" + mint.tag("ORG", m.group(1)),
        guard=_hashtag_guard,
        requires_context=ORG_CTX_RE, window=80, confidence="context",
        needs="#",               # the hash sign of the match
    ),

    # 7) mixed-script single token (deliberate brand stylization). low + should
    #    be column-scoped by the caller (set .scope); very FP-prone globally.
    Rule(
        "homoglyph_mixed_script",
        re.compile(r"(?:(?<=\\n)|(?<![\w@.\\]))"
                   r"([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9]{3,30})" + R),
        mask=lambda m: mint.tag("ORG", _homoglyph_canon(m.group(1))),
        guard=_homoglyph_guard, confidence="low",
    ),
]


# --------------------------------------------------------------------------- #
# ambiguous_name_negctx — INVERTED gate, exposed as a factory.
#   A common-noun-colliding brand ("Магнит"/"Лампа"): mask by DEFAULT, skip only
#   if a *lowercase* form of the same word appears within ±window chars (that
#   signals the common object, not the company). This needs the actual literal
#   (corpus data → a known_values pack), so we ship the MECHANISM and the guard,
#   not any specific word.
# --------------------------------------------------------------------------- #
def ambiguous_name_negctx(literal: str, *, window: int = 130, confidence: str = "context",
                          kind: str = "ORG", forms=None) -> Rule:
    """Rule that masks a Capitalised ``literal`` (+ RU inflection tail) unless a
    lowercase form of the same word is nearby — the ±window lowercase-form probe
    is the whole FP-guard. ``kind`` — the mint class (ORG by default;
    PERSON_NAME for single homograph names «Вера/Вер»).
    ``forms`` — an explicit list of EXACT forms instead of the open tail ``[а-яё]*``:
    for short homograph stems the tail ate other words («Вер[а-яё]*» →
    Вернусь/Версии/Верстал)."""
    esc = re.escape(literal)
    low = literal.lower()
    if forms:
        alt = "|".join(re.escape(f) for f in sorted(set(forms), key=len, reverse=True))
        rx = re.compile(CL + r"(" + alt + r")" + CR)
        # and the guard is exact: a lowercase form as a WHOLE word («вер»), not any
        # word starting with «вер…» («версии»/«вернусь» must not silence «Вер,»)
        low_alt = "|".join(re.escape(f.lower())
                           for f in sorted(set(forms), key=len, reverse=True))
        low_rx = re.compile(r"(?<![А-ЯЁA-Zа-яё])(?:" + low_alt + r")(?![а-яё])")
    else:
        rx = re.compile(CL + r"(" + esc + r"[а-яё]*)" + CR)
        low_rx = re.compile(r"(?<![А-ЯЁA-Z])" + re.escape(low) + r"[а-яё]*")

    def _guard(m):
        s, i, e = m.string, m.start(1), m.end(1)
        left, right = s[max(0, i - window):i], s[e:e + window]
        if is_own_token(m.group(1), part=True):
            return True
        # lowercase common-noun form nearby => it's the object, not the company.
        return bool(low_rx.search(left) or low_rx.search(right))

    return Rule(
        f"ambiguous:{low}", rx,
        mask=lambda m: mint.tag(kind, literal),
        value_group=1, guard=_guard, confidence=confidence,
    )


# --------------------------------------------------------------------------- #
# brand_variation — parameterised: one brand's many surface forms → ONE token.
#   Spec = {
#     "token_seed":     "acme",              # canonical seed → every form = 1 token
#     "latin":          ["Acme", "AcmeCorp"],# whole-word, case-insensitive literals
#     "cyr_stems":      ["акме", "экме"],    # inflected: stem + [а-яё]* tail
#     "require_substr": "кме",               # discriminating substring the match
#                                            #   MUST contain (brand stem, not a common word)
#     "caps_only":      False,               # True => latin literal must be ALL-CAPS
#     "exact_case":     False,               # True => latin literal as written (no IGNORECASE)
#     "ws_class":       False,               # a form's space — a run of whitespace without
#                                            #   a line break (boundary.HWS_CLASS)
#     "dash_class":     False,               # a form's hyphen — "-" or a Unicode dash
#                                            #   (boundary.DASH_CLASS)
#     "glue":           False,               # gluing the brand with a digit/Latin
#                                            #   («Ромашка2024», «РомашкаBot») — the same token
#   }
#   Everything here is GENERAL; the concrete spec is corpus data (known_values).
#   Rules of a multi-word brand (a space or a hyphen/dash in any form) are marked
#   rule.early = "cyr" | "latin" — Pipeline raises them before the detectors
#   orgs/names, so that «Директор Ромашкин Технолоджис» does not fall apart into a person and a tail:
#   Cyrillic ones — before the legal-form rule, the Latin one — after the domain rules
#   (domain*: a host with the brand in a label — RESOURCE whole). The flag
#   is derived from the spec, it is not part of rule_digest.
# --------------------------------------------------------------------------- #
def _class_escape(escaped: str, spec: dict) -> str:
    """An escaped alternation/stem with whitespace and dash classes by the spec keys:
    re.escape writes a space as «\\ », a hyphen as «\\-» — only those are substituted; without the keys —
    byte for byte the old pattern."""
    if spec.get("ws_class"):
        escaped = escaped.replace("\\ ", HWS_CLASS)
    if spec.get("dash_class"):
        escaped = escaped.replace("\\-", DASH_CLASS)
    return escaped


def _multiword(spec: dict) -> bool:
    forms = list(spec.get("latin", [])) + list(spec.get("cyr_stems", [])) \
        + list(spec.get("cyr_forms", []))
    return any(" " in f or "-" in f or any(d in f for d in DASHES) for f in forms)


# class folding in the require_substr guard: a whitespace run → a space, a dash → a hyphen
_HWS_RE = re.compile(HWS_CLASS)
_DASH_TR = str.maketrans({d: "-" for d in DASHES})


def brand_variation(spec: dict, *, confidence: str = "context") -> list[Rule]:
    seed = spec["token_seed"]
    # the mint seed may differ from the rule name seed — an organisation name from
    # a registry cell is minted by the name itself (like opf_name and the profile), the rule name is ASCII
    mseed = spec.get("mint_seed") or seed
    require = spec.get("require_substr")
    rules: list[Rule] = []
    early = _multiword(spec)

    # explicit whole-word latin literals only need the not-already-a-token guard.
    def _lit_guard(m):
        return is_own_token(m.group(1), part=True)

    # cyrillic inflected stems need the discriminating-substring guard too, so a
    # stem+[а-яё]* over-match on a common word (stem heading an ordinary word) is rejected.
    # With the ws_class/dash_class keys the match has a whitespace run (NBSP,
    # a double space, a tab) or a Unicode dash, while require_substr («ка гру», «та-со») has an ordinary
    # space and hyphen: without folding the guard silently dropped a class match, the pack promised a class
    # that was not there. We fold BOTH sides with the same class (a line break is not a space);
    # without the keys the comparison is as before.
    fold_ws, fold_dash = bool(spec.get("ws_class")), bool(spec.get("dash_class"))

    def _fold(s: str) -> str:
        if fold_ws:
            s = _HWS_RE.sub(" ", s)
        if fold_dash:
            s = s.translate(_DASH_TR)
        return s

    req_low = _fold(require.lower()) if require else None

    def _stem_guard(m):
        g = m.group(1)
        if is_own_token(g, part=True):
            return True
        return bool(require) and req_low not in _fold(g.lower())

    # latin whole-word literals (longest first so a short form can't pre-empt).
    # A Russian case tail of a Latin brand («в Slackом», «для Figmы»)
    # is part of the match and goes under the token WHOLE: the R boundary no longer
    # lets Cyrillic through on the right, otherwise «Slackом» would either leak whole
    # or (as before) give the welding «ORG_12ом» (deglue log: «ORG_4386-ом»).
    # exact_case: a short literal (≤3 letters, «Lee»/«Nil»)
    # matches only in the approved.tsv spelling — case-insensitively it ate
    # the word «lee» and `return nil` in code; caps_only — the old ALL-CAPS mode.
    lats = spec.get("latin", [])
    if lats:
        flags = 0 if (spec.get("caps_only") or spec.get("exact_case")) \
            else re.IGNORECASE
        rules.append(Rule(
            f"brand:{seed}:latin",
            re.compile(L + r"(" + _class_escape(alt_longest_first(lats), spec) + r")[а-яё]*" + R,
                       flags),
            mask=lambda m: mint.tag("ORG", mseed),
            value_group=1, guard=_lit_guard, confidence=confidence,
        ))

    # cyrillic inflected stems: CL stem[а-яё]* CR (right boundary cuts the
    # inflection that would flip the meaning: ромашк vs ромашковым).
    for stem in sorted(spec.get("cyr_stems", []), key=len, reverse=True):
        rules.append(Rule(
            f"brand:{seed}:cyr:{stem}",
            re.compile(CL + r"(" + _class_escape(re.escape(stem), spec) + r"[а-яё]*)" + CR),
            mask=lambda m: mint.tag("ORG", mseed),
            value_group=1, guard=_stem_guard, confidence=confidence,
        ))
    # cyrillic EXACT forms (no open tail): a short stem («Аль» from «Аля»,
    # «Дим» from «Дима») with the tail [а-яё]* ate Альвина/Альфа/Димитров — the magnitude
    # gate (156 hits with 9 expected). packgen puts such brands
    # into cyr_forms: an explicit alternation of case forms, longest first.
    forms = spec.get("cyr_forms", [])
    if forms:
        rules.append(Rule(
            f"brand:{seed}:cyr-exact",
            re.compile(CL + r"(" + _class_escape(alt_longest_first(forms), spec) + r")" + CR),
            mask=lambda m: mint.tag("ORG", mseed),
            value_group=1, guard=_stem_guard, confidence=confidence,
        ))
    # (owner's decision): gluing the brand with a digit or Latin — «Ромашка2024»,
    # «РомашкаBot», «2024Ромашка», «Ромашкой_2024» — ONE token of the same entity, whole
    # (the tails are part of the match: the token is not welded to a digit/letter — the GLUE_L/GLUE_R boundaries stand
    # outside). The glue key is written by packgen (ANON_PACKGEN_GLUE) only for a brand without negctx/require_substr/
    # exact_case/caps_only with a seed > 3 letters and a stem ≥ 4 — a short stem with a tail would fan out.
    # Cyrillic right on the left («ДоговорсРомашка») is another word, not taken: the tail is [0-9A-Za-z_]
    # and the boundary is outside. On the right of a Cyrillic brand — also a Cyrillic tail with a Capital made only of
    # twin letters (_GLUE_CAMEL_TWIN): lowercase Cyrillic on the right is impossible (it is eaten by the stem's case
    # tail), and the engine's twin layer shows the rules «РомашкаBot» as
    # «РомашкаВот» (B, o, t are twins) — without this branch gluing with Latin made only of twins
    # would stay open. Any CamelCase Cyrillic on the right used to eat another
    # approved brand («АльфаБанк» with «Альфа» — one token for two entities); «Б» is not a twin,
    # the twin branch does not take it; for a neighbour made only of twins («…Маркет») packgen does not write the glue key
    # for the head. Tails — GLUE_TAIL: a minted token is not part of a tail.
    # A Latin brand: the tail is only digits/"_" («Romashka2024»), letter gluing is not
    # taken — «Roma» in «Romashka». These rules are the last: ordinary forms and "_" glues
    # («Ромашка_2024» — stem + "_" boundary) are masked by the earlier rules with the same token.
    if spec.get("glue"):
        def _glue_guard(m):
            # a glue — at least one tail is non-empty; a pure word is the earlier rules' job (they
            # have already masked it, there is no repeated match); already a token — left alone
            if not (m.group(1) or m.group(3)):
                return True
            return is_own_token(m.group(0), part=True)
        cyr_alt = [_class_escape(re.escape(s), spec) + r"[а-яё]*"
                   for s in sorted(spec.get("cyr_stems", []), key=len, reverse=True)]
        if forms:
            cyr_alt.append(_class_escape(alt_longest_first(forms), spec))
        if cyr_alt:
            rules.append(Rule(
                f"brand:{seed}:glue",
                re.compile(GLUE_L + r"(" + GLUE_TAIL + r"{0,24})(" + "|".join(cyr_alt)
                           + r")(" + GLUE_TAIL + r"{0,24}|" + _GLUE_CAMEL_TWIN + r")" + GLUE_R),
                mask=lambda m: mint.tag("ORG", mseed),
                guard=_glue_guard, confidence=confidence,
            ))
        if lats:
            flags = 0 if (spec.get("caps_only") or spec.get("exact_case")) \
                else re.IGNORECASE
            rules.append(Rule(
                f"brand:{seed}:glue:latin",
                re.compile(GLUE_L + r"([0-9_]{0,24})(" + _class_escape(alt_longest_first(lats), spec)
                           + r")([0-9_]{0,24})" + GLUE_R, flags),
                mask=lambda m: mint.tag("ORG", mseed),
                guard=_glue_guard, confidence=confidence,
            ))
    if early:
        for r in rules:
            # "latin" — after the orgs domain rules, "cyr" — before the legal-form rule (Pipeline.__init__)
            r.early = "latin" if ":latin" in r.name else "cyr"
    return rules


# --------------------------------------------------------------------------- #
# company_column_mask — OPTIONAL structured-CSV pass.
#   Not a regex Rule; the engine's value-level CSV pass (engine.formats.
#   mask_csv_cells) drives it. We expose `company_cell_fn(columns)` → a value_fn
#   the CSV pass calls per cell (engine.pipeline builds it from the columns).
#   Each cell: protect existing tokens + LEGAL_FORMS + GENERIC_DESCRIPTORS
#   verbatim, replace every maximal plaintext name-run with ONE CRM-range ORG
#   token (so bulk third-party names don't collide with low-numbered brand
#   tokens). Contact-name (person) columns are the caller's to exclude.
# --------------------------------------------------------------------------- #
_KEEP_WORD = {w.upper() for w in LEGAL_FORMS} | {w.upper() for w in GENERIC_DESCRIPTORS}
# Conjunctions/articles that GLUE a multi-word name («Рога И Копыта», «Coca AND Cola»):
# inside a name-run they are part of the name, not a structural descriptor — otherwise the run
# breaks on every «и»/«and» and one name gives several tokens. The same
# set as _LINK in the regex rules. Kept SEPARATE from _KEEP_WORD: at the start/
# end of a cell a lone conjunction stays verbatim (we do not seed «и» into a token).
_LINK_WORDS = {"И", "THE", "OF", "AND"}
_COMPANY_HEADERS = {
    # only UNambiguous company headers auto-detect. "наименование" is excluded:
    # it more often labels a product/document/item column than a company, and
    # masking it wholesale is the top false-positive source. Name it explicitly
    # via owner.crm_columns when a given corpus really uses it for companies.
    "компания", "организация", "company", "organization",
    "юрлицо", "контрагент", "фирма", "контрагенты",
}
# split a cell into word-runs and separators; keep the separators verbatim.
_SPLIT = re.compile(r"([^\wА-Яа-яЁё]+)")


def company_cell_fn(columns=None):
    """Return a value_fn(cell, header) for engine.formats.mask_csv_cells.

    Masks each plaintext name-run in a company cell to one CRM-range ORG token,
    keeping already-placed tokens, legal forms and generic descriptors verbatim.
    """
    cols = None
    if columns:
        cols = {c.lower() if isinstance(c, str) else c for c in columns}

    def fn(cell, header=None):
        if not cell or not cell.strip():
            return cell
        # column gate: explicit columns => only those; else header must look
        # like a company column (never touch arbitrary CSV cells — a ragged
        # row's overflow cell arrives with header=None and must stay verbatim).
        if cols is not None:
            if not (header and header.lower() in cols):
                return cell
        elif header is None or header.lower() not in _COMPANY_HEADERS:
            return cell
        if is_own_token(cell.strip()):
            return cell
        # client role designations in a cell — each with its own ORG token, before
        # the CRM column minter (otherwise «Исполнитель 1, Исполнитель 4» — one crm_org for both roles)
        return _mask_name_runs(sub_client_roles(cell),
                               lambda words: mint.crm_org(words.lower()))

    return fn


def _mask_name_runs(cell: str, minter) -> str:
    """A cell → each maximal name-run under ONE token ``minter(run)``;
    already minted tokens, legal forms/descriptors and tech terms — verbatim. A common
    building block of the CRM pass (crm_org) and the column rules (pname / tag)."""
    out, run = [], []

    def flush():
        # run — a maximal name-run: name words AND inner separators between
        # them («Рога и Копыта»). Edge separators are not part of the name —
        # we cut them outward verbatim and seed the REST as one token, so that
        # one name = one token (the docstring promises a maximal name-run;
        # the old branch flush()-ed on every space → «Рога и Копыта» =
        # three tokens).
        if not run:
            return
        i, j = 0, len(run)
        while i < j and _SPLIT.fullmatch(run[i]):
            out.append(run[i]); i += 1
        trail = []
        while j > i and _SPLIT.fullmatch(run[j - 1]):
            trail.append(run[j - 1]); j -= 1
        core = run[i:j]
        run.clear()
        if not core:
            out.extend(reversed(trail))
            return
        words = "".join(core)
        if is_own_token(words) or words.upper() in _KEEP_WORD \
                or words.lower() in TECH_TERMS:
            out.append(words)
        else:
            out.append(minter(words))
        out.extend(reversed(trail))

    for seg in _SPLIT.split(cell):
        if not seg:
            continue
        if _SPLIT.fullmatch(seg):           # a separator run
            if run:
                run.append(seg)             # may be an INNER separator
            else:
                out.append(seg)
        elif run and seg.upper() in _LINK_WORDS:
            run.append(seg)                  # a conjunction INSIDE the name — part of the run
        elif is_own_token(seg) or seg.upper() in _KEEP_WORD:
            flush()                          # keep legal form / token verbatim
            out.append(seg)
        else:                                # part of a name-run
            run.append(seg)
    flush()
    return "".join(out)


# --------------------------------------------------------------------------- #
# csv_column_rule — a CSV/TSV column WHOLE (idea from presidio-structured).
#   The column family = klass of the candidate of the review.CSV_COLUMN_COHORT cohort
#   (the tools/csv_profile profile in mine → one verdict → one pack rule).
#   The family minter is the same as for the family's literal rules, otherwise one
#   column would be a mix of two placeholder forms (the regex pass runs BEFORE
#   the column pass and has already put its form into "normal" cells; the column
#   picks up the rest — a rare surname, a malformed phone):
#     person → mint.pname   (PERSON_NAME_*, like fullname_pair/pack person-*)
#     org    → mint.tag ORG (like opf_name/brand_variation: «Ромашка» in text
#              and in a column = one token; NOT crm_org — that one isolates bulk names
#              of third parties from brands, while the column verdict is about the client's entities)
#     email/phone/inn → family constants (contacts/financial: EMAIL_REDACTED,
#              PHONE_REDACTED, INN/OGRN/ACCOUNT/CARD_REDACTED by the digit count)
#   A cell — only WHOLE values (not prose); already minted tokens and
#   placeholders are left alone (idempotency: apply → a repeated dry = 0).
# --------------------------------------------------------------------------- #
COLUMN_FAMILIES = ("person", "org", "email", "phone", "inn")


def _id_placeholder(cell: str) -> str:
    """The inn family placeholder by the digit count — as in the financial rules:
    13/15 → OGRN, 20 → account, 16 → card, otherwise INN."""
    n = sum(ch.isdigit() for ch in cell)
    if n in (13, 15):
        return P.OGRN
    if n == 20:
        return P.ACCOUNT
    if n == 16:
        return P.CARD
    return P.INN


def column_cell_fn(family: str, columns=None):
    """value_fn(cell, header) for mask_csv_cells: the whole cell of a column of the family
    ``family`` → the family token/placeholder. ``columns`` — headers (by
    strip); a cell outside them and a cell without a header (overflow of a ragged
    row, header=None) are returned verbatim."""
    if family not in COLUMN_FAMILIES:
        raise SystemExit(f"orgs.column_cell_fn: family {family!r} not in "
                         f"{COLUMN_FAMILIES}")
    cols = {str(c).strip() for c in columns} if columns else None
    if family == "person":
        minter = lambda words: mint.pname(words.lower())          # noqa: E731
    elif family == "org":
        minter = lambda words: mint.tag("ORG", words.lower())     # noqa: E731
    else:
        minter = None

    def fn(cell, header=None):
        if not cell or not cell.strip():
            return cell
        if cols is not None and not (header and header.strip() in cols):
            return cell
        s = cell.strip()
        # already masked (a repeated round / the regex pass earlier) — verbatim
        if is_own_token(s) or s in P.ALL:
            return cell
        if not re.search(r"\w", s):              # «-», «—», «?» — not a value
            return cell
        if minter is not None:
            # client role designations («Исполнитель 1, Исполнитель 4») —
            # with the same ORG token as the engine rule in text, each its own, BEFORE the family
            # minter (the column is applied before the regex passes — the rule does not see the cell)
            return _mask_name_runs(sub_client_roles(cell), minter)
        # email/phone/inn: the whole value → the family constant; edge
        # whitespace is kept (structure intact, as in _mask_name_runs)
        lead = cell[: len(cell) - len(cell.lstrip())]
        trail = cell[len(cell.rstrip()):]
        if family == "email":
            token = P.EMAIL
        elif family == "phone":
            token = P.PHONE
        else:
            token = _id_placeholder(s)
        return lead + token + trail

    return fn


def csv_column_rule(name: str, family: str, columns, scope=None,
                    note: str = "") -> Rule:
    """``kind='csv_column'`` column Rule: the pipeline applies it to
    .csv/.tsv (AFTER the regex passes) via mask_csv_cells by ``columns``
    (headers), only on a relpath from ``scope`` (None = any file).
    Carries ``.columns``/``.family``/``.value_fn``; for the regex pass it never
    matches (apply_rules skips kind csv_column)."""
    if family not in COLUMN_FAMILIES:
        raise SystemExit(f"orgs.csv_column_rule {name!r}: family "
                         f"{family!r} not in {COLUMN_FAMILIES}")
    cols = [str(c) for c in columns]
    if not cols:
        raise SystemExit(f"orgs.csv_column_rule {name!r}: empty column list")
    r = Rule(
        name, re.compile(r"(?!x)x"), mask=lambda m: m.group(0),
        kind="csv_column", confidence="high",
        scope=set(scope) if scope else None,
        note=note or f"csv-column {family}: {', '.join(cols)}",
    )
    r.columns = cols
    r.family = family
    r.value_fn = column_cell_fn(family, cols)
    return r
