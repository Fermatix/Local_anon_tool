"""Full-name dictionaries (russiannames) — lazy frozensets + case normalization.

Data: anonymizer/engine/data/ru_{surnames,first_names,patronymics}.txt and
ru_name_homonyms.txt — exported by tools/build_name_dicts.py from datacoon/
russiannames (© Ivan Begtin / datacoon, CC-BY-SA 4.0, the commit pin is in the header
of the files and in NOTICE). Format: lines starting with "#" are the header, then one form per
line, lowercase, "ё" kept, ordered by source frequency, descending
(the first N lines = top N; ranked()).

Key canon — as the conveyor's _seed_person: lowercase + ё->е (norm()). All
sets and all queries are folded the same way, so «Артём»/«Артем»,
«Фёдоров»/«Федоров» are one key.

Case normalization of queries (is_*): an oblique form is reduced to the dictionary form
through CLOSED sets of endings — surnames: the same cut as
detectors/names._strip_case_tail (the ending is cut ONLY if the rest
ends with a surname suffix from vocab.SURNAME_ENDINGS; «-ского/-ская» ->
«-ский»); first names/patronymics: an ending from a closed set + the base ending
(«Марии» -> «мария», «Сергея» -> «сергей», «Ильёй» -> «илья», «Ивановну» ->
«ивановна»). The helper lives here, not in detectors/names — engine does not import
detectors (a cycle), names may import from here.

Homonyms (is_homonym): dictionary forms that are themselves the lemma of a common noun
(Мороз / Волк / Заяц / Роман / Вера / Король; built with pymorphy2, a lemma
rule) + the built-in BUILTIN_HOMONYMS (a fallback without the file). Oblique
homographs (Быков = «быков», Иванов = possessive) are NOT here — they are
guarded by the corpus lower-case dominance of triage (tools/ner_local.triage_rows).

Zero-dependency core: no files -> empty sets, all is_* -> False,
one WARN to stderr per process (available() checks without a WARN). Loading
is lazy (first call), the result is a frozenset in the module cache; predicates
are cached with lru_cache per word. reset() clears the caches (tests, a DATA_DIR change).

    is_surname("Ивановой") / is_first_name("Марии") / is_patronymic("Ильича")
    is_homonym("Мороз") # True -> a dictionary match is no evidence
    ranked("first_names", 3000) -> ["александр", ...]   # top by frequency
    strip_case_tail("ивановой") -> "иванов"
"""
from __future__ import annotations

import os
import re
import sys
from functools import lru_cache

from .vocab import SURNAME_ENDINGS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = {
    "surnames": "ru_surnames.txt",
    "first_names": "ru_first_names.txt",
    "patronymics": "ru_patronymics.txt",
    "homonyms": "ru_name_homonyms.txt",
}
KINDS = ("surnames", "first_names", "patronymics")

# Fallback in case ru_name_homonyms.txt is missing (and the FP set of the tests):
# frequent common-noun lemmas that match a dictionary surname / first name.
BUILTIN_HOMONYMS = frozenset({
    # animals / nature — surnames
    "волк", "заяц", "орел", "кот", "лебедь", "сокол", "соловей", "воробей",
    "сорока", "синица", "чайка", "муха", "комар", "жук", "рак", "лось", "лис",
    "крот", "баран", "козел", "медведь", "дрозд", "кулик", "соболь", "гриб",
    "мороз", "гром", "зима", "весна", "береза", "калина", "дуб", "гора", "река",
    "поле", "шило", "дубина", "середа", "коляда",
    # roles / crafts / ethnonyms — surnames
    "король", "мельник", "кузнец", "гончар", "бондарь", "швец", "ткач",
    "рыбак", "пастух", "казак", "купец", "гайдук", "коваль", "колесник",
    "чумак", "литвин", "швед", "лях", "русак", "бортник", "кулак",
    # adjectives — surnames
    "белый", "черный", "толстой", "добрый", "молодой", "горький", "бойко",
    # first names — common nouns
    "вера", "надежда", "любовь", "роман", "лев", "роза", "лилия", "лада",
    "мила", "марта", "слава", "идея", "воля", "искра", "заря", "рада", "майя",
    "май", "август", "октябрь", "гений", "алмаз", "булат", "султан", "ислам",
    "марс", "муза", "лира", "флора", "аврора", "нора", "виктория",
    "олимпиада", "риф", "граф", "базар", "логин", "абонент", "максим", "мир",
})

_sets: dict = {}          # kind -> frozenset(norm forms)
_ranked: dict = {}        # kind -> tuple(raw lowercase forms in file order)
_warned = False


def norm(word: str) -> str:
    """Key canon: lowercase + ё->е (as _seed_person)."""
    return word.strip().lower().replace("ё", "е")


def path(kind: str) -> str:
    return os.path.join(DATA_DIR, FILES[kind])


def _warn_missing(missing: list) -> None:
    global _warned
    if _warned:
        return
    _warned = True
    sys.stderr.write(
        "name_dicts: WARN full-name dictionaries not found (" + ", ".join(missing) +
        ") — is_surname/is_first_name/is_patronymic are always False, the name "
        "gazetteer stays built-in (~300); to build: python3 tools/"
        "build_name_dicts.py (see NOTICE)\n")
    sys.stderr.flush()


def _read(kind: str) -> tuple:
    """Raw lowercase forms of the file in its order ("ё" kept); no file ->
    empty + one WARN per process."""
    hit = _ranked.get(kind)
    if hit is not None:
        return hit
    p = path(kind)
    forms: list = []
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                w = line.strip().lower()
                if w:
                    forms.append(w)
    elif kind != "homonyms":
        _warn_missing([FILES[k] for k in KINDS if not os.path.isfile(path(k))])
    out = tuple(forms)
    _ranked[kind] = out
    return out


def load(kind: str, top: int | None = None) -> frozenset:
    """frozenset of norm forms of dictionary kind (lazy, module cache); ``top`` —
    only the first N forms of the file (top by source frequency): a gate for weak
    evidence (a lower-case token of an ASR transcript against a rare surname, count 2–5)."""
    key = (kind, top)
    hit = _sets.get(key)
    if hit is None:
        forms = frozenset(norm(w) for w in (_read(kind)[:top] if top else _read(kind)))
        if kind == "homonyms":
            forms = forms | BUILTIN_HOMONYMS
        hit = _sets[key] = forms
    return hit


def ranked(kind: str, limit: int | None = None) -> list:
    """Dictionary forms in file order (frequency descending), raw lowercase
    ("ё" kept — the gazetteer needs both spellings)."""
    forms = _read(kind)
    return list(forms[:limit] if limit else forms)


def available() -> bool:
    """Whether at least the surname or first-name dictionary exists (no WARN)."""
    return any(os.path.isfile(path(k)) for k in KINDS)


def reset() -> None:
    """Clear the caches (tests: a substituted DATA_DIR)."""
    global _warned
    _sets.clear()
    _ranked.clear()
    _warned = False
    for f in (is_surname, is_first_name, is_patronymic, is_homonym, name_kinds):
        f.cache_clear()


# --------------------------------------------------------------------------- #
# Case normalization
# --------------------------------------------------------------------------- #
# Surname ending in an oblique case (the same set as names._TAIL_STRIP):
# longer before shorter, «-ого» as a whole, not as «-о».
_SURN_TAILS = sorted(
    ("а", "у", "е", "ы", "ой", "ою", "ым", "ом", "ых", "им",
     "его", "ого", "ему", "ому"), key=len, reverse=True)
_SURN_END_RE = re.compile(SURNAME_ENDINGS + r"$")
# «-ского / -ской / -ская / -цкую …» -> «-ский / -цкий» (as names._SK_TAIL)
_SK_TAIL = re.compile(r"(?<=ск|цк)(?:ий|ого|ому|им|ом|ая|ой|ую|ие|ых|ими)$")
# first name / patronymic: an oblique-case ending (closed set, longest first)
# + the base ending of the dictionary form
_GIVEN_TAILS = ("ью", "ей", "ой", "ем", "ом", "ии", "а", "я", "у", "ю", "е",
                "и", "ы", "й", "ь")
_GIVEN_BASES = ("", "а", "я", "й", "ь", "ий", "ия", "ья")
_PATR_TAILS = ("ем", "ой", "ою", "а", "у", "е", "ы", "и")
_PATR_BASES = ("", "а")


def strip_case_tail(surname: str) -> str:
    """Oblique case of a surname -> lowercase nominative: the ending is cut ONLY
    when the rest ends with a surname suffix («Ивановой» -> «иванов»,
    «Марина» is left alone); «-ского/-ская» -> «-ский». Mirrors
    detectors/names._strip_case_tail + _surname_nominative."""
    low = surname.lower()
    if _SK_TAIL.search(low):
        return _SK_TAIL.sub("ий", low)
    for t in _SURN_TAILS:
        if low.endswith(t) and len(low) - len(t) >= 3:
            stem = low[:-len(t)]
            if _SURN_END_RE.search(stem):
                return stem
    return low


def _surname_forms(n: str):
    yield n
    s = strip_case_tail(n)
    if s != n:
        yield s


def _inflected_forms(n: str, tails, bases):
    yield n
    for t in tails:
        if n.endswith(t) and len(n) - len(t) >= 2:
            stem = n[:-len(t)]
            for b in bases:
                c = stem + b
                if len(c) >= 3 and c != n:
                    yield c


def _lookup(kind: str, forms, top=None) -> bool:
    s = load(kind, top)
    if not s:
        return False
    return any(f in s for f in forms)


@lru_cache(maxsize=1 << 16)
def is_surname(word: str, top: int | None = None) -> bool:
    """The form (any case, any letter case) is a dictionary surname; ``top`` — only
    among the first N by frequency (see load)."""
    n = norm(word)
    return bool(n) and _lookup("surnames", _surname_forms(n), top)


@lru_cache(maxsize=1 << 16)
def is_first_name(word: str, top: int | None = None) -> bool:
    n = norm(word)
    return bool(n) and _lookup("first_names",
                               _inflected_forms(n, _GIVEN_TAILS, _GIVEN_BASES), top)


@lru_cache(maxsize=1 << 16)
def is_patronymic(word: str, top: int | None = None) -> bool:
    n = norm(word)
    return bool(n) and _lookup("patronymics",
                               _inflected_forms(n, _PATR_TAILS, _PATR_BASES), top)


@lru_cache(maxsize=1 << 16)
def name_kinds(word: str) -> tuple:
    """Which dictionaries know the form: a subset of ("surname", "first_name",
    "patronymic") — for a report/hint; empty = not in a dictionary."""
    out = []
    if is_surname(word):
        out.append("surname")
    if is_first_name(word):
        out.append("first_name")
    if is_patronymic(word):
        out.append("patronymic")
    return tuple(out)


@lru_cache(maxsize=1 << 16)
def is_homonym(word: str) -> bool:
    """The form (or its nominative by the surname cut) is the lemma of a common noun:
    a dictionary match by itself is NOT evidence of a name."""
    n = norm(word)
    if not n:
        return False
    h = load("homonyms")
    return n in h or strip_case_tail(n) in h
