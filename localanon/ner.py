"""NER layers of the local run: the name model and Natasha — people's names as extra literals.
Natasha follows the main pipeline's local NER tool without its pool, cache and run directory; the
same filters as the main pipeline mining.

The model (``[ner].model``, localanon/nermodel.py; ``model_rules``): its mentions do NOT go
through triage — the model is trained on chats and has already decided it is a person. Mention →
words: given name (any case, by gazetteer forms), patronymic, initial, the rest — surname. A given
name alone is not masked; a single non-name word is a surname. A person with a surname gets the
roster rules (``roster.person_rules``: the pair in any order and case, initials, surname
Capitalized/CAPS) with a token by the nominative surname — the same token as the roster and the
engine give in other files; on top of them — lowercase surname forms (``lower``) and the mention
itself as a literal (lowercase pair "иван петров", "Иван Сергеевич"). Common word guards:
COMMON_WORDS, places and organizations (``place_word``), a dictionary reading by pymorphy
(``common_reading``): such a word Capitalized is a homonym surname (masked not at a sentence
start, like the roster), lowercase — skipped.

Natasha (``[ner].natasha``): NewsNERTagger (PER in capitalized prose) + NamesExtractor (lowercase
names); the text is cut into chunks at line boundaries (yargy is superlinear in memory).
Triage (``triage``): a fragment with punctuation — the span core; more than 4 words — out; all
words are plain dictionary words without a name reading (natasha morphology) — out; STRONG — all
words capitalized, none "mostly lowercase" in the corpus, and (two+ words or frequency ≥ 2);
DICTIONARY — all words in the russiannames name dictionaries (engine/name_dicts), not homonyms,
without a non-name reading; the rest is weak: in the main pipeline a local model or the cloud
judges it, here there is no verdict — not masked (a counter goes to the report).

Natasha literal → rules: the full form "Given Surname" — person rules (roster.person_rules, the
same token as the pair in text); a single word — a surname rule only for a dictionary surname
(russiannames, top-N cutoff), not a common-noun homonym and not a place/organization; a given
name alone is not masked.

Places and organizations: LOC/ORG spans of the natasha tagger block NamesExtractor matches in the
same place; a single candidate word is dropped if it has a pymorphy reading with Geox/Orgn/Trad
(the same sign as the engine's "name evidence" in engine/lexicon) or is in the stop list of the
engine's name detector (``_NOT_A_SURNAME``). There are no separate dictionary files of places and
organizations in ``anonymizer/engine/data`` — this is the engine's geo/org dictionary
(morphology + stop lists).
"""
from __future__ import annotations

import contextlib
import os
import re
import warnings
from collections import Counter

from anonymizer.detectors.names import (_GEO_MARKERS, _NOT_A_SURNAME, _seed_person,
                                        _surname_nominative)
from anonymizer.engine import mint, name_dicts
from anonymizer.engine.boundary import CL, CR, MINT_FRAG_RE
from anonymizer.engine.lit_hygiene import literal_parts
from anonymizer.engine.rule import Rule
from anonymizer.engine.vocab import COMMON_EN, LEGAL_FORMS, TECH_TERMS
from anonymizer.structured.entities import (_LAT_COMMON, _cyr_surname_like, _lat_given, canonical,
                                            person_parts)

from . import roster as _roster

_CHUNK = 32 * 1024
_MIN_LEN = 3
_MAX_WORDS = 4
_LC_DOMINANCE = 3
# a single candidate word is a surname only from the top of the russiannames dictionary (top-N
# for dictionary candidates); the dictionary tail is places and noise ("Самара" #48993, "Москва")
SURNAME_TOP = 30000
_PLACE_TAGS = ("Geox", "Orgn", "Trad")
_PLACE_STOP = frozenset(w.lower() for w in _NOT_A_SURNAME | _GEO_MARKERS)
_FRAGMENT_RE = re.compile(r"[.!?,;:()\"«»…]")
_INITIALS_RE = re.compile(r"(?:[A-ZА-ЯЁ]\.){1,3}")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё][\w-]*")
_TOKENISH = re.compile(r"PERSON_NAME|USER_ID|HANDLE_|ORG_|RESOURCE_|SECRET_|_REDACTED")
_HAS_CYR = re.compile(r"[А-Яа-яЁё]")
# common words that NamesExtractor takes for names (the core of the main pipeline's
# residual.COMMON_WORDS is not moved here — it holds corpus data; here is a general minimum)
COMMON_WORDS = frozenset(w.casefold() for w in COMMON_EN | TECH_TERMS | {
    "привет", "спасибо", "пожалуйста", "хорошо", "давай", "готово", "ладно", "ок", "окей",
    "да", "нет", "вера", "надежда", "любовь", "роза", "лилия", "мира", "воля", "слава",
    "роман", "лев", "май", "август", "марта", "мила", "лада", "рада", "идея", "искра",
    "заказчик", "исполнитель", "клиент", "команда", "проект", "задача", "релиз", "спринт",
    "дизайн", "макет", "ссылка", "файл", "папка", "сайт", "почта", "телефон", "договор",
    "счёт", "счет", "акт", "оплата", "срок", "встреча", "звонок", "созвон", "чат", "канал",
})


# --------------------------------------------------------------------------- Natasha
@contextlib.contextmanager
def _quiet_pkg_resources():
    """pymorphy2 (under natasha) imports pkg_resources, and setuptools 67-80 warns about it with a
    UserWarning on every run. Only that warning, only from pymorphy / pkg_resources, only around our
    natasha imports — anything else still shows."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated", category=UserWarning,
                                module=r"(pymorphy|pkg_resources)")
        yield


def natasha_available() -> bool:
    try:
        with _quiet_pkg_resources():
            import natasha  # noqa: F401
    except ImportError:
        return False
    return True


class NatashaEngine:
    """One-time build of the natasha models; extract — spans (literal, klass, start, stop)."""

    def __init__(self):
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        with _quiet_pkg_resources():
            import natasha
            self.Doc = natasha.Doc
            self.segmenter = natasha.Segmenter()
            self.tagger = natasha.NewsNERTagger(natasha.NewsEmbedding())
            self.morph = natasha.MorphVocab()
            self.names = natasha.NamesExtractor(self.morph)

    def extract(self, text: str) -> list:
        out = []
        for a, b in chunk_bounds(text):
            out.extend(self._piece(text[a:b], a))
        return out

    def _piece(self, text: str, offset: int) -> list:
        if not text.strip():
            return []
        out, taken = [], []
        doc = self.Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.tagger)
        for span in doc.spans:
            # tagger LOC/ORG are not people: their place is closed for NamesExtractor too ("в Москве",
            # "Самара" it reads as names)
            taken.append((span.start, span.stop))
            if span.type != "PER":
                continue
            out.append((text[span.start:span.stop], "person", span.start + offset,
                        span.stop + offset))
        for m in self.names(text):
            a, b = m.start, m.stop
            if any(a < tb and ta < b for ta, tb in taken):
                continue
            lit = text[a:b]
            if len(lit) < _MIN_LEN or lit.casefold() in COMMON_WORDS:
                continue
            out.append((lit, "person", a + offset, b + offset))
        out.sort(key=lambda t: (t[2], t[3]))
        return out

    def dictionary_common(self, word: str) -> bool:
        """All readings are known to the dictionary and none is a name reading."""
        parses = self.morph.parse(word)
        if not parses:
            return False
        for p in parses:
            if not p.is_known:
                return False
            tag = str(p.tag)
            if any(t in tag for t in _NAME_TAGS):
                return False
        return True

    def common_reading(self, word: str) -> bool:
        for p in self.morph.parse(word.lower()):
            tag = str(p.tag)
            if p.is_known and "Poss" not in tag and not any(t in tag for t in _NAME_TAGS):
                return True
        return False

    def place_word(self, word: str) -> bool:
        """Place/organization: a known reading with Geox/Orgn/Trad or the engine stop list."""
        return place_word(word, self.morph)


_NAME_TAGS = ("Name", "Surn", "Patr", "Geox", "Orgn", "Trad")
_CUT_PUNCT = ".,;:!?)»…"
_MORPH = None


def _morph():
    """natasha MorphVocab to check places and common words without the full NatashaEngine (model
    layer, literal_rules); no natasha — None."""
    global _MORPH
    if _MORPH is None and natasha_available():
        with _quiet_pkg_resources():
            import natasha
            _MORPH = natasha.MorphVocab()
    return _MORPH


def place_word(word: str, morph=None) -> bool:
    """The word is a place or organization by the engine dictionaries: a pymorphy (natasha) reading with
    Geox/Orgn/Trad (like engine/lexicon._PII_TAGS) or the detectors/names stop list
    (``_NOT_A_SURNAME``, ``_GEO_MARKERS``: "Москва", "Россия", "Город"…)."""
    w = word.strip().strip("«»\"'")
    if not w:
        return False
    if w.lower().replace("ё", "е") in _PLACE_STOP:
        return True
    if morph is None:
        morph = _morph()
    if not morph:                       # False — the stop list only (tests without natasha)
        return False
    for p in morph.parse(w):
        if p.is_known and any(t in str(p.tag) for t in _PLACE_TAGS):
            return True
    return False


def chunk_bounds(text: str) -> list:
    """Chunks ≤ _CHUNK at line boundaries."""
    if not text.strip():
        return []
    if len(text) <= _CHUNK:
        return [(0, len(text))]
    out, pos = [], 0
    while pos < len(text):
        end = min(len(text), pos + _CHUNK)
        if end < len(text):
            cut = text.rfind("\n", pos, end)
            if cut >= pos:
                end = cut + 1
            else:
                lo = max(pos, end - _CHUNK // 2)
                k = end - 1
                while k > lo:
                    ch = text[k]
                    if ch.isspace() or ch in _CUT_PUNCT:
                        end = k + 1
                        break
                    k -= 1
        out.append((pos, end))
        pos = end
    return out


# --------------------------------------------------------------------------- triage
_OPF_HEAD = re.compile(r"^(?:" + "|".join(re.escape(f) for f in sorted(LEGAL_FORMS, key=len,
                                                                       reverse=True))
                       + r")\.?\s+")


def span_core(lit: str) -> str:
    """Span core with edge punctuation: the legal form at the head, initials and quotes at the edges
    are cut ("ООО «Ромашка»" → "Ромашка", "Петров А." → "Петров")."""
    words = _OPF_HEAD.sub("", lit.strip()).strip("«»„“”\"' ").split()
    while words and _INITIALS_RE.fullmatch(words[-1]):
        words.pop()
    while words and _INITIALS_RE.fullmatch(words[0]):
        words.pop(0)
    return " ".join(words).strip("«»„“”\"' ")


def word_freq(texts) -> Counter:
    """Corpus word frequencies (case kept) for the lowercase dominance guard."""
    freq: Counter = Counter()
    for t in texts:
        freq.update(_WORD_RE.findall(t))
    return freq


def dict_name(word: str) -> bool:
    top = {"surnames": 30000, "first_names": 5000, "patronymics": 3000} \
        if word[:1].islower() else {}
    known = (name_dicts.is_surname(word, top.get("surnames"))
             or name_dicts.is_first_name(word, top.get("first_names"))
             or name_dicts.is_patronymic(word, top.get("patronymics")))
    return known and not name_dicts.is_homonym(word) and word.casefold() not in COMMON_WORDS


def triage(counts: Counter, corpus_freq: Counter, common_word=None, common_reading=None,
           in_dict=None, place=None) -> tuple:
    """{literal: frequency} → (accepted literals, statistics). Predicate injection is for tests
    without natasha; the default is natasha morphology (if present) and the engine name
    dictionaries. ``place`` — place/organization: a single place word and a literal of places only
    ("Нижний Новгород") are not candidates (stats["place"])."""
    common = common_word or (lambda w: False)
    reading = common_reading or (lambda w: False)
    in_dict = in_dict or dict_name
    place = place or place_word
    stats = Counter()
    have = set(counts)
    accepted: list = []

    def lc_dominant(word: str) -> bool:
        lo = corpus_freq.get(word.lower(), 0)
        up = corpus_freq.get(word, 0)
        return lo >= _LC_DOMINANCE * max(up, 1)

    for lit, freq in sorted(counts.items()):
        if _FRAGMENT_RE.search(lit):
            core = span_core(lit)
            if core and core != lit and not _FRAGMENT_RE.search(core) and core not in have:
                have.add(core)
                lit = core
        words = lit.split()
        if _FRAGMENT_RE.search(lit) or len(words) > _MAX_WORDS or not words:
            stats["dropped"] += 1
            continue
        if MINT_FRAG_RE.search(lit) or _TOKENISH.search(lit) \
                or any(w.casefold() in COMMON_WORDS for w in words):
            stats["dropped"] += 1               # token fragment ("nPERSON_NAME"), common word
            continue
        if all(common(w) for w in words):
            stats["dictionary"] += 1
            continue
        if (len(words) == 1 and place(words[0])) or all(place(w) for w in words):
            stats["place"] += 1                 # city, river, region, organization
            continue
        capitalized = all(w[:1].isupper() for w in words)
        strong = capitalized and not any(lc_dominant(w) for w in words) \
            and (len(words) >= 2 or freq >= 2)
        by_dict = all(in_dict(w) and not lc_dominant(w) and not reading(w) for w in words)
        if strong or by_dict:
            stats["strong" if strong else "dict"] += 1
            accepted.append(lit)
        else:
            stats["weak"] += 1
    return accepted, stats


# --------------------------------------------------------------------------- literals → rules
def surname_candidate(word: str, place=None) -> bool:
    """A single word is a surname for a rule: in the dictionary (russiannames, top SURNAME_TOP),
    not a common-noun homonym, not a place/organization, not a given name."""
    place = place or place_word
    if word in _roster.gaz.RU_FIRST_NAMES_ALL or word in _roster.gaz.RU_FIRST_NAMES_DIM \
            or name_dicts.is_first_name(word) and not name_dicts.is_surname(word):
        return False
    if not name_dicts.is_surname(word, SURNAME_TOP) or name_dicts.is_homonym(word):
        return False
    return not place(word)


def literal_rules(literals, prefix: str = "ner", place=None) -> tuple:
    """(rules, number of people, number of surnames). A full form — person rules; one word — a
    surname rule only by surname_candidate (dictionary, not a homonym, not a place); a given name
    alone — skipped."""
    rules, persons, surnames = [], 0, 0
    seen: set = set()
    for lit in literals:
        c = canonical(lit)
        if c:
            if c in seen:
                continue
            seen.add(c)
            given, surname = person_parts(c)
            p = _roster.Person(c, given, surname, not _HAS_CYR.search(surname))
            seen.add(p.seed)
            rules.extend(_roster.person_rules(p, prefix))
            persons += 1
            continue
        words = lit.split()
        if len(words) != 1:
            continue
        w = words[0]
        if not _HAS_CYR.search(w) or not w[:1].isupper():
            continue
        if not _cyr_surname_like(w) or not surname_candidate(w, place):
            continue
        p = _roster.Person(w, "", w, False)
        if p.seed in seen:
            continue
        seen.add(p.seed)
        rules.append(_roster.person_rules(p, prefix)[-1])     # the surname rule only
        surnames += 1
    return rules, persons, surnames


# --------------------------------------------------------------------------- Natasha over views
_PIECES_PER_BATCH = 4                   # pieces (≤ _CHUNK each) per pool message


def natasha_pieces(views) -> list:
    """Views → the pieces ``NatashaEngine.extract`` walks (``chunk_bounds``), in order."""
    out = []
    for _rel, text in views:
        out.extend(text[a:b] for a, b in chunk_bounds(text))
    return out


def _natasha_init(_slot):
    return NatashaEngine()


def _natasha_batch(eng, batch: list) -> list:
    """Pool worker: pieces → the raw span literals of each piece (the offset does not change them)."""
    return [[raw for raw, _klass, _a, _b in eng._piece(piece, 0)] for piece in batch]


def natasha_literals(views, note=None, slots=None, ctx=None) -> tuple:
    """Natasha spans over texts → accepted literals and statistics. ``views`` — [(rel, text)].

    ``slots`` — CPU slots of the hw plan (stage ``natasha``): with more than one the pieces go to
    worker processes (``devpool``, each loads Natasha once); the counts are summed, so the literals
    and statistics are the same as in one process. ``ctx`` — for pool notes."""
    if not natasha_available():
        if note:
            note("natasha is not installed: NER layer skipped (pip install natasha)")
        return [], Counter()
    eng = NatashaEngine()
    counts: Counter = Counter()
    texts = [text for _rel, text in views]
    if slots and len(slots) > 1:
        from . import devpool
        raws = devpool.run(_natasha_batch, natasha_pieces(views), slots, init=_natasha_init,
                           batch_size=_PIECES_PER_BATCH, ctx=ctx)
    else:
        raws = ([raw for raw, _klass, _a, _b in eng.extract(text)] for text in texts)
    for piece_raws in raws:
        for raw in piece_raws:
            for lit in literal_parts(raw):
                lit = " ".join(lit.split())
                if lit:
                    counts[lit] += 1
    freq = word_freq(texts)
    return triage(counts, freq, eng.dictionary_common, eng.common_reading, place=eng.place_word)


# --------------------------------------------------------------------------- model: mentions → rules
LOWER_POLICIES = ("off", "seen", "all")
_EDGE = ",;:!?()[]{}<>«»„“”\"'…—–*_"
_INITIAL_RE = re.compile(r"(?:[A-Za-zА-Яа-яЁё]\.){1,3}|[A-ZА-ЯЁ]")
_CYR_WORD = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?")
_LAT_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
_PATR_ANY = re.compile(r"[а-яё]{2,}(?:(?:ович|евич|ьич|ич)(?:а|у|е|ем|и)?"
                       r"|(?:овн|евн|ичн|иничн)(?:а|у|е|ой|ою|ы|и))")
_PATR_M = re.compile(r"(.+?(?:ович|евич|ьич|ич))(?:а|у|е|ем|и)?")
_PATR_F = re.compile(r"(.+?(?:овн|евн|ичн|иничн))(?:а|у|е|ой|ою|ы|и)")
_NAME_ONLY = ("Name", "Surn", "Patr", "Poss")    # "петрову" is also a possessive of "Пётр"
_SP_RX = r"[ \u00a0\u202f]+"
_GIVEN: dict | None = None


def mentions_of(texts, spans) -> Counter:
    """Model spans by lines → {mention (spaces collapsed): frequency}."""
    out: Counter = Counter()
    for text, sp in zip(texts, spans):
        for s, e in sp:
            m = " ".join(text[s:e].split())
            if m:
                out[m] += 1
    return out


def _given_table() -> dict:
    """Gazetteer given name forms (lowercase, all cases) → the nominative name; once per process."""
    global _GIVEN
    if _GIVEN is None:
        tab: dict = {}
        for n in sorted(_roster.gaz.RU_FIRST_NAMES_ALL | _roster.gaz.RU_FIRST_NAMES_DIM):
            tab.setdefault(n.lower(), n)
            for f in _roster.given_forms(n):
                tab.setdefault(f, n)
        _GIVEN = tab
    return _GIVEN


def given_nominative(word: str) -> str:
    """Given name (any case and letter case) → nominative from the gazetteer; Latin — by the engine
    name dictionary (English and transliterated Russian); not a name — ''."""
    if _HAS_CYR.search(word):
        return _given_table().get(word.lower(), "")
    return word[:1].upper() + word[1:].lower() if _lat_given(word) else ""


def patronymic_nominative(word: str) -> str:
    low = word.lower()
    m = _PATR_F.fullmatch(low)
    if m:
        return m.group(1) + "а"
    m = _PATR_M.fullmatch(low)
    return m.group(1) if m else low


def common_reading(word: str) -> bool:
    """There is a known pymorphy (natasha) reading without name tags — the word can be common ("сокол",
    "кузнецов" — genitive plural). Without natasha — COMMON_WORDS only."""
    if word.casefold() in COMMON_WORDS:
        return True
    morph = _morph()
    if not morph:
        return False
    for p in morph.parse(word.lower()):
        tag = str(p.tag)
        if p.is_known and not any(t in tag for t in _NAME_ONLY):
            return True
    return False


_PERSON_TAGS = ("Name", "Surn", "Patr")


def plain_word(word: str) -> bool:
    """A Cyrillic word that is only an ordinary dictionary word, never a person: COMMON_WORDS, or
    pymorphy (natasha) knows it with no Name/Surn/Patr reading — and the name dictionaries of the
    engine know it neither as a surname nor as a given name («Техническое», «Задание»). Surnames
    that are also common words («Кузнецов», «Сокол», «Белых») are not plain. Latin — False."""
    w = word.strip(_EDGE)
    if not w or not _HAS_CYR.search(w) or not _CYR_WORD.fullmatch(w):
        return False
    if name_dicts.is_surname(w) or name_dicts.is_first_name(w) or given_nominative(w):
        return False
    if w.casefold() in COMMON_WORDS:
        return True
    morph = _morph()
    if not morph:
        return False
    known = [str(p.tag) for p in morph.parse(w.lower()) if p.is_known]
    return bool(known) and not any(t in tag for tag in known for t in _PERSON_TAGS)


def guard_filename_words() -> None:
    """File and folder names: a lone capitalised segment that is a plain dictionary word
    (``plain_word``: «Техническое» in «Техническое задание.docx» has the «-ск-» shape of a surname)
    is kept unless a given name stands in the same name («Иван Петров», «Петров отчёт» are still
    masked). Wraps the segment classifier of ``anonymizer.detectors.filenames`` once per process,
    so every caller of ``filenames.mask_name`` (renaming, Telegram/YouTrack file fields) gets it;
    the engine source stays untouched (proposed to the engine in the la-tail report)."""
    from anonymizer.detectors import filenames as fn
    orig = fn._classify_segment
    if getattr(orig, "localanon_guard", False):
        return

    def classify(seg, is_dir, cue=frozenset()):
        if seg and not cue and plain_word(seg):
            return None, None
        return orig(seg, is_dir, cue)

    classify.localanon_guard = True
    fn._classify_segment = classify


def _cap_word(w: str) -> str:
    return "-".join(x[:1].upper() + x[1:].lower() for x in w.split("-"))


def _mention_words(mention: str):
    """[(kind, word)]: "init" — an initial, "word" — a word of one alphabet; anything else (digits,
    mixed alphabets, fragments) — None."""
    out = []
    for raw in mention.split():
        w = raw.strip(_EDGE)
        if not w:
            continue
        if _INITIAL_RE.fullmatch(w):
            out.append(("init", w))
            continue
        w = w.rstrip(".")
        if _CYR_WORD.fullmatch(w) or _LAT_WORD.fullmatch(w):
            out.append(("word", w))
        else:
            return None
    return out


def _surname_reject(w: str, place) -> str:
    """Why the word is not a surname (statistics key) or ''."""
    if w.casefold() in COMMON_WORDS:
        return "common"
    if len(w) < 3:
        return "short"
    if not _HAS_CYR.search(w):
        return "common" if w.lower() in _LAT_COMMON else ""
    return "place" if place(w) else ""


def _mention_rx(surfaces) -> str:
    alts = sorted(set(surfaces), key=len, reverse=True)
    return "|".join(_SP_RX.join(re.escape(x) for x in s.split()) for s in alts)


def _lookup_rule(name: str, table: dict, pattern: str) -> Rule:
    def mask(m, t=table):
        return t[" ".join(m.group(1).split())]
    return Rule(name, re.compile(CL + "(" + pattern + ")" + CR), mask=mask, value_group=1,
                guard=_roster._guard, confidence="high")


def model_rules(mentions, prefix: str = "ner:model", lower: str = "seen", common=None,
                place=None) -> tuple:
    """Model mentions {surface: frequency} → (early rules, pack rules, statistics).

    Early (``Engine(before_names=…)``): multi-word mentions as literals, pairs and initials of
    people. Pack: surnames (Capitalized/CAPS; Latin — case-insensitive), then lowercase surname
    forms by ``lower``: ``off`` — none, ``seen`` — for surnames the model saw in lowercase,
    ``all`` — for all (except homonyms); a lowercase form with a common reading (``common``) — skipped."""
    if lower not in LOWER_POLICIES:
        raise ValueError(f"lower: {lower!r} is not one of {LOWER_POLICIES}")
    common = common or common_reading
    place = place or place_word
    stats: Counter = Counter()
    persons: dict = {}               # seed → {"cap", "latin", "homonym", "given": set, "surf": set, "lower"}
    surfaces: dict = {}              # multi-word mention → token

    def person(w: str, given: str) -> dict:
        latin = not _HAS_CYR.search(w)
        cap = w if latin else _cap_word(w)
        p = _roster.Person(cap, "", cap, latin)
        e = persons.get(p.seed)
        if e is None:
            homonym = not latin and (name_dicts.is_homonym(cap)
                                     or (common(w.lower()) and not _cyr_surname_like(cap)))
            e = persons[p.seed] = {"cap": cap, "latin": latin, "homonym": homonym, "given": set(),
                                   "surf": set(), "lower": False, "token": p.token}
        if given and bool(_HAS_CYR.search(given)) != latin:     # given name of the same alphabet
            e["given"].add(given)
        e["surf"].add(w)
        if w[:1].islower():
            e["lower"] = True
        return e

    for mention in sorted(mentions):
        if MINT_FRAG_RE.search(mention) or _TOKENISH.search(mention):
            stats["token"] += 1
            continue
        ws = _mention_words(mention)
        if not ws or len(ws) > _MAX_WORDS + 1:
            stats["dropped"] += 1
            continue
        words = [w for k, w in ws if k == "word"]
        given, patr, surn = [], [], []
        for w in words:
            g = given_nominative(w)
            if g:
                given.append(g)
            elif len(words) > 1 and _PATR_ANY.fullmatch(w.lower()):
                patr.append(w)
            else:
                surn.append(w)
        norm = " ".join(mention.split()).strip(_EDGE)
        if not surn and not patr:
            stats["first"] += 1                 # a given name alone (or with an initial) — not masked
            continue
        if not surn and given:                  # "Иван Сергеевич": token by given name and patronymic
            seed = _seed_person(given[0] + " " + patronymic_nominative(patr[0]))
            surfaces.setdefault(norm, mint.pname(seed))
            stats["first_patr"] += 1
            continue
        if not surn:
            surn = patr
        keep = []
        for w in surn:
            why = _surname_reject(w, place)
            if not why and not given and plain_word(w):
                why = "plain"                   # «Техническое» without a given name next to it
            if why:
                stats[why] += 1
            else:
                keep.append(w)
        if not keep:
            continue
        one = len(keep) == 1
        for w in keep:
            e = person(w, given[0] if given and one else "")
        stats["mentions"] += 1
        if one and len(ws) > 1:
            surfaces.setdefault(norm, e["token"])

    early, pack, lower_tab = [], [], {}
    if surfaces:
        early.append(_lookup_rule(f"{prefix}:mention", surfaces, _mention_rx(surfaces)))
    for seed in sorted(persons, key=lambda s: (-len(persons[s]["cap"]), s)):
        e = persons[seed]
        cap, latin = e["cap"], e["latin"]
        for g in sorted(e["given"]):
            p = _roster.Person(f"{g} {cap}", g, cap, latin, homonym=e["homonym"])
            early.extend(_roster.person_rules(p, prefix)[:-1])
        p = _roster.Person(cap, "", cap, latin, homonym=e["homonym"])
        pack.append(_roster.person_rules(p, prefix)[-1])
        if latin or e["homonym"] or lower == "off" or (lower == "seen" and not e["lower"]):
            continue
        forms = set(_roster.surname_forms(_surname_nominative(cap)))
        forms |= {w.lower() for w in e["surf"]}
        for f in forms:
            if f.casefold() in COMMON_WORDS or common(f):
                stats["lower_common"] += 1
                continue
            lower_tab.setdefault(f, e["token"])
    if lower_tab:
        pack.append(_lookup_rule(f"{prefix}:surname-lc", lower_tab, _mention_rx(lower_tab)))
    stats["persons"] = len(persons)
    stats["homonyms"] = sum(1 for e in persons.values() if e["homonym"])
    stats["surfaces"] = len(surfaces)
    stats["lower_forms"] = len(lower_tab)
    return early, pack, stats
