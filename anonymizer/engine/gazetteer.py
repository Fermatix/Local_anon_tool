"""General given-name gazetteers + a case-aware first-name matcher factory.

Data only (NOT corpus-specific): the most common Russian and English given
names in their base (nominative) form. Corpus-specific literal name lists
(specific employees / clients / handles) still belong in
``anonymizer/known_values/`` — these sets are just the frequent, generic
first-names that a bare-name sweep or an anchored ``dict_first_name`` rule
draws on.

RU_FIRST_NAMES is the hand-written core (~300, not extended: packgen classifies
parts of a literal by it); RU_FIRST_NAMES_DICT is the top 3000 names of the
russiannames dictionary (engine/name_dicts, without common-noun homonyms);
RU_FIRST_NAMES_ALL = core + dictionary — the set for the detectors/names matcher. Without
data files the dictionary sets are empty.

The matcher, ``first_name_rx(names)``, compiles a (trie-shaped) alternation of
the base forms and matches each name **with the bounded case-suffix whitelist of its
own declension class** (``а|ы|е|у|ой|…``) rather than an open ``[а-яё]*`` tail
or a cross-class union — the open tail is what turns ``Иван`` into ``Иваново``
/ ``Никит`` into ``Никитин``, and the cross-class union is what turns ``Верю``
/ ``Адом`` into "names", so both are deliberately avoided.

Canonicalisation (one entity -> one token): a masc/fem name is reduced to its
inflection-invariant STEM (drop a trailing а/я/й/ь), the stem is the alternation
atom AND the seed key, so ``Сергей`` / ``Сергея`` / ``Сергею`` all collapse to a
single ``PERSON_NAME_x`` token. Cyrillic no-fuse boundaries (CL / CR) plus a
Latin/digit right guard keep the minted token from welding onto an adjacent
Russian/Latin letter or digit.
"""
from __future__ import annotations

import re

from .boundary import CL, CR

# --------------------------------------------------------------------------- #
# Russian given names — base (nominative) forms, ~300 common. Plain set.
# --------------------------------------------------------------------------- #
RU_FIRST_NAMES = {
    # --- male ---
    "Александр", "Алексей", "Анатолий", "Андрей", "Антон", "Аркадий", "Арсений",
    "Артём", "Артем", "Артур", "Богдан", "Борис", "Вадим", "Валентин", "Валерий",
    "Василий", "Виктор", "Виталий", "Владимир", "Владислав", "Вячеслав",
    "Геннадий", "Георгий", "Герман", "Глеб", "Григорий", "Даниил", "Данил",
    "Денис", "Дмитрий", "Евгений", "Егор", "Ефим", "Захар", "Иван", "Игнат",
    "Игорь", "Илья", "Иннокентий", "Кирилл", "Клим", "Константин", "Лев",
    "Леонид", "Макар", "Максим", "Марк", "Матвей", "Мирон", "Михаил", "Назар",
    "Никита", "Николай", "Олег", "Пётр", "Петр", "Павел", "Платон", "Прохор",
    "Роберт", "Родион", "Роман", "Ростислав", "Руслан", "Савелий", "Святослав",
    "Семён", "Семен", "Сергей", "Спартак", "Станислав", "Степан", "Тарас",
    "Тимофей", "Тимур", "Трофим", "Фёдор", "Федор", "Феликс", "Филипп", "Эдуард",
    "Эльдар", "Эмиль", "Юрий", "Яков", "Ян", "Ярослав",
    "Валера", "Гавриил", "Демьян", "Елисей", "Ждан", "Игорёк", "Лаврентий",
    "Мстислав", "Наум", "Остап", "Пантелей", "Рустам", "Самуил", "Тихон",
    "Устин", "Фадей", "Харитон", "Цезарь", "Юлиан", "Ефрем", "Всеволод",
    # --- female ---
    "Августа", "Агата", "Аглая", "Ада", "Аида", "Александра", "Алёна", "Алена",
    "Алиса", "Алина", "Алла", "Анастасия", "Ангелина", "Анжела", "Анна", "Антонина",
    "Анфиса", "Валентина", "Валерия", "Варвара", "Василиса", "Вера", "Вероника",
    "Виктория", "Виолетта", "Владислава", "Галина", "Дарья", "Диана", "Дина",
    "Евгения", "Ева", "Евдокия", "Екатерина", "Елена", "Елизавета", "Жанна",
    "Зинаида", "Зоя", "Инга", "Инесса", "Инна", "Ирина", "Камилла", "Карина",
    "Кира", "Клавдия", "Кристина", "Ксения", "Лада", "Лариса", "Лидия", "Лилия",
    "Любовь", "Людмила", "Маргарита", "Марина", "Мария", "Марьяна", "Милана",
    "Надежда", "Наталья", "Наталия", "Нелли", "Нина", "Оксана", "Олеся", "Ольга",
    "Полина", "Раиса", "Регина", "Рената", "Римма", "Роза", "Светлана", "Серафима",
    "Снежана", "Софья", "София", "Таисия", "Тамара", "Татьяна", "Ульяна", "Фаина",
    "Эвелина", "Эльвира", "Эмма", "Юлия", "Яна", "Ярослава",
    "Агнесса", "Анжелика", "Белла", "Влада", "Гелена", "Дарина", "Есения",
    "Злата", "Иванна", "Изабелла", "Карен", "Лана", "Марта", "Мила", "Нонна",
    "Прасковья", "Стелла", "Эльза", "Юна",
}
# The hand-written set above is the CORE of the gazetteer: RU_FIRST_NAMES as an OBJECT is not extended
# (packgen classifies parts of a person literal by `part in RU_FIRST_NAMES`:
# a name outside the core goes down the surname branch with a transliteration fan — «Артемий Ким» → Artemiy;
# extending the object would silently change pack contents; mine — klass freq-contrast).
RU_FIRST_NAMES_CORE = frozenset(RU_FIRST_NAMES)

# Extension with the russiannames dictionary (engine/name_dicts, data in engine/data/
# ru_first_names.txt, file order = source frequency): the first DICT_TOP_N
# forms minus common-noun homonyms (name_dicts.is_homonym — «Гений» / «Алмаз» /
# «Идея» / «Логин» are not taken; Роман / Лев / Вера of the core stay under its
# discipline _HOMOGRAPH_STEMS + NAME_CTX), minus hyphenated ones and those shorter than 3 letters.
# «ё» — both spellings, as in the core (Артём/Артем). No file — empty, gazetteer =
# core (the core has zero dependencies; name_dicts prints a WARN once).
#   RU_FIRST_NAMES_DICT — dictionary names only (~2.7k), RU_FIRST_NAMES_ALL — core
# + dictionary: the set for the MATCHER (detectors/names: _FIRST_NAME_RX / dict_first_
# name / cue of transliteration pairs _LAT_FIRST_KEYS — activated in names.py) —
# there width = recall, and classification of pack parts is not affected. The matcher
# first_name_rx over ~3k stems is compiled as a trie alternation (see below):
# the cost per position is the word depth, not the number of names.
DICT_TOP_N = 3000
_DICT_FORM = re.compile(r"^[а-яё]{3,}$")


def dict_first_names(limit: int = DICT_TOP_N) -> set:
    """Capitalized base forms of the top ``limit`` dictionary names (without homonyms)."""
    from . import name_dicts
    out: set = set()
    for w in name_dicts.ranked("first_names", limit):
        if not _DICT_FORM.match(w) or name_dicts.is_homonym(w):
            continue
        cap = w.capitalize()
        out.add(cap)
        if "ё" in cap:
            out.add(cap.replace("ё", "е"))
    return out


RU_FIRST_NAMES_DICT = frozenset(dict_first_names())
RU_FIRST_NAMES_ALL = frozenset(RU_FIRST_NAMES) | RU_FIRST_NAMES_DICT


# bare «-ич» forms are taken from the top DICT_PATR_TOP patronymics by frequency: further on
# the dictionary has typos of productive forms («Николавич», «Владимирвич»)
DICT_PATR_TOP = 1000


# Diminutive / hypocoristic given-name stems live in ``vocab.DIMINUTIVES``
# (they need a vocative anchor and richer suffix handling); this set is only the
# extra common short forms not already there, still as base forms.
RU_FIRST_NAMES_DIM = {
    "Саша", "Женя", "Дима", "Митя", "Костя", "Толя", "Гриша", "Гоша", "Жора",
    "Слава", "Стас", "Витя", "Валя", "Веня", "Гена", "Даня", "Егорка", "Захарка",
    "Ксюша", "Маша", "Даша", "Паша", "Наташа", "Настя", "Катя", "Лена", "Оля",
    "Таня", "Света", "Вера", "Люба", "Люда", "Галя", "Зина", "Тома", "Аня",
    "Юля", "Соня", "Поля", "Варя", "Ляля", "Нюра", "Рита", "Алла", "Инна",
}

# --------------------------------------------------------------------------- #
# English given names — ~200 common. Plain set. Matched case-insensitively by
# callers; stored capitalised. No Russian case tail applies to these.
# --------------------------------------------------------------------------- #
EN_FIRST_NAMES = {
    # --- male ---
    "Aaron", "Adam", "Adrian", "Alan", "Albert", "Alex", "Alexander", "Andrew",
    "Anthony", "Antonio", "Arthur", "Austin", "Benjamin", "Bernard", "Bill",
    "Bobby", "Brandon", "Brian", "Bruce", "Bryan", "Carl", "Carlos", "Charles",
    "Chris", "Christopher", "Clarence", "Craig", "Dan", "Daniel", "David",
    "Dennis", "Derek", "Donald", "Douglas", "Dylan", "Earl", "Edward", "Elijah",
    "Eric", "Ernest", "Ethan", "Eugene", "Evan", "Frank", "Fred", "Gabriel",
    "Gary", "George", "Gerald", "Gordon", "Greg", "Gregory", "Harold", "Harry",
    "Henry", "Howard", "Ian", "Isaac", "Jack", "Jacob", "Jake", "James", "Jason",
    "Jeffrey", "Jeremy", "Jerry", "Jesse", "Jim", "Joe", "John", "Johnny",
    "Jonathan", "Jordan", "Joseph", "Joseph", "Joshua", "Juan", "Justin", "Keith",
    "Kenneth", "Kevin", "Kyle", "Larry", "Lawrence", "Leonard", "Logan", "Louis",
    "Luke", "Marcus", "Mark", "Martin", "Mason", "Matthew", "Michael", "Nathan",
    "Nicholas", "Noah", "Oliver", "Patrick", "Paul", "Peter", "Philip", "Ralph",
    "Randy", "Raymond", "Richard", "Robert", "Roger", "Ronald", "Roy", "Russell",
    "Ryan", "Samuel", "Scott", "Sean", "Sebastian", "Shawn", "Stephen", "Steve",
    "Steven", "Terry", "Thomas", "Timothy", "Todd", "Tom", "Tyler", "Victor",
    "Vincent", "Walter", "Wayne", "William", "Willie", "Zachary",
    # --- female ---
    "Abigail", "Alice", "Amanda", "Amber", "Amy", "Andrea", "Angela", "Ann",
    "Anna", "Ashley", "Barbara", "Betty", "Beverly", "Brenda", "Brittany",
    "Carol", "Carolyn", "Catherine", "Charlotte", "Cheryl", "Christina",
    "Christine", "Cynthia", "Deborah", "Debra", "Denise", "Diana", "Diane",
    "Donna", "Doris", "Dorothy", "Elizabeth", "Ella", "Emily", "Emma", "Evelyn",
    "Frances", "Gloria", "Grace", "Hannah", "Heather", "Helen", "Isabella",
    "Jacqueline", "Jane", "Janet", "Janice", "Jean", "Jennifer", "Jessica",
    "Joan", "Joyce", "Judith", "Judy", "Julia", "Julie", "Karen", "Katherine",
    "Kathleen", "Kayla", "Kelly", "Kimberly", "Laura", "Lauren", "Linda", "Lisa",
    "Lori", "Madison", "Margaret", "Maria", "Marie", "Marilyn", "Martha", "Mary",
    "Megan", "Melissa", "Michelle", "Mia", "Nancy", "Natalie", "Nicole", "Olivia",
    "Pamela", "Patricia", "Rachel", "Rebecca", "Rose", "Ruth", "Samantha", "Sandra",
    "Sara", "Sarah", "Sharon", "Shirley", "Sophia", "Stephanie", "Susan", "Teresa",
    "Theresa", "Tiffany", "Victoria", "Virginia", "Wanda",
}

# --------------------------------------------------------------------------- #
# Matcher factory
# --------------------------------------------------------------------------- #
# Russian case-suffix whitelists — one CLOSED tail set PER declension class,
# never a global union: with a union every stem accepts every class's tails,
# so 'Вер'+ю/ь (Верю, Верь) and 'Ад'+ом (Адом) turn ordinary words into
# PERSON_NAME tokens. A stem only ever matches with the tails of ITS OWN class
# (enforced by a per-stem lookahead in ``first_name_rx``). Deliberately NOT
# [а-яё]* — an open tail matches derivations (Иваново, Никитин, Ольгинск) and
# shreds ordinary morphology. Only "cons" contains the empty tail: a
# vowel/soft-sign-class stem is not a word form by itself (bare 'Ольг' — or
# 'Ад' — must never match). 'ая' is gone entirely: no base-form given name in
# the lists above declines with it (Аглая stems to Агла + я).
_CLASS_TAILS = {
    "cons": ("", "а", "у", "е", "ом"),          # Иван: Иван/Ивана/Иваном
    "y":    ("й", "я", "ю", "е", "ем"),         # Серге-: Сергей/Сергею/Сергеем
    "iy":   ("й", "я", "ю", "и", "ем"),         # Дмитри-: Дмитрий/Дмитрии
    "a":    ("а", "ы", "и", "е", "у", "ой"),    # Ольг-: Ольга/Ольги/Ольгой
    "ya":   ("я", "и", "е", "ю", "ей"),         # Кат-: Катя/Кати/Катей
    "iya":  ("я", "и", "е", "ю", "ей", "ёй"),   # Мари-/Иль-: Марии/Ильёй
    "soft": ("ь", "я", "ю", "е", "и", "ем", "ём", "ью"),  # Игор-/Любов-: Игоря/Любовью
}

# Tail-capture UNION of the class sets above (group 2 of the compiled pattern).
# The per-stem class lookahead does the real gating; this union only needs to
# re-consume whatever tail that lookahead already validated.
_RU_CASE_TAIL = r"(?:ью|ой|ей|ёй|ём|ем|ом|а|я|ы|и|е|у|ю|й|ь|)"

# Stems whose paradigm collides with an ordinary word (verb/noun roots) —
# Вер+у/ы (вера), Ад+а (ад), Мил+ой (милый), Лад+ы (лады), Роз+ы (розы),
# Март+а («8 Марта» / a month), Милан+а/е (a city: «из Милана»), Август+а/е
# (a month: «5 Августа») — a global review: dates/cities next to NAME_CTX were minted
# as PERSON. «Любов» is the real stem of «Любовь» (_stem cuts only the final
# «ь»; the old entry «Люб» did not silence it and was dead): «Любовь»/«Любовью»
# is a common noun at the start of a sentence. «Люб» is kept as protection in case
# RU_FIRST_NAMES_DIM is compiled («Люба»: Люб+ой = «Любой»). Never compiled
# into the matcher; the names they came from (Вера, Ада, Мила, Марта, Милана,
# Августа, Любовь…) are left to the context-anchored rules that carry stronger
# evidence than the bare gazetteer sweep.
_HOMOGRAPH_STEMS = {"Вер", "Ад", "Мил", "Лад", "Люб", "Роз",
                    "Любов", "Март", "Милан", "Август"}

# Right no-fuse guard: CR blocks a following Cyrillic letter; the second
# lookahead blocks Latin letters and digits, so the minted PERSON_NAME token
# (which ends in DIGITS) can never weld onto 'Иван2026' / 'ИванX' — a fused
# 'PERSON_NAME_1331002026' would later parse as a different token, and the
# GLUE_RE structural gate cannot detect digit-on-digit fusion.
_NAME_RIGHT = CR + r"(?![A-Za-z0-9а-яё])"

_VOWELS = "аеёиоуыэюя"


def _stem(name: str) -> str:
    """Inflection-invariant stem of a base-form (nominative) Russian given name.

    Strips the class-forming ending so ``stem + <case tail>`` reconstructs every
    case of the name (and only that name's paradigm):
        -ий / -<vowel>й (Дмитрий, Сергей, Николай)  -> drop the final й
        -ья / -ия       (Илья, Наталья, Мария)      -> drop the final я
        -а / -я         (Ольга, Анна, Никита)        -> drop it
        -ь              (Игорь, Любовь)              -> drop it
        consonant       (Иван, Пётр, Максим)         -> keep whole
    """
    n = name
    if len(n) > 2:
        if n.endswith("й") and (n.endswith("ий") or n[-2] in _VOWELS):
            return n[:-1]
        if n.endswith("я") and n[-2] in ("ь", "и"):
            return n[:-1]
        if n[-1] in ("а", "я", "ь"):
            return n[:-1]
    return n


def _decl_class(name: str) -> str:
    """Declension class of a base-form name — mirrors the ``_stem`` branches,
    so ``_stem(n) + <tail in _CLASS_TAILS[_decl_class(n)]>`` spans exactly the
    name's own paradigm (and only that paradigm)."""
    n = name
    if len(n) > 2:
        if n.endswith("й") and (n.endswith("ий") or n[-2] in _VOWELS):
            return "iy" if n.endswith("ий") else "y"
        if n.endswith("я") and n[-2] in ("ь", "и"):
            return "iya"
        if n[-1] == "а":
            return "a"
        if n[-1] == "я":
            return "ya"
        if n[-1] == "ь":
            return "soft"
    return "cons"


def first_name_rx(names) -> re.Pattern:
    """Compile a matcher for the given base-form Russian names.

    Returns a compiled pattern with two capture groups:
        group 1  -> the inflection-invariant STEM (seed key: one token/name)
        group 2  -> the (optional) case tail
    and group 0 spanning stem+tail. Each stem carries a lookahead restricting
    the tail to its OWN declension class, so cross-class homographs (Верю /
    Верь / Адом / Милая) never match. Cyrillic no-fuse boundaries (CL/CR) plus
    the ``(?![A-Za-z0-9а-яё])`` right guard keep the minted token from welding
    onto a following Russian/Latin letter OR digit ('Иван2026' is left alone
    instead of becoming a corrupted 'PERSON_NAME_1331002026') and block
    open-tail derivations.

    Usage in a Rule:
        Rule("dict_first_name", first_name_rx(RU_FIRST_NAMES),
             mask=lambda m: mint.pname(m.group(1).lower()),
             value_group=0, requires_context=NAME_CTX, confidence="context")
    """
    # stem -> union of the tail sets of every base form sharing that stem
    # (Ян/Яна, Валерий/Валерия legitimately merge two classes on one stem)
    tails_by_stem: dict[str, set] = {}
    for n in names:
        if not (n and n[0].isupper()):
            continue
        s = _stem(n)
        if s in _HOMOGRAPH_STEMS:
            continue
        tails_by_stem.setdefault(s, set()).update(_CLASS_TAILS[_decl_class(n)])
    # longest-first so a short stem can't pre-empt a longer superstring; the
    # per-stem lookahead validates <its class tails + right guard>, then the
    # shared tail group re-consumes exactly the tail the lookahead validated
    # (tails are pure Cyrillic, so the no-fuse boundary pins a unique span).
    # A short stem (≤2 letters: Ан-/Ол-/Юл-/Мы-) with an empty tail would match itself
    # as a word — «Мы работаем» → PERSON_NAME (found on a batch of calls,
    # person-names). An empty tail for such a stem is allowed only if exactly
    # this form is a name from the list («Ян»).
    exact = {n for n in names if _stem(n) == n}
    leaves: dict[str, str] = {}      # stem -> lookahead of its tails
    for s in sorted(tails_by_stem, key=len, reverse=True):
        tset = tails_by_stem[s]
        if len(s) <= 2 and s not in exact:
            tset = tset - {""}
            if not tset:
                continue
        tails = sorted(tset, key=len, reverse=True)  # '' sorts last
        leaves[s] = r"(?=(?:" + "|".join(tails) + r")" + _NAME_RIGHT + r")"
    if not leaves:
        # An empty alternation r"()" matches the EMPTY string at every position —
        # the whole corpus turns into a mush of tokens (found on a smoke run:
        # decorative parts of a discord display_name '4ое]'/'по' do not pass the isupper
        # filter above). Silent damage is forbidden — fail loudly.
        raise SystemExit(
            "first_name_rx: not a single valid base form in "
            f"{sorted(set(names))!r} (a capitalized Cyrillic word is needed) — an empty "
            "alternation would match the empty string at every position")
    # ~3k dictionary stems — a flat alternation would cost 3k comparisons
    # at every position (measured: ×10 for the names pack on 1 MB); a trie splits by
    # the first character, longest-first is kept (the terminal after the children).
    alt = _trie_alternation(leaves)
    return re.compile(CL + r"(" + alt + r")(" + _RU_CASE_TAIL + r")" +
                      _NAME_RIGHT)


def _trie_alternation(leaves: dict) -> str:
    """Stems as a regex trie: a common prefix is one branch, nested ``(?:…)``
    by the next character, a chain of single descendants is glued into a literal.
    The terminal of a stem (lookahead of its tails, ``leaves[stem]``) stands AFTER
    the child branches — a long stem is tried before a short one, like the old
    longest-first. The branches of a node are mutually exclusive by the first character, so
    their order does not affect the result (sorting is for a deterministic pattern
    text). The caller's group semantics do not change: the whole alternation
    is wrapped in group 1 (the stem)."""
    root: dict = {}
    for stem, look in leaves.items():
        node = root
        for ch in stem:
            node = node.setdefault(ch, {})
        node[""] = look

    def emit(node: dict) -> str:
        term = node.get("")
        parts = []
        for ch in sorted(k for k in node if k):
            child, lit = node[ch], ch
            while len(child) == 1 and "" not in child:   # a chain without branching
                (nxt, child), = child.items()
                lit += nxt
            parts.append(re.escape(lit) + emit(child))
        if term is not None:
            parts.append(term)
        return parts[0] if len(parts) == 1 else r"(?:" + "|".join(parts) + r")"

    return emit(root)
