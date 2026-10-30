"""Name + handle detectors.

Personal names are the hardest PII class: short, common, inflected, transliterated,
homoglyph-stylised, hidden in slugs / filenames / non-Latin scripts. The discipline
here mirrors the rest of the pipeline:

  * one entity -> one token: every rule seeds ``pname`` on a *canonical key*
    (lowercase nominative surname / given-name stem) so all cases, spellings and
    transliterations of one person collapse to a single ``PERSON_NAME_x`` token.
  * rare / unique / structurally-anchored forms (full-name pairs, initials+surname,
    the official «Фамилия Имя Отчество» / «Фамилия И.О.» order, long
    uncommon surnames, ``/in/`` slugs) fire with boundary guards only -> 'high'.
  * short / common / weak forms (bare given names, 3-4 letter vocatives, common
    surname roots, translit/leet canonicalisation) require a nearby context cue or
    an intro/vocative frame -> 'context'; the truly global sweeps (bare common
    first-names, global common-surname) are 'low' and off by default.
  * geo homographs (Ивановск / Фёдоровск) are split to GEO *before* the person
    rules run, so a village / oilfield / region never becomes a PERSON.
  * a leet / homoglyph-stylised Cyrillic surname («Иван0в», «Ив@новым»,
    «Достоевск0го») reads back to Cyrillic and seeds on the same nominative
    key as the plain forms — in the context rule, in the given +
    surname pair and in the official order (stylised slot).
  * a minted token must never fuse onto an adjacent letter/digit — CL/CR on the
    Cyrillic side, L/R on the Latin side, plus a re-entrancy guard so an already
    placed ``*_\\d`` token is never re-processed.

NO corpus-specific literals live here (specific surnames / handles / brands belong
to optional ``known_values`` packs). These rules are general morphology, dictionaries
and structural anchors only.
"""
from __future__ import annotations

import functools
import os
import re

from ..engine import mint, name_dicts, placeholders as P
from ..engine.boundary import L, R, CL, CR, _ESC, is_own_token, own_token_re
from ..engine.gazetteer import (
    EN_FIRST_NAMES, RU_FIRST_NAMES, RU_FIRST_NAMES_ALL, first_name_rx,
)
from ..engine.rule import Rule, keep_line_breaks, no_guard
from ..engine.vocab import (
    COMMON_EN,
    DELEET,
    DIMINUTIVES,
    HOMOGLYPH,
    SCRIPT_RANGES,
    SURNAME_ENDINGS,
    TECH_TERMS,
    TRANSLIT,
)

# --------------------------------------------------------------------------- #
# helpers / shared sub-patterns
# --------------------------------------------------------------------------- #

# A Cyrillic capitalised word (name-shaped), hyphen allowed (Римский-Корсаков).
_CAP_CYR = r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
# A Cyrillic surname STEM: capital + letters, no inner hyphen (for morphology).
_STEM_CYR = r"[А-ЯЁ][а-яё]+"

# Case-tail after a surname/given-stem (nominative + oblique inflections). Kept
# tight (a bounded suffix set) so it can't run into the next word.
_TAIL = r"(?:а|у|е|ы|ой|ою|ым|ом|ых|им|его|ого|ему|ому|)"

# Re-entrancy: never treat an already-minted token as a name candidate.
_IS_TOKEN = re.compile(r"[A-ZА-Я][A-ZА-Я0-9]*_\d")

# Service broadcast mentions of a chat (Discord/Slack): @everyone/@here/@channel/
# @all — not nicknames but broadcast markers; masking them as HANDLE is pointless.
_BROADCAST_MENTIONS = {"everyone", "here", "channel", "all"}

# Common Cyrillic words that collide with initial-pair "surnames" — the second
# capitalised word after "И.И." is only a surname if it is NOT one of these.
_NOT_A_SURNAME = {
    "начальник", "директор", "нужно", "можно", "этот", "как", "что", "если",
    "затем", "далее", "также", "поэтому", "однако", "который", "которая",
    "москва", "россия", "компания", "клиент", "заказчик", "договор",
}

# Geo / role marker words that precede a capitalised word but are NOT given names
# (so a "<marker> <cap-surname-shaped>" span is not a person pair).
_GEO_MARKERS = {
    "Село", "Город", "Деревня", "Посёлок", "Поселок", "Станица", "Хутор",
    "Улица", "Проспект", "Область", "Край", "Район", "Республика",
}

# An ordinary Capitalized word before a surname (an adverb/conjunction/role at the start
# of a sentence) is NOT a given name: «Однако Иванов согласовал» was swallowed as a whole pair,
# and «Однако» disappeared from the text (global review). A word from the stop set
# is kept, only the surname is minted (the orgs._opf_mask discipline: keep the structure,
# mask only the name). A superset of _NOT_A_SURNAME.
_PAIR_FIRST_STOP = _NOT_A_SURNAME | {
    "например", "потом", "сначала", "итак", "кстати", "впрочем", "наконец",
    "возможно", "теперь", "сегодня", "завтра", "вчера", "почему", "когда",
    "хотя", "пока", "чтобы", "только", "кроме", "после", "перед", "именно",
    "видимо", "конечно", "правда", "значит", "может", "пусть", "здесь",
    "потому", "вообще", "иначе", "снова", "опять",
}

# Initial pairs that are really abbreviations, not people (и.о., т.е., т.к. …).
_ABBR_INITIALS = {("И", "О"), ("Т", "Е"), ("Т", "К"), ("Т", "Д"),
                  ("Т", "П"), ("И", "Т"), ("Т", "Ч"), ("Н", "Э")}

# Vocative cue words: a diminutive right after one of these (whole-word) is a
# person being addressed. Whole-word left boundary + trailing space/comma.
_VOCATIVE_CUE = re.compile(
    r"(?i)(?<![А-Яа-яЁё])(привет|да|нет|слушай|давай|смотри|тогда|ну|эй|слышь|"
    r"окей|подожди|прости|говорит|типа|блин|короче|дружище|спасибо|здравствуй|"
    r"здравствуйте|алло|извини|извините|скажи|скажите|послушай)[ ,]+$"
)

# Intro frame: name that follows a self-introduction / address frame.
# (?<![A-Za-z]) before the «i'm» branch: without a left boundary the suffix «im» of ANY word
# («interim», «claim», «vim» + space) was read as the «I'm» frame and minted
# the next Capitalized word as PERSON (global review). The frame tail is
# horizontal whitespace [^\S\r\n]: ``\s+$`` crossed the line, and
# the first word of the next message («коллеги\nНапоминаю») went to PERSON.
_INTRO_FRAME = re.compile(
    r"(?i)(меня\s+зовут|мо[её]\s+имя|это\s+я[, ]|коллег[аеиу]|my\s+name\s+is|"
    r"(?<![A-Za-z])i\s*['`]?m|this\s+is)[^\S\r\n]+$"
)

# Non-name words after an intro frame («this is Test», «I'm Admin»): intro_
# frame_name has no dictionary guard, any Capitalized word was taken for a name.
_INTRO_STOP = COMMON_EN | TECH_TERMS | {
    "test", "prod", "sparta", "admin", "root", "here", "back", "sorry", "done",
    "ready", "online", "offline", "good", "fine", "ok", "okay",
}

# Left-window anchor that a person is nearby (for context-tier surname/given).
NAME_CTX = re.compile(
    r"(?i)(фамили|имя|отчеств|ф\.?и\.?о|зовут|господин|госпожа|гражданин|"
    r"сотрудник|менеджер|директор|клиент|автор|подпис|от\s+кого|адресат|"
    # (?<!person_)name: a placed PERSON_NAME_… token must NOT itself serve
    # as a context label — otherwise a repeated apply finishes off neighbouring names and
    # the fixpoint is reached on the 3rd run instead of the 2nd (an E2 smoke on a real
    # corpus: +3 replacements on the second apply). display_name/author_name still work.
    r"contact|(?<!person_)name|author|from|manager|owner|signed|mr\.?|ms\.?|mrs\.?)"
)

# Gazetteer first-name matcher wrapped in the Latin/digit no-fuse pair: the
# gazetteer's own CL/CR + (?![а-яё]) guards only see Cyrillic, but the minted
# PERSON_NAME token is Latin+digits — L / R stop it welding onto an adjacent
# Latin letter or digit (L stays JSONL-aware: a value after an escaped \n
# still matches). The matcher set is RU_FIRST_NAMES_ALL (a core of ~300 +
# the top 3k names of the russiannames dictionary without common-noun homonyms, a trie
# alternation): dict_first_name, the name neighbour of a bare «-ич» (_patronymic_guard)
# and the middle slot of the official order (_official_guard) see Тагир / Гульнара / Айрат.
# RU_FIRST_NAMES (the core) stays the set for classifying parts in packgen.
_FIRST_NAME_RX = re.compile(L + first_name_rx(RU_FIRST_NAMES_ALL).pattern + R)


# Cyrillic leet — visual substitutions of the Russian leet tradition
# (0=о, 3=з, 4=ч, 6=б, 8=в, @=а, $=с; 1/2/5/7/9 — as DELEET via reverse
# translit), so that a token with LEADING Cyrillic is restored into a Cyrillic
# word: «Иван0в» -> «иванов», «Ку3нецов» -> «кузнецов». DELEET does not fit here:
# it is Latin (3=e, 4=a) and would give «куенецов». The table is only for
# Cyrillic tokens; the Latin path (Pol1karpova / bel0va) is DELEET, as before.
_DELEET_CYR = {"0": "о", "1": "и", "2": "з", "3": "з", "4": "ч", "5": "с",
               "6": "б", "7": "т", "8": "в", "9": "г", "@": "а", "$": "с"}
# A Latin look-alike letter -> Cyrillic (reverse HOMOGLYPH): «Пeтpов» / «Иванoв».
_HOMOGLYPH_REV = {lat: cyr for cyr, lat in HOMOGLYPH.items()}
_CYR_LEAD = re.compile(r"[А-Яа-яЁё]")
_HAS_LAT = re.compile(r"[A-Za-z]")


def _cyr_word(token: str) -> str:
    """The Cyrillic reading of a stylised token with leading Cyrillic:
    leet chars -> letters via _DELEET_CYR; Latin look-alike letters -> Cyrillic,
    but ONLY when all Latin letters of the token are look-alikes («Пeтpов», «Иванoв»,
    «Рoмaн» — a Cyrillic word in disguise); otherwise the Latin tail is translit
    («Иvan0v», «Сidorov», «Кuznetsov») and stays Latin. The rest (hyphen,
    dot) is dropped. Lowercase."""
    low = token.lower()
    lat = [ch for ch in low if "a" <= ch <= "z"]
    fold = bool(lat) and all(ch in _HOMOGLYPH_REV for ch in lat)
    out = []
    for ch in low:
        if ch in _DELEET_CYR:
            out.append(_DELEET_CYR[ch])
        elif fold and ch in _HOMOGLYPH_REV:
            out.append(_HOMOGLYPH_REV[ch])
        elif ch.isalpha():
            out.append(ch)
    return "".join(out)


# A word of a full-name slot (a pair / the official order) is stylised — it contains
# a leet char or a Latin look-alike letter.
_STYLISED = re.compile(r"[0-9@$A-Za-z]")


def _slot_cyr(word: str):
    """A word from a full-name slot -> lowercase Cyrillic for morphology and the seed:
    an ordinary word — as is (the hyphen of a double surname is kept);
    a stylised one («Иван0в», «ИВАН0ВА», «Иванoв») — the Cyrillic reading
    _cyr_word; a translit tail («Иvanov») is not read as Cyrillic -> None."""
    if not _STYLISED.search(word):
        return word.lower()
    cyr = _cyr_word(word)
    return None if _HAS_LAT.search(cyr) else cyr


def _canon_lat(token: str) -> str:
    """Homoglyph + de-leet + translit canonicalisation of a stylised token to a
    plain lowercase Latin key (Pol1karpova / bel0va / Рoмaн -> a comparable key).

    A token with LEADING Cyrillic is canonicalised TRANSLIT-first — first
    the Cyrillic reading (_cyr_word: leet -> letters, look-alikes -> Cyrillic),
    then the whole TRANSLIT table; HOMOGLYPH does not take part. The previous order
    (HOMOGLYPH before TRANSLIT: в/н/р/с -> b/h/p/c) gave «Иван0в» -> ``ibah0b``
    and the canon did not pass _LAT_SURN_END — Cyrillic leet was not caught at all.
    Now «Иван0в» / «ИВАН0В» / «Иванoв» / «Пeтpов» -> ``ivanov`` / ``petrov``,
    as for «Ivanov» / «Petrov». Latin and mixed tokens with leading Latin
    — as before (HOMOGLYPH-first: «Кuznetsov» ``kuznecov``, «1van0v» is not a token)."""
    if _CYR_LEAD.match(token):
        s = "".join(TRANSLIT.get(ch, ch) for ch in _cyr_word(token))
    else:
        out = []
        for ch in token.lower():
            if ch in HOMOGLYPH:            # Cyrillic look-alike -> Latin
                out.append(HOMOGLYPH[ch])
            elif ch in TRANSLIT:           # remaining Cyrillic -> Latin translit
                out.append(TRANSLIT[ch])
            elif ch in DELEET:             # leet digit / @ / $ -> letter
                out.append(DELEET[ch])
            elif ch.isalpha():
                out.append(ch)
            # drop everything else (punctuation, stray marks)
        s = "".join(out)
    for a, b in (("kh", "h"), ("ph", "f"), ("ts", "c"), ("zh", "j"), ("yu", "u"),
                 ("ya", "a"), ("sch", "s"), ("sh", "s"), ("ch", "c")):
        s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)    # squeeze doubles
    s = s.replace("y", "i")
    return s


# Canonical Latin keys of stylised-word BLOCKLIST (dictionary words that survive
# canonicalisation but must never mint a PERSON token).
_CANON_STOP = {_canon_lat(w) for w in (
    COMMON_EN | TECH_TERMS | {
        "hello", "world", "please", "thanks", "cheers", "regards", "support",
        "account", "invoice", "welcome", "success", "failure", "example",
    }
)}

# Latin surname-ending signal on a *canonicalised* token (translit collapsed:
# kh->h, y->i, doubles squeezed) — a token whose canon key ends here is likely a
# transliterated Russian / Armenian surname.
# «ckii» is the canon of «-цкий» (Троцкий -> ``trockii``): «skii» was there, but
# «-цкий» passed only in the truncated form «cki» (Trotsky) — Cyrillic
# leet «Тр0цкий» / «Троцк1й» was not caught while «Дост0евский» was.
_LAT_SURN_END = re.compile(
    r"(?:ov|ev|in|[iy]n|ova|eva|ina|ski|skii|skaia|cki|ckii|enko|uk|iuk|chuk|"
    r"ian|iana|ovic|evic|ich)$"
)

# A stop set for Cyrillic leet (translit_leet_name, a token with leading
# Cyrillic). Russian morphology makes «-ов/-ев/-ин/-ина/-овая» an ending of
# ORDINARY words (genitive plural «документ0в», «г0тов», «маг@зин»,
# «н0вая», cities «Р0стов»), not only of surnames — the Latin _CANON_STOP
# (COMMON_EN/TECH_TERMS) does not know them. Keys are the canon (_canon_lat, TRANSLIT-
# first), so any leet substitution in a word is compared with the same key:
# «г0тов» / «гoтов» / «ГОТ0В» -> ``gotov``. Of RU_FUNCTION_WORDS (tools/
# packgen) and RU_COMMON_CAPS (tools/residual) 11 words pass the surname ending
# — they are here; the rest are frequent genitive plurals of the business/
# chat vocabulary, common nouns in -ин/-ина, adjectives in -овая,
# toponyms. The set is separate from _CANON_STOP so as not to suppress Latin pairs
# («Ivan Chasov» in _lat_pair_guard). The Cyrillic list itself
# (_SURNAME_SHAPED_WORDS) is shared by the surname-slot stop set of the official
# full-name order rules (_SURN_SLOT_STOP) — one source of "a common noun with
# a surname ending".
_SURNAME_SHAPED_WORDS = {
    # RU_FUNCTION_WORDS / RU_COMMON_CAPS that pass _LAT_SURN_END
    "готов", "готова", "дизайн", "дизайна", "клиентов", "месяцев", "новая",
    "проектов", "снова", "таков", "часов",
    # short adjectives / adverbs / pronouns
    "здоров", "здорова", "суров", "нова", "один",
    "какова", "такова", "каков", "основ", "основа", "слов", "слова", "голов",
    "голова", "корова", "подкова", "остров", "островов", "покров", "кров",
    "улов", "плов", "блин", "ужин", "джин",
    "клин", "камин", "финн", "аршин", "кувшин", "дельфин", "пингвин",
    "бассейн", "бензин", "витамин", "кофеин", "магазин", "магазинов",
    # chat / technical vocabulary in -ин («Скр1н», «Л0гин», «Ск1ну»)
    "скрин", "скин", "логин", "плагин", "админ", "термин",
    # genitive plural in -ян («-ян» is allowed in the leet path)
    "дворян", "селян", "россиян", "киевлян", "землян",
    "машина", "машин", "кабина", "кабин", "витрина", "картина", "картин",
    "причина", "причин", "половина", "вершина", "глубина", "ширина", "длина",
    "община", "дисциплина", "медицина", "резина", "вакцина", "корзина",
    "мужчина", "мужчин", "женщина", "женщин", "малина", "калина", "рябина",
    "равнина", "долина", "пружина", "плотина", "турбина", "рубин", "сатин",
    "апельсин", "мандарин", "спин", "тын", "былина", "крупина", "низина",
    # adjectives in -овая / -евая (ya->a + squeeze give «ova»/«eva»)
    "готовая", "здоровая", "деловая", "торговая", "почтовая", "цифровая",
    "базовая", "типовая", "грузовая", "легковая", "розовая", "боевая",
    "бытовая", "годовая", "часовая", "целевая", "правовая", "финансовая",
    "налоговая", "столовая", "ценовая", "плановая", "разовая", "массовая",
    "ключевая", "кочевая", "трудовая", "кредитовая", "долевая", "нулевая",
    "рублевая", "рублёвая", "валовая", "сетевая", "силовая", "тепловая",
    "бортовая", "фронтовая", "пиковая",
    # genitive plural — business / chat vocabulary
    "документов", "договоров", "заказов", "товаров", "счетов", "сотрудников",
    "годов", "вопросов", "ответов", "файлов", "результатов", "отчетов",
    "отчётов", "менеджеров", "специалистов", "участников", "номеров",
    "адресов", "телефонов", "размеров", "вариантов", "пунктов", "этапов",
    "планов", "звонков", "продуктов", "объектов", "элементов", "процессов",
    "ресурсов", "сервисов", "серверов", "долларов", "процентов", "километров",
    "метров", "литров", "банков", "офисов", "городов", "домов", "столов",
    "компьютеров", "тестов", "кодов", "логинов", "аккаунтов", "контактов",
    "актов", "чеков", "расходов", "доходов", "налогов", "кредитов", "бонусов",
    "баллов", "очков", "голосов", "отзывов", "постов", "чатов", "каналов",
    "ботов", "скриншотов", "текстов", "символов", "знаков", "языков", "шагов",
    "примеров", "типов", "видов", "случаев", "багов", "релизов", "коммитов",
    "братьев", "разработчиков", "партнеров", "партнёров", "поставщиков",
    "подрядчиков", "заказчиков", "инвесторов", "экспертов", "авторов",
    "директоров", "юристов", "бухгалтеров", "дизайнеров", "программистов",
    "тестировщиков", "аналитиков", "инженеров", "операторов", "курьеров",
    "студентов", "учеников", "фондов", "взносов", "тарифов", "лимитов",
    "статусов", "запросов", "лайков", "репостов", "подписчиков", "просмотров",
    "переходов", "кликов", "визитов", "заходов", "входов", "выходов", "выводов",
    "итогов", "выборов", "выпусков", "форматов", "шаблонов", "макетов",
    "экранов", "блоков", "пакетов", "скриптов", "сайтов", "доменов", "портов",
    "хостов", "кластеров", "контейнеров", "образов", "дисков", "потоков",
    "сигналов", "логов", "алертов", "инцидентов", "тикетов", "фиксов",
    "хотфиксов", "апдейтов", "билдов", "рисков", "плюсов", "минусов", "пиков",
    "трудов", "работников",
    "материалов", "инструментов", "приборов", "механизмов", "рынков", "секторов", "регионов", "районов", "участков",
    "складов", "заводов", "отделов", "департаментов", "филиалов",
    "признаков", "факторов", "графиков", "протоколов", "приказов", "законов", "судов",
    "исков", "штрафов", "сроков", "периодов", "кварталов",
    "рейсов", "маршрутов", "билетов", "тендеров", "лотов", "контрактов",
    # toponyms (city/river/region) with a surname ending
    "ростов", "саратов", "тамбов", "львов", "киров", "псков", "серпухов",
    "кишинев", "кишинёв", "могилев", "могилёв", "харьков", "дмитров", "ковров",
    "азов", "таллин", "берлин", "дублин", "пекин", "сахалин", "женева",
    "нева", "украина", "аргентина", "палестина",
}
_CYR_LEET_STOP = {_canon_lat(w) for w in _PAIR_FIRST_STOP | _SURNAME_SHAPED_WORDS}

# The structural trigger of translit_leet_name, BUILT INTO the pattern (perf): a match
# is allowed only if the token contains (letter·leet[0-9@$]·letter) OR
# a mix of Latin and Cyrillic in one token — exactly what the guard used to check
# after the fact on EVERY word. Keep in sync with the guard branch
# «embedded leet / cyr+latin mix» of the translit_leet_name rule.
# PERF (robustness): previously all three branches stood on ``_LD*`` — on a run
# of letters without spaces (minified/base64-like text) the nested
# ``_LD*…_LD*`` of the «mix» branches gave O(n²) from one position: 100k letters ≈ 30 s
# (Latin) / 38 s (Cyrillic) on ONE rule, × 3 fixpoint passes.
# The token is ≤30 chars anyway (``{2,28}`` + two letters), so the lookahead
# is bounded by the same length: the leet branch is ``_LD{0,27}`` + a triplet (one repetition,
# ≤28 backtracks); the «mix» branches — the first letter of the token sets the alphabet, then
# ``{0,28}`` chars ONLY of this alphabet/digits up to the first letter of the other one —
# the classes do not intersect, there are no nested repetitions, O(30) per position. The set
# of candidates passing the guard is the same: the structure must lie inside
# group(1) (≤30 chars from the match start), and the guard rechecks it.
_LET = r"[A-Za-zА-Яа-яЁё]"
_LD = r"[A-Za-zА-Яа-яЁё0-9@$]"
_TRIGGER = (r"(?=" + _LD + r"{0,27}" + _LET + r"[0-9@$]" + _LET + r"|"
            r"[A-Za-z][A-Za-z0-9@$]{0,28}[А-Яа-яЁё]|"
            r"[А-Яа-яЁё][А-Яа-яЁё0-9@$]{0,28}[A-Za-z])")
# A stylised full-name slot (the surname in fullname_pair_leet / surname_given_
# patronymic / surname_initials, the given name in fullname_pair_leet): a Cyrillic capital,
# a Cyrillic prefix, the first leet char / Latin look-alike letter, then
# any mix, a letter at the end; no hyphen. No lookahead: on an ordinary word
# the prefix ``[А-Яа-яЁё]{0,27}`` hits the boundary and backtracks no more than
# the word length — cheaper than _TRIGGER, and the morphology and reading purity are checked by
# the guard (_leet_surname_nom). The last leet char of a token is always before
# a letter, so _LEET_NEEDS stays a necessary condition of a match.
_LEET_CAP = (r"[А-ЯЁ][А-Яа-яЁё]{0,27}[0-9@$A-Za-z][A-Za-zА-Яа-яЁё0-9@$]{0,27}"
             r"[A-Za-zА-Яа-яЁё]")
_LEET_SLOT = r"(?:" + _LEET_CAP + r")"
# The needs of rules with a stylised token (translit_leet_name, fullname_
# pair_leet): a leet char BETWEEN letters or a Latin/Cyrillic junction (the char
# before the alphabet switch is a letter or a leet char of the same run) — a necessary
# condition of any _TRIGGER branch, and hence of a match.
_LEET_NEEDS = re.compile(r"[A-Za-zА-Яа-яЁё][0-9@$][A-Za-zА-Яа-яЁё]"
                         r"|[A-Za-z0-9@$][А-Яа-яЁё]|[А-Яа-яЁё0-9@$][A-Za-z]")
# needs of rules with a stylised CYRILLIC slot (_LEET_CAP: fullname_pair_
# leet, surname_*_leet, cyr_leet_surname_low) — the prefix of the slot itself; on
# Latin / base64 / prose without leet the rule is skipped entirely.
_LEET_NEEDS_CYR = re.compile(r"[А-ЯЁ][А-Яа-яЁё]{0,27}[0-9@$A-Za-z]")


def _lat_first_key(token: str) -> str:
    """The key of a KNOWN given name for the translit-pair cue: Cyrillic is FIRST
    transliterated with the TRANSLIT table, and only then comes the Latin canon
    (``_canon_lat``: HOMOGLYPH / de-leet / folds). A bare ``_canon_lat`` takes
    HOMOGLYPH before TRANSLIT (в/н/р/с -> b/h/p/c), so the keys of the RU gazetteer
    came out as garbage («Иван» -> ``ibah``, «Анна» -> ``aha``, «Сергей» ->
    ``cepgei``) and the cue on a Russian name in Latin in ``_lat_pair_guard`` did not
    fire: «Ivan Ilyin» / «Sergey Kalinin» were lost. Here «Иван» /
    «Ivan» / «1van» / «Иvan» converge to one ``ivan``. ``x`` -> ``ks``
    (Alexey/Aleksey, Maxim/Максим) and a repeated squeeze after the y->i fold
    (Дмитрий ``dmitrii`` / Dmitry ``dmitri``) — a mirror of ``filenames._lat_key``.
    For pure Latin without ``x`` / ``iy`` — exactly ``_canon_lat`` (seeds and
    the leet detection of translit_leet_name are NOT touched: there the canon of a stylised
    token stays ``_canon_lat``)."""
    s = "".join(TRANSLIT.get(ch, ch) for ch in token.lower())
    return re.sub(r"(.)\1+", r"\1", _canon_lat(s.replace("x", "ks")))


# Canonical Latin keys of common given names (EN gazetteer + translit of the RU
# gazetteer) — used as the first-word seed cue for the CLEAN-translit ФИО pair
# rule ("Ivan Petrov" / "Anna Sidorova" without any leet substitution). The
# surname ending is the primary anchor; the first name in this set strengthens
# it (and is the fallback evidence when a surname ending is borderline).
# The key is built via ``_lat_first_key`` (TRANSLIT -> canon), not via a bare
# ``_canon_lat`` — otherwise the RU half of the set is garbage and the cue stays silent.
_LAT_FIRST_KEYS = {_lat_first_key(w) for w in EN_FIRST_NAMES | RU_FIRST_NAMES_ALL}


def _seed_person(canon: str) -> str:
    # ё/е fold into one key: the gazetteer deliberately keeps both spellings
    # (Артём/Артем, Пётр/Петр, Фёдор/Федор) — without the fold one entity gives two
    # different PERSON_NAME_x (seed drift, global review). The canonical key =
    # the lowercase nominative with ё→е, so «Артём»/«Артем», «Фёдоров»/«Федоров»
    # are minted into ONE token.
    return canon.strip().lower().replace("ё", "е")


# The case tail of a surname/given name in the genitive/instrumental etc. — the same
# set as _TAIL, but for cutting from the END of the captured form, plus «-ыми»
# (instrumental plural «Ивановыми» — the cut is needed by the leet rule and
# initials_surname; it is not in the _TAIL of the patterns — the pair / official order
# do not meet such a form). Longer before shorter, so that «-ого»
# is cut whole, not as «-о».
_TAIL_STRIP = sorted(
    ("а", "у", "е", "ы", "ой", "ою", "ым", "ом", "ых", "им",
     "его", "ого", "ему", "ому", "ыми"),
    key=len, reverse=True,
)
# A surname ending in the nominative: cut the tail only if the REST ends with it —
# otherwise «Иванова»→«Иванов» is fine, and an ordinary «Марина» is not mutilated.
_SURN_END_RE = re.compile(SURNAME_ENDINGS + r"$")


# A double surname in the official order
# and with initials («Петрова-Водкина Анна Сергеевна», «И.И. Петровой-Водкиной») was seeded whole —
# the case-tail cut saw only the second part («петрова-водкин»), and the token matched neither
# the pair «Анна Петрова-Водкина» (fullname_pair_double: «петров-водкин»), nor the masculine form
# «Петров-Водкин Иван» (seed drift of one entity). Now the nominative is taken per part across
# the hyphen (tests/test_gaps_detectors.py).


def _strip_case_tail(surname: str) -> str:
    """Reduce an oblique case of a surname to the nominative, so that «И.И. Иванова» and
    «Иван Иванов» give ONE token (initials_surname seeded group(3) whole —
    seed drift, global review). The tail is cut ONLY when the rest
    ends with a surname suffix (otherwise a name like «Марина» is left alone).
    A double surname — per part."""
    low = surname.lower()
    if "-" in low:
        return "-".join(_strip_case_tail(p) for p in low.split("-"))
    for t in _TAIL_STRIP:
        if low.endswith(t) and len(low) - len(t) >= 3:
            stem = low[: -len(t)]
            if _SURN_END_RE.search(stem):
                return stem
    return low


def _pair_mask(m):
    """A pair «Имя Фамилия» -> ONE token seeded by the surname. If the first
    word is an ordinary Capitalized word from the stop set (adverb/conjunction/role), it
    is kept verbatim and only the surname is minted.
    A triple «Word Name Surname» (groups 4/5 — an optional third
    word with a surname ending). A given name in an oblique case itself looks like
    a surname («Марине» = «Марин» + «е»), and the pair was taken from the first two
    words: «Передайте Марине Смирновой» -> «TOKEN Смирновой» — the surname
    stayed open (a finding on Telegram exports). If the second word is
    a form of a gazetteer name and the first one is not a name, the real pair is
    the second and third word: the first is kept, the token is seeded by the surname.
    Otherwise — the previous behaviour, the third word is returned as it was (the
    next fixpoint pass will take it)."""
    rest = ""
    # the word separator is as in the text (NBSP/U+202F stay themselves)
    sp1 = m.string[m.end(1):m.start(2)]
    if m.group(4):
        if (_FIRST_NAME_RX.fullmatch(m.group(2) + m.group(3))
                and not _FIRST_NAME_RX.fullmatch(m.group(1))):
            return m.group(1) + sp1 + mint.pname(_seed_person(m.group(4)))
        rest = m.string[m.end(3):m.start(4)] + m.group(4) + m.group(5)
    tok = mint.pname(_seed_person(m.group(2)))
    if m.group(1).lower() in _PAIR_FIRST_STOP:
        return m.group(1) + sp1 + tok + rest
    return tok + rest


def _lat_pair_mask(m):
    """A clean translit pair «Имя Фамилия» in Latin -> ONE token seeded by the
    CANON of the surname, so that «Поляков»/«Polyakov» link into one PERSON_NAME."""
    return mint.pname(_canon_lat(m.group(2)))


def _lat_pair_guard(m):
    """True => leave alone. Catch only a real translit full name: the canon key of the SECOND word
    must end with a Latin surname suffix (ov/ova/enko…)
    and not be a dictionary word; with a borderline suffix the first word must
    be a known given name (EN gazetteer ∪ translit of the RU gazetteer)."""
    if _IS_TOKEN.search(m.group(0)):
        return True
    first, surn = _canon_lat(m.group(1)), _canon_lat(m.group(2))
    if surn in _CANON_STOP or first in _CANON_STOP:
        return True
    if not _LAT_SURN_END.search(surn):
        return True
    # a surname suffix is strong evidence; a known given name strengthens it, but
    # is mandatory only for the ambiguous short «-in/-yn» tail (the cities
    # Berlin/Austin), so as not to swallow toponyms. The name key is _lat_first_key:
    # the same canon _LAT_FIRST_KEYS is built with.
    if surn.endswith(("in", "yn")) and _lat_first_key(m.group(1)) not in _LAT_FIRST_KEYS:
        return True
    return False


# --- a bare «-ич» (patronymic): a guard against common-noun homographs ---------- #
# Productive patronymic suffixes are evidence by themselves (as before).
_ICH_PRODUCTIVE = re.compile(r"(?:ович|евич|инич|овна|евна|ична)$")
# Common nouns in «-ич» that surface Capitalized at the start of a sentence
# («Москвич припаркован», «Кирпич лежал») — never a patronymic.
_ICH_COMMON = {
    "москвич", "омич", "томич", "костромич", "кирпич", "кулич", "паралич",
    "гринвич", "сэндвич", "сендвич", "спич",
}


def _patronymic_guard(m):
    """True => leave alone. A bare «-ич» (without -ович/-евич/-инич/-овна/-евна/
    -ична) is a homograph of common nouns: a stop-dictionary word is always silent, the rest
    need a name neighbour on the left («Владимир Ильич», gazetteer) or a person cue
    (NAME_CTX) in the left window — the neighbouring-name requirement in the spirit of
    dict_first_name. Real Ильич/Кузьмич next to a name/label still work."""
    if _IS_TOKEN.search(m.group(0)):
        return True
    w = m.group(1)
    if _ICH_PRODUCTIVE.search(w):
        return False
    if w.lower() in _ICH_COMMON:
        return True
    a = max(0, m.start(1) - 60)
    left = m.string[a:m.start(1)]
    prev = _PATR_PREV.search(left)
    if prev and _FIRST_NAME_RX.fullmatch(prev.group(1)):
        return False
    # (as the engine's context gate, context.left_hit): the window is in the full
    # text, (?<!person_)name sees the text before the window; a slice started inside
    # «PERSON_NAME_…» gave «NAME_…» — an extra mask
    return not NAME_CTX.search(m.string, a, m.start(1))


# --- the official full-name order: «Фамилия Имя Отчество» / «Фамилия И.О.» ------- #
# The standard order of documents / CRM / HR exports; fullname_pair knows
# only «Имя Фамилия», initials_surname — only «И.И. Иванов», so
# «Сотрудник: Иванов Иван Иванович» gave «Иванов PERSON_NAME PERSON_NAME»
# (the surname open, the given name and patronymic — two different tokens), and «Иванов И.И.» and
# «ИВАНОВ ИВАН ИВАНОВИЧ» were not touched at all (review, high); the NER layer without
# natasha does not pick up the surname.
#
# The evidence is structural and morphological, without a surname dictionary: the surname slot is
# a surname ending (SURNAME_ENDINGS + the case tail _TAIL, as in
# fullname_pair; plus the oblique and feminine forms of «-ский/-цкий» — Достоевского /
# Достоевской / Достоевская, which _TAIL does not give after «-ск»; a hyphenated
# double surname is allowed), the patronymic slot is
# a productive suffix (-ович/-евич/-инич/-овна/-евна/-ична with a case
# tail: Ивановича/Ивановичем, Ивановны/Ивановне/Ивановной; Turkic
# «Гусейн оглы» / «Мамед кызы») or a bare «-ич» ONLY with a gazetteer name
# in the middle slot (the _patronymic_guard discipline: a bare «-ич» is a homograph, it
# needs a name neighbour). The middle slot is any Capitalized word: the gazetteer (~300
# names) does not know Тагир/Азамат/Гульнара, and a surname leak costs more; the exception
# is a word that itself looks like a patronymic («Ирина Ивановна Абрамович» — the order
# «Имя Отчество Фамилия», not ours). Three words -> ONE token, the seed is the surname in
# the nominative (_surname_nominative: the case-tail cut as in
# initials_surname, «-ск*» -> «-ский»): the same key as for fullname_pair /
# initials_surname / a bare «Иванов» in display_name — one person, one
# PERSON_NAME. The neuter «-ское» (the village Ивановское — «Ивановс»+«ко»+«е»
# passes SURNAME_ENDINGS) and the feminine «-ая» with a masculine patronymic
# («Ивановская Иван Иванович» — a street + a name) are not taken for a surname.
#
# CAPS («ИВАНОВ ИВАН ИВАНОВИЧ», «ИВАНОВ И.И.») and a CAPS surname with an ordinary
# given name and patronymic («ИВАНОВ Иван Иванович» — signatures, requisites): the same pattern
# bodies raised to upper case (_up), the slots are an alternation of the ordinary and
# CAPS forms. A comma after the surname («Иванов, Иван Иванович» — columns
# «Фамилия, Имя Отчество») is allowed. The separator is ONE space or NBSP
# (Word typography «Иванов И. И.»), never \t / \n: TSV cells and lines do not
# merge (the fullname_pair «[ ]» discipline).
#
# The surname slot is not a word from _SURN_SLOT_STOP (the stop set of the pair's first word +
# common nouns/toponyms with surname morphology + forms of address and roles in
# -ин/-ец/-ук/-ман/-ян, adverbs in -ко, pronominal forms in -их/-ых/-им):
# «Господин Иван Иванович», «Продавец Иван Иванович», «Голова Ивана
# Ивановича», «Из Пскова, Иван Иванович», «Один И.И.» — the word stays, the given name and
# patronymic are picked up by the previous rules (dict_first_name / patronymic), as
# before. A slot WITHOUT surname morphology («Ромашка Ирина Петровна», «Заяц
# Иван Иванович») is invisible to the rule: a stop set of positions/projects of an open
# class («Директор / Проект Иван Иванович») would be leakier than morphology; such
# surnames are left to a literal pack / NER, as before.


def _up(body: str) -> str:
    """The body of a Cyrillic pattern in upper case (CAPS variants). The
    body has no ``\\`` escapes with letters (``\\s`` / ``\\u``) — upper() would
    break them; the L/CL/CR/R boundaries are attached outside."""
    return body.upper()


# The space between full-name words is a regular one, NBSP (U+00A0) or a narrow no-break
# one (U+202F: macOS, web, Word); not \s — not across \t / \n.
_SP = "[ \u00a0\u202f]"
_PAIR_SP = _SP
# neighbours of the same rules: the name to the left of a bare «-ич»
# (_patronymic_guard), the fullname_pair_leet separator — _PAIR_SP; the geo word
# to the left of the official order (_GEO_LEFT) — _SP
_PATR_PREV = re.compile(r"([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)" + _PAIR_SP + r"$")
# oblique and feminine forms of «-ский/-цкий» (SURNAME_ENDINGS knows only
# the nominative «-ский» and «-ской»): Достоевского / Достоевскому / Достоевским /
# Достоевском / Достоевская / Достоевскую
_SK_FORMS = r"(?:ск|цк)(?:ого|ому|им|ом|ая|ую|ие|ых|ими)"
_SURN_CAP = (r"(?:" + _STEM_CYR + r"-)?" + _STEM_CYR + r"(?:" + SURNAME_ENDINGS +
             _TAIL + r"|" + _SK_FORMS + r")")
# the surname slot is a whole word: (1) the ordinary form | CAPS
_SLOT_SURN = r"(" + _SURN_CAP + r"|" + _up(_SURN_CAP) + r")"
# the surname slot is a stylised word («Иван0в Иван Иванович», «ИВАН0В
# И.И.») for the leet twins of the official order rules; morphology and stop
# sets — by the Cyrillic reading (_official_guard -> _leet_surname_nom)
_SLOT_SURN_LEET = r"(" + _LEET_SLOT + r")"
_SLOT_GIVEN = r"(" + _CAP_CYR + r"|" + _up(_CAP_CYR) + r")"
_SK_TAIL = re.compile(r"(?<=ск|цк)(?:ий|ого|ому|им|ом|ая|ой|ую|ие|ых|ими)$")
# patronymic: masculine -ич + tail, feminine -овн/-евн/-ичн + tail, Turkic
# «Гусейн оглы» / «Гусейн-оглы» / «Мамед кызы»
_PATR_CAP = (r"(?:" + _STEM_CYR + r"ич(?:а|у|ем|е|)|" + _STEM_CYR +
             r"(?:овн|евн|ичн)(?:а|ы|е|у|ой|ою)|" + _CAP_CYR + r"(?:" + _SP +
             r"|-)?(?:[Оо]глы|[Кк]ызы|[Уу]лы|[Гг]ызы))")
_SLOT_PATR = r"(" + _PATR_CAP + r"|" + _up(_PATR_CAP) + r")"
# a productive patronymic suffix is evidence by itself; a bare «-ич» — only with
# a gazetteer name in the middle slot
_PATR_PRODUCTIVE = re.compile(r"(?:ович|евич|инич|овн|евн|ичн|оглы|кызы|улы|гызы)")
# a feminine patronymic (agreement with «-ая» in the surname slot)
_PATR_FEM = re.compile(r"(?:овн|евн|ичн)(?:а|ы|е|у|ой|ою)$|(?:кызы|гызы)$")
# a word that itself looks like a patronymic in the given-name slot — not our order
_PATR_SHAPE = re.compile(r"(?:ич|овн|евн|ичн)(?:а|у|ем|е|ы|ой|ою|)$")
# a geo word right to the left of the surname slot: «из Пушкина, Иван Иванович» /
# «в Королёв Иван Иванович» — a city-surname + an address, not a full name. Toponym
# surnames (Пушкин / Чехов / Королёв / Гагарин) are NOT put into the stop set: they are
# frequent surnames, a leak costs more. «г.» is deliberately not a geo label — it is also «год»
# («12.05.2024 г. Иванов Иван Иванович подписал»).
_GEO_LEFT = re.compile(r"(?i)(?<![а-яёa-z])(?:из|в|во|город[аеу]?|сел[оае]|"
                       r"деревн[яеи]|пос\.|ул\.)" + _SP + r"$")

# The stop set of the surname slot (keys — _seed_person: lowercase, ё->е). Compared
# both by the whole form («Голова») and by the stem without the tail («Головы» -> «голов»).
_SURN_SLOT_STOP = {_seed_person(w) for w in _PAIR_FIRST_STOP | _SURNAME_SHAPED_WORDS | {
    # forms of address, roles, common nouns in -ин / -ец / -ук / -ман / -ян before a name
    "господин", "гражданин", "хозяин", "старшина", "кузина", "родина", "грузин",
    "армянин", "татарин", "славянин", "христианин", "мещанин", "дворянин",
    "крестьянин", "горожанин", "осетин", "болгарин", "блондин", "семьянин",
    "продавец", "владелец", "совладелец", "молодец", "отец", "певец", "боец",
    "жилец", "купец", "гонец", "немец", "испанец", "китаец", "японец",
    "американец", "европеец", "украинец", "иностранец", "первенец", "младенец",
    "образец", "конец", "дворец", "умелец", "беглец", "пловец", "чтец",
    "творец", "ловец", "юнец", "малец", "внук", "правнук", "звук", "стук",
    "паук", "барсук", "сундук", "индюк", "каблук", "роман",
    "карман", "туман", "обман", "шаман", "атаман", "боцман", "лоцман",
    "штурман", "гетман", "крестьян", "славян", "армян", "обезьян", "баян",
    "кальян", "бурьян", "изъян",
    # adverbs / numerals in -ко («однако» / «только» are in _PAIR_FIRST_STOP)
    "сколько", "столько", "несколько", "настолько", "нисколько", "далеко",
    "недалеко", "глубоко", "широко", "высоко", "легко", "нелегко", "редко",
    "ярко", "близко", "низко", "резко", "громко", "крепко", "мелко", "гладко",
    "сладко", "жарко", "ловко", "неловко", "робко", "звонко", "жестко",
    "четко", "коротко", "тонко", "ломко", "хрупко", "мягко", "горько",
    "молоко", "яблоко", "облако", "войско",
    # pronominal and adjectival forms in -их / -ых, «один»
    "этих", "наших", "ваших", "своих", "других", "многих", "самих", "самых",
    "каких", "таких", "всяких", "любых", "некоторых", "остальных", "прочих",
    "первых", "вторых", "третьих", "обоих", "обеих", "иных", "новых", "старых",
    "данных", "указанных", "следующих", "уважаемых", "дорогих",
    "ответственных", "молодых", "больших", "малых", "разных", "общих",
    "прошлых", "текущих", "будущих", "нужных", "лучших", "худших", "крупных",
    "мелких", "ближайших", "последних", "отдельных", "основных", "главных",
    "важных", "ключевых", "целевых", "рабочих", "домашних", "личных",
    "один", "одних",
    # adjectives in -ский/-цкий (compared by the nominative «-ий»,
    # so one form covers Русского / Русская / Русской)
    "русский", "российский", "московский", "советский", "детский",
    "городской", "сельский", "гражданский", "банковский", "заводской",
    "морской", "английский", "немецкий", "французский", "польский",
    "турецкий", "украинский", "белорусский", "казахский", "татарский",
    "еврейский", "мужской", "женский", "воинский", "царский", "людской",
}}


def _surname_nominative(surname: str) -> str:
    """A surname from the slot -> the lowercase nominative for the seed: «-ск*/-цк*» in any
    form (Достоевского / Достоевской / Достоевская) -> «достоевский» — the
    same key as for fullname_pair «Иван Достоевский»; the rest — the case-tail
    cut, as in initials_surname (_strip_case_tail: Ивановой ->
    «иванов», Черных -> «черных»). CAPS is lowercased right here. A double
    surname — per part («Римской-Корсаковой» -> «римский-корсаков»)."""
    low = surname.lower()
    if "-" in low:
        return "-".join(_surname_nominative(p) for p in low.split("-"))
    if _SK_TAIL.search(low):
        return _SK_TAIL.sub("ий", low)
    return _strip_case_tail(low)


def _official_mask(m, leet=False):
    """The whole span -> ONE token by the surname in the nominative; a stylised slot
    (leet=True) — by the Cyrillic reading: «Иван0в» and «Иванов» are
    one key."""
    surn = _slot_cyr(m.group(1)) if leet else m.group(1)
    return mint.pname(_seed_person(_surname_nominative(surn)))


def _official_guard(m, given=None, patr=None, leet=False):
    """True => leave alone: an already minted token in the span; a word from
    _SURN_SLOT_STOP in the surname slot (whole or the nominative); a stylised
    slot (leet=True) — a translit tail, not a surname form or a stop by
    _leet_surname_nom; the neuter «-ское»; a geo word on the left (_GEO_LEFT);
    a feminine «-ая» with a masculine patronymic; the given-name slot itself looks like a patronymic;
    a bare «-ич» without a gazetteer name in the given-name slot (a CAPS name is checked against
    the gazetteer in Capitalized form). An ordinary slot goes the previous way without
    decoding (perf: dense CAPS/full-name text — tens of thousands of matches)."""
    if _IS_TOKEN.search(m.group(0)):
        return True
    if leet:
        surn = _slot_cyr(m.group(1))
        if surn is None or _leet_surname_nom(m.group(1)) is None:
            return True
    else:
        surn = m.group(1).lower()
        if _seed_person(surn) in _SURN_SLOT_STOP \
                or _seed_person(_surname_nominative(surn)) in _SURN_SLOT_STOP:
            return True
    if surn.endswith(("ское", "цкое")) \
            or _GEO_LEFT.search(m.string[max(0, m.start() - 12):m.start()]):
        return True
    if given is None:
        return False
    if surn.endswith("ая") and not _PATR_FEM.search(patr.lower()):
        return True
    if _PATR_SHAPE.search(given.lower()):
        return True
    if not _PATR_PRODUCTIVE.search(patr.lower()):
        cap = given if given[1:].islower() else given.capitalize()
        return not _FIRST_NAME_RX.fullmatch(cap)
    return False


# --- leet surname — oblique cases, pair slots and the official order ----------- #
# translit_leet_name caught Cyrillic leet only in the nominative and the
# feminine «-а» («Иван0в», «Иван0ва»); «Иван0ву», «Ив@новым», «Иван0вой»,
# «Кузнец0вым», «Достоевск0го» stayed open, and «Иван Иван0в» /
# «Иван0в Иван Иванович» were taken neither by fullname_pair, nor by the official order
# (a leet char in the surname slot), nor by the leet rule without a context label. Now:
#   * an oblique case of a leet surname -> the same nominative as for fullname_pair /
#     initials_surname / the official order (_surname_nominative): «Иванов»,
#     «Иван0в», «Иван0ву», «Ив@новым», «Иван0выми» — ONE PERSON_NAME;
#   * a tail is allowed only in combination with an ending for which a SURNAME
#     has such a form (_leet_nominative), and the stem of an adjective / common
#     noun in an oblique form is a stop (_CYR_LEET_OBLIQUE_STOP): exactly there
#     «Б0евой» / «Байт0вого» / «Б0левым» gave +39 FP on docs prose;
#   * a stylised surname slot in fullname_pair_leet (the given name must be in the
#     gazetteer — weaker evidence than for an ordinary pair) and in the leet twins
#     of the official order surname_given_patronymic_leet / surname_initials_
#     leet — the same _leet_surname_nom, the seed is the Cyrillic nominative.
# Without a context label a bare leet surname is still not a candidate: the genitive
# plural Capitalized («П0дарков», «К0тов», «Ст0лбов») is an open
# class, the stop set does not close it.

# Tails by the declension type of a surname. The possessive type (-ов/-ев/-ёв/-ин/-ын):
# Иванов / Иванова / Иванову / Ивановым / Иванове / Ивановой / Ивановою /
# Ивановы / Ивановых / Ивановыми; «-ого/-ому/-ом/-ую/-ые/-им» with such a stem
# exist only for an adjective (Байт0вого / Б0левому / Тепл0вом / Н0вую) —
# rejected (the price — foreign «Дарвином» / «Чаплином»). The noun
# type (-ук/-юк/-чук/-ец/-ян/-ман/-берг/-штейн/…): Шевчук / Шевчука / Шевчуку /
# Шевчуком / Шевчуке / Шевчуки. «-ский/-цкий» — all adjectival forms
# (_SK_TAIL); «-енко/-ко/-ых/-их/-дзе/-швили» are not declined (the tail is empty —
# _strip_case_tail does not cut them: the rest has no surname ending).
_POSSESSIVE_END = re.compile(r"(?:ов|ев|ёв|ин|ын)$")
_POSSESSIVE_TAILS = frozenset(("", "а", "у", "ым", "е", "ой", "ою", "ы", "ых", "ыми"))
_NOUN_TAILS = frozenset(("", "а", "у", "ом", "е", "ы"))
# «-ический / -еский / -ческий» is a productive adjectival suffix of business
# prose (технический, коммерческий, юридический, математический), there are practically no
# surnames with it; in the leet path it is rejected in any form («Техн1ческие»,
# «Коммерческ0го» next to a person cue — OCR noise, not a surname). «-овский /
# -енский / -инский» are the most frequent surname types, they still work (Достоевский,
# Раменский, Белинский); frequent adjectives of these types are in
# _SURN_SLOT_STOP (московский, банковский, …).
_SK_ADJ_SUFFIX = re.compile(r"(?:ическ|ческ|еск)(?:ий|ого|ому|им|ом|ая|ой|ую|ие|ых|ими)$")


def _leet_nominative(cyr: str):
    """A Cyrillic reading (lowercase) -> (the surname nominative, whether the form is
    oblique) or None if the form is not a surname one: the rest after the tail cut
    carries a surname ending (SURNAME_ENDINGS, as in fullname_pair), the tail is
    a case form of a SURNAME of this declension type; «-ск*/-цк*» in any form ->
    «-ский» (as in _surname_nominative), except the adjectival «-ическ/-еск»."""
    if _SK_TAIL.search(cyr):
        if _SK_ADJ_SUFFIX.search(cyr):
            return None
        return _SK_TAIL.sub("ий", cyr), not cyr.endswith(("ий", "ой"))
    stem = _strip_case_tail(cyr)
    if not _SURN_END_RE.search(stem):
        return None
    tail = cyr[len(stem):]
    allowed = _POSSESSIVE_TAILS if _POSSESSIVE_END.search(stem) else _NOUN_TAILS
    if tail not in allowed:
        return None
    return stem, bool(tail)


# A stop of OBLIQUE forms of a leet surname (keys — _seed_person of the nominative). In
# the nominative an adjective in -овый/-евый/-иный and a surname in -ов/-ев/-ин do not
# coincide (Розов / розовый), in oblique cases — letter for letter (Р0зовым /
# Б0евой / Ключ3вых / Ед1ными), as do common nouns in -ин/-ина (Маг@зину /
# Маш1ной / Дл1не), so the stem is a veto ONLY for a form with a tail; in
# the nominative only whole-word stops apply (_CYR_LEET_STOP and the surname
# slot _SURN_SLOT_STOP). The stems are derived from the stop set of the surname
# slot (_SURN_SLOT_STOP ⊇ _SURNAME_SHAPED_WORDS ∪ _PAIR_FIRST_STOP:
# готовая -> готов, машина -> машин, нова -> нов, столовая -> столов) and from the
# adjectives / common nouns below (in the same feminine form / nominative
# as the _SURNAME_SHAPED_WORDS list). The price: rare surnames homographic to a stem (Розов / Боев /
# Рублёв / Турбин / Крупин) are not minted in an oblique leet form — in the nominative
# and in a pair / the official order in ordinary Cyrillic they still work.
_OBLIQUE_SHAPED_WORDS = {
    # adjectives in -овая / -евая (business / technical / everyday vocabulary)
    "болевая", "байтовая", "битовая", "тестовая", "текстовая", "файловая",
    "звуковая", "видовая", "фоновая", "кодовая", "числовая", "весовая",
    "ходовая", "пусковая", "дисковая", "рисковая", "лицевая", "передовая",
    "ролевая", "продуктовая", "домовая", "дворовая", "садовая", "полевая",
    "ветровая", "газовая", "паровая", "световая", "цветовая", "теневая",
    "дождевая", "речевая", "мозговая", "лобовая", "носовая", "ножевая",
    "мировая", "классовая", "ранговая", "вековая", "ледовая", "меховая",
    "пуховая", "шелковая", "шёлковая", "сырьевая", "угловая", "круговая",
    "боковая", "концевая", "краевая", "кольцевая", "строевая", "гостевая",
    "вкусовая", "курсовая", "ростовая", "сортовая", "паевая", "кладовая",
    "дуговая", "шаговая", "стартовая", "призовая", "игровая", "медовая",
    "ореховая", "вишневая", "вишнёвая", "малиновая", "персиковая", "березовая",
    "берёзовая", "дубовая", "сосновая", "еловая", "кедровая", "липовая",
    "кленовая", "буровая", "тяговая", "рулевая", "маркетинговая", "лизинговая",
    "брендовая", "флаговая", "языковая", "знаковая", "аналоговая", "растровая",
    "кадровая", "групповая", "долларовая", "евровая", "биржевая", "оптовая",
    "дебетовая", "залоговая", "чековая", "кассовая", "стеновая", "половая",
    "лифтовая", "мостовая", "портовая", "спиртовая", "зерновая", "бобовая",
    "фруктовая", "снеговая", "штормовая", "грозовая", "ливневая",
    # adjectives in -иная (possessives from animals) and «единая»
    "единая", "куриная", "гусиная", "лошадиная", "звериная", "змеиная",
    "мышиная", "орлиная", "утиная", "козлиная", "крысиная", "пчелиная",
    "соловьиная", "голубиная", "лебединая", "петушиная", "ослиная", "тигриная",
    "львиная", "воробьиная", "журавлиная", "комариная", "муравьиная", "осиная",
    "совиная",
    # common nouns in -ин (oblique forms «-ина/-ину/-ине/-ины»)
    "керосин", "вазелин", "аспирин",
    "инсулин", "адреналин", "глицерин", "желатин", "никотин", "протеин",
    "гемоглобин", "ванилин", "маргарин", "парафин", "нафталин", "формалин",
    "гоблин", "муслин", "габардин", "жасмин", "розмарин", "тмин", "карантин",
    "серпантин", "балдахин", "графин", "мокасин", "властелин", "паладин",
    "исполин", "мезонин",
    # common nouns in -ина (oblique forms «-ину/-ине/-ины/-иной»)
    "гардина", "перина", "горошина", "жемчужина", "льдина", "трещина",
    "морщина", "паутина", "скотина", "лучина", "маслина", "смородина",
    "осина", "древесина", "холстина", "парусина", "стремнина", "теснина",
    "пучина", "толщина", "величина", "старина", "дружина", "чужбина",
    "годовщина", "четвертина", "десятина", "детина", "дубина", "образина",
    "махина", "громадина", "домина", "холодина", "рыбина", "тина", "вина",
    "кручина", "лавина", "доктрина", "рутина", "гильотина", "балерина",
    "ветчина", "конина", "свинина", "говядина", "баранина", "телятина",
    "курятина", "осетрина", "лососина", "буженина", "солонина", "пластина",
    "платина", "щетина", "хворостина", "ложбина", "впадина", "котловина",
    "середина", "сердцевина", "целина", "вотчина", "личина", "лещина",
    "крушина", "бузина", "мешковина",
}


def _oblique_stop_keys(words) -> set:
    keys = set()
    for w in words:
        w = _seed_person(w)
        keys.add(w)
        keys.add(_strip_case_tail(w))          # машина -> машин, нова -> нов
        if w.endswith("ая"):
            keys.add(w[:-2])                   # готовая -> готов, боевая -> боев
    return keys


_CYR_LEET_OBLIQUE_STOP = _oblique_stop_keys(_SURN_SLOT_STOP | _OBLIQUE_SHAPED_WORDS)


@functools.lru_cache(maxsize=4096)
def _leet_surname_nom(word: str):
    """A surname slot (stylised or ordinary) -> the lowercase nominative or
    None: a purely Cyrillic reading (_slot_cyr), surname morphology with
    an allowed tail (_leet_nominative), not a stop word of the surname slot
    (the whole form and the nominative — _SURN_SLOT_STOP), in an oblique form — not the stem
    of an adjective / common noun (_CYR_LEET_OBLIQUE_STOP). A pure function
    of the word — lru_cache: a surname repeats in the text, four passes over the
    token (reading, tail cut, canon, stops) are computed once."""
    cyr = _slot_cyr(word)
    if cyr is None:
        return None
    nf = _leet_nominative(cyr)
    if nf is None:
        return None
    nom, oblique = nf
    if _seed_person(cyr) in _SURN_SLOT_STOP \
            or _seed_person(nom) in _SURN_SLOT_STOP:
        return None
    if oblique and _seed_person(nom) in _CYR_LEET_OBLIQUE_STOP:
        return None
    return nom


def _leet_pair_guard(m):
    """True => leave alone. The given name (stylised or ordinary) reads as Cyrillic
    and is in the RU gazetteer — with leet in a pair the evidence is weaker than for fullname_pair
    («Отдел Заказ0в» / «Файл Ба3ов» — not a pair); the surname — _leet_surname_nom."""
    if _IS_TOKEN.search(m.group(0)):
        return True
    given = _slot_cyr(m.group(1) or m.group(3))
    if given is None or not _FIRST_NAME_RX.fullmatch(given.capitalize()):
        return True
    return _leet_surname_nom(m.group(2) or m.group(4)) is None


def _leet_pair_mask(m):
    """A pair with a stylised word -> ONE token by the Cyrillic
    nominative of the surname (the same key as for fullname_pair / «фамилия Иван0в»)."""
    return mint.pname(_seed_person(_leet_surname_nom(m.group(2) or m.group(4))))


# translit_leet_name: the seed and the guard. The structural requirement of the rule — a leet char
# BETWEEN letters or a Cyrillic+Latin mix in one token — is the same as built into _TRIGGER
# (kept in the guard too as a cheap recheck of the canon).
_LEET_STRUCT = re.compile(r"[A-Za-zА-Яа-яЁё][0-9@$][A-Za-zА-Яа-яЁё]")


def _leet_seed(token: str) -> str:
    """The seed of translit_leet_name. A token with leading Cyrillic whose
    Cyrillic reading (_cyr_word) is pure Cyrillic («Иван0в», «ИВАН0В»,
    «Иванoв», «Пeтpов», «Иван0ву», «Достоевск0го») is seeded like the Cyrillic
    rules: the lowercase nominative with ё->е (_seed_person + _surname_nominative,
    as fullname_pair / initials_surname / the official order) — «Иванов» /
    «Иван0в» / «Иван0ва» / «Иван0ву» / «Ив@новым» / «И.И. Иванова» give ONE
    PERSON_NAME. Translit tokens («Иvan0v», «Сidorov», «Petr0v», «bel0va») —
    the Latin canon, as before (linking with «Ivanov» / «Sidorov» from
    translit_fullname_pair)."""
    if _CYR_LEAD.match(token):
        cyr = _cyr_word(token)
        if not _HAS_LAT.search(cyr):
            return _seed_person(_surname_nominative(cyr))
    return _canon_lat(token)


def _leet_guard(m):
    """True => leave alone. The structure (leet between letters / Cyrillic+Latin mix), a canon
    with a surname ending (_LAT_SURN_END) and not a dictionary word (_CANON_STOP)
    — as before. A token with leading Cyrillic and a PURELY Cyrillic reading
    («Иван0в», «Иванoв») — the Cyrillic path _cyr_leet_reject; a translit tail
    («Иvan0v», «сidorov», «Гotov») goes the Latin path, its canon is
    additionally not in _CYR_LEET_STOP."""
    tok = m.group(1)
    if _IS_TOKEN.search(m.group(0)):
        return True
    if not (_LEET_STRUCT.search(tok)
            or (_HAS_LAT.search(tok) and _CYR_LEAD.search(tok))):
        return True
    cyr_lead = bool(_CYR_LEAD.match(tok))
    if cyr_lead:
        cyr = _cyr_word(tok)
        if not _HAS_LAT.search(cyr):
            return _cyr_leet_reject(tok, cyr)
    canon = _canon_lat(tok)
    return (not _LAT_SURN_END.search(canon) or canon in _CANON_STOP
            or (cyr_lead and canon in _CYR_LEET_STOP))


@functools.lru_cache(maxsize=4096)
def _cyr_leet_reject(tok: str, cyr: str) -> bool:
    """The Cyrillic path of translit_leet_name (oblique cases included).
    A pure function of the token — lru_cache (see _leet_surname_nom): on dense
    leet text the guard is called for every match, token repeats are free.
    (1) only Capitalized (Иван0в / ИВАН0В — a surname form): a lowercase
    «документ0в» / «г0тов» / «маг@зин» is a genitive plural and
    common nouns with a surname ending, OCR noise («0» for «о») is full of them
    next to any person cue; Latin has no such morphology, there lowercase
    bel0va / k0zlov still work; (2) a surname by the CYRILLIC morphology of the nominative
    after cutting the case tail (_leet_nominative: SURNAME_ENDINGS, the tail is
    a surname form of this declension type; a productive patronymic -ович/-евич is
    evidence by itself) AND by the Latin canon of the nominative (_LAT_SURN_END):
    the Latin canon alone lets through «Н0вая» (novaia->nova), «Диз@йна»,
    «Авт0матически», the Cyrillic «-ко/-ец/-ых/-ман» alone — «Ябл0ко»,
    «К0нец», «Д@нных», «Рoмaн»; the intersection holds both classes; (3) the canon
    of the form and the canon of the nominative are not in _CYR_LEET_STOP (Capitalized/CAPS common nouns
    and toponyms: «Заказ0в», «Сл0ва», «Р0СТОВ», «Берл1н») and not in _CANON_STOP,
    the nominative is not in the surname-slot stop set (_SURN_SLOT_STOP: forms of address,
    roles, -ский adjectives — «Господ1н», «Русск1й», «М0сковским»);
    (4) an oblique form — the nominative is not in _CYR_LEET_OBLIQUE_STOP (stems
    of adjectives / common nouns: «Б0евой», «Б0левым», «Маш1ной»)."""
    if not tok[0].isupper() or _canon_lat(cyr) in _CYR_LEET_STOP:
        return True
    if _ICH_PRODUCTIVE.search(cyr):
        nom, oblique = cyr, False
    else:
        nf = _leet_nominative(cyr)
        if nf is None:
            return True
        nom, oblique = nf
    canon = _canon_lat(nom)
    # «-ян/-янц» (Armenian surnames): TRANSLIT «я» -> «ya» -> the fold «a» loses
    # «i», the Latin suffix «ian» does not reach the canon («саркисян» ->
    # ``sarkisan``); the Cyrillic morphology here is unambiguous — common nouns
    # in -ян (баян, кальян, армян, россиян, …) are in _SURN_SLOT_STOP.
    if not (_LAT_SURN_END.search(canon) or nom.endswith(("ян", "янц"))) \
            or canon in _CANON_STOP or canon in _CYR_LEET_STOP:
        return True
    key = _seed_person(nom)
    return key in _SURN_SLOT_STOP or (oblique and key in _CYR_LEET_OBLIQUE_STOP)


# --- display_name_field: a value with already minted tokens inside ----------- #
# NAME_CTX sees the field key («…name»), so dict_first_name / patronymic
# manage to mint a name INSIDE the value before the field rule, and the previous guard
# _IS_TOKEN discarded such a value entirely — «Петров Иван» left
# «Петров» open (review). Now the tokens inside the value are kept, and
# each remaining plaintext run is minted with its own token (the seed is the run itself:
# «Петров» gives the same key as fullname_pair). A repeated pass: the runs are
# separators only, nothing to mint, the guard is silent (idempotency).
_DN_STOP = COMMON_EN | TECH_TERMS | {"null", "none", "true", "false", "admin",
                                     "bot", "system", "deleted", "unknown"}
_DN_EDGE = re.compile(r"^([^A-Za-zА-Яа-яЁё0-9]*)(.*?)([^A-Za-zА-Яа-яЁё0-9]*)$",
                      re.S)


def _dn_token_re():
    """A token inside the value — only ours, as a separate word
    («QWERTY_2001» / «SUPER_USER_2001» is a nickname, minted like a
    lowercase one)."""
    return own_token_re(word=True)


def _dn_maskable(core: str) -> bool:
    """A display_name value run is worth minting: not shorter than 4, not a number, not
    a stop word. WITHOUT an .isupper() branch: «IVAN PETROV» / «ИВАНОВ» are the same full names
    in caps (a leak, a major of the global review); acronyms are covered by len<4 and
    the stop dictionary (lower() is already case-insensitive)."""
    return not (len(core) < 4 or core.isdigit() or core.lower() in _DN_STOP)


def _dn_runs(val: str):
    """((prefix, core, suffix), token) over the value: a plaintext run between
    already minted tokens, the edge non-letter chars of the run go out
    verbatim; the last element is the tail with an empty token."""
    pos = 0
    for t in _dn_token_re().finditer(val):
        yield _DN_EDGE.match(val[pos:t.start()]).groups(), t.group(0)
        pos = t.end()
    yield _DN_EDGE.match(val[pos:]).groups(), ""


def _display_name_mask(m):
    val = m.group(2)
    if not _dn_token_re().search(val):
        return m.group(1) + mint.pname(_seed_person(val)) + m.group(3)
    out = []
    for (lead, core, trail), tok in _dn_runs(val):
        if core and _dn_maskable(core):
            core = mint.pname(_seed_person(core))
        out.append(lead + core + trail + tok)
    return m.group(1) + "".join(out) + m.group(3)


def _display_name_guard(m):
    """True => leave alone: nothing to mint — a stop word / short / a number
    as a whole or not a single suitable run between tokens."""
    val = m.group(2)
    if not _dn_token_re().search(val):
        return not _dn_maskable(val.strip())
    return not any(core and _dn_maskable(core)
                   for (_, core, _), _ in _dn_runs(val))


# orgs hands «ИП <Фамилия>» over here (opf_name does not take it), but there was no such
# rule — «ИП Петров», «ИП Сидоренко Ольга», «индивидуальный предприниматель Кузнецов»
# stayed open. ip_surname: the label «ИП» as a whole word CASE-SENSITIVE (not added to
# NAME_CTX: with (?i) «ип» is inside «тип», «принцип») or «индивидуальн… предпринимател…»; then
# a surname with a surname ending (the fullname_pair slot) + optionally a given name (and patronymic) or
# initials. The label stays, the full name is one token, the seed is the surname nominative (as for the pair and
# initials_surname).
_IP_SURNAME = re.compile(
    # on the left — also a JSONL escape (boundary._ESC: «инфо\nИП Петров» in
    # raw .jsonl stayed open — the letter «n» of the escape was taken for a neighbour)
    r"(?:" + _ESC + r"|(?<![А-Яа-яЁёA-Za-z0-9_]))"
    r"((?:ИП|[Ии]ндивидуальн[а-яё]*" + _PAIR_SP + r"+предпринимател[а-яё]*)"
    + _PAIR_SP + r"+)"
    r"(" + _STEM_CYR + SURNAME_ENDINGS + _TAIL
    + r"(?:" + _PAIR_SP + r"[А-ЯЁ]\.[  ]?[А-ЯЁ]\.|" + _PAIR_SP + _STEM_CYR
    + r"(?:" + _PAIR_SP + _STEM_CYR + r"(?:вич|вна|ична|чна|оглы|кызы)[а-яё]*)?)?)"
    + CR + R)


def _ip_surname_mask(m):
    surname = re.split(r"[   ]", m.group(2), maxsplit=1)[0]
    return m.group(1) + mint.pname(_seed_person(_strip_case_tail(surname)))


def _ip_surname_guard(m) -> bool:
    return bool(_IS_TOKEN.search(m.group(2)))


# The surname slot of fullname_pair has no hyphen, and «Мария Иванова-Петрова» gave
# «PERSON_NAME_…-Петрова» (the second part open, the seed by the first — the same token as for «Иван
# Иванов»). fullname_pair_double stands BEFORE fullname_pair and takes only a pair with a double
# surname: the same geometry + «-Фамилия», the seed is both parts (the second — without the case tail:
# «Иванова-Петрова» and «Ивановой-Петровой» — one key «иванов-петров»), a stop word as the first
# word stays, as in _pair_mask.


def _pair_double_mask(m):
    sp1 = m.string[m.end(1):m.start(2)]
    tok = mint.pname(_seed_person(m.group(2) + "-" + _strip_case_tail(m.group(4)[1:])))
    if m.group(1).lower() in _PAIR_FIRST_STOP:
        return m.group(1) + sp1 + tok
    return tok


def _pair_double_guard(m) -> bool:
    return bool(_IS_TOKEN.search(m.group(0))) or m.group(1) in _GEO_MARKERS


# --------------------------------------------------------------------------- #
# RULES  (ordered: most-specific / longest / geo-before-person first)
# --------------------------------------------------------------------------- #
RULES: list[Rule] = [
    # «ИП Петров» / «индивидуальный предприниматель Кузнецов Иван» (see _IP_SURNAME)
    Rule("ip_surname", _IP_SURNAME, mask=_ip_surname_mask, value_group=2, confidence="high",
         guard=_ip_surname_guard,
         needs=re.compile(r"ИП|(?i:предпринимател)")),
    # «ИП Фамилия И.О.» with a surname WITHOUT surname morphology («ИП Бондарь Р.С.», «ИП Ким В.Р.»):
    # a pair of initials after «ИП» is evidence by itself; the seed is the surname nominative, the
    # same token as for a register cell (structured.tabular.party_value: «ИП » + PERSON_NAME)
    Rule("ip_surname_initials",
         re.compile(r"(?:" + _ESC + r"|(?<![А-Яа-яЁёA-Za-z0-9_]))(ИП" + _PAIR_SP + r"+)"
                    r"([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?" + _PAIR_SP + r"+[А-ЯЁ]\.[  ]?[А-ЯЁ]\.)"),
         mask=lambda m: m.group(1) + mint.pname(_seed_person(_surname_nominative(
             m.group(2).split()[0]))),
         value_group=2, confidence="high",
         guard=lambda m: bool(_IS_TOKEN.search(m.group(2))),
         needs=re.compile(r"ИП")),

    # === surname -> GEO split (roots that are ALSO place names) ===
    # Runs FIRST so villages / oilfields / regions (…овск / …овк / …овское)
    # collapse to GEO, never becoming a PERSON in the person rules below.
    # ё and е fold together.
    Rule("surname_geo_split",
         re.compile(L + CL + r"(" + _STEM_CYR + r"(?:ов|ев|[её]в|ин)(?:ск|к)"
                    r"(?:ое|ий|ая|ого|ому|ом|е|а|у|)|" + _STEM_CYR +
                    r"градск(?:ий|ая|ое|ом|))" + CR + R),
         mask=lambda m: P.GEO, value_group=0, confidence="low",
         guard=lambda m: bool(_IS_TOKEN.search(m.group(0))),
         # the suffix (ов|ев|ёв|ин)+(ск|к) or «градск» is a mandatory
         # part of the match (the stem is lowercase Cyrillic, the case is exact)
         needs=("овск", "овк", "евск", "евк", "ёвск", "ёвк", "инск", "инк",
                "градск")),

    # === official order / surname + initials with a STYLISED surname ===
    # «Иван0в Иван Иванович», «ИВАН0ВА Мария Ивановна», «Иван0в, Иван Иванович»,
    # «Иван0в И.И.», «ИВАН0В И. И.»: the same bodies as surname_given_patronymic /
    # surname_initials, but the surname slot is a stylised word (_SLOT_SURN_LEET);
    # morphology, the case tail and stops — by the Cyrillic reading
    # (_official_guard(leet=True) -> _leet_surname_nom), the seed is the same nominative
    # (_official_mask(leet=True) -> _slot_cyr): «Иван0в Иван Иванович» and
    # «Иванов И.И.» are one PERSON_NAME. Separate rules, not a third branch of
    # _SLOT_SURN: needs=_LEET_NEEDS_CYR skips them entirely on text without leet,
    # dense CAPS/full-name text does not pay for an extra branch, the ordinary
    # official-order rules are byte-for-byte as they were. They stand BEFORE the ordinary ones: «@» / «$» /
    # a digit inside a token is not a letter for CL, and the ordinary rule would otherwise bite off
    # the tail of a stylised word («П@ТРОВ ИВАН ИВАНОВИЧ» -> «П@PERSON_NAME»).
    Rule("surname_given_patronymic_leet",
         re.compile(r"(?=[А-ЯЁ])" + L + CL + _SLOT_SURN_LEET + r",?" + _SP +
                    _SLOT_GIVEN + _SP + _SLOT_PATR + CR + R),
         mask=lambda m: _official_mask(m, leet=True),
         value_group=0, confidence="high",
         guard=lambda m: _official_guard(m, m.group(2), m.group(3), leet=True),
         needs=_LEET_NEEDS_CYR),       # the stylised slot is mandatory
    Rule("surname_initials_leet",
         re.compile(r"(?=[А-ЯЁ])" + L + CL + _SLOT_SURN_LEET + r",?" + _SP +
                    r"([А-ЯЁ])\." + _SP + r"?([А-ЯЁ])\." + CR + R),
         mask=lambda m: _official_mask(m, leet=True),
         value_group=0, confidence="high",
         guard=lambda m: _official_guard(m, leet=True),
         needs=_LEET_NEEDS_CYR),       # the stylised slot is mandatory

    # === official order «Фамилия Имя Отчество» -> ONE token, seed = surname ===
    # Runs BEFORE fullname_pair / dict_first_name / patronymic so the
    # given+patronymic tail is never eaten halfway (two tokens, surname open).
    # Slots: surname (morphology + _SURN_SLOT_STOP) / any Capitalized given /
    # patronymic (productive suffix, or bare -ич with a gazetteer given) — the
    # official-order block above.  «Иванов, Иван Иванович», «ИВАНОВ ИВАН ИВАНОВИЧ»,
    # «ИВАНОВ Иван Иванович», «Мамедов Али Гусейн оглы» included; oblique cases
    # via _TAIL + patronymic tails («Ивановой Марии Петровне»).
    # PERF: the leading (?=[А-ЯЁ]) cuts a position with one check BEFORE the three
    # lookbehind branches of L/CL — a 1 MB scan is three times cheaper (0.03 -> 0.01 s), the matches
    # are the same (the surname slot starts with a capital anyway).
    Rule("surname_given_patronymic",
         re.compile(r"(?=[А-ЯЁ])" + L + CL + _SLOT_SURN + r",?" + _SP +
                    _SLOT_GIVEN + _SP + _SLOT_PATR + CR + R),
         mask=_official_mask,
         value_group=0, confidence="high",
         guard=lambda m: _official_guard(m, m.group(2), m.group(3))),

    # === surname + initials («Иванов И.И.», «Иванов И. И.», «ИВАНОВ И.И.») ===
    # Mirror of initials_surname for the surname-first order; the seed is the
    # nominative surname (same token as «И.И. Иванов» / «Иван Иванов»).  No
    # _ABBR_INITIALS guard here: after a surname «И.О.» is a name pair (Иван
    # Олегович) — the role abbreviation «и.о.» stands BEFORE its noun.
    Rule("surname_initials",
         re.compile(r"(?=[А-ЯЁ])" + L + CL + _SLOT_SURN + r",?" + _SP +
                    r"([А-ЯЁ])\." + _SP + r"?([А-ЯЁ])\." + CR + R),
         mask=_official_mask,
         value_group=0, confidence="high",
         guard=_official_guard,
         # the pair of initials «И.И.» / «И. И.» is a mandatory part of the match
         # (the same key as initials_surname: one search per pass)
         needs=re.compile(r"[А-ЯЁ]\.\s?[А-ЯЁ]\.")),

    # === full-name PAIR with a stylised (leet / homoglyph) word ===
    # «Иван Иван0в» / «Ив@н Иванов» / «Мария ПЕТР0ВА» / «Ивану Иван0ву»:
    # fullname_pair sees only pure Cyrillic. The evidence is weaker (leet is OCR
    # noise / codes / nicknames), so the given name must be in the RU gazetteer (by
    # the Cyrillic reading), and the surname slot passes _leet_surname_nom
    # (morphology + surname tail + slot stop sets). The seed is
    # the Cyrillic nominative of the surname: the same token as for «Иван Иванов» /
    # «Иванов И.И.» / «фамилия Иван0в». At least one slot is stylised
    # (_LEET_SLOT), otherwise the pair is fullname_pair's business; groups (1,2) | (3,4).
    # Stands BEFORE fullname_pair (a whole stylised token before any
    # partial match inside it) and is skipped by needs on text without
    # leet. (?=[А-ЯЁ]) cuts a position before the lookbehind branches of L/CL (perf).
    Rule("fullname_pair_leet",
         re.compile(r"(?=[А-ЯЁ])" + L + CL +
                    r"(?:(" + _LEET_SLOT + r")" + _PAIR_SP + r"(" + _LEET_SLOT + r"|" +
                    _CAP_CYR + r")|(" + _CAP_CYR + r")" + _PAIR_SP + r"(" + _LEET_SLOT +
                    r"))" +
                    CR + R),
         mask=_leet_pair_mask,
         value_group=0, confidence="high",
         guard=_leet_pair_guard,
         # a stylised Cyrillic slot is a mandatory part of the match
         needs=_LEET_NEEDS_CYR),

    # === full-name PAIR (given + surname) — mask both, seed on the surname ===
    # Longest / most specific, so the first name can never fuse to the token.
    # Single-space separator only (never crosses a newline / sentence break),
    # and the leading word must not be a geo marker (Село / Город / Деревня…).
    # The case tail sits in its OWN group so the seed is the nominative
    # stem+ending only ("Иван Петров" / "Ивана Петрова" -> one token),
    # mirroring surname_morph.  L/R keep the minted Latin token off adjacent
    # Latin letters / digits (CL/CR only see Cyrillic).
    # a pair with a double surname «Мария Иванова-Петрова» — whole (see _pair_double_mask)
    Rule("fullname_pair_double",
         re.compile(L + CL + r"(" + _CAP_CYR + r")" + _PAIR_SP + r"(" + _STEM_CYR +
                    SURNAME_ENDINGS + r")(" + _TAIL + r")(-[А-ЯЁ][а-яё]+)" + CR + R),
         mask=_pair_double_mask, value_group=0, confidence="high",
         guard=_pair_double_guard,
         needs=re.compile(r"[а-яё]" + _PAIR_SP + r"[А-ЯЁ][а-яё]+-[А-ЯЁ][а-яё]")),

    # an optional third word with a surname ending (groups 4/5)
    # — _pair_mask decides which pair is real in «Word Name Surname».
    Rule("fullname_pair",
         re.compile(L + CL + r"(" + _CAP_CYR + r")" + _PAIR_SP + r"(" + _STEM_CYR +
                    SURNAME_ENDINGS + r")(" + _TAIL + r")"
                    r"(?:" + _PAIR_SP + r"(" + _STEM_CYR + SURNAME_ENDINGS + r")(" +
                    _TAIL + r"))?" + CR + R),
         mask=_pair_mask,
         value_group=0, confidence="high",
         guard=lambda m: bool(_IS_TOKEN.search(m.group(0)))
                         or m.group(1) in _GEO_MARKERS,
         # «…lowercase SPACE Capital lowercase…» — the junction of the given name and
         # surname (a single space is a pattern literal; also NBSP/U+202F)
         needs=re.compile(r"[а-яё]" + _PAIR_SP + r"[А-ЯЁ][а-яё]")),

    # === initials + surname ("И.И.Иванов", "Г.В. Плеханова") ===
    # Explicit boundaries on BOTH edges: the minted token is Latin+digits, so
    # CL/CR alone would let it weld onto a neighbouring Latin letter / digit.
    Rule("initials_surname",
         re.compile(L + CL + r"([А-ЯЁ])\.\s?([А-ЯЁ])\.\s?(" + _CAP_CYR + r")" +
                    CR + R),
         # «\s?» between initials/surname may be a line break — keep it
         mask=lambda m: keep_line_breaks(mint.pname(_seed_person(_strip_case_tail(m.group(3)))),
                                         m.group(0)),
         value_group=0, confidence="high",
         guard=lambda m: (m.group(1), m.group(2)) in _ABBR_INITIALS
                         or m.group(3).lower() in _NOT_A_SURNAME
                         or bool(_IS_TOKEN.search(m.group(0))),
         needs=re.compile(r"[А-ЯЁ]\.\s?[А-ЯЁ]\.")),   # see surname_initials

    # === surname morphology (RU) ===
    # stem + productive family ending + inflection tail, seeded on the nominative
    # base so every case collapses to one token.  Negative lookahead blocks the
    # place / adjective homographs (…ск / …бар).  'low' by default: bare -ов/-ин/
    # -ский morphology is inherently ambiguous (маркетингов, магазин, готов…) and
    # over-masks ordinary prose, so it is opt-in for max recall / manual review.
    # High-precision names come from patronymic / full-pair / initials / gazetteer.
    Rule("surname_morph",
         re.compile(L + CL + r"(" + _STEM_CYR + r"?" + SURNAME_ENDINGS + r")"
                    r"(?!ск)(?!бар)" + r"(" + _TAIL + r")" + CR + R),
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=0, requires_context=NAME_CTX, confidence="low",
         guard=lambda m: bool(_IS_TOKEN.search(m.group(0)))
                         or m.group(1).lower() in _NOT_A_SURNAME,
         # the context gate — without NAME_CTX in left_ctx+text there is no replacement
         # (a shared search with translit_fullname_pair / nonlatin_script_word)
         needs=NAME_CTX),

    # === bare dictionary surname ===
    # A Capitalized word with surname morphology (the same pattern as
    # surname_morph), KNOWN to the russiannames surname dictionary (engine/
    # name_dicts: ~200k forms, count>=2, case normalisation), with a person cue
    # in the left window (NAME_CTX) -> 'context'. surname_morph stays 'low'
    # (bare morphology: маркетингов / магазин), here the evidence is the dictionary; not
    # minted: common-noun homonyms (Королева / Мельник — name_dicts.
    # is_homonym), _NOT_A_SURNAME, the surname-slot stop set (Готов /
    # Заказов / Сотрудников). The seed is the nominative (group 1 without the tail): the same
    # key as for fullname_pair / initials_surname / surname_given_patronymic.
    # No dictionary files — is_surname is always False, the rule is silent.
    Rule("bare_surname_dict",
         re.compile(L + CL + r"(" + _STEM_CYR + r"?" + SURNAME_ENDINGS + r")"
                    r"(?!ск)(?!бар)" + r"(" + _TAIL + r")" + CR + R),
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=0, requires_context=NAME_CTX, confidence="context",
         guard=lambda m: bool(_IS_TOKEN.search(m.group(0)))
                         or m.group(1).lower() in _NOT_A_SURNAME
                         or _seed_person(m.group(1)) in _SURN_SLOT_STOP
                         or name_dicts.is_homonym(m.group(0))     # Королева
                         or name_dicts.is_homonym(m.group(1))     # Мельник
                         or not name_dicts.is_surname(m.group(0)),
         needs=NAME_CTX),        # the context gate, as in surname_morph

    # === patronymic (-ович / -евич / -ич + tail; latin SERGEEVICH) ===
    # A bare «-ич» is additionally guarded (_patronymic_guard): common-noun
    # homographs Capitalized («Москвич»/«Кирпич») are not a patronymic.
    Rule("patronymic",
         re.compile(L + CL + r"(" + _STEM_CYR + r"(?:ович|евич|ьевич|инич|ич|овна|евна|"
                    r"ична|инична))(?:а|ем|у|е|ы|)" + CR + R),
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=0, confidence="context",
         guard=_patronymic_guard),
    Rule("patronymic_lat",
         re.compile(L + r"([A-Z][a-z]+(?:evich|ovich|evic|ovic))" + R),
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=1, confidence="context"),

    # === clean-transliteration ФИО pair (Latin, NO leet) ===
    # «Ivan Petrov» / «Anna Sidorova» without embedded leet substitutions are not caught by translit_
    # leet_name (it requires a digit/@/$ or a Cyrillic+Latin mix IN the token) — and
    # this is the most frequent kind of full-name translit (FN, global review).  The anchor is
    # the Latin surname suffix of the canon of the SECOND word (ov/ova/enko/…);
    # NAME_CTX in the left window keeps the rule at 'context'.  The seed is the surname canon,
    # so that «Поляков»/«Polyakov»/«Polyakova» link into one token.  L/R
    # keep the token from welding to adjacent Latin/digits.  Between the words —
    # horizontal whitespace [^\S\r\n]: ``\s+`` matched across a line
    # break and merged CSV/TSV records «Moscow\nPetrov»; NBSP/thin spaces
    # of Word exports remain.
    Rule("translit_fullname_pair",
         re.compile(L + r"([A-Z][a-z]+)[^\S\r\n]+([A-Z][a-z]+)" + R),
         mask=_lat_pair_mask,
         value_group=0, requires_context=NAME_CTX, confidence="context",
         guard=_lat_pair_guard, needs=NAME_CTX),        # the context gate

    # === dict-first-name (gazetteer-backed, given-name suffix WHITELIST) ===
    # Given names from the general RU_FIRST_NAMES gazetteer, matched with a closed
    # Russian case-suffix whitelist (NOT [а-яё]*) — the whitelist + Cyrillic
    # boundaries are what block surname / city derivations (Никитин, Ивановск,
    # Ольгинск) from matching.  A bare common first name is weak, so it is
    # ANCHORED: it fires only when a person cue (фамилия/имя/зовут/господин/…,
    # a vocative, or an intro frame) sits in the left window -> 'context'.  The
    # truly global bare-name sweep is the separate 'low' rule (off by default).
    # The matcher's group 1 is the inflection-invariant STEM, so every case of a
    # name seeds one PERSON_NAME token; group 0 spans stem+tail (the value).
    Rule("dict_first_name",
         _FIRST_NAME_RX,
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=0, requires_context=NAME_CTX, confidence="context",
         guard=lambda m: bool(_IS_TOKEN.search(m.group(0)))),

    # === transliteration + homoglyph + de-leet canonicalisation ===
    # Only fires on an alpha-led token that actually CONTAINS an EMBEDDED
    # substitution — a leet digit / @ / $ *between* letters (Pol1karpova, bel0va),
    # or a Cyrillic+Latin homoglyph mix in one word (Рoмaн) — AND whose canonical
    # key ends in a surname-ending and is not a dictionary word.  A leading '@'
    # is excluded here (that is a handle, matched below).  'context'-anchored.
    # PERF (bounded-lookahead trauma, global review): previously the pattern matched
    # EVERY word of ≥4 letters, and has_context ran for each match (a large
    # NAME_CTX alternation with a lookbehind over the window) + guard — on a 1.2MB line
    # ~2.0s of 2.4s of the whole pack (100k matches). The structural requirement (a leet
    # digit/@/$ BETWEEN letters OR a Cyrillic+Latin mix) is moved RIGHT into the pattern via
    # the ``_TRIGGER`` lookahead — matches became a handful, has_context is called only
    # on real candidates. Tokens are bounded by {2,28}, the lookahead is linear.
    # Cyrillic leet («Иван0в» / «ИВАН0В» / «Иванoв») — the canon is
    # TRANSLIT-first (_canon_lat), the seed is Cyrillic, as for fullname_pair
    # (_leet_seed), the guard — a capital + _CYR_LEET_STOP (_leet_guard).
    # Oblique cases («Иван0ву», «Ив@новым», «Достоевск0го») — the seed is by
    # the nominative, the tail is a surname form, an adjective stem is a stop.
    Rule("translit_leet_name",
         re.compile(L + _TRIGGER +
                    r"(?![@$])([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9@$]{2,28}"
                    r"[A-Za-zА-Яа-яЁё])" + R),
         mask=lambda m: mint.pname(_leet_seed(m.group(1))),
         value_group=1, requires_context=NAME_CTX, confidence="context",
         guard=_leet_guard,
         # the structural requirement of _TRIGGER — a leet char BETWEEN letters
         # or a Latin/Cyrillic junction — must be present in the text
         needs=_LEET_NEEDS),

    # === bare Cyrillic leet-surname sweep (OFF by default: 'low') ===
    # The same Cyrillic path as translit_leet_name (a capital, surname
    # morphology of the nominative / oblique case, stop sets), but WITHOUT a person cue
    # in the window: a bare leet surname «Иван0в пришёл». Estimate: on an OCR-like
    # substitution in Capitalized words of a real chat corpus ~5 % of
    # occurrences pass, of them the false ones after the stop sets are names in -ина (the same seed
    # as dict_first_name), rare -ские adjectives, «Вова/Сева» — for
    # a rule without a label the surface is large, as for surname_morph; hence 'low'
    # (opt-in via rule_tiers on a corpus where leet is the norm), and for
    # review — a harvest in tools/mine (a candidate, not a replacement). Leading Cyrillic
    # (_LEET_CAP) — Latin leet («Ivan0v») without a label stays out of scope.
    Rule("cyr_leet_surname_low",
         re.compile(L + r"(" + _LEET_CAP + r")" + R),
         mask=lambda m: mint.pname(_leet_seed(m.group(1))),
         value_group=1, confidence="low",
         guard=_leet_guard,
         needs=_LEET_NEEDS_CYR),

    # === diminutive / vocative (RU) — REQUIRE a vocative cue anchor ===
    # Dictionary of hypocoristic stems, but only when addressed: a whole-word cue
    # sits immediately left ("Слушай, Саш,").  Homograph-prone stems never fire bare.
    Rule("diminutive_vocative",
         re.compile(CL + r"(" + "|".join(sorted((re.escape(d) for d in DIMINUTIVES),
                                                 key=len, reverse=True)) +
                    r")(?:а|у|е|ой|ей|ь|и)?" + CR + r"(?=[ ,.!?…]|$)"),
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=0, requires_context=_VOCATIVE_CUE, window=40,
         confidence="context",
         guard=lambda m: bool(_IS_TOKEN.search(m.group(0)))),

    # === intro-frame name ("меня зовут X" / "My name is X") ===
    # Explicit boundaries: the frame anchors the left, but the token itself
    # needs the full no-fuse pair (CL/CR + L/R) on both edges.
    Rule("intro_frame_name",
         re.compile(L + CL + r"([А-ЯЁ][а-яё]+|[A-Z][a-z]+)" + CR + R),
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=1, requires_context=_INTRO_FRAME, window=30,
         confidence="context",
         guard=lambda m: bool(_IS_TOKEN.search(m.group(0)))
                         or m.group(1).lower() in _INTRO_STOP,
         # the context gate (the introduction frame) WITHOUT the «space+$» anchor:
         # _INTRO_FRAME is designed for a search over a window slice (the slice end =
         # the start of the name), on the full text «$» will not fire — needs looks for the
         # frame itself anywhere in left_ctx+text (necessary: without the frame there is no replacement)
         needs=re.compile(_INTRO_FRAME.pattern.removesuffix(r"[^\S\r\n]+$"))),

    # === linkedin /in/ slug -> HANDLE ===
    # Slug class excludes '/' and Cyrillic (so the token never fuses onto a path
    # continuation or a following Russian word); a slug that is already a token
    # is left alone.
    # (Telegram re-run audit: a slug with a URL-encoded non-ASCII letter «…%C3%A9…» was cut
    # at «%» — the slug tail stayed open): %XX is part of the slug
    # A dot/hyphen at the end is sentence punctuation, not part of the slug («/in/ivan-petrov.» gave
    # a seed with a dot and a foreign token): a slug ends with a letter/digit/«_» or %XX, as in label rules;
    # dots inside are part of the slug (a greedy class), the length of 3–61 units is unchanged
    Rule("linkedin_slug",
         re.compile(r"(/in/)((?:[A-Za-z]|%[0-9A-Fa-f]{2})(?:[A-Za-z0-9._-]|%[0-9A-Fa-f]{2}){1,59}"
                    r"(?:[A-Za-z0-9_]|%[0-9A-Fa-f]{2}))"),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="high",
         # "already a token" — ours, as a separate word of the slug
         guard=lambda m: is_own_token(m.group(2), word=True)
                         or m.group(2).lower() in COMMON_EN),

    # === handle / username literal (label-anchored) ===
    # A leading '@' or an explicit nick / username / login / handle label is the
    # evidence; a bare word is never harvested (that path-segment homograph FP).
    # A nickname of 31–63 chars whole («@aaaa…ab» of 32 chars was not taken by handle_at at all, and
    # «@имя_фамилия_город_…» was cut at the 31st char — the tail open). Before handle_at, the same
    # guard and seed; a nickname longer than 63 — as before.
    # A dot/hyphen at the end is sentence punctuation, not part of the
    # nickname («@ник.» at the end of a sentence gave a seed with a dot and a second token of the same nickname; «@admin.»
    # bypassed the stop list): a nickname ends with a letter/digit/«_», as in label rules; dots inside are
    # part of the nickname (a greedy class); the lengths are unchanged (31–63 and 3–31). handle_at_long has a lookahead "do not
    # cut a nickname at the length limit" — in the login_label form: not a letter/digit/«_» and not «.»/«-» +
    # a letter/digit/«_» (a hyphen before a space/end is punctuation, outside the nickname).
    Rule("handle_at_long",
         re.compile(r"(?:(?<=\\n)|(?<![\w@./\\-]))(@)([A-Za-z][A-Za-z0-9._-]{29,61}[A-Za-z0-9_])"
                    r"(?![A-Za-z0-9_]|[.-][A-Za-z0-9_])" + R),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="context",
         guard=lambda m: is_own_token(m.group(2), word=True)
         or m.group(2).lower() in COMMON_EN | TECH_TERMS | _BROADCAST_MENTIONS,
         needs=re.compile(r"@[A-Za-z][A-Za-z0-9._-]{30}")),
    Rule("handle_at",
         # JSONL-aware left edge: an @handle right after an escaped \n still
         # matches (the '@' itself can never sit ON the escape's 'n').
         re.compile(r"(?:(?<=\\n)|(?<![\w@./\\-]))(@)([A-Za-z][A-Za-z0-9._-]{1,29}[A-Za-z0-9_])" + R),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="context",
         # @everyone/@here/@channel/@all — service broadcast mentions
         # (Discord/Slack — the pipeline targets such corpora, detectors/
         # discord_ids.py), NOT nicknames: without a stop every line with them inflates
         # the map with dummy tokens (global review).
         # "already a token" — only ours (a known prefix) and as a separate
         # word of the nickname: «@KOZLOV_2001», «@SUPER_USER_2001» (a prefix after «_»)
         # — a nickname, masked.
         guard=lambda m: is_own_token(m.group(2), word=True)
                         or m.group(2).lower() in
                         COMMON_EN | TECH_TERMS | _BROADCAST_MENTIONS,
         needs="@"),              # the «@» of the match (a probe shared with email)
    # A dot/hyphen at the end is sentence punctuation, not part of the value («login: admin.»,
    # «ник: user.name.» gave a seed with a dot and a foreign token; «admin.» bypassed the stop list): the value
    # ends with a letter/digit/«_», as in contacts.login_label; dots inside («user.name») are part
    # of the value (a greedy class), the length of 3–31 is unchanged
    Rule("handle_label",
         re.compile(r"(?i)((?:username|user\s*name|nick(?:name)?|handle|login|"
                    r"ник|логин)\s*[:=]\s*@?)([A-Za-z][A-Za-z0-9._-]{1,29}[A-Za-z0-9_])"),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="context",
         # placeholder constants (EMAIL_REDACTED etc.) are tokens too: «login=<e-mail>»
         # after encoding/contacts is already EMAIL_REDACTED, and the rule re-minted it as HANDLE
         # "already a token" — as in handle_at («ник: QWERTY_2001»)
         guard=lambda m: is_own_token(m.group(2), word=True)
                         or "_REDACTED" in m.group(2)
                         or m.group(2).lower() in COMMON_EN | TECH_TERMS),

    # === display_name / global_name harvest from structured fields ===
    # Value of a display-name field in JSON is masked; generic tokens are stopped.
    # The value class is ESCAPE-AWARE ((?:[^"\\\n]|\\.)): a \" or \\ inside the
    # value is part of it, so the closing-quote group is the real closing quote
    # and the masked line stays valid JSON with no name remainder leaking.
    # Tokens already minted INSIDE the value (dict_first_name / patronymic run
    # first — NAME_CTX sees the «…name» key) are kept and the plaintext runs
    # around them are minted (see _display_name_mask): «Петров Иван» no
    # longer leaves «Петров» open.
    Rule("display_name_field",
         re.compile(r'(?i)("(?:display_name|global_name|full_name|real_name|nick|'
                    r'author_name)"\s*:\s*")((?:[^"\\\n]|\\.){3,40})(")'),
         mask=_display_name_mask,
         value_group=2, confidence="context",
         guard=_display_name_guard),

    # === name-in-filename field (first-last slug in name / path fields) ===
    # Exact first-last slug inside a filename / path field; keeps ext + separators.
    Rule("filename_firstlast",
         # (?i) ONLY on the key name: the Name-Surname slug must be case-sensitive,
         # otherwise "scan-Ivan" in "scan-Ivan-Petrov2.pdf" is read as a name pair and
         # masked with partial garbage (PERSON_NAME_...-Petrov2).
         # (?!-[A-Za-z]) on the right: a THIRD Cap segment («Node-Red-Setup») is a sign
         # of a product slug, not a full name; the bare key "name" is the most frequent
         # JSON key, and without this anchor + the dictionary guard it shredded slugs
         # (Node-Red-Setup.zip -> PERSON_NAME_x-Setup.zip, global review).
         re.compile(r'("(?i:filename|attachment_name|file_name|name|local_path|'
                    r'relative_path)"\s*:\s*"[^"\n]*?)'
                    r'(?<![A-Za-z-])([A-Z][a-z]{2,20}-[A-Z][a-z]{2,20})(?![A-Za-z0-9])(?!-[A-Za-z])'),
         mask=lambda m: m.group(1) + mint.pname(_seed_person(m.group(2))),
         value_group=2, confidence="high",
         # neither part of the slug may be a dictionary word (Node/Red/Setup/Test…) —
         # a product/project (Node-Red, Type-Script) is not a person.
         # A «Name-Surname» slug without «_» and digits — our token does not
         # occur in it in any mode; the guard is the same function for uniformity
         guard=lambda m: is_own_token(m.group(2), word=True)
                         or any(p.lower() in COMMON_EN | TECH_TERMS
                                for p in m.group(2).split("-")),
         # the slug «…a-Bcd…» is a mandatory part of the match (the case is exact)
         needs=re.compile(r"[a-z]-[A-Z][a-z]{2}")),

    # === non-Latin-script name (Armenian -yan / -ian surname suffix) ===
    # Latin regexes are blind to these scripts; the Armenian family suffix (-յան)
    # is a reliable surname signal.  Georgian / Arabic / CJK stay 'low' (need review).
    Rule("armenian_surname",
         re.compile(L + r"(?<![Ա-֏])([Ա-Ֆ][ա-ֆ]{2,20}(?:յան|եան))(?![ա-ֆ])" + R),
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=1, confidence="context"),
    Rule("nonlatin_script_word",
         re.compile(r"([" + SCRIPT_RANGES["georgian"] + SCRIPT_RANGES["armenian"] +
                    r"]{3,30})"),
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=1, requires_context=NAME_CTX, confidence="low",
         # at least one char of the Georgian/Armenian script
         needs=re.compile(r"[" + SCRIPT_RANGES["georgian"]
                          + SCRIPT_RANGES["armenian"] + r"]")),

    # === bare common first-name sweep (OFF by default: 'low') ===
    Rule("bare_first_name_low",
         re.compile(L + CL + r"(Александр|Сергей|Андрей|Дмитрий|Иван|Максим|Николай|"
                    r"Михаил|Владимир|Алексей|Артём|Артем|Роман|Егор|Кирилл|Денис|"
                    r"Антон|Игорь|Олег|Павел|Виктор|Елена|Ольга|Наталья|Татьяна|"
                    r"Ирина|Анна|Мария|Светлана|Юлия|Екатерина)" + CR + r"(?![а-яё])" + R),
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=1, confidence="low",
         guard=lambda m: bool(_IS_TOKEN.search(m.group(0)))),

    # === global common-surname sweep (OFF by default: 'low') ===
    Rule("common_surname_low",
         re.compile(L + CL + r"(" + _STEM_CYR + r"(?:ов|ев|[её]в|ин|ын)" + _TAIL + r")"
                    r"(?!ск)" + CR + R),
         mask=lambda m: mint.pname(_seed_person(m.group(1))),
         value_group=1, confidence="low",
         guard=lambda m: bool(_IS_TOKEN.search(m.group(0)))
                         or m.group(1).lower() in _NOT_A_SURNAME,
         # the surname suffix (ов|ев|ёв|ин|ын) is a mandatory part of the match
         needs=("ов", "ев", "ёв", "ин", "ын")),
]
