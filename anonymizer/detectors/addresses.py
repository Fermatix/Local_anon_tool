"""Address / geo / DOB detectors.

Everything here is FORMAT-ONLY or COMMON — a bare street name, a 6-digit ZIP, a
decimal coordinate pair, a `дом 12`, a `dd.mm.yyyy` date are all ubiquitous in
ordinary text and in structured logs. So every rule in this pack is gated:

* street / house / flat / city — a marker word is required in the match itself,
  and (for street/house/flat/city) an ``ADDRESS_CTX`` cue must sit in the left
  window.
* postal index — never global (it is a substring of epoch-ms timestamps); it
  fires only with ``ADDRESS_CTX`` nearby.
* region — «Свердловская обл.» / «Краснодарскому краю» / «Республика
  Татарстан»: the adjective must carry the -ск-/-цк- suffix and a region
  context (address, issuing authority, birth place, passport block) must sit
  in the left window. The postal index is now anchored by a region on its
  right as well («620075, Свердловская обл., …»).
* birth place / foreign address — label-anchored («Место рождения: …»,
  «Address: …» with a digit and a comma in the value).
* decimal coords — ``V.coord_in_range`` range-check AND ``GEO_CTX`` within 120.
* DMS coords (incl. %-encoded °'") — self-anchoring format.
* DOB — label-anchored via ``BIRTH_CTX``, year-bound 1935..2010, and it skips
  ``00.00.0000``, QA/placeholder contexts and values followed by an ``HH:MM``.

Markers (ул./д./г./кв.) are KEPT verbatim; only the value is replaced (return
``m.group(1) + token``). Tokens sit on Cyrillic / digit no-fuse boundaries so
they never weld onto a neighbouring letter or digit.
"""
from __future__ import annotations

import re

from ..engine import placeholders as P
from ..engine.boundary import _ESC
from ..engine.context import ADDRESS_CTX, GEO_CTX, BIRTH_CTX
from ..engine.rule import Rule, no_guard
from .financial import _keep_lines, _VS
from ..engine import validators as V

# ---- marker vocabularies -----------------------------------------------------
# Horizontal whitespace inside an address: `\s` matched a line
# break — «ул. Ленина\n2024-09-25 приехали» / «г. Москва,\nЛенина, 5, 12\nx»
# glued lines/CSV records into one token, the content between them disappeared.
# [^\S\r\n] — any whitespace except \r/\n (NBSP from Word/Confluence is intact).
_HS = r"[^\S\r\n]"
# Street-type markers (kept verbatim; anchor the street name that follows).
_STREET_MARKER = (
    r"(?:ул\.|улиц[аеы]|проспект|просп\.|пр-?кт|пр-?т|переулок|пер\.|"
    r"набережн\w*|наб\.|шоссе|ш\.|бульвар|б-р|проезд|тупик|аллея|"
    r"микрорайон|мкр\.?|квартал|кв-л)"
)
# City / settlement markers (marker is MANDATORY — a bare capitalised word is not
# a city). Abbreviations — incl. the ambiguous single letters — only count WITH
# their dot; full words and dotless contractions (пгт/ст-ца/рп) with a space.
_CITY_MARKER = (
    r"(?:(?:г|гор|с|пос|дер|д|ст|х)\.\s?"
    r"|(?:город|село|станица|хутор|пгт|пос[её]лок|деревня|ст-ца|рп)\s+)"
)
# Capitalised words that follow a marker but are NOT a street/city name.
_STREET_STOP = re.compile(
    r"(?i)^(?:вопрос|этап|версия|период|раздел|которо?\w*|дом|корпус|строен\w*|"
    r"адрес|индекс|город|район|области?|региона?)$"
)
# A year / 'г.р.' marker GLUED to a DOB ('01.05.1985г.', '…1985г.р.') is consumed
# into the masked value so DATE_REDACTED never welds onto the 'г'. Spaced forms
# ('01.05.1985 г.') don't weld and are kept verbatim (that 'г.' may even be the
# city marker of the rule below).
# «г.\s?р.» took a line break («1985г.\nр.» — a match across lines, a skip with a
# WARN, the date open) — horizontal whitespace financial._VS.
_DOB_TAIL = r"(?:(?:г\." + _VS + r"?р\.?|года?|г\.?)(?![А-Яа-яЁёA-Za-z0-9]))?"


# Left open were «Дата рождения: 5 марта 1985 г.» (a month in words),
# «дата рождения: 12.05.2015» / «…12.05.1934» (years outside 1935–2010), «DOB: 1990-05-12», «ДР
# 12.05.1990» (labels missing from BIRTH_CTX). Now: a form with the month in the genitive case
# (dob_month, BIRTH_CTX, the same guard); years 1930–2029 — only with a STRONG label on the left in the
# same line (date/year of birth, д.р., родил…, birth, DOB, «ДР» as a whole word in CAPITALS), «выдан» —
# still only 1935–2010 (doc_issue_date covers it); short labels «DOB»/«ДР» right next to
# the date — dob_labeled.
_DOB_STRONG = re.compile(
    r"(?i:дата\s*рожд|(?<![а-яёa-z])д\.\s*р\.?(?![а-яёa-z])|год\s*рожд|родил|birth|"
    r"(?<![a-z])dob(?![a-z]))|(?<![А-Яа-яЁё])ДР(?![А-Яа-яЁё])")
# to the left of «DOB» — also a JSONL escape (boundary._ESC: «инфо\nDOB:
# 1990-05-12» in a raw .jsonl stayed open — the escape's letter "n" counted as a neighbour)
_DOB_LABEL = (r"(?:(?:" + _ESC + r"|(?<![A-Za-z]))(?i:dob|d\.o\.b\.?)(?![A-Za-z])"
              r"|(?<![А-Яа-яЁё])ДР(?![А-Яа-яЁё]))"
              r"[^\S\r\n]*[:\-–—]?[^\S\r\n]*")
_MONTHS_GEN = r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"


def _dob_strong_label(m: re.Match) -> bool:
    """A strong birth label to the left of the date in the same line (window 60) — years 1930–2029."""
    left = m.string[max(0, m.start(0) - 60): m.start(0)]
    left = left[max(left.rfind("\n"), left.rfind("\r")) + 1:]
    return bool(_DOB_STRONG.search(left))


def _dob_guard(m: re.Match) -> bool:
    """True => leave untouched. Skip 00.00.0000, implausible years, QA/placeholder
    context, and dates that are really a timestamp date (followed by HH:MM)."""
    whole = m.group(0)
    if re.match(r"^\D*0{1,2}[.\-/]0{1,2}[.\-/]0{2,4}\b", whole):
        return True
    try:
        y = int(m.group("y"))
    except (ValueError, IndexError):
        return True
    if not (1935 <= y <= 2010) and not _dob_strong_label(m):
        return True
    tail = m.string[m.end(0): m.end(0) + 12]
    if re.match(r"(?i)[ T]?(?:в|at\s)?\s*\d{1,2}:\d{2}", tail):  # a datetime, not a DOB
        return True
    left = m.string[max(0, m.start(0) - 80): m.start(0)]
    if re.search(r"(?i)(подставляет|автоматическ|дефолт|default|границ|проверк|"
                 r"ошибк|минимальн|максимальн|пример|example|placeholder|тест)", left):
        return True
    return False


def _coord_guard(m: re.Match) -> bool:
    """True => leave untouched. Reject out-of-range / version-pair coordinates."""
    try:
        lat, lon = float(m.group(1)), float(m.group(2))
    except (ValueError, IndexError):
        return True
    return not V.coord_in_range(lat, lon)


def _flat_guard(m: re.Match) -> bool:
    """True => leave untouched. 'кв. 2021 г./года' is a year, not a flat. Probe
    from the end of the digits so a swallowed letter ('кв. 2021г.') cannot hide
    the year cue."""
    val = m.group(2)
    digits = re.match(r"\d+", val).group(0)
    tail = val[len(digits):] + m.string[m.end(0): m.end(0) + 6]
    return bool(re.match(r"(?i)^\s*(?:г\.?\s|год|20\d\d)", tail))


# --- Russian region (audit escalations) ---------------------------------------
# «620075, Свердловская обл., г. Екатеринбург» — the most common address form
# of company details (221 occurrences in a gdocs corpus) passed a round half done:
# city/street/office were masked, while the postcode and region stayed. Reasons: (1)
# the postcode anchor _ZIP_RIGHT expected a city/street marker or «Слово,» after the comma
# — «Свердловская обл.,» (adjective + region marker) did not
# fit it; (2) there was no rule for the region itself at all («Свердловская обл.»,
# «Краснодарскому краю», «Волгоградской области», «…РАЙОНЕ ГОР. …» — CITY
# audit findings).
# A Russian region name is a relative adjective in -ск-/-цк- (Свердловская,
# Краснодарский, Ненецкий): the suffix is required so that «Новый район», «Наш край»,
# «Этот округ» in an address context do not become GEO.
_REGION_MARKER = (
    r"(?:обл\.|област(?:ь|и|ью|ей)|кра(?:й|я|ю|ем|е)|район(?:а|е|у|ом)?|"
    r"р-н(?:а|е|у)?|округ(?:а|е|у|ом)?)"
)
_REGION_ADJ = (
    r"[А-ЯЁ](?:[а-яё]+|[А-ЯЁ]+)(?:-[А-ЯЁ](?:[а-яё]+|[А-ЯЁ]+))?"
)
_REGION_ADJ_END = re.compile(
    r"(?i)(?:ск|цк)(?:ая|ой|ую|ою|ий|ого|ому|им|ом|ое)$")
# Region context is wider than the address one: a document issuing authority («Выдан ГУ МВД
# России по Краснодарскому краю»), place of birth, a passport block.
_REGION_CTX = re.compile(
    ADDRESS_CTX.pattern
    + r"|(?i:выдан|мвд|уфмс|место\s+рожд|паспорт|" + re.escape(P.PASSPORT) + r")")
# label words that may follow a place of birth without a separator
# («Место рождения: гор. Тверь Паспорт серия …») — not part of the city name
_FIELD_WORDS = (
    r"Паспорт|Серия|Номер|Дата|Адрес|Гражданство|Пол|СНИЛС|ИНН|Выдан|Код|Место|"
    r"Телефон|Тел|Почта|Регистрац\w*|Проживан\w*|Образование|Должность"
)
# a field that starts the NEXT company detail after an English address
_EN_NEXT_FIELD = (
    r"Registration|Reg\.|Taxpayer|Tax\b|VAT\b|TIN\b|Phone|Tel\b|Tel\.|E-?mail|"
    r"IBAN|SWIFT|BIC\b|Account|Bank\s+name|Bank\s+address|Company|Director|"
    r"Contact|Website|Web\b"
)


# generic adjectives in -ск-: a territory type, not its name
# («Городской округ», «Сельский район» — not a region). «российск*» was removed
# (owner: the whole Russian trace is masked) — «Российской
# Федерации/области» in an address = GEO; bare toponyms and «российский» outside an address
# are covered by the known_values/ru_trace dictionary
_REGION_GENERIC = re.compile(r"(?i)^(?:городск|сельск|поселковск)")
_NOT_POSTAL = re.compile(
    r"(?i)(?:ip|ipv[46]|mac|e-?mail|web|url|site|wallet|contract|dns|host|"
    r"server|proxy)[^\S\r\n]*$")


def _region_guard(m: re.Match) -> bool:
    """True => do not mask: the word before the region marker is not a name
    (no -ск-/-цк- suffix) or is already a token."""
    name = m.group(1)
    if "_REDACTED" in name or _REGION_GENERIC.match(name):
        return True
    return not _REGION_ADJ_END.search(name)


def _en_address_guard(m: re.Match) -> bool:
    """True => do not mask: the value after "Address:" does not look like an address —
    no digit or comma ("Address: see below", "Address: TBD") or it is already
    tokens/URL."""
    v = m.group(2)
    if "_REDACTED" in v or "://" in v:
        return True
    # «IP address: 10.0.0.1, 10.0.0.2», «MAC address: …», «E-mail address: …»,
    # "wallet/contract address: 0x…" — not a postal address
    left = m.string[max(0, m.start(1) - 14): m.start(1)]
    if _NOT_POSTAL.search(left):
        return True
    return not (any(c.isdigit() for c in v) and "," in v)


_ZIP_LABEL = re.compile(r"(?i)(индекс|почтов\w*|zip|postal|post\s*code)\D{0,4}$")
# Leading position of an inline address: the postcode comes FIRST («адрес: 101000, г.
# Москва…» / «195112, Санкт-Петербург, пр. …») — a comma only on the right, and
# the former "commas on both sides" anchor did not see it.
# A tight anchor on the right: right after the comma — a city/street marker or
# «Слово,» (a city without a marker in postal format). The engine's ADDRESS_CTX window
# is still required.
# + a region after the comma («190000, Ленинградская обл., …», «…,
# Республика Татарстан, …») and an already placed GEO_REDACTED (the region rule
# stands below postcode, but the second pass fixpoint sees the token).
_ZIP_RIGHT = re.compile(
    r"^,\s*(?:(?i:" + _CITY_MARKER + r"|" + _STREET_MARKER + r")"
    r"|[А-ЯЁ][А-Яа-яЁё-]{2,25},"
    r"|" + _REGION_ADJ + r"[^\S\r\n]+(?i:" + _REGION_MARKER + r")"
    r"|(?i:республик[аиеу]|респ\.)[^\S\r\n]+[А-ЯЁ]"
    r"|" + re.escape(P.GEO) + r")"
)


# With v1 minting (old tenant.yaml without mint_version) a token ends with
# six digits — «Адрес: PERSON_NAME_113900, г. Москва» → «PERSON_NAME_POSTCODE_REDACTED»
# (the link to the person, restore and entity_map of this place broke, the gates were silent). A postcode right
# after "_" is the tail of an identifier, not a postcode.


def _postcode_after_underscore(m: re.Match) -> bool:
    i = m.start(1)
    return i > 0 and m.string[i - 1] == "_"


def _postcode_anchored(m: re.Match) -> bool:
    """A 6-digit ZIP is trusted only with a TIGHT anchor (the 120-char ADDRESS_CTX
    window alone is too loose — it lets a task/order id near address words slip
    through). Accept only when an index/zip/postal label sits right before, OR the
    value is comma-anchored on both sides (CSV cell / inline address), OR the ZIP
    LEADS an inline address (comma right + city/street marker or «Город,»)."""
    s = m.string
    left = s[max(0, m.start(1) - 16): m.start(1)]
    if _ZIP_LABEL.search(left):
        return True
    before = s[:m.start(1)].rstrip()
    after = s[m.end(1):].lstrip()
    if before.endswith(",") and after.startswith(","):
        return True
    # a leading postcode: right after it «, г. …» / «, ул. …» / «, Город, …»
    return bool(_ZIP_RIGHT.match(s[m.end(1):]))


RULES: list[Rule] = [
    # --- full inline address (CSV): city, street, house, flat — most specific ---
    Rule("address_csv",
         re.compile(
             r"([А-ЯЁ][А-Яа-яЁё.-]{2,30},)" + _HS + r"*"      # city,
             r"([А-ЯЁ][А-Яа-яЁё .-]{3,30})," + _HS + r"*"     # street,
             r"(\d{1,4}[а-яёА-ЯЁ]?)," + _HS + r"*"            # house,
             r"(\d{1,4})(?!\d)"),                             # flat
         mask=lambda m: P.ADDRESS, value_group=0,
         requires_context=ADDRESS_CTX, confidence="context",
         # the ", house, flat" tail is a mandatory part of the match
         needs=re.compile(r",[^\S\r\n]*\d{1,4}[а-яёА-ЯЁ]?,[^\S\r\n]*\d")),

    # --- street name (marker + capitalised name), keep marker -> STREET; a
    #     glued house-number run ('ул. Ленина8') is swallowed into the value so
    #     the token never welds onto a digit ---
    Rule("street",
         re.compile(
             r"((?i:" + _STREET_MARKER + r")" + _HS + r"+)"
             r"([А-ЯЁ][А-Яа-яЁё-]+(?:" + _HS + r"+(?:им\.?" + _HS + r"+)?"
             r"[А-ЯЁ0-9][А-Яа-яЁё0-9-]+){0,2}"
             r"(?:\d{1,4}[а-яё]?)?)"),
         mask=lambda m: m.group(1) + P.STREET, value_group=2,
         requires_context=ADDRESS_CTX, confidence="context",
         guard=lambda m: bool(_STREET_STOP.match(m.group(2).split()[0])),
         # context gate — without ADDRESS_CTX in left_ctx+text there is no replacement;
         # one shared search per pass for street/house/flat/city/postcode
         needs=ADDRESS_CTX),

    # --- house / building (д./дом/корп/стр); left-boundary excludes letter-hyphen
    #     so 'пр-д' (проезд) never reads as 'д.' ---
    Rule("house",
         re.compile(
             r"(?i)(?<![А-Яа-яЁёA-Za-z-])"
             r"((?:дом|д\.|владени\w*|влд?\.?|строени\w*|стр\.|корпус|корп\.|к\.|"
             r"пом\.?|помещени\w*)\s?)"
             r"(\d{1,4}[а-яёa-z]?(?:" + _VS + r"?[кс]\.?" + _VS + r"?\d{1,3})?)"
             r"(?![\dа-яёА-ЯЁA-Za-z])"),      # корпус (building) — in the same line (_VS)
         mask=lambda m: m.group(1) + P.HOUSE, value_group=2,
         requires_context=ADDRESS_CTX, confidence="context",
         guard=lambda m: bool(re.match(r"^\d+-", m.group(2))),  # skip ordinal 25-й
         needs=ADDRESS_CTX),

    # --- flat / office (кв./оф.); the flat letter ('кв. 12а') is swallowed into
    #     the value so the token never welds; year-guard so 'кв. 2021 года'
    #     isn't a flat ---
    Rule("flat",
         re.compile(
             r"(?i)((?:квартир\w*|кв\.?|офис|оф\.?)\s?)"
             r"(\d{1,4}[а-яё]?[a-zA-Z]?)(?![\dа-яёА-ЯЁA-Za-z])"),
         mask=lambda m: m.group(1) + P.FLAT, value_group=2,
         requires_context=ADDRESS_CTX, confidence="context",
         guard=_flat_guard, needs=ADDRESS_CTX),

    # --- city with mandatory marker (г. + dot / город|село|пгт + space) -> GEO ---
    Rule("city",
         re.compile(
             r"(?<![А-Яа-яЁёA-Za-z])(" + _CITY_MARKER + r")"
             r"([А-ЯЁ][А-Яа-яЁё-]{2,}(?:[- ][А-ЯЁ][А-Яа-яЁё-]+){0,2})"),
         mask=lambda m: m.group(1) + P.GEO, value_group=2,
         requires_context=ADDRESS_CTX, confidence="context",
         guard=lambda m: bool(_STREET_STOP.match(m.group(2).split()[0])),
         needs=ADDRESS_CTX),

    # --- postal index (6-digit, first ∈ 1-6). NEVER global: it is a substring of
    #     epoch-ms timestamps and collides with task/order ids. Requires BOTH an
    #     ADDRESS_CTX window (engine) AND a tight anchor — an index/zip/postal
    #     label right before, or a comma on both sides (CSV / inline address).
    #     Right guard rejects a glued letter ('101000г.') — masking it would
    #     weld the token. ---
    Rule("postcode",
         re.compile(r"(?<![\d.])([1-6]\d{5})(?![\dA-Za-zА-Яа-яЁё])"),
         mask=lambda m: P.POSTCODE, value_group=1,
         requires_context=ADDRESS_CTX, confidence="context",
         guard=lambda m: _postcode_after_underscore(m) or not _postcode_anchored(m),
         needs=ADDRESS_CTX),

    # --- Russian region: «Свердловская обл.» / «Краснодарскому краю» /
    #     «ВОЛГОГРАДСКОЙ ОБЛАСТИ» / «Орджоникидзевском районе» -> GEO, the marker
    #     stays. The adjective must carry the -ск-/-цк- suffix
    #     (_region_guard); context — address, issuing authority, place of birth,
    #     passport block (_REGION_CTX). COMES AFTER postcode: the postcode anchor
    #     looks at the raw region on the right. ---
    Rule("region",
         re.compile(r"(?<![А-Яа-яЁёA-Za-z])(" + _REGION_ADJ + r")"
                    r"([^\S\r\n]+(?i:" + _REGION_MARKER + r"))"
                    r"(?![А-Яа-яЁёA-Za-z])"),
         mask=lambda m: P.GEO + m.group(2), value_group=1,
         requires_context=_REGION_CTX, confidence="context",
         guard=_region_guard,
         needs=re.compile(r"(?i)обл\.|област|кра[йяюе]|район|р-н|округ")),
    Rule("region_republic",
         re.compile(r"(?<![А-Яа-яЁёA-Za-z])((?:[Рр]еспублик[аиеу][^\S\r\n]+"
                    r"|[Рр]есп\.[^\S\r\n]*))"
                    r"([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
                    r"(?:[^\S\r\n]+\([А-ЯЁ][а-яё]+\))?)"),
         mask=lambda m: m.group(1) + P.GEO, value_group=2,
         requires_context=_REGION_CTX, confidence="context",
         needs=re.compile(r"[Рр]еспублик|[Рр]есп\.")),
    # --- a city in CAPS with a marker («ГОР. ТВЕРЬ», «Г. МОСКВА» in headers of
    #     contracts/certificates): city above is case-sensitive (single «с./д./х.»
    #     in caps are initials), here only the unambiguous «Г./ГОР./ГОРОД» ---
    Rule("city_caps",
         re.compile(r"(?<![А-Яа-яЁёA-Za-z])((?:ГОР\.|Г\.|ГОРОД)[^\S\r\n]?)"
                    r"([А-ЯЁ]{3,}(?:-[А-ЯЁ]{2,}){0,2})(?![А-Яа-яЁё])"),
         mask=lambda m: m.group(1) + P.GEO, value_group=2,
         requires_context=_REGION_CTX, confidence="context",
         needs=re.compile(r"ГОР\.|Г\.|ГОРОД")),
    # --- place of birth: label-anchored, no context needed. The city
    #     marker is optional; a second capitalised word is taken only if
    #     it is not the next form label («…гор. Тверь Паспорт серия…»). ---
    Rule("birthplace",
         re.compile(
             r"((?i:место[^\S\r\n]+рождения)[^\S\r\n]*[:\-–—]?[^\S\r\n]*"
             r"(?:(?:г|гор|с|пос|дер|п)\.[^\S\r\n]?"
             r"|(?:город|село|пос[её]лок|деревня)[^\S\r\n]+)?)"
             r"([А-ЯЁ][а-яё]+(?:-[А-ЯЁа-яё][а-яё]+){0,2}"
             r"(?:[^\S\r\n](?!(?:" + _FIELD_WORDS + r")(?![а-яё]))[А-ЯЁ][а-яё]+)?)"),
         mask=lambda m: m.group(1) + P.GEO, value_group=2, confidence="context",
         guard=lambda m: bool(re.fullmatch(r"(?:" + _FIELD_WORDS + r")", m.group(2))),
         needs=re.compile(r"(?i)место[^\S\r\n]+рожд")),
    # --- a foreign address under an English label ("Address: Netherlands,
    #     8 …straat, 1234AB City, Netherlands" — a postal format without Russian
    #     markers was seen by nobody). A colon after the label is mandatory; the value
    #     runs to the end of the line/cell/double space or to the next field with
    #     a colon ("… Registration number:"). Guard: the value has a digit and
    #     a comma; "IP/MAC/E-mail/Web address:" is not a postal address. ---
    Rule("address_label_en",
         re.compile(
             r"(?i)(?<![a-z])((?:(?:bank|legal|registered|postal|billing|office|"
             r"company|mailing|business)[^\S\r\n]+)?address[^\S\r\n]*:[^\S\r\n]*)"
             r"([^\n\r|*<>\\\"]{8,160}?)"
             r"(?=[^\S\r\n]{2,}|[^\S\r\n]*(?:[\r\n|*<>\\\"]|$)"
             r"|[^\S\r\n]+(?:" + _EN_NEXT_FIELD + r")[^\n\r:|]{0,30}:)"),
         mask=lambda m: m.group(1) + P.ADDRESS, value_group=2,
         confidence="context", guard=_en_address_guard,
         needs=re.compile(r"(?i)address[^\S\r\n]*:")),

    # --- decimal geo coordinate pair — range-check + GEO_CTX within 120 ---
    Rule("coords_decimal",
         re.compile(r"(?<![\d.,\w])(-?\d{1,3}\.\d{4,15})\s*,\s*(-?\d{1,3}\.\d{4,15})(?![\d.])"),
         # «55.7558,\n37.6173» — line breaks of the match are kept (financial._keep_lines)
         mask=lambda m: _keep_lines(P.GEO, m), value_group=0,
         requires_context=GEO_CTX, confidence="context",
         guard=_coord_guard,
         needs=re.compile(r"\d\.\d{4}")),     # a fraction with ≥4 digits

    # --- DMS coordinates (incl. %-encoded ° ' ") — self-anchoring format ---
    Rule("coords_dms",
         re.compile(
             r"(?i)(\d{1,3}(?:°|%C2%B0)" + _VS + r"?\d{1,2}(?:'|′|%27)" + _VS + r"?"
             r"\d{1,2}(?:\.\d+)?(?:\"|″|%22)?" + _VS + r"?[NSEWСЮВЗ])"),   # _VS
         mask=lambda m: P.GEO, value_group=1, confidence="high",
         needs=("°", "%C2%B0")),               # the degree marker is mandatory

    # --- DOB (label-anchored via BIRTH_CTX, year-bound 1935..2010) ---
    Rule("dob",
         re.compile(
             r"(?<![\d.])((?:0[1-9]|[12]\d|3[01])[.\-/](?:0[1-9]|1[0-2])[.\-/]"
             r"(?P<y>(?:19[3-9]\d|20[0-2]\d))" + _DOB_TAIL + r")(?!\d)"),
         mask=lambda m: P.DATE, value_group=1,
         requires_context=BIRTH_CTX, confidence="context",
         guard=_dob_guard, needs=BIRTH_CTX),   # context gate, shared search
    # «5 марта 1985 г.» — the month in words in the genitive case, the same label and guard
    Rule("dob_month",
         re.compile(
             r"(?<![\d.])((?:0?[1-9]|[12]\d|3[01])[^\S\r\n]+" + _MONTHS_GEN + r"[^\S\r\n]+"
             r"(?P<y>(?:19[3-9]\d|20[0-2]\d))" + _DOB_TAIL + r")(?!\d)", re.I),
         mask=lambda m: P.DATE, value_group=1,
         requires_context=BIRTH_CTX, confidence="context",
         guard=_dob_guard, needs=BIRTH_CTX),
    # a short label right next to the date — «DOB: 1990-05-12», «ДР 12.05.1990»
    Rule("dob_labeled",
         re.compile(
             r"(" + _DOB_LABEL + r")"
             r"((?:(?:0[1-9]|[12]\d|3[01])[.\-/](?:0[1-9]|1[0-2])[.\-/](?P<y>19[3-9]\d|20[0-2]\d)"
             r"|(?P<y2>19[3-9]\d|20[0-2]\d)-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
             r"|(?:0?[1-9]|[12]\d|3[01])[^\S\r\n]+" + _MONTHS_GEN + r"[^\S\r\n]+"
             r"(?P<y3>19[3-9]\d|20[0-2]\d))" + _DOB_TAIL + r")(?![\d.])"),
         mask=lambda m: m.group(1) + P.DATE, value_group=2, confidence="context",
         guard=no_guard,
         needs=re.compile(r"(?i)dob|d\.o\.b|ДР")),
    # ISO-form DOB (yyyy-mm-dd), same label + year gate
    Rule("dob_iso",
         re.compile(
             r"(?<![\d.])((?P<y>19[3-9]\d|20[0-2]\d)-(?:0[1-9]|1[0-2])-"
             r"(?:0[1-9]|[12]\d|3[01])" + _DOB_TAIL + r")(?!\d)"),
         mask=lambda m: P.DATE, value_group=1,
         requires_context=BIRTH_CTX, confidence="context",
         guard=_dob_guard, needs=BIRTH_CTX),
]
