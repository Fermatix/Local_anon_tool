"""Structured profile of CSV/TSV table registries: a column is the unit of decision.

Why. A real batch had 21 CSV registries (144 MB, cp1251, ";", quoted values),
rows like "personal account; full name; settlement; street; house; building; flat;
…; organization; …". The old path made a separate literal out of every full name/address/
organization cell: 928k control values, a 133 MB pack, and every stage grew with
that number; streets in a separate column ("ул Ленина" without a house in the same cell)
were invisible to the address detector — they were closed by hand with group decisions.

What the profile does (following the Telegram/YouTrack structured profiles, where
"people in fields → the same token as in text"):

  * recognises a registry: .csv/.tsv, the delimiter — as in the value passes
    (formats.csv_delimiter), a regular table (≥ 2 columns, ≥ MIN_ROWS rows,
    ≥ 90 % of sample rows with the same field count), the header — by the shape of row 0
    (header hints, or "the data below is numeric/email/phone and row 0
    is not"); without a header — column statistics only;
  * decides the column kind (``KINDS``): full name/subscriber (person), organization
    (org), email, phone, INN, SNILS, a whole address and its parts — settlement,
    street, house, flat, postcode; personal account, passport, date of birth.
    Statistical kinds (email, phone, INN, SNILS, full name, legal form, street/
    city markers) are recognised by the share of sample cells, even without a header; "header
    only" kinds (house, flat, postcode, personal account, passport, date of birth —
    bare numbers cannot be told from codes) need a header hint AND the cell
    form. A header with a hint but cells of the wrong form — the column is not
    recognised → the normal path (rules on text). A code header ("код улицы",
    "id") is not PII;
  * masks the cells of PII columns WHOLE with the same token the
    text path would give the same value (tenant salt, mint): full name →
    PERSON_NAME by the surname nominative (detectors.names._official_mask:
    "Иванов Иван Иванович" in text and in a cell is one token; "ООО …" in a
    subscriber column → ORG like opf_name; "ИП Иванов И.И." → "ИП PERSON_NAME");
    organization → ORG by the name without the legal form (legal form and quotes stay); email/phone/
    INN/SNILS/account/passport/date → family constants; address parts → the marker
    stays ("г. ", "ул. ", "д. ", "кв. "), the value → GEO/STREET/HOUSE/FLAT/
    POSTCODE_REDACTED; a whole address → ADDRESS_REDACTED. Columns without PII
    (readings, amounts, tariff codes, dates) are not touched; free text
    (comments) takes the normal path — the engine and pack rules run on the
    file AFTER the profile and do not touch already closed cells (TOKEN guards);
  * writing — by substituting changed cells into the raw record (the same discipline
    as formats.mask_csv_cells: cell quoting, delimiters, the record line ending
    and everything outside the replaced cells — byte for byte); streamed by record,
    the file is never held as a whole list of rows;
  * for mining and NER (``blank_text``) the cells of PII columns are replaced with spaces
    of the same length (offsets intact): no candidate literals, pack or
    control values grow from them; a repeat pass over an already masked file gives
    0 replacements (token cells do not count as kind evidence, the profile is not
    recognised).

Entry points: ``profile_text(rel, text)`` → TableProfile | None;
``mask_text(text, prof, report)``; ``blank_text(text, prof)``.
The switch is shared by the structured profiles: ANON_STRUCTURED_PROFILES=0
(engine.profiles_enabled) — the file takes the old flat path.
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field

from ..detectors import addresses as _addr
from ..detectors import names as _names
from ..detectors import orgs as _orgs
from ..engine import mint
from ..engine import placeholders as P
from ..engine import validators as V
from ..engine.boundary import is_own_token
from .engine import STATS
from ..engine.formats import (CsvStructureError, _csv_cell_text, _csv_quote,
                              _csv_record_cells, _csv_record_eol,
                              _LineTracker, csv_delimiter, iter_csv_cells,
                              raise_csv_field_limit)
from ..engine.gazetteer import (EN_FIRST_NAMES, RU_FIRST_NAMES_ALL,
                                RU_FIRST_NAMES_DIM)
from ..engine.vocab import SURNAME_ENDINGS

# --- column kinds ------------------------------------------------------------
PERSON = "person"          # full name / subscriber / payer (a cell with a legal form → org)
ORG = "org"                # organization (a full-name cell without a legal form → person)
EMAIL = "email"
PHONE = "phone"
INN = "inn"
SNILS = "snils"
ACCOUNT = "account"        # personal account
PASSPORT = "passport"
DOB = "dob"                # date of birth
ADDRESS = "address"        # whole address
SETTLEMENT = "settlement"  # settlement
STREET = "street"
HOUSE = "house"            # house / building / structure
FLAT = "flat"              # flat / premises
POSTCODE = "postcode"
KINDS = (PERSON, ORG, EMAIL, PHONE, INN, SNILS, ACCOUNT, PASSPORT, DOB,
         ADDRESS, SETTLEMENT, STREET, HOUSE, FLAT, POSTCODE)
# kinds recognised by cell statistics without a header (sample share threshold)
STAT_KINDS = ((EMAIL, "email", 0.9), (PHONE, "phone", 0.9), (INN, "inn", 0.9),
              (SNILS, "snils", 0.9), (PERSON, "person", 0.9), (ORG, "opf", 0.8),
              (ADDRESS, "address", 0.8), (STREET, "street", 0.8),
              (SETTLEMENT, "settlement", 0.8))
# confirmation by cell form when there is a header hint: feature, share threshold
HEADER_NEED = {
    EMAIL: ("email", 0.5), PHONE: ("phone", 0.5), INN: ("inn_digits", 0.8),
    SNILS: ("snils_digits", 0.8), PERSON: ("person_loose", 0.5),
    ORG: ("letters", 0.5), STREET: ("letters", 0.5), SETTLEMENT: ("letters", 0.5),
    ADDRESS: ("letters", 0.5), HOUSE: ("house", 0.8), FLAT: ("flat", 0.8),
    POSTCODE: ("postcode", 0.8), ACCOUNT: ("account", 0.8),
    PASSPORT: ("passport", 0.8), DOB: ("date", 0.8),
}
# strong statistics override the header hint (an email in a "Contact" column)
_STAT_OVERRIDES = frozenset({EMAIL, PHONE, INN, SNILS})

HEAD_CHARS = 2 << 20       # text head for the sample (chars)
MAX_SAMPLE_ROWS = 3000     # sample records
MIN_ROWS = 5               # fewer data rows — not a registry
MIN_CELLS = 5              # fewer non-empty cells — the column kind is not judged
REGULAR_SHARE = 0.9        # share of sample rows with the modal field count

# --- header hints (normalised phrases; order is priority) ---------------------
HEADER_HINTS = (
    (EMAIL, ("email", "e mail", "mail", "почта", "эл почта", "электронная почта",
             "адрес электронной почты", "электронной почты")),
    (PHONE, ("телефон", "телефона", "тел", "phone", "mobile", "моб", "мобильный", "сотовый",
             "номер телефона", "контактный телефон", "телефоны")),
    (INN, ("инн", "inn")),
    (SNILS, ("снилс", "snils")),
    (PASSPORT, ("паспорт", "паспортные данные", "серия и номер", "серия номер",
                "документ удостоверяющий личность", "passport")),
    (DOB, ("дата рождения", "д р", "др", "год рождения", "birth", "dob",
           "birth date", "date of birth")),
    (POSTCODE, ("индекс", "почтовый индекс", "zip", "postcode", "postal code")),
    (FLAT, ("квартира", "квартиры", "кв", "номер квартиры", "помещение", "пом", "офис",
            "flat", "apartment", "apt")),
    (HOUSE, ("дом", "дома", "д", "номер дома", "корпус", "корп", "строение", "стр",
             "house", "building")),
    (STREET, ("улица", "улицы", "ул", "street")),
    (SETTLEMENT, ("населенный пункт", "населенного пункта", "нас пункт", "нп", "н п", "город", "гор",
                  "село", "поселок", "деревня", "поселение", "city", "town",
                  "settlement", "locality")),
    (ADDRESS, ("адрес", "адрес регистрации", "адрес проживания", "место жительства",
               "адрес объекта", "address")),
    (ACCOUNT, ("лицевой счет", "лицевой счёт", "лицевого счета", "л с", "лс", "лиц счет", "лиц счёт",
               "номер лицевого счета", "номер л с", "account", "acct", "личный счет")),
    (ORG, ("организация", "компания", "контрагент", "юр лицо", "юрлицо",
           "наименование организации", "поставщик", "управляющая компания",
           "organization", "organisation", "company", "vendor", "supplier")),
    (PERSON, ("фио", "ф и о", "фамилия", "имя", "отчество", "абонент", "плательщик",
              "собственник", "владелец", "потребитель", "заявитель", "наниматель",
              "квартиросъемщик", "застрахованный", "сотрудник", "работник",
              "контактное лицо", "руководитель", "директор", "ответственный",
              "name", "full name", "fullname", "first name", "last name", "surname",
              "contact", "customer name")),
)
# a code/lookup word in the header: the column is not PII, even with a hint
CODE_WORDS = frozenset({"код", "code", "id", "идентификатор", "тип", "type",
                        "статус", "status", "признак", "кол", "количество",
                        "сумма", "тариф", "флаг", "ключ", "key"})
_NORM_RX = re.compile(r"[^\w]+|_", re.UNICODE)


def norm_header(h: str) -> str:
    """Header → space-separated words: casefold, ё→е, punctuation → space."""
    s = h.replace("﻿", "").casefold().replace("ё", "е")
    return " ".join(_NORM_RX.sub(" ", s).split())


def header_kind(h: str):
    """Kind from the header: ``kind`` | ``"code"`` (lookup/code) | None."""
    norm = norm_header(h)
    if not norm:
        return None
    pad = " " + norm + " "
    if any(" " + w + " " in pad for w in CODE_WORDS):
        return "code"
    for kind, hints in HEADER_HINTS:
        for hint in hints:
            if " " + hint + " " in pad:
                return kind
    return None


# --- cell forms --------------------------------------------------------------
_LETTER = re.compile(r"[A-Za-zА-Яа-яЁё]")
_DIGITS = re.compile(r"\d")
_EMAIL_RX = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")
_NUMERIC = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_DATE_RX = re.compile(r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?")
_HOUSE_RX = re.compile(r"(?i)(?:(?:д|дом|влд|вл|владение|стр|строение|корп|корпус|к)\.?\s*)?"
                       r"\d{1,4}[а-яёa-z]?(?:\s*[/\-]\s*\d{1,3}[а-яёa-z]?)?"
                       r"(?:\s*(?:к|корп|корпус|стр|с|лит|литер)\.?\s*[\dа-яёa-z]{1,3})?")
_FLAT_RX = re.compile(r"(?i)(?:(?:кв|квартира|пом|помещение|оф|офис)\.?\s*)?\d{1,5}[а-яёa-z]?")
_PASSPORT_RX = re.compile(r"\d{2}\s?\d{2}\s?\d{6}|\d{4}\s?\d{6}|\d{10}")
_INITIALS = re.compile(r"[А-ЯЁA-Z]\.?|(?:[А-ЯЁA-Z]\.){1,2}[А-ЯЁA-Z]\.?")
_CYR_CAP = re.compile(r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁа-яё][а-яё]+)?")
_CYR_UP = re.compile(r"[А-ЯЁ]{2,}(?:-[А-ЯЁ]{2,})?")
_LAT_CAP = re.compile(r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?")
_LAT_UP = re.compile(r"[A-Z]{2,}(?:-[A-Z]{2,})?")
_PATR = re.compile(r"(?:ович|евич|ьич|ич|овна|евна|ична|инична)$")
_SURN_END = re.compile(SURNAME_ENDINGS + r"$")
_LAT_SURN = re.compile(r"(?:ov|ev|in|yn|ova|eva|ina|sky|ski|skiy|skaya|enko|uk|yuk|"
                       r"chuk|yan|ian|ich|vich|dze|shvili)$", re.IGNORECASE)
# legal form at the cell start — the same alternation as the text path's opf_name rule
# (LEGAL_FORMS + HOUSING_FORMS: ТСЖ/ЖСК/СНТ…: one token for cell ↔ text)
_OPF_HEAD = re.compile(r"^(?P<opf>(?:" + _orgs._OPF + r")\.?)(?P<sp>\s+|(?=[«\"']))")
# street marker: the address detector vocabulary + abbreviations without a dot ("ул Ленина" —
# a registry form the detector does not see in text without "ул.")
_STREET_WORD = (r"(?:" + _addr._STREET_MARKER + r"|ул|улица|пр|просп|пр-?кт|пр-?т|пер|наб|ш|"
                r"б-р|бульвар|пр-?д|проезд|туп|тупик|аллея|мкр|кв-?л|пл|площадь|линия|тракт)\.?")
_STREET_HEAD = re.compile(r"^(?P<m>(?i:" + _STREET_WORD + r")(?:\s+|(?<=\.)))(?=\S)")
_STREET_TAIL = re.compile(r"(?P<m>\s+(?i:" + _STREET_WORD + r"))$")
_CITY_HEAD = re.compile(r"^(?P<m>(?:(?i:г|гор|город|с|село|пос|п|поселок|посёлок|дер|д|деревня|"
                        r"ст|станица|х|хутор|рп|пгт|снт|мкр)\.?\s+|(?i:г|с|д|п)\.))")
_HOUSE_HEAD = re.compile(r"^(?P<m>(?i:д|дом|влд|вл|владение)\.?\s*)(?=\d)")
_FLAT_HEAD = re.compile(r"^(?P<m>(?i:кв|квартира|пом|помещение|оф|офис)\.?\s*)(?=\d)")
_QUOTES = "«»„“”\"'"
_EN_GIVEN = frozenset(n.lower() for n in EN_FIRST_NAMES)
_RU_GIVEN = frozenset(RU_FIRST_NAMES_ALL) | frozenset(RU_FIRST_NAMES_DIM)


def _cap(w: str) -> str:
    """ALL-CAPS word → Capitalized for the dictionaries (ИВАНОВ → Иванов)."""
    return w.capitalize() if w.isupper() and len(w) > 1 else w


def _name_words(s: str) -> list:
    return [w for w in re.split(r"[\s,]+", s.strip(_QUOTES + " ()")) if w]


def _person_parse(s: str):
    """Name cell → (words, surname | None, evidence: bool, latin: bool) | None.
    Words: 1–5, each Cap/ALL-CAPS in one alphabet, initials or a patronymic; evidence —
    a known given name, a patronymic, surname morphology."""
    words = _name_words(s)
    if not 1 <= len(words) <= 5 or _DIGITS.search(s):
        return None
    cyr = lat = False
    given, patr, cands, evid = [], [], [], False
    for w in words:
        if _INITIALS.fullmatch(w):
            continue                     # initial(s): "И.", "И.И.", "И.И"
        w = w.strip(".")
        if not w:
            continue
        c = _cap(w)
        if _CYR_CAP.fullmatch(c) or _CYR_UP.fullmatch(w):
            cyr = True
            if c in _RU_GIVEN:
                given.append(c)
                evid = True
            elif _PATR.search(c.lower()) and len(c) > 5:
                patr.append(c)
                evid = True
            else:
                cands.append(c)
                if _SURN_END.search(_names._surname_nominative(c)):
                    evid = True
        elif _LAT_CAP.fullmatch(c) or _LAT_UP.fullmatch(w):
            lat = True
            if c.lower() in _EN_GIVEN:
                given.append(c)
                evid = True
            else:
                cands.append(c)
                if _LAT_SURN.search(c):
                    evid = True
        else:
            return None
    if cyr and lat:
        return None
    if not (given or patr or cands):
        return None                      # initials only
    surname = cands[0] if cands else (given[0] if given else patr[0])
    return words, surname, evid, lat


def cell_features(cell: str) -> set:
    """Form features of the WHOLE cell (for column statistics). Already a token /
    placeholder — empty (evidence neither for nor against)."""
    s = cell.strip().strip(_QUOTES + " ")
    if not s or is_own_token(s, part=True):
        return set()
    f = set()
    d = sum(ch.isdigit() for ch in s)
    if _LETTER.search(s):
        f.add("letters")
    if _EMAIL_RX.fullmatch(s):
        f.add("email")
    if d >= 10 and not _LETTER.search(s) and len(s) <= 24 \
            and re.fullmatch(r"\+?[\d\s\-().]+", s):
        digits = re.sub(r"\D", "", s)
        if (len(digits) in (10, 11) and V.phone_ru_ok(digits)) \
                or (s.startswith("+") and 10 <= len(digits) <= 15):
            f.add("phone")
    if re.fullmatch(r"\d+", s):
        f.add("digits")
        if len(s) in (10, 12):
            f.add("inn_digits")
            if V.inn_ok(s):
                f.add("inn")
        if len(s) == 11:
            f.add("snils_digits")
            if V.snils_ok(s):
                f.add("snils")
        if len(s) == 6 and s[0] in "123456":
            f.add("postcode")
        if 3 <= len(s) <= 20:
            f.add("account")
    elif re.fullmatch(r"\d{3}-\d{3}-\d{3}[\s-]\d{2}", s):
        f.add("snils_digits")
        if V.snils_ok(s):
            f.add("snils")
    if _NUMERIC.fullmatch(s):
        f.add("numeric")
    if _DATE_RX.fullmatch(s):
        f.add("date")
    if _HOUSE_RX.fullmatch(s):
        f.add("house")
    if _FLAT_RX.fullmatch(s):
        f.add("flat")
    if _PASSPORT_RX.fullmatch(s):
        f.add("passport")
    if _OPF_HEAD.match(s) or (s[:1] == "«" and s[-1:] == "»" and _LETTER.search(s)):
        f.add("opf")                     # legal form or «quotes» — an organization (like quoted in text)
    elif "letters" in f:
        p = _person_parse(s)
        if p is not None:
            f.add("person_loose")
            if p[2]:
                f.add("person")
    if "letters" in f:
        if _STREET_HEAD.match(s) or _STREET_TAIL.search(s):
            f.add("street")
            if d:
                f.add("address")
        if _CITY_HEAD.match(s):
            f.add("settlement")
        if "," in s and d and re.search(r"(?i)" + _addr._STREET_MARKER + r"|д\.|дом|кв\.", s):
            f.add("address")
    return f


# --- profile -----------------------------------------------------------------
@dataclass
class TableProfile:
    delimiter: str
    has_header: bool
    ncols: int
    headers: list                      # headers (empty without a header row)
    kinds: dict                        # column index → kind
    rows: int = 0                      # data rows in the sample
    how: dict = field(default_factory=dict)   # index → "header"|"stat"|"both"

    def summary(self) -> str:
        parts = []
        for i in sorted(self.kinds):
            h = self.headers[i] if i < len(self.headers) and self.headers[i] else f"#{i}"
            parts.append(f"{h}: {self.kinds[i]}")
        return (f"columns {self.ncols}, header {'yes' if self.has_header else 'no'}, "
                f"PII {len(self.kinds)} ({', '.join(parts)})")

    def by_header(self) -> dict:
        """{header | "#i": kind} — for state/tabular_profiles.json."""
        out = {}
        for i, k in sorted(self.kinds.items()):
            h = self.headers[i] if i < len(self.headers) and self.headers[i].strip() else f"#{i}"
            out[h] = k
        return out


def _sample_rows(text: str, delimiter: str) -> list:
    head = text
    if len(text) > HEAD_CHARS:
        cut = text.rfind("\n", 0, HEAD_CHARS)
        head = text[:cut + 1] if cut > 0 else text[:HEAD_CHARS]
    # raise the csv field limit up to the file cap (it was not raised here before, so the
    # "registry or not" decision on a head with a cell > 128 KB depended on process history)
    raise_csv_field_limit()
    rows = []
    for row in csv.reader(io.StringIO(head), delimiter=delimiter):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        rows.append(row)
        if len(rows) >= MAX_SAMPLE_ROWS:
            break
    if len(text) > len(head) and rows and len(rows) < MAX_SAMPLE_ROWS:
        rows.pop()                          # the cut-off last record of the head
    return rows


def _col_stats(rows: list, ncols: int) -> list:
    stats = [Counter() for _ in range(ncols)]
    for row in rows:
        for i, cell in enumerate(row[:ncols]):
            if not cell or not cell.strip():
                continue
            f = cell_features(cell)
            st = stats[i]
            st["n"] += 1
            if not f:
                st["token"] += 1
            for k in f:
                st[k] += 1
    return stats


def _looks_header(row0: list, hkinds: list, stats: list) -> bool:
    """Row 0 is a header: it has no numeric/email/phone/date cells and
    either has a header hint, or at least one data column is ≥ 80 %
    numeric/email/phone/date while the row 0 cell is text."""
    for c in row0:
        f = cell_features(c)
        if f & {"numeric", "email", "phone", "date", "digits"}:
            return False
    if any(k is not None for k in hkinds):
        return True
    for i, st in enumerate(stats):
        n = st["n"]
        if n >= MIN_CELLS and (st["numeric"] + st["email"] + st["phone"] + st["date"]) >= 0.8 * n:
            return True
    return False


def _decide(hk, st: Counter):
    n = st["n"]
    if n < MIN_CELLS or hk == "code":
        return None, None
    share = lambda k: st[k] / n            # noqa: E731
    stat = None
    for kind, feat, thr in STAT_KINDS:
        if share(feat) >= thr:
            stat = kind
            break
    if hk is None:
        return stat, "stat"
    if stat in _STAT_OVERRIDES and stat != hk:
        return stat, "stat"
    feat, thr = HEADER_NEED[hk]
    if share(feat) >= thr:
        return hk, "both" if stat == hk else "header"
    return None, None


def profile_text(rel: str, text: str):
    """Registry profile from the text head, or None (not a table / no PII columns /
    delimiter not recognised). csv.Error never escapes."""
    if not rel.lower().endswith((".csv", ".tsv")) or not text:
        return None
    try:
        delimiter = csv_delimiter(rel, text)
        rows = _sample_rows(text, delimiter)
    except (csv.Error, CsvStructureError, ValueError):
        return None
    if len(rows) < MIN_ROWS + 1:
        return None
    ncols = Counter(len(r) for r in rows).most_common(1)[0][0]
    if ncols < 2 or sum(1 for r in rows if len(r) == ncols) < REGULAR_SHARE * len(rows):
        return None
    row0 = rows[0] if len(rows[0]) == ncols else [""] * ncols
    hkinds = [header_kind(h) for h in row0]
    stats = _col_stats(rows[1:], ncols)
    has_header = len(rows[0]) == ncols and _looks_header(row0, hkinds, stats)
    if not has_header:
        hkinds = [None] * ncols
        stats = _col_stats(rows, ncols)
        row0 = [""] * ncols
    data_rows = len(rows) - (1 if has_header else 0)
    if data_rows < MIN_ROWS:
        return None
    kinds, how = {}, {}
    for i in range(ncols):
        k, w = _decide(hkinds[i], stats[i])
        if k is not None:
            kinds[i] = k
            how[i] = w
    if not kinds:
        return None
    return TableProfile(delimiter, has_header, ncols, list(row0), kinds, data_rows, how)


# --- cell tokens --------------------------------------------------------------
def person_token(s: str) -> str:
    """PERSON_NAME for a name cell: the seed is the surname nominative, as in
    detectors.names (official order / pair / initials); latin — the
    canonical surname (_lat_pair_mask)."""
    p = _person_parse(s)
    if p is None:
        words = _name_words(s) or [s.strip()]
        return mint.pname(_names._seed_person(_names._surname_nominative(words[0])))
    _w, surname, _e, lat = p
    if lat:
        return mint.pname(_names._canon_lat(surname))
    return mint.pname(_names._seed_person(_names._surname_nominative(surname)))


def org_token(name: str) -> str:
    """ORG for an organization name without the legal form: leading descriptors ("Компания",
    "Группа") stay, as in opf_name; the seed is the rest."""
    kept, rest = _orgs._split_leading_descriptors(name.strip())
    if not rest.strip():
        return mint.tag("ORG", name.strip())
    return kept + mint.tag("ORG", rest.strip())


def _strip_quotes(s: str):
    """(opening quote, body, closing quote)."""
    oq = cq = ""
    if s and s[0] in _QUOTES:
        oq, s = s[0], s[1:]
    if s and s[-1] in _QUOTES:
        cq, s = s[-1], s[:-1]
    return oq, s.strip(), cq


def party_value(s: str, default: str) -> str:
    """Subscriber/organization value: legal form → ORG (legal form and quotes stay; "ИП" →
    PERSON by the surname); otherwise the column default, but a full name with evidence in
    an org column → person, a legal-form-like brand in a person column → org."""
    m = _OPF_HEAD.match(s)
    if m:
        opf, sp = m.group("opf"), m.group("sp")
        rest = s[m.end():]
        oq, body, cq = _strip_quotes(rest.strip())
        if not body:
            return s
        if _orgs._IP_RE.match(opf.rstrip(".")):
            return opf + (sp or " ") + oq + person_token(body) + cq
        return opf + sp + oq + org_token(body) + cq
    oq, body, cq = _strip_quotes(s)
    if not body:
        return s
    if default == ORG:
        p = _person_parse(body)
        if p is not None and p[2] and len(p[0]) >= 2:
            return oq + person_token(body) + cq
        return oq + org_token(body) + cq
    return oq + person_token(body) + cq


def _const(token: str, need=None):
    def fn(s: str) -> str:
        if need is not None and not need(s):
            return s
        return token
    return fn


def _marker_value(head_rx, token: str, tail_rx=None):
    def fn(s: str) -> str:
        if not _LETTER.search(s) and not _DIGITS.search(s):
            return s
        m = head_rx.match(s) if head_rx is not None else None
        if m:
            return m.group("m") + token
        if tail_rx is not None:
            t = tail_rx.search(s)
            if t:
                return token + t.group("m")
        return token
    return fn


_STREET_ABBR = re.compile(r"(?i)^(?:ул|пер|пр-?кт|пр-?т|просп|пр|наб|ш|б-р|пр-?д|туп|мкр|"
                          r"кв-?л|пл)\.?\s")


def _street_value(s: str) -> str:
    """Street: a marker at the end goes after the token ("Набережная ул." / "Набережная улица" →
    STREET + marker: the street name itself looks like a marker, so the tail marker is checked first
    unless the head is an abbreviation marker), a marker at the start stays ("ул. Набережная" →
    "ул. " + STREET), otherwise the whole cell."""
    if not _LETTER.search(s) and not _DIGITS.search(s):
        return s
    t = _STREET_TAIL.search(s)
    if t and t.start() > 0 and not _STREET_ABBR.match(s):
        return P.STREET + t.group("m")
    m = _STREET_HEAD.match(s)
    if m and s[m.end():].strip():
        return m.group("m") + P.STREET
    return P.STREET


def _house_value(head_rx, token):
    def fn(s: str) -> str:
        if not _DIGITS.search(s):
            return s
        m = head_rx.match(s)
        return (m.group("m") if m else "") + token
    return fn


_has_letter = lambda s: bool(_LETTER.search(s))                 # noqa: E731
_has_digit = lambda s: bool(_DIGITS.search(s))                  # noqa: E731
_ndigits = lambda s, n: sum(ch.isdigit() for ch in s) >= n      # noqa: E731

VALUE_FN = {
    PERSON: lambda s: party_value(s, PERSON),
    ORG: lambda s: party_value(s, ORG),
    EMAIL: _const(P.EMAIL, lambda s: "@" in s),
    PHONE: _const(P.PHONE, lambda s: _ndigits(s, 6)),
    INN: _const(P.INN, lambda s: _ndigits(s, 5)),
    SNILS: _const(P.SNILS, lambda s: _ndigits(s, 9)),
    ACCOUNT: _const(P.ACCOUNT, lambda s: bool(re.search(r"\w", s))),
    PASSPORT: _const(P.PASSPORT, _has_digit),
    DOB: _const(P.DATE, _has_digit),
    ADDRESS: _const(P.ADDRESS, _has_letter),
    SETTLEMENT: _marker_value(_CITY_HEAD, P.GEO),
    STREET: _street_value,
    HOUSE: _house_value(_HOUSE_HEAD, P.HOUSE),
    FLAT: _house_value(_FLAT_HEAD, P.FLAT),
    POSTCODE: _const(P.POSTCODE, lambda s: bool(re.fullmatch(r"\d{6}", s.strip()))),
}


def cell_value(cell: str, kind: str, cache: dict | None = None) -> str:
    """A cell of a ``kind`` column → the masked cell (edge spaces
    intact; already a token / no letters or digits — verbatim)."""
    s = cell.strip()
    if not s or is_own_token(s, part=True):
        return cell
    # client role labels in a cell ("Исполнитель 12", "Исполнитель 1, Исполнитель
    # 4") get the same ORG token in a column of any kind as the rule gives in text
    # (orgs.CLIENT_ROLE_WORDS); the rest of the cell next to a label ("Исполнитель 1,
    # Иванов Иван") gets the column kind value, as the pack column rule cuts it
    roles = _orgs.client_roles_split(s, VALUE_FN[kind])
    if roles is not None:
        return cell[: len(cell) - len(cell.lstrip())] + roles + cell[len(cell.rstrip()):]
    if cache is not None:
        hit = cache.get((kind, s))
        if hit is not None:
            new = hit
        else:
            new = VALUE_FN[kind](s)
            if len(cache) < _CACHE_MAX:
                cache[(kind, s)] = new
    else:
        new = VALUE_FN[kind](s)
    if new == s:
        return cell
    lead = cell[: len(cell) - len(cell.lstrip())]
    trail = cell[len(cell.rstrip()):]
    return lead + new + trail


_CACHE_MAX = 1_000_000
# kinds whose values go to the roster (mine candidates and the group decision store): approved class
ROSTER_KINDS = {PERSON: "person", ORG: "org", STREET: "geo", SETTLEMENT: "geo"}


# --- roster: the cell value literal for the text path -------------------------
def _person_literal(body: str):
    p = _person_parse(body)
    if p is not None:
        return _cap(p[1])
    words = _name_words(body)
    return _cap(words[0]) if words and _LETTER.search(words[0]) else None


def roster_literal(kind: str, cell: str):
    """(approved class, literal) of a roster column cell value for the text path: the surname
    as in the cell (person; "ИП Фамилия И.О." — the surname), the organization name without
    legal form and quotes (org), the street / settlement name without the marker (geo).
    None — no literal (no letters)."""
    s = cell.strip()
    if not s or not _LETTER.search(s) or is_own_token(s, part=True):
        return None
    cores = _orgs.client_roles_rest(s)
    if cores is not None:
        # a client role label is not a roster candidate; the literal comes from the rest of
        # the cell next to the label ("Исполнитель 1, Иванов Иван" → "Иванов", not "Исполнитель")
        cores = [c for c in cores if _LETTER.search(c)]
        if not cores:
            return None
        s = cores[0]
    if kind in (PERSON, ORG):
        m = _OPF_HEAD.match(s)
        if m:
            _oq, body, _cq = _strip_quotes(s[m.end():].strip())
            if not body or not _LETTER.search(body):
                return None
            if _orgs._IP_RE.match(m.group("opf").rstrip(".")):
                lit = _person_literal(body)
                return ("person", lit) if lit else None
            return "org", body
        _oq, body, _cq = _strip_quotes(s)
        if not body:
            return None
        if kind == PERSON:
            lit = _person_literal(body)
            return ("person", lit) if lit else None
        p = _person_parse(body)
        if p is not None and p[2] and len(p[0]) >= 2:
            return "person", _cap(p[1])
        return "org", body
    if kind == STREET:
        t = _STREET_TAIL.search(s)
        if t and t.start() > 0 and not _STREET_ABBR.match(s):
            name = s[:t.start()]
        else:
            m = _STREET_HEAD.match(s)
            name = s[m.end():].strip() if m and s[m.end():].strip() else s
        return ("geo", name) if _LETTER.search(name) else None
    if kind == SETTLEMENT:
        m = _CITY_HEAD.match(s)
        name = s[m.end():].strip() if m else s
        return ("geo", name) if _LETTER.search(name) else None
    return None


_ROSTER_WORD = re.compile(r"[A-Za-zА-Яа-яЁё][\w-]*")


def roster_stems(klass: str, literal: str) -> list:
    """Word stems of the literal for finding matches outside registry cells (with a margin: a surname
    in an oblique case, without initials, "Ромашка/Ромашки", lower case): lower, ё→е, for a surname
    the case ending is cut (name_dicts.strip_case_tail), then one final vowel/ь/й is dropped
    if the rest is at least 4 chars ("Бондарь" → "бондар" catches "Бондарю")."""
    from ..engine import name_dicts
    out = []
    for w in _ROSTER_WORD.findall(literal):
        low = w.lower().replace("ё", "е")
        if len(low) < 3:
            continue
        if klass == "person":
            low = name_dicts.strip_case_tail(low)
        if len(low) > 4 and low[-1] in "аяуюеиоыьй":
            low = low[:-1]
        out.append(low)
    return out


# --- walking records with the layout in mind ----------------------------------------
def _looks_new_header(row: list) -> bool:
    """A data row looks like the header of another table: no digits in any cell, at least
    two cells carry a header hint (kind or code), no cell looks like a full name/email/
    phone or is longer than 40 chars."""
    if any(ch.isdigit() for f in row for ch in f):
        return False
    hinted = 0
    for f in row:
        if len(f) > 40:
            return False
        feats = cell_features(f)
        if feats & {"person", "email", "phone"}:
            return False
        if header_kind(f) is not None:
            hinted += 1
    return hinted >= 2


def _records(text: str, delimiter: str):
    """csv.reader records with the raw text of each. The csv field limit is raised up to the file
    cap (raise_csv_field_limit: a cell over 128 KB in a registry tail used to crash the mine/NER
    blank with csv.Error). Other csv.Error (CR in an unquoted field) escape: the caller
    gives a WARN and takes the normal path."""
    raise_csv_field_limit()
    tracker = _LineTracker(io.StringIO(text))
    for row in csv.reader(tracker, delimiter=delimiter):
        yield row, tracker.take()


def iter_layout(text: str, prof: TableProfile):
    """File records with the column kinds in force for each: (raw, row, kinds | None).
    kinds None — the profile does not touch the record: the profile header and its repeats,
    a row with another field count (a shift — the normal path), a header row of another table
    (the layout is re-profiled from the next up to MAX_SAMPLE_ROWS rows — own kinds for
    the rows after it; no kinds — the rows take the normal path until the next header)."""
    from collections import deque
    it = _records(text, prof.delimiter)
    buf: deque = deque()
    headers = list(prof.headers) if prof.has_header else None
    kinds, ncols = dict(prof.kinds), prof.ncols
    first = True
    while True:
        if buf:
            row, raw = buf.popleft()
        else:
            try:
                row, raw = next(it)
            except StopIteration:
                return
        if first:
            first = False
            if prof.has_header:
                yield raw, row, None
                continue
        if headers is not None and row == headers:
            yield raw, row, None                      # header repeat
            continue
        if len(row) >= 2 and _looks_new_header(row):
            ahead = []
            for _ in range(MAX_SAMPLE_ROWS):
                try:
                    ahead.append(next(it))
                except StopIteration:
                    break
            buf.extend(ahead)
            n = len(row)
            stats = _col_stats([r for r, _raw in ahead if len(r) == n], n)
            hk = [header_kind(h) for h in row]
            kinds = {}
            for i in range(n):
                k, _w = _decide(hk[i], stats[i])
                if k is not None:
                    kinds[i] = k
            headers, ncols = list(row), n
            STATS["tabular_relayout"] += 1
            yield raw, row, None
            continue
        if len(row) != ncols:
            yield raw, row, None                      # field shift — the normal path
            continue
        yield raw, row, kinds


def cell_changes(cell: str, kind: str) -> bool:
    """Whether the profile will replace the cell (without minting: mine/NER run without the
    tenant salt). For person/org — the same conditions as party_value (letters, not a token,
    a non-empty name after the legal form and quotes); other kinds — value_fn itself
    (constants and markers, no mint)."""
    s = cell.strip()
    if not s or is_own_token(s, part=True):
        return False
    if kind in (PERSON, ORG):
        if not _LETTER.search(s):
            return False
        m = _OPF_HEAD.match(s)
        body = _strip_quotes((s[m.end():] if m else s).strip())[1]
        return bool(body)
    return VALUE_FN[kind](s) != s


def _new_row(row: list, kinds: dict, cache: dict, counts: Counter, roster=None,
             blank: bool = False):
    """The masked record (a list), or None if no cell changed.
    ``blank`` — only the replacement flag (cell_changes), no minting: cell → ''."""
    new_row = None
    for i, kind in kinds.items():
        if i >= len(row):
            continue
        cell = row[i]
        if not cell or not cell.strip():
            continue
        if blank:
            new = "" if cell_changes(cell, kind) else cell
        else:
            new = cell_value(cell, kind, cache)
        if new != cell:
            if new_row is None:
                new_row = list(row)
            new_row[i] = new
            counts[kind] += 1
            if roster is not None and kind in ROSTER_KINDS:
                lit = roster_literal(kind, cell)
                if lit is not None:
                    roster[lit] = roster.get(lit, 0) + 1
    return new_row


def _subst_record(raw: str, row: list, new_row: list, delimiter: str, blank: bool):
    """The record with changed cells substituted (blank — spaces of the same length instead of
    the token: quotes and line breaks inside the cell stay intact); None — the cell scanner
    could not parse the record (mask — csv.writer, blank — the record as is)."""
    spans = _csv_record_cells(raw, delimiter)
    if spans is None or len(spans) != len(row) \
            or not all(_csv_cell_text(raw, sp) == c for sp, c in zip(spans, row)):
        return None
    piece, pos = [], 0
    for sp, old, new in zip(spans, row, new_row):
        if new == old:
            continue
        piece.append(raw[pos:sp[0]])
        if blank:
            piece.append(_BLANK_RE.sub(" ", raw[sp[0]:sp[1]]))
        else:
            piece.append(_csv_quote(new, delimiter, sp[2] or (len(spans) == 1 and not new)))
        pos = sp[1]
    piece.append(raw[pos:])
    return "".join(piece)


# --- masking and blanking -----------------------------------------------------
def mask_text(text: str, prof: TableProfile, report=None, rule_prefix: str = "tab_") -> str:
    """Registry text with the cells of PII columns masked. Writing is as in
    formats.mask_csv_cells (substitution by raw record spans; a record the
    scanner could not parse is rewritten by csv.writer alone), streamed by record (iter_layout:
    the header and its repeats, rows with shifted fields and tables of another layout — by their own rules).
    ``report`` gets one ``<prefix><kind>`` entry per changed cell (dry == apply)."""
    delimiter = prof.delimiter
    raise_csv_field_limit()
    cache: dict = {}
    counts: Counter = Counter()
    out: list = []
    consumed = 0
    changed = warned = False
    n = 0
    for raw, row, kinds in iter_layout(text, prof):
        n += 1
        consumed += len(raw)
        new_row = _new_row(row, kinds, cache, counts) if kinds else None
        if new_row is None:
            out.append(raw)
            continue
        changed = True
        piece = _subst_record(raw, row, new_row, delimiter, blank=False)
        if piece is not None:
            out.append(piece)
            continue
        if not warned:
            print(f"tabular: WARN csv: record {n} not parsed by the cell scanner — "
                  "rebuilding the record with csv.writer", flush=True)
            warned = True
        eol = _csv_record_eol(raw)
        buf = io.StringIO()
        csv.writer(buf, delimiter=delimiter, lineterminator=eol or "\n").writerow(new_row)
        out.append(buf.getvalue() if eol else buf.getvalue()[:-1])
    if not changed:
        return text
    if consumed != len(text):
        print("tabular: WARN csv: raw records do not match the text — file left without profile edits",
              flush=True)
        return text
    if report is not None:
        for kind, c in sorted(counts.items()):
            for _ in range(c):
                report.record(rule_prefix + kind, "<cell>", f"<{kind}>")
    return "".join(out)


_BLANK_RE = re.compile(r'[^"\r\n]')


def blank_text(text: str, prof: TableProfile, roster: dict | None = None) -> str:
    """Text where the cells the profile ACTUALLY replaces (cell_value ≠ cell) → spaces
    of the same length (quotes and line breaks inside the cell stay: length, line count and the
    offsets of other characters do not change) — the input of the mine literal harvesters and NER layers.
    A PII column cell the kind does not take (letters under "House", a field shift, another table) stays:
    the normal path judges it. ``roster`` collects roster values {(class, literal): cell
    count} (roster_literal) from the replaced cells of ROSTER_KINDS kinds."""
    cache: dict = {}
    counts: Counter = Counter()
    out: list = []
    consumed = 0
    changed = False
    for raw, row, kinds in iter_layout(text, prof):
        consumed += len(raw)
        new_row = _new_row(row, kinds, cache, counts, roster, blank=True) if kinds else None
        piece = _subst_record(raw, row, new_row, prof.delimiter, blank=True) \
            if new_row is not None else None
        if piece is None:
            out.append(raw)
        else:
            changed = True
            out.append(piece)
    if not changed or consumed != len(text):
        return text
    return "".join(out)
