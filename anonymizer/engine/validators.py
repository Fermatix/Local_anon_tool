"""Checksum / structural validators.

Self-validating detectors (INN, OGRN, IBAN, card-Luhn, SNILS, 20-digit account)
run globally WITHOUT a label because the checksum itself is the evidence — this
is what kills false positives on random long digit strings. Format-only
detectors (bare phone, 6-digit ZIP, solid-16 card) still need a context label.

Lifted and completed from the project's read-only `probe_checksum.py`; SNILS
mod-101 and card Luhn (which the corpus never implemented) are added here.
"""
from __future__ import annotations

import os

# ---- region code in INN/KPP --------------------------------------------------
# The first 4 digits of INN and KPP are the tax office code: 2 digits of the Russian region + 2
# digits of the office number within the region. Only the region is checked: the office number
# cannot be checked without the historical registry (offices merged, INNs of the 1990s were issued
# by closed offices, "00" occurs in fixtures), while the list of regions is
# stable and vendored in data/inn_regions.txt (source and date are in the file header;
# abolished autonomous areas 80–88 and 99 inter-regional are kept: their INNs are valid).
# An honest cut (200k random numbers): the checksum of 10-digit ones passes in
# ~10 %, the region cuts 4 codes of 100 (00, 96, 97, 98) → ~9.6 %; epoch numbers
# (leading 14..18) are not cut by the region at all — financial._needs_id_ctx holds them.
_REGIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "inn_regions.txt")


def _load_inn_regions(path: str = _REGIONS_FILE) -> frozenset:
    """Set of 2-digit codes from the table; no file / a broken line /
    a suspiciously short table — SystemExit (fail loudly: a silent
    empty set would reject EVERY INN)."""
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as e:
        raise SystemExit(f"validators: no INN region table {path} ({e})")
    codes = set()
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        code = ln.split("\t", 1)[0].strip()
        if len(code) != 2 or not code.isdigit():
            raise SystemExit(f"validators: bad line in the region table {path}: {ln!r}")
        codes.add(code)
    if len(codes) < 80:
        raise SystemExit(f"validators: INN region table {path} is incomplete: {len(codes)} codes")
    return frozenset(codes)


INN_REGIONS = _load_inn_regions()


# ---- INN (10 / 12 digit) -----------------------------------------------------
_W10 = [2, 4, 10, 3, 5, 9, 4, 6, 8]
_W12A = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
_W12B = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]


def inn_ok(s: str) -> bool:
    if not s.isdigit():
        return False
    if len(set(s)) == 1:  # 0000000000-style dummy: the checksum matches, but it is no INN
        return False
    if s[:2] not in INN_REGIONS:  # region code (00, 96–98 are not issued)
        return False
    d = [int(c) for c in s]
    if len(s) == 10:
        return (sum(w * x for w, x in zip(_W10, d[:9])) % 11) % 10 == d[9]
    if len(s) == 12:
        c11 = (sum(w * x for w, x in zip(_W12A, d[:10])) % 11) % 10
        c12 = (sum(w * x for w, x in zip(_W12B, d[:11])) % 11) % 10
        return c11 == d[10] and c12 == d[11]
    return False


# ---- OGRN / OGRNIP (13 / 15 digit) -------------------------------------------
# Structure (same for OGRN and OGRNIP): position 1 — flag, 2–3 — year
# of registration (YY), 4–5 — region code (RR, 01..99). The checksum alone
# let through 10% of unix epochs in milliseconds (13 digits: 1.42e12..1.89e12 →
# YY = 42..89); the YY/RR check cuts them to 0 without losing recall (the registry exists since
# 2002; epochs collide with real years not before ~2034).
OGRN_YEAR_MIN = 2      # the registries are kept since 2002 (old legal entities — YY=02)
OGRN_YEAR_MAX = 27     # current year + 1 (2026-09); raise once a year


def _ogrn_struct_ok(s: str) -> bool:
    yy, kk = int(s[1:3]), int(s[3:5])
    return OGRN_YEAR_MIN <= yy <= OGRN_YEAR_MAX and 1 <= kk <= 99


def ogrn_ok(s: str) -> bool:
    if not s.isdigit():
        return False
    if len(set(s)) == 1:  # 1111111111111-style dummy
        return False
    if len(s) == 13:
        if s[0] not in "125" or not _ogrn_struct_ok(s):
            return False
        return int(s[:12]) % 11 % 10 == int(s[12])
    if len(s) == 15:
        if s[0] not in "34" or not _ogrn_struct_ok(s):
            return False
        return int(s[:14]) % 13 % 10 == int(s[14])
    return False


# ---- SNILS (mod-101) ---------------------------------------------------------
def snils_ok(digits: str) -> bool:
    d = "".join(c for c in digits if c.isdigit())
    if len(d) != 11:
        return False
    num, ctrl = d[:9], int(d[9:])
    if len(set(num)) == 1:  # 000000000 style dummy
        return False
    s = sum(int(num[i]) * (9 - i) for i in range(9))
    s = s % 101
    if s == 100:
        s = 0
    return s == ctrl


# ---- IBAN (mod-97) -----------------------------------------------------------
# IBAN length by country (ISO 13616, SWIFT registry) is fixed, so a tail
# glued to an IBAN without a space («…0130 002024», two IBANs back to back) is split off
# by length (financial._iban_prefix). Countries outside the table — mod-97 only.
IBAN_LEN = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16,
    "BG": 22, "BH": 22, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28,
    "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24,
    "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18,
    "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IQ": 23,
    "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28, "LC": 32,
    "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24, "ME": 22,
    "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24,
    "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "RU": 33,
    "SA": 24, "SC": 31, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25,
    "SV": 28, "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24,
    "XK": 20,
}


def iban_ok(s: str) -> bool:
    v = s.replace(" ", "").upper()
    if not (15 <= len(v) <= 34) or not v[:2].isalpha() or not v[2:4].isdigit():
        return False
    rot = v[4:] + v[:4]
    num = "".join(str(int(c, 36)) if c.isalpha() else c for c in rot)
    try:
        return int(num) % 97 == 1
    except ValueError:
        return False


# ---- Payment card (Luhn / mod-10) -------------------------------------------
def luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not (13 <= len(d) <= 19):
        return False
    if len(set(d)) == 1:
        return False
    checksum, parity = 0, len(d) % 2
    for i, x in enumerate(d):
        if i % 2 == parity:
            x *= 2
            if x > 9:
                x -= 9
        checksum += x
    return checksum % 10 == 0


# ---- 20-digit bank account structure ----------------------------------------
_BAL = {"40702", "40802", "40817", "40820", "30101", "40503", "40701", "42301"}
_CUR = {"810", "643", "840", "978", "156", "392"}


def account_struct(s: str) -> bool:
    if not (s.isdigit() and len(s) == 20):
        return False
    if len(set(s)) <= 2:  # all-3s dummy etc.
        return False
    return s[:5] in _BAL and s[5:8] in _CUR


# ---- department code (region 01..92, not 000) --------------------------------
def dept_ok(a: str, b: str) -> bool:
    """Department code «RRL-NNN»: RR — region code (01…99), L — level
    of the department (0 — head office, 1 — city, 2 — district, 3 —
    passport desk/settlement), NNN — number. The old check
    ``1 <= int(a) <= 92`` read ALL three digits as the region — only
    codes 001…092 passed, and real ones (region 23/34/66 + level: 230, 340, 660 …)
    were rejected, so dept_code never fired (department codes
    survived the round and the audit fixer had to catch them)."""
    if not (a.isdigit() and b.isdigit() and len(a) == 3 and len(b) == 3):
        return False
    return 1 <= int(a[:2]) <= 99 and a[2] in "0123"


# ---- KPP (9 digits, no checksum) ---------------------------------------------
# NNNN PP XXX: 1–4 — tax office code (region by the table, the office is not
# checked, as for INN), 5–6 — reason of registration (01–50 Russian
# organizations, 51–99 foreign; 00 does not exist), 7–9 — registration
# number (001–999). The structure is weak (~94 % of random 9-digit numbers pass),
# so a bare KPP without a label is taken only next to an INN
# (financial.kpp_near_inn); with the label «КПП» no structure is required.
def kpp_struct(s: str) -> bool:
    if not (s.isdigit() and len(s) == 9):
        return False
    if len(set(s)) <= 2:  # 000000000 / 770000000-style dummy
        return False
    return s[:2] in INN_REGIONS and s[4:6] != "00" and s[6:] != "000"


# ---- geo coordinates ---------------------------------------------------------
def coord_in_range(lat: float, lon: float) -> bool:
    if not (abs(lat) <= 90 and abs(lon) <= 180):
        return False
    if abs(lat) < 1 and abs(lon) < 1:  # software version pair, not a location
        return False
    return True


# ---- public IPv4 -------------------------------------------------------------
def public_ipv4(s: str) -> bool:
    parts = s.split(".")
    if len(parts) != 4:
        return False
    try:
        octs = [int(p) for p in parts]
    except ValueError:
        return False
    if any(not (0 <= o <= 255) for o in octs):
        return False
    if any(len(p) > 1 and p[0] == "0" for p in parts):  # leading-zero = timecode
        return False
    if sum(1 for o in octs if o >= 10) < 2:  # kills 1.7.8.6-style version strings
        return False
    a, b = octs[0], octs[1]
    if a in (0, 10, 127) or a >= 224 or a == 255:
        return False
    if a == 192 and b == 168:
        return False
    if a == 172 and 16 <= b <= 31:
        return False
    if a == 169 and b == 254:
        return False
    return True


# ---- phone plausibility (RU) -------------------------------------------------
def phone_ru_ok(raw: str) -> bool:
    d = "".join(c for c in raw if c.isdigit())
    if len(set(d)) <= 4 or d.count("0") >= 5:  # dummy / 000-0000
        return False
    if len(d) == 11 and d[0] in "78" and d[1] in "3489":
        return True
    if len(d) == 10 and d[0] in "3489":
        return True
    return False


# ---- phone plausibility (CIS) ------------------------------------------------
# Length of the national part by country code: exactly the table length means a phone, not
# an amount/version/id; a match is evidence without a context word, so
# contacts.phone_cis is in high (phone_intl for other codes stays in
# context). +1 is deliberately absent (amounts «+1 500 000», versions). +7 7xx — Kazakhstan
# (10 digits); +7 3/4/8/9xx — Russia, phone_ru_ok takes them.
PHONE_CIS_LEN = {"374": 8, "375": 9, "380": 9, "998": 9, "995": 9, "996": 9,
                 "992": 9, "993": 8, "994": 9, "373": 8}


def phone_cis_ok(raw: str) -> bool:
    d = "".join(c for c in raw if c.isdigit())
    if len(set(d)) <= 4:  # dummy 000-0000 / 111 (as in phone_ru_ok)
        return False
    if d.startswith("77"):  # +7 7xx — KZ: 1 + 10
        return len(d) == 11
    n = PHONE_CIS_LEN.get(d[:3])
    return n is not None and len(d) == 3 + n


# ---- phone plausibility (world) ----------------------------------------------
# Codes outside Russia/CIS: the length of the national part varies by country and number type
# (UK 9–10, DE 5–13, AT 4–13…), a length table like PHONE_CIS_LEN will not do —
# the validator asks libphonenumber (package phonenumbers, optional,
# requirements-optional.txt): is_valid_number = length + prefix by the country
# metadata. No package → False, and contacts.phone_intl (context tier, format regex)
# works as before. The import is lazy — at the first candidate; measured (9.0.40,
# py3.14): import 8 ms, country metadata is loaded on demand (~1 ms per
# country), 11 µs per number when warm, 46 MB on disk. +1 (NANP) and +7 are deliberately
# not validated: +7 — phone_ru/phone_cis, +1 — amounts «+1 500 000» and the contract
# of test_fp_guards («+1 415 555 01 23» is not high). ANON_PHONENUMBERS=0 —
# turn off (the "no package" behaviour; to test both paths in one venv).
# +1 and +7 are the only one-digit codes: the first digit of the number = the country code,
# so such a candidate is cut off BEFORE the package import (Russian/Kazakh/US numbers
# do not wake the import).
PHONE_INTL_SKIP_CC = frozenset({"1", "7"})
_PN = None          # None — not tried yet; False — no package / turned off


def _phonenumbers():
    global _PN
    if _PN is None:
        if os.environ.get("ANON_PHONENUMBERS", "1") == "0":
            _PN = False
        else:
            try:
                import phonenumbers   # noqa: F401 — optional dependency
                _PN = phonenumbers
            except ImportError:
                _PN = False
    return _PN


def phone_intl_available() -> bool:
    """Whether the validator is available (the package imports and is not turned off)."""
    return bool(_phonenumbers())


def phone_intl_ok(raw: str) -> bool:
    """«+country code …» — a real number by libphonenumber metadata. Raw
    text (not only digits): «(0)» after the code is the UK/DE trunk prefix, which
    the parser drops itself («+44 (0)20 7946 0958» → 20 7946 0958)."""
    d = "".join(c for c in raw if c.isdigit())
    if len(set(d)) <= 4 or not (8 <= len(d) <= 15):  # dummy / E.164 ≤ 15 digits
        return False
    if d[0] in PHONE_INTL_SKIP_CC:                   # +1 / +7 — not ours
        return False
    pn = _phonenumbers()
    if not pn:
        return False
    try:
        num = pn.parse(raw.strip(), None)
    except pn.NumberParseException:
        return False
    return bool(pn.is_valid_number(num))
