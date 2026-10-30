"""Fixed redaction constants (the `*_REDACTED` family).

Minted `PREFIX_NNNN` tokens (see mint.py) preserve linkage — the same value gets
the same token everywhere. `*_REDACTED` constants are for values where linkage
is undesirable or meaningless (a one-off secret, a coordinate, a birth date).
"""
SECRET = "SECRET_REDACTED"
ENCODED = "ENCODED_REDACTED"
EMAIL = "EMAIL_REDACTED"
PHONE = "PHONE_REDACTED"
GEO = "GEO_REDACTED"
DATE = "DATE_REDACTED"
ADDRESS = "ADDRESS_REDACTED"
STREET = "STREET_REDACTED"
HOUSE = "HOUSE_REDACTED"
FLAT = "FLAT_REDACTED"
POSTCODE = "POSTCODE_REDACTED"
SNILS = "SNILS_REDACTED"
IP = "IP_REDACTED"
MAC = "MAC_REDACTED"
CARD = "CARD_REDACTED"
IBAN = "IBAN_REDACTED"
BIK = "BIK_REDACTED"
KPP = "KPP_REDACTED"
OKPO = "OKPO_REDACTED"
OGRN = "OGRN_REDACTED"
INN = "INN_REDACTED"
ACCOUNT = "ACCOUNT_REDACTED"
SWIFT = "SWIFT_REDACTED"
PASSPORT = "PASSPORT_REDACTED"
DEPTCODE = "DEPTCODE_REDACTED"
AWS_ACCOUNT = "AWS_ACCOUNT_REDACTED"
COOKIE = "COOKIE_REDACTED"
PLATE = "PLATE_REDACTED"
HEALTH = "HEALTH_REDACTED"
ETHNICITY = "ETHNICITY_REDACTED"
# Russian traces behind switches (anonymizer/known_values/ru_trace.py): ruble currency and
# Russian regulations (152-FZ, Civil Code, GOST) — off by default, the owner decides
CURRENCY = "CURRENCY_REDACTED"
LAW = "LAW_REDACTED"

ALL = {v for k, v in dict(globals()).items() if isinstance(v, str) and v.endswith("_REDACTED")}
