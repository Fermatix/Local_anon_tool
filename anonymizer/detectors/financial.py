"""Financial / registration-id detectors.

Self-validating ids (INN, OGRN, IBAN, card-Luhn, 20-digit account) fire
GLOBALLY without a label — the checksum is the evidence, which is what kills FPs
on random long digit strings. Solid-16 cards (``card_solid``) are also
self-validating: Luhn is a checksum, so a bare card is minted without
a context word (the same as card_grouped/IBAN). Label-only ids (KPP, OKPO,
SWIFT, passport) and pure-format ids (SNILS) require a bank/id
context word nearby.
"""
from __future__ import annotations

import os
import re

from ..engine import mint as _mint
from ..engine import placeholders as P
from ..engine.boundary import JSON_SAFE as _JS, NUM_L, NUM_R, _ESC
from ..engine.context import BANK_CTX, PASSPORT_CTX
from ..engine.rule import Rule, keep_line_breaks, no_guard
from ..engine import validators as V

_HEX = set("0123456789abcdefABCDEF")


def _inside_hex_run(m) -> bool:
    """Guard (True => do not mask): a digit run inside a long hex string
    (commit hashes, asset fingerprints: ``…2312253c<digits>a0f385…``). Checksums
    (INN/account) match random digits ~1/10 of the time — on large corpora
    hex blobs inevitably give "valid" ids (caught by the batch gate: an INN
    inside a gitlab hash, a corrupted value). Criterion: >=2 hex chars RIGHT
    on BOTH sides (a lone hex letter on the left — "id7707…" — does not silence
    a legitimate id)."""
    s, i, e = m.string, m.start(1), m.end(1)
    left = s[max(0, i - 2):i]
    right = s[e:e + 2]
    return (len(left) == 2 and all(c in _HEX for c in left)
            and len(right) == 2 and all(c in _HEX for c in right))


def _inside_id_run(m) -> bool:
    """INN guard: True => do not mask. The existing hex criterion
    (>=2 hex on both sides) PLUS: the digits are the tail of a base36/id token if
    RIGHT on the left there is a continuous ascii-alnum run that ITSELF contains a digit
    (ysclid=m1apnldzf<10 digits> — a click id, not an INN; it gave the structural glue
    m1apnldzfINN_REDACTED). A short purely alphabetic label ('id7707…',
    'payment id…') has no digit in the prefix → masked as before."""
    if _inside_hex_run(m):
        return True
    s, i, e = m.string, m.start(1), m.end(1)
    # (a) the digits are RIGHT surrounded by ascii-alnum on BOTH sides → this is
    # the inside of a structural alnum token (patent AU2011123456B2, SKU,
    # serial number), not a bare INN. A one-sided ascii label on the left
    # ('id7707…') is not silenced — there the digits are the id (clause b + test).
    if (i > 0 and s[i - 1].isascii() and s[i - 1].isalnum()
            and e < len(s) and s[e].isascii() and s[e].isalnum()):
        return True
    # (b) RIGHT on the left an ascii-alnum run that ITSELF contains a digit
    # (ysclid=m1apnldzf<digits> — a click id, not an INN):
    j = i
    while j > 0 and s[j - 1].isascii() and s[j - 1].isalnum():
        j -= 1
    return any(c.isdigit() for c in s[j:i])


def _hex_run_group(m, g: int) -> bool:
    """The same hex guard for rules where the digits are not group(1) (label ids)."""
    s, i, e = m.string, m.start(g), m.end(g)
    left = s[max(0, i - 2):i]
    right = s[e:e + 2]
    return (len(left) == 2 and all(c in _HEX for c in left)
            and len(right) == 2 and all(c in _HEX for c in right))


def _inside_hex_run2(m) -> bool:
    return _hex_run_group(m, 2)


def _inside_hex_run3(m) -> bool:
    return _hex_run_group(m, 3)


# --- A BARE NUMBER WITHOUT AN ID WORD -----------------------------------------
# The INN/OGRN checksum matches ~10% of random numbers, so a bare number
# of 10/13 digits in the unix-epoch range (sec 1.0e9…2.1e9 ≈ 2001…2036, ms
# 1.0e12…2.1e12: Slack ts, Telegram date_unixtime, Jira/Notion/Firebase, logs
# {"ts": 1616940109}) is a "valid INN" in every tenth case: a JSON number
# became {"ts": INN_REDACTED} — broken JSON → structural BLOCK of the whole
# round; a string ts was lost silently (thread linking, a false inn counter).
# Rule: such a number is an INN/OGRN only with an id word nearby. For a JSON NUMBER
# (``"key": 1700000008,`` — unquoted) the word is looked up in the field KEY, for
# other positions — in a window of ±_CTX_WIN chars WITHIN THE LINE (the stream branch
# sees one line at a time; a window across "\n" would diverge from the in-memory
# byte oracle). Words — only requisite labels: prose words
# «налог/расчёт/реквизиты» next to a ts in a chat would bring back FPs. The mask in
# a JSON number position is quoted (like discord_ids._mask_field): the JSON
# stays valid. ANON_ID_CTX_GUARD=0 — turn off (the old behaviour).
_ID_CTX_GUARD_ON = os.environ.get("ANON_ID_CTX_GUARD", "1") != "0"
_CTX_WIN = 40
_ID_CTX = re.compile(
    r"(?i)(?<![a-zа-яё])(?:инн|огрн(?:ип)?|кпп|бик|inn|ogrn|kpp|bik|tin|vat|"
    r"taxpayer|tax[ _-]?id|р/с|к/с)(?![a-zа-яё])"
)
_EPOCH = {10: (1_000_000_000, 2_100_000_000),
          13: (1_000_000_000_000, 2_100_000_000_000)}
# ``"key": <number>`` / ``"key": "<number>"`` — the field key on the left (no escapes:
# JSON inside a JSON string does not get here — the mask's quotes would break
# the OUTER document), on the right — the end of the value (, } ] or end of text) or
# a closing quote of the same form.
_JSON_KEY_L = re.compile(r"""(["'])([^"'\\\n]{1,64})\1\s*:\s*(["']?)$""")
_JSON_VAL_R = re.compile(r"^\s*(?:[,}\]]|$)")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
# A negative key (Presidio research): a time field is not an INN/OGRN even with
# a checksum and even outside the epoch range ("never expires" 4102444806, a string
# thread_ts). The key is matched whole, camelCase is split (createdAt →
# created_At).
_TIME_KEY = re.compile(
    r"(?i)(?:\w+_)?(?:ts|time|timestamp|created|updated|edited|deleted|"
    r"expires|exp|iat|nbf|date|unixtime)|\w+_(?:at|on)")


def _epoch_like(v: str) -> bool:
    """10 digits — unix seconds 2001…2036, 13 — the same in milliseconds."""
    rng = _EPOCH.get(len(v))
    return bool(rng) and rng[0] <= int(v) < rng[1]


def _json_key(m):
    """(key, is number?) for the value (group 1) in a JSON field position:
    ``"ts": 1700000008,`` → ("ts", True); ``"ts": "1700000008"`` →
    ("ts", False); otherwise (None, False). camelCase is split with ``_``."""
    s, i, e = m.string, m.start(1), m.end(1)
    k = _JSON_KEY_L.search(s[max(0, i - 96):i])
    if not k:
        return None, False
    key, q = _CAMEL.sub("_", k.group(2)), k.group(3)
    if q:
        return (key, False) if s[e:e + 1] == q else (None, False)
    return (key, True) if _JSON_VAL_R.match(s[e:e + 16]) else (None, False)


def _ctx_near(m, ctx: re.Pattern, window: int = _CTX_WIN) -> bool:
    """ctx in a ±window around the value (group 1), without crossing the line
    boundaries. engine.context.has_context looks only to the left — in tables and
    JSON the label may be on the right too («1700000008 (ИНН)», ``"…": "ИНН"``)."""
    s, i, e = m.string, m.start(1), m.end(1)
    left = s[max(0, i - window):i]
    left = left[left.rfind("\n") + 1:]
    right = s[e:e + window].split("\n", 1)[0]
    return bool(ctx.search(left) or ctx.search(right))


# An INN of legal entities of regions 10–20 (1700000008) looks like unix time, and the id word
# was required on the same line — the «ИНН» column of a counterparty table and the JSON key "tax_number"
# left it open. Now the id context is also a CSV/TSV column header (value pass
# id_cell_mask, pipeline._apply_id_columns) and a JSON key with a part inn/ogrn/kpp/tax/tin/vat
# (camelCase is split: taxNumber → tax_Number). Prose without a label — as before.
_ID_KEY = re.compile(r"(?i)(?:^|_)(?:inn|ogrn(?:ip)?|kpp|tax(?:payer)?|tin|vat)(?=_|$)")
_ID_CELL = re.compile(r"(\s*)(\d{10}|\d{12}|\d{13}|\d{15})(\s*)")


def id_header(header) -> bool:
    """A CSV/TSV column header is an INN/OGRN id context (ИНН, INN, ОГРН, Tax ID, taxNumber)."""
    if not isinstance(header, str) or not header.strip():
        return False
    return bool(_ID_CTX.search(header) or _ID_KEY.search(_CAMEL.sub("_", header.strip())))


def id_cell_mask(cell: str) -> str:
    """A cell of a column with an id header: INN/OGRN by checksum → token (surrounding spaces intact)."""
    m = _ID_CELL.fullmatch(cell)
    if not m:
        return cell
    v = m.group(2)
    if len(v) in (10, 12) and V.inn_ok(v):
        tok = P.INN
    elif len(v) in (13, 15) and V.ogrn_ok(v):
        tok = P.OGRN
    else:
        return cell
    return m.group(1) + tok + m.group(3)


def _needs_id_ctx(m) -> bool:
    """Guard (True => do not mask): a time field (JSON key) is not an INN/OGRN;
    the checksum is weak evidence (a JSON number or an epoch-like number),
    and there is no id word nearby (for a JSON number — in the key: orgInn → org_Inn)."""
    if not _ID_CTX_GUARD_ON:
        return False
    key, is_num = _json_key(m)
    if key is not None:
        if _TIME_KEY.fullmatch(key):
            return True
        if _ID_KEY.search(key):
            return False                 # tax_number / orgInn / ogrn_ip
        if is_num:
            return not _ID_CTX.search(key)
    return _epoch_like(m.group(1)) and not _ctx_near(m, _ID_CTX)


def _num_mask(m, token: str) -> str:
    """A placeholder instead of the number; in a JSON number position — quoted
    ({"inn": 7707083893} → {"inn": "INN_REDACTED"}), otherwise the JSON breaks."""
    return f'"{token}"' if _json_key(m)[1] else token


def _inn_guard(m) -> bool:
    return _inside_id_run(m) or _needs_id_ctx(m)


def _ogrn_guard(m) -> bool:
    return _inside_hex_run(m) or _needs_id_ctx(m)


# --- KPP WITHOUT A LABEL -------------------------------------------------------
# A KPP has no checksum, the structure (V.kpp_struct) passes ~94 % of random
# 9-digit numbers, so a bare number is a KPP only with a valid structure AND
# an INN nearby on THE SAME line (the stream sees one line at a time — like _ctx_near):
# an already minted INN_REDACTED (inn/inn_labeled stand earlier in RULES) or
# a still live 10/12-digit INN with a checksum (e.g. under the JSON time guard) —
# «ИНН/КПП 7707083893/770701001», «7707083893 / 770701001» in requisites;
# or a JSON field key with kpp/кпп («"kpp": 770701001» — the kpp rule does not see
# a label through a quote). The window is ±_CTX_WIN as for id words.
_KPP_KEY = re.compile(r"(?i)(?<![a-zа-яё])(?:кпп|kpp)(?![a-zа-яё])")
_INN_NEAR = re.compile(re.escape(P.INN) + r"|(?<![\d_])(\d{10}|\d{12})(?![\d_])")


def _inn_near(m) -> bool:
    """An INN (a token or a number with a checksum) in a ±_CTX_WIN window around group(1)
    within the line, or kpp/кпп in the JSON field key of the value."""
    key, _ = _json_key(m)
    if key is not None and _KPP_KEY.search(key):
        return True
    s, i, e = m.string, m.start(1), m.end(1)
    left = s[max(0, i - _CTX_WIN):i]
    left = left[left.rfind("\n") + 1:]
    right = s[e:e + _CTX_WIN].split("\n", 1)[0]
    for seg in (left, right):
        for mm in _INN_NEAR.finditer(seg):
            if mm.group(1) is None or V.inn_ok(mm.group(1)):
                return True
    return False


def _kpp_guard(m) -> bool:
    """Guard (True => do not mask): the inside of a hex/alnum token (as for INN)
    or no INN nearby."""
    return _inside_id_run(m) or not _inn_near(m)


# --- IBAN ----------------------------------------------------------------------
# ``[A-Z]{2}\d{2}`` + BBAN: a run of UPPER/digits with single spaces
# (paper format in groups of 4). The old ``(?:[ ]?[A-Z0-9]{2,6}){2,7}`` was
# ambiguous (how to cut the run into groups of 2–6) → catastrophic backtracking
# on long UPPER tokens (5000 SKUs — seconds). Here every char is
# determined unambiguously (a space — only before alnum), backtracking only on
# the counter → linear. The match tail may grab the next UPPER word
# («…013000 BIC»): the validator/mask check PREFIXES at spaces and mint
# only a valid one (before, the validator rejected the glued run whole — the IBAN leaked).
_IBAN_RE = re.compile(
    r"\b([A-Z]{2}\d{2}(?:[A-Z0-9]|[ ](?=[A-Z0-9])){10,43}[A-Z0-9])\b")


def _iban_prefix(v: str) -> int:
    """Length of the longest valid IBAN prefix of ``v`` (0 — no valid
    one). "Compression" (Presidio research): the country is known (V.IBAN_LEN) →
    the length is fixed, there is one candidate — the first n alnum chars, wherever
    the spaces are (a tail «…0130 002024» / a second IBAN right after cannot be separated
    by groups; a mod-97 accidentally matching on an extra group «… 00 20» no longer
    lengthens the mask). No match / country not in the table → by groups from the end
    (space boundaries: tail «… 00 2024», the next UPPER word «BIC»)."""
    n = V.IBAN_LEN.get(v[:2], 0)
    if n:
        cnt = 0
        for i, ch in enumerate(v):
            if ch != " ":
                cnt += 1
                if cnt == n:
                    if V.iban_ok(v[:i + 1]):
                        return i + 1
                    break
    parts = v.split(" ")
    for k in range(len(parts), 0, -1):
        cand = " ".join(parts[:k])
        if V.iban_ok(cand):
            return len(cand)
    return 0


def _iban_mask(m) -> str:
    """A valid prefix → placeholder; the tail («BIC …», a second IBAN in a row)
    goes out as is, but itself passes this same regex once more — otherwise an IBAN
    that got into the tail of another match would leak (sub continues AFTER the match).
    A tail glued WITHOUT a space («…002024», a second IBAN right after) goes
    under the placeholder together with the IBAN: «IBAN_REDACTED2024» — token welding
    (GLUE_RE, structural gate), and the tail itself is not PII (or is also an IBAN).
    The nested replacement is not counted in the Report (the example shows the whole match)."""
    v = m.group(1)
    tail = v[_iban_prefix(v):]
    if tail and tail[0] != " ":
        return P.IBAN
    return P.IBAN + _IBAN_RE.sub(
        lambda mm: _iban_mask(mm) if _iban_prefix(mm.group(1)) else mm.group(0),
        tail)


# --- DOCUMENT REQUISITES (audit escalations) -------------------------------------
# What survived the round and why:
#  * «Паспорт серия NNN номер NNNNNN» — a 3-digit series (typo/truncation):
#    the passport rule expects exactly 4 series digits → a critical leak;
#  * «диплом бакалавра NNNNNN NNNNNNN» — the series and blank number without the words
#    «серия/номер», next to it «регистрационный номер NNN-NN-N/NN» — a compound
#    number, reg_number took only solid 6–10 digits;
#  * «Registration number: NNNNNNNN», «Taxpayer Identification Code: CZNNNNNNNN»
#    — English labels of foreign requisites were known to nobody;
#  * «Выдан DD.MM.YYYY» in a passport block — dob takes years 1935…2010, the issue date
#    of a recent passport is out of range.
# All rules are label-anchored (a label on the left = evidence): bare numbers are not
# touched — amounts, dates, task and order numbers without a document label stay.
# A strict passport context for the WEAK form (a series not of 4 digits): the words
# «серия/номер» of the general PASSPORT_CTX are not evidence here — they are in the match themselves.
_PASS_CTX_STRICT = re.compile(r"(?i)(паспорт|passport|удостоверени\w*\s+личности)")
# The "this is a document block" context for an issue date: passport/ID card/series/
# department code/issuing authority or an already placed PASSPORT_REDACTED
# (passport stands earlier and by this point has eaten «серия … номер …»).
_DOC_CTX = re.compile(
    r"(?i)(паспорт|passport|удостоверени|серия|код\s*подраздел|мвд|уфмс|"
    + re.escape(P.PASSPORT) + r")")
# «г.»/«года» welded to a date without a space («21.10.2016г.») go under the
# mask together with the date — otherwise DATE_REDACTED is welded to a letter (GLUE_RE,
# structural gate); the same trick as addresses._DOB_TAIL
_YEAR_TAIL = r"(?:(?:года?|г\.?)(?![А-Яа-яЁёA-Za-z0-9]))?"
# horizontal whitespace: new rules do not match across a line break —
# the stream branch sees one line at a time, in-memory must behave the same
_HS = r"[^\S\r\n]"
_DOC_LABEL = r"(?:диплом|аттестат|свидетельств|удостоверени|лицензи|сертификат)[а-яё]*"
_DOC_NEEDS = re.compile(r"(?i)диплом|аттестат|свидетельств|удостоверени|лицензи|сертификат")
# a series value: 1–4 chars (digits/CAPITALS) + an optional second group
# («77 АВ», «II-АК», «45 12»)
_SERIES_VAL = r"[0-9A-ZА-ЯЁ]{1,4}(?:[ \t-]?[0-9A-ZА-ЯЁ]{1,3})?"


# «Паспорт 4510 123456», «паспорт: 45 10 123456», «паспорт 4510123456»,
# «Паспорт РФ 4510 №123456» — the most common written form, while passport waited for the word «серия».
# The label is strict (паспорт / паспортные данные / passport, «РФ» is part of the label), without the general
# context with the word «номер»; the label stays, series and number — PASSPORT_REDACTED.
_PASS_LABEL = (r"(?:паспорт(?:а|е|у|ом)?(?:" + _HS + r"+данные)?|паспортные" + _HS + r"+данные"
               r"|passport(?:" + _HS + r"+(?:no\.?|number|#))?)"
               r"(?:" + _HS + r"+(?:гражданина" + _HS + r"+)?(?:РФ|России|RF))?")


# A match across a line break («Паспорт серия 45\n12 …», «СНИЛС 123 456 789
# 01\n12 марта»): the mask merged the lines — the safety net let it through with a WARN, the value stayed
# open. Now the mask keeps the line breaks of the match (rule.keep_line_breaks), the SNILS value
# under a label — within the line (snils_label_line).


def _keep_lines(token: str, m) -> str:
    return keep_line_breaks(token, m.group(0))


# Watchdogs (tests/test_gaps_detectors.py): "\s" in the VALUE of a rule
# (snils_fmt, snils_label; addresses: house, coords_dms, the _DOB_TAIL tail of dob/dob_iso/dob_month/
# dob_labeled) caught a line break — the safety net let a cross-line match through with a WARN
# MaskChangesLineCount, the value stayed open («СНИЛС 112-233-445\n95», «…1985г.\nр.»).
# Path (b) of the comment above the test: horizontal whitespace [^\S\r\n] instead of \s — the rule does not
# cross a line (the rest on the next line stays as is, without a WARN).
_VS = _HS


def _series_guard(m) -> bool:
    """Guard (True => do not mask): «серия» without signs of a document series —
    the value must contain a digit or be 2–4 CAPITAL letters
    («серия АВ № 123456»). «Серия игр номер 12345» / «серия про номер…» is
    ordinary prose."""
    s = m.group(2)
    return not (any(c.isdigit() for c in s)
                or re.fullmatch(r"[A-ZА-ЯЁ]{2,4}", s))


def _doc_seed(*parts) -> str:
    """DOC_ID seed of a requisite: no spaces/hyphens, upper case — one
    and the same "series+number" pair written differently gives one token."""
    return re.sub(r"[\s-]+", "", "".join(parts)).upper()


RULES: list[Rule] = [
    # --- self-validating (checksum) — global, no label needed ---
    # needs: a bare digit run of the required length (or its mandatory
    # part) is a necessary condition of the match; the pattern starts with a lookbehind,
    # the prefilter cannot derive a probe, and a needs regex with a class at the start
    # is scanned by sre faster than the rule itself and is not called when False.
    Rule("ogrn", re.compile(NUM_L + r"(\d{13}|\d{15})" + NUM_R),
         mask=lambda m: _num_mask(m, P.OGRN), value_group=1, validator=V.ogrn_ok,
         guard=_ogrn_guard, confidence="high",
         needs=re.compile(r"\d{13}")),
    Rule("inn", re.compile(NUM_L + r"(\d{10}|\d{12})" + NUM_R),
         mask=lambda m: _num_mask(m, P.INN), value_group=1, validator=V.inn_ok,
         guard=_inn_guard, confidence="high",
         needs=re.compile(r"\d{10}")),
    Rule("iban", _IBAN_RE, mask=_iban_mask, value_group=1,
         validator=lambda v: _iban_prefix(v) > 0, confidence="high",
         needs=re.compile(r"[A-Z]{2}\d{2}")),
    # --- label ids WITHOUT a checksum: non-Russian requisites (audit: an Armenian
    # «ИНН 01234567» of 8 digits and «рег. № 12345678» survived the release). A label
    # on the left = evidence; the checksummed inn stands EARLIER and eats
    # valid Russian INNs first. The left (?<!letter) cuts off «Финн 12345678»;
    # the Latin "INN" is deliberately not included (Inn = hotel, FP). ---
    Rule("inn_labeled",
         re.compile(r"(?i)(?<![А-Яа-яЁёA-Za-z])(ИНН\s*[:№]?\s*)(\d{8,12})"
                    + NUM_R),
         mask=lambda m: m.group(1) + P.INN, value_group=2,
         guard=_inside_hex_run2, confidence="context"),
    # the value — solid 6–10 digits OR a compound number of digit
    # groups joined by "-"/"/" («123-45-6/19» — a diploma/order reg. number); in
    # a compound one ≥5 digits in total (validator), so that «рег. № 12-3» does not go.
    # The compound alternative FIRST: otherwise «123456-78» would be taken as «123456»
    # and the tail «-78» would stick to the token.
    Rule("reg_number",
         re.compile(r"(?i)(?<![А-Яа-яЁёA-Za-z])"
                    r"(рег(?:истрационн\w*)?\.?\s*(?:№|номер))"
                    r"(\s*[:№]?\s*)(\d{1,6}(?:[-/]\d{1,6}){1,4}|\d{6,10})" + NUM_R),
         mask=lambda m: m.group(1) + m.group(2)
         + _mint.tag("DOC_ID", m.group(3)),
         value_group=3, guard=_inside_hex_run3, confidence="context",
         validator=lambda v: sum(c.isdigit() for c in v) >= 5),
    # an English label of a company/record registration number
    # («Registration number: 04567123», «Reg. No. AB1234567», «Company
    # registration no 12345678»). The label is mandatory; a letter prefix ≤3.
    Rule("reg_number_en",
         re.compile(r"(?i)(?<![a-z])((?:company" + _HS + r"+)?(?:registration|reg\.)"
                    + _HS + r"*(?:number|no\.?|№|#)" + _HS + r"*[:#]?" + _HS + r"*)"
                    r"([A-Z]{0,3}\d{6,12})"
                    + NUM_R),
         mask=lambda m: m.group(1) + _mint.tag("DOC_ID", m.group(2).upper()),
         value_group=2, guard=_inside_hex_run2, confidence="context",
         needs=re.compile(r"(?i)regist|reg\.")),
    # a foreign tax number under an English label («Taxpayer
    # Identification Code: CZ04567123», «VAT number DE123456789», «Tax ID:
    # 123456789», «TIN 1234567890»). The value — 8–12 digits, optionally with
    # a two-letter country code; «VAT 20%» and other prose do not fit.
    Rule("tax_id_en",
         re.compile(r"(?i)(?<![a-z])((?:taxpayer" + _HS + r"+identification" + _HS + r"+"
                    r"(?:code|number|no\.?)|tax(?:payer)?" + _HS + r"*id(?:entification)?"
                    r"(?:" + _HS + r"+(?:code|number|no\.?))?|vat(?:" + _HS
                    + r"+(?:reg(?:istration)?\.?" + _HS + r"+)?(?:number|no\.?|id|code))?"
                    r"|tin)" + _HS + r"*[:#]?" + _HS + r"*)"
                    r"([A-Z]{2}[ ]?\d{8,12}|\d{8,12})" + NUM_R),
         mask=lambda m: m.group(1) + P.INN, value_group=2,
         guard=_inside_hex_run2, confidence="context",
         needs=re.compile(r"(?i)tax|vat|tin")),
    # the series and blank number under a document label without the words «серия/номер»
    # («диплом бакалавра 107704 0012345», «свидетельство 77 АВ 1234567»):
    # 6+7 digits joined by a space or «NN ББ NNNNNN(N)». Up to 4 words between the label and
    # the value («диплом о высшем образовании …»).
    Rule("doc_blank_number",
         re.compile(r"(?i)(?<![а-яё])(" + _DOC_LABEL
                    + r"(?:[ \t]+[а-яё]+){0,4}?[ \t]*(?:№|N|номер)?[ \t]*[:№]?[ \t]*)"
                    r"(\d{6}[ \t]\d{7}|\d{2}[ \t]?[А-ЯЁA-Z]{2}[ \t]?\d{6,7})"
                    + NUM_R),
         mask=lambda m: m.group(1) + _mint.tag("DOC_ID", _doc_seed(m.group(2))),
         value_group=2, confidence="context", needs=_DOC_NEEDS),
    Rule("account20", re.compile(NUM_L + r"(\d{20})" + NUM_R),
         mask=lambda m: _num_mask(m, P.ACCOUNT), value_group=1,
         validator=V.account_struct, guard=_inside_hex_run, confidence="high",
         needs=re.compile(r"\d{20}")),
    Rule("card_grouped",
         re.compile(NUM_L + r"(\d{4}[ \-_]\d{4}[ \-_]\d{4}[ \-_]\d{4})" + NUM_R),
         mask=lambda m: P.CARD, value_group=1,
         validator=lambda v: V.luhn_ok(v), confidence="high",
         needs=re.compile(r"\d{4}[ \-_]\d{4}[ \-_]\d{4}[ \-_]\d{4}")),

    # --- label-anchored ids ---
    Rule("bik", re.compile(r"(?i)(?:БИК|BIK|BIC)\s*[:№]?\s*(04[45]\d{6})\b"),
         mask=lambda m: m.group(0).replace(m.group(1), P.BIK), value_group=1, confidence="context"),
    Rule("bik_ctx", re.compile(NUM_L + r"(04[45]\d{6})" + NUM_R),
         mask=lambda m: P.BIK, value_group=1, requires_context=BANK_CTX, ctx_right=40,
         confidence="context",    # «044525225 (БИК)» — the label on the right
         needs=re.compile(r"04[45]\d{6}")),
    Rule("kpp", re.compile(r"(?i)(?:КПП|KPP)\s*[:№]?\s*(\d{9})\b"),
         mask=lambda m: m.group(0).replace(m.group(1), P.KPP), value_group=1, confidence="context"),
    # KPP without a label — structure (region/reason/sequence) + an INN nearby
    # on the same line or a kpp key of the JSON field (see _inn_near); the mask in
    # a JSON number position is quoted, as for inn.
    Rule("kpp_near_inn", re.compile(NUM_L + r"(\d{9})" + NUM_R),
         mask=lambda m: _num_mask(m, P.KPP), value_group=1, validator=V.kpp_struct,
         guard=_kpp_guard, confidence="context",
         needs=re.compile(r"\d{9}")),
    Rule("okpo", re.compile(r"(?i)(?:ОКПО|OKPO)\s*[:№]?\s*(\d{8,10})\b"),
         mask=lambda m: m.group(0).replace(m.group(1), P.OKPO), value_group=1, confidence="context"),
    Rule("swift", re.compile(r"(?i)(?:SWIFT|BIC)\s*[:№]?\s*([A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b"),
         mask=lambda m: m.group(0).replace(m.group(1), P.SWIFT), value_group=1,
         requires_context=BANK_CTX, confidence="context"),
    Rule("account_label",
         re.compile(r"(?i)(?:р/с|расч[её]тн\w*\s+сч[её]т\w*|лицев\w*\s+сч[её]т\w*|account)\s*[:№]?\s*(\d(?:[ \-]?\d){11,24})"),
         mask=lambda m: m.group(0).replace(m.group(1), P.ACCOUNT), value_group=1, confidence="context"),

    # --- card / SNILS / passport (format + context) ---
    Rule("card_solid", re.compile(NUM_L + r"([2-6]\d{15})" + NUM_R),
         mask=lambda m: _num_mask(m, P.CARD), value_group=1,
         validator=lambda v: V.luhn_ok(v), confidence="context",
         needs=re.compile(r"[2-6]\d{15}")),
    Rule("snils_fmt", re.compile(NUM_L + r"(\d{3}-\d{3}-\d{3}(?:" + _VS + r"|-)\d{2})" + NUM_R),
         mask=lambda m: P.SNILS, value_group=1, validator=lambda v: V.snils_ok(v), confidence="high",
         needs=re.compile(r"\d{3}-\d{3}-\d{3}")),
    # the SNILS value — within the line (snils_label took «… 01\n12» and lost the line
    # break → the safety net let the match through, the number open); stands before snils_label
    Rule("snils_label_line",
         re.compile(r"(?i)СНИЛС\s*[:№]?\s*(\d(?:\d|[^\S\r\n]|[.-]){9,14}\d)"),
         mask=lambda m: m.group(0).replace(m.group(1), P.SNILS), value_group=1,
         guard=no_guard, confidence="context",
         needs=re.compile(r"(?i)снилс")),
    Rule("snils_label", re.compile(r"(?i)СНИЛС\s*[:№]?\s*(\d(?:\d|" + _VS + r"|[.-]){9,14}\d)"),
         mask=lambda m: m.group(0).replace(m.group(1), P.SNILS), value_group=1, confidence="context"),
    # JSON-safe: the gaps of the match (replaced whole) do not cross «"»/«\» of raw JSON
    # («"серия": "45 10", "номер": "123456"» — the quotes and the key went under the token, the line broke)
    Rule("passport", re.compile(r"(?i)(?:серия[^\d\n" + _JS + r"]{0,10}(\d{2}\s?\d{2})[^\d\n" + _JS
                                + r"]{0,20}(?:номер|№)?[^\d\n" + _JS + r"]{0,6}(\d{6}))"),
         mask=lambda m: _keep_lines(P.PASSPORT, m), value_group=0, requires_context=PASSPORT_CTX,
         confidence="context"),
    # a passport with a series NOT of 4 digits (2–3 — a typo/truncation in a form):
    # a weak form, so the word «номер/№» is mandatory, the number is exactly 6 digits,
    # and on the left — a strict passport context (паспорт/passport/удостоверение
    # личности), not the general PASSPORT_CTX. 4 digits are taken by passport above.
    Rule("passport_short_series",
         re.compile(r"(?i)серия[^\d\n" + _JS + r"]{0,10}(\d{2,3})[^\d\n" + _JS + r"]{0,6}(?:номер|№)"
                    r"[^\d\n" + _JS + r"]{0,6}(\d{6})(?!\d)"),
         mask=lambda m: P.PASSPORT, value_group=0,
         requires_context=_PASS_CTX_STRICT, confidence="context",
         needs=re.compile(r"(?i)серия")),
    # a passport label + 4 series digits (2+2) + №? + 6 number digits, without «серия». On the left —
    # also a JSONL escape (boundary._ESC: «\n», «\t», «\uXXXX» — the escape letter is not a neighbour;
    # «инфо\nПаспорт 4510 123456» in raw .jsonl stayed open)
    Rule("passport_labeled",
         re.compile(r"(?i)(?:" + _ESC + r"|(?<![а-яёa-z]))(" + _PASS_LABEL + _HS + r"*[:№#-]?"
                    + _HS + r"*)"
                    r"(\d{2}" + _HS + r"?\d{2}" + _HS + r"*(?:№|N|номер)?" + _HS + r"*\d{6})"
                    r"(?!\d)"),
         mask=lambda m: m.group(1) + P.PASSPORT, value_group=2,
         guard=no_guard, confidence="context",
         needs=re.compile(r"(?i)паспорт|passport")),
    # «серия X номер/№ Y» of any document (diploma, certificate,
    # licence, driving licence) outside a passport context → one DOC_ID per pair.
    # Both words are mandatory, the number 5–8 digits, the series — with a digit or 2–4
    # CAPITALS (_series_guard): «серия 2, номер 15» of a magazine/TV series and «серия
    # игр номер …» do not fit.
    Rule("doc_series_number",
         re.compile(r"(?<![А-Яа-яЁё])((?:[Сс]ерия|СЕРИЯ)[ \t]*[:№]?[ \t]*)"
                    r"(" + _SERIES_VAL + r")"
                    r"([ \t]*,?[ \t]*(?:[Нн]омер|НОМЕР|№|N)[ \t]*[:№]?[ \t]*)"
                    r"(\d{5,8})(?!\d)"),
         mask=lambda m: m.group(1)
         + _mint.tag("DOC_ID", _doc_seed(m.group(2), m.group(4))),
         value_group=4, guard=_series_guard, confidence="context",
         needs=re.compile(r"[Сс]ерия|СЕРИЯ")),
    # V.dept_ok used to reject all real codes (read 3 digits as
    # a region ≤ 92) — the rule never fired; see validators.dept_ok.
    Rule("dept_code", re.compile(r"(?i)(код\s*подраздел\w*\s*[:№]?\s*)(\d{3})\s?-?\s?(\d{3})"),
         mask=lambda m: m.group(1) + P.DEPTCODE, value_group=2,
         guard=lambda m: not V.dept_ok(m.group(2), m.group(3)), confidence="context"),
    # a document issue date («Выдан 05.06.2020», «Дата выдачи: …») in
    # a document block (_DOC_CTX on the left). dob (addresses) takes «выдан» only
    # for 1935…2010 — recent passports kept their date. Years 1980…2039.
    Rule("doc_issue_date",
         re.compile(r"(?i)(?<![а-яё])((?:выдан[аоы]?|дата" + _HS + r"+выдачи)"
                    + _HS + r"*[:\-–—]?" + _HS + r"*)"
                    r"((?:0[1-9]|[12]\d|3[01])[.\-/](?:0[1-9]|1[0-2])[.\-/]"
                    r"(?:19[89]\d|20[0-3]\d)" + _YEAR_TAIL + r")(?!\d)"),
         mask=lambda m: m.group(1) + P.DATE, value_group=2,
         requires_context=_DOC_CTX, window=160, confidence="context",
         needs=re.compile(r"(?i)выдан|дата" + _HS + r"+выдачи")),
    # (Telegram re-run audit: a cadastral number «NN:NN:NNNNNNN:NNNN» of a real estate object in
    # a message — DOC_ID): the format «district:area:block:object» is recognised by itself — times
    # (groups of 2 digits) and IPv6 (hex groups) do not look like this
    Rule("cadastral_number",
         re.compile(r"(?<![\d:.])(\d{2}:\d{2}:\d{6,7}:\d{1,6})(?![\d:])"),
         mask=lambda m: _mint.tag("DOC_ID", m.group(1)), value_group=1, confidence="context",
         needs=re.compile(r"\d{2}:\d{2}:\d{6}")),
    Rule("aws_account", re.compile(r"(?i)(?:AWS|Account\s*ID)[^\d\n]{0,20}(\d{12})\b"),
         mask=lambda m: m.group(0).replace(m.group(1), P.AWS_ACCOUNT), value_group=1, confidence="context"),
]
