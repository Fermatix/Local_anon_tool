"""DECODE-BEFORE-SCAN — catch secrets/PII smuggled through encoding.

A password, API key or email can slip past every plaintext rule by riding
inside a base64 / base64url / hex / percent-encoded blob — or by being spelled
out under a *transport* encoding (punycode ``xn--`` in a domain) or split apart
by *invisible* characters (soft-hyphen U+00AD, zero-width space/joiner). This
module finds each such carrier, DECODES / REASSEMBLES it, and tests the decoded
text for a real secret/PII signal (entropy-credential, email, `password=…`, a
vendor key signature, a hidden Cyrillic name/brand). Only if the decode yields
something genuinely sensitive does it replace the encoded blob with ``P.ENCODED``.

Carriers handled here:
  * base64 / base64url  — MIME + url-safe alphabets;
  * hex                 — even-length lowercase runs (>=12 bytes);
  * percent-encoding    — ``%XX`` escapes (>=3), possibly interleaved with text;
  * punycode (``xn--``) — an IDNA/ACE host label that decodes to a real
    (Cyrillic) name/brand or carries a PII signal — a `.рф`/`.укр` smuggle;
  * soft-hyphen / zero-width reassembly — a run broken up by U+00AD / U+200B /
    U+200C / U+200D / U+FEFF that, once the invisibles are stripped, becomes one
    of the carriers above OR a plaintext secret label/value.

FP discipline is the whole game here: base64 is also JS sourcemaps, inlined
images, fonts, CSS data-URIs, protobuf, and gzip; ``xn--`` is also innocuous
transliteration; soft-hyphens are legitimate line-break hints. Masking any of
these blindly would shred ordinary documents. So every rule:
  1. only matches a carrier whose DECODE is *mostly printable text* (binary =>
     skip: image/font/gzip, not a smuggled credential), and
  2. only fires when that decoded text carries a secret/PII signal.
Both checks live in the rule ``guard`` (``True`` => leave untouched), so the
expensive decode runs only on the handful of candidate blobs the regex finds.

confidence='context' — the signal is the decoded payload, not the blob shape.
Runs AFTER detectors/secrets (so plaintext signatures/labels are already gone)
and BEFORE detectors/financial.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import re
from urllib.parse import unquote_to_bytes

from ..engine import ac
from ..engine import placeholders as P
from ..engine.boundary import L
from ..engine.rule import Rule
from ..engine.secret import is_secret_value

# ---------------------------------------------------------------------------
# decoded-text signal detection
# ---------------------------------------------------------------------------

# A vendor-key / bearer-token signature *inside* the decoded text — the same
# self-identifying prefixes secrets.py matches in plaintext. If the decode
# reveals one of these, it's unambiguously a smuggled credential.
_VENDOR_SIG = re.compile(
    r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\."   # jwt
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}"                       # aws
    r"|AIza[\w-]{30,}"                                  # google api
    r"|gh[pousr]_[A-Za-z0-9]{30,}"                      # github pat
    r"|xox[abprs]-[\w-]{10,}"                           # slack
    r"|SG\.[\w-]{16,}\.[\w-]{16,}"                       # sendgrid
    r"|(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"    # stripe
    r"|sk-(?:proj-)?[A-Za-z0-9_-]{20,}"                # openai
    r"|hf_[A-Za-z0-9]{20,}"                             # huggingface
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"             # pem
)

# A secret-ish label followed by a value: `password=…`, `api-key: …`, `токен …`.
_LABEL_VALUE = re.compile(
    r"(?i)(?:парол\w*|password|passwd|pwd|passcode|секрет|secret|token|токен|"
    r"api[_-]?key|access[_-]?key|accesskey|client[_-]?secret|ключ|"
    r"bearer|authorization|credential)"
    r"\s*[:=]\s*\S{4,}"
)

# An email address anywhere in the decoded text.
# PERF: without a left boundary and limits, `[A-Za-z0-9][\w.+-]{1,}@` was O(n²)
# on a long \w run before "@" (40 KB → 3.9 s, 1 MB ≈ 40 min for ONE blob):
# every position of the run reopened a greedy scan to its end. The lookbehind
# lets a match start only at the beginning of an ASCII run (O(1) per position), the
# RFC limits 64/255 bound the scan at the start by a constant. Same semantics: the run
# before "@" contains an alphanumeric not in the last place (`.john@` / `_john@` — as
# before, the prefix [._+-]* without nested ambiguity); the lookbehind class is
# ASCII, so Cyrillic right on the left («почтаjohn@…») does not hide the address.
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9_.+-])[._+-]{0,64}[A-Za-z0-9][\w.+-]{1,64}"
    r"@[A-Za-z0-9][\w.-]{0,253}\.[A-Za-z]{2,}"
)

# A run of >=3 Cyrillic letters — a real Russian word (name / brand / place),
# the thing a punycode `.рф` host or a percent-encoded label is smuggling. A lone
# accented latin char (xn--bda -> "ê") or a 1-2 letter fragment is NOT a signal,
# so benign transliteration IDNs stay untouched.
_CYRILLIC_WORD = re.compile(r"[А-Яа-яЁё]{3,}")

# Invisible characters used to SPLIT a token so it hides from a plaintext scan:
# soft-hyphen, zero-width space/non-joiner/joiner, word-joiner, BOM/ZWNBSP.
_INVISIBLE = "­​‌‍⁠﻿"
_INVISIBLE_RE = re.compile("[" + _INVISIBLE + "]")
# a visible character of an invisible-split run (the run class minus the invisibles)
_VISIBLE = r"[^\s\"'<>)\]}" + _INVISIBLE + r"]"


def _printable_ratio(bs: bytes) -> float:
    """Fraction of bytes that are plain printable ASCII (letters/digits/punct/
    common whitespace). Low ratio => binary payload (image/font/gzip)."""
    if not bs:
        return 0.0
    ok = sum(1 for b in bs if 0x20 <= b <= 0x7E or b in (0x09, 0x0A, 0x0D))
    return ok / len(bs)


_TARGETS: frozenset = frozenset()
_TARGET_AC = None       # Aho-Corasick over the targets (casefold) — per-process cache


def set_targets(literals) -> None:
    """Dynamic targets of the RUN (mask literals / control forms of the pack): a decode
    containing a target = smuggling a known entity through an encoding
    (critical: base64 → acme.example, percent-encoding → «Ромашка» bypassed
    the static PII signal — the layer did not know the brands of a given run).
    casefold substrings; length >= 4 (shorter is noise).

    PERF: checking `t in low` for EVERY target on EVERY decode is
    O(targets × blob): 12k targets → 25 s on 10k small base64 lines of one
    mail (.eml/.mbox — lines of 76 chars), 3.7 s per 1 MB of decode, × 3
    fixpoint passes → hours of the stage. The automaton is built once per
    set_targets and gives one pass over the decode (0.03 s / 0.05 s); the semantics
    are the same — a casefold substring (the automaton folds targets and text itself)."""
    global _TARGETS, _TARGET_AC
    lits = [t for t in literals if isinstance(t, str) and len(t.strip()) >= 4]
    _TARGETS = frozenset(t.casefold() for t in lits)
    _TARGET_AC = ac.Automaton(lits, casefold=True) if lits else None


def _decoded_is_sensitive(text: str) -> bool:
    """True if a decoded string carries a real secret/PII signal.

    A single positive signal is enough. Order cheapest-first."""
    if not text:
        return False
    if _TARGET_AC is not None:
        # the first hit stops (a presence gate, positions are not needed)
        if next(_TARGET_AC.iter(text), None) is not None:
            return True
    if _VENDOR_SIG.search(text):
        return True
    # _LABEL_VALUE requires `:`/`=` — without them the (?i) alternation over
    # 1 MB of prose (~0.1 s) is not needed
    if (":" in text or "=" in text) and _LABEL_VALUE.search(text):
        return True
    if "@" in text and _EMAIL.search(text):
        return True
    # a whole decoded value that is itself a credential (high-entropy key/pw).
    # `is_secret_value` judges a single spaceless value, so test the raw decode
    # AND its whitespace-split tokens (covers `password=Secret123` -> token).
    if is_secret_value(text):
        return True
    for tok in re.split(r"[\s,;<>|]+", text):
        if len(tok) < 8:                 # is_secret_value: shorter than 8 — never
            continue
        # the value half of a `k=v` / `k:v` pair is the interesting part
        for part in re.split(r"[:=]", tok, maxsplit=1)[-1:]:
            if is_secret_value(part):
                return True
    return False


_FIRST_NAMES_LC: frozenset | None = None


def _decoded_has_trace(text: str) -> bool:
    """Geo in URL-encoded cookies/links (a city / country / region + a postal code,
    CITY/COUNTRY major): a percent decode containing a Russian trace from the
    known_values/ru_trace dictionary (city/region/country/brand/government body) or a name
    from the core of the name gazetteer is PII smuggled through an encoding. Cyrillic by
    itself is NOT a signal (Wikipedia titles, search phrases without names/geo stay as
    they were): only the dictionary and names."""
    words = _CYRILLIC_WORD.findall(text)
    if not words:
        return False
    global _FIRST_NAMES_LC
    if _FIRST_NAMES_LC is None:
        from ..engine.gazetteer import RU_FIRST_NAMES_CORE
        _FIRST_NAMES_LC = frozenset(n.casefold() for n in RU_FIRST_NAMES_CORE)
    if any(w.casefold() in _FIRST_NAMES_LC for w in words):
        return True
    from ..known_values import ru_trace as _rt
    if not _rt.enabled():
        return False
    toks = re.findall(r"[А-Яа-яЁё][А-Яа-яЁё-]*", text)
    for n in (3, 2, 1):                       # multi-word dictionary entities before single ones
        for i in range(len(toks) - n + 1):
            if _rt.kind(" ".join(toks[i:i + n])):
                return True
    return False


def _pct_guard(m) -> bool:
    """Guard of a percent carrier (True => leave alone): the decode is printable AND carries a
    secret/PII (_decoded_is_sensitive) or a Russian trace/name (_decoded_has_trace) — else the blob stays."""
    d = _decode_percent(m.group(1))
    return not (d and (_decoded_is_sensitive(d) or _decoded_has_trace(d)))


# Guard (tests/test_gaps_detectors.py): «почта ivan.petrov@xn--e1afmkfd.xn--p1ai»
# — the encoding detector runs before contacts, encoded_punycode took the host → "ivan.petrov@
# ENCODED_REDACTED", the local part open, and contacts.email_wide no longer saw the whole
# address with an IDN zone. A punycode host right after "@" with a local part on the left is
# email_wide's job (EMAIL_REDACTED whole); without email_wide (contacts detector off) the host
# is masked here. Yielding on the single sign "[\w.+-]@" on the left left the host OPEN where
# email_wide is silent (a local part without a letter — its guard; "XN--" in upper case — the
# pattern has no re.I; a local part longer than 64 — the cap; "x@y@host" — does not start after
# "@"; the contacts detector is off) — with knob 0 the host was under ENCODED. Yield only after
# the rule itself matches (guard test test_email_punycode_host_stays_masked).
_EMAIL_LOCAL_RUN = re.compile(r"[\w.+-]+@$")     # email_wide local part ≤ 64 + "@"
EMAIL_WIDE_RULE_ACTIVE = True   # Pipeline: False when the contacts detector is off (email_wide does not run)


def _punycode_host_of_email(m) -> bool:
    """True => the punycode host is part of a mail address that contacts.email_wide takes
    whole: its match from the start of the local part (its finditer starts there too: on the
    left is not an address character) ends exactly at this host, the guard passes. Otherwise the host is masked here."""
    if not EMAIL_WIDE_RULE_ACTIVE:
        return False
    s, at = m.string, m.start(1)
    run = _EMAIL_LOCAL_RUN.search(s, max(0, at - 66), at)
    if run is None or (run.start() > 0 and s[run.start() - 1] == "@"):
        return False
    from . import contacts as _ct
    em = _ct._EMAIL_WIDE.match(s, run.start())
    return em is not None and em.end(1) == m.end(1) and not _ct._email_wide_guard(em)


def _decoded_carries_name(text: str) -> bool:
    """As ``_decoded_is_sensitive`` but ALSO true when the decode reveals a real
    Cyrillic name / brand / place word. Used for the *transport* carriers
    (punycode, invisible-split reassembly) where a hidden Russian entity — not a
    high-entropy credential — is the thing being smuggled through the encoding."""
    if _decoded_is_sensitive(text):
        return True
    return bool(_CYRILLIC_WORD.search(text))


# ---------------------------------------------------------------------------
# per-encoding decoders — each returns the decoded UTF-8 text, or "" if the
# blob does not cleanly decode to mostly-printable text (=> not a candidate).
# ---------------------------------------------------------------------------

def _try_bytes(bs: bytes | None) -> str:
    """Common tail: require the bytes exist, decode as UTF-8 (latin-1
    fallback), and be mostly printable TEXT. Empty string => reject.

    Printability is measured on the DECODED text, not on the raw bytes:
    utf-8 Cyrillic (0xD0/0xD1) is non-ASCII bytes, and the byte metric cut
    a Russian decode as "binary" — percent-encoded «Ромашка» / base64 with Russian
    context were not decoded at all, the smuggling channel was open."""
    if not bs or len(bs) < 6:
        return ""
    try:
        text = bs.decode("utf-8")
    except UnicodeDecodeError:
        # not utf8: the old byte threshold (latin-1 decodes everything — without
        # the printable gate any binary became "text")
        if _printable_ratio(bs) < 0.85:
            return ""
        try:
            text = bs.decode("latin-1")
        except UnicodeDecodeError:
            return ""
    ok = sum(1 for ch in text if ch.isprintable() or ch in "\t\n\r")
    if ok / len(text) < 0.85:
        return ""
    return text


def _decode_b64(blob: str) -> str:
    # standard base64. Strip any padding the blob CARRIES first, then re-pad
    # from the body length — naively padding the already-padded string appends
    # extra '=' and validate=True rejects a perfectly good blob.
    s = blob.rstrip("=")
    pad = (-len(s)) % 4
    if pad == 3:                      # no valid base64 body has len%4 == 1
        return ""
    try:
        bs = base64.b64decode(s + "=" * pad, validate=True)
    except (binascii.Error, ValueError):
        return ""
    return _try_bytes(bs)


def _decode_b64url(blob: str) -> str:
    s = blob.rstrip("=")              # same re-padding discipline as _decode_b64
    pad = (-len(s)) % 4
    if pad == 3:
        return ""
    try:
        bs = base64.urlsafe_b64decode(s + "=" * pad)
    except (binascii.Error, ValueError):
        return ""
    return _try_bytes(bs)


def _decode_hex(blob: str) -> str:
    try:
        bs = bytes.fromhex(blob)
    except ValueError:
        return ""
    return _try_bytes(bs)


def _decode_percent(blob: str) -> str:
    try:
        bs = unquote_to_bytes(blob)
    except Exception:  # noqa: BLE001 - unquote is defensive, never crash a pass
        return ""
    return _try_bytes(bs)


def _decode_punycode(host: str) -> str:
    """Decode an ACE/IDNA host (one or more ``xn--`` labels) to its Unicode form.

    Label-wise so we don't depend on the whole host being valid IDNA; a label
    that starts ``xn--`` but does not decode aborts the host (=> ""). Non-ACE
    labels (a plain ``.ru`` TLD next to an ``xn--`` label) pass through. Returns
    "" when there is no ``xn--`` label at all (nothing was actually encoded)."""
    parts = host.split(".")
    if not any(p.lower().startswith("xn--") and len(p) > 4 for p in parts):
        return ""
    out = []
    for p in parts:
        if p.lower().startswith("xn--") and len(p) > 4:
            try:
                out.append(codecs.decode(p[4:].encode("ascii"), "punycode"))
            except (UnicodeError, ValueError):
                return ""
        else:
            out.append(p)
    return ".".join(out)


def _reassemble(blob: str) -> str:
    """Strip the invisible splitters from a run and RE-TEST the reassembled text
    as any of the plaintext carriers or a bare secret. The value returned is the
    reassembled text ONLY when it now reads as a smuggled secret/PII — else "".

    This catches ``pass­word=Secret1`` (a label/value fractured by a
    soft-hyphen) and ``АБВ­Г0000…`` (an id split so it dodges a literal
    scan): the invisibles are a line-break/obfuscation hint with no semantic
    meaning, so removing them is loss-free."""
    joined = _INVISIBLE_RE.sub("", blob)
    if joined == blob or len(joined) < 6:
        return ""
    # a reassembled base64/hex/percent carrier -> decode that and test it too
    for dec in (_decode_b64, _decode_b64url, _decode_hex, _decode_percent):
        d = dec(joined)
        if d and _decoded_is_sensitive(d):
            return joined
    # else the reassembled text itself may be a plaintext label/value/secret —
    # OR a hidden Cyrillic name/brand/id («АБВ\xadГ0000…»), exactly what the
    # module/function docstrings promise. invisible-split is a TRANSPORT carrier
    # (like punycode), so the re-test goes through _decoded_carries_name, not
    # _decoded_is_sensitive (that one rejects Cyrillic, secret.py) — without this
    # the documented Cyrillic smuggling channel was open (0 replacements).
    if _decoded_carries_name(joined):
        return joined
    return ""


# base64/hex of «Иван Петров», «+7 900 000-00-00» or JSON with them went out
# open — _decoded_is_sensitive looks for a secret, an email and a label, but not a phone or a
# full name (JSON with one phone was masked by accident, by entropy). Now in a base64/base64url/hex
# decode the signal is also a Russian phone (phone_ru form + V.phone_ru_ok) and a "First Last" /
# "Last First" pair (a first name from the gazetteer core + a surname ending): a single
# «надежда», «роман» (bare _decoded_has_trace) is not a signal.
_DEC_PHONE = re.compile(r"(?<![\d])((?:\+?7|8)[\s\-()]{0,3}\d{3}[\s\-()]{0,3}\d{3}[\s\-()]{0,3}"
                        r"\d{2}[\s\-()]{0,3}\d{2})(?!\d)")
_DEC_PAIR = re.compile(r"(?<![А-Яа-яЁё])([А-ЯЁ][а-яё]{1,30})[ \u00a0]([А-ЯЁ][а-яё]{1,30})(?![А-Яа-яЁё])")
_DEC_SURN = re.compile(r"(?:ов|ев|ёв|ин|ын|ский|цкий|ской|цкой|енко|ук|юк|чук|ян|дзе|швили)"
                       r"(?:а|у|е|ы|ой|ым|ом|им|ого|ому|)$")


def _decoded_has_pii(text: str) -> bool:
    """A Russian phone or a "First Last" pair (in any order) in the decode."""
    from ..engine import validators as _V
    for m in _DEC_PHONE.finditer(text):
        if _V.phone_ru_ok(m.group(1)):
            return True
    global _FIRST_NAMES_LC
    if _FIRST_NAMES_LC is None:
        from ..engine.gazetteer import RU_FIRST_NAMES_CORE
        _FIRST_NAMES_LC = frozenset(n.casefold() for n in RU_FIRST_NAMES_CORE)
    for m in _DEC_PAIR.finditer(text):
        a, b = m.group(1), m.group(2)
        if (a.casefold() in _FIRST_NAMES_LC and _DEC_SURN.search(b)
                and b.casefold() not in _FIRST_NAMES_LC) \
                or (b.casefold() in _FIRST_NAMES_LC and _DEC_SURN.search(a)
                    and a.casefold() not in _FIRST_NAMES_LC):
            return True
    return False


def _guard(decoder):
    """Build a guard: True => leave untouched. We keep the blob unless it
    decodes AND the decode looks sensitive (decode failure => keep).
    Also a phone / "First Last" in the decode."""
    def guard(m):
        decoded = decoder(m.group(1))
        return not (decoded and (_decoded_is_sensitive(decoded)
                                 or _decoded_has_pii(decoded)))
    return guard


# ---------------------------------------------------------------------------
# rules — ordered longest/most-specific first. base64url before base64 so a
# blob containing `-`/`_` isn't half-eaten by the stricter base64 class; hex and
# percent are disjoint alphabets so their order relative to base64 is free.
# ---------------------------------------------------------------------------

def _eats_json_escape(m) -> bool:
    """Guard component (True => suppress): the match starts right after a
    backslash and starts with a JSON/string escape letter (``\\n``,
    ``\\t``…) — the mask would eat the pair's letter, leaving a lone ``\\`` =>
    broken JSON (caught by the structural gate: ``\\ENCODED_REDACTED``,
    a cascading shift of the file's lines)."""
    s, i = m.string, m.start(1)
    return i > 0 and s[i - 1] == "\\" and s[i] in "ntrbfu"


# Left boundary of a base64 carrier. The class used to contain "=" —
# ``(?<![A-Za-z0-9+/=_-])`` — and a blob in the position of a PARAMETER VALUE did not match
# at all: right to its left is "=" (``?eid=<base64>``, ``&token=<base64>``,
# ``key=<base64>``), and any later start is already inside the run. This is
# the most common form of smuggling: a Google Calendar link ``event?eid=…`` carries
# base64 of "<event id> <participant e-mail>" — critical, an employee e-mail
# in every meeting link (dozens of files). "=" on the left is either
# a key=value separator (the blob must be checked) or the padding of the PREVIOUS
# blob (no data follows padding; two glued blobs are rare, and the
# decode guard judges them anyway). The guard still decides: a decode that is not printable
# or has no PII/secret signal → the blob is left alone.
_B64_L = r"(?<![A-Za-z0-9+/_-])"

RULES: list[Rule] = [
    # base64url: distinguished by `-`/`_` (must contain at least one, else the
    # plain-base64 rule handles it). No `+`/`/`. Optional `=` padding.
    Rule("encoded_base64url",
         re.compile(_B64_L +
                    r"([A-Za-z0-9_-]{16,}={0,2})"
                    r"(?![A-Za-z0-9+/=])"),
         mask=lambda m: P.ENCODED, value_group=1, confidence="context",
         guard=lambda m: _eats_json_escape(m)
                         or ("-" not in m.group(1) and "_" not in m.group(1))
                         or _guard(_decode_b64url)(m),
         # a class run of length ≥16 is a necessary condition for a match
         needs=re.compile(r"[A-Za-z0-9_-]{16}")),

    # standard / MIME base64: A-Za-z0-9+/ with optional `=` padding.
    Rule("encoded_base64",
         re.compile(_B64_L +
                    r"([A-Za-z0-9+/]{16,}={0,2})"
                    r"(?![A-Za-z0-9+/=])"),
         mask=lambda m: P.ENCODED, value_group=1, confidence="context",
         guard=lambda m: _eats_json_escape(m) or _guard(_decode_b64)(m),
         needs=re.compile(r"[A-Za-z0-9+/]{16}")),

    # hex: even length, >=24 nibbles (>=12 bytes) so short ids/colors don't hit.
    Rule("encoded_hex",
         re.compile(r"(?<![0-9A-Za-z])"
                    r"((?:[0-9a-f]{2}){12,})"
                    r"(?![0-9A-Za-z])"),
         mask=lambda m: P.ENCODED, value_group=1, confidence="context",
         guard=lambda m: _eats_json_escape(m) or _guard(_decode_hex)(m),
         needs=re.compile(r"[0-9a-f]{24}")),      # (?:[0-9a-f]{2}){12,} ⊇ 24 hex

    # (critical EMAIL — all URL-encoded): an e-mail whose "@" is
    # encoded by ONE %40 escape ("login=ivanov%40example.com" in links and cookies).
    # encoded_percent below requires ≥3 escapes, contacts.email looks for "@" — the address went
    # out open. The mask is the same EMAIL as for a plain address. Start: after a JSONL \n escape,
    # after another %XX escape ("login%3Divanov%40…") or not after an address character.
    Rule("encoded_email_pct",
         re.compile(r"(?:(?<=\\n)|(?<=%[0-9A-Fa-f]{2})|(?<![\w.@+%\\-]))"
                    r"([A-Za-z0-9][\w.+-]{0,63}%40[A-Za-z0-9][\w-]{0,62}"
                    r"(?:\.[\w-]{1,63}){0,8}\.[A-Za-z]{2,24})(?![\w-])"),
         mask=lambda m: P.EMAIL, value_group=1, confidence="high",
         guard=lambda m: "_REDACTED" in m.group(1),
         needs="%40"),

    # percent-encoding: a URL-ish run carrying >=3 %XX escapes (escapes may be
    # interleaved with plaintext, as in real query strings:
    # `password%3DSecret123abc%26x%3D1`, `john%40example.com`). unquote then
    # test — the >=3-escape floor and the decode signal are enforced in the guard.
    # The decode signal is also a Russian trace from the dictionary / a name (_pct_guard): geo in
    # URL-encoded cookies (a city / country / region + a postal code) went out open.
    Rule("encoded_percent",
         re.compile(r"(?<![A-Za-z0-9%_.~=@:&?+/-])"
                    r"((?:[A-Za-z0-9._~=@:&?+/-]*%[0-9A-Fa-f]{2}){3,}"
                    r"[A-Za-z0-9._~=@:&?+/-]*)"),
         mask=lambda m: P.ENCODED, value_group=1, confidence="context",
         guard=lambda m: _eats_json_escape(m) or _pct_guard(m),
         needs=re.compile(r"%[0-9A-Fa-f]{2}")),   # ≥3 escape pairs ⊇ one

    # punycode host: `xn--…` label(s), optionally dotted with more labels / a
    # TLD (`xn----7sba3bilngrk4c.xn--p1ai` = «ромашка-тур.рф»). Fires only
    # when the IDNA decode yields a real Cyrillic name/brand/place OR a PII
    # signal — so a benign transliteration IDN (`xn--bda.cc` -> "ê.cc") is left
    # untouched. L keeps it off an adjacent word; the right lookahead stops the
    # host from swallowing a following path/word char.
    Rule("encoded_punycode",
         re.compile(L + r"(xn--[A-Za-z0-9-]{1,59}(?:\.(?:xn--[A-Za-z0-9-]{1,59}"
                    r"|[A-Za-z]{2,18})){0,4})" + r"(?![A-Za-z0-9-])", re.I),
         mask=lambda m: P.ENCODED, value_group=1, confidence="context",
         guard=lambda m: _punycode_host_of_email(m) or not (
             (d := _decode_punycode(m.group(1))) and _decoded_carries_name(d)
         )),

    # invisible-split reassembly: a run whose characters are interleaved with
    # soft-hyphen / zero-width space-joiner-etc. (`pass­word=Secret1`,
    # `АБВ­Г0000123456`). Strip the invisibles, re-test the reassembly as a
    # carrier or a bare secret label/value; mask only when the reassembly reads
    # as a smuggled secret/PII. The run must actually CONTAIN an invisible char
    # (the (?=…) assertion), so ordinary text never enters the reassembly path.
    # An invisible character at the EDGE of a run is not a split. A UTF-8 BOM (U+FEFF) at
    # the start of a file (Excel "CSV UTF-8", Notepad, PowerShell) made the first run
    # a candidate: `﻿Имя,Фамилия,Телефон` → the reassembly stripped the BOM, Cyrillic
    # ≥3 letters → the whole CSV header → ENCODED_REDACTED; likewise a trailing ZWSP after
    # a word. Now the group starts and ends with a VISIBLE character (edge
    # invisibles are outside the match; a start is allowed right after them), and an invisible
    # only at the edge ⇒ joined == blob in _reassemble ⇒ not a candidate.
    # PERF: the lookahead is bounded by the group window (200): previously `[^\s…]*` from every
    # start (after ",", "=", ":"…) scanned the run to its end — a CSV/URL line
    # without spaces of 150 KB → 14 s (O(n²)). An invisible further than 200 chars did not get
    # into the group before either (the guard was silent) — same semantics.
    Rule("encoded_invisible_split",
         re.compile(r"(?<![^\s\"'>(\[{=:;," + _INVISIBLE + r"])"
                    r"(?=" + _VISIBLE + r"[^\s\"'<>)\]}]{0,198}[" + _INVISIBLE + r"])"
                    r"(" + _VISIBLE + r"[^\s\"'<>)\]}]{4,198}" + _VISIBLE + r")"),
         mask=lambda m: P.ENCODED, value_group=1, confidence="context",
         guard=lambda m: not _reassemble(m.group(1)),
         # the lookahead requires an invisible character inside the run — without it
         # there is no match in the text (the rule does not touch ordinary text)
         needs=tuple(_INVISIBLE)),
]
