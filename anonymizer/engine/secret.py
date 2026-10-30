"""Entropy-based 'is this value a credential?' gate + vault-context detection.

The crown-jewel FP-guard block from credfinal/megafix: label-free secret
detection is dangerous, so it (a) only judges a *whole value* (JSON leaf / CSV
cell), and (b) fires only inside a vault/access zone. Vendor-signature and
label-anchored secret rules (see detectors/secrets.py) don't need this — the
signature/label is evidence enough.
"""
from __future__ import annotations

import math
import re

from .boundary import is_placeholder

COMMON = {
    "true", "false", "null", "none", "nan", "localhost", "example", "test",
    "utf-8", "application", "json", "text", "html", "string", "number", "object",
    "array", "boolean", "default", "unknown", "success", "error", "warning",
    "password", "username", "email", "phone", "address", "name", "title",
    "description", "content", "message", "status", "active", "enabled",
}

_HEX = re.compile(r"^[0-9a-fA-F]+$")
_SEMVER = re.compile(r"^\d+(\.\d+){1,3}([.\-+]\w+)*$")
# (a coordinator signal: «"created_at": "2026-04-08T11:00:00+00:00"» in a JSONL file named like a
# vault zone → SECRET_REDACTED): an ISO 8601 timestamp / date is not a credential. Length ≥20 and
# two character classes (digits + «-:+T») passed the «cc ≥ 2 and len ≥ 20» branch of the entropy gate
# for ANY year; no year range is needed here.
_DATETIME = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d{1,9})?)?)?"
    r"(?:Z|[+-]\d{2}(?::?\d{2})?)?"
    r"|\d{2}[./-]\d{2}[./-]\d{4}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d{1,9})?)?)?"
    r"|\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?)$")
# Template/tag/path markers a whole-value credential never contains. NB: bare
# '=' and '/' are deliberately NOT here — base64 padding ends in '=' and the
# base64 / AWS-secret-key alphabets contain '/'; those two are judged in
# _looks_pathy so real credentials aren't whitelisted by their own alphabet.
_TEMPLATE = re.compile(r"\\|\.\.|\{\{|\$\{|<|>")
_PAD = re.compile(r"={1,2}$")                     # base64 padding (max 2)
# A file/attachment name with an extension (IMG_20240101.png,
# Report-Q3_final.xlsx, backup_2024.tar) is a name value, not a credential:
# its high entropy comes from the date/number. A tail «.<letter><up to 4 alnum>» —
# base64/hex/JWT signatures never have a dot with a short letter tail.
_FILENAME_EXT = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,4}$")
# A slash-separated segment that reads as a plain word (`usr`, `local`,
# `octet-stream`) — not a mixed-case/digit credential chunk like `K7MDENG`.
_WORD_SEG = re.compile(r"[A-Za-z][a-z]*(?:-[a-z]+)*")


def _looks_pathy(v: str) -> bool:
    """True for template/path/k=v shapes that disqualify a whole-value secret.

    Base64-compatible on purpose: the caller strips trailing ``=`` padding
    first, so a remaining ``=`` is a real k=v pair; ``/`` alone is NOT pathy
    (AWS secret keys contain it) — only ``/`` next to a dot-segment or
    word-like segments reads as a real path/MIME shape."""
    if _TEMPLATE.search(v):
        return True
    if "=" in v:                    # internal k=v pair, not a whole value
        return True
    if "/" not in v:
        return False
    if "." in v:                    # file.ext / host.tld — no dots in base64
        return True
    segs = v.split("/")
    words = sum(1 for s in segs if _WORD_SEG.fullmatch(s))
    # `usr/local/bin`, `application/octet-stream` — words between slashes;
    # `wJalrXUtnFEMI/K7MDENG/bPxRf…` — zero word segments, stays a candidate
    return words >= 2 or (len(segs) >= 3 and words >= 1)


def shannon(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def char_classes(s: str) -> int:
    return (
        bool(re.search(r"[a-z]", s))
        + bool(re.search(r"[A-Z]", s))
        + bool(re.search(r"\d", s))
        + bool(re.search(r"[^A-Za-z0-9]", s))
    )


def is_secret_value(v: str, hash_ok: bool = False) -> bool:
    """True if a whole value looks like a credential (entropy shape).

    hash_ok — hex 32/40/64 (md5/sha1/sha256 shape) WITHOUT
    evidence is a digest, not a secret: a bare sha256 in prose/logs, a git commit
    SHA, a file md5, a hex URL segment. The same shape UNDER a label (api_key=,
    password:, secret=) or in a vault zone is a live key (Mailchimp/Trello/
    OpenWeather — 32 hex, GitHub classic PAT — 40, VirusTotal — 64), and it used to
    leak to the client unmasked. A caller that already has the label/zone
    (guard label_value/bearer in detectors/secrets, _secret_value_fn in
    pipeline) passes hash_ok=True — hex is judged by entropy like any other
    value. Unlabelled calls (encoding, canary) keep the default False."""
    v = v.strip()
    if not (8 <= len(v) <= 200):
        return False
    if " " in v or is_placeholder(v):
        return False
    if re.search(r"[А-Яа-яЁё]", v):          # Russian prose, not a key
        return False
    if v.lower() in COMMON or _looks_pathy(_PAD.sub("", v)):
        return False
    if v.isdigit() or _SEMVER.match(v) or _DATETIME.match(v):
        return False
    if _FILENAME_EXT.search(v):               # a file name with an extension
        return False
    # md5/sha1/sha256 — repaired elsewhere, not a secret… while there is no label/zone
    # (under a label the same string is a key; the caller passes hash_ok=True)
    if not hash_ok and _HEX.match(v) and len(v) in (32, 40, 64):
        return False
    ent = shannon(v)
    cc = char_classes(v)
    if cc >= 3 and ent >= 2.8:
        return True
    if cc >= 2 and (ent >= 3.4 or len(v) >= 20):
        return True
    # word + digits + symbol password shape (keklol1234!, Forest521)
    if re.fullmatch(r"[A-Za-z]+\d{2,}[!@#$%^&*_.-]*", v) and len(v) >= 8:
        return True
    if len(v) >= 32 and ent >= 3.0:
        return True
    return False


VAULT_KEY = re.compile(
    r"(?i)(парол|password|passwd|pwd|passcode|секрет|secret|токен|token|"
    r"api[_-]?key|access[_-]?key|accesskey|client[_-]?secret|ключ|доступ|"
    r"логин|login|credential|auth|bearer|supervisor)"
)


def is_vault_context(*hints) -> bool:
    """True if any hint (JSON key, CSV header, filename, record title) marks a
    secret zone. Weak label-free secret detection fires only here."""
    return any(h and VAULT_KEY.search(str(h)) for h in hints)
