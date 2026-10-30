"""HMAC pseudonymisation of *linking* identifiers.

``mint.py`` collapses one entity to one token via a *public* SHA-1 (so the same
brand / name / thread gets the same token everywhere — canonicalisation is the
goal, and reversibility by brute force is acceptable because the token carries no
standalone value).

Linking ids are different. A Discord/snowflake ``id``, a ``channel_id``, a
``thread_id`` or a YouTrack ``ringId`` is *only* useful for JOINING: it lets an
attacker stitch a de-identified export back to the live platform, or correlate
two exports of the same conversation. We must keep those ids **internally
consistent** (so reply-chains, thread membership and message order survive the
anonymisation and the corpus stays analysable) while making them **unlinkable to
the source**.

The tool for that is a keyed hash. Unlike ``mint``'s public formula, the key
here is the tenant salt (``runs/<co>/tenant.yaml``, 0600) that is **NEVER
placed in the archive**: without it the pseudonyms are a one-way dead end, with
it (held only by the data owner) the same original always maps to the same
pseudonym so the internal graph is intact.

    link_id("175928847299117063")     -> 481930275514   # stable 12-digit int
    link_hex("<sha256>", 64)           -> a fake but well-formed 64-char hex
    coarse_ts("2026-08-11T14:37:52Z")  -> "2026-08-11T14:00:00Z"  # hour bucket

The salt and the run directory are set by the MANDATORY ``set_tenant(salt_hex, run_dir)``.
There is no legacy ``~/.anonymizer_salt`` file any more: linking without a
tenant is a SystemExit, there is no silent fallback. The seen sidecar of issued
pseudonyms (see ``detectors.linking``) lives in
``<run_dir>/state/linking_seen.txt`` — outside the exported corpus and outside
the audit agent's subtree.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re

# --- tenant salt ---------------------------------------------------------------
# A module global, as in mint: one process = one tenant;
# switching the tenant = another set_tenant with a different run_dir.
_SALT: bytes | None = None
_RUN_DIR: str | None = None


def set_tenant(salt_hex: str, run_dir: str) -> None:
    """Set the tenant salt (the linking HMAC key) and the sidecar run directory.

    Key = 64 hex characters (urandom(32) from runs/<co>/tenant.yaml).
    ``run_dir`` must contain ``state/`` (init creates it) — the seen sidecar
    moves there from the home directory. The seen cache in detectors.linking
    is reset lazily: it is keyed by the sidecar path and re-read
    as soon as seen_path() changes. The salt is never printed in errors — it is a secret.
    """
    global _SALT, _RUN_DIR
    if not isinstance(salt_hex, str) or len(salt_hex) != 64:
        raise SystemExit("pseudo.set_tenant: the salt must be a string of 64 hex characters "
                         "(runs/<co>/tenant.yaml, key salt)")
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        raise SystemExit("pseudo.set_tenant: the salt is not a hex string "
                         "(runs/<co>/tenant.yaml, key salt)")
    state = os.path.join(run_dir, "state")
    if not os.path.isdir(state):
        raise SystemExit(f"pseudo.set_tenant: no directory {state!r} — "
                         "init the tenant first or create state/ by hand")
    _SALT = salt
    _RUN_DIR = os.path.abspath(run_dir)


def require_tenant() -> bytes:
    """Fail loudly: linking without a tenant is forbidden (there is no legacy mode)."""
    if _SALT is None:
        raise SystemExit("pseudo: the salt is not set — call "
                         "set_tenant(salt_hex, run_dir)")
    return _SALT


def seen_path() -> str:
    """Path of the current tenant's seen sidecar (append-only registry of issued
    pseudonyms, consumed by ``detectors.linking``). Lives in ``state/`` —
    a zone that is never packed into a release."""
    if _RUN_DIR is None:
        raise SystemExit("pseudo: run_dir is not set — call "
                         "set_tenant(salt_hex, run_dir)")
    return os.path.join(_RUN_DIR, "state", "linking_seen.txt")


def _hmac_hex(value: str) -> str:
    return hmac.new(require_tenant(), str(value).encode("utf-8", "surrogateescape"),
                    hashlib.sha256).hexdigest()


def link_id(value, digits: int = 12) -> int:
    """Stable keyed pseudonym for a linking id, as an ``int`` of ``digits`` digits.

    Deterministic per salt, so every occurrence of one original id (across files,
    reply-chains and thread membership) maps to the same pseudonym — the internal
    join graph is preserved — but the mapping can't be inverted without the
    (unarchived) salt.

    Exactly as many hex characters are taken from the HMAC as needed for the source
    range to cover the modulus ``10**digits``. Historically [:16] = 64
    bits ≈ 1.844e19 were taken; for ``digits>=20`` (10^20 > 2^64) the mod degenerated into a no-op —
    every 20+-digit pseudonym started with '1' (a uniform fingerprint "the field
    was an id"). While the modulus fits in 64 bits (``digits<=19``, all
    snowflake/ring widths), the slice stays [:16] — byte for byte with the old
    formula, corpora already issued do not drift; wider — the whole digest is taken
    (sha256 = 256 bits, uniform for any practical width)."""
    h = _hmac_hex(value)
    hexlen = 16 if 10 ** digits <= 2 ** 64 else len(h)
    return int(h[:hexlen], 16) % (10 ** digits)


def link_str(value, digits: int = 12) -> str:
    """``link_id`` rendered as a fixed-width, leading-zero-free numeric string so
    it stays *shaped* like the snowflake/id it replaces (first digit forced 1-9).
    """
    n = link_id(value, digits)
    s = str(n).rjust(digits, "0")
    if s[0] == "0":                       # keep it a full-width id, no lost length
        s = "1" + s[1:]
    return s


def link_hex(value, length: int = 64) -> str:
    """A fake but well-formed lowercase-hex pseudonym of ``length`` for hash/etag
    fields (sha256/sha1/md5/checksum). Keyed + stable, same shape as the source.

    The input is case-canonicalized (``.strip().lower()``): the rule
    ``link_hash_field`` matches ``[0-9a-fA-F]{32,64}`` — an upper-case
    MD5/etag is legal. The same digest in different case MUST give the same
    pseudonym, otherwise a file with an UPPER hash and another with a lower rendering of the
    same control value stop linking. The output is always
    lower-case, so values already in lower case do not change."""
    canon = str(value).strip().lower()
    out = ""
    i = 0
    while len(out) < length:
        out += hmac.new(require_tenant(), (canon + str(i)).encode("utf-8", "surrogateescape"),
                        hashlib.sha256).hexdigest()
        i += 1
    return out[:length]


# ---- timestamp coarsening ----------------------------------------------------
# ISO-8601 with a preserved trailing zone / fractional-less tail.
_ISO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}):\d{2}:\d{2}(?:\.\d+)?(.*)$")
# bare epoch: 10-digit seconds or 13-digit milliseconds (with optional .fraction).
_EPOCH_RE = re.compile(r"^(\d{10}|\d{13})(\.\d+)?$")
_HOUR = 3600
_HOUR_MS = 3600 * 1000


def coarse_ts(ts):
    """Truncate a timestamp to the hour, KEEPING the input's format.

    * ISO ``2026-08-11T14:37:52.418Z`` -> ``2026-08-11T14:00:00Z`` (zone kept).
    * epoch seconds ``1754923072``     -> ``1754922000`` (hour bucket, still s).
    * epoch millis  ``1754923072123``  -> ``1754920800000`` (hour bucket, still ms).

    Returns ``None`` if ``ts`` is not a recognisable timestamp (caller then leaves
    the value untouched). Coarsening to the hour drops the fine-grained timing
    that fingerprints a specific user's activity while preserving day/hour-level
    analysability.
    """
    s = str(ts)
    m = _ISO_RE.match(s)
    if m:
        return m.group(1) + ":00:00" + (m.group(2) or "")
    m = _EPOCH_RE.match(s)
    if m:
        digits = m.group(1)
        n = int(digits)
        if len(digits) == 13:             # milliseconds
            return str(n - (n % _HOUR_MS))
        return str(n - (n % _HOUR))       # seconds
    return None
