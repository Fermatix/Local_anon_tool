"""Deterministic token minting (per-tenant salted, ADR-001 / E1-04).

Same input value under the same tenant salt always yields the same token →
canonicalization is free (every spelling / inflection / duplicate of one entity
collapses to one token, and threads / reply-chains stay linkable across files),
while tokens of two different tenants never line up (salt isolation).

The salt is ALWAYS required: ``set_tenant(salt_hex)`` before the first minting, otherwise
SystemExit — there is no legacy "no salt" mode.

Minting version (tenant.yaml ``mint_version``, read together with the salt:
``set_tenant(salt_hex, mint_version)``; a missing field in tenant.yaml = 1):

v1 — token shapes (kept identical to the original project so output is compatible):
    tag("ORG", "acme")            -> "ORG_1234"          # %9000+1000
    tag("RESOURCE", "acme.ru")    -> "RESOURCE_12345"    # %40000+1000
    tag("USER_ID", "175…")        -> "USER_ID_912345678901"  # wide, collision-safe
    pname("иванов")               -> "PERSON_NAME_133100" # %900+1000 then "00"

v2 — wide fixed-width ranges (on a CSV column of 5000 full names
v1 put ~5–6 people into one PERSON_NAME — 900 slots; the same in prose):
    tag("ORG", "acme")            -> "ORG_1234567"           # %9e6+1e6, 7 digits
    tag("RESOURCE"/"DOC_ID"/"HANDLE"/"USER"/"PERSON", …)   # the same 7 digits
    crm_org("acme corp")          -> "ORG_23456789"          # %8e7+2e7, 8 digits
    tag("USER_ID", "175…")        -> "USER_ID_912345678901"  # as v1 (already wide)
    pname("иванов")               -> "PERSON_NAME_1234567"   # %9e6+1e6, 7 digits
Expected number of collisions n²/(2m): 5000 names → ~1.4 pairs (≤0.1 %), 50 000 →
~140 pairs (~0.5 %); in v1 the same 5000 names fell into ≤900 tokens. The tail
width is fixed (offset ≥ mod/9), the v1 and v2 shapes do not overlap (6 digits with
a "00" tail vs 7; ORG 4 vs 7; CRM 5 vs 8) — a token cannot be
confused with minting of another version. The "00" tail of v1 is compatibility with
the original project, nothing in the code depends on it (the regexes
TOKEN_RE/GLUE_RE/identity._TOKEN are width-agnostic), v2 does not need it.
The ``_h`` formula and seed canonicalization are SHARED by both versions: "one entity —
one token" within a tenant holds, only the projection onto the range changes.
Existing runs (INCR rounds) stay on v1 and mint the same people into the same
tokens; ``conveyor init`` writes mint_version: 2 only for NEW workspaces.

Phones are masked with the CONSTANT ``P.PHONE`` (``detectors.contacts``), not
a digit surrogate — there is no format-preserving digit minting in the conveyor.

`*_REDACTED` placeholders (SECRET_REDACTED, GEO_REDACTED, …) are plain constants,
not minted — see engine.placeholders.
"""
from __future__ import annotations

import hashlib

# mod ranges per prefix; anything not listed falls back to DEFAULT_MOD.
# CRM/high-cardinality columns use a wide, offset range so bulk third-party
# names don't collide with the meaningful low-numbered brand/entity tokens.
#
# USER_ID is a high-cardinality family too: ``detectors.discord_ids`` (default ON)
# mints EVERY snowflake (id/author_id/reply_to/mentions) through
# ``tag("USER_ID", digits)``. With DEFAULT_MOD=9000 at Discord scale different
# users merged into one token en masse (12k ids → ~44% merged, measured):
# false attribution of reply chains, a human verdict on one entity
# leaks to another. A wide fixed-width range
# (9·10^11 + [0..10^11) → USER_ID_900000000000..999999999999) keeps
# the 12-digit shape and does not collide at corpus scale.
MOD = {
    "ORG": 9000,
    "HANDLE": 9000,
    "USER": 9000,
    "USER_ID": (10 ** 11, 9 * 10 ** 11),  # (mod, offset) -> USER_ID_9e11..1e12-1
    "PERSON": 9000,
    "RESOURCE": 40000,
    "DOC_ID": 40000,
    "CRM": (80000, 20000),  # (mod, offset) -> ORG_20000..99999
}
DEFAULT_MOD = 9000

# --- minting version ----------------------------------------------------------
# v1 — the tables above (MOD/DEFAULT_MOD + pname %900+1000 and the "00" tail), byte for byte
# as before versions existed: existing runs keep minting the same people into the same
# tokens (otherwise INCR rounds would diverge from the already masked work/ and
# entity_map). v2 — wide fixed-width ranges (mod, offset):
# offset ≥ mod/9 keeps the number of digits constant (7 for all families, 8 for CRM), and
# the v1 and v2 shapes do not overlap. USER_ID is already wide — unchanged.
# Other kinds (ORG/HANDLE/USER/PERSON/RESOURCE/DOC_ID and unknown
# pack prefixes) suffer from the same 9000/40000 tightness on bulk columns
# (5000 companies in 9000 slots → ~28 % share a token), so v2 widens
# ALL families with one rule. CRM keeps a separate range (bulk names
# of third parties do not overlap with brands; pipeline.CRM_ORG_RE counts them).
MINT_VERSIONS = (1, 2)
_WIDE = (9_000_000, 1_000_000)        # (mod, offset) -> PREFIX_1000000..9999999
MOD_V2 = {
    "ORG": _WIDE,
    "HANDLE": _WIDE,
    "USER": _WIDE,
    "USER_ID": MOD["USER_ID"],        # USER_ID_9e11..1e12-1 — as v1
    "PERSON": _WIDE,
    "RESOURCE": _WIDE,
    "DOC_ID": _WIDE,
    "CRM": (80_000_000, 20_000_000),  # (mod, offset) -> ORG_20000000..99999999
}
DEFAULT_MOD_V2 = _WIDE
PNAME_V1 = (900, 1000)                # + "00" tail -> PERSON_NAME_100000..189900
PNAME_V2 = _WIDE                      # PERSON_NAME_1000000..9999999
_TABLES = {1: (MOD, DEFAULT_MOD), 2: (MOD_V2, DEFAULT_MOD_V2)}

# --- tenant salt ---------------------------------------------------------------
# A module global: minimal diff, one
# process = one tenant. The salt comes from runs/<co>/tenant.yaml;
# the minting version comes from there too (mint_version) and is set TOGETHER with the salt.
_SALT: bytes | None = None
_VERSION: int = 1


def set_tenant(salt_hex: str, mint_version: int = 1) -> None:
    """Set the tenant salt: 64 hex characters (urandom(32) from tenant.yaml)
    and the minting version (tenant.yaml ``mint_version``; default 1 = minting before versions,
    a missing field in tenant.yaml means the same).

    It is mixed in as a PREFIX to the already canonicalized value in ``_h`` —
    seed canonicalization does not change, so "one entity → one token"
    holds within a tenant, while tokens of different tenants never overlap.
    The salt value is never printed in error messages — it is a secret.
    """
    global _SALT, _VERSION
    if not isinstance(salt_hex, str) or len(salt_hex) != 64:
        raise SystemExit("mint.set_tenant: the salt must be a string of 64 hex characters "
                         "(runs/<co>/tenant.yaml, key salt)")
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        raise SystemExit("mint.set_tenant: the salt is not a hex string "
                         "(runs/<co>/tenant.yaml, key salt)")
    # bool is a subclass of int: True would silently pass as version 1
    if (isinstance(mint_version, bool) or not isinstance(mint_version, int)
            or mint_version not in MINT_VERSIONS):
        raise SystemExit(f"mint.set_tenant: mint_version {mint_version!r} is not in "
                         f"{MINT_VERSIONS} (runs/<co>/tenant.yaml, key mint_version)")
    _SALT = salt
    _VERSION = mint_version


def require_tenant() -> bytes:
    """Fail loudly: minting without a tenant salt is forbidden (there is no legacy mode)."""
    if _SALT is None:
        raise SystemExit("mint: the salt is not set — call set_tenant(salt_hex) "
                         "(the tenant salt from runs/<co>/tenant.yaml)")
    return _SALT


def mint_version() -> int:
    """Current minting version (1|2); without a tenant — SystemExit, like require_tenant."""
    require_tenant()
    return _VERSION


def _h(value: str) -> int:
    # sha1(salt + canon(value)): the salt is a prefix, canonicalization is unchanged.
    return int(hashlib.sha1(require_tenant() +
                            value.strip().lower().encode("utf-8", "surrogateescape")).hexdigest(), 16)


def tag(prefix: str, value: str) -> str:
    """Mint a `PREFIX_NNNN` token deterministically from `value`.

    PERSON_NAME has a DEDICATED minter ``pname`` (shape PERSON_NAME_10000..
    18999, its own namespace). Without routing, ``tag("PERSON_NAME", x)`` went
    to DEFAULT_MOD (PERSON_NAME_1000..9999) — the same entity, masked
    by an engine rule (``pname``) in one file and by a hotfix line
    (``tag``, tools/llm_hotfix) in another, got TWO tokens of two shapes.
    One class — one minting path: PERSON_NAME always goes through pname."""
    if prefix == "PERSON_NAME":
        return pname(value)
    h = _h(value)                      # require_tenant() inside — the version is valid
    table, default = _TABLES[_VERSION]
    spec = table.get(prefix, default)
    if isinstance(spec, tuple):
        mod, off = spec
    else:
        mod, off = spec, 1000
    return f"{prefix}_{h % mod + off}"


def crm_org(value: str) -> str:
    """ORG token in an isolated range (bulk CRM company columns): v1
    ORG_20000..99999, v2 ORG_20000000..99999999 (pipeline.CRM_ORG_RE)."""
    h = _h(value)
    mod, off = _TABLES[_VERSION][0]["CRM"]
    return f"ORG_{h % mod + off}"


def pname(value: str) -> str:
    """Mint a PERSON_NAME token. Seed on the canonical (lowercase nominative) key.
    v1: 900 slots + "00" tail (PERSON_NAME_100000..189900, the shape of the original
    project); v2: PERSON_NAME_1000000..9999999 (see the module header)."""
    h = _h(value)
    if _VERSION == 1:
        mod, off = PNAME_V1
        return f"PERSON_NAME_{h % mod + off}00"
    mod, off = PNAME_V2
    return f"PERSON_NAME_{h % mod + off}"
