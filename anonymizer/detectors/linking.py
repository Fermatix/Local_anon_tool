"""Linking-id pseudonymiser (OPT-IN: ``config.detectors.linking = False``).

The general engine masks *personal data*. It deliberately leaves the structural
plumbing of a chat/issue export alone: snowflake ``id`` s, ``channel_id`` /
``guild_id`` / ``thread_id`` / ``parent_id`` references, YouTrack ``ringId`` s,
message/attachment ids, ``sha256`` file digests, exact millisecond timestamps.
None of that is PII on its own — but together it is a *linkage* surface: it lets
anyone stitch a de-identified export back to the live platform (look the id up in
Discord/YouTrack) or correlate two exports of the same conversation.

This detector closes that gap WITHOUT breaking analysability. It pseudonymises
each linking id through the keyed, salted ``engine.pseudo.link_id`` — so the same
original id always maps to the same pseudonym (reply-chains, thread membership
and message order survive) but the mapping can't be inverted without the
unarchived salt. Obvious high-resolution millisecond/ISO timestamps are coarsened
to the hour (kills per-user activity fingerprinting; keeps day/hour analytics).

FP-safety: every rule is anchored to a JSON-ish ``"field": value`` context (a
quoted key from a known id/hash/timestamp field set, then ``:`` , then the
value). A bare number in prose is never touched — the key is the evidence, the
same discipline ``financial.py`` uses for label-only ids. OFF by default; enable
per corpus after eyeballing the dry-run counts.

Idempotency (invariant 4): ``link_str`` / ``link_hex`` emit digit/hex strings
*shaped like their own input*, so a naive re-run would re-match and re-HMAC the
pseudonyms (``link(link(x)) != link(x)``) and partially re-run corpora would
stop sharing pseudonyms for one original id. The pseudonyms are not token-shaped
either, so ``boundary.TOKEN_RE`` cannot protect them. Two guards close this:
every minted pseudonym is recorded in a sidecar under the tenant's
``state/`` dir (unarchived, like the salt) and skipped on later passes, and
digit pseudonyms are nudged so they can
never checksum as INN/OGRN/bank account/card and be eaten by ``financial.py``'s
label-free rules, which run earlier in ``DETECTOR_ORDER``.
"""
from __future__ import annotations

import os
import re

from ..engine import pseudo
from ..engine import validators as V
from ..engine.rule import Rule

# --- field name sets (mirrors the study's ID_FIELDS / TS_FIELDS / SHA_FIELDS) --
_ID_FIELDS = (
    "id", "reply_to", "in_reply_to", "attachment_id", "parent_id", "ringId",
    "ring_id", "message_id", "channel_id", "guild_id", "server_id", "author_id",
    "user_id", "member_id", "thread_id", "referenced_message_id", "webhook_id",
    "application_id", "interaction_id", "role_id", "emoji_id", "sticker_id",
)
_TS_FIELDS = (
    "created_at", "edited_at", "timestamp", "created", "updated", "resolved",
    "exported_at", "date", "joined_at", "premium_since", "edited_timestamp",
)
_SHA_FIELDS = ("sha256", "sha1", "md5", "hash", "checksum", "digest", "etag")


def _keys(names):
    """Regex alternation of JSON keys, longest-first (so ``message_id`` wins over
    ``id``); matched whole between the quotes."""
    return "|".join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))


# The lead-in that is KEPT verbatim: opening quote, key, closing quote, colon,
# optional whitespace, and the opening quote of a string value if present.
_LEAD = r'("(?:%s)"\s*:\s*"?)'
# Numeric id value: 6..25 digits (snowflakes are 17-20; YouTrack ring ids vary).
_IDVAL = r"(\d{6,25})"
# Trailing optional closing quote for string-form ids, kept verbatim.
_TAIL = r'("?)'


# --- emitted-pseudonym sidecar (invariant 4: re-run on own output = no-op) ----
# Deterministic per-salt hashing keeps SEPARATE runs consistent, but only if
# every value is processed exactly ONCE — a second pass over an already-masked
# file would map pseudonym -> new pseudonym and break the join graph. The salt
# gives determinism per original; this sidecar is the missing "already
# processed" oracle. It lives in <run_dir>/state/linking_seen.txt (E1-05):
# outside the exported corpus AND outside the audit-agent subtree, 0600, one
# pseudonym per line. A false skip — a fresh original id that equals an earlier
# pseudonym — has probability ~10^-len per id and merely leaves that one id
# unmasked, which is the safe direction for a *linking* (non-PII) field.
# The path is taken LAZILY from pseudo.seen_path(): there is no import-time path,
# without pseudo.set_tenant linking fails openly (SystemExit) on first use.
_SEEN: set[str] | None = None
_SEEN_FOR: str | None = None      # path the cache was loaded for: a tenant
                                  # change changes seen_path() -> reload


def _seen() -> set[str]:
    """The persisted set of pseudonyms we ever emitted, loaded once per tenant."""
    global _SEEN, _SEEN_FOR
    path = pseudo.seen_path()
    if _SEEN is None or _SEEN_FOR != path:
        try:
            with open(path, encoding="ascii") as fh:
                _SEEN = {ln.strip() for ln in fh if ln.strip()}
        except FileNotFoundError:
            _SEEN = set()   # no first run yet — normal
        except OSError as e:
            # an existing but unreadable registry is MORE DANGEROUS than a missing
            # one: a silent empty set re-mints pseudonyms on a repeated apply
            # (link(link(x)) != link(x)) and breaks the join graph.
            raise SystemExit(f"linking: cannot read the pseudonym registry {path!r}: {e}")
        _SEEN_FOR = path
    return _SEEN


def _remember(token: str) -> None:
    """Record a freshly minted pseudonym (append-only, private like the salt).
    If the sidecar can't be written, the in-memory set still guards this run."""
    s = _seen()
    if token in s:
        return
    s.add(token)
    try:
        fd = os.open(pseudo.seen_path(), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (token + "\n").encode("ascii"))
        finally:
            os.close(fd)
    except OSError as e:
        # falling back to memory does not survive the process: the next run
        # re-mints pseudonyms and breaks reply chains. Fail loudly.
        raise SystemExit(f"linking: cannot write the pseudonym registry "
                         f"{pseudo.seen_path()!r}: {e}")


def _already_minted(m: re.Match) -> bool:
    """Rule guard (True => leave untouched): the candidate value is one of our
    own previously emitted pseudonyms — re-pseudonymising it would violate
    invariant 4. ``link_hex`` is always lowercase, so exact match suffices."""
    return m.group(2) in _seen()


def _unfinancial(s: str) -> str:
    """Nudge a digit pseudonym so it can never checksum as a financial id.

    ``financial.py``'s label-free rules (INN 10/12, OGRN 13/15, 20-digit
    account, solid-16 Luhn card) run BEFORE this detector, so on a later pass a
    pseudonym that happens to validate would be eaten into ``INN_REDACTED`` /
    ``OGRN_REDACTED`` and the join graph would lose that id. The nudge is a
    pure function of the HMAC output, so "same original -> same pseudonym"
    still holds; it never touches the first digit, so the id keeps full width.
    """
    if V.inn_ok(s) or V.ogrn_ok(s) or (len(s) == 16 and s[0] in "256" and V.luhn_ok(s)):
        # these all carry their control digit LAST -> any change there breaks it
        s = s[:-1] + str((int(s[-1]) + 1) % 10)
    while V.account_struct(s):        # structure lives in s[:8] -> bump 2nd digit
        s = s[0] + str((int(s[1]) + 1) % 10) + s[2:]
    return s


def _mask_id(m: re.Match) -> str:
    """Replace the id value with a keyed pseudonym of the SAME length (so a
    snowflake stays snowflake-shaped), keeping key + quoting verbatim."""
    val = m.group(2)
    out = _unfinancial(pseudo.link_str(val, len(val)))
    _remember(out)
    return m.group(1) + out + m.group(3)


def _mask_hash(m: re.Match) -> str:
    val = m.group(2)
    out = pseudo.link_hex(val, len(val))
    _remember(out)
    return m.group(1) + out + m.group(3)


def _mask_ts(m: re.Match) -> str:
    val = m.group(2)
    coarse = pseudo.coarse_ts(val)
    if coarse is None:                       # not a timestamp we recognise -> leave
        return m.group(0)
    return m.group(1) + coarse + m.group(3)


RULES: list[Rule] = [
    # === linking id in a JSON id field -> keyed pseudonym (same length) ===
    # "id":"175928847299117063"  /  "channel_id": 481930275514  ->  pseudonym.
    Rule("link_id_field",
         re.compile(_LEAD % _keys(_ID_FIELDS) + _IDVAL + _TAIL),
         mask=_mask_id, value_group=2, guard=_already_minted, confidence="context",
         note="pseudonymise snowflake/discord/message/thread/ring ids via HMAC salt"),

    # === content hash / etag field -> keyed hex pseudonym (same length) ===
    Rule("link_hash_field",
         re.compile(_LEAD % _keys(_SHA_FIELDS) + r"([0-9a-fA-F]{32,64})" + _TAIL),
         mask=_mask_hash, value_group=2, guard=_already_minted, confidence="context",
         note="re-key sha256/sha1/md5/checksum/etag digests (unlinkable, same shape)"),

    # === high-resolution timestamp field -> coarsen to the hour, keep format ===
    # ISO string ("2026-08-11T14:37:52.418Z") or bare epoch s/ms (1754923072123).
    Rule("link_ts_field",
         re.compile(_LEAD % _keys(_TS_FIELDS) +
                    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?[^\"\n]*"
                    r"|\d{13}(?:\.\d+)?(?!\d)|\d{10}(?:\.\d+)?(?!\d))"
                    + _TAIL),
         mask=_mask_ts, value_group=2, confidence="context",
         note="coarsen obvious ms/ISO activity timestamps to the hour"),
]
