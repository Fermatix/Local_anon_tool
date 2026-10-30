"""Structural document identity — the idempotency / dup / collision oracle.

Two documents are 'the same document, differently masked' when they share a
*structural skeleton*: identical apart from which values got tokenised and which
digits got surrogated. ``struct_hash`` reduces a text to that skeleton and
fingerprints it, so the pipeline can answer three questions cheaply:

  * idempotency — re-running the pipeline over already-masked output must not
    change the skeleton (``struct_hash(once) == struct_hash(twice)``);
  * dup detection — two source files that mask to the same skeleton are the same
    document (mirror the ``findlost`` normalise-then-sha1 sweep);
  * collision — a source and its masked counterpart share a skeleton, so a
    candidate source can be paired with a masked file without a filename match.

Normalisation (mirrors the original ``findlost.py``):

  * every mask token of ours (``ORG_1234`` / ``PERSON_NAME_133100`` /
    ``EMAIL_REDACTED`` — known prefixes only) collapses to a single
    ``#`` — masking a value must not change the skeleton;
  * every run of digits collapses to a single ``0`` — a digit-surrogate
    (e.g. ``pseudo.link_str``) or a re-numbered token must not change the skeleton;
  * every whitespace run collapses to one space, ends stripped — reflow /
    re-serialisation must not change the skeleton.

Then sha1 over the ``surrogateescape`` bytes (the corpus is not clean UTF-8, see
``engine.walk``), so the fingerprint round-trips on every file in the tree.
"""
from __future__ import annotations

import hashlib
import re

from .boundary import own_token_re

# A mask token in any of the shapes the engine mints:
#   PREFIX_123, PREFIX_123_456, PERSON_NAME_133100, FOO_REDACTED.
# Kept in sync with ``engine.boundary.TOKEN_RE`` (same body without the outer
# group) — the two MUST stay in sync. Quantifiers are bounded and the pattern starts
# with (?<![A-ZА-Я0-9]) — the same performance guard as in boundary: unbounded tails
# on a long UPPER/underscore run made normalize() scan in O(n²) (see the comment
# at TOKEN_RE). The prefix may itself be underscore-joined (``PERSON_NAME``), so
# match a run of UPPER/digit segments joined by ``_``, ending in either a
# numeric tail (``_133100`` / ``_20000_1``) or ``_REDACTED``.
_TOKEN = re.compile(
    r"(?<![A-ZА-Я0-9])[A-ZА-Я][A-ZА-Я0-9]{0,31}(?:_[A-ZА-Я0-9]{1,32}){0,8}"
    r"_(?:\d{1,32}(?:_\d{1,32}){0,7}|REDACTED)"
)
# Only OUR token (known prefixes, boundary.own_token_re) collapses to "#" — a
# foreign "PETROV_77" stays part of the skeleton like any word. _TOKEN stays the
# boundary (a test keeps it in sync with TOKEN_RE).
_OWN = own_token_re()
_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Reduce a text to its structural skeleton (token -> ``#``, digits -> ``0``,
    whitespace -> single space, stripped). Order matters: collapse tokens FIRST
    so a token's trailing digits become part of the ``#`` and never leak a ``0``.
    """
    text = _OWN.sub("#", text)
    text = _DIGITS.sub("0", text)
    text = _WS.sub(" ", text)
    return text.strip()


def struct_hash(text: str) -> str:
    """sha1 hex of the normalised skeleton — the structural identity of a doc.

    Equal hashes mean 'same document, different masking' (idempotent re-run,
    duplicate source, or source<->masked pair). ``surrogateescape`` so byte-dirty
    corpus files hash without blowing up (see ``engine.walk``)."""
    skel = normalize(text)
    return hashlib.sha1(skel.encode("utf-8", "surrogateescape")).hexdigest()
