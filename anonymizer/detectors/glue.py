"""A token inside a compound identifier — glue post-rules.

A pack literal (roster nick, name, brand) matches between the ``L``/``R``
boundaries, for which "_" is not a letter: a literal INSIDE a snake_case
identifier is replaced, while the neighbouring segments stay welded to the
token through "_":

    <nick>_god               →  HANDLE_NNN_god          (the nick tail leaked)
    North_Romashka_South     →  North_ORG_NNN_South  (a client name in
                                local_path of a Discord export)

Literal rules do not know they sit inside an identifier, and their left
boundary cannot change: a pack pattern must start with the literal
alternation (the AC prefilter derives probes from the leading literal; an
optional prefix before it would disable the prefilter on a pack of tens of
thousands of forms). So the glue is removed by a SEPARATE pass AFTER all
rules (Pipeline adds RULES last, after the packs; the fixpoint picks up
tokens minted on the second pass):

* ``glue_handle`` — HANDLE/USER: a nick is a single identifier, so ANY
  alphanumeric segments welded with a single "_" go under the same token
  ("mr_HANDLE_1_god" → "HANDLE_1").
* ``glue_proper`` — ORG/PERSON_NAME: only NAME segments go under the token
  (a Capitalized/CamelCase word or an all-caps word of letters):
  "North_ORG_1_South" → "ORG_1", "Ivan_PERSON_NAME_5" → "PERSON_NAME_5".
  Lowercase code tails ("ORG_1_backend_url", "PERSON_NAME_5_id") are left
  alone — they are not a name.

What the rules do NOT do: they leave a double underscore alone
("USER_ID_9__file.jpg" — an attachment name separator, not glue), a digit
segment right after the token ("_2024" cannot be told from the minted form
PREFIX_1_2) and a neighbouring OTHER token ("HANDLE_1_ORG_2" — two separate
tokens, must not merge). USER_ID (a numeric Discord id) is not covered. The
replacement is the token itself: no new entity is minted, the identity is
the one the pack literal gave.

PERF: a match starts only at the START of an alphanumeric run (strict
lookbehind) and takes at most 12 segments of ≤64 chars on each side — one
pass per run, no quadratic scan of starts inside long base64url strings;
``needs`` skips the rule when there are no tokens of the family in the text.
"""
from __future__ import annotations

import re

from ..engine.boundary import _PREFIX_ALT
from ..engine.rule import Rule

_AN = r"A-Za-zА-Яа-яЁё0-9"
# the neighbouring segment is (or contains) a minted token — do not merge
_HAS_TOKEN = re.compile(r"(?:" + _PREFIX_ALT + r")_(?:\d|REDACTED)")
_HAS_LETTER = re.compile(r"[A-Za-zА-Яа-яЁё]")
# name segment: a Capitalized/CamelCase word or all caps of ≥2 letters (Latin/Cyrillic)
_NAME_SEG = re.compile(r"(?:(?:[A-ZА-ЯЁ][a-zа-яё]+)+|[A-ZА-ЯЁ]{2,})$")


def _any_seg(seg: str) -> bool:
    return bool(_HAS_LETTER.search(seg))


def _name_seg(seg: str) -> bool:
    return bool(_NAME_SEG.match(seg))


def _glue_rule(name: str, token_alt: str, absorbable, needs) -> Rule:
    rx = re.compile(
        r"(?<![" + _AN + r"_])"
        r"((?:[" + _AN + r"]{1,64}_){0,12})"
        r"((?:" + token_alt + r")_\d+)"
        r"((?:_[" + _AN + r"]{1,64}){0,12})"
        r"(?![" + _AN + r"])")

    def mask(m):
        head, tok, tail = m.group(1), m.group(2), m.group(3)
        if not head and not tail:
            return m.group(0)
        keep_head, keep_tail = head, tail
        if head and not _HAS_TOKEN.search(head):
            segs = head[:-1].split("_")
            while segs and absorbable(segs[-1]):
                segs.pop()
            keep_head = "".join(s + "_" for s in segs)
        if tail and not _HAS_TOKEN.search(tail):
            segs = tail[1:].split("_")
            # a digit segment right after the token may be the tail of a minted
            # form (PREFIX_1_2), not glue: stop
            i = 0
            while i < len(segs) and absorbable(segs[i]):
                i += 1
            keep_tail = "".join("_" + s for s in segs[i:])
        return keep_head + tok + keep_tail

    return Rule(name, rx, mask=mask, value_group=0, confidence="high",
                needs=needs)


RULES: list[Rule] = [
    _glue_rule("glue_handle", r"HANDLE|USER", _any_seg, ("HANDLE_", "USER_")),
    _glue_rule("glue_proper", r"PERSON_NAME|ORG", _name_seg,
               ("ORG_", "PERSON_NAME_")),
]
