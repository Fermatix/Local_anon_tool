"""Pack "client names — always mask": the MECHANISM lives in the repo, the DATA only in the run dir.

Names of clients and their projects are client material: they do not belong in code or git history
(the repo is shared by different companies; the ru_trace dictionary holds only well-known Russian
companies/services/places). The pack file lives in the run dir and is wired by the decisions.yaml key
``known_files`` — a path inside the run (compile finds the file in the run dir and fails if it is
missing). Template (runs/<co>/known_values/clients.py):

    from anonymizer.known_values.client_pack import client_rules

    RULES = client_rules([
        # one name — one ORG token for all spellings
        {"seed": "acme77", "names": ["ACME77", "АКМЕ77"]},
        {"seed": "romashka", "names": ["romashka.market"], "exact": ["Ромашка Маркет", "Romashka"],
         "segments": ["Romashka", "romashka"]},
    ])

Entry fields:
  names    — whole words, case-insensitive (Latin and Cyrillic), also as a host/path/file segment
             ("acme77-prod-db-1", "/home/acme77", "acme77_dev1", "app.acme77.local"); a Russian case
             ending goes under the token; a name ending in a digit also matches as a glued identifier
             prefix ("acme77dev1"). Not touched: "ACME 77", "xacme77", "acme770";
  exact    — the same, but case-SENSITIVE: for names that are also common words ("Ромашка Маркет" — yes,
             "ромашка в поле" — no);
  segments — a case-SENSITIVE stem as a segment of a code identifier: after a lowercase letter/digit/"_"
             or at a word start, followed by a non-lowercase char ("CompanyRomashkaController",
             "isRomashka", "romashka_api").
"""
from __future__ import annotations

import re

from ..engine import mint
from ..engine.boundary import L, R, is_own_token
from ..engine.rule import Rule

_FIELDS = ("seed", "names", "exact", "segments")
_LISTS = ("names", "exact", "segments")


def _alt(forms) -> str:
    return "|".join(re.escape(f) for f in sorted(set(forms), key=lambda x: (-len(x), x)))


# A tail of "[а-яё]*" took any continuation: client "Нова" in exact masked "Новая версия",
# "Мир" masked "Мировой рекорд", "Акме" in names masked "Акмеизм". Now the tail is only case
# endings (shared by names and exact); "Новой", "Миром", "Мира" are still a token.
_CASE_TAIL = (r"(?:ами|ями|ого|его|ому|ему|ой|ою|ей|ею|ом|ем|ам|ям|ах|ях|ов|ев|ым|им|"
              r"а|у|е|ы|и|ю)?")


def _tail() -> str:
    return _CASE_TAIL


def _guard(m) -> bool:
    return is_own_token(m.group(1), part=True)       # only our own token


def _check(sp, seeds) -> tuple:
    if not isinstance(sp, dict) or set(sp) - set(_FIELDS):
        raise SystemExit(f"client_pack: entry {sp!r} — allowed fields: {', '.join(_FIELDS)}")
    seed = sp.get("seed")
    if not (isinstance(seed, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]*", seed, re.ASCII)) or seed in seeds:
        raise SystemExit(f"client_pack: seed {seed!r} — lowercase Latin letters/digits, unique in the pack")
    lists = {k: sp.get(k) or [] for k in _LISTS}
    if not any(lists.values()) or not all(
            isinstance(v, list) and all(isinstance(x, str) and x.strip() == x and x for x in v)
            for v in lists.values()):
        raise SystemExit(f"client_pack: {seed}: need at least one non-empty list of strings "
                         f"{'/'.join(_LISTS)} (no leading/trailing spaces)")
    return seed, lists


def client_rules(specs) -> list:
    """Client pack rules from entries (see the module header). A malformed entry is a SystemExit: a pack
    that silently does not mask a client leaks an "always mask" value."""
    rules, seeds = [], set()
    for sp in specs:
        seed, lists = _check(sp, seeds)
        seeds.add(seed)

        def mask(m, s=seed):
            return mint.tag("ORG", s)
        for kind, flags in (("names", re.IGNORECASE), ("exact", 0)):
            forms = lists[kind]
            if not forms:
                continue
            rules.append(Rule(f"client:{seed}:{kind}",
                              re.compile(L + "((?:" + _alt(forms) + r")" + _tail() + ")" + R, flags),
                              mask=mask, value_group=1, guard=_guard, confidence="high"))
            tail_digit = [f for f in forms if f[-1].isdigit()]
            if tail_digit:
                rules.append(Rule(f"client:{seed}:{kind}:glued",
                                  re.compile(L + "(" + _alt(tail_digit) + r")(?=[A-Za-zА-Яа-яЁё])", flags),
                                  mask=mask, value_group=1, guard=_guard, confidence="high"))
        for stem in lists["segments"]:
            left = r"(?:(?<=[a-z0-9_])|(?<![A-Za-z0-9]))" if stem[:1].isupper() \
                else r"(?:(?<=_)|(?<![A-Za-z0-9]))"
            rules.append(Rule(f"client:{seed}:segment:{stem}",
                              re.compile(left + "(" + re.escape(stem) + r")(?![a-z])"),
                              mask=mask, value_group=1, guard=_guard, confidence="high"))
    return rules
