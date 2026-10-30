"""Links: placeholder replacement and domain masking. A link is never removed — the message, field
or document with it stays, only the URL itself changes.

* ``[links].delete_hosts`` and their subdomains, hosts with ``[links].delete_host_prefixes``
  ("confluence.company.com") — the whole URL (scheme, host, port, path, query, fragment) is
  replaced with a placeholder — exactly ``RESOURCE_REDACTED``. Without a scheme too ("figma.com/file/…",
  "disk.yandex.ru/d/…"), a bare host mention too. These rules go FIRST in the chain
  (engine_adapter): otherwise the engine's ``domain`` rule turns the host into a RESOURCE token earlier.
  In Telegram ``text_entities`` (``link``, ``text_link``: ``href``) the same rule works through the
  profile slots; trailing punctuation and the closing bracket of a markdown link stay outside.
* ``mask_other_domains`` — the host of any other link (with a scheme, with "www." or with a path) →
  ``RESOURCE_…`` with the same seed as the engine's ``domain`` (host in lower case); the path stays.
  The engine itself keeps hosts from TECH_TERMS (github.com, gitlab…) — but "repositories stay, the
  domain → RESOURCE", so here they are masked (a rule in the pack slot).
"""
from __future__ import annotations

import re

from anonymizer.engine import mint
from anonymizer.engine import placeholders as P
from anonymizer.engine.boundary import alt_longest_first, is_own_token
from anonymizer.engine.rule import Rule

REDACTED = "RESOURCE_REDACTED"
_LABEL = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
_HOST = _LABEL + r"(?:\." + _LABEL + r")*\.[a-z]{2,24}"
# path/query/fragment ("?…", "#…" and without "/": miro.com?share_link_id=…): up to a space, quote,
# angle bracket, "\" (JSON escape — ``\n``/``\"`` stay intact); the JSON escape "\/" in .txt/.log
# (no transport view there) is part of the path
_PATH = r"(?:(?:\\/|[/?#])(?:[^\s\"'<>«»\\]|\\/)*)?"
_TRAIL = ".,;:!?»”’"
_LEFT = r"(?:(?<=\\[ntrbfv])|(?<![\w.@\\/-]))"
# after the host: not a letter/digit/hyphen and not ".label" (figma.community); "_" is allowed — the
# file name "figma.com_export.txt" and its path stay without the host
_HOST_END = r"(?![^\W_]|-|\.[\w-])"
_SCHEME = r"(?:[a-z][a-z0-9+.-]{1,10}:(?://|\\/\\/)(?:[^\s/@\"'<>\\]{1,64}@)?)?"


def _strip_trail(s: str) -> tuple:
    """(body, tail): trailing punctuation and unpaired closing brackets stay outside."""
    i = len(s)
    while i > 0:
        ch = s[i - 1]
        if ch in _TRAIL:
            i -= 1
        elif ch == ")" and s[:i].count(")") > s[:i].count("("):
            i -= 1
        elif ch == "]" and s[:i].count("]") > s[:i].count("["):
            i -= 1
        else:
            break
    return s[:i], s[i:]


def delete_rules(hosts, prefixes) -> list:
    hosts = sorted({h.strip().lower().lstrip(".") for h in hosts if h and h.strip()})
    prefixes = sorted({p.strip().lower() for p in prefixes if p and p.strip()})
    alts = []
    if hosts:
        alts.append(r"(?:" + _LABEL + r"\.)*(?:" + alt_longest_first(hosts) + r")")
    for p in prefixes:
        p = p.rstrip(".")
        alts.append(re.escape(p) + r"(?:\." + _LABEL + r")+")
    if not alts:
        return []
    rx = re.compile(_LEFT + _SCHEME + r"(?:" + "|".join(alts) + r")(?::\d{2,5})?"
                    + _HOST_END + _PATH, re.IGNORECASE)

    def mask(m):
        body, tail = _strip_trail(m.group(0))
        return REDACTED + tail

    return [Rule("link:delete", rx, mask=mask, value_group=0, confidence="high")]


def _other_guard(m) -> bool:
    return is_own_token(m.group(2), part=True)


def _other_mask(m):
    return m.group(1) + mint.tag("RESOURCE", m.group(2).lower())


def other_domain_rules() -> list:
    """Host of a link with a scheme / www. / a path → RESOURCE_… (seed — host in lower case)."""
    rx = re.compile(r"(?i)((?:[a-z][a-z0-9+.-]{1,10}://)(?:[^\s/@\"'<>\\]{1,64}@)?)(" + _HOST
                    + r")(?![\w-])")
    rx_www = re.compile(r"(?i)" + _LEFT + r"()(www\." + _HOST + r")(?![\w-])")
    rx_path = re.compile(r"(?i)" + _LEFT + r"()(" + _HOST + r")(?=/[\w%~-])")
    return [
        Rule("link:host", rx, mask=_other_mask, value_group=2, guard=_other_guard,
             confidence="high", needs="://"),
        Rule("link:host:www", rx_www, mask=_other_mask, value_group=2, guard=_other_guard,
             confidence="high", needs="www."),
        Rule("link:host:path", rx_path, mask=_other_mask, value_group=2, guard=_other_guard,
             confidence="high", needs="/"),
    ]


def rules(cfg_links) -> tuple:
    """(first rules — replacement with RESOURCE_REDACTED, rules in the pack slot — other domains)."""
    first = delete_rules(cfg_links.delete_hosts, cfg_links.delete_host_prefixes)
    pack = other_domain_rules() if cfg_links.mask_other_domains else []
    return first, pack


def host_residual_rx(hosts, prefixes):
    """Regex for the control gate: a mention of a host from delete_hosts (any subdomain) or a prefixed host."""
    hosts = sorted({h.strip().lower().lstrip(".") for h in hosts if h and h.strip()})
    prefixes = sorted({p.strip().lower().rstrip(".") for p in prefixes if p and p.strip()})
    alts = []
    if hosts:
        alts.append(r"(?:" + _LABEL + r"\.)*(?:" + alt_longest_first(hosts) + r")")
    for p in prefixes:
        alts.append(re.escape(p) + r"(?:\." + _LABEL + r")+")
    if not alts:
        return None
    return re.compile(r"(?i)(?<![\w.-])(?:" + "|".join(alts) + r")" + _HOST_END)


__all__ = ["REDACTED", "P", "rules", "delete_rules", "other_domain_rules", "host_residual_rx"]
