"""Contact detectors: phone / email / IPv4 / MAC / cookie-device-signature.

Also: values of the Cookie header (curl -b / "Cookie:"), id headers of curl -H
("DeviceID: …"), a 10-digit phone after a label, a login without a colon and a Cyrillic
login with digits, a nickname after "/@" in a platform link, Dockerfile MAINTAINER,
a VK page id as the word "id<digits>" next to a VK label; a VK object with the owner's token is not a nickname.
"""
from __future__ import annotations

import re
from urllib.parse import unquote

from ..engine import mint, placeholders as P
from ..engine.boundary import OWN_TOKEN_RE, R, _ESC, _MINT_PREFIXES
from ..engine.context import PHONE_CTX
from ..engine.rule import Rule, keep_line_breaks, no_guard
from ..engine import validators as V

# JSONL-hardened left boundaries. In raw JSONL a newline is the two chars
# ``\`` + ``n`` (see boundary.L): a value at line start sits right after that
# escape, so a plain lookbehind MISSES it — worse, a match that starts ON the
# escape's own ``n`` swallows it and leaves an invalid ``\E…`` escape in the
# line. The fallback lookbehind therefore also rejects a preceding backslash.
_L = r"(?:(?<=\\n)|(?<![A-Za-z0-9\\]))"
# «_» is a boundary for _L, so the digits of our own token starting with 89/79
# plus "-NNNN" (a year, the glue «ORG_8916168-2024») read as the phone «8 916 168-20-24» —
# the token was destroyed («ORG_PHONE_REDACTED»), the link to the entity lost. A token boundary: the number
# does not start right after the head of a minted token "<PREFIX>_" (all families of boundary._MINT_PREFIXES;
# a fixed-width lookbehind for each). Not a rule guard: a guard writes "guard: ok" into the
# why block of dry_report.json for every number.
_PHONE_L = _L + "".join(f"(?<!{p}_)" for p in sorted(_MINT_PREFIXES))
# email: same idiom with the wider address-glue fallback (\w / dot / @ / + / -),
# plus a right guard so the TLD can't weld onto a following word/number.
# "+" MUST be in the lookbehind: the body class [\w.+-] contains it, and without it
# every "+" inside a slash-free alphanumeric run reopened a candidate start
# whose greedy scan [\w.+-]{1,} ran to the end of the run — O(runlen) per "+" =>
# quadratic on base64 blobs ("+"≈1/64). No FN risk: a legitimate email
# does not start right after "+" (interior "user+tag@" pluses are eaten by
# the body class from the real start character).
_EMAIL_L = r"(?:(?<=\\n)|(?<![\w.@+\\-]))"
_EMAIL_R = r"(?![A-Za-z0-9-])"
# ipv4: fallback also rejects a preceding dot (never match a fragment of a
# longer dotted run). The right boundary MIRRORS the left one: besides letter/digit
# we also reject a following ".digit" — otherwise a 5-part build/version run
# "17.25.19.14.3" would match the 4-octet prefix and leave a ".3" stub.
_IP_L = r"(?:(?<=\\n)|(?<![A-Za-z0-9.\\]))"
_IP_R = r"(?![A-Za-z0-9]|\.\d)"

# --- international phone ------------------------------------------------------
_PHONE_SEP = re.compile(r"(?:[^\S\r\n]|[\-()])+")


def _phone_intl_prefix(v: str) -> int:
    """Length of the longest prefix of ``v`` at digit group boundaries that
    V.phone_intl_ok accepts as a number (0 — none). First the whole match,
    then without the last group, etc.: a whole number matters more than its valid
    piece. Without the phonenumbers package always 0 — the rule is silent; the
    validator rejects +1/+7 candidates by the first digit BEFORE the lazy import."""
    if V.phone_intl_ok(v):
        return len(v)
    for sep in reversed(list(_PHONE_SEP.finditer(v))):
        cand = v[:sep.start()]
        if V.phone_intl_ok(cand):
            return sep.start()
    return 0


def _phone_intl_mask(m) -> str:
    v = m.group(1)
    return P.PHONE + v[_phone_intl_prefix(v):]


# --- social network profile ---------------------------------------------------
_SOCIAL_HOST = (r"(?:vk\.com|vkontakte\.ru|t\.me|telegram\.me|instagram\.com|"
                r"facebook\.com|fb\.com|twitter\.com|x\.com|ok\.ru|tiktok\.com|"
                r"behance\.net|dribbble\.com)")
_SOCIAL_NEEDS = re.compile(r"(?i)" + _SOCIAL_HOST + r"/")
# The first path segment that is NOT a nickname: service sections and platform objects
# (vk.com/feed, t.me/joinchat/…, instagram.com/p/<post>, facebook.com/groups/…,
# x.com/intent/…). VK objects with a numeric id (wall-1_2, club123, photo1_2) —
# by regex; "id12345" is a numeric USER id, that is a nickname and is masked.
_SOCIAL_SERVICE = frozenset("""
feed im wall share share.php away away.php joinchat addstickers addlist addemoji
addtheme proxy socks iv reel reels explore stories tv groups pages profile.php
events watch sharer sharer.php dialog intent hashtag search home login signup
about help settings dev apps docs doc topic market video videos audio photo
photos album albums clip clips music games app blog press legal terms privacy
policy business ads developers support download tos jobs careers direct
accounts web plugins l.php login.php story.php permalink.php photo.php video.php
media status statuses messages notifications compose oauth authorize method api
contact setlanguage boost embed widget widgets tag tags discover trending
live marketplace gaming fundraisers saved bookmarks lists communities premium
artist artists gallery galleries shots static idnone audios podcasts podcast
invite collections playlist playlists stickers sticker channel channels
""".split())
# An object with a numeric id (post/photo/video/document…) is not a nickname. Communities
# (club/public/event + id) are NOT included: that is an organization page, it
# is masked as a company nickname.
_SOCIAL_OBJ = re.compile(
    r"(?i)^(?:wall|album|video|photo|topic|doc|app|market|audio|clip|story|"
    r"note|page|board|poll|write|article|product)-?\d")
_HANDLE_TOKEN = re.compile(r"^(?:[A-Z][A-Z0-9_]*_(?:\d+|REDACTED))$")
# the same object after discord_ids.vk_object_owner — "wall-USER_ID_N_<number>" (the owner is already
# a token; measured on raw tg: without this social_profile on the next pass folded 54 such
# links into HANDLE whole)
_SOCIAL_OBJ_TOK = re.compile(
    r"^(?i:wall|album|video|photo|topic|doc|clip|audio|market|product|board|note|poll|story|"
    r"podcast)s?-USER_ID_\d")


# service values after "login:" — not a person's login (nothing to mask)
_OS_USER_STOP = frozenset("public default shared administrator admin root user guest runner ubuntu "
                          "ec2-user vagrant www-data nobody desktop appdata all".split())
_LOGIN_STOP = frozenset("admin administrator root user test guest demo null none true false "
                        "default anonymous example".split())


def _bitrix_cookie_mask(m) -> str:
    # BITRIX_SM_LOGIN — the user's login (the same seed as for a nickname); other BITRIX_SM_* — guest/
    # session/cart ids → COOKIE
    if m.group(2).upper() == "LOGIN":
        return m.group(1) + mint.tag("HANDLE", unquote(m.group(3)).lower())
    return m.group(1) + P.COOKIE


def _social_guard(m) -> bool:
    """Guard (True => do not mask): the first path segment is a service section
    or a platform object, or already a token."""
    slug = m.group(2)
    low = slug.lower()
    if (low in _SOCIAL_SERVICE or low.endswith(".php") or _SOCIAL_OBJ.match(slug)
            or _SOCIAL_OBJ_TOK.match(slug)):
        return True
    return bool(_HANDLE_TOKEN.match(slug))


# --- cookies/curl headers, logins, nicknames in links -------------------------------------------
_COOKIE_ATTR = frozenset("path domain expires max-age samesite secure httponly priority version "
                         "comment partitioned".split())
_SESS_NAME = re.compile(r"(?i)(?:sess|token|auth|csrf|xsrf|secret|key|pass|sid$)")
_CK_PAIR = re.compile(r"(^|;)([^\S\n]*)([^=;\s]{1,128})=([^;]*)")


def _cookie_idlike(name: str, v: str) -> bool:
    """A cookie value is an identifier/session: ≥6 chars with a digit or ≥16 chars; Set-Cookie
    attributes, flags/languages/themes (short, one lowercase word), our token — no."""
    if name.lower() in _COOKIE_ATTR or not v or OWN_TOKEN_RE.search(v) or "_REDACTED" in v:
        return False
    if len(v) < 6 or re.fullmatch(r"[a-z]+", v):
        return False
    return re.search(r"\d", v) is not None or len(v) >= 16


def _cookie_header_mask(m) -> str:
    """Each "name=value" pair of the header: an identifier value → COOKIE_REDACTED, a session/
    token/csrf by name → SECRET_REDACTED; the name and separators stay, as does a cmd.exe "^" escape
    before the closing quote."""
    def fix(pm):
        val = pm.group(4)
        core = val.strip()
        tail = ""
        while core.endswith("^"):
            core, tail = core[:-1], "^" + tail
        if not _cookie_idlike(pm.group(3), core):
            return pm.group(0)
        tok = P.SECRET if _SESS_NAME.search(pm.group(3)) else P.COOKIE
        lead = val[:len(val) - len(val.lstrip())]
        trail = val[len(val.rstrip()):]
        return pm.group(1) + pm.group(2) + pm.group(3) + "=" + lead + tok + tail + trail
    return m.group(1) + _CK_PAIR.sub(fix, m.group(2))


# curl -H headers without PII (agent, formats, cache, request tracing) — left alone
_HDR_SKIP = re.compile(r"(?i)^(?:user-agent|accept(?:-[\w-]+)?|content-[\w-]+|referer|origin|host|"
                       r"connection|cache-control|pragma|sec-[\w-]+|upgrade-insecure-requests|dnt|te|"
                       r"priority|if-[\w-]+|x-requested-with|cookie|authorization|range|"
                       r"(?:x-)?(?:request|trace|correlation|span|parent)-?id)$")
_HDR_SECRET = re.compile(r"(?i)(?:token|secret|signature|passw|api[-_]?key|access[-_]?key|csrf|"
                         r"xsrf|session|auth|bearer)")
_HDR_ID = re.compile(r"(?i)(?:id|uid|guid|uuid|udid|imei|fingerprint)$")


def _curl_header_mask(m) -> str:
    """A curl -H header value: by a secret name (token/auth/session/csrf/api-key…) →
    SECRET, by an identifier name (…Id/…ID/uuid/udid/imei/fingerprint) → DOC_ID (the seed is
    the value: one device id — one token). One "word" ≥6 with a digit (or ≥16)."""
    name, val = m.group(3), m.group(5)
    v = val.strip()
    if (_HDR_SKIP.match(name) or not v or len(v) < 6 or " " in v or OWN_TOKEN_RE.search(v)
            or "_REDACTED" in v or not (re.search(r"\d", v) or len(v) >= 16)):
        return m.group(0)
    if _HDR_SECRET.search(name):
        tok = P.SECRET
    elif _HDR_ID.search(name):
        tok = mint.tag("DOC_ID", v.lower())
    else:
        return m.group(0)
    lead = val[:len(val) - len(val.lstrip())]
    trail = val[len(val.rstrip()):]
    return m.group(1) + name + m.group(4) + lead + tok + trail + m.group(6)


_LOGIN_VERSION = re.compile(r"(?i)v\d+(?:[._]\d+)*(?:[-+][\w.]+)?")
_LOGIN_LETTER_DIGITS = re.compile(r"[A-Za-z](\d+)")


def _login_spaced_guard(m) -> bool:
    """True => do not mask. A login without a colon: a value with a digit, not a service word, not
    our token; not a version ("login v2.0 flow") and not "a letter + up to 5 digits" — a codec/model
    («ник h264»). The form "a letter + ≥6 digits" is a login (a123456789 in audit findings)."""
    v = m.group(2)
    if not re.search(r"\d", v) or v.lower() in _LOGIN_STOP or OWN_TOKEN_RE.search(v) is not None:
        return True
    if _LOGIN_VERSION.fullmatch(v):
        return True
    ld = _LOGIN_LETTER_DIGITS.fullmatch(v)
    return ld is not None and len(ld.group(1)) < 6


def _phone10_ok(v: str) -> bool:
    """10 digits after a phone label: a Russian number (V.phone_ru_ok) or a national format with
    a leading 0 ("0XX XXX XX XX": Ukraine etc.). The label is the evidence, so only explicit ones
    count as dummies: ≤2 distinct digits, ≥7 zeros, "0123456789"."""
    if v.startswith("0"):
        return len(set(v)) > 2 and v.count("0") < 7 and v not in "01234567890"
    return _phone_pretty_ok(v)


# V.phone_ru_ok treats a number with ≤4 distinct digits or ≥5 zeros as a dummy —
# "pretty" (often public) numbers «8 999 111-11-11», «8 (495) 100-00-01», «моб: 9169000090»
# stayed open even with a label. With a phone label (PHONE_CTX nearby — phone_ru_pretty;
# a label right next to it — phone_labeled10) a dummy is only an explicit one: ≤2 distinct digits or ≥7 zeros; the code
# (7/8 + 3/4/8/9, 10 digits — 3/4/8/9) is checked as before. Without a label — the old phone_ru_ok.


def _phone_pretty_ok(raw: str) -> bool:
    d = "".join(c for c in raw if c.isdigit())
    if len(d) == 11:
        code = d[0] in "78" and d[1] in "3489"
    elif len(d) == 10:
        code = d[0] in "3489"
    else:
        return False
    return code and len(set(d)) > 2 and d.count("0") < 7


# (owner: a rule for all clients): a compound login "XXX_First_Last" — a prefix
# in Latin capitals (often a company abbreviation) and 2–3 capitalised words joined by "_" (Latin or
# Cyrillic) is an account, not a mention of a name: → HANDLE whole (like logins, nicknames, profiles and
# OS users in this module; the seed is lowercase, one login — one token). Whole and BEFORE the packs: a pack
# literal (first/last name) would otherwise cut it at "_" ("HANDLE_N_god"). A code constant of English words
# ("UI_Button_Primary", "DB_Connection_Pool") is not a login: every word is a dictionary English word and none
# is a name (lexicon: en_words without en_proper).
_COMPOUND_LOGIN = re.compile(r"(?:(?<=\\[ntr])|(?<![\w@./\\-]))"
                             r"([A-Z]{2,6}(?:_(?:[A-Z][a-z]{1,30}|[А-ЯЁ][а-яё]{1,30})){2,3})"
                             r"(?![\w@-]|\.\w)")


def _compound_login_guard(m) -> bool:
    """True => do not mask: our token, a code constant of English words."""
    v = m.group(1)
    if OWN_TOKEN_RE.search(v):
        return True
    from ..engine import lexicon
    words, proper = lexicon._en("words"), lexicon._en("proper")
    parts = [p.lower() for p in v.split("_")[1:]]
    return all(p in words and p not in proper for p in parts)


# email (self-evident) — keep it simple and high-precision (the email rule in RULES)
_EMAIL_RX = re.compile(_EMAIL_L + r"([A-Za-z0-9][\w.+-]{1,}@[A-Za-z0-9][\w.-]*\.[A-Za-z]{2,})"
                       + _EMAIL_R)
# email took only an ASCII address starting with a letter/digit and forbade \w
# on the left — «ИВАН.ПЕТРОВ@MAIL.RU», «_ivan@mail.ru», «иван@почта.рф», «ivan@mail.xn--p1ai»,
# «почтуivan.petrov@gmail.com» stayed open (the domain went to ORG), the "_" guard at the start was
# dead. email_wide (after email, picks up what it did not take): the left boundary is an ASCII class,
# like encoding._EMAIL; the local part is Unicode \w, "_" at the start; the zone is Latin, Cyrillic
# or xn--; lengths are bounded (linear time).
_EMAIL_WIDE = re.compile(
    r"(?:(?<=\\n)|(?<![A-Za-z0-9_.@+\\-]))"
    r"(\w[\w.+-]{0,63}@[^\W_][\w-]{0,62}(?:\.[^\W_][\w-]{0,62}){0,8}?"
    r"\.(?:xn--[A-Za-z0-9-]{2,59}(?![A-Za-z0-9-])|[A-Za-z]{2,24}(?![A-Za-z0-9-])"
    r"|[А-ЯЁа-яё]{2,24}(?![А-ЯЁа-яё0-9-])))")


def _email_wide_guard(m) -> bool:
    """True => do not mask: already a mask; an address without a letter in the local part;
    an address that the email rule takes with the same span — it is email's (email_wide does not duplicate it: a failure
    or a guard of email is not silently replaced by another rule)."""
    v = m.group(1)
    if "_REDACTED" in v or not any(c.isalpha() for c in v.split("@", 1)[0]):
        return True
    old = _EMAIL_RX.match(m.string, m.start(1))
    return old is not None and old.group(1) == v


# In a URL-encoded link («ref=https%3A%2F%2Fvk.com%2Fivan_petrov_88»,
# «phone=%2B79001234567») a word boundary sees a letter/digit of the escape («%2F», «%2B») — the nickname and the phone
# did not match, and domain started at «2F» and broke the escape («%RESOURCE_…»). An allowing branch
# (?<=%XX), as in encoding.encoded_email_pct: social_profile_pct (path separator %2F),
# phone_ru_pct (after %2B, separators also %20), orgs.domain_pct (a host right after %XX, BEFORE
# domain); domain no longer starts right after "%".
_PCT_SP = re.compile(r"%20")


# a nickname in a link path after "/@" (YouTube/Medium/Threads/Dzen/vc.ru/teletype…)
_AT_HOSTS = (r"(?:(?:www\.|m\.)?youtube\.com|medium\.com|threads\.net|dzen\.ru|zen\.yandex\.ru|"
             r"vc\.ru|teletype\.in|tenchat\.ru|pikabu\.ru|substack\.com|mastodon\.social)")


RULES: list[Rule] = [
    # email (self-evident) — keep it simple and high-precision
    Rule("email", _EMAIL_RX,
         mask=lambda m: P.EMAIL, value_group=1, confidence="high",
         guard=lambda m: "_REDACTED" in m.group(1) or "_" == m.group(1)[0],
         needs="@"),                          # no address without "@"
    # Cyrillic / IDN / "_" at the start / right next to a Russian word (see _EMAIL_WIDE)
    Rule("email_wide", _EMAIL_WIDE, value_group=1, mask=lambda m: P.EMAIL, confidence="high",
         guard=_email_wide_guard, needs="@"),

    # RU phone — normalize + range-check via validator (kills timestamps/dummies).
    # Letter-aware boundaries: a digit run glued to letters is an identifier,
    # not a phone (no `orderPHONE_REDACTEDx` welds).
    # Separators are \s, a number can be broken by a line break («+7 916\n123-45-67»);
    # the mask keeps the line breaks of the match (keep_line_breaks), otherwise the safety net
    # skipped such a match whole (tg: 20 numbers stayed open)
    Rule("phone_ru",
         re.compile(_PHONE_L + r"((?:\+?7|8)[\s\-()]{0,3}\d{3}[\s\-()]{0,3}\d{3}[\s\-()]{0,3}\d{2}[\s\-()]{0,3}\d{2})(?:\.0)?" + R),
         mask=lambda m: keep_line_breaks(P.PHONE, m.group(0)), value_group=1,
         validator=V.phone_ru_ok, confidence="high",
         # the digit skeleton of the number without the code/boundaries is a mandatory part of the match
         needs=re.compile(r"\d{3}[\s\-()]{0,3}\d{3}[\s\-()]{0,3}\d{2}[\s\-()]{0,3}\d{2}")),
    # the same form, a "pretty" number — with a phone label nearby (see _phone_pretty_ok)
    Rule("phone_ru_pretty",
         re.compile(_PHONE_L + r"((?:\+?7|8)[\s\-()]{0,3}\d{3}[\s\-()]{0,3}\d{3}[\s\-()]{0,3}\d{2}[\s\-()]{0,3}\d{2})(?:\.0)?" + R),
         mask=lambda m: keep_line_breaks(P.PHONE, m.group(0)), value_group=1,
         validator=_phone_pretty_ok, requires_context=PHONE_CTX, confidence="context",
         guard=no_guard,
         needs=re.compile(r"\d{3}[\s\-()]{0,3}\d{3}[\s\-()]{0,3}\d{2}[\s\-()]{0,3}\d{2}")),
    # a URL-encoded number after «%2B» («phone=%2B79001234567», «%2B7%20916%20…»)
    Rule("phone_ru_pct",
         re.compile(r"(?i)(?<=%2B)(7(?:[\-()]|%20){0,3}\d{3}(?:[\-()]|%20){0,3}\d{3}"
                    r"(?:[\-()]|%20){0,3}\d{2}(?:[\-()]|%20){0,3}\d{2})(?![0-9A-Za-z])"),
         mask=lambda m: P.PHONE, value_group=1, confidence="high",
         validator=lambda v: V.phone_ru_ok(_PCT_SP.sub(" ", v)),
         guard=no_guard,
         needs=re.compile(r"(?i)%2B7")),
    # CIS: a country code from the table + the national part of exactly the table
    # length (V.phone_cis_ok) — evidence without a context word → high;
    # phone_intl below stays the context tier for other codes (+1 deliberately
    # not in the table: amounts/versions). +7 7xx is Kazakhstan: phone_ru above
    # rejects it (the second digit is not 3/4/8/9), the number reaches here untouched.
    Rule("phone_cis",
         re.compile(_L + r"(\+(?:7|37[345]|380|99[2-8])(?:[\s\-()]{0,3}\d){8,10})" + R),
         mask=lambda m: keep_line_breaks(P.PHONE, m.group(0)), value_group=1,
         validator=V.phone_cis_ok, confidence="high"),
    # world: a country code + a national part of variable length (6–12 digits,
    # separators as in phone_cis, no line break) — the evidence comes from
    # V.phone_intl_ok (libphonenumber via the optional phonenumbers package:
    # length and prefix by country metadata) → high without a context word. No
    # package → the validator is False, the number goes to phone_intl below (context,
    # format regex, as before). A greedy match may grab an adjacent digit group
    # («+44 20 7946 0958 12», «… 0958 (2 шт)»): the validator/mask take
    # the longest valid prefix at group boundaries (_phone_intl_prefix,
    # like _iban_prefix), the tail goes out as is. A lazy quantifier does not
    # work: it stopped at the FIRST space («+44 20 7946»). +1/+7
    # the validator rejects itself (phone_ru/phone_cis above; +1 — context).
    Rule("phone_intl_valid",
         re.compile(_L + r"(\+\d{1,3}(?:(?:[^\S\r\n]|[\-()]){0,3}\d){6,12})" + R),
         mask=_phone_intl_mask, value_group=1,
         validator=lambda v: _phone_intl_prefix(v) > 0,
         confidence="high", needs=re.compile(r"\+\d")),   # a country code "+digit"
    # intl phone — explicit country code required
    Rule("phone_intl",
         re.compile(_L + r"(\+\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})" + R),
         mask=lambda m: keep_line_breaks(P.PHONE, m.group(0)), value_group=1,
         confidence="context",
         needs=re.compile(r"\+\d")),          # a country code starts with "+digit"

    # public IPv4 only (validator drops private/version/timecode); letter-glued
    # dotted runs (`v81.19.70.3`) are version-like identifiers → skip.
    Rule("ipv4",
         re.compile(_IP_L + r"((?:\d{1,3}\.){3}\d{1,3})" + _IP_R),
         mask=lambda m: P.IP, value_group=1, validator=V.public_ipv4, confidence="high",
         needs=re.compile(r"\d\.\d{1,3}\.\d{1,3}\.\d")),   # three dots between digits

    # MAC address (structural)
    Rule("mac",
         re.compile(r"(?<![0-9A-Fa-f:])((?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})(?![0-9A-Fa-f:])"),
         mask=lambda m: P.MAC, value_group=1, confidence="high",
         guard=lambda m: m.group(1).lower() in ("ff:ff:ff:ff:ff:ff", "00:00:00:00:00:00"),
         needs=re.compile(r"(?:[0-9A-Fa-f]{2}:){3}[0-9A-Fa-f]{2}")),   # 5 pairs ⊇ 3

    # cookie / device signature — keep the param name, redact the value
    Rule("cookie_dvs",
         re.compile(r"((?:_dvs|dSesn|bxmaker|__af|sbjs_[\w]+|sessionid|jsessionid|phpsessid|csrftoken)=)([A-Za-z0-9%+/=_.:~-]{12,400})"),
         mask=lambda m: m.group(1) + P.COOKIE, value_group=2, confidence="context"),

    # «BITRIX_SM_LOGIN=<login>» in cookies from logs/headers:
    # the BITRIX_SM_LOGIN value → HANDLE, other BITRIX_SM_* (GUEST_ID, SALE_UID, UIDH…) → COOKIE.
    Rule("bitrix_sm_cookie",
         re.compile(r"(?i)((?<![A-Za-z0-9_])BITRIX_SM_([A-Z0-9_]{2,40})=)([^;\s&\"'<>\\]{1,200})"),
         mask=_bitrix_cookie_mask, value_group=3, confidence="context",
         guard=lambda m: "_REDACTED" in m.group(3) or bool(_HANDLE_TOKEN.match(m.group(3))),
         needs=re.compile(r"(?i)bitrix_sm_")),
    # A curl from DevTools in a message:
    # «-b '_ym_uid=…; _ga=GA1.1.…; xxx_session=…'», «Cookie: …», cmd escapes «^"…^"»: the value
    # of each identifier cookie of the header (analytics, session, device id) — COOKIE/SECRET.
    # The header body runs to a quote, a line break or "\" (a JSON escape / line continuation).
    Rule("cookie_header",
         re.compile(r"(?i)((?<![\w-])(?:set-)?cookie[^\S\n]{0,3}:[^\S\n]{0,3}|"
                    r"(?<![\w-])(?:-b|--cookie)[^\S\n]+\^?[\"']\^?)([^\n\"'\\]{3,16000})"),
         mask=_cookie_header_mask, value_group=2, confidence="context",
         needs=re.compile(r"(?i)(?:cookie|-b[^\S\n])")),
    # curl -H «DeviceID: <hex>», «…SessionId:
    # …»: the value of an identifier header → DOC_ID, of a secret header → SECRET
    Rule("curl_header_id",
         re.compile(r"""(?i)((?<![\w-])(?:-H|--header)[^\S\n]+\^?(["'])\^?)([A-Za-z][\w-]{1,60})"""
                    r"""([^\S\n]*:[^\S\n]*)([^"'\n\\]{1,4096}?)(\^?\2)"""),
         mask=_curl_header_mask, value_group=5, confidence="context",
         needs=re.compile(r"(?i)(?:-H|--header)[^\S\n]")),

    # «login: ivan_petrov», «логин — ipetrov» in chats:
    # the value after a login/username label → HANDLE (the same token as for @nickname and t.me/nickname).
    # A mail address (with "@") does not go here: email takes it. Service «admin/root/test…» are left alone.
    Rule("login_label",
         # a dash only with spaces around: «login-form»/«login-page» is a compound word, not a pair;
         # a dot/hyphen at the end is phrase punctuation, not part of the login («login: ivan.petrov.»)
         re.compile(r"(?i)((?<![A-Za-z0-9_])(?:login|логин|username|user[_\s-]?name|юзернейм|никнейм)"
                    r"(?:[^\S\n]{0,3}[:=][^\S\n]{0,3}|[^\S\n]{1,3}[—–-][^\S\n]{1,3}))"
                    r"([A-Za-z0-9][A-Za-z0-9_.-]{1,62}[A-Za-z0-9_])(?![A-Za-z0-9_@]|[.-][A-Za-z0-9_])"),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="context",
         guard=lambda m: m.group(2).lower() in _LOGIN_STOP or bool(_HANDLE_TOKEN.match(m.group(2)))),
    # «Логин a123456789», «Никнейм
    # nickname12» without a colon; «логин: ЖЖЖЖ12345» in Cyrillic: a label followed by a space — the value
    # only with a digit (an ordinary word after «логин» is not a login); Cyrillic — only with a digit
    Rule("login_label_spaced",
         re.compile(r"(?i)((?<![\w])(?:login|логин|username|юзернейм|никнейм|ник)[^\S\n]{1,3})"
                    r"([A-Za-z][A-Za-z0-9_.-]{2,62}[A-Za-z0-9_])(?![\w@]|[.-]\w)"),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="context",
         # our token inside the value («Логин ORG_N_xxx» after a pack) — not re-minted; a version and
         # "a letter + up to 5 digits" (h264) — not a login
         guard=_login_spaced_guard),
    # a dot/hyphen at the end is phrase punctuation, not part of the login («логин: ЖЖЖЖ12345.» gave
    # a seed with a dot and a foreign token), as in login_label above; length 3–63 as before
    Rule("login_label_cyr",
         re.compile(r"(?i)((?<![\w])(?:login|логин|username|юзернейм|никнейм|ник)[^\S\n]{0,3}[:=]"
                    r"[^\S\n]{0,3})([A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_.-]{1,61}"
                    r"[A-Za-zА-Яа-яЁё0-9_])(?![\w@])"),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="context",
         guard=lambda m: (not (re.search(r"[А-Яа-яЁё]", m.group(2)) and re.search(r"\d", m.group(2)))
                          or OWN_TOKEN_RE.search(m.group(2)) is not None)),

    # a phone without +7/8 — «916 123 45 67», «(495) 123-45-67»: the grouping
    # 3-3-2-2 with separators is a phone form; code 3/4/8/9 and not a "dummy" — V.phone_ru_ok
    Rule("phone_ru_local",
         re.compile(_L + r"((?:\(\d{3}\)[ ]?|\d{3}[ -])\d{3}[ -]\d{2}[ -]\d{2})" + R),
         mask=lambda m: P.PHONE, value_group=1, validator=V.phone_ru_ok, confidence="context",
         needs=re.compile(r"\d{3}[ -]\d{2}[ -]\d{2}")),     # the number skeleton is mandatory
    # JSON in a message — {"phone": "9001234567"}: 10 digits of a mobile without
    # a country code, the whole string value
    Rule("phone_ru_quoted10",
         re.compile(r'(?<=")(9\d{9})(?=")'),
         mask=lambda m: P.PHONE, value_group=1, validator=V.phone_ru_ok, confidence="context",
         needs=re.compile(r'"9\d{9}"')),
    # «телефон:<10 digits>» under a link and «id:»;
    # a number with a leading 0 is a national format outside Russia, phone_ru does not see it: a phone label is
    # the evidence; 10 digits — a Russian number (V.phone_ru_ok) or a non-dummy «0XX XXX XX XX»;
    # to the left of the label — also a JSONL escape (boundary._ESC)
    Rule("phone_labeled10",
         re.compile(r"(?i)((?:" + _ESC + r"|(?<![\w]))(?:телефон\w*|тел|phone|mobile|моб(?:ильный)?|сотовый|whatsapp|"
                    r"ватсап|вотсап|viber|вайбер)\.?[^\S\n]{0,3}[:=-]?[^\S\n]{0,3})([03489]\d{9})(?!\d)"),
         mask=lambda m: m.group(1) + P.PHONE, value_group=2, validator=_phone10_ok,
         confidence="context", needs=re.compile(r"[03489]\d{9}")),
    # an OS user name in a profile path — «C:\Users\<name>\…», «/Users/<name>/»,
    # «/home/<name>/» → HANDLE. Service profiles (Public, Default, root, ubuntu…) are left alone;
    # «/home/» inside a link («site.ru/home/news/») is not an OS path.
    # In raw JSON a single "\" is always the head of an escape
    # («C:\\Users\n----+----»: the separator before Users is doubled, the one after is a single «\n», and the rule
    # took «n----…» for a name, breaking the escape). The separator after Users must have the same
    # multiplicity as the one before it: «C:\Users\nick\» (text) and «C:\\Users\\nick\\» (JSON) — the mask
    # stays, «\\Users\n…» — no.
    Rule("os_user_path",
         re.compile(r"(?i)((?:(?<=[^\\]\\)|(?<=^\\)|(?<![A-Za-z0-9\\]))(?:Users|Documents and Settings)"
                    r"\\(?!\\)|(?<=\\\\)(?:Users|Documents and Settings)\\\\|"
                    r"(?<![\w./-])/(?:Users|home)/)([^\\/:*?\"<>|\s]{2,64})(?=\\|/)"),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="context",
         guard=lambda m: (m.group(2).lower() in _OS_USER_STOP or "_REDACTED" in m.group(2)
                          or bool(_HANDLE_TOKEN.match(m.group(2))))),

    # a compound login "XXX_First_Last" → HANDLE whole (see _COMPOUND_LOGIN)
    Rule("compound_login", _COMPOUND_LOGIN,
         mask=lambda m: mint.tag("HANDLE", m.group(1).lower()),
         value_group=1, confidence="context", guard=_compound_login_guard,
         needs=re.compile(r"[A-Z]{2}_(?:[A-Z][a-z]|[А-ЯЁ][а-яё])")),

    # SSH login -> USER token (username identity), keep the surrounding literal
    Rule("ssh_login", re.compile(r"(\bssh\s+)([a-z][a-z0-9._-]{2,20})(@)"),
         mask=lambda m: m.group(1) + mint.tag("USER", m.group(2)) + m.group(3),
         value_group=2, confidence="context"),

    # --- social network profile: the nickname in a URL path -> HANDLE whole ---
    # «vk.com/<nick>», «t.me/<nick>», «instagram.com/<nick>» in CRM/HR tables and
    # chats were masked by nobody: the domain went under RESOURCE (orgs.domain), and
    # the nickname stayed; then a pack literal (first/last name/roster handle) got
    # INSIDE the nickname at a "_" boundary and cut it in half — «HANDLE_NNN_god»
    # (an audit finding; the fixer picked up another dozen such nicknames). The rule is in
    # contacts — BEFORE orgs (the host is still in place) and before the packs: the nickname goes under a token
    # whole, nothing is left to cut. The seed is the lowercase nickname, the same as for
    # handle_at/handle_label: «@nick» and «t.me/nick» give ONE token.
    # GitHub/GitLab are deliberately not included (owner's decision: OSS handles
    # are not masked).
    Rule("social_profile",
         re.compile(r"(?i)(?<![A-Za-z0-9.-])((?:https?://)?(?:www\.|m\.|mobile\.)?"
                    + _SOCIAL_HOST + r"/(?:s/)?@?)"
                    r"([A-Za-z0-9_][A-Za-z0-9_.-]{1,62}[A-Za-z0-9_])"),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="high", guard=_social_guard,
         needs=_SOCIAL_NEEDS),
    # the same profile URL-encoded — «vk.com%2F<nick>» after «%2F»/«https%3A%2F%2F»
    Rule("social_profile_pct",
         re.compile(r"(?i)(?:(?<=%[0-9A-Fa-f]{2})|(?<![A-Za-z0-9.%-]))((?:https?%3A%2F%2F)?"
                    r"(?:www\.|m\.|mobile\.)?" + _SOCIAL_HOST + r"%2F(?:s%2F)?(?:%40)?)"
                    r"([A-Za-z0-9_][A-Za-z0-9_.-]{1,62}[A-Za-z0-9_])(?![A-Za-z0-9_])"),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="high",
         guard=_social_guard,
         needs=re.compile(r"(?i)" + _SOCIAL_HOST + r"%2F")),
    # «vk_id:» and below it the line «id<digits>»: a VK page id
    # as the word «id123…» with a VK label nearby → HANDLE with the same seed as the link «vk.com/id123…»
    # (social_profile): one page — one token
    Rule("vk_id_word",
         re.compile(r"(?<![\w/.@-])(id\d{5,12})(?![\w-])"),
         mask=lambda m: mint.tag("HANDLE", m.group(1).lower()), value_group=1, confidence="context",
         requires_context=re.compile(r"(?i)(?<![\w])(?:vk[_ ]?id|vk\.com|вконтакте|вк[_ ]?id)"),
         window=160, needs=re.compile(r"id\d{5}")),
    # «youtube.com/@nick», «medium.com/@nick»,
    # «teletype.in/@nick»: a nickname after "/@" in a link of a platform with @-nicknames → HANDLE (the seed is lowercase,
    # the same as for @nick in text). Hosts are an explicit list: "/@scope/pkg" of npm/CDN is not a nickname.
    Rule("url_at_handle",
         re.compile(r"(?i)((?<![\w.-])(?:https?://)?" + _AT_HOSTS + r"/@)"
                    r"([A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9_])?)"),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="high", guard=lambda m: bool(OWN_TOKEN_RE.search(m.group(2))),
         needs=re.compile(r"(?i)/@\w")),
    # A Dockerfile in a message «MAINTAINER <nick>»,
    # «LABEL maintainer="<nick>"»: one "word" after the label → HANDLE; email takes mail addresses.
    # A dot/hyphen at the end is punctuation, not part of the nickname («MAINTAINER nick.»,
    # «maintainer="nick."» gave a seed with a dot and a foreign token), as in login_label; length 2–63 as before
    Rule("maintainer_label",
         re.compile(r"(?im)((?:^[^\S\n]*MAINTAINER[^\S\n]+)|(?<![\w-])maintainer[^\S\n]*=[^\S\n]*\"?)"
                    r"([A-Za-z0-9_][A-Za-z0-9_.-]{0,61}[A-Za-z0-9_])(?=[.-]?[^\S\n]*(?:$|[\"<]))"),
         mask=lambda m: m.group(1) + mint.tag("HANDLE", m.group(2).lower()),
         value_group=2, confidence="context",
         guard=lambda m: bool(OWN_TOKEN_RE.search(m.group(2))) or m.group(2).lower() in _LOGIN_STOP,
         needs=re.compile(r"(?i)maintainer")),
]
