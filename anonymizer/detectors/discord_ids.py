"""Discord snowflake ids: raw user ids → USER_ID tokens (default-ON).

A release audit confirmed a CRITICAL hole: "author.id must be
USER_ID_###, NOT a raw snowflake" — while snowflake masking lived only in
the opt-in linking detector (HMAC pseudonyms that keep the digit form).
The default release shipped with real Discord ids and inline mentions
<@1234567890…> in content.

Three rules (default-ON, minted with mint.tag("USER_ID", digits) — the same
id in a field and in a mention gives ONE token, linking within the corpus stays intact):
  * discord_snowflake_field — JSON fields of the id family with a value of 17-19
    digits (id/author_id/user_id/reply_to/…); the unquoted variant
    is wrapped in quotes ("id": 175… → "id": "USER_ID_1234") so that
    the JSONL stays valid for the structural gate;
  * discord_mention_array — arrays "mentions"/"user_ids": [id, id, …];
  * discord_mention — <@id> / <@!id> in text: the brackets and @ stay, the digits
    are minted.

The boundaries are deliberately strict: 17-19 digits do not overlap with checksummed
INN 10/12, OGRN 13/15, card 16, account 20 and epoch timestamps 10/13;
created_at is not in the key set. channel_id/guild_id/thread_id
remain the zone of opt-in linking (its form is pinned by tests). Idempotent
by construction: the value must start with a digit, USER_ID is in
TOKEN_AHEAD/MINT_FRAG_RE.

Numeric ids and UUIDs in the TEXT of messages (Telegram audit): an id with an
id-family label, an id in brackets after a nickname (a bot greeting template), ouid in Google
links, a number as a whole message/line, a number of 11–13 digits in text, a UUID next to
a label or as a cookie value. Millisecond timestamps and numbers inside links are left alone;
a bare JSON number gets a quoted token (the document stays valid).

Also (Telegram re-run audit): an id in square brackets after a name (a bot service
message "Name [id] …"), a bare label "ID: <7–10 digits>" capitalised (lowercase "id:" is
a code field, see labeled_user_id), a negative Bot API chat id, links with ids: t.me/c/<channel>,
t.me start parameters with a referrer id, the owner of a VK object, a calendar event and layer,
a Zoom/Google Meet meeting (link and labelled number), an hh resume, a foreign passport as a line, "(<name>, id)!"
without a nickname, a Facebook id in a link parameter, a labelled Account ID. The link host is also recognised by
its RESOURCE token (the next round runs over the output of the previous one).
"""
from __future__ import annotations

import re

from ..engine import mint
from ..engine import placeholders as P
from ..engine import validators as V
from ..engine.boundary import JSON_SAFE as _JS, OWN_TOKEN_RE
from ..engine.rule import Rule
from ..engine.vocab import TECH_TERMS
from .secrets import host_token

_KEYS = (r"(?:id|author_id|user_id|member_id|owner_id|recipient_id|"
         r"reply_to|in_reply_to|message_id|referenced_message_id|"
         r"webhook_id|interaction_id)")

# items of mention arrays: "123…" or bare 123… inside [ ]
_ARR_ITEM = re.compile(r'(?<![0-9A-Za-z_."])(")?(\d{17,19})(?(1)")'
                       r'(?![0-9A-Za-z_."])')


def _mask_field(m) -> str:
    # quotes are normalised: both "id": 175… and "id": "175…" → "USER_ID_x"
    return f'{m.group(1)}"{mint.tag("USER_ID", m.group(3))}"'


# --- numeric IDs in message text (Telegram audit: critical USER_ID 134, DOC_ID 38) ---
# An id-family label before the number: «user_id: 5123456789», «uid=…», «chat_id …», «айди 123…».
# A bare "id" is not included (task/order/row id is too generic), only with a qualifier.
_ID_LABEL = (r"(?:user[_\s-]?id|userid|uid|chat[_\s-]?id|from[_\s-]?id|peer[_\s-]?id|"
             r"sender[_\s-]?id|owner[_\s-]?id|tg[_\s-]?id|telegram[_\s-]?id|"
             r"айди|юзер[_\s-]?айди|ид[^\S\n]+пользователя|id[^\S\n]+пользователя)")
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
# to the right of a standalone number — units/currency: that is an amount or a quantity, not an identifier
_UNIT_RIGHT = re.compile(r"(?i)[^\S\n]{0,2}(?:руб|₽|\$|€|usd|rub|eur|коп|тыс|млн|млрд|byte|байт|"
                         r"кб|мб|гб|kb|mb|gb|ms|мс|сек|sec|rows|строк|шт|%)")
_MONEY_LEFT = re.compile(r"(?i)(?:\$|€|₽|руб\.?|сумм\w*|стоимост\w*|цен\w*|бюджет\w*|оплат\w*|итого)"
                         r"[^\S\n]{0,3}[:=]?[^\S\n]{0,3}$")


def _owned_elsewhere(d: str) -> bool:
    """A number masked by a dedicated detector with its own semantics: a Russian phone
    (contacts.phone_ru), a valid INN 10/12 and OGRN 13 (financial) — not intercepted."""
    if len(d) == 11 and d[0] in "78" and V.phone_ru_ok(d):
        return True
    if len(d) in (10, 12) and V.inn_ok(d):
        return True
    return len(d) == 13 and V.ogrn_ok(d)


def _epoch_ms(d: str) -> bool:
    """13 digits in the millisecond timestamp range (2001–2036): not a document number —
    module contract: epoch timestamps are left alone (tests/test_discord_ids: negatives)."""
    return len(d) == 13 and 1_000_000_000_000 <= int(d) <= 2_100_000_000_000


_JSON_ARR_OPEN = re.compile(r'\[[\s\d.,"+-]*$')


def _qtok(tok: str, s: str, a: int, b: int) -> str:
    """A token in place of a BARE JSON number — quoted: .json outside the structural profile goes
    to the rules as raw text, and an unquoted token would make the document invalid (the same trick as
    discord_snowflake_field). A bare number is the value «"key": 123» on the same line, an element
    of an inline array «[1, 2, 123]» or an element of a "pretty" array: the number is alone on its line (with
    or without a comma), and the line above ends with "," or "["."""
    ls = s.rfind("\n", 0, a) + 1
    le = s.find("\n", b)
    le = len(s) if le < 0 else le
    left = s[max(ls, a - 5000):a].rstrip()
    right_line = s[b:le].strip()
    r1 = right_line[:1]
    if left.endswith(":") and left[:-1].rstrip().endswith('"') and r1 in ("", ",", "}", "]"):
        return f'"{tok}"'
    if r1 in (",", "]") and (left.endswith("[") or (left.endswith(",")
                                                     and _JSON_ARR_OPEN.search(left))):
        return f'"{tok}"'
    if not left and right_line in ("", ",") and ls > 0:
        if s[max(0, ls - 2000):ls - 1].rstrip().endswith((",", "[")):
            return f'"{tok}"'
    return tok


def _bare_cls(d: str) -> str:
    # 9–10 digits — the range of numeric Telegram/social network ids; 11–13 — a document/order/invoice number
    return "USER_ID" if len(d) <= 10 else "DOC_ID"


def _bare_line_mask(m) -> str:
    v = m.group(2)
    if not v.isdigit():                  # «4510 123456» — passport series and number as a line
        return m.group(1) + P.PASSPORT + m.group(3)
    tok = _qtok(mint.tag(_bare_cls(v), v), m.string, m.start(2), m.end(2))
    return m.group(1) + tok + m.group(3)


def _money_ctx(m) -> bool:
    s, a, b = m.string, m.start(1), m.end(1)
    return bool(_UNIT_RIGHT.match(s, b) or _MONEY_LEFT.search(s[max(0, a - 24):a]))


_TOKEN_STOP = " \t\n\"'<>()[]{}\ue000"


def _in_link(m) -> bool:
    """The number is part of a link (a path segment, a query parameter): a product/post id/timestamp.
    Measured on Telegram output: ~3/4 of hits on 11–13 digit numbers were in links, while among
    confirmed DOC_ID findings only 3 of 63 were (and those longer/with a prefix) — links are left alone."""
    s, a = m.string, m.start(1)
    w = s[max(0, a - 300):a]
    tok = w[max(w.rfind(c) for c in _TOKEN_STOP) + 1:]
    return "://" in tok or tok.startswith("www.") or "/" in tok


# to the left of a bare ID label — the subject whose number it is: a meeting (meeting_id_label → DOC_ID), an order,
# a payment, a request, an error — not a user id
_BARE_ID_SUBJECT = re.compile(r"(?i)(?:meeting|conference|webinar|конференции|встречи|вебинара|"
                              r"order|заказ\w*|transaction|транзакци\w*|payment|платеж\w*|платёж\w*|"
                              r"invoice|счет\w*|счёт\w*|ticket|тикет\w*|task|задач\w*|issue|request|"
                              r"запрос\w*|error|ошибк\w*)[^\S\n]*$")


def _bare_id_guard(m) -> bool:
    """True => do not mask. Units/currency on the right — an amount or a quantity; a number inside a link —
    a site object id; a subject word on the left («Meeting ID», «Order ID») — a meeting/order/
    payment number, not a user's."""
    if _UNIT_RIGHT.match(m.string, m.end(2)) or _in_link(m):
        return True
    return _BARE_ID_SUBJECT.search(m.string[max(0, m.start() - 24):m.start()]) is not None


def _mask_array(m) -> str:
    body = _ARR_ITEM.sub(
        lambda x: f'"{mint.tag("USER_ID", x.group(2))}"', m.group(2))
    return m.group(1) + body + m.group(3)


# --- ids in text and links (Telegram re-run audit: USER_ID, DOC_ID, RESOURCE critical) ---
def _uid_seed(v: str) -> str:
    """USER_ID seed of a numeric id. A negative Bot API id: "-100<id>" of a supergroup/channel →
    "channel<id>", "-<id>" of a group → "chat<id>" — the same seeds as a peer id in Telegram
    export fields (structured.telegram.peer_seed/chat_seed): one chat — one token."""
    d = v.lstrip("-")
    if not v.startswith("-"):
        return d
    if v.startswith("-100") and len(d) >= 13:
        return "channel" + d[3:]
    return "chat" + d


def _host_guard(*hosts):
    """Guard for a link recognised by the host's RESOURCE token: the token must be a token of one of
    ``hosts`` (otherwise it is another site with a similar path)."""
    def guard(m) -> bool:
        tok = re.search(r"RESOURCE_\d+", m.group(1))
        return tok is not None and tok.group(0) not in host_token(*hosts)
    return guard


_TME_HOSTS = ("t.me", "telegram.me", "www.t.me")
_START_PARAMS = frozenset({"start", "startapp", "startgroup", "startattach", "startchannel",
                           "ref", "referrer"})
_DIGRUN = re.compile(r"(?<!\d)\d{6,13}(?!\d)")
_HEX_SEG = re.compile(r"[0-9a-fA-F]{8,}")


def _start_value_mask(val: str) -> str:
    """Numeric runs of 6–13 digits in a start parameter value → USER_ID (a referrer/
    bot user id). Left alone: runs inside our token and inside a hex segment of 8+
    chars with two or more letters a-f (a link hash/signature). A letter right on the left — the token goes after
    "-" (not welded to the word)."""
    toks = [t.span() for t in OWN_TOKEN_RE.finditer(val)]
    segs = [(x.start(), x.end()) for x in re.finditer(r"[^_\-.:~]+", val)
            if _HEX_SEG.fullmatch(x.group(0)) and len(re.findall(r"[a-fA-F]", x.group(0))) >= 2]

    def one(d):
        if any(a <= d.start() < b for a, b in toks + segs):
            return d.group(0)
        sep = "-" if d.start() > 0 and val[d.start() - 1].isalpha() else ""
        return sep + mint.tag("USER_ID", d.group(0))
    return _DIGRUN.sub(one, val)


def _tme_query_mask(m) -> str:
    def fix(pm):
        if pm.group(1).lower() not in _START_PARAMS:
            return pm.group(0)
        return pm.group(1) + "=" + _start_value_mask(pm.group(2))
    return m.group(1) + re.sub(r"([^&=?#]+)=([^&#]*)", fix, m.group(2))


# --- issue tracker key (owner 2026-10-06: a rule for all clients, not a prefix list
# of one client) — «ABC-123», in a branch «feature/ABC-123-fix», in a link «…/browse/ABC-123» → DOC_ID
# whole, the number is not left open. One key — one token (seed "issue:<KEY>").
# Not keys: standards, encodings, hashes and ciphers, protocols and interfaces, product models/versions,
# diseases and classifiers, crypto token standards, months and fiscal periods (dates), article numbers.
# Threshold: a 2-letter prefix — only with a tracker sign nearby (a link/branch via "/" on the left, a task
# word in the line): a two-letter "XX-NN" in messages is more often a model or a standard (MD-5, IP-65, PS-5,
# RS-232, CP-1251, ES-6, FP-16), while tracker project keys are usually 3+ letters. By the dictionary
# engine/lexicon.py: an English word as a prefix does not cancel a key (DEV/OPS/WEB/CORE/TASK are common
# project keys and all are en_words); the stop list is only standards/technologies from TECH_TERMS
# (utf/sha/md/http/tls/gpt…) and related ones below.
_ISSUE_STD = frozenset((
    "UTF UCS ISO IEC IEEE ECMA ES RFC PEP PSR WCAG OWASP CWE CVE NIST SOC ITIL COBIT TOGAF PMBOK "
    "HL EN GB JIS DIN ASTM ANSI GOST UN SHA MD CRC AES RSA DES DSA ECDSA HMAC PBKDF BASE CP KOI WIN "
    "TLS SSL HTTP HTTPS IPV IP TCP UDP USB DDR HDMI PCI PCIE SATA LTE WIFI RS COM LPT MP MPEG AV VP "
    "DX CSS HTML PHP JDK GPT LLAMA ARM AMD INT UINT FP BF COVID SARS MERS HIV ICD ERC BEP TRC BRC "
    "TOP FY RC SP LTS BETA ALPHA JAN FEB MAR APR MAY JUN JUL AUG SEP SEPT OCT NOV DEC SKU EAN UPC "
    "ISBN ISSN GTIN MPN PN PS GTA CS AK SU TU IL AN MIG").split())
# a time zone offset («UTC-3», «GMT-5», «MSK-1») and a technology/hardware with a version
# or model («JAVA-17», «MYSQL-8», «RTX-3080», «RAID-5», «CAT-6») are not keys. Technologies — all of
# TECH_TERMS (java/mysql/redis/vue/react/api…) and related ones below. Deliberate project keys — an explicit
# list: the TECH_TERMS stop does not cancel them (DEV/OPS/WEB/CORE).
_ISSUE_TZ = frozenset((
    "UTC GMT MSK EST EDT CST CDT MST MDT PST PDT AKST HST CET CEST EET EEST WET WEST BST IST JST "
    "KST AEST AEDT ACST AWST NZST NZDT HKT SGT ICT SAMT YEKT OMST KRAT IRKT YAKT VLAT MAGT").split())
_ISSUE_TECH = frozenset((
    "RTX GTX RX RAID CAT EXT FAT NODE RUBY PERL LUA CUDA IOS MACOS RHEL GDDR LPDDR HBM NVME SFP QSFP "
    "POE GSM GEMMA QWEN PHI").split())
_ISSUE_PROJECTS = frozenset({"DEV", "OPS", "WEB", "CORE"})
# MAC/hex pairs joined by "-"/":" («AB-12-CD-34-EF-56») — not a key, nor a piece of one
_ISSUE_HEXPAIRS = re.compile(r"[0-9A-Fa-f]{2}(?:[-:][0-9A-Fa-f]{2}){2,}(?![0-9A-Za-z])")
# "_" after the number — a key in a branch name («feature/ABC-123_add_login»): no letter/digit on the right
_ISSUE_KEY = re.compile(r"(?:(?<=\\[ntr])|(?<![\w.\\-]))([A-Z]{2,5})-(\d{1,6})(?![^\W_]|[.,:]\d|-\d)")
_ISSUE_CTX = re.compile(r"(?i)(?:задач|тикет|таск|issue|ticket|task|jira|джир|youtrack|ютрек|трекер|"
                        r"tracker|баг|bug|ветк|branch|коммит|commit|merge|мерж|эпик|epic|story|"
                        r"(?<![a-z])(?:pr|mr)(?![a-z]))")
# an article/model on the left — a product number, not a task key
_ISSUE_ARTICLE = re.compile(r"(?i)(?:артикул\w*|арт\.|sku|модел\w*|model|part|p/n|каталож\w*)"
                            r"[^\S\n]*[:#№]?[^\S\n]*$")


def _issue_guard(m) -> bool:
    """True => not a task key (see _ISSUE_STD and the threshold above)."""
    pre = m.group(1)
    if pre in _ISSUE_STD or pre in _ISSUE_TZ or pre in _ISSUE_TECH \
            or (pre.lower() in TECH_TERMS and pre not in _ISSUE_PROJECTS):
        return True
    s, a, b = m.string, m.start(1), m.end(2)
    if _ISSUE_HEXPAIRS.match(s, a):
        return True
    if _ISSUE_ARTICLE.search(s, max(0, a - 24), a):
        return True
    if len(pre) == 2 and not (a > 0 and s[a - 1] == "/"):
        ls = s.rfind("\n", 0, a) + 1
        le = s.find("\n", b)
        le = len(s) if le < 0 else le
        return not (_ISSUE_CTX.search(s, max(ls, a - 60), a) or _ISSUE_CTX.search(s, b, min(le, b + 60)))
    return False


def _calendar_mask(m) -> str:
    rest = re.sub(r"(?i)([?&]layerId=)(\d{3,20})",
                  lambda x: x.group(1) + mint.tag("DOC_ID", "calendar-layer:" + x.group(2)),
                  m.group(3))
    return m.group(1) + mint.tag("DOC_ID", "calendar-event:" + m.group(2)) + rest


RULES: list[Rule] = [
    Rule("discord_snowflake_field",
         re.compile(r'("' + _KEYS + r'"\s*:\s*)("?)(\d{17,19})("?)'
                    r"(?![0-9A-Za-z_])"),
         mask=_mask_field, value_group=3, confidence="high"),
    Rule("discord_mention_array",
         # the body was raised 2000→20000: Discord allows up to 100 mentions per
         # message, 100 quoted 19-digit snowflakes ≈ 2.4KB — the old
         # cap 2000 silently dropped a mass-ping array (raw snowflakes leaked,
         # CRITICAL). The limit stays BOUNDED and lazy, and the anchor on
         # the literal key + [^\][] keep the scan constant (not quadratic).
         re.compile(r'("(?:mentions|mention_ids|user_ids)"\s*:\s*\[)'
                    r"([^\][]{0,20000}?)(\])"),
         mask=_mask_array, value_group=2,
         # guard: the array contains a raw snowflake at all — otherwise a no-op
         # rule would chew every array on every pass
         guard=lambda m: _ARR_ITEM.search(m.group(2)) is None,
         confidence="high"),
    Rule("discord_mention",
         # <@id> / <@!id> / <@&role-id> (audit: 49 raw role mentions
         # identified the guild and its roles)
         re.compile(r"(<@[!&]?)(\d{6,20})(>)"),
         mask=lambda m: m.group(1)
         + mint.tag("USER_ID", m.group(2)) + m.group(3),
         value_group=2, confidence="high",
         needs="<@"),                        # the opening pair of the match
    # snowflakes in URL PATHS (audit, critical: a guild id in links consistently
    # matches the directory → the ORG_#### pseudonym is reversible for the whole dataset;
    # cdn attachments carry channel/message ids). Anchored on a path segment
    # attachments/ | channels/ — outside them 17-20 digit numbers are left alone (INN/
    # account/epoch in prose are not touched). Each number is minted with THE SAME
    # mint.tag("USER_ID") — linking within the corpus stays intact, reversibility outward
    # is broken. Idempotent: the token is alphabetic, a repeat does not match.
    Rule("discord_url_snowflake",
         re.compile(r"((?:attachments|channels)/)"
                    r"(\d{17,20}(?:/\d{17,20}){0,2})(?!\d)"),
         mask=lambda m: m.group(1) + "/".join(
             mint.tag("USER_ID", d) for d in m.group(2).split("/")),
         value_group=2, confidence="high"),
    # attachments[].local_path: «attachments/<channel-name>/852…__file.png» —
    # the id segment comes AFTER the name, not right after attachments/ (audit: 334
    # raw snowflakes in 54 files restored a CDN link and a ms timestamp).
    # The anchor is the id segment itself between / and __/./" — numbers in prose are not touched.
    Rule("discord_localpath_snowflake",
         re.compile(r"(attachments/[^\"\\/]{1,80}/)"
                    r"(\d{17,20})(?=__|[./\"])"),
         mask=lambda m: m.group(1) + mint.tag("USER_ID", m.group(2)),
         value_group=2, confidence="high"),

    # (Telegram audit, critical USER_ID): a numeric id with an id-family label — in prose, logs,
    # JSON fragments and links tg://user?id=… The label stays, the number is minted USER_ID (the same
    # token as this id anywhere else in the corpus).
    # A negative Bot API id («chat_id: -100…» — a channel/supergroup, the seed as for channel<id>
    # of the export), a parameter label «user_id_<something>=» and a link «t.me/user?id=»
    Rule("labeled_user_id",
         re.compile(r"(?i)((?<![A-Za-z0-9_])(?:" + _ID_LABEL + r"|user_id_[a-z]{2,12})[\"']?"
                    r"[^\S\n]{0,3}[:=#№-]?[^\S\n]{0,3}[\"']?|tg://user\?id=|\.me/user\?id=)"
                    r"(-?\d{5,13})(?![0-9A-Za-z_])"),
         mask=lambda m: m.group(1) + _qtok(mint.tag("USER_ID", _uid_seed(m.group(2))), m.string,
                                           m.start(2), m.end(2)),
         value_group=2, confidence="high"),
    # (Telegram audit: ~70 critical USER_ID — a bot greeting template «Имя (@ник, 5123456789)!»):
    # a numeric id in brackets right after a nickname → USER_ID. By this pass the nickname may already be
    # a token (HANDLE_n) — that fits too.
    Rule("paren_handle_id",
         re.compile(r"(\(@?[A-Za-z][A-Za-z0-9_]{2,31},[^\S\n]{0,2})(\d{6,13})(?=[^\S\n]{0,2}\))"),
         mask=lambda m: m.group(1) + mint.tag("USER_ID", m.group(2)),
         value_group=2, confidence="context",
         needs=re.compile(r",[^\S\n]{0,2}\d{6}")),
    # (Telegram re-run audit: the same greeting template «… (<name>, 5123456789)!» when there is no
    # nickname and the name in brackets is not Latin — paren_handle_id does not see it): «(<word>, <id>)!» —
    # the template's exclamation mark is mandatory
    Rule("paren_name_id",
         re.compile(r"(\([^\s(),]{1,32},[^\S\n]{0,2})(\d{6,13})(?=[^\S\n]{0,2}\)!)"),
         mask=lambda m: m.group(1) + mint.tag("USER_ID", m.group(2)),
         value_group=2, confidence="context",
         needs=re.compile(r",[^\S\n]{0,2}\d{6,13}[^\S\n]{0,2}\)!")),
    # (Telegram audit: a Google account id in document "share" links: …&ouid=<21 digits>&…)
    Rule("google_ouid",
         re.compile(r"([?&]ouid=)(\d{15,25})(?!\d)"),
         mask=lambda m: m.group(1) + mint.tag("USER_ID", m.group(2)),
         value_group=2, confidence="high", needs="ouid="),
    # (Telegram audit: a device/session/user/key UUID next to a label — «mindboxDeviceUUID»,
    # «device_id=…», «session: …», «"guid": "…"», «Client Secret» on the line above): UUID → DOC_ID
    # (minted lowercase — one id = one token). A bare UUID without a label is left alone (task/
    # object ids in logs are not PII). Short labels (key/token/secret/uuid/guid) — only at a word
    # boundary or capitalised in camelCase («clientSecret»), not as a word tail («monkey»).
    Rule("labeled_uuid",
         re.compile(r"(?i)((?:(?:device|install(?:ation)?|session|client|user|account|visitor|guest)"
                    r"[_\s-]?(?:id|uuid|guid)?|(?-i:(?=[A-Z])|(?<![A-Za-z]))"
                    r"(?:secret|token|api[_\s-]?key|key|uuid|guid))"
                    r"[\"']?(?:[^\S\n]{0,3}[:=][^\S\n]{0,3}|[^\S\n]*\n[^\S\n]*)[\"']?)"
                    r"(" + _UUID + r")(?![0-9a-f-])"),
         mask=lambda m: m.group(1) + mint.tag("DOC_ID", m.group(2).lower()),
         value_group=2, confidence="context",
         needs=re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-")),
    # (Telegram audit: a UUID as a cookie value in the header «Cookie: a=1; vid=<uuid>; …»)
    Rule("cookie_uuid",
         re.compile(r"(?i)((?:;|cookie:)[^\S\n]*[\w.-]{1,40}=)(" + _UUID + r")(?=[;\s]|$)"),
         mask=lambda m: m.group(1) + mint.tag("DOC_ID", m.group(2).lower()),
         value_group=2, confidence="context",
         needs=re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-")),
    # (Telegram audit: an id/number sent as a SEPARATE message — «"text": "5123456789"»):
    # a line consisting of one number of 9–13 digits (or a passport "series number" 4+6). In the structural
    # profile field values go as separate lines (the slot separator is a line break), in
    # text — a list line. Phones/INN/OGRN go to dedicated detectors (_owned_elsewhere),
    # millisecond timestamps are left alone.
    # «NN NNNNNNN» as a line — series and number of a foreign passport (under the line with the name,
    # above «Дата выдачи …»)
    Rule("bare_number_line",
         re.compile(r"(?m)(^[^\S\n]*)(\d{9,13}|\d{4}[^\S\n]\d{6}|\d{2}[^\S\n]\d{7})([^\S\n]*)$"),
         mask=_bare_line_mask, value_group=2, confidence="context",
         guard=lambda m: _owned_elsewhere(m.group(2)) or _epoch_ms(m.group(2)),
         needs=re.compile(r"\d{4}[^\S\n]?\d{5}|\d{2}[^\S\n]\d{7}")),
    # (Telegram re-run audit: USER_ID critical ×33+41 — a bot service message «Привет/
    # Пользователь <имя> [<id>] …», the id typed as code between brackets): 8–10 digits in square
    # brackets right after a word/name/token (via a space); in findings and on raw tg — only
    # 9–10, shorter ones are error and task numbers («код ошибки [123456]»). A JSON/Python array («= [123…]»,
    # «: [123…]»), a markdown link «[123…](url)», a line start (a log label) — not here. The space
    # before the bracket — any horizontal one (the bot puts a non-breaking one); between the bracket and the digits
    # there may be invisible characters (U+200B…U+200D, U+2060, U+FEFF) — those too.
    Rule("bracket_user_id",
         re.compile(r"(?<=[^\s=:,(\[{<>\"'`][^\S\n])(\[[\u200b-\u200d\u2060\ufeff]*)(\d{8,10})"
                    r"([\u200b-\u200d\u2060\ufeff]*\])(?![(\[:=\w])"),
         mask=lambda m: m.group(1) + mint.tag("USER_ID", m.group(2)) + m.group(3),
         value_group=2, confidence="context",
         needs=re.compile(r"\d{8}[\u200b-\u200d\u2060\ufeff]*\]")),
    # (Telegram re-run audit: USER_ID ×9+13 — «Your ID: …», «ID: …», «9. ID …», «Id …» under
    # an email): a bare ID label and 7–10 digits (the Telegram/VK id range). The label — only capitalised
    # (ID/Id/ИД/Ид): lowercase «id: …» is a code/log field, a task or order id — the contract of
    # labeled_user_id (it takes an id only with a qualifier, tests/test_detectors_141). A label with
    # a qualifier (user_id, chat ID) — labeled_user_id's; «ID заказа 123…» — not here (a word follows
    # the label, not a number); «Order ID», «Meeting ID», units/currency on the right — not a user id
    # (_bare_id_guard). A JSON key «"ID": …» does not match (a quote after the label); a link parameter
    # «?id=/&id=» and a path «/id/…» — a site object id, not here. A bare JSON number — a quoted token.
    Rule("bare_id_label",
         re.compile(r"((?<![\w.?&/=#-])(?:ID|Id|ИД|Ид)(?:[^\S\n]{0,3}[:#№=—–-][^\S\n]{0,3}|"
                    r"[^\S\n]{1,3}))(\d{7,10})(?![\w]|[.,:-]\d)"),
         mask=lambda m: m.group(1) + _qtok(mint.tag("USER_ID", m.group(2)), m.string,
                                           m.start(2), m.end(2)),
         value_group=2, confidence="context", guard=_bare_id_guard,
         needs=re.compile(r"(?:ID|Id|ИД|Ид)[^\S\n]{0,3}[:#№=—–-]?[^\S\n]{0,3}\d{7}")),
    # (Telegram re-run audit: RESOURCE/USER_ID ×7+7): a link to
    # a message of a private channel/supergroup «t.me/c/<channel id>/<message>» — the channel id →
    # USER_ID with the seed «channel<id>» (the same token as from_id of this channel in the export)
    Rule("tme_private_link",
         re.compile(r"(?i)((?<![\w.-])(?:https?://)?(?:www\.)?(?:(?:t|telegram)\.me|RESOURCE_\d{1,12})"
                    r"/c/)(\d{6,14})(?!\d)"),
         mask=lambda m: m.group(1) + mint.tag("USER_ID", "channel" + m.group(2)),
         value_group=2, confidence="high", guard=_host_guard(*_TME_HOSTS),
         needs=re.compile(r"(?i)/c/\d{6}")),
    # (Telegram re-run audit: USER_ID ×4+3 — «t.me/<bot>?start=ref_<id>», «…/app?startapp=<id>»,
    # «?ref=Id<id>»): numeric runs of 6–13 digits in the value of start/startapp/ref parameters of a
    # t.me link — the id of the user who invited via a referral link
    Rule("tme_start_param",
         re.compile(r"(?i)((?<![\w.-])(?:https?://)?(?:www\.)?(?:(?:t|telegram)\.me|RESOURCE_\d{1,12})"
                    r"/[^\s?#\"'<>]{1,80}\?)([^\s#\"'<>" + _JS + r"]{1,1024})"),   # JSON-safe: no "\"
         mask=_tme_query_mask, value_group=2, confidence="context", guard=_host_guard(*_TME_HOSTS),
         needs=re.compile(r"(?i)[?&](?:start\w*|ref\w*)=[^&\s]*\d{6}")),
    # (Telegram re-run audit: USER_ID/RESOURCE ×6+8 — «vk.com/wall-<owner>_<post>»,
    # «?w=wall-…», «video_ext.php?oid=…»): the owner id of a VK object (a user or a community)
    # → USER_ID (seed «vk<id>» — its own number space); the object number stays. Between the object
    # type and the token — "-", the token is not welded to the word.
    Rule("vk_object_owner",
         re.compile(r"(?i)((?<![\w.-])(?:https?://)?(?:www\.|m\.)?(?:vk\.com|vkontakte\.ru|"
                    r"RESOURCE_\d{1,12})/(?:[\w.]{1,64}\?(?:[^\s&#\"'<>]{0,200}&)?[wz]=)?"
                    r"(?:wall|photo|video|album|topic|clip|doc|audio|market|product|board|note|poll|"
                    r"story|podcast)s?)(-?)(\d{1,12})(_\d{1,12})"),
         mask=lambda m: (m.group(1) + "-" + mint.tag("USER_ID", "vk" + m.group(2) + m.group(3))
                         + m.group(4)),
         value_group=3, confidence="high",
         guard=_host_guard("vk.com", "m.vk.com", "www.vk.com", "vkontakte.ru"),
         needs=re.compile(r"\d_\d")),
    Rule("vk_owner_param",
         re.compile(r"(?i)((?<![\w.-])(?:https?://)?(?:www\.|m\.)?(?:vk\.com|vkontakte\.ru|"
                    r"RESOURCE_\d{1,12})/[\w.]{1,64}\?(?:[^\s&#\"'<>]{0,200}&)?oid=)(-?\d{1,12})(?!\d)"),
         mask=lambda m: m.group(1) + mint.tag("USER_ID", "vk" + m.group(2)),
         value_group=2, confidence="high",
         guard=_host_guard("vk.com", "m.vk.com", "www.vk.com", "vkontakte.ru"),
         needs=re.compile(r"(?i)[?&]oid=-?\d")),
    # («https://<calendar>/event/<id>?applyToFuture=0&event_date=…&
    # layerId=<id>» — 32 links in raw tg): the calendar event and layer ids → DOC_ID. A host with
    # "calendar" in its name or a RESOURCE token with applyToFuture/layerId parameters; public
    # event pages (timepad …/event/<n>) are left alone.
    Rule("calendar_event_link",
         re.compile(r"(?i)((?<![\w.-])(?:https?://)?(?:(?:[a-z0-9-]+\.)*calendar(?:\.[a-z0-9-]+)+|"
                    r"RESOURCE_\d{1,12}(?=/(?:[\w.-]+/){0,4}event/\d{5,20}[^\s\"'<>]{0,1024}"
                    r"[?&](?:layerId|applyToFuture)=))/(?:[\w.-]+/){0,4}event/)(\d{5,20})"
                    r"([^\s\"'<>]{0,1024})"),
         mask=_calendar_mask, value_group=2, confidence="high",
         needs=re.compile(r"(?i)/event/\d{5}")),
    # (meeting invitations in messages): a Zoom meeting number
    # (in a /j/ link and with the label «Meeting ID / Идентификатор конференции») and a Google Meet code
    # (meet.google.com/<code>, tel.meet/<code>) → DOC_ID; one number — one token in the link and in
    # the label (seed «meeting:<digits>»). PIN and passcode — detectors/secrets.
    Rule("zoom_meeting_link",
         re.compile(r"(?i)((?<![\w.-])(?:https?://)?(?:[a-z0-9-]+\.)?zoom\.us/(?:j|w|s|wc(?:/join)?)/)"
                    r"(\d{9,11})(?!\d)"),
         mask=lambda m: m.group(1) + mint.tag("DOC_ID", "meeting:" + m.group(2)),
         value_group=2, confidence="high", needs=re.compile(r"(?i)zoom\.us/")),
    Rule("meeting_id_label",
         re.compile(r"(?i)((?<![\w])(?:meeting id|идентификатор конференции|идентификатор встречи|"
                    r"id конференции|id встречи|conference id|webinar id|идентификатор вебинара)"
                    r"[^\S\n]{0,3}:?[^\S\n]{0,3})(\d{3} ?\d{3,4} ?\d{3,4})(?!\d)"),
         mask=lambda m: m.group(1) + mint.tag("DOC_ID", "meeting:" + m.group(2).replace(" ", "")),
         value_group=2, confidence="high",
         needs=re.compile(r"(?i)(?:meeting|идентификатор|конференции|встречи|conference|webinar)")),
    Rule("meet_code",
         re.compile(r"(?i)((?<![\w.-])(?:https?://)?(?:meet\.google\.com|tel\.meet|RESOURCE_\d{1,12})"
                    r"/(?:lookup/)?)([a-z]{3}-[a-z]{4}-[a-z]{3})(?![\w-])"),
         mask=lambda m: m.group(1) + mint.tag("DOC_ID", "meet:" + m.group(2).lower()),
         value_group=2, confidence="high", guard=_host_guard("meet.google.com", "tel.meet"),
         needs=re.compile(r"(?i)/(?:lookup/)?[a-z]{3}-[a-z]{4}-[a-z]{3}")),
    # (Telegram audit: «m.facebook.com/<page>.php?…&id=<15 digits>» — a Facebook profile/
    # page id in a parameter) → USER_ID (seed «fb<id>»); social_profile skips «*.php»
    Rule("facebook_id_param",
         re.compile(r"(?i)((?<![\w.-])(?:https?://)?(?:www\.|m\.|mbasic\.|web\.)?(?:facebook\.com|fb\.com|"
                    r"RESOURCE_\d{1,12})/[\w.]{1,40}\.php\?(?:[^\s&#\"'<>]{0,300}&)?id=)(\d{5,20})(?!\d)"),
         mask=lambda m: m.group(1) + mint.tag("USER_ID", "fb" + m.group(2)),
         value_group=2, confidence="high",
         guard=_host_guard("facebook.com", "www.facebook.com", "m.facebook.com", "fb.com"),
         needs=re.compile(r"(?i)\.php\?\S*\bid=\d{5}")),
    # (Telegram re-run audit: «<Service> Account ID — <letters-digits>»): a labelled account id →
    # USER_ID; 12 digits after «Account ID» — AWS (financial.aws_account), not here
    Rule("account_id_label",
         re.compile(r"(?i)((?<![\w])(?:account[ _-]?id|id аккаунта|аккаунт id)[^\S\n]{0,3}[:=—–-][^\S\n]{0,3})"
                    r"([A-Za-z0-9_-]{6,64})(?![\w-])"),
         mask=lambda m: m.group(1) + mint.tag("USER_ID", m.group(2)),
         value_group=2, confidence="context",
         guard=lambda m: (not re.search(r"\d", m.group(2)) or re.fullmatch(r"\d{12}", m.group(2)) is not None
                          or OWN_TOKEN_RE.search(m.group(2)) is not None),
         needs=re.compile(r"(?i)(?:account[ _-]?id|id аккаунта|аккаунт id)")),
    # (Telegram re-run audit: a link to an hh.ru resume «…hh.ru/resume/<hex>») → DOC_ID; the path
    # «/resume/<long hex>» — a resume on a masked host too
    Rule("hh_resume_id",
         re.compile(r"(?i)((?<![\w.-])(?:https?://)?(?:[a-z0-9-]+\.)?(?:hh\.ru|hh\.kz|headhunter\.ru|"
                    r"RESOURCE_\d{1,12})/resume/)([0-9a-f]{20,64})(?![0-9a-z])"),
         mask=lambda m: m.group(1) + mint.tag("DOC_ID", m.group(2).lower()),
         value_group=2, confidence="high", needs=re.compile(r"(?i)/resume/[0-9a-f]{20}")),

    # (Telegram audit, DOC_ID critical/major): a standalone number of 11–13 digits in text
    # (a contract/order/invoice/SNILS number without a format) — DOC_ID. Left alone: parts of
    # decimals/versions/times/ranges (".,:-" with a digit before them; after a letter — a label:
    # «тел:7…»), amounts and quantities (currency/units nearby), numbers inside links, millisecond
    # timestamps, phones and valid INN/OGRN (dedicated detectors). A bare JSON number — a quoted
    # token.
    Rule("bare_doc_number",
         re.compile(r"(?<![\w+])(?<!\d[.,:-])(\d{11,13})(?![\w]|[.,:-]\d)"),
         mask=lambda m: _qtok(mint.tag("DOC_ID", m.group(1)), m.string, m.start(1), m.end(1)),
         value_group=1, confidence="context",
         guard=lambda m: (_owned_elsewhere(m.group(1)) or _epoch_ms(m.group(1))
                          or _money_ctx(m) or _in_link(m)),
         needs=re.compile(r"\d{11}")),

    # issue tracker key «ABC-123» (in text, a branch, a link …/browse/ABC-123) → DOC_ID whole;
    # glued to text at the edges («xABC-123», «ABC-123x»; "_" on the right — branch «ABC-123_fix», a key),
    # a continuation with a number («AB-123-45», «ABC-1.2»), MAC, time zones and standards/technologies/models/
    # months/article numbers (_issue_guard) — not a key.
    Rule("issue_key", _ISSUE_KEY,
         mask=lambda m: mint.tag("DOC_ID", "issue:" + m.group(1) + "-" + m.group(2)),
         value_group=0, confidence="context", guard=_issue_guard,
         needs=re.compile(r"[A-Z]{2}-\d")),
]
