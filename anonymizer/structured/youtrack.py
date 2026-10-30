"""YouTrack export profile (JSONL: one issue per line).

File kind — ``.jsonl`` / ``.ndjson``, every non-empty line is an issue JSON
object. Two line forms (signature by content, not by name):
  * exporter wrapper — ``{"entity": "issue", "exported_at": …, "payload":
    {"issue": {…}, "comments": […], "attachments": […]}}`` (the client's batch);
  * bare REST API issue — ``{"$type": "Issue", "idReadable": …, …}``.
Any other JSONL does not belong to the profile and takes the old path.

What the profile does instead of "read a JSON line as flat text":
  * SKIP — service fields (id, idReadable, ringId, numbers, dates, counters,
    flags, $type, schema enums: link type, colour, field metadata) go
    nowhere; relative instance addresses (files, icons, avatars:
    /api/files/<id>?sign=…) — too; an absolute address (the host is the client's name)
    goes to the rules;
  * PEOPLE — a user object (reporter, updater, author, owner, watchers,
    voters, people in customFields values — anywhere, by object shape):
    fullName / name → registry entity token (structured.entities.Registry,
    the same as in Telegram: token = what the normal rule chain gives the
    canonical "First Last" with the tenant salt), login → HANDLE (mint.tag
    on the nick, like handle_at in text), email → EMAIL_REDACTED (like the
    contacts detector), avatarUrl — login/email/name inside the address are replaced
    with the same tokens;
  * FREE TEXT — summary, description, comments[].text, string and
    text values of customFields — to rules, mining and NER; HTML copies
    (wikifiedDescription, textPreview) — to rules and mining without NER;
    project names and short names, the issue key (idReadable), field
    names, tags, enum values — isolated, once per value per
    file (a probe on echo26: the project short name and the issue key hold the client's
    brand, rules brand/issue_key; it must not stay a service value).

The ``FIELD_POLICY`` table is data: "anchor.key → action". A key missing from
the table is NOT skipped: the string goes to free text (the safe
side), a user object is recognised by its shape, the key is counted in
engine.STATS["unknown_key"]. SKIP applies only to scalars: a number, bool, null
and a string of its own service form (_SKIP_FORM: id, $type, date, mime, …);
a string of another form is free text, an object/list under SKIP is walked as
unknown (strings as text, isolated: the schema repeats). A string is never
lost: a string in a list under any anchor, a string instead of an object — as
text; a relative address not of service form — to the rules.
"""
from __future__ import annotations

import re
from urllib.parse import quote

from ..detectors import orgs as _orgs
from ..engine import mint
from ..engine import placeholders as P
from ..engine.boundary import is_own_token
from . import engine as E
from .entities import Registry

# --- actions ----------------------------------------------------------------
SKIP = "skip"            # do not process and do not show to anyone
TEXT = "text"            # free text: rules + mining + NER
AUX = "aux"              # text without natural language (HTML): rules + mining
ITEXT = "itext"          # a name (project, tag, enum value):
#                          isolated, once per value per file
USER = "user"            # a user object (or a list of them)
FNAME = "fname"          # attachment file name: file name detector + rules
CFVALUE = "cfvalue"      # customField value: by value type
DESCEND = "descend"      # nested object/list; arg — the anchor of its items
URL = "url"              # instance address: relative (/api/files/…) —
#                          service; absolute (client host) — to the rules
#                          isolated, without NER
# user object fields
LOGIN = "login"
PNAME = "pname"
EMAIL = "email"
AVATAR = "avatar"

# anchors (JSON paths — for a human reading the table; "[]" — array item)
ANCHOR_PATHS = {
    "record": "$ (обёртка экспортёра)",
    "payload": "$.payload",
    "issue": "$.payload.issue | $ (голая задача)",
    "comment": "<payload|issue>.comments[]",
    "attachment": "<payload|issue|comment>.attachments[]",
    "reaction": "<comment>.reactions[]",
    "issue_ref": "<comment>.issue | <link>.issues[] | <link>.trimmedIssues[]"
                 " | <vote_dup>.issue",
    "project": "<issue|issue_ref>.project",
    "custom_field": "<issue>.customFields[]",
    "pcf": "<custom_field>.projectCustomField",
    "cf_def": "<pcf>.field (определение поля: имя, тип)",
    "cf_value": "<custom_field>.value | .value[] (справочник, период, текст)",
    "link": "<issue>.links[] | <issue>.parent | <issue>.subtasks",
    "tag": "<issue>.tags[]",
    "voters": "<issue>.voters",
    "vote_dup": "<voters>.duplicate[]",
    "watchers": "<issue>.watchers",
    "watcher": "<watchers>.issueWatchers[] | .duplicateWatchers[]",
    "visibility": "<issue|comment>.visibility",
    "group": "<visibility>.permittedGroups[]",
    "user": "любой объект пользователя (по форме: $type User / login+fullName)",
}

_COMMON_SKIP = ("$type", "id")
_DATES = ("created", "updated", "resolved")

FIELD_POLICY = (
    # === exporter wrapper ====================================================
    ("record", "entity", SKIP, None),            # record kind ("issue")
    ("record", "exported_at", SKIP, None),       # export time
    ("record", "payload", DESCEND, "payload"),
    ("payload", "issue", DESCEND, "issue"),
    ("payload", "comments", DESCEND, "comment"),
    ("payload", "attachments", DESCEND, "attachment"),

    # === issue ===============================================================
    *(("issue", k, SKIP, None) for k in _COMMON_SKIP + _DATES + (
        "numberInProject", "commentsCount", "votes", "isDraft",
        "externalIssue")),
    # issue key "ABC-123": the prefix is the project short name; in the client's batch in
    # a third of the links it is the client's brand (echo26 probe: brand/issue_key rules);
    # in text the key is closed by issue_key → DOC_ID — the field gets the same token
    ("issue", "idReadable", ITEXT, None),
    ("issue", "summary", TEXT, None),
    ("issue", "description", TEXT, None),
    ("issue", "wikifiedDescription", AUX, None),   # HTML copy of description
    ("issue", "reporter", USER, None),
    ("issue", "updater", USER, None),
    ("issue", "draftOwner", USER, None),
    ("issue", "project", DESCEND, "project"),
    ("issue", "customFields", DESCEND, "custom_field"),
    ("issue", "links", DESCEND, "link"),
    ("issue", "parent", DESCEND, "link"),
    ("issue", "subtasks", DESCEND, "link"),
    ("issue", "tags", DESCEND, "tag"),
    ("issue", "voters", DESCEND, "voters"),
    ("issue", "watchers", DESCEND, "watchers"),
    ("issue", "visibility", DESCEND, "visibility"),
    ("issue", "comments", DESCEND, "comment"),       # bare REST API issue
    ("issue", "attachments", DESCEND, "attachment"),

    # === issue reference (comment, links, subtasks) =========================
    *(("issue_ref", k, SKIP, None) for k in _COMMON_SKIP + _DATES + (
        "numberInProject",)),
    ("issue_ref", "idReadable", ITEXT, None),
    ("issue_ref", "summary", TEXT, None),
    ("issue_ref", "project", DESCEND, "project"),

    # === project: the name is a client/project name (isolated) ==============
    *(("project", k, SKIP, None) for k in _COMMON_SKIP + (
        "archived", "template")),
    ("project", "name", ITEXT, None),
    ("project", "shortName", ITEXT, None),      # client/project abbreviation
    ("project", "iconUrl", URL, None),

    # === issue field: service part — skip, value — by type ==================
    # field names and empty-value text are set by the admin: may be the
    # client's name (echo26 probe: brand/pack in a field name) — isolated
    *(("custom_field", k, SKIP, None) for k in _COMMON_SKIP),
    ("custom_field", "name", ITEXT, None),
    ("custom_field", "projectCustomField", DESCEND, "pcf"),
    ("custom_field", "value", CFVALUE, None),
    *(("pcf", k, SKIP, None) for k in _COMMON_SKIP + (
        "canBeEmpty", "hasRunningJob", "isPublic", "ordinal")),
    ("pcf", "emptyFieldText", ITEXT, None),
    ("pcf", "field", DESCEND, "cf_def"),
    *(("cf_def", k, SKIP, None) for k in _COMMON_SKIP + (
        "fieldType", "hasRunningJob", "isAutoAttached",
        "isDisplayedInIssueList", "isUpdateable", "ordinal")),
    ("cf_def", "name", ITEXT, None),
    ("cf_def", "localizedName", ITEXT, None),
    ("cf_def", "aliases", ITEXT, None),
    # enum element value (enum, state, version, build,
    # subsystem, group), period, text value
    *(("cf_value", k, SKIP, None) for k in _COMMON_SKIP + (
        "color", "isResolved", "archived", "released", "releaseDate",
        "ordinal", "minutes", "presentation", "ringId", "usersCount",
        "teamForProject", "allUsersGroup")),
    ("cf_value", "name", ITEXT, None),
    ("cf_value", "localizedName", ITEXT, None),
    ("cf_value", "description", TEXT, None),
    ("cf_value", "text", TEXT, None),
    ("cf_value", "owner", USER, None),

    # === comment =============================================================
    *(("comment", k, SKIP, None) for k in _COMMON_SKIP + (
        "created", "updated", "deleted", "pinned")),
    ("comment", "text", TEXT, None),
    ("comment", "textPreview", AUX, None),          # HTML copy of text
    ("comment", "author", USER, None),
    ("comment", "issue", DESCEND, "issue_ref"),
    ("comment", "attachments", DESCEND, "attachment"),
    ("comment", "reactions", DESCEND, "reaction"),
    ("comment", "visibility", DESCEND, "visibility"),

    # === attachment: file name — file name detector; addresses — signed links
    # to the file without user data (/api/files/<id>?sign=…) =================
    *(("attachment", k, SKIP, None) for k in _COMMON_SKIP + (
        "created", "updated", "size", "extension", "charset", "mimeType",
        "metaData", "draft", "removed")),
    ("attachment", "name", FNAME, None),
    ("attachment", "url", URL, None),
    ("attachment", "thumbnailURL", URL, None),
    ("attachment", "author", USER, None),
    ("attachment", "issue", DESCEND, "issue_ref"),
    ("attachment", "comment", DESCEND, "comment"),
    ("attachment", "visibility", DESCEND, "visibility"),

    # === reaction ============================================================
    *(("reaction", k, SKIP, None) for k in _COMMON_SKIP + ("reaction",)),
    ("reaction", "author", USER, None),

    # === issue link: link type is schema =====================================
    *(("link", k, SKIP, None) for k in _COMMON_SKIP + (
        "direction", "linkType")),
    ("link", "issues", DESCEND, "issue_ref"),
    ("link", "trimmedIssues", DESCEND, "issue_ref"),

    # === tag ================================================================
    *(("tag", k, SKIP, None) for k in _COMMON_SKIP + (
        "color", "untagOnResolve")),
    ("tag", "name", ITEXT, None),
    ("tag", "owner", USER, None),

    # === votes and watchers =================================================
    *(("voters", k, SKIP, None) for k in _COMMON_SKIP + ("hasVote",)),
    ("voters", "original", USER, None),
    ("voters", "duplicate", DESCEND, "vote_dup"),
    *(("vote_dup", k, SKIP, None) for k in _COMMON_SKIP),
    ("vote_dup", "user", USER, None),
    ("vote_dup", "issue", DESCEND, "issue_ref"),
    *(("watchers", k, SKIP, None) for k in _COMMON_SKIP + ("hasStar",)),
    ("watchers", "issueWatchers", DESCEND, "watcher"),
    ("watchers", "duplicateWatchers", DESCEND, "watcher"),
    *(("watcher", k, SKIP, None) for k in _COMMON_SKIP + ("isStarred",)),
    ("watcher", "user", USER, None),

    # === visibility =========================================================
    *(("visibility", k, SKIP, None) for k in _COMMON_SKIP),
    ("visibility", "permittedUsers", USER, None),
    ("visibility", "permittedGroups", DESCEND, "group"),
    *(("group", k, SKIP, None) for k in _COMMON_SKIP + ("ringId",)),
    ("group", "name", ITEXT, None),

    # === user object =========================================================
    # id and ringId are internal instance ids (like the issue id): skip;
    # banned/guest/online are flags
    *(("user", k, SKIP, None) for k in _COMMON_SKIP + (
        "ringId", "banned", "guest", "online")),
    ("user", "login", LOGIN, None),
    ("user", "fullName", PNAME, None),
    ("user", "name", PNAME, None),
    ("user", "email", EMAIL, None),
    ("user", "avatarUrl", AVATAR, None),
)

POLICY: dict = {}
for _anchor, _key, _act, _arg in FIELD_POLICY:
    POLICY.setdefault(_anchor, {})[_key] = (_act, _arg)
del _anchor, _key, _act, _arg


_USER_TYPES = frozenset({"User", "Me", "VcsUnresolvedUser"})
_EMAIL_RX = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")
_ABS_URL = re.compile(r"(?i)[a-z][a-z0-9+.-]*://|//")

# service form of a string under SKIP (key → fullmatch regex); no key — id form
# (starts with a digit, no spaces: "2-123", "0-1", ringId UUID). Another form
# means the field holds data (full name, email, brand): free text, not a skip
_ID_RX = re.compile(r"[0-9][0-9A-Za-z._:\[\]*-]{0,63}"
                    r"|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_DT_RX = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
                    r"(?:Z|[+-]\d{2}:?\d{2})?|\d{1,20}")
_SKIP_FORM = {
    "$type": re.compile(r"[A-Z][A-Za-z0-9]{0,63}"),
    "id": _ID_RX, "ringId": _ID_RX,
    "entity": re.compile(r"[a-z][a-z_]{0,31}"),
    "exported_at": _DT_RX, "created": _DT_RX, "updated": _DT_RX,
    "resolved": _DT_RX, "releaseDate": _DT_RX,
    "extension": re.compile(r"[A-Za-z0-9]{1,10}"),
    "charset": re.compile(r"[A-Za-z0-9_.:-]{1,32}"),
    "mimeType": re.compile(r"[A-Za-z0-9.+-]{1,64}/[A-Za-z0-9.+-]{1,128}"
                           r"(?:;\s*[A-Za-z0-9_.-]+=[A-Za-z0-9_.\"-]+)*"),
    "metaData": re.compile(r"\d{1,6}x\d{1,6}"),
    "direction": re.compile(r"[A-Z_]{1,16}"),
    "reaction": re.compile(r"[a-z0-9_+-]{1,40}"),
    "presentation": re.compile(r"[0-9wdhm ]{1,32}"),
}
for _k in ("numberInProject", "commentsCount", "votes", "size", "ordinal",
           "minutes", "usersCount"):
    _SKIP_FORM[_k] = re.compile(r"-?\d{1,20}")
del _k
# service relative instance address: segments are YouTrack/Hub address words
# or ids, parameters are its own (signature, update mark, size); anything else goes to the rules
_URL_QUERY_KEYS = frozenset({"sign", "updated", "s", "etag", "dpr", "size", "c",
                             "w", "h"})
_URL_SEG_RX = re.compile(r"[0-9][0-9A-Za-z._-]*"
                         r"|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_URL_VAL_RX = re.compile(r"[A-Za-z0-9_%.=+-]*")
# instance address words (/hub/api/rest/avatar/…, /api/files/…): a login equal to
# such a word is not searched for in the avatar address
_URL_WORDS = frozenset({"hub", "api", "rest", "avatar", "avatars", "users", "user",
                        "files", "file", "default", "static", "youtrack", "http",
                        "https", "www", "etag", "size", "dpr", "png", "svg", "jpg"})


def service_url(val: str) -> bool:
    """A relative instance address without user data
    (/api/files/8-1?sign=…&updated=…, /hub/api/rest/avatar/<ringId>?s=48&etag=…)."""
    from urllib.parse import urlsplit
    if not val.startswith("/") or val.startswith("//") or any(c.isspace() for c in val):
        return False
    sp = urlsplit(val)
    segs = [x for x in sp.path.split("/") if x]
    if not segs or any(x.lower() not in _URL_WORDS and not _URL_SEG_RX.fullmatch(x)
                       for x in segs):
        return False
    if sp.fragment:
        return False
    for part in (sp.query.split("&") if sp.query else ()):
        k, _eq, v = part.partition("=")
        if k not in _URL_QUERY_KEYS or not _URL_VAL_RX.fullmatch(v):
            return False
    return True


def skip_ok(key, val) -> bool:
    """A scalar under SKIP is a service value (skipped): a number, bool, a string of its own form."""
    if isinstance(val, bool) or isinstance(val, (int, float)):
        return True
    if not isinstance(val, str):
        return False
    if E.is_token(val):
        return True                      # already a token (repeat pass)
    return (_SKIP_FORM.get(key) or _ID_RX).fullmatch(val) is not None


# the fix layer (fix_engine.control_sweep, llm_hotfix) does not touch a service value in a
# profile file: a string under a key that is service for at least one anchor (SKIP), of its
# own form (skip_ok), and the record kind under ``entity`` — the profile signature (after
# post-cleaning the file is recognised by the same profile). Free-text keys (summary, name,
# text, …) are not here, nor a string of the wrong form under a service key
# (externalIssue: a full name): the fix layer cleans those
SKIP_KEYS = frozenset(k for _a, k, act, _g in FIELD_POLICY if act == SKIP)
KEEP_KEYS = SKIP_KEYS | {"entity"}


def keep_value(key, val) -> bool:
    """The record value ``key: val`` is a service value; the fix layer does not change it."""
    if key == "entity":
        return True
    return key in SKIP_KEYS and skip_ok(key, val)


def is_user(node) -> bool:
    """A YouTrack user object — by type or by shape: login + name
    (fullName/name), email or ringId."""
    if not isinstance(node, dict):
        return False
    t = node.get("$type")
    if t in _USER_TYPES:
        return True
    return isinstance(node.get("login"), str) and (
        "fullName" in node or "ringId" in node or "name" in node
        or "email" in node)


def user_key(u: dict):
    """Entity key in the registry: the instance login (stable within an
    export and between exports of one YouTrack), otherwise ringId / id."""
    for k, pre in (("login", "yt:"), ("ringId", "yt:ring:"), ("id", "yt:id:")):
        v = u.get(k)
        if isinstance(v, str) and v.strip() and not E.is_token(v):
            return pre + v.strip().lower()
    return None


def user_name(u: dict):
    """The user's display name: fullName, otherwise name."""
    for k in ("fullName", "name"):
        v = u.get(k)
        if isinstance(v, str) and v.strip() and not E.is_token(v):
            return v
    return None


def login_token(login: str) -> str:
    """Login → token: an email — EMAIL_REDACTED (like contacts.email in text),
    otherwise HANDLE on the lower-cased nick (like names.handle_at for "@nick").
    A login that is a client role label ("Исполнитель_12") gets the same
    ORG token the engine rule gives in the export text; a label inside a login
    ("Исполнитель_12.dev") — ORG token + the rest as HANDLE on the nick."""
    s = login.strip()
    if _EMAIL_RX.fullmatch(s):
        return P.EMAIL
    role = _orgs.client_role_seed(s)
    if role is not None:
        return mint.tag("ORG", role)
    mixed = _orgs.client_roles_split(s, lambda rest: mint.tag("HANDLE", rest.lower()))
    if mixed is not None:
        return mixed
    return mint.tag("HANDLE", s.lower())


def _masked(s: str) -> bool:
    """The field value is already masked: our token as a whole, or (a client role
    label inside the value — ORG token + the rest as a field token) our token as a separate word."""
    return E.is_token(s) or (s[:1] == "@" and E.is_token(s[1:])) or is_own_token(s, word=True)


class _Session:
    """State of processing one file: profile + registry + report.
    ``report is None`` — projection mode (mining/NER/audit): no point edits
    are built and nothing is minted, only text slots are collected."""

    def __init__(self, profile, registry, report):
        self.p = profile
        self.masking = report is not None
        self.report = report
        self.reg = registry if registry is not None or not self.masking \
            else Registry()
        self._acts = {
            TEXT: self._text, AUX: self._aux, ITEXT: self._itext,
            USER: self._user_act, FNAME: self._fname, CFVALUE: self._cfvalue,
            DESCEND: self._descend, LOGIN: self._login, PNAME: self._pname,
            EMAIL: self._email, AVATAR: self._avatar, URL: self._url,
        }

    # ---- entry ----
    def plan(self, pat, obj, header: bool):
        plan = E.Plan()
        if isinstance(obj, dict):
            self._visit(root_anchor(obj), obj, (), plan)
        elif isinstance(obj, list):
            self._visit("?", obj, (), plan)
        return plan if plan.edits or plan.slots else None

    def _visit(self, anchor, node, path, plan, iso=False) -> None:
        """Walk an object/list by the anchor table. ``iso`` — strings of an unknown
        anchor ("?") go isolated (subtree of a service key: the schema repeats across
        the whole file — once per value)."""
        if isinstance(node, list):
            for i, item in enumerate(node):
                if isinstance(item, (dict, list)):
                    self._visit(anchor, item, path + (i,), plan, iso)
                elif isinstance(item, str) and item:
                    # a string in a list — as text under any anchor (never lost)
                    self._free(item, path + (i,), None, plan, iso)
            return
        if anchor != "user" and is_user(node):
            anchor = "user"
        table = POLICY.get(anchor) or {}
        acts = self._acts
        for key, val in node.items():
            if val is None or val == "":
                continue
            p = path + (key,)
            row = table.get(key)
            if row is None:
                self._unknown(anchor, key, val, p, plan, iso)
                continue
            act = row[0]
            if act == SKIP:
                self._skip(key, val, p, plan)
            elif act in _SCALAR_ACTS and isinstance(val, (dict, list)):
                self._visit("?", val, p, plan)      # an object instead of a string
            elif act == DESCEND and isinstance(val, str):
                plan.slot(val, p, key)              # a string instead of an object
            else:
                acts[act](node, key, val, p, plan, row[1])

    def _free(self, val, p, key, plan, iso=False) -> None:
        if iso and val.strip() and not E.is_token(val):
            plan.slot(val, p, key, iso=True)
        else:
            plan.slot(val, p, key)

    def _skip(self, key, val, p, plan) -> None:
        """SKIP: a scalar of service form — skipped; a string of another form —
        text; an object/list — walked as unknown (strings isolated)."""
        if isinstance(val, (dict, list)):
            self._visit("?", val, p, plan, iso=True)
        elif isinstance(val, str) and not skip_ok(key, val):
            E.STATS["skip_text"] += 1
            plan.slot(val, p, key)

    def _unknown(self, anchor, key, val, p, plan, iso=False) -> None:
        if anchor != "?":
            E.STATS["unknown_key"] += 1
            self.p.unknown.add((anchor, key))
        if isinstance(val, str):
            self._free(val, p, key, plan, iso)
        elif isinstance(val, (dict, list)):
            self._visit("?", val, p, plan, iso)

    def _mask(self, rule, plan, p, before, token) -> None:
        plan.edits[p] = token
        self.report.record(rule, before, token)

    # ---- free text ----
    def _text(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, str):
            plan.slot(val, p, key)
        elif isinstance(val, (dict, list)):
            self._visit("?", val, p, plan)

    def _aux(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, str):
            # HTML copy of the text: rules on the HTML form (&#NNN;, &nbsp;)
            plan.slot(val, p, key, ner=False, html=True)
        elif isinstance(val, (dict, list)):
            self._visit("?", val, p, plan)

    def _itext(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, str):
            if val.strip() and not E.is_token(val):
                plan.slot(val, p, key, iso=True)
        elif isinstance(val, (dict, list)):
            self._visit("?", val, p, plan)

    def _url(self, node, key, val, p, plan, arg, orig=None) -> None:
        """Address: a service relative one — skipped (edit ``val`` instead of
        ``orig`` — avatar tokens already substituted); any other (absolute — the
        client's host, relative with a file name/data) — to the rules,
        isolated, without NER."""
        if not isinstance(val, str) or E.is_token(val):
            return
        orig = val if orig is None else orig
        if not _ABS_URL.match(val) and service_url(val):
            if val != orig:
                plan.edits[p] = val
            return
        plan.slots.append(E.Slot(val, [p], key, False, orig, "", True))

    def _fname(self, node, key, val, p, plan, arg) -> None:
        """Attachment file name: rules first (people in the name get the same
        token as in text), then the file name detector on their output."""
        if isinstance(val, str):
            plan.slots.append(E.Slot(val, [p], key, False, val, "",
                                     post=_fname_post if self.masking else None))

    def _cfvalue(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, str):
            plan.slot(val, p, key)       # string field (SimpleIssueCustomField)
        elif isinstance(val, dict):
            self._cf_item(val, p, plan)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    self._cf_item(item, p + (i,), plan)
                elif isinstance(item, str) and item:
                    plan.slot(item, p + (i,), key)
        # numbers (date, number, period in minutes), bool — service values

    def _cf_item(self, item: dict, p, plan) -> None:
        # a user (Single/MultiUser) is recognised by shape in _visit;
        # an enum element, period, text value — the cf_value table
        self._visit("cf_value", item, p, plan)

    def _descend(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, (dict, list)):
            self._visit(arg, val, p, plan)

    # ---- people ----
    def _user_act(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, dict):
            self._user(val, p, plan)
        elif isinstance(val, list):
            for i, u in enumerate(val):
                if isinstance(u, dict):
                    self._user(u, p + (i,), plan)
                elif isinstance(u, list):
                    self._visit("?", u, p + (i,), plan)
                elif isinstance(u, str) and u:
                    plan.slot(u, p + (i,), key)     # login/name as a string — as text
        elif isinstance(val, str):
            plan.slot(val, p, key)       # not an object — as free text

    def _user(self, u: dict, p, plan) -> None:
        self._visit("user", u, p, plan)

    def _pname(self, node, key, val, p, plan, arg) -> None:
        if not isinstance(val, str):
            return
        if not self.masking or _masked(val) or not val.strip():
            return
        tok = self._person_token(node, val)
        self._mask("yt_person", plan, p, val, tok)

    def _person_token(self, node, name) -> str:
        k = user_key(node)
        if k is not None:
            return self.reg.user_token(k, name)
        return self.reg.name_token(name)

    def _login(self, node, key, val, p, plan, arg) -> None:
        if not isinstance(val, str) or not self.masking:
            return
        if not val.strip() or _masked(val):
            return
        tok = login_token(val)
        self._mask("yt_email" if tok == P.EMAIL else "yt_login", plan, p,
                   val, tok)

    def _email(self, node, key, val, p, plan, arg) -> None:
        if not isinstance(val, str) or not self.masking:
            return
        if val.strip() and not E.is_token(val) and P.EMAIL not in val:
            self._mask("yt_email", plan, p, val, P.EMAIL)

    def _avatar(self, node, key, val, p, plan, arg) -> None:
        """Avatar address: the login, email and user name inside it (raw and
        URL-encoded) become tokens of the same entity; otherwise it is treated as an
        instance address (URL). An address like /hub/api/rest/avatar/<ringId>?s=48&etag=…
        (the client's export) carries no user data."""
        if not isinstance(val, str) or not val.strip() or E.is_token(val):
            return
        new = val
        if self.masking:
            low = new.lower()
            pairs = []
            login = node.get("login")
            if isinstance(login, str) and len(login.strip()) >= 3 \
                    and not _masked(login.strip()):
                pairs.append((login.strip(), login_token(login)))
            email = node.get("email")
            if isinstance(email, str) and "@" in email and not _masked(email.strip()):
                pairs.append((email.strip(), P.EMAIL))
            name = user_name(node)
            if name is not None and len(name.strip()) >= 3:
                pairs.append((name.strip(), self._person_token(node, name)))
            for raw, tok in pairs:
                if raw.lower() in _URL_WORDS:
                    continue             # login "api"/"rest" is an address word, not a person
                for form in sorted({raw, quote(raw, safe=""), quote(raw)},
                                   key=len, reverse=True):
                    if form and form.lower() in low:
                        new = _replace_ci(new, form, tok)
                        low = new.lower()
            if new != val:
                self.report.record("yt_avatar", val, new)
        # host and other address data — by the rules (like any instance address)
        self._url(node, key, new, p, plan, arg, orig=val)


_SCALAR_ACTS = frozenset({TEXT, AUX, ITEXT, FNAME, LOGIN, PNAME, EMAIL, AVATAR, URL})


def _fname_post(text: str, report) -> str:
    """File name detector on the rules output (FNAME: rules first)."""
    from ..detectors import filenames
    out = filenames.mask_name(text)
    if out != text:
        report.record("yt_filename", text, out)
    return out


def _replace_ci(s: str, sub: str, rep: str) -> str:
    """Replace ``sub`` as an address component (case-insensitive): on both sides —
    not a letter/digit/"_.%+-" (login "rest" does not cut "/hub/api/rest/…")."""
    return re.sub(r"(?<![\w.%+-])" + re.escape(sub) + r"(?![\w%+-])",
                  lambda _m: rep, s, flags=re.IGNORECASE)


def root_anchor(obj: dict) -> str:
    """Anchor of the line root: exporter wrapper or bare issue."""
    if "payload" in obj or "entity" in obj:
        return "record"
    if obj.get("$type") == "Issue" or "idReadable" in obj \
            or "customFields" in obj:
        return "issue"
    return "?"


def iter_users(node):
    """All user objects of a record (in document order)."""
    if isinstance(node, dict):
        if is_user(node):
            yield node
            return
        for v in node.values():
            if isinstance(v, (dict, list)):
                yield from iter_users(v)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                yield from iter_users(v)


class YouTrackProfile:
    """YouTrack export: JSONL, record = issue (``jsonl`` — the engine's
    "line = unit" layout)."""
    name = "youtrack"
    kind = "jsonl"
    jsonl = True
    # a rules batch is 16 issues: free text per issue ≈ 4k chars
    # (measured on the client's export: 64M chars of text over 16k
    # issues), a batch ≈ 64k chars — like the big CSV window:
    # hundreds of rules have a pack probe, not all 84k.
    batch_units = 16
    spine = frozenset()
    anchors: dict = {}
    # service values the fix layer does not touch (engine.jsonl_keep)
    keep_keys = KEEP_KEYS
    keep_value = staticmethod(keep_value)

    def __init__(self):
        self.unknown: set = set()        # (anchor, key) pairs missing from the table

    def session(self, registry, report) -> _Session:
        return _Session(self, registry, report)

    # ---- entity registry collection ----
    def collect(self, pat, obj, header: bool, reg: Registry, rel) -> None:
        for u in iter_users(obj):
            name = user_name(u)
            key = user_key(u)
            if name is not None:
                if key is not None:
                    reg.add_user(key, name, rel)
                else:
                    reg.add_loose(name, rel)
            login = u.get("login")
            if isinstance(login, str):
                reg.add_account(login, rel)      # mining: yt-users, _handle_ok
            reg.add_person(u.get("fullName"), rel)


# signature: YouTrack issue keys that never occur together in other JSONL
_ISSUE_MARKS = ('"idReadable"', '"numberInProject"', '"customFields"',
                '"wikifiedDescription"', '"commentsCount"', '"reporter"')
_WRAP_RX = re.compile(r'\{\s*"entity"\s*:\s*"issue"\s*,')


class _Sniffer:
    """Format signature from the file head → profile or None."""
    name = "youtrack"

    def __init__(self):
        self._prof = YouTrackProfile()

    def sniff(self, head: str):
        return None                      # .json is not this profile

    def sniff_jsonl(self, head: str):
        line = head.lstrip("﻿ \t\r\n").split("\n", 1)[0]
        if line[:1] != "{":
            return None
        keys = E.top_keys(line)
        marks = sum(1 for m in _ISSUE_MARKS if m in line)
        if keys.get("payload", "")[:1] == "{" and _WRAP_RX.match(line):
            # exporter wrapper: {"entity": "issue", …, "payload": {"issue":
            if re.search(r'"payload"\s*:\s*\{\s*"issue"\s*:\s*\{', line) \
                    and marks >= 1:
                return self._prof
            return None
        if re.match(r'"Issue"', keys.get("$type", "")) and marks >= 2:
            return self._prof
        if "idReadable" in keys and "summary" in keys and marks >= 3:
            return self._prof
        return None


PROFILE = _Sniffer()
