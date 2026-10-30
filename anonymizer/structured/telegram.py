"""Telegram Desktop export profile (``result.json``).

Two kinds of file, recognised by content (not by name):
  * full account export — root with ``chats`` / ``left_chats`` (an
    ``about`` + ``list`` object), ``personal_information``, ``contacts`` …;
  * single-chat export — root ``name`` / ``type`` / ``id`` / ``messages``.
Any other JSON does not belong to the profile and takes the normal pipeline path.

What the profile does instead of "read JSON as flat text":
  * SKIP — service fields (message ids, dates, sizes, mime, image sizes,
    reaction emoji, action types, export paths) go nowhere;
  * IDENTITY — by field, without LLM or mining: a person's name → entity
    token (entities.Registry: the id links the names, the token is the same
    one the text pass gives), peer id → USER_ID, phone → PHONE_REDACTED,
    nickname → HANDLE, coordinates → GEO_REDACTED;
  * TYPED text ENTITIES (email / phone / bank_card / mention /
    mention_name) — by type, straight into their class;
  * FREE TEXT (message parts, titles, polls, button labels, bio,
    channel and group names) — only this goes to rules, mining and NER.

The ``FIELD_POLICY`` table below is data: "anchor.key → action". An anchor is
a document object; its absolute JSON paths are listed in ``ANCHOR_PATHS``.
A key missing from the table is NOT skipped: the string goes to free text
(the safe side), and the key is counted in engine.STATS["unknown_key"].
"""
from __future__ import annotations

import re

from ..engine import mint
from ..engine import placeholders as P
from ..engine.boundary import is_own_token
from ..engine.vocab import COMMON_EN, TECH_TERMS
from . import engine as E
from .entities import Registry, canonical

# --- actions ----------------------------------------------------------------
SKIP = "skip"            # do not process and do not show to anyone
TEXT = "text"            # free text: rules + mining + NER
AUX = "aux"              # text without natural language: rules + mining
RICH = "rich"            # text_entities: text parts + the text mirror
MIRROR = "mirror"        # text: mirror of text_entities (rebuilt from them)
RICHV = "richv"          # string or list of parts (poll, to-do item)
PERSON = "person"        # a person's name; arg — the id key in the same object
RESOLVE = "resolve"      # name without id or kind: a person or a channel
MEMBERS = "members"      # list of people's names
NAMEPART = "namepart"    # first_name/last_name; arg — the id key (or None)
CHATNAME = "chatname"    # chat name: personal chat — a person, otherwise text
FCNAME = "fcname"        # frequent contact name (by type)
PEER = "peer"            # peer id as a string: user<N> / channel<N> / chat<N>
UID = "uid"              # numeric id; arg: user | chat | fc (how to seed)
PHONE = "phone"
HANDLE = "handle"
GEO = "geo"
FNAME = "fname"          # file name: file name detector + rules
PATH = "path"            # export path: directory untouched, name — as FNAME
DESCEND = "descend"      # nested object/list; arg — the anchor of its items

# absolute JSON paths of the anchors ("[]" — array item; in a single-chat
# export the chat anchor is the document root, message — messages[])
ANCHOR_PATHS = {
    "root": "$",
    "section": "$.chats | $.left_chats",
    "chat": "$.chats.list[] | $.left_chats.list[] | $ (один чат)",
    "message": "<chat>.messages[]",
    "entity": "<message>.text_entities[] | <message>.text[]",
    "reaction": "<message>.reactions[]",
    "recent": "<reaction>.recent[]",
    "poll": "<message>.poll",
    "poll_answer": "<poll>.answers[]",
    "todo_list": "<message>.todo_list",
    "todo_item": "<todo_list>.answers[]",
    "button": "<message>.inline_bot_buttons[][]",
    "contact_information": "<message>.contact_information",
    "location_information": "<message>.location_information",
    "personal_information": "$.personal_information",
    "contacts": "$.contacts",
    "contact": "$.contacts.list[]",
    "frequent_contacts": "$.frequent_contacts",
    "frequent_contact": "$.frequent_contacts.list[]",
    "profile_picture": "$.profile_pictures[]",
    "story": "$.stories[]",
}

FIELD_POLICY = (
    # === SKIP ================================================================
    # the exporter's own text ("Here is the data you requested…") — the same
    # in every export, no user data in it
    ("root", "about", SKIP, None),
    ("section", "about", SKIP, None),
    ("contacts", "about", SKIP, None),
    ("frequent_contacts", "about", SKIP, None),
    # schema enum values (chat/message/action/media/reaction type, call
    # discard reason, contact category) — format vocabulary, not data
    ("chat", "type", SKIP, None),
    ("message", "type", SKIP, None),
    ("message", "action", SKIP, None),
    ("message", "media_type", SKIP, None),
    ("message", "discard_reason", SKIP, None),
    ("message", "mime_type", SKIP, None),
    ("entity", "type", SKIP, None),
    ("reaction", "type", SKIP, None),
    ("button", "type", SKIP, None),
    ("frequent_contact", "category", SKIP, None),
    ("frequent_contact", "type", SKIP, None),
    # message sequence numbers inside a chat and links to them — not people's ids
    ("message", "id", SKIP, None),
    ("message", "reply_to_message_id", SKIP, None),
    ("message", "message_id", SKIP, None),
    ("todo_item", "id", SKIP, None),
    ("message", "completed", SKIP, None),
    ("message", "incompleted", SKIP, None),
    # dates and timestamps (the structure gate forbids clobbering timestamp fields)
    ("message", "date", SKIP, None),
    ("message", "date_unixtime", SKIP, None),
    ("message", "edited", SKIP, None),
    ("message", "edited_unixtime", SKIP, None),
    ("message", "schedule_date", SKIP, None),
    ("recent", "date", SKIP, None),
    ("contact", "date", SKIP, None),
    ("contact", "date_unixtime", SKIP, None),
    ("profile_picture", "date", SKIP, None),
    ("profile_picture", "date_unixtime", SKIP, None),
    ("story", "date", SKIP, None),
    ("story", "date_unixtime", SKIP, None),
    ("story", "expires", SKIP, None),
    ("story", "expires_unixtime", SKIP, None),
    # sizes, durations, counters, flags (numbers and bool; the structure gate
    # forbids a token in size/count/duration/width/height)
    ("message", "file_size", SKIP, None),
    ("message", "photo_file_size", SKIP, None),
    ("message", "thumbnail_file_size", SKIP, None),
    ("message", "contact_vcard_file_size", SKIP, None),
    ("message", "width", SKIP, None),
    ("message", "height", SKIP, None),
    ("message", "duration_seconds", SKIP, None),
    ("message", "duration", SKIP, None),
    ("message", "period", SKIP, None),
    ("message", "live_location_period_seconds", SKIP, None),
    ("message", "boosts", SKIP, None),
    ("message", "days", SKIP, None),
    ("message", "months", SKIP, None),
    ("message", "cost", SKIP, None),
    ("message", "media_spoiler", SKIP, None),
    ("message", "new_icon_emoji_id", SKIP, None),
    ("reaction", "count", SKIP, None),
    ("poll", "closed", SKIP, None),
    ("poll", "total_voters", SKIP, None),
    ("poll_answer", "voters", SKIP, None),
    ("poll_answer", "chosen", SKIP, None),
    ("todo_list", "others_can_append", SKIP, None),
    ("todo_list", "others_can_complete", SKIP, None),
    ("frequent_contact", "rating", SKIP, None),
    ("story", "pinned", SKIP, None),
    ("entity", "collapsed", SKIP, None),
    # emoji: sticker, reaction, topic icon; custom emoji id/path
    ("message", "sticker_emoji", SKIP, None),
    ("message", "emoticon", SKIP, None),
    ("reaction", "emoji", SKIP, None),
    ("reaction", "document_id", SKIP, None),
    ("entity", "document_id", SKIP, None),
    # code block language label ("python", "json")
    ("entity", "language", AUX, None),   # typed by the user: "```word"

    # === IDENTITY (by field, without LLM or mining) ==========================
    # sender / action author / forward source: name + peer id
    ("message", "from", PERSON, "from_id"),
    ("message", "from_id", PEER, None),
    ("message", "actor", PERSON, "actor_id"),
    ("message", "actor_id", PEER, None),
    ("message", "forwarded_from", PERSON, "forwarded_from_id"),
    ("message", "forwarded_from_id", PEER, None),
    ("message", "reply_to_peer_id", PEER, None),
    ("recent", "from", PERSON, "from_id"),
    ("recent", "from_id", PEER, None),
    # channel post signature and service message members — people without an id
    ("message", "author", PERSON, None),
    ("message", "members", MEMBERS, None),
    # "saved from" / "invited by": peer kind unknown — a person or a channel
    ("message", "saved_from", RESOLVE, None),
    ("message", "inviter", RESOLVE, None),
    ("message", "via_bot", HANDLE, None),
    # chat: personal — a person's name and id; group/channel — name as text, peer id
    ("chat", "name", CHATNAME, None),
    ("chat", "id", UID, "chat"),
    # the export owner
    ("personal_information", "user_id", UID, "user"),
    ("personal_information", "first_name", NAMEPART, "user_id"),
    ("personal_information", "last_name", NAMEPART, "user_id"),
    ("personal_information", "phone_number", PHONE, None),
    ("personal_information", "username", HANDLE, None),
    # address book and frequent contacts
    ("contact", "first_name", NAMEPART, None),
    ("contact", "last_name", NAMEPART, None),
    ("contact", "phone_number", PHONE, None),
    ("frequent_contact", "id", UID, "fc"),
    ("frequent_contact", "name", FCNAME, None),
    # contact shared in a message; location
    ("contact_information", "first_name", NAMEPART, None),
    ("contact_information", "last_name", NAMEPART, None),
    ("contact_information", "phone_number", PHONE, None),
    ("location_information", "latitude", GEO, None),
    ("location_information", "longitude", GEO, None),
    # user id in a mention_name entity
    ("entity", "user_id", UID, "user"),

    # === FREE TEXT (rules + mining + NER) ====================================
    ("message", "text_entities", RICH, None),
    ("message", "text", MIRROR, None),
    ("message", "title", TEXT, None),
    ("message", "new_title", TEXT, None),
    ("message", "invoice_information", TEXT, None),
    ("personal_information", "bio", TEXT, None),
    ("poll", "question", RICHV, None),
    ("poll_answer", "text", RICHV, None),
    ("todo_list", "title", RICHV, None),
    ("todo_item", "text", RICHV, None),
    ("button", "text", TEXT, None),

    # === TEXT WITHOUT NATURAL LANGUAGE (rules + mining, no NER) ==============
    ("entity", "href", AUX, None),              # text_link url
    ("button", "data", AUX, None),              # button url
    ("button", "dataBase64", AUX, None),        # button callback data
    ("message", "file_name", FNAME, None),      # attachment name
    # export paths: "chats/chat_NN/photos/photo_1@…jpg" and "(File not
    # included…)" marks — skipped; the original file name at the path tail — as file_name
    ("message", "file", PATH, None),
    ("message", "thumbnail", PATH, None),
    ("message", "photo", PATH, None),
    ("message", "contact_vcard", PATH, None),
    ("profile_picture", "photo", PATH, None),
    ("story", "media", PATH, None),

    # === NESTED OBJECTS ======================================================
    ("message", "reactions", DESCEND, "reaction"),
    ("reaction", "recent", DESCEND, "recent"),
    ("message", "poll", DESCEND, "poll"),
    ("poll", "answers", DESCEND, "poll_answer"),
    ("message", "todo_list", DESCEND, "todo_list"),
    ("todo_list", "answers", DESCEND, "todo_item"),
    ("message", "inline_bot_buttons", DESCEND, "button"),
    ("message", "contact_information", DESCEND, "contact_information"),
    ("message", "location_information", DESCEND, "location_information"),
    ("contacts", "list", DESCEND, "contact"),
    ("frequent_contacts", "list", DESCEND, "frequent_contact"),
)

POLICY: dict = {}
for _anchor, _key, _act, _arg in FIELD_POLICY:
    POLICY.setdefault(_anchor, {})[_key] = (_act, _arg)
del _anchor, _key, _act, _arg


CHAT_TYPES = frozenset({
    "personal_chat", "bot_chat", "saved_messages", "private_group",
    "private_supergroup", "public_supergroup", "private_channel",
    "public_channel"})
_PERSON_CHATS = ("personal_chat", "bot_chat")

_PEER_RX = re.compile(r"(user|channel|chat)(\d+)")

# A numeric peer id (from_id / actor_id / forwarded_from_id / reply_to_peer_id as a
# number — non-standard input) used to stay open: _peer took only strings. An unknown key
# with a peer id ("user123", "channel456" as a string; a number under a *_id key about a
# person or chat) was not masked either. Now it is USER_ID with the same seed as "user<N>"
# (counter STATS["unknown_peer_id"]). And an old TDesktop export without date_unixtime is
# recognised by the profile (sniffed by "messages"): before, it took the flat path and
# "from" and the chat id stayed open.
# (both fixes go together: without numeric ids the profile would open them on an old
# export, while the flat path closed the numbers).
_PEER_ID_KEY = re.compile(
    r"(?:^|_)(?:user|from|actor|peer|sender|author|owner|member|inviter|admin|bot|chat|channel|"
    r"contact|creator)(?:_peer)?_id$")
_NICK_RX = re.compile(r"@?[A-Za-z0-9_]{2,64}")
# file name produced by the exporter itself (photo_12@01-02-2023_10-11-12.jpg,
# sticker_3.webp_thumb.jpg) — no user data in it
_GENERATED = re.compile(
    r"(?:photo|sticker|video|audio|animation|file|contact)_\d+"
    r"(?:@\d\d-\d\d-\d{4}_\d\d-\d\d-\d\d)?(?:\.\w{1,5})?(?:_thumb\.jpg)?")
# service @-words — not nicknames (mirrors the names.handle_at guard: the same nick in
# a mention entity and in free text must get the same decision)
_NOT_NICK = frozenset(COMMON_EN) | frozenset(TECH_TERMS) | {
    "everyone", "here", "channel", "all"}


def peer_parts(pid):
    """Peer id → (kind, digits): ('user', '123') / ('channel', '456') /
    (None, None) — no id, already a token or an unknown form."""
    if isinstance(pid, bool):
        return None, None
    if isinstance(pid, int):
        return "user", str(pid)
    if isinstance(pid, str):
        m = _PEER_RX.fullmatch(pid)
        if m:
            return m.group(1), m.group(2)
        if pid.isdigit():
            return "user", pid
    return None, None


def peer_seed(kind: str, digits: str) -> str:
    """Seed of a peer's USER_ID token: for a user — bare digits (one token with
    the numeric user_id in mention_name / personal_information / personal
    chat id), for a channel or group — with the kind (numbers from different
    spaces do not merge)."""
    return digits if kind == "user" else kind + digits


def chat_seed(chat_type, digits: str) -> str:
    if chat_type in ("personal_chat", "bot_chat", "saved_messages"):
        return digits
    if chat_type == "private_group":
        return "chat" + digits
    return "channel" + digits


# values the exporter itself writes into name fields (not user data):
# inviter = "Group" for the join_group_by_link action
_EXPORTER_NAMES = frozenset({"Group"})


def title_seed(name: str) -> str:
    """Seed of a group/channel/bot title token: case- and extra-space-
    insensitive — one title gives one token in all fields and exports."""
    return " ".join(name.split()).casefold()


def _masked(s: str) -> bool:
    """The field value is already masked: our token as a whole, or (a client role
    label inside the value — ORG token + the rest as a field token) our token as a separate word."""
    return E.is_token(s) or (s[:1] == "@" and E.is_token(s[1:])) or is_own_token(s, word=True)


def _has_token(s: str) -> bool:
    """The typed entity text already holds a token — the entity was masked
    earlier (an original email/phone has no tokens). Only our own
    token: mention "@KOZLOV_2001" is a nickname, not a mask."""
    return is_own_token(s, part=True)


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
            TEXT: self._text, AUX: self._aux, RICH: self._rich,
            MIRROR: self._mirror, RICHV: self._richv_act,
            PERSON: self._person, RESOLVE: self._resolve,
            MEMBERS: self._members, NAMEPART: self._namepart,
            CHATNAME: self._chatname, FCNAME: self._fcname, PEER: self._peer,
            UID: self._uid, PHONE: self._phone, HANDLE: self._handle,
            GEO: self._geo, FNAME: self._fname, PATH: self._path,
            DESCEND: self._descend,
        }

    # ---- entry ----
    def plan(self, pat, obj, header: bool):
        anchor = self.p.anchors.get((pat, header), "?")
        plan = E.Plan()
        self._visit(anchor, obj, (), plan)
        return plan if plan.edits or plan.slots else None

    def _visit(self, anchor, node, path, plan) -> None:
        if isinstance(node, list):
            for i, item in enumerate(node):
                if isinstance(item, (dict, list)):
                    self._visit(anchor, item, path + (i,), plan)
                elif isinstance(item, str) and item and anchor == "?":
                    plan.slot(item, path + (i,), None)
            return
        table = POLICY.get(anchor) or {}
        acts = self._acts
        for key, val in node.items():
            if val is None or val == "":
                continue
            row = table.get(key)
            if row is None:
                self._unknown(anchor, key, val, path + (key,), plan)
            elif row[0] != SKIP:
                acts[row[0]](node, key, val, path + (key,), plan, row[1])

    def _unknown(self, anchor, key, val, p, plan) -> None:
        if anchor != "?":
            E.STATS["unknown_key"] += 1
            self.p.unknown.add((anchor, key))
        if self.masking and self._unknown_peer(key, val, p, plan):
            return
        if isinstance(val, str):
            plan.slot(val, p, key)
        elif isinstance(val, (dict, list)):
            self._visit("?", val, p, plan)

    def _unknown_peer(self, key, val, p, plan) -> bool:
        """Peer id under an unknown key → USER_ID (True — masked)."""
        if isinstance(val, bool) or not isinstance(key, str):
            return False
        if isinstance(val, int):
            if not _PEER_ID_KEY.search(key.lower()):
                return False
        elif not (isinstance(val, str) and _PEER_RX.fullmatch(val)):
            return False
        kind, digits = peer_parts(val)
        if kind is None:
            return False
        E.STATS["unknown_peer_id"] += 1
        self._mask("tg_peer_id", plan, p, val, mint.tag("USER_ID", peer_seed(kind, digits)))
        return True

    def _mask(self, rule, plan, p, before, token) -> None:
        plan.edits[p] = token
        self.report.record(rule, before, token)

    def _title(self, plan, p, key, val, prefix="ORG", rule="tg_title") -> None:
        """Group/channel/bot title in an IDENTITY FIELD (from/actor/
        forwarded_from/saved_from of a channel, name of a non-personal chat). It goes
        to mining and rules as text (an isolated slot: the same title in
        messages gets its verdict the usual way, and if the pack approved it the
        token is shared), but always leaves the field AS A TOKEN: a private group
        title is the name of a client, a project or people (a check on 26 real
        exports: 583 chat titles and 6.6k sender values stayed open while
        the rules decided)."""
        whole = None
        if self.masking and val.strip():
            whole = (rule, mint.tag(prefix, title_seed(val)))
        plan.slot(val, p, key, iso=True, whole=whole)

    # ---- free text ----
    def _text(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, str):
            plan.slot(val, p, key)
        else:
            self._richv(val, p, plan, key)

    def _aux(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, str):
            plan.slot(val, p, key, ner=False)

    def _richv_act(self, node, key, val, p, plan, arg) -> None:
        self._richv(val, p, plan, key)

    def _richv(self, val, p, plan, key) -> None:
        if isinstance(val, str):
            plan.slot(val, p, key)
        elif isinstance(val, list):
            self._rich_list(val, p, plan, key)
        elif isinstance(val, dict):
            self._visit("?", val, p, plan)

    def _mirror(self, node, key, val, p, plan, arg) -> None:
        if isinstance(node.get("text_entities"), list):
            return                       # rebuilt from text_entities
        self._richv(val, p, plan, key)

    def _rich(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, list):
            self._rich_list(val, p, plan, key, node.get("text"),
                            p[:-1] + ("text",))

    def _rich_list(self, items, p, plan, key, mirror=None, mp=None) -> None:
        parts, raws, paths, extras = [], [], [], []
        for i, e in enumerate(items):
            if not isinstance(e, dict):
                s = e if isinstance(e, str) else ""
                parts.append(s)
                raws.append(s)
                paths.append(p + (i,))
                continue
            s = e.get("text")
            if not isinstance(s, str):
                s = ""
            part = s
            if self.masking and s and not _has_token(s):
                tok = self._typed(e, s)
                if tok is not None:
                    self.report.record("tg_entity_" + e["type"], s, tok)
                    part = tok
            parts.append(part)
            raws.append(s)
            paths.append(p + (i, "text"))
            if len(e) > 2:
                for k, v in e.items():
                    if k != "type" and k != "text" and v is not None and v != "":
                        extras.append((i, k, v, e))
        kind = None
        if isinstance(mirror, str):
            if mirror == "".join(raws):
                kind = "str"
        elif isinstance(mirror, list) and len(mirror) == len(items):
            if all((m == raws[i]) if isinstance(m, str)
                   else (isinstance(m, dict) and m.get("text") == raws[i])
                   for i, m in enumerate(mirror)):
                kind = "list"
        table = POLICY["entity"]
        for i, k, v, e in extras:
            ep = p + (i, k)
            row = table.get(k)
            n = len(plan.slots)
            if row is None:
                self._unknown("entity", k, v, ep, plan)
            elif row[0] != SKIP:
                self._acts[row[0]](e, k, v, ep, plan, row[1])
            if kind == "list" and isinstance(mirror[i], dict) \
                    and mirror[i].get(k) == v:
                # the same field in the text mirror — the same edit / the same slot
                mpath = mp + (i, k)
                if ep in plan.edits:
                    plan.edits[mpath] = plan.edits[ep]
                for s in plan.slots[n:]:
                    if s.paths == [ep]:
                        s.paths.append(mpath)
        if parts:
            plan.slots.append(E.RichSlot(parts, raws, paths, key, mirror, mp,
                                         kind))
        if kind is None and mirror is not None and mirror != "" \
                and mirror != []:
            # the mirror did not match the parts (never seen in data) — it is
            # processed on its own rather than lost
            if parts:
                E.STATS["mirror_mismatch"] += 1
            self._richv(mirror, mp, plan, "text")

    def _typed(self, e: dict, s: str):
        """Token of a typed text entity, or None (plain text)."""
        t = e.get("type")
        if t == "email":
            return P.EMAIL
        if t == "phone":
            return P.PHONE
        if t == "bank_card":
            return P.CARD
        if t == "mention":
            nick = s[1:] if s[:1] == "@" else s
            if s[:1] != "@" or not _NICK_RX.fullmatch(nick) \
                    or nick.lower() in _NOT_NICK:
                return None
            return "@" + mint.tag("HANDLE", nick.lower())
        if t == "mention_name":
            kind, digits = peer_parts(e.get("user_id"))
            if kind == "user":
                return self.reg.user_token(digits, s)
            return self.reg.name_token(s)
        return None

    # ---- identity ----
    def _person(self, node, key, val, p, plan, arg) -> None:
        if not isinstance(val, str):
            return
        pid = node.get(arg) if arg else None
        kind, digits = peer_parts(pid)
        if kind in ("channel", "chat") or (kind is None and E.is_token(pid)):
            # channel/group name — free text. The id is already a token but the name
            # is not — a repeat pass over the same case (a person would have got a token
            # together with their id; a value with an ORG label token and a rest
            # token is also already closed, _masked)
            if not _masked(val):
                self._title(plan, p, key, val)
            return
        if not self.masking or _masked(val) or not val.strip():
            return
        tok = self.reg.user_token(digits, val) if kind == "user" \
            else self.reg.name_token(val)
        self._mask("tg_person", plan, p, val, tok)

    def _resolve(self, node, key, val, p, plan, arg) -> None:
        if not isinstance(val, str) or _masked(val) \
                or val in _EXPORTER_NAMES:
            return
        kind = self.reg.name_kind(val) if self.reg is not None \
            else ("person" if canonical(val) else "")
        if kind in ("user", "person"):
            if self.masking:
                self._mask("tg_person", plan, p, val, self.reg.name_token(val))
        else:
            self._title(plan, p, key, val)

    def _members(self, node, key, val, p, plan, arg) -> None:
        if not self.masking or not isinstance(val, list):
            return
        for i, name in enumerate(val):
            if isinstance(name, str) and name.strip() and not _masked(name):
                self._mask("tg_person", plan, p + (i,), name,
                           self.reg.name_token(name))

    def _namepart(self, node, key, val, p, plan, arg) -> None:
        if not self.masking or not isinstance(val, str):
            return
        first = node.get("first_name")
        last = node.get("last_name")
        first = first if isinstance(first, str) and not _masked(first) else ""
        last = last if isinstance(last, str) and not _masked(last) else ""
        if key == "last_name" and first.strip():
            return                       # the pair was handled on first_name
        full = (first + " " + last).strip()
        if not full:
            return
        kind, digits = peer_parts(node.get(arg)) if arg else (None, None)
        tok = self.reg.user_token(digits, full) if kind == "user" \
            else self.reg.name_token(full)
        base = p[:-1]
        if first.strip():
            plan.edits[base + ("first_name",)] = tok
        if last.strip():
            plan.edits[base + ("last_name",)] = tok
        self.report.record("tg_person", full, tok)

    def _chatname(self, node, key, val, p, plan, arg) -> None:
        if not isinstance(val, str):
            return
        if node.get("type") not in _PERSON_CHATS:
            if not _masked(val):
                # group/channel name: text for mining, a token on output
                self._title(plan, p, key, val)
            return
        if not self.masking or _masked(val) or not val.strip():
            return
        kind, digits = peer_parts(node.get("id"))
        tok = self.reg.user_token(digits, val) if kind == "user" \
            else self.reg.name_token(val)
        self._mask("tg_person", plan, p, val, tok)

    def _fcname(self, node, key, val, p, plan, arg) -> None:
        if not isinstance(val, str):
            return
        if node.get("type") not in (None, "user"):
            if not _masked(val):
                self._title(plan, p, key, val)
            return
        if not self.masking or _masked(val) or not val.strip():
            return
        kind, digits = peer_parts(node.get("id"))
        tok = self.reg.user_token(digits, val) if kind == "user" \
            else self.reg.name_token(val)
        self._mask("tg_person", plan, p, val, tok)

    def _peer(self, node, key, val, p, plan, arg) -> None:
        if self.masking and isinstance(val, int) and not isinstance(val, bool):
            kind, digits = peer_parts(val)           # a number = "user<N>"
            self._mask("tg_peer_id", plan, p, val,
                       mint.tag("USER_ID", peer_seed(kind, digits)))
            return
        if not self.masking or not isinstance(val, str) or E.is_token(val):
            return
        kind, digits = peer_parts(val)
        seed = peer_seed(kind, digits) if kind else val
        self._mask("tg_peer_id", plan, p, val, mint.tag("USER_ID", seed))

    def _uid(self, node, key, val, p, plan, arg) -> None:
        if not self.masking:
            return
        kind, digits = peer_parts(val)
        if kind != "user":
            return                       # already a token / not an id
        if arg == "chat":
            seed = chat_seed(node.get("type"), digits)
        elif arg == "fc" and node.get("type") not in (None, "user"):
            seed = str(node.get("type")) + digits
        else:
            seed = digits
        self._mask("tg_peer_id", plan, p, val, mint.tag("USER_ID", seed))

    def _phone(self, node, key, val, p, plan, arg) -> None:
        if self.masking and isinstance(val, str) and not E.is_token(val) \
                and any(c.isdigit() for c in val):
            self._mask("tg_phone", plan, p, val, P.PHONE)

    def _handle(self, node, key, val, p, plan, arg) -> None:
        if not isinstance(val, str):
            return
        if not _NICK_RX.fullmatch(val):
            if not _masked(val):
                # does not look like a nick — rules as for text, a token out of the field
                self._title(plan, p, key, val, "HANDLE", "tg_handle")
            return
        nick = val.lstrip("@")
        if self.masking and nick and not E.is_token(nick):
            self._mask("tg_handle", plan, p, val, val[:len(val) - len(nick)]
                       + mint.tag("HANDLE", nick.lower()))

    def _geo(self, node, key, val, p, plan, arg) -> None:
        if self.masking and not isinstance(val, bool) \
                and (isinstance(val, (int, float))
                     or (isinstance(val, str) and not E.is_token(val))):
            self._mask("tg_geo", plan, p, val, P.GEO)

    # ---- file names and paths ----
    def _name_slot(self, plan, p, key, name: str, prefix: str) -> None:
        """File name: file name detector, then rules (a person's token in the
        name is shared with the text), then the detector again on the rules output
        (``post``: the rules minted DOC_ID from a number — the detector picks up the
        id tail; if it changed something — another rules pass, up to a fixed point
        in the same pass; otherwise a repeat dry run on the output gave a
        tg_filename replacement and the batch stopped on the idempotency oracle)."""
        text, post = name, None
        if self.masking:
            text = _fname_detect(name, self.report)
            post = _fname_detect
        # orig — the name without the directory: Slot.commit compares the result with it
        # and puts the path's directory back as a prefix
        plan.slots.append(E.Slot(text, [p], key, False, name, prefix, post=post))

    def _fname(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, str):
            self._name_slot(plan, p, key, val, "")

    def _path(self, node, key, val, p, plan, arg) -> None:
        if not isinstance(val, str) or val[:1] == "(":
            return                       # exporter mark "(File not included…)"
        head, sep, base = val.rpartition("/")
        if not base or _GENERATED.fullmatch(base):
            return
        self._name_slot(plan, p, key, base, head + sep)

    def _descend(self, node, key, val, p, plan, arg) -> None:
        if isinstance(val, (dict, list)):
            self._visit(arg, val, p, plan)


def _fname_detect(text: str, report) -> str:
    """File name detector on an attachment name (before the rules and on their output)."""
    from ..detectors import filenames
    out = filenames.mask_name(text)
    if out != text:
        report.record("tg_filename", text, out)
    return out


def _full(first, last) -> str:
    first = first if isinstance(first, str) else ""
    last = last if isinstance(last, str) else ""
    return (first + " " + last).strip()


class TelegramProfile:
    """File kind: ``full`` — account export, ``chat`` — single chat."""
    name = "telegram"

    def __init__(self, kind: str):
        self.kind = kind
        self.unknown: set = set()        # (anchor, key) pairs missing from the table
        if kind == "full":
            self.spine = frozenset({
                (), ("chats",), ("chats", "list"), ("chats", "list", "*"),
                ("chats", "list", "*", "messages"),
                ("left_chats",), ("left_chats", "list"),
                ("left_chats", "list", "*"),
                ("left_chats", "list", "*", "messages")})
            self.anchors = {((), True): "root",
                            (("personal_information",), False):
                                "personal_information",
                            (("profile_pictures",), False): "profile_picture",
                            (("stories",), False): "story",
                            (("contacts",), False): "contacts",
                            (("frequent_contacts",), False):
                                "frequent_contacts"}
            for sect in ("chats", "left_chats"):
                self.anchors[((sect,), True)] = "section"
                self.anchors[((sect, "list", "*"), True)] = "chat"
                self.anchors[((sect, "list", "*", "messages", "*"), False)] = \
                    "message"
        else:
            self.spine = frozenset({(), ("messages",)})
            self.anchors = {((), True): "chat",
                            (("messages", "*"), False): "message"}

    def session(self, registry, report) -> _Session:
        return _Session(self, registry, report)

    # ---- entity registry collection ----
    def collect(self, pat, obj, header: bool, reg: Registry, rel) -> None:
        anchor = self.anchors.get((pat, header))
        if anchor == "message":
            self._collect_message(obj, reg, rel)
        elif anchor == "chat":
            name = obj.get("name")
            if obj.get("type") in _PERSON_CHATS:
                kind, digits = peer_parts(obj.get("id"))
                if kind == "user":
                    reg.add_user(digits, name, rel)
            else:
                reg.add_title(name)
        elif anchor == "personal_information":
            kind, digits = peer_parts(obj.get("user_id"))
            full = _full(obj.get("first_name"), obj.get("last_name"))
            if kind == "user":
                reg.add_user(digits, full, rel)
            else:
                reg.add_loose(full, rel)
            reg.add_handle(obj.get("username"), rel, owner=True)
        elif anchor == "contacts":
            for c in obj.get("list") or ():
                if isinstance(c, dict):
                    reg.add_loose(_full(c.get("first_name"),
                                        c.get("last_name")), rel)
        elif anchor == "frequent_contacts":
            for c in obj.get("list") or ():
                if isinstance(c, dict) and c.get("type") in (None, "user"):
                    kind, digits = peer_parts(c.get("id"))
                    if kind == "user":
                        reg.add_user(digits, c.get("name"), rel)

    @staticmethod
    def _pair(reg, name, pid, rel, loose=True) -> None:
        if not isinstance(name, str):
            return
        kind, digits = peer_parts(pid)
        if kind == "user":
            reg.add_user(digits, name, rel)
        elif kind is not None:
            reg.add_title(name)
        elif loose and pid is None:
            reg.add_loose(name, rel)

    def _collect_message(self, m: dict, reg: Registry, rel) -> None:
        pair = self._pair
        pair(reg, m.get("from"), m.get("from_id"), rel, loose=False)
        if "actor" in m:
            pair(reg, m.get("actor"), m.get("actor_id"), rel, loose=False)
        if "forwarded_from" in m:
            pair(reg, m.get("forwarded_from"), m.get("forwarded_from_id"), rel)
        if "author" in m:
            reg.add_loose(m.get("author"), rel)
        if "members" in m:
            for name in m.get("members") or ():
                reg.add_loose(name, rel)
        for key in ("saved_from", "inviter"):
            name = m.get(key)
            if isinstance(name, str) and canonical(name):
                reg.add_loose(name, rel)
        ci = m.get("contact_information")
        if isinstance(ci, dict):
            reg.add_loose(_full(ci.get("first_name"), ci.get("last_name")), rel)
        for r in m.get("reactions") or ():
            if isinstance(r, dict):
                for rc in r.get("recent") or ():
                    if isinstance(rc, dict):
                        pair(reg, rc.get("from"), rc.get("from_id"), rel,
                             loose=False)
        ents = m.get("text_entities")
        if ents:
            for e in ents:
                if not isinstance(e, dict):
                    continue
                t = e.get("type")
                if t == "mention_name":
                    pair(reg, e.get("text"), e.get("user_id"), rel)
                elif t == "mention":
                    s = e.get("text")
                    if isinstance(s, str) and s[1:].lower() not in _NOT_NICK \
                            and not s.lower().endswith("bot"):
                        reg.add_handle(s, rel)


class _Sniffer:
    """Format signature from the file head → profile kind or None."""
    name = "telegram"

    def __init__(self):
        self._kinds = {"full": TelegramProfile("full"),
                       "chat": TelegramProfile("chat")}

    def sniff(self, head: str):
        old = '"messages"' in head     # TDesktop without date_unixtime
        if '"date_unixtime"' not in head and '"chats"' not in head \
                and '"personal_information"' not in head and not old:
            return None                  # cheap filter before parsing keys
        keys = E.top_keys(head)
        kind = None
        msgs = keys.get("messages", "")
        if msgs[:1] == "[" and "id" in keys and "type" in keys:
            m = re.match(r'"([a-z_]+)"', keys["type"])
            if (m and m.group(1) in CHAT_TYPES) or '"date_unixtime"' in head:
                kind = "chat"
        if kind is None:
            sect = [k for k in ("chats", "left_chats")
                    if keys.get(k, "")[:1] == "{"]
            if sect and '"list"' in head and (
                    "about" in keys or "personal_information" in keys
                    or '"date_unixtime"' in head):
                kind = "full"
            elif keys.get("personal_information", "")[:1] == "{" \
                    and '"user_id"' in head and '"phone_number"' in head:
                kind = "full"
            elif "about" in keys and '"list"' in head and (
                    (keys.get("contacts", "")[:1] == "{"
                     and '"date_unixtime"' in head)
                    or (keys.get("frequent_contacts", "")[:1] == "{"
                        and '"rating"' in head)):
                kind = "full"
        return self._kinds[kind] if kind else None


PROFILE = _Sniffer()
