"""DiscordChatExporter JSON: root ``guild``, ``channel``, ``messages[]``."""
from __future__ import annotations

from .generic import is_image_name, strip_media_keys

DIRECT_TYPES = frozenset({"DirectTextChat", "DirectGroupTextChat", "dm", "group_dm"})


def detect(obj) -> bool:
    return (isinstance(obj, dict) and isinstance(obj.get("messages"), list)
            and isinstance(obj.get("channel"), dict)
            and ("guild" in obj or "id" in obj["channel"]))


def chat_label(obj) -> tuple[str, str, str]:
    ch = obj.get("channel") or {}
    cid = ch.get("id")
    name = ch.get("name") or ""
    g = obj.get("guild") or {}
    if g.get("name") and name and ch.get("type") not in DIRECT_TYPES:
        name = f"{g['name']}/{name}"
    return ("" if cid is None else str(cid)), str(name), str(ch.get("type") or "")


def set_message_count(obj) -> None:
    """Recount ``messageCount`` after removing messages (if the field was there)."""
    if "messageCount" in obj and isinstance(obj.get("messages"), list):
        obj["messageCount"] = len(obj["messages"])


def chat_aliases(obj) -> tuple[str, ...]:
    """The channel name without the guild — as the owner sees it in Discord."""
    ch = obj.get("channel") or {}
    name = ch.get("name")
    return (str(name),) if name else ()


_USER_TYPES = frozenset({None, "", "Default", "Reply", "default", "reply"})


def is_service(m: dict) -> bool:
    """A system message (join, pin, ThreadCreated …): the message filter does not touch it."""
    return m.get("type") not in _USER_TYPES


def message_text(m: dict) -> str:
    parts = []
    c = m.get("content")
    if isinstance(c, str) and c.strip():
        parts.append(c)
    for e in m.get("embeds") or ():
        if isinstance(e, dict):
            for k in ("title", "description"):
                v = e.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v)
    return "\n".join(parts)


# forwarded copies of other messages (Discord API: message_snapshots[].message)
SNAPSHOT_KEYS = ("message_snapshots", "messageSnapshots")


def snapshot_lists(m: dict) -> list:
    return [m[k] for k in SNAPSHOT_KEYS if isinstance(m.get(k), list)]


def snapshot_message(s):
    """The message inside a snapshot (``message_snapshots[].message``; a bare object — as is)."""
    if isinstance(s, dict) and isinstance(s.get("message"), dict):
        return s["message"]
    return s


def snapshot_text(s) -> str:
    s = snapshot_message(s)
    return message_text(s) if isinstance(s, dict) else ""


def _attachment_is_image(a) -> bool:
    if not isinstance(a, dict):
        return False
    return is_image_name(a.get("fileName")) or is_image_name(a.get("url"))


# nested copy of the message being replied to (Discord API: referenced_message)
REF_COPY_KEYS = ("referencedMessage", "referenced_message")


def ref_copy_matches(m: dict, host: dict) -> list[dict]:
    """Copies of the host ``host`` in message ``m``: ``referenced_message`` with its ``id``, or
    one with a ``reference``/``message_reference`` to it."""
    hid = str(host.get("id"))
    ref = m.get("reference") or m.get("message_reference")
    by_ref = isinstance(ref, dict) and str(ref.get("messageId") or ref.get("message_id")) == hid
    out = []
    for k in REF_COPY_KEYS:
        cp = m.get(k)
        if isinstance(cp, dict) and (str(cp.get("id")) == hid or (by_ref and cp.get("id") is None)):
            out.append(cp)
    return out


# message fields of DiscordChatExporter and the Discord API; under other keys a copy of another
# message (an object with content/attachments/embeds) is not parsed — only counted
_KNOWN_KEYS = frozenset({
    "id", "type", "timestamp", "timestampEdited", "callEndedTimestamp", "isPinned", "content",
    "author", "attachments", "embeds", "stickers", "reactions", "mentions", "inlineEmojis",
    "reference", "channel_id", "guild_id", "edited_timestamp", "tts", "mention_everyone",
    "mention_roles", "mention_channels", "pinned", "webhook_id", "activity", "application",
    "application_id", "flags", "message_reference", "interaction", "interaction_metadata",
    "thread", "components", "sticker_items", "position", "poll", "call", "nonce",
    "role_subscription_data", "resolved", *REF_COPY_KEYS, *SNAPSHOT_KEYS})
_MSG_SIGNS = ("content", "attachments", "embeds")


def unparsed_copy_keys(m: dict) -> list[str]:
    """Unknown message fields holding an object (or a list of objects) that looks like a
    message — a possible forward under another key."""
    out = []
    for k, v in m.items():
        if k in _KNOWN_KEYS:
            continue
        items = v if isinstance(v, list) else [v]
        if any(isinstance(x, dict) and any(s in x for s in _MSG_SIGNS) for x in items):
            out.append(k)
    return out


def clean_message(m: dict) -> dict:
    """Avatars, picture attachments, embed pictures, emoji and stickers. → counters.
    The nested copy of the parent (``referenced_message``) and forwarded copies
    (``message_snapshots[].message``) are cleaned the same way; a forwarded picture-only copy goes."""
    c = _clean_one(m)
    for k in REF_COPY_KEYS:
        cp = m.get(k)
        if isinstance(cp, dict):
            _add(c, _clean_one(cp))
    dropped = False
    for lst in snapshot_lists(m):
        left = []
        for sn in lst:
            msg = snapshot_message(sn)
            if isinstance(msg, dict):
                rc = _clean_one(msg)
                _add(c, rc)
                if rc["image_only"]:
                    dropped = True
                    continue
            left.append(sn)
        if len(left) != len(lst):
            lst[:] = left
    if c["image_only"] or dropped:
        c["image_only"] = int(not has_content(m) and not any(snapshot_lists(m)))
    return c


def _add(c: dict, rc: dict) -> None:
    for key in ("avatars", "attachments", "embeds", "emoji"):
        c[key] += rc[key]


def _clean_one(m: dict) -> dict:
    c = {"avatars": 0, "attachments": 0, "embeds": 0, "emoji": 0}
    had_media = False
    att = m.get("attachments")
    if isinstance(att, list):
        keep = []
        for a in att:
            if _attachment_is_image(a):
                c["attachments"] += 1
                had_media = True
            else:
                keep.append(a)
        att[:] = keep
    embeds = m.get("embeds")
    if isinstance(embeds, list):
        left = []
        for e in embeds:
            if isinstance(e, dict):
                for k in ("thumbnail", "image", "images"):
                    if k in e:
                        del e[k]
                        c["embeds"] += 1
                        had_media = True
                if not e:
                    continue             # the embed was only a picture
            left.append(e)
        embeds[:] = left
    st = m.get("stickers")
    if isinstance(st, list) and st:
        had_media = True
        for s in st:
            if isinstance(s, dict) and "sourceUrl" in s:
                del s["sourceUrl"]
                c["emoji"] += 1
    for em in m.get("inlineEmojis") or ():
        if isinstance(em, dict) and "imageUrl" in em:
            del em["imageUrl"]
            c["emoji"] += 1
    for r in m.get("reactions") or ():
        emoji = r.get("emoji") if isinstance(r, dict) else None
        if isinstance(emoji, dict) and "imageUrl" in emoji:
            del emoji["imageUrl"]
            c["emoji"] += 1
    c["avatars"] += strip_media_keys(m)  # author.avatarUrl, mentions[].avatarUrl, iconUrl …
    c["image_only"] = int(had_media and not has_content(m))
    return c


def has_content(m: dict) -> bool:
    if message_text(m).strip():
        return True
    if m.get("attachments"):
        return True
    if any(isinstance(e, dict) and (e.get("url") or e.get("fields")) for e in m.get("embeds") or ()):
        return True
    return is_service(m)                 # a system message (join, pin …)
