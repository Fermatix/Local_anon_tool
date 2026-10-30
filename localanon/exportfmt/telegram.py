"""Telegram Desktop JSON: a full export (``result.json`` with ``chats.list``) and a single-chat
export (root ``name``/``type``/``id``/``messages``). The signature is by content.
"""
from __future__ import annotations

from pathlib import Path

PERSONAL_SECTIONS = ("personal_information", "contacts", "frequent_contacts", "sessions",
                     "profile_pictures", "other_data", "stories", "web_sessions")
CHAT_SECTIONS = ("chats", "left_chats")
CHAT_TYPES = frozenset({
    "personal_chat", "bot_chat", "saved_messages", "private_group", "private_supergroup",
    "public_supergroup", "private_channel", "public_channel"})
PERSONAL_CHAT_TYPES = frozenset({"personal_chat"})

# message fields that point to an export file, and their companions
_MEDIA_FIELDS = {
    "photo": ("photo_file_size", "width", "height", "self_destruct_period_seconds"),
    "thumbnail": ("thumbnail_file_size",),
    "file": ("file_name", "file_size", "mime_type", "media_type", "sticker_emoji",
             "duration_seconds", "width", "height", "performer", "title"),
    "contact_vcard": ("contact_vcard_file_size",),
}
# message content that makes it not "picture only"
_CONTENT_KEYS = ("poll", "contact_information", "location_information", "todo_list",
                 "inline_bot_buttons", "game_title", "invoice_information", "giveaway_information",
                 "place_name", "address", "paid_stars_amount", "live_location_period_seconds")


def detect(obj) -> str | None:
    """→ 'telegram_full' | 'telegram_chat' | None."""
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("messages"), list) and "name" in obj and "type" in obj and "id" in obj:
        if obj.get("type") in CHAT_TYPES or any(
                isinstance(m, dict) and "date" in m and ("text" in m or "action" in m)
                for m in obj["messages"][:5]):
            return "telegram_chat"
    for sect in CHAT_SECTIONS:
        s = obj.get(sect)
        if isinstance(s, dict) and isinstance(s.get("list"), list):
            return "telegram_full"
    if isinstance(obj.get("personal_information"), dict) and "user_id" in obj["personal_information"]:
        return "telegram_full"
    return None


def iter_chats(obj, kind: str):
    """→ [(section or None, container list or None, chat)]."""
    if kind == "telegram_chat":
        return [(None, None, obj)]
    out = []
    for sect in CHAT_SECTIONS:
        s = obj.get(sect)
        if isinstance(s, dict) and isinstance(s.get("list"), list):
            for chat in s["list"]:
                if isinstance(chat, dict):
                    out.append((sect, s["list"], chat))
    return out


def strip_sections(obj) -> list[str]:
    """Remove the personal sections of a full export. → names of the removed ones."""
    gone = [k for k in PERSONAL_SECTIONS if k in obj]
    for k in gone:
        del obj[k]
    return gone


def message_text(m: dict) -> str:
    """Flat message text (text_entities, then text — a string or a list of parts)."""
    ents = m.get("text_entities")
    if isinstance(ents, list) and ents:
        return "".join(e.get("text", "") if isinstance(e, dict) else str(e) for e in ents)
    t = m.get("text")
    if isinstance(t, str):
        return t
    if isinstance(t, list):
        return "".join(p if isinstance(p, str) else str(p.get("text", "")) if isinstance(p, dict) else ""
                       for p in t)
    return ""


def _file_present(base: Path, value) -> bool:
    if not isinstance(value, str) or not value or value.startswith("("):
        return False                     # "(File not included…)" — no file
    try:
        return (base / value).is_file()
    except (OSError, ValueError):
        return False


def clean_message_media(m: dict, base: Path) -> int:
    """Drop fields pointing to missing files (pictures were removed earlier). → how many fields."""
    n = 0
    for key, mates in _MEDIA_FIELDS.items():
        if key in m and not _file_present(base, m[key]):
            del m[key]
            n += 1
            for k in mates:
                if k in m and not any(other in m for other in _MEDIA_FIELDS if other != key
                                      and k in _MEDIA_FIELDS[other]):
                    del m[k]
    return n


# service messages whose photo is a chat avatar ([media].images = "avatars")
AVATAR_ACTIONS = frozenset(("edit_group_photo", "suggest_profile_photo"))


def avatar_photo(m) -> str | None:
    """Path of the chat avatar in a group-photo service message, else None."""
    if not isinstance(m, dict) or m.get("type") != "service" or m.get("action") not in AVATAR_ACTIONS:
        return None
    v = m.get("photo")
    return v if isinstance(v, str) and v and not v.startswith("(") else None


def strip_avatar_photos(chat: dict, base: Path) -> int:
    """Drop the photo fields of group-photo service messages whose file is gone. → how many."""
    n = 0
    for m in chat.get("messages") or ():
        if (isinstance(m, dict) and m.get("type") == "service" and m.get("action") in AVATAR_ACTIONS
                and "photo" in m and not _file_present(base, m["photo"])):
            del m["photo"]
            for k in _MEDIA_FIELDS["photo"]:
                m.pop(k, None)
            n += 1
    return n


def media_paths(m: dict, base: Path) -> list[Path]:
    """Attachment files of the message (photo, file, preview, vCard) that are on disk."""
    return [base / m[k] for k in _MEDIA_FIELDS if k in m and _file_present(base, m[k])]


def has_content(m: dict) -> bool:
    """Whether the message has anything left besides service fields (text, a live file, a poll…)."""
    if m.get("type") != "message":
        return True                      # a service message is not "picture only"
    if message_text(m).strip():
        return True
    if any(k in m for k in _MEDIA_FIELDS):
        return True
    return any(k in m for k in _CONTENT_KEYS)


def chat_label(chat: dict) -> tuple[str, str, str]:
    """→ (id as a string, name, type)."""
    cid = chat.get("id")
    cid = "" if cid is None else str(cid)
    name = chat.get("name")
    name = "" if name is None else str(name)
    return cid, name, str(chat.get("type") or "")
