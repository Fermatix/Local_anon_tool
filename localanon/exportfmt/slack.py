"""Slack: ``users.json``, ``channels.json`` (``groups.json``, ``dms.json``, ``mpims.json``) and
``<channel>/<date>.json`` — a list of messages with ``ts``. The signature is by content."""
from __future__ import annotations

import re

from .generic import is_image_name, strip_avatar_keys, strip_media_keys

_TS_RX = re.compile(r"^\d{6,}\.\d+$")
_IMAGE_TYPES = frozenset({"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif", "heic", "svg"})


def detect(obj) -> str | None:
    """→ 'slack_users' | 'slack_channels' | 'slack_messages' | None."""
    if not isinstance(obj, list) or not obj or not all(isinstance(x, dict) for x in obj[:20]):
        return None
    head = obj[:20]
    if all("profile" in x and "id" in x and ("name" in x or "real_name" in x) for x in head):
        return "slack_users"
    if all(isinstance(x.get("ts"), str) and _TS_RX.match(x["ts"]) and
           ("type" in x or "text" in x or "user" in x) for x in head):
        return "slack_messages"
    if all("id" in x and ("members" in x or "is_channel" in x or "is_group" in x or "is_im" in x
                          or ("name" in x and "created" in x)) for x in head):
        return "slack_channels"
    return None


# subtypes with a person behind them: such messages are classified; other subtypes
# (channel_join, channel_topic, bot_message, pinned_item …) are service ones, the filter leaves them
_USER_SUBTYPES = frozenset({"thread_broadcast", "me_message", "file_share", "file_comment",
                            "reply_broadcast"})


def is_service(m: dict) -> bool:
    sub = m.get("subtype")
    return bool(sub) and sub not in _USER_SUBTYPES


_MSG_URL_RX = re.compile(r"/archives/[^/?#]+/p\d+")


def is_copy(a, host: dict | None = None) -> bool:
    """The attachment is a copy of another message: a forward ("Share message": ``is_share``,
    ``is_msg_unfurl``, ``from_url`` to a message), ``ts`` together with message signs
    (``author_id``/``channel_id``/``message_blocks``), or the parent copy of an old
    ``reply_broadcast`` (``ts``). ``ts`` alone is not enough: in attachments of integration bots
    (GitHub, Jira, Sentry …) it is the footer time — they are judged together with the bot message."""
    if not isinstance(a, dict):
        return False
    if a.get("is_share") or a.get("is_msg_unfurl"):
        return True
    url = a.get("from_url")
    if isinstance(url, str) and _MSG_URL_RX.search(url):
        return True
    if not a.get("ts"):
        return False
    if any(a.get(k) for k in ("author_id", "channel_id", "message_blocks")):
        return True
    return isinstance(host, dict) and host.get("subtype") == "reply_broadcast"


def copies_of(m: dict) -> list:
    """Copy attachments of message ``m`` (in list order)."""
    atts = m.get("attachments")
    if not isinstance(atts, list):
        return []
    return [a for a in atts if is_copy(a, m)]


def copy_text(a: dict) -> str:
    """Text of a copy: title + text; without text — ``pretext``/``fallback``."""
    parts = []
    for k in ("title", "text"):
        v = a.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    if not (isinstance(a.get("text"), str) and a["text"].strip()):
        for k in ("pretext", "fallback"):
            v = a.get(k)
            if isinstance(v, str) and v.strip() and v not in parts:
                parts.append(v)
                break
    return "\n".join(parts)


def message_text(m: dict, copies: bool = True) -> str:
    """Text of the message and its attachments; ``copies=False`` — without nested copies of other
    messages (the message filter judges them separately)."""
    parts = []
    t = m.get("text")
    if isinstance(t, str) and t.strip():
        parts.append(t)
    for a in m.get("attachments") or ():
        if isinstance(a, dict) and (copies or not is_copy(a, m)):
            for k in ("title", "text", "fallback"):
                v = a.get(k)
                if isinstance(v, str) and v.strip() and v not in parts:
                    parts.append(v)
                    break
    return "\n".join(parts)


def _file_is_image(f) -> bool:
    if not isinstance(f, dict):
        return False
    mt = f.get("mimetype")
    if isinstance(mt, str) and mt.lower().startswith("image/"):
        return True
    ft = f.get("filetype")
    if isinstance(ft, str) and ft.lower() in _IMAGE_TYPES:
        return True
    return is_image_name(f.get("name")) or is_image_name(f.get("title"))


def clean_profile(p: dict) -> int:
    n = 0
    for k in [k for k in p if k.startswith("image_") or k == "avatar_hash"]:
        del p[k]
        n += 1
    return n


def clean_user(u: dict) -> int:
    n = 0
    p = u.get("profile")
    if isinstance(p, dict):
        n += clean_profile(p)
    n += strip_media_keys(u)
    return n


_PROFILE_KEYS = ("user_profile", "bot_profile", "profile", "icons")


def strip_avatars(node) -> int:
    """Avatars only: profile/icons ``image_*`` and ``avatar_hash`` at any depth (messages, thread
    roots, forwarded copies, users.json) plus generic avatar keys. Files stay. → how many."""
    n = 0
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for key in _PROFILE_KEYS:
                p = cur.get(key)
                if isinstance(p, dict):
                    n += clean_profile(p)
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend(x for x in cur if isinstance(x, (dict, list)))
    return n + strip_avatar_keys(node)


def clean_message(m: dict) -> dict:
    """profile.image_*, avatar_hash, bot icons, picture files and their thumbnails. → counters.
    The nested copy of the thread parent (``root`` of thread_broadcast) and forwarded copies in
    ``attachments`` are cleaned the same way."""
    c = _clean_one(m)
    root = m.get("root")
    if isinstance(root, dict):
        rc = _clean_one(root)
        c["profiles"] += rc["profiles"]
        c["files"] += rc["files"]
    return c


def _clean_one(m: dict) -> dict:
    c = {"profiles": 0, "files": 0}
    had_media = False
    for key in ("user_profile", "bot_profile", "profile"):
        p = m.get(key)
        if isinstance(p, dict):
            c["profiles"] += clean_profile(p)
            icons = p.get("icons")
            if isinstance(icons, dict):
                c["profiles"] += clean_profile(icons)
    icons = m.get("icons")
    if isinstance(icons, dict):
        c["profiles"] += clean_profile(icons)
    files = m.get("files")
    if isinstance(files, list):
        keep = []
        for f in files:
            if _file_is_image(f):
                c["files"] += 1
                had_media = True
            else:
                if isinstance(f, dict):
                    for k in [k for k in f if k.startswith("thumb_")]:
                        del f[k]
                keep.append(f)
        files[:] = keep
    atts = m.get("attachments")
    if isinstance(atts, list):
        left = []
        for a in atts:
            if isinstance(a, dict):
                for k in [k for k in a if k in ("image_url", "thumb_url", "author_icon", "footer_icon")
                          or k.startswith("image_")]:
                    del a[k]
                    c["files"] += 1
                if is_copy(a, m) or isinstance(a.get("files"), list):
                    # a forwarded copy is cleaned like the message itself; a picture-only copy goes
                    ac = _clean_one(a)
                    c["profiles"] += ac["profiles"]
                    c["files"] += ac["files"]
                    if ac["image_only"]:
                        had_media = True
                        continue
            left.append(a)
        if len(left) != len(atts):
            atts[:] = left
    c["profiles"] += strip_media_keys(m)   # recursively, including inside root
    c["image_only"] = int(had_media and not has_content(m))
    return c


def has_content(m: dict) -> bool:
    if message_text(m).strip():
        return True
    if m.get("files"):
        return True
    sub = m.get("subtype")
    if sub and sub not in ("file_share", "bot_message", "thread_broadcast"):
        return True                      # channel_join, pinned_item … — a service message
    return False
