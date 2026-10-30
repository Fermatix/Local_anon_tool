"""Generic path for any JSON: avatar and photo keys with a URL or path value are removed.

Keys: ``avatar*``, ``photo*``, ``profile_pic*``, ``profile_image*``, ``image_url``, ``icon_url``
(and their camelCase variants: ``avatarUrl``, ``iconUrl``, ``imageUrl``).

Avatars only (``[media].images = "avatars"``): ``avatar*``, ``profile_pic*``, ``profile_image*``,
``profile_photo*``, ``userpic*``, ``icon_url`` (server/bot icon), ``author_icon``; photos in
messages (``photo``, ``image_url``) stay.
"""
from __future__ import annotations

import re

IMAGE_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif",
    ".svg", ".ico", ".avif", ".jfif", ".psd", ".tgs", ".webm_thumb"})

_KEY_RX = re.compile(r"^(?:avatar|photo|profile_?pic|profile_?image)|^(?:image_?url|icon_?url)$",
                     re.IGNORECASE)
_URL_RX = re.compile(r"^(?:https?:)?//\S+$|^data:image/", re.IGNORECASE)
# a file path: segments split by / or \, an extension at the end
_PATH_RX = re.compile(r"^[/\\]?(?:[^/\\\n\t]+[/\\])*[^/\\\n\t]+\.[A-Za-z0-9]{1,5}$")


_AVATAR_KEY_RX = re.compile(r"^(?:avatar|profile_?pic|profile_?image|profile_?photo|user_?pic)"
                            r"|^(?:icon_?url|author_?icon)$", re.IGNORECASE)


def avatar_key(key) -> bool:
    return isinstance(key, str) and bool(_AVATAR_KEY_RX.search(key))


def _avatar_hit(k, v) -> bool:
    if not avatar_key(k):
        return False
    if isinstance(k, str) and k.lower() == "avatar_hash":
        return isinstance(v, str) and bool(v)
    return url_or_path(v)


def strip_avatar_keys(node) -> int:
    """Recursively drop avatar keys whose value is a URL or a path. → how many."""
    n = 0
    if isinstance(node, dict):
        for k in [k for k, v in node.items() if _avatar_hit(k, v)]:
            del node[k]
            n += 1
        for v in node.values():
            if isinstance(v, (dict, list)):
                n += strip_avatar_keys(v)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                n += strip_avatar_keys(v)
    return n


def avatar_paths(node) -> list[str]:
    """Local paths (not URLs) under avatar keys: files of the export to delete."""
    out: list[str] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(v, (dict, list)):
                    stack.append(v)
                elif (_avatar_hit(k, v) and k.lower() != "avatar_hash"
                      and not _URL_RX.match(v.strip())):
                    out.append(v.strip())
        elif isinstance(cur, list):
            stack.extend(x for x in cur if isinstance(x, (dict, list)))
    return out


def media_key(key) -> bool:
    return isinstance(key, str) and bool(_KEY_RX.search(key))


def url_or_path(value) -> bool:
    if not isinstance(value, str) or not value or len(value) > 2048:
        return False
    v = value.strip()
    if _URL_RX.match(v):
        return True
    if " " in v and "/" not in v and "\\" not in v:
        return False                     # a phrase, not a path
    return bool(_PATH_RX.match(v))


def is_image_name(name) -> bool:
    if not isinstance(name, str):
        return False
    base = name.rsplit("?", 1)[0].rsplit("#", 1)[0]
    base = base.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for ext in IMAGE_EXTS:
        if base.endswith(ext):
            return True
    return False


def strip_media_keys(node) -> int:
    """Recursively remove avatar/photo keys with a URL/path value. → how many were removed."""
    n = 0
    if isinstance(node, dict):
        for k in [k for k, v in node.items() if media_key(k) and url_or_path(v)]:
            del node[k]
            n += 1
        for v in node.values():
            if isinstance(v, (dict, list)):
                n += strip_media_keys(v)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                n += strip_media_keys(v)
    return n
