"""Stage "images".

``[media].images`` (DESIGN.md §9b):

* ``avatars`` (default, owner's list of 27.10) — only avatars and profile photos are removed:
  files in profile folders (``profile_pictures``, ``avatars``, …), files named like an avatar
  (``avatar*``, ``userpic*``, ``profile_photo*``…), files that an export points to from an avatar
  field (Discord ``avatarUrl``/``iconUrl`` with ``--media``, Mattermost ``profile_image``, any
  JSON ``avatar*``/``profile_image*``) and group-photo service messages of Telegram. Screenshots
  and other pictures stay (``images:kept``); logos on them are task L-2;
* ``keep`` — pictures stay as they are, the stage only counts them (``images:kept:<kind>``);
* ``delete`` — every image is removed: raster and vector, Telegram stickers (webp/tgs and their
  previews), avatars and ``profile_pictures``, ``*_thumb.*`` previews. A picture is recognised
  by extension and by the signature of its first bytes (renamed to ``.dat`` goes too).

Each removal is ``ctx.remove(path, 'image', '<kind>')``; the kind is a short reason without content.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from ..office import sniff_image

NAME = "images"

# extension → kind (the reason in the report)
IMAGE_EXT = {
    ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".jpe": "jpeg", ".jfif": "jpeg",
    ".gif": "gif", ".webp": "webp", ".bmp": "bmp", ".dib": "bmp", ".tif": "tiff", ".tiff": "tiff",
    ".ico": "ico", ".cur": "ico", ".heic": "heic", ".heif": "heic", ".avif": "avif",
    ".svg": "svg", ".svgz": "svg", ".psd": "psd", ".ai": "ai", ".eps": "eps",
    ".tgs": "стикер Telegram (tgs)", ".cr2": "raw-фото", ".nef": "raw-фото", ".dng": "raw-фото",
    ".emf": "emf", ".wmf": "wmf", ".pict": "pict", ".pct": "pict", ".xcf": "xcf", ".jp2": "jpeg2000",
}
# a Telegram sticker and its preview — exporter names
_STICKER_RX = re.compile(r"^sticker(?:_\d+)?(?:@[\d_-]+)?(?:\s\(\d+\))?\.\w{1,5}(?:_thumb\.\w{1,5})?$", re.I)
_THUMB_RX = re.compile(r"_thumb(?:\.\w{1,5})?$", re.I)
# directories with profile photos
_PROFILE_DIRS = frozenset(("profile_pictures", "avatars", "avatar", "profile_photos",
                           "profile_images", "userpics"))
# file names of avatars and profile photos
_AVATAR_NAME_RX = re.compile(r"^(?:avatar|userpic|user_pic|profile[_ -]?(?:pic|picture|photo|image))"
                             r"(?![a-z])", re.I)
_REFS_MAX = 200_000_000         # JSON larger than this is not scanned for avatar paths

_HEAD = 4096


def _sniff(head: bytes) -> str | None:
    """Image kind from the first bytes, or None (the shared sniffer is office.sniff_image)."""
    return sniff_image(head)


def _why(path: Path, head: bytes) -> str | None:
    """Removal reason, or None — not an image."""
    name = path.name
    parts = {p.lower() for p in path.parts[:-1]}
    ext = path.suffix.lower()
    kind_ext = IMAGE_EXT.get(ext)
    sniffed = _sniff(head) if head else None
    if ext == ".svgz" and head.startswith(b"\x1f\x8b"):
        sniffed = "svg"
    kind = kind_ext or sniffed
    if kind is None:
        return None
    if _STICKER_RX.match(name):
        return "стикер Telegram" + (" (превью)" if _THUMB_RX.search(name) else "")
    if parts & _PROFILE_DIRS:
        return f"аватарка ({kind})"
    if _THUMB_RX.search(name) or _THUMB_RX.search(path.stem):
        return f"превью ({kind})"
    if kind_ext:
        return kind
    return f"сигнатура {sniffed} под расширением {ext or '(нет)'}"


def _kind(why: str) -> str:
    """Kind for the counter: without the extension under a signature and without details in brackets."""
    if why.startswith("сигнатура "):
        return why.split()[1]
    return why.split(" (")[0]


def _avatar_refs(ctx) -> set:
    """Files of the work copy that exports point to from avatar fields."""
    from .. import exportfmt
    from ..exportfmt import generic as G
    from ..exportfmt import telegram as T
    work = ctx.work.resolve()
    out: set = set()
    for p in ctx.files():
        if p.suffix.lower() not in (".json", ".jsonl") or not p.is_file():
            continue
        try:
            if p.stat().st_size > _REFS_MAX:
                continue
            parsed = exportfmt.detect(p)
        except (OSError, ValueError):
            continue
        if parsed is None:
            continue
        refs = G.avatar_paths(parsed.obj)
        if parsed.kind in ("telegram_full", "telegram_chat"):
            for _sect, _lst, chat in T.iter_chats(parsed.obj, parsed.kind):
                refs.extend(x for x in map(T.avatar_photo, chat.get("messages") or ()) if x)
        for ref in refs:
            ref = ref.replace("\\", "/").lstrip("/")
            for base in (p.parent, ctx.work):
                try:
                    q = (base / ref).resolve()
                except (OSError, ValueError):
                    continue
                if work in q.parents and q.is_file():
                    out.add(q)
    return out


def _avatar_why(path: Path, why: str, refs: set) -> str | None:
    """Reason to remove in ``avatars`` mode, else None (the picture stays)."""
    if why.startswith("аватарка"):
        return why
    if _AVATAR_NAME_RX.match(path.name):
        return f"аватарка ({_kind(why)})"
    try:
        if path.resolve() in refs:
            return f"аватарка ({_kind(why)}, поле выгрузки)"
    except OSError:
        pass
    return None


def run(ctx) -> None:
    mode = ctx.config.media.images
    refs = _avatar_refs(ctx) if mode == "avatars" else set()
    n = 0
    touched_dirs: set[Path] = set()
    for p in ctx.files():
        if not p.is_file():
            continue
        try:
            with open(p, "rb") as fh:
                head = fh.read(_HEAD)
        except OSError:
            head = b""
        why = _why(p, head)
        if why is None:
            continue
        if mode == "avatars":
            why = _avatar_why(p, why, refs) or _keep(ctx, why)
            if why is None:
                continue
        elif mode != "delete":
            _keep(ctx, why)
            continue
        ctx.remove(p, "image", why)
        touched_dirs.add(p.parent)
        n += 1
    # emptied image directories (profile_pictures, stickers) are not needed in the archive
    for d in sorted(touched_dirs, key=lambda x: len(x.parts), reverse=True):
        while d != ctx.work and d.exists() and d.is_dir() and not os.listdir(d):
            try:
                d.rmdir()
            except OSError:
                break
            d = d.parent


def _keep(ctx, why: str) -> None:
    ctx.count("images:kept")
    ctx.count("images:kept:" + _kind(why))
