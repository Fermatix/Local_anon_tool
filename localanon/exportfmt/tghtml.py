"""Telegram Desktop HTML (``messages.html``, ``messages2.html`` …): ``div.message`` blocks.

Parsing without dependencies: a block runs from ``<div class="message …"`` to the matching
``</div>`` (nesting is counted). Avatars (``userpic_wrap``), photos and stickers (``photo_wrap``,
``sticker_wrap``, ``img``) and links to files missing from the working copy are stripped from the
block; a message with no text and no remaining attachments is removed whole. The text is
``div.text`` without tags.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

_MSG_OPEN = re.compile(r'<div class="message(?:\s[^"]*)?"[^>]*>')
_TAG = re.compile(r"<(/?)div\b[^>]*>", re.IGNORECASE)
_ID = re.compile(r'\bid="message(-?\d+)"')
_CLASS = re.compile(r'^<div class="([^"]*)"')
_TEXT_DIV = re.compile(r'<div class="text">')
_FROM_DIV = re.compile(r'<div class="from_name">')
_REPLY_DIV = re.compile(r'<div class="reply_to details">')
_GOTO = re.compile(r"GoToMessage\((-?\d+)\)|#go_to_message(-?\d+)")
_A_OPEN = re.compile(r'<a class="([^"]*)"[^>]*>')
_HREF = re.compile(r'href="([^"]*)"')
_IMG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_USERPIC = re.compile(r'<div class="(?:pull_left |pull_right )?userpic_wrap[^"]*">')
_HEADER_NAME = re.compile(r'<div class="page_header">.*?<div class="text bold">(.*?)</div>', re.S)

_MEDIA_A = ("photo_wrap", "sticker_wrap", "media_photo", "animated_wrap", "video_file_wrap")


def is_telegram_html(head: str) -> bool:
    return ('<div class="message' in head and
            ('class="history"' in head or 'class="page_wrap"' in head or "Telegram" in head))


def _block_end(text: str, start: int) -> int:
    """End of the div block opened at start (the position after the matching </div>)."""
    depth = 0
    for m in _TAG.finditer(text, start):
        if m.group(1):
            depth -= 1
            if depth == 0:
                return m.end()
        else:
            depth += 1
    return len(text)


def _div_span(text: str, open_rx, start: int = 0, end: int | None = None):
    m = open_rx.search(text, start, end if end is not None else len(text))
    if not m:
        return None
    return m.start(), _block_end(text, m.start())


def blocks(text: str) -> list[tuple[int, int]]:
    """Ranges of top-level message blocks."""
    out = []
    pos = 0
    while True:
        m = _MSG_OPEN.search(text, pos)
        if not m:
            break
        end = _block_end(text, m.start())
        out.append((m.start(), end))
        pos = end
    return out


def msg_id(block: str) -> str:
    m = _ID.search(block[:300])
    return m.group(1) if m else ""


def msg_classes(block: str) -> list[str]:
    m = _CLASS.match(block)
    return m.group(1).split() if m else []


def is_service(block: str) -> bool:
    return "service" in msg_classes(block)


def text_of(block: str) -> str:
    sp = _div_span(block, _TEXT_DIV)
    if not sp:
        return ""
    inner = block[sp[0]:sp[1]]
    inner = inner[inner.index(">") + 1:]
    if inner.endswith("</div>"):
        inner = inner[:-6]
    inner = _BR.sub("\n", inner)
    return html.unescape(_TAGS.sub("", inner)).strip()


def reply_to(block: str) -> str:
    sp = _div_span(block, _REPLY_DIV)
    if not sp:
        return ""
    m = _GOTO.search(block, sp[0], sp[1])
    return (m.group(1) or m.group(2)) if m else ""


def drop_reply(block: str) -> str:
    sp = _div_span(block, _REPLY_DIV)
    if not sp:
        return block
    return block[:sp[0]].rstrip(" \t") + block[sp[1]:].lstrip(" \t")


def chat_name(text: str) -> str:
    m = _HEADER_NAME.search(text[:20_000])
    return html.unescape(_TAGS.sub("", m.group(1))).strip() if m else ""


def _file_present(base: Path, href: str) -> bool:
    if not href or href.startswith(("http:", "https:", "#", "tg:")):
        return True                      # an external link is not an export file
    try:
        return (base / html.unescape(href)).is_file()
    except (OSError, ValueError):
        return False


def clean_block(block: str, base: Path) -> tuple[str, dict]:
    """Strip avatars and attachments pointing to missing files. → (block, counters)."""
    c = {"avatars": 0, "attachments": 0}
    had_media = False
    while True:
        sp = _div_span(block, _USERPIC)
        if not sp:
            break
        block = block[:sp[0]] + block[sp[1]:]
        c["avatars"] += 1
    pos = 0
    while True:
        m = _A_OPEN.search(block, pos)
        if not m:
            break
        classes = m.group(1).split()
        hm = _HREF.search(m.group(0))
        href = hm.group(1) if hm else ""
        media = any(k in classes for k in _MEDIA_A)
        if "media" in classes and not media and not _file_present(base, href):
            media = True                 # an export file that is gone
        if media and (not _file_present(base, href) or any(k in classes for k in _MEDIA_A)):
            end = block.find("</a>", m.end())
            end = len(block) if end < 0 else end + 4
            block = block[:m.start()] + block[end:]
            c["attachments"] += 1
            had_media = True
            pos = m.start()
        else:
            pos = m.end()
    n_img = len(_IMG.findall(block))
    if n_img:
        block = _IMG.sub("", block)
        c["attachments"] += n_img
        had_media = True
    c["image_only"] = int(had_media and not is_service(block) and not text_of(block)
                          and not _A_OPEN.search(block))
    return block, c


def strip_page_pics(text: str) -> tuple[str, int]:
    """Avatars outside messages (the page header)."""
    n = 0
    while True:
        sp = _div_span(text, _USERPIC)
        if not sp:
            break
        text = text[:sp[0]] + text[sp[1]:]
        n += 1
    return text, n


def unjoin(block: str, from_name_div: str) -> str:
    """The message after a removed "joined" message gets the sender name."""
    m = _CLASS.match(block)
    if m and "joined" in m.group(1).split():
        cls = " ".join(k for k in m.group(1).split() if k != "joined")
        block = block[:m.start(1)] + cls + block[m.end(1):]
    if from_name_div and not _FROM_DIV.search(block):
        sp = _div_span(block, re.compile(r'<div class="body">'))
        if sp:
            m2 = re.search(r'<div class="pull_right date details"[^>]*>.*?</div>', block[sp[0]:sp[1]], re.S)
            if m2:
                at = sp[0] + m2.end()
                return block[:at] + from_name_div + block[at:]
    return block


def from_name_div(block: str) -> str:
    sp = _div_span(block, _FROM_DIV)
    return block[sp[0]:sp[1]] if sp else ""
