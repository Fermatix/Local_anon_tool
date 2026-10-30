"""Reading and writing JSON/JSONL keeping the input's look: indent, separators, BOM, line
endings, non-ASCII escaping. ``json`` itself keeps the key order (dict is ordered).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_INDENT_RX = re.compile(r"[\[{]\r?\n([ \t]*)\S")
_SPACED_ITEMS = re.compile(r'(?:"|\d|true|false|null|\]|\}), (?:"|\d|true|false|null|\[|\{)')


@dataclass
class Style:
    indent: int | str | None = None      # None — compact, on one line
    item_sep: str = ","
    key_sep: str = ":"
    ensure_ascii: bool = False
    bom: bool = False
    newline: str = "\n"
    trailing_newline: bool = True

    def dumps(self, obj) -> str:
        text = json.dumps(obj, ensure_ascii=self.ensure_ascii, indent=self.indent,
                          separators=(self.item_sep, self.key_sep))
        if self.newline != "\n":
            text = text.replace("\n", self.newline)
        if self.trailing_newline:
            text += self.newline
        return ("\ufeff" if self.bom else "") + text


def sniff_style(text: str) -> Style:
    """The look of JSON text by its head: indent (number of spaces or a tab), separators, etc."""
    st = Style()
    if text.startswith("\ufeff"):
        st.bom = True
        text = text[1:]
    head = text[:200_000]
    st.newline = "\r\n" if "\r\n" in head else "\n"
    st.trailing_newline = text.endswith("\n")
    m = _INDENT_RX.search(head)
    if m:
        ws = m.group(1)
        st.indent = "\t" if ws.startswith("\t") else len(ws)
        st.item_sep, st.key_sep = ",", ": "
    else:
        st.indent = None
        st.item_sep = ", " if _SPACED_ITEMS.search(head) else ","
        st.key_sep = ": " if re.search(r'": ', head) else ":"
    if "\\u" in head and not any(ord(c) > 127 for c in head):
        st.ensure_ascii = True
    return st


def load(path: Path):
    """→ (object, style). A parse error is json.JSONDecodeError / UnicodeDecodeError."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    st = sniff_style(text)
    return json.loads(text.lstrip("\ufeff")), st


def save(path: Path, obj, style: Style) -> None:
    Path(path).write_text(style.dumps(obj), encoding="utf-8", newline="")


def load_lines(path: Path):
    """JSONL → (list of objects, style). Empty lines are skipped."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    bom = text.startswith("\ufeff")
    if bom:
        text = text[1:]
    nl = "\r\n" if "\r\n" in text[:100_000] else "\n"
    objs = []
    for line in text.splitlines():
        if line.strip():
            objs.append(json.loads(line))
    st = Style(indent=None, bom=bom, newline=nl, trailing_newline=text.endswith("\n"))
    first = text.lstrip()[:100_000]
    st.item_sep = ", " if _SPACED_ITEMS.search(first) else ","
    st.key_sep = ": " if re.search(r'": ', first) else ":"
    if "\\u" in first and not any(ord(c) > 127 for c in first):
        st.ensure_ascii = True
    return objs, st


def save_lines(path: Path, objs, style: Style) -> None:
    lines = [json.dumps(o, ensure_ascii=style.ensure_ascii, separators=(style.item_sep, style.key_sep))
             for o in objs]
    text = style.newline.join(lines)
    if style.trailing_newline and lines:
        text += style.newline
    Path(path).write_text(("\ufeff" if style.bom else "") + text, encoding="utf-8", newline="")


def is_jsonl_text(text: str) -> bool:
    """Several lines, each non-empty one is a standalone JSON object."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 1:
        return False
    if len(lines) == 1:
        return False                     # one line — plain JSON
    for ln in lines[:50]:
        s = ln.strip()
        if s[:1] not in "{[":
            return False
        try:
            json.loads(s)
        except ValueError:
            return False
    return True
