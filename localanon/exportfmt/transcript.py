"""``*.transcript.txt`` transcripts: lines "[HH:MM:SS-HH:MM:SS] SPEAKER_00: text".
A line without a timestamp continues the previous utterance."""
from __future__ import annotations

import re
from pathlib import Path

LINE_RX = re.compile(r"^\[\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\s*-\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\]\s*"
                     r"(?P<speaker>[^:\n]{1,64}):\s?(?P<text>.*)$")
SUFFIX = ".transcript.txt"


def is_transcript(path: Path, head: str) -> bool:
    if not path.name.lower().endswith(SUFFIX):
        return False
    return any(LINE_RX.match(ln) for ln in head.splitlines()[:50])


def parse(text: str) -> list[dict]:
    """→ [{lines: [str], text: str}] — utterances together with their continuation lines."""
    items = []
    for ln in text.splitlines():
        m = LINE_RX.match(ln)
        if m:
            items.append({"lines": [ln], "text": m.group("text")})
        elif items:
            items[-1]["lines"].append(ln)
            if ln.strip():
                items[-1]["text"] += "\n" + ln.strip()
        else:
            items.append({"lines": [ln], "text": "", "head": True})
    return items


def render(items: list[dict], newline: str) -> str:
    out = []
    for it in items:
        out.extend(it["lines"])
    return newline.join(out) + (newline if out else "")
