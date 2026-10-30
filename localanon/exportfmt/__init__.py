"""Parsing of chat export formats. The format is detected by content.

``detect(path)`` → file kind:
  telegram_full, telegram_chat — Telegram Desktop JSON;
  telegram_html                — Telegram Desktop HTML (messages*.html);
  discord                      — DiscordChatExporter JSON;
  slack_users, slack_channels, slack_messages — Slack;
  json, jsonl                  — other JSON (the generic path);
  transcript                   — a *.transcript.txt transcript;
  None                         — not our file.
The parsed object is kept alongside (``Parsed``), so the file is not read twice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import discord, jsonio, slack, telegram, tghtml, transcript

_MAX_SNIFF = 64 * 1024


@dataclass
class Parsed:
    kind: str
    obj: object = None                   # a JSON object, a list of JSONL objects or text
    style: jsonio.Style | None = None


def _head(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(_MAX_SNIFF)
    except OSError:
        return ""


def detect(path: Path) -> Parsed | None:
    path = Path(path)
    name = path.name.lower()
    if name.endswith(transcript.SUFFIX):
        head = _head(path)
        if transcript.is_transcript(path, head):
            return Parsed("transcript")
        return None
    if name.endswith((".html", ".htm")):
        head = _head(path)
        if tghtml.is_telegram_html(head):
            return Parsed("telegram_html")
        return None
    if not name.endswith((".json", ".jsonl", ".ndjson", ".txt")):
        return None
    head = _head(path).lstrip("\ufeff \t\r\n")
    if head[:1] not in "{[":
        return None
    try:
        obj, style = jsonio.load(path)
    except (ValueError, UnicodeDecodeError):
        if name.endswith((".jsonl", ".ndjson", ".json", ".txt")):
            try:
                text = path.read_text(encoding="utf-8")
                if jsonio.is_jsonl_text(text):
                    objs, style = jsonio.load_lines(path)
                    return Parsed("jsonl", objs, style)
            except (ValueError, UnicodeDecodeError, OSError):
                pass
        return None
    kind = telegram.detect(obj)
    if kind:
        return Parsed(kind, obj, style)
    if discord.detect(obj):
        return Parsed("discord", obj, style)
    sk = slack.detect(obj)
    if sk:
        return Parsed(sk, obj, style)
    if name.endswith(".txt"):
        return None
    return Parsed("json", obj, style)


def save(path: Path, parsed: Parsed) -> None:
    if parsed.kind == "jsonl":
        jsonio.save_lines(path, parsed.obj, parsed.style)
    else:
        jsonio.save(path, parsed.obj, parsed.style)


def valid_json(path: Path) -> bool:
    try:
        json.loads(Path(path).read_text(encoding="utf-8").lstrip("\ufeff"))
        return True
    except (ValueError, OSError):
        return False
