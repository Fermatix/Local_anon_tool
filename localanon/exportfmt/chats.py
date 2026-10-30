"""Chat selection by [chats] and the chat registry for the data owner's report — ``ctx.data['chats']``.

Chat record: ``{id, name, type, messages, kept, source, reason, dropped, messages_kept}``:
  messages      — messages before the filter;
  kept          — False if the chat is removed whole (exclude / include_only / drop_personal_chats
                  / emptied);
  reason        — why it was removed;
  dropped       — {category: how many messages the filter removed};
  messages_kept — messages after the filter.
"""
from __future__ import annotations

from ..context import Removal
from .telegram import PERSONAL_CHAT_TYPES
from .discord import DIRECT_TYPES


def _norm(s) -> str:
    return " ".join(str(s).split()).casefold()


def _match(entry: str, cid: str, name: str, aliases=()) -> bool:
    e = _norm(entry)
    if not e:
        return False
    if e == _norm(cid) or (bool(name) and e == _norm(name)):
        return True
    return any(a and e == _norm(a) for a in aliases)


def decision(cfg_chats, cid: str, name: str, ctype: str, personal: bool | None = None,
             aliases=()) -> str:
    """'' — keep, otherwise the removal reason. ``aliases`` — other names of the same chat
    (Discord: the channel name without the guild, as the owner sees it)."""
    if cfg_chats.include_only:
        if not any(_match(e, cid, name, aliases) for e in cfg_chats.include_only):
            return "not in [chats].include_only"
    if any(_match(e, cid, name, aliases) for e in cfg_chats.exclude):
        return "in [chats].exclude"
    if cfg_chats.drop_personal_chats:
        if personal is None:
            personal = ctype in PERSONAL_CHAT_TYPES or ctype in DIRECT_TYPES
        if personal:
            return "personal chat ([chats].drop_personal_chats)"
    return ""


def records(ctx) -> list[dict]:
    return ctx.data.setdefault("chats", [])


def record(ctx, source: str, cid: str, name: str, ctype: str, messages: int) -> dict:
    """Find or create a chat record (key — file + id + name)."""
    for r in records(ctx):
        if r["source"] == source and r["id"] == cid and r["name"] == name:
            return r
    r = {"id": cid, "name": name, "type": ctype, "messages": messages, "kept": True,
         "source": source, "reason": "", "dropped": {}, "messages_kept": messages}
    records(ctx).append(r)
    return r


def duplicate_names(chats) -> set[str]:
    """Names shared by several chats of one file (the report tells them apart by [id])."""
    seen, dups = set(), set()
    for chat in chats:
        name = _norm(chat.get("name") if isinstance(chat, dict) else "")
        if name in seen:
            dups.add(name)
        seen.add(name)
    return dups


def drop_inline(ctx, rec: dict, rel: str, reason: str, category: str = "chat",
                dup: bool = False) -> None:
    """A chat removed from inside a shared file: the report record is ``<file>#<chat name>``;
    chats with the same name (``dup``) get ``<name> [id]``."""
    rec["kept"] = False
    rec["reason"] = reason
    rec["messages_kept"] = 0
    label = rec["name"] or rec["id"] or "?"
    if dup and rec["id"] and rec["name"]:
        label = f"{label} [{rec['id']}]"
    ctx.removals.append(Removal(f"{rel}#{label}", category, reason))
    ctx.counters["removed:" + category] += 1


def drop_file(ctx, rec: dict, path, reason: str, category: str = "chat") -> None:
    rec["kept"] = False
    rec["reason"] = reason
    rec["messages_kept"] = 0
    ctx.remove(path, category, reason)
