"""Stage "exports".

Structural cleaning of exports, the format is detected by content (localanon/exportfmt):
  * Telegram JSON: personal_information, contacts, frequent_contacts, sessions,
    profile_pictures, other_data go from result.json; in messages — photo/thumbnail/file/media_type
    pointing to files no longer in the working copy (the images stage removed the pictures); a
    message with only a picture and no text is removed whole;
  * Telegram HTML: div.message blocks — avatars, photos, stickers, links to removed files;
  * Discord (DiscordChatExporter): avatarUrl, picture attachments, thumbnail/image of embeds,
    emoji and sticker pictures;
  * Slack: profile.image_*, avatar_hash, picture files;
  * any JSON/JSONL: keys avatar*/photo*/profile_pic*/profile_image*/image_url/icon_url with a URL
    or path value.
The media cleaning above runs only with ``[media].images = "delete"``. With ``avatars`` (default,
DESIGN.md §9b) only avatar fields go: Telegram HTML ``userpic_wrap``, Discord ``avatarUrl`` and
``iconUrl``, Slack ``profile.image_*``/``avatar_hash``/``icons``, any JSON ``avatar*``/
``profile_image*``/``profile_pic*`` (Mattermost bulk export), the photo of Telegram group-photo
service messages; photos in messages and messages with them stay. With ``keep`` pictures and
avatars stay: the stage only picks chats and strips the service sections of Telegram.
Chat selection by [chats] (exclude / include_only / drop_personal_chats): a chat file is removed
via ctx.remove, a chat inside result.json — from chats.list with the record "result.json#<name>".
The chat registry for the report is ctx.data['chats'] (see exportfmt/chats.py). JSON stays valid.
"""
from __future__ import annotations

import re
from pathlib import Path

from .. import exportfmt
from ..exportfmt import chats as C
from ..exportfmt import discord as D
from ..exportfmt import generic as G
from ..exportfmt import slack as S
from ..exportfmt import telegram as T
from ..exportfmt import tghtml as H

NAME = "exports"


def strip_media(ctx) -> bool:
    """Whether to strip picture links and "picture only" messages (pictures are removed)."""
    return ctx.config.media.images == "delete"


def strip_avatars(ctx) -> bool:
    """Only avatar fields go ([media].images = "avatars")."""
    return ctx.config.media.images == "avatars"


def run(ctx) -> None:
    strip = strip_media(ctx)
    avatars = strip_avatars(ctx)
    slack_meta = _slack_meta(ctx)
    html_noted = False
    n_removed = len(ctx.removals)
    slack_done: set = set()              # Slack day files processed together with their channel
    C.records(ctx)                       # the chat map exists once the exports are parsed, even empty (anonymize)
    for path in ctx.files():
        if not path.is_file() or path in slack_done:
            continue
        parsed = exportfmt.detect(path)
        if parsed is None:
            continue
        rel = ctx.rel(path)
        kind = parsed.kind
        if kind in ("telegram_full", "telegram_chat"):
            _telegram(ctx, path, rel, parsed)
        elif kind == "telegram_html":
            if not html_noted:
                ctx.note("Telegram HTML: div.message blocks parsed, message text taken from div.text; "
                         "a JSON export is more reliable")
                html_noted = True
            _telegram_html(ctx, path, rel)
        elif kind == "discord":
            _discord(ctx, path, rel, parsed)
        elif kind == "slack_messages":
            files = slack_channel_files(path, parsed)
            slack_done.update(p for p, _ in files)
            _slack_channel(ctx, files, slack_meta)
        elif avatars:
            if kind == "slack_users":
                n = sum(S.strip_avatars(u) for u in parsed.obj if isinstance(u, dict))
                ctx.count("exports:slack_profile_images", n)
            else:
                n = G.strip_avatar_keys(parsed.obj)
                ctx.count("exports:json_keys", n)
            if n:
                exportfmt.save(path, parsed)
        elif not strip:
            continue                     # pictures stay: other JSON is not changed
        elif kind == "slack_users":
            n = sum(S.clean_user(u) for u in parsed.obj if isinstance(u, dict))
            ctx.count("exports:slack_profile_images", n)
            exportfmt.save(path, parsed)
        elif kind == "slack_channels":
            ctx.count("exports:json_keys", G.strip_media_keys(parsed.obj))
            exportfmt.save(path, parsed)
        elif kind in ("json", "jsonl"):
            n = G.strip_media_keys(parsed.obj)
            if n:
                ctx.count("exports:json_keys", n)
                exportfmt.save(path, parsed)
    _prune_empty_dirs(ctx, n_removed)


# ---------------------------------------------------------------------------------------------
# Telegram JSON
# ---------------------------------------------------------------------------------------------
def _telegram(ctx, path: Path, rel: str, parsed) -> None:
    obj, kind = parsed.obj, parsed.kind
    cfg = ctx.config.chats
    base = path.parent
    if kind == "telegram_full":
        gone = T.strip_sections(obj)
        ctx.count("exports:tg_sections", len(gone))
    chats = T.iter_chats(obj, kind)
    dups = C.duplicate_names(chat for _, _, chat in chats)
    for sect, lst, chat in chats:
        cid, name, ctype = T.chat_label(chat)
        msgs = chat.get("messages") if isinstance(chat.get("messages"), list) else []
        rec = C.record(ctx, rel, cid, name, ctype, len(msgs))
        dup = C._norm(name) in dups
        reason = C.decision(cfg, cid, name, ctype)
        if reason:
            if lst is None:
                C.drop_file(ctx, rec, path, reason)
                return
            lst.remove(chat)
            C.drop_inline(ctx, rec, rel, reason, dup=dup)
            ctx.count("exports:tg_chats_dropped")
            continue
        if strip_avatars(ctx):
            ctx.count("exports:tg_avatar_photos", T.strip_avatar_photos(chat, base))
            continue
        if not strip_media(ctx):
            continue
        fields, dropped = clean_telegram_chat(chat, base)
        ctx.count("exports:tg_media_fields", fields)
        ctx.count("exports:tg_media_messages", dropped)
        rec["messages_kept"] = len(chat.get("messages") or [])
        if not chat.get("messages") and msgs:
            reason = "empty after removing attachments"
            if lst is None:
                C.drop_file(ctx, rec, path, reason)
                return
            lst.remove(chat)
            C.drop_inline(ctx, rec, rel, reason, dup=dup)
    if strip_media(ctx):
        ctx.count("exports:json_keys", G.strip_media_keys(obj))
    elif strip_avatars(ctx):
        ctx.count("exports:json_keys", G.strip_avatar_keys(obj))
    if kind == "telegram_full" and not any(
            (obj.get(s) or {}).get("list") for s in T.CHAT_SECTIONS if isinstance(obj.get(s), dict)):
        ctx.remove(path, "chat", "no chats left in the export")
        return
    exportfmt.save(path, parsed)


def clean_telegram_chat(chat: dict, base: Path) -> tuple[int, int]:
    """Fields pointing to removed files and "picture only" messages. → (fields, messages)."""
    msgs = chat.get("messages")
    if not isinstance(msgs, list):
        return 0, 0
    fields = 0
    keep, gone = [], set()
    for m in msgs:
        if not isinstance(m, dict):
            keep.append(m)
            continue
        fields += T.clean_message_media(m, base)
        if T.has_content(m):
            keep.append(m)
        else:
            gone.add(m.get("id"))
    if gone:
        for m in keep:
            if isinstance(m, dict) and m.get("reply_to_message_id") in gone:
                del m["reply_to_message_id"]
                m.pop("reply_to_peer_id", None)
    chat["messages"] = keep
    return fields, len(msgs) - len(keep)


# ---------------------------------------------------------------------------------------------
# Telegram HTML
# ---------------------------------------------------------------------------------------------
def _telegram_html(ctx, path: Path, rel: str) -> None:
    with open(path, "r", encoding="utf-8", newline="") as fh:   # CRLF is kept as is
        text = fh.read()
    name = H.chat_name(text)
    spans = H.blocks(text)
    rec = C.record(ctx, rel, "", name, "html", len(spans))
    reason = C.decision(ctx.config.chats, "", name, "html", personal=False)
    if reason:
        C.drop_file(ctx, rec, path, reason)
        return
    if strip_avatars(ctx):
        text, pics = H.strip_page_pics(text)       # every userpic_wrap of the page
        ctx.count("exports:html_avatars", pics)
        if pics:
            path.write_text(text, encoding="utf-8", newline="")
        return
    if not strip_media(ctx):
        return                           # pictures stay: the page is not changed
    text, pics = H.strip_page_pics(text)
    spans = H.blocks(text)
    out, pos = [], 0
    avatars, attachments, dropped = pics, 0, 0
    gone, pending_from = set(), ""
    kept_blocks = 0
    for s, e in spans:
        out.append(text[pos:s])
        block, c = H.clean_block(text[s:e], path.parent)
        avatars += c["avatars"]
        attachments += c["attachments"]
        if c["image_only"]:
            gone.add(H.msg_id(block))
            dropped += 1
            fn = H.from_name_div(block)
            if fn and "joined" not in H.msg_classes(block):
                pending_from = fn
            pos = e
            continue
        if pending_from:
            block = H.unjoin(block, pending_from)
            pending_from = ""
        elif "joined" not in H.msg_classes(block):
            pending_from = ""
        out.append(block)
        kept_blocks += 1
        pos = e
    out.append(text[pos:])
    new = "".join(out)
    if gone:
        new = _html_fix_replies(new, gone)
    ctx.count("exports:html_avatars", avatars)
    ctx.count("exports:html_attachments", attachments)
    ctx.count("exports:tg_media_messages", dropped)
    rec["messages_kept"] = kept_blocks
    if kept_blocks == 0 and spans:
        C.drop_file(ctx, rec, path, "empty after removing attachments")
        return
    path.write_text(new, encoding="utf-8", newline="")


def _html_fix_replies(text: str, gone: set) -> str:
    out, pos = [], 0
    for s, e in H.blocks(text):
        out.append(text[pos:s])
        block = text[s:e]
        if H.reply_to(block) in gone:
            block = H.drop_reply(block)
        out.append(block)
        pos = e
    out.append(text[pos:])
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------------------------
def _discord(ctx, path: Path, rel: str, parsed) -> None:
    obj = parsed.obj
    cid, name, ctype = D.chat_label(obj)
    msgs = obj.get("messages")
    rec = C.record(ctx, rel, cid, name, ctype, len(msgs))
    reason = C.decision(ctx.config.chats, cid, name, ctype, personal=ctype in D.DIRECT_TYPES,
                        aliases=D.chat_aliases(obj))
    if reason:
        C.drop_file(ctx, rec, path, reason)
        return
    if strip_avatars(ctx):
        n = G.strip_avatar_keys(obj)                # author/mentions avatarUrl, guild iconUrl
        ctx.count("exports:discord_avatars", n)
        if n:
            exportfmt.save(path, parsed)
        return
    if not strip_media(ctx):
        return                           # pictures stay: the file is not changed
    totals = {"avatars": 0, "attachments": 0, "embeds": 0, "emoji": 0}
    keep, gone = [], set()
    for m in msgs:
        if not isinstance(m, dict):
            keep.append(m)
            continue
        c = D.clean_message(m)
        for k in totals:
            totals[k] += c[k]
        if c["image_only"]:
            gone.add(str(m.get("id")))
        else:
            keep.append(m)
    obj["messages"] = keep
    D.set_message_count(obj)
    discord_fix_replies(keep, gone)
    for k, v in totals.items():
        ctx.count(f"exports:discord_{k}", v)
    ctx.count("exports:discord_media_messages", len(gone))
    ctx.count("exports:json_keys", G.strip_media_keys(obj))
    rec["messages_kept"] = len(keep)
    if not keep and msgs:
        C.drop_file(ctx, rec, path, "empty after removing attachments")
        return
    exportfmt.save(path, parsed)


def discord_fix_replies(msgs: list, gone: set) -> None:
    """Drop a reference to a removed message; a nested copy of it (``referenced_message``)
    goes whole."""
    if not gone:
        return
    for m in msgs:
        if not isinstance(m, dict):
            continue
        ref = m.get("reference")
        hit = isinstance(ref, dict) and str(ref.get("messageId")) in gone
        if hit:
            m["reference"] = None
        for k in D.REF_COPY_KEYS:
            cp = m.get(k)
            if isinstance(cp, dict) and (hit or str(cp.get("id")) in gone):
                m[k] = None


# ---------------------------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------------------------
_SLACK_DM_DIR = re.compile(r"^[DG][0-9A-Z]{8,}$")


def _slack_meta(ctx) -> dict:
    """Slack channels from channels.json / groups.json / dms.json / mpims.json: name → (id, personal)."""
    meta = {}
    for path in ctx.files():
        if path.suffix.lower() != ".json" or path.stat().st_size > 50_000_000:
            continue
        head = exportfmt._head(path).lstrip("\ufeff \t\r\n")
        if head[:1] != "[" or not any(k in head for k in ('"is_channel"', '"is_im"', '"is_mpim"',
                                                              '"is_group"', '"members"')):
            continue
        parsed = exportfmt.detect(path)
        if parsed is None or parsed.kind != "slack_channels":
            continue
        for ch in parsed.obj:
            if not isinstance(ch, dict):
                continue
            cid = str(ch.get("id") or "")
            name = str(ch.get("name") or cid)
            personal = bool(ch.get("is_im") or ch.get("is_mpim")) or not ch.get("name")
            meta[name] = (cid, personal)
            if cid:
                meta[cid] = (cid, personal)
    return meta


def slack_channel_files(path: Path, parsed) -> list[tuple[Path, object]]:
    """All day files of a Slack channel (the directory of ``path``), in a stable order.
    Slack threads span day files, so the channel is processed whole."""
    out = []
    for q in sorted(path.parent.iterdir()):
        if not q.is_file():
            continue
        pr = parsed if q == path else exportfmt.detect(q)
        if pr is not None and pr.kind == "slack_messages":
            out.append((q, pr))
    return out


def slack_channel_record(ctx, files: list, meta: dict) -> tuple[dict, str, bool]:
    """Registry record for a Slack channel (directory). → (record, id, whether personal)."""
    folder = files[0][0].parent
    dirname = folder.name
    cid, personal = meta.get(dirname, ("", bool(_SLACK_DM_DIR.match(dirname))))
    ctype = "dm" if personal else "channel"
    total = sum(len(pr.obj) for _, pr in files)
    rec = C.record(ctx, ctx.rel(folder), cid, dirname, ctype, total)
    return rec, cid, personal


def _slack_channel(ctx, files: list, meta: dict) -> None:
    rec, cid, personal = slack_channel_record(ctx, files, meta)
    reason = C.decision(ctx.config.chats, cid, rec["name"], rec["type"], personal=personal)
    if reason:
        for q, _ in files:
            C.drop_file(ctx, rec, q, reason)
        return
    if strip_avatars(ctx):
        n = 0
        for q, pr in files:
            k = S.strip_avatars(pr.obj)
            if k:
                exportfmt.save(q, pr)
            n += k
        ctx.count("exports:slack_profile_images", n)
        return
    if not strip_media(ctx):
        return                           # pictures stay: day files are not changed
    totals = {"profiles": 0, "files": 0}
    gone, keep_all = set(), []
    for q, pr in files:
        keep = []
        for m in pr.obj:
            if not isinstance(m, dict):
                keep.append(m)
                continue
            c = S.clean_message(m)
            for k in totals:
                totals[k] += c[k]
            if c["image_only"]:
                gone.add(m.get("ts"))
            else:
                keep.append(m)
        pr.obj[:] = keep
        keep_all.extend(keep)
    slack_fix_threads(keep_all, gone)
    ctx.count("exports:slack_profile_images", totals["profiles"])
    ctx.count("exports:slack_files", totals["files"])
    ctx.count("exports:slack_media_messages", len(gone))
    ctx.count("exports:json_keys", sum(G.strip_media_keys(pr.obj) for _, pr in files))
    _slack_save(ctx, rec, files, len(keep_all), "empty after removing attachments")


def _slack_save(ctx, rec: dict, files: list, kept: int, empty_reason: str) -> None:
    """Save the channel day files; remove emptied ones (the whole channel emptied — kept=False)."""
    rec["messages_kept"] = kept
    if kept == 0:
        for q, _ in files:
            C.drop_file(ctx, rec, q, empty_reason)
        return
    for q, pr in files:
        if pr.obj:
            exportfmt.save(q, pr)
        else:
            ctx.remove(q, "chat", empty_reason)


def slack_fix_threads(msgs: list, gone: set) -> None:
    """Drop thread_ts/replies pointing to removed messages (``msgs`` — the whole channel, all days).
    Nested copies of a removed message (``root`` of thread_broadcast, a forwarded message in
    ``attachments`` with its ``ts``) go whole; a copy of a live parent gets its
    ``replies``/``reply_count``/``latest_reply`` cleaned like the parent itself."""
    if not gone:
        return
    for m in msgs:
        if not isinstance(m, dict):
            continue
        _slack_fix_refs(m, gone)
        root = m.get("root")
        if isinstance(root, dict):
            if root.get("ts") in gone:
                del m["root"]
            else:
                _slack_fix_refs(root, gone)
        atts = m.get("attachments")
        if isinstance(atts, list):
            left = [a for a in atts if not _slack_quotes_gone(a, gone)]
            if len(left) != len(atts):
                atts[:] = left


def _slack_fix_refs(m: dict, gone: set) -> None:
    if m.get("thread_ts") in gone and m.get("thread_ts") != m.get("ts"):
        m.pop("thread_ts", None)
        m.pop("parent_user_id", None)
    reps = m.get("replies")
    if isinstance(reps, list):
        left = [r for r in reps if not (isinstance(r, dict) and r.get("ts") in gone)]
        if len(left) != len(reps):
            m["replies"] = left
            if "reply_count" in m:
                m["reply_count"] = len(left)
            if "reply_users" in m:
                m["reply_users"] = sorted({r.get("user") for r in left if isinstance(r, dict)
                                           and r.get("user")})
                m["reply_users_count"] = len(m["reply_users"])
            if "latest_reply" in m:
                ts = [r.get("ts") for r in left if isinstance(r, dict) and r.get("ts")]
                if ts:
                    m["latest_reply"] = max(ts)
                else:
                    del m["latest_reply"]


def _slack_quotes_gone(a, gone: set) -> bool:
    """A forward attachment ("Share message"): a copy of the message with its ``ts``."""
    if not isinstance(a, dict):
        return False
    if a.get("ts") in gone:
        return True
    return any(isinstance(b, dict) and b.get("ts") in gone for b in a.get("message_blocks") or ())


def _prune_empty_dirs(ctx, start: int) -> None:
    """Directories emptied by our removals (``ctx.removals[start:]``), bottom up.
    Directories empty in the input or emptied by other stages are not touched."""
    work = ctx.work.resolve()
    for r in ctx.removals[start:]:
        p = (ctx.work / r.rel).parent
        while True:
            try:
                rp = p.resolve()
                if rp == work or work not in rp.parents or not p.is_dir() or any(p.iterdir()):
                    break
                p.rmdir()
            except OSError:
                break
            p = p.parent
