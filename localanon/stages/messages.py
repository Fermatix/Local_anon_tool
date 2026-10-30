"""Stage "messages".

By default (data owner's decision of 27.10) the stage is off: ``[messages].filter = "off"``
or an empty ``drop_categories`` — no message and no chat is removed, the classifier does not run
(counter 'messages:filter_off', a note in the report); all correspondence goes to anonymize.
Below is the behaviour with the filter explicitly on.

The message filter: messages of all exports (Telegram JSON and HTML, Discord, Slack) and
utterances of ``*.transcript.txt`` transcripts → localanon/msgfilter.classify in batches →
messages of the [messages].drop_categories categories are removed whole (the message object);
reply_to pointing to a removed message is dropped; empty chats and files are removed; JSON stays
valid. Service messages (Telegram ``service``, Slack ``subtype`` channel_join/bot_message/…,
Discord system types, HTML ``service``) are not passed to the classifier — the structure decides
their fate.
Nested copies of other messages (Slack ``attachments[]`` forwards, see ``slack.is_copy``;
Discord ``message_snapshots``), including copies in copies (depth ≤ 3), are separate messages of
the same chat: a copy of a drop_categories category goes whole (also in the host's copy — Slack
``root``, Discord ``referenced_message``), the host is judged by its own text; a host without its
own text and without remaining attachments is removed; a parent copy emptied after the nested one
goes too (counted the same way). The original may be in another channel or outside the
export. A Discord forward under an unknown key is not judged — counter
'messages:unparsed:<key>' and a note.
Telegram: a dropped message goes together with its live attachment files (ctx.remove with the
category 'messages:<category>', counter 'messages:attachments_removed'); a message emptied after
transcribe (no text, no file) is removed whole — 'messages:emptied'.
Accounting: ctx.count('messages:<category>'), 'messages:removed'; per chat — ctx.data['chats']
(dropped: {category: n}, messages_kept), no message text.
Also 'messages:removed:<category>' — removed messages of the chats by category, and
'messages:doubtful' — messages kept by the model with p(delete) just below the threshold
(``msgfilter.DOUBT_BAND``).

With the model the messages of all chats go to the classifier in one call before the chats are
processed (``_prefetch``: the hw plan spreads it over the devices); a chat finds its categories by its
exact input (texts, chat type, copies), a chat read otherwise is classified on its own, as before.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .. import exportfmt, msgfilter
from ..exportfmt import chats as C
from ..exportfmt import discord as D
from ..exportfmt import slack as S
from ..exportfmt import telegram as T
from ..exportfmt import tghtml as H
from ..exportfmt import transcript as TR
from . import exports as X

NAME = "messages"
CONTEXT = 3
_CACHE: dict = {}                    # (chat type, texts, copies) → (categories, doubt); see _prefetch


def run(ctx) -> None:
    drop = set(ctx.config.messages.drop_categories)
    if ctx.config.messages.filter == "off" or not drop:
        ctx.count("messages:filter_off")
        ctx.note("messages: message filter is off ([messages].filter = off or empty drop_categories) — "
                 "messages are not removed, only anonymized")
        return
    n_removed = len(ctx.removals)
    _CACHE.clear()
    try:
        _CACHE.update(_prefetch(ctx))
    except Exception as e:  # noqa: BLE001 — a speed-up only: the chats are classified one by one
        ctx.note(f"messages: one call for all chats failed ({type(e).__name__}) — chat by chat")
    try:
        _run(ctx, drop)
    finally:
        _CACHE.clear()
    X._prune_empty_dirs(ctx, n_removed)


def _run(ctx, drop: set) -> None:
    slack_meta = None                    # Slack channels are read lazily, once
    slack_done: set = set()              # Slack day files processed together with their channel
    for path in ctx.files():
        if not path.is_file() or path in slack_done:
            continue
        parsed = exportfmt.detect(path)
        if parsed is None:
            continue
        rel = ctx.rel(path)
        kind = parsed.kind
        if kind in ("telegram_full", "telegram_chat"):
            _telegram(ctx, path, rel, parsed, drop)
        elif kind == "telegram_html":
            _telegram_html(ctx, path, rel, drop)
        elif kind == "discord":
            _discord(ctx, path, rel, parsed, drop)
        elif kind == "slack_messages":
            if slack_meta is None:
                slack_meta = X._slack_meta(ctx)
            files = X.slack_channel_files(path, parsed)
            slack_done.update(p for p, _ in files)
            _slack_channel(ctx, files, slack_meta, drop)
        elif kind == "transcript":
            _transcript(ctx, path, rel, drop)


def _prefetch(ctx) -> dict:
    """Categories of the messages of all chats in one classifier call: chat by chat, a small chat runs
    in this process on one device slot, and each chat from ``msgmodel.POOL_MIN`` messages starts a pool
    and loads the model again. The chats are read here (a fresh parse, nothing is written) the way the
    handlers read them; Slack channels (spread over day files) are left to the handlers. Without the
    model (rules) — nothing to gain, {}. → {(chat type, texts, copies): (categories, doubt)}."""
    if not msgfilter.model_available(ctx.config, ctx.notes):
        return {}
    keys: dict = {}
    for path in ctx.files():
        if not path.is_file():
            continue
        parsed = exportfmt.detect(path)
        if parsed is None:
            continue
        if parsed.kind in ("telegram_full", "telegram_chat"):
            for _sect, _lst, chat in T.iter_chats(parsed.obj, parsed.kind):
                if not isinstance(chat.get("messages"), list):
                    continue
                X.clean_telegram_chat(chat, path.parent)
                keys.setdefault((T.chat_label(chat)[2], tuple(_telegram_texts(chat["messages"])), ()), None)
        elif parsed.kind == "telegram_html":
            keys.setdefault(("html", tuple(_html_texts(_html_blocks(_read(path))[1])), ()), None)
        elif parsed.kind == "discord":
            msgs = parsed.obj["messages"]
            texts, copies = _discord_inputs(msgs)
            keys.setdefault((D.chat_label(parsed.obj)[2], tuple(texts), tuple(_discord_copy_texts(copies))),
                            None)
        elif parsed.kind == "transcript":
            keys.setdefault(("call", tuple(it["text"] for it in _replicas(TR.parse(_read(path)))), ()), None)
    allm, bounds = [], []
    for key in keys:
        m = _msgs(list(key[1]), key[0], key[2])
        bounds.append((key, len(allm), len(m)))
        allm.extend(m)
    if not allm:
        return {}
    doubt: list = []
    cats = msgfilter.classify(allm, ctx.config, ctx.notes, ctx=ctx, doubt=doubt)
    return {key: (cats[s:s + n], doubt[s:s + n]) for key, s, n in bounds}


def _msgs(texts: list[str], chat_kind: str, copies=()) -> list:
    def prev(i):
        return [p for p in texts[max(0, i - 8):i] if p.strip()][-CONTEXT:]

    msgs = [msgfilter.Msg(t, prev(i), chat_kind) for i, t in enumerate(texts)]
    return msgs + [msgfilter.Msg(t, prev(i), chat_kind) for i, t in copies]


def _classify(ctx, texts: list[str], chat_kind: str, copies=()) -> list[str]:
    """Categories for a sequence of texts of one chat (context — the previous 1–3).
    ``copies`` — (host index, text) of nested copies: each is judged as a chat message in the
    host's place (its context) and does not go into the context of other messages; their
    categories come last. Categories found by ``_prefetch`` for the same input are taken from there."""
    hit = _CACHE.get((chat_kind, tuple(texts), tuple(copies)))
    if hit is not None:
        cats, doubt = list(hit[0]), list(hit[1])
    else:
        doubt = []
        cats = msgfilter.classify(_msgs(texts, chat_kind, copies), ctx.config, ctx.notes, ctx=ctx,
                                  doubt=doubt)
    for c in cats[:len(texts)]:
        ctx.count("messages:" + c)
    ctx.count("messages:doubtful", sum(1 for d in doubt[:len(texts)] if d))
    return cats


def _apply(ctx, rec: dict, cats: list[str], drop: set) -> set:
    """Indexes of messages to remove; per-chat accounting."""
    gone = set()
    for i, c in enumerate(cats):
        if c in drop:
            gone.add(i)
            rec["dropped"][c] = rec["dropped"].get(c, 0) + 1
            ctx.count("messages:removed:" + c)
    ctx.count("messages:removed", len(gone))
    return gone


COPY_DEPTH = 3                           # copy in a copy: collection depth


def _copies(m: dict, children, parent=None, depth: int = COPY_DEPTH) -> list:
    """Copies of message ``m`` and copies inside them (depth ≤ ``depth``), a parent before nested ones:
    (owner list, copy, parent copy or None). ``children(msg)`` →
    (owner list, item, message inside the item)."""
    out = []
    if depth <= 0:
        return out
    for lst, item, inner in children(m):
        out.append((lst, item, parent))
        if isinstance(inner, dict):
            out.extend(_copies(inner, children, item, depth - 1))
    return out


def _slack_children(m: dict) -> list:
    return [(m["attachments"], a, a) for a in S.copies_of(m)]


def _discord_children(m: dict) -> list:
    return [(lst, sn, D.snapshot_message(sn)) for lst in D.snapshot_lists(m) for sn in lst]


def _drop_copies(ctx, rec: dict, msgs: list, copies: list, cats: list[str], drop: set,
                 gone: set, is_empty, copy_empty) -> tuple[set, dict]:
    """Drop copies (``copies`` — (host index, owner list, copy, parent copy)) of the ``drop``
    categories whole: text, author, link, files; copies nested in a dropped copy are not counted;
    a parent copy emptied by this (``copy_empty``) goes too. →
    (indexes of hosts left with neither own text nor attachments — they are removed as empty,
    counted under the dropped copy's category; dropped copies by host index)."""
    emptied: dict[int, str] = {}
    removed: dict[int, list] = {}
    dead: set[int] = set()
    n = 0
    for (i, lst, item, parent), c in zip(copies, cats):
        if parent is not None and id(parent) in dead:
            dead.add(id(item))
            continue
        if c not in drop or i in gone:
            continue
        lst[:] = [x for x in lst if x is not item]
        dead.add(id(item))
        n += 1
        emptied.setdefault(i, c)
        removed.setdefault(i, []).append(item)
    # a parent copy left with neither text nor attachments after the nested ones went is an
    # empty shell (author, link): it goes whole, up the chain, counted the same way
    where = {id(item): (lst, parent) for _, lst, item, parent in copies}
    for i, items in list(removed.items()):
        for item in list(items):
            par = where[id(item)][1]
            while par is not None and id(par) not in dead and copy_empty(par):
                lst, up = where[id(par)]
                lst[:] = [x for x in lst if x is not par]
                dead.add(id(par))
                n += 1
                items.append(par)
                par = up
    ctx.count("messages:copies_removed", n)
    hosts = {i for i in emptied if is_empty(msgs[i])}
    for i in hosts:
        rec["dropped"][emptied[i]] = rec["dropped"].get(emptied[i], 0) + 1
    ctx.count("messages:removed", len(hosts))
    return hosts, {i: items for i, items in removed.items() if i not in hosts}


def _slack_sync_roots(msgs: list, removed: dict) -> None:
    """A dropped copy also goes from the copy of a live host — ``root`` of thread_broadcast
    (matched by ``ts``/``from_url``, nested copies included)."""
    keys = {}
    for i, items in removed.items():
        keys[msgs[i].get("ts")] = {v for a in items for v in (a.get("ts"), a.get("from_url")) if v}
    for m in msgs:
        root = m.get("root") if isinstance(m, dict) else None
        if isinstance(root, dict) and root.get("ts") in keys:
            _slack_strip(root, keys[root["ts"]], COPY_DEPTH)


def _slack_strip(m: dict, keys: set, depth: int) -> None:
    atts = m.get("attachments")
    if not isinstance(atts, list) or depth <= 0:
        return
    left = [a for a in atts if not (isinstance(a, dict)
                                    and (a.get("ts") in keys or a.get("from_url") in keys))]
    if len(left) != len(atts):
        atts[:] = left
    for a in left:
        if isinstance(a, dict):
            _slack_strip(a, keys, depth - 1)


def _discord_sync_refs(msgs: list, removed: dict) -> None:
    """A dropped copy also goes from the copy of a live host — ``referenced_message`` of a reply
    (matched by ``id``/``message_reference``): the copy's snapshots are made equal to the host's."""
    for i, items in removed.items():
        host = msgs[i]
        for m in msgs:
            if not isinstance(m, dict) or m is host:
                continue
            for cp in D.ref_copy_matches(m, host):
                for k in D.SNAPSHOT_KEYS:
                    if isinstance(cp.get(k), list):
                        src = host.get(k)
                        cp[k] = (copy.deepcopy(src) if isinstance(src, list)
                                 else [x for x in cp[k] if x not in items])


def _discord_unparsed(ctx, msgs: list) -> None:
    """A forward under an unknown key is not judged — a counter and one note per key."""
    seen: dict[str, int] = {}
    for m in msgs:
        if isinstance(m, dict):
            for k in D.unparsed_copy_keys(m):
                seen[k] = seen.get(k, 0) + 1
    for k, n in sorted(seen.items()):
        ctx.count("messages:unparsed:" + k, n)
        note = (f"Discord: message field '{k}' looks like a nested copy but is not parsed — "
                f"the filter does not judge it (counter messages:unparsed:{k})")
        if note not in ctx.notes:
            ctx.note(note)


# ---------------------------------------------------------------------------------------------
def _telegram(ctx, path: Path, rel: str, parsed, drop: set) -> None:
    obj, kind = parsed.obj, parsed.kind
    base = path.parent
    chats = T.iter_chats(obj, kind)
    dups = C.duplicate_names(chat for _, _, chat in chats)
    for sect, lst, chat in chats:
        cid, name, ctype = T.chat_label(chat)
        msgs = chat.get("messages")
        if not isinstance(msgs, list):
            continue
        rec = C.record(ctx, rel, cid, name, ctype, len(msgs))
        _, emptied = X.clean_telegram_chat(chat, base)      # files gone after transcribe
        ctx.count("messages:emptied", emptied)              # a message without text and attachments — whole
        msgs = chat["messages"]
        cats = _classify(ctx, _telegram_texts(msgs), ctype)
        gone = _apply(ctx, rec, cats, drop)
        gone_ids = {msgs[i].get("id") for i in gone if isinstance(msgs[i], dict)}
        keep = [m for i, m in enumerate(msgs) if i not in gone]
        _telegram_files(ctx, base, msgs, gone, cats, keep)
        for m in keep:
            if isinstance(m, dict) and m.get("reply_to_message_id") in gone_ids:
                del m["reply_to_message_id"]
                m.pop("reply_to_peer_id", None)
        chat["messages"] = keep
        rec["messages_kept"] = len(keep)
        if not keep:
            reason = "all messages removed by the filter" if msgs else "chat is empty"
            if lst is None:
                C.drop_file(ctx, rec, path, reason)
                return
            lst.remove(chat)
            C.drop_inline(ctx, rec, rel, reason, dup=C._norm(name) in dups)
    if kind == "telegram_full" and not any(
            (obj.get(s) or {}).get("list") for s in T.CHAT_SECTIONS if isinstance(obj.get(s), dict)):
        ctx.remove(path, "chat", "no chats left in the export")
        return
    exportfmt.save(path, parsed)


def _telegram_texts(msgs: list) -> list[str]:
    return [T.message_text(m) if isinstance(m, dict) and m.get("type") == "message" else "" for m in msgs]


def _telegram_files(ctx, base: Path, msgs: list, gone: set, cats: list[str], keep: list) -> None:
    """A message goes together with its attachments: live files of the dropped message (file, photo,
    preview, vCard) are removed from the working copy under the same category; a file referenced
    by a kept message and a path outside the working copy are not touched."""
    work = Path(ctx.work).resolve()
    kept = {p.resolve() for m in keep if isinstance(m, dict) for p in T.media_paths(m, base)}
    n = 0
    for i in sorted(gone):
        m = msgs[i]
        for p in (T.media_paths(m, base) if isinstance(m, dict) else ()):
            rp = p.resolve()
            if rp in kept or not rp.is_file() or not rp.is_relative_to(work):
                continue
            ctx.remove(p, "messages:" + cats[i], f"attachment of a message of category {cats[i]}")
            n += 1
    ctx.count("messages:attachments_removed", n)


def _read(path: Path) -> str:
    with open(path, "r", encoding="utf-8", newline="") as fh:   # CRLF is kept as is
        return fh.read()


def _html_blocks(text: str) -> tuple[list, list[str]]:
    spans = H.blocks(text)
    return spans, [text[s:e] for s, e in spans]


def _html_texts(blocks: list[str]) -> list[str]:
    return ["" if H.is_service(b) else H.text_of(b) for b in blocks]


def _telegram_html(ctx, path: Path, rel: str, drop: set) -> None:
    text = _read(path)
    spans, blocks = _html_blocks(text)
    rec = C.record(ctx, rel, "", H.chat_name(text), "html", len(spans))
    cats = _classify(ctx, _html_texts(blocks), "html")
    gone = _apply(ctx, rec, cats, drop)
    gone_ids = {H.msg_id(blocks[i]) for i in gone}
    out, pos, pending_from, kept = [], 0, "", 0
    for i, (s, e) in enumerate(spans):
        out.append(text[pos:s])
        pos = e
        block = blocks[i]
        if i in gone:
            fn = H.from_name_div(block)
            if fn and "joined" not in H.msg_classes(block):
                pending_from = fn
            continue
        if pending_from:
            block = H.unjoin(block, pending_from)
            pending_from = ""
        if H.reply_to(block) in gone_ids:
            block = H.drop_reply(block)
        out.append(block)
        kept += 1
    out.append(text[pos:])
    rec["messages_kept"] = kept
    if kept == 0:
        C.drop_file(ctx, rec, path, "all messages removed by the filter" if spans else "chat is empty")
        return
    path.write_text("".join(out), encoding="utf-8", newline="")


def _discord(ctx, path: Path, rel: str, parsed, drop: set) -> None:
    obj = parsed.obj
    cid, name, ctype = D.chat_label(obj)
    msgs = obj["messages"]
    rec = C.record(ctx, rel, cid, name, ctype, len(msgs))
    texts, copies = _discord_inputs(msgs)
    _discord_unparsed(ctx, msgs)
    cats = _classify(ctx, texts, ctype, _discord_copy_texts(copies))
    gone = _apply(ctx, rec, cats[:len(texts)], drop)
    hosts, removed = _drop_copies(ctx, rec, msgs, copies, cats[len(texts):], drop, gone, _discord_empty,
                                   _discord_copy_empty)
    gone |= hosts
    _discord_sync_refs(msgs, removed)
    gone_ids = {str(msgs[i].get("id")) for i in gone if isinstance(msgs[i], dict)}
    keep = [m for i, m in enumerate(msgs) if i not in gone]
    X.discord_fix_replies(keep, gone_ids)
    obj["messages"] = keep
    D.set_message_count(obj)
    rec["messages_kept"] = len(keep)
    if not keep:
        C.drop_file(ctx, rec, path, "all messages removed by the filter" if msgs else "chat is empty")
        return
    exportfmt.save(path, parsed)


def _discord_inputs(msgs: list) -> tuple[list[str], list]:
    texts = [D.message_text(m) if isinstance(m, dict) and not D.is_service(m) else "" for m in msgs]
    copies = [(i, lst, sn, par) for i, m in enumerate(msgs) if isinstance(m, dict)
              for lst, sn, par in _copies(m, _discord_children)]
    return texts, copies


def _discord_copy_texts(copies: list) -> list:
    return [(i, D.snapshot_text(sn)) for i, _, sn, _ in copies]


def _slack_channel(ctx, files: list, meta: dict, drop: set) -> None:
    """A whole Slack channel (all day files): context and thread links span days."""
    rec, _, _ = X.slack_channel_record(ctx, files, meta)
    msgs = [m for _, pr in files for m in pr.obj]
    texts = [S.message_text(m, copies=False) if isinstance(m, dict) and not S.is_service(m) else ""
             for m in msgs]
    copies = [(i, lst, a, par) for i, m in enumerate(msgs) if isinstance(m, dict)
              for lst, a, par in _copies(m, _slack_children)]
    cats = _classify(ctx, texts, rec["type"], [(i, S.copy_text(a)) for i, _, a, _ in copies])
    gone = _apply(ctx, rec, cats[:len(texts)], drop)
    hosts, removed = _drop_copies(ctx, rec, msgs, copies, cats[len(texts):], drop, gone, _slack_empty,
                                   _slack_copy_empty)
    gone |= hosts
    _slack_sync_roots(msgs, removed)
    gone_ts, keep_all, start = set(), [], 0
    for _, pr in files:
        msgs = pr.obj
        keep = []
        for j, m in enumerate(msgs):
            if start + j in gone:
                if isinstance(m, dict):
                    gone_ts.add(m.get("ts"))
            else:
                keep.append(m)
        start += len(msgs)
        pr.obj[:] = keep
        keep_all.extend(keep)
    X.slack_fix_threads(keep_all, gone_ts)
    X._slack_save(ctx, rec, files, len(keep_all), "all messages removed by the filter")


def _slack_empty(m: dict) -> bool:
    t = m.get("text")
    return not (isinstance(t, str) and t.strip()) and not m.get("attachments") and not m.get("files")


def _slack_copy_empty(a: dict) -> bool:
    """``fallback``/``pretext`` do not count: they are rendering (author, time, nested copy text)."""
    own = any(isinstance(a.get(k), str) and a[k].strip() for k in ("text", "title"))
    return not own and not a.get("attachments") and not a.get("files")


def _discord_copy_empty(sn) -> bool:
    m = D.snapshot_message(sn)
    return not isinstance(m, dict) or _discord_empty(m)


def _discord_empty(m: dict) -> bool:
    return not D.has_content(m) and not any(D.snapshot_lists(m))


def _transcript(ctx, path: Path, rel: str, drop: set) -> None:
    text = _read(path)
    nl = "\r\n" if "\r\n" in text else "\n"
    items = TR.parse(text)
    replicas = _replicas(items)
    rec = C.record(ctx, rel, "", path.name, "call", len(replicas))
    cats = _classify(ctx, [it["text"] for it in replicas], "call")
    gone = _apply(ctx, rec, cats, drop)
    gone_items = {id(replicas[i]) for i in gone}
    keep = [it for it in items if id(it) not in gone_items]
    kept = sum(1 for it in keep if not it.get("head"))
    rec["messages_kept"] = kept
    if kept == 0:
        C.drop_file(ctx, rec, path, "all utterances removed by the filter" if replicas else "transcript is empty")
        return
    path.write_text(TR.render(keep, nl), encoding="utf-8", newline="")


def _replicas(items: list) -> list:
    return [it for it in items if not it.get("head")]
