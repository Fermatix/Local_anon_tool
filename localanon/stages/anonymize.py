"""Stage "anonymize".

Anonymization by the anonymizer/ engine with the [project].salt salt: the roster of people from the
exports with all cases (localanon/roster.py), the engine's full-name rules, the name layer by [ner]
(``names_layer``: the name model localanon/nermodel.py, gliner, Natasha — localanon/ner.py),
emails, logins, nicknames, domains → RESOURCE_…, links to [links].delete_hosts are replaced with
the RESOURCE_REDACTED placeholder (localanon/links.py), the contractor from [executor] →
ORG_…/RESOURCE_… (localanon/executor.py), file and folder names. Office and PDF go through the
views of localanon/office.py and localanon/pdf.py. The chain is built in localanon/engine_adapter.py.

UTF-16/32 text with a BOM is first re-encoded to UTF-8 without a BOM (the engine reads UTF-8 only).

Every part uses the slots of the hw plan: document views are read ahead in worker processes
(``engine_adapter.prefetch_views``), the models and Natasha — in their pools, the documents are
masked in forked engine processes (``Engine.map``) and the tree — in the engine's own pool.

In ctx.data: ``roster`` (people and nicknames — for the data owner's report), ``gate_controls``
(forms for the control checks), ``mask_rules`` (counters by rule, no values), ``file_marks``
(file record: ``{rel: {"transcoded_from": encoding}}``). Counters ``masked:<family>`` and
``masked:transcoded`` (re-encoded files) go to the manifest.
"""
from __future__ import annotations

from pathlib import Path

from .. import engine_adapter as EA
from .. import executor as _executor
from .. import hw
from .. import links as _links
from .. import ner as _ner
from .. import nermodel
from .. import roster as _roster
from ..context import UNPROCESSED

NAME = "anonymize"

_FAMILY = (
    ("link_deleted", ("link:delete",)),
    ("person", ("person", "name", "surname", "patronymic", "initials", "roster:pair",
                "ner:", "diminutive", "translit", "tab_person")),
    ("handle", ("handle", "mention", "slug", "maintainer")),
    ("email", ("email",)),
    ("phone", ("phone",)),
    ("resource", ("domain", "link:host", "resource", "gdrive", "figma", "yadisk", "dropbox",
                  "url", "doc_id", "docid")),
    ("org", ("org", "opf", "brand", "executor", "client", "quoted", "title", "hashtag",
             "suffix", "crm", "company")),
    ("user_id", ("user_id", "uid", "peer", "snowflake", "discord")),
    ("secret", ("secret", "token", "key", "password", "cookie", "entropy")),
    ("financial", ("inn", "ogrn", "kpp", "card", "iban", "account", "bik", "snils", "passport")),
    ("address", ("address", "street", "house", "flat", "postcode", "geo", "city", "dob")),
)


def family_of(rule: str) -> str:
    low = rule.lower()
    for fam, keys in _FAMILY:
        if any(k in low for k in keys):
            return fam
    return "other"


# lowercase forms of surnames the model found (ner.model_rules): seen — only for surnames the
# model saw in lowercase; chosen by measurement (docs/MODELS.md §2)
LOWER = "all"            # lowercase forms of all found surnames (except homonyms): measured, MODELS.md §2


def _lines(views) -> list[str]:
    """Unique non-empty view lines (models work line by line; no need to process a repeated line)."""
    seen: dict = {}
    for _rel, text in views:
        for line in text.split("\n"):
            line = line.strip()
            if line and line not in seen:
                seen[line] = None
    return list(seen)


# files of data, not of text: tables, logs, JSON/XML/YAML data, configs, code — the model's organization names
# found only there are not masked (executor.model_org_pack, text_lines); a chat export is text whatever its extension
_DATA_EXT = frozenset((
    ".csv", ".tsv", ".xlsx", ".log", ".json", ".jsonl", ".ndjson", ".xml", ".yaml", ".yml", ".ini", ".cfg", ".conf",
    ".toml", ".properties", ".env", ".sql", ".py", ".js", ".ts", ".java", ".go", ".cs", ".php", ".rb", ".kt", ".c",
    ".cpp", ".h", ".sh"))
_CHAT_KINDS = frozenset(("telegram_full", "telegram_chat", "telegram_html", "discord", "slack_messages",
                         "transcript"))


def _chat_sources(ctx):
    """Files and folders (a Slack channel) of the chats the export stages parsed (``ctx.data["chats"]``);
    None — the stages did not run (``--stages``), the chats are told by content."""
    recs = ctx.data.get("chats")
    return None if recs is None else {r["source"] for r in recs}


def _text_view(path: Path, rel: str, chats) -> bool:
    """A view of text — a chat, a document, a prose file — or of data (``_DATA_EXT``): the share of the
    model's names in text (Л-23)."""
    if chats is not None:
        parts = rel.split("/")
        if any("/".join(parts[:i]) in chats for i in range(1, len(parts) + 1)):
            return True
    elif path.suffix.lower() in (".json", ".jsonl", ".ndjson"):
        from .. import exportfmt
        parsed = exportfmt.detect(path)
        if parsed is not None and parsed.kind in _CHAT_KINDS:
            return True
    return path.suffix.lower() not in _DATA_EXT


def _union(a, b):
    out = []
    for s, e in sorted(list(a) + list(b)):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _model_spans(ctx, lines, model_dir, info):
    """Name model spans per line → (PER spans with gliner on top, ORG spans or None). ORG — when
    ``[ner].orgs`` is on and the tier's model has the ORG class (one pass) or the index has an
    ``org_tier`` (a second pass of that tier for ORG only — the names stay with the chosen tier);
    no model — (None, None) and a note."""
    cfg = ctx.config.ner
    tier = None if cfg.model == "auto" else cfg.model
    why = nermodel.missing_reason(model_dir, tier)
    if why:
        ctx.note(f"ner_model: {why} — names are found by Natasha" if cfg.natasha is not False
                 else f"ner_model: {why} — names are found only by the roster and engine rules")
        return None, None
    try:
        sp = hw.stage_plan(ctx, "ner_model", hw.Workload(n_texts=len(lines)))
    except Exception as e:  # noqa: BLE001 — the plan does not crash the stage: CPU
        ctx.note(f"ner_model: hardware plan not built ({type(e).__name__}) — model on CPU")
        sp = None
    if model_dir and tier is None and sp is not None:      # own directory: the tier from its index
        tier = nermodel.default_tier(model_dir, gpu=sp.on_gpu) or None
    chosen = tier or (sp.model if sp is not None else None)
    own = cfg.orgs and "ORG" in nermodel.classes_of(model_dir, chosen)
    second = nermodel.org_tier(model_dir, chosen) if cfg.orgs and not own else None
    if cfg.orgs and not own and not second:
        ctx.note("ner_model: the model of this tier does not find organizations — only the engine's "
                 "legal-form rules and [executor] mask them")
    res = nermodel.predict_classes(model_dir, lines, classes=("PER", "ORG") if own else ("PER",), plan=sp,
                                   tier=tier, ctx=ctx, info=info)
    spans = [d["PER"] for d in res]
    org_spans = [d["ORG"] for d in res] if own else None
    if second:
        info["org_tier"] = second
        org_spans = [d["ORG"] for d in nermodel.predict_classes(model_dir, lines, classes=("ORG",), plan=sp,
                                                                tier=second, ctx=ctx)]
    if cfg.gliner:
        from .. import gliner_ner
        why = gliner_ner.missing_reason()
        if why:
            ctx.note(f"ner_gliner: {why} — the name model only")
        else:
            g = gliner_ner.spans(lines, cfg.gliner_threshold, cfg.device, info=info)
            ctx.count("ner:gliner:spans", sum(map(len, g)))
            spans = [_union(a, b) for a, b in zip(spans, g)]
    return spans, org_spans


def names_layer(ctx, files, model_dir=None) -> tuple:
    """Name layer by [ner] → (pack rules, early rules, model info — only numbers and tier/device
    names, organizations — ``executor.Pack`` of the names the model found). The name model works
    without triage (ner.model_rules); Natasha follows ``natasha``: ``auto`` — only without the model,
    ``true`` — with it too (through triage), ``false`` — off. Organizations (``[ner].orgs``, a model
    with the ORG class) — rules of the contractor kind (executor.model_org_pack)."""
    cfg = ctx.config.ner
    pack, early, info, orgs = [], [], {}, _executor.Pack()
    if cfg.model == "off" and cfg.natasha is False:
        return pack, early, info, orgs
    paths = []
    views = []
    for p, rel, _kind, text in EA.iter_views(ctx.work, files, ner=True, notes=ctx.note):
        paths.append(p)
        views.append((rel, text))
    model_on = False
    if cfg.model != "off":
        lines = _lines(views)
        spans, org_spans = _model_spans(ctx, lines, model_dir, info) if lines else ([], None)
        persons = _ner.mentions_of(lines, spans or [])
        if spans is not None:
            model_on = True
            e, pk, stats = _ner.model_rules(persons, lower=LOWER)
            early.extend(e)
            pack.extend(pk)
            ctx.count("ner:model:lines", len(lines))
            for k, v in stats.items():
                ctx.count(f"ner:model:{k}", v)
        if org_spans is not None:
            chats = _chat_sources(ctx)
            text_lines = set(_lines([v for p, v in zip(paths, views) if _text_view(p, v[0], chats)]))
            orgs = _executor.model_org_pack(_ner.mentions_of(lines, org_spans), persons=persons, lines=lines,
                                            workers=_engine_workers(ctx), text_lines=text_lines)
            for k, v in orgs.counts.items():
                ctx.count(f"ner:org:{k}", v)
    if cfg.natasha is True or (cfg.natasha == "auto" and not model_on):
        lits, stats = _ner.natasha_literals(views, ctx.note, slots=_natasha_slots(ctx, views), ctx=ctx)
        for k, v in stats.items():
            ctx.count(f"ner:natasha:{k}", v)
        rules, persons, surnames = _ner.literal_rules(lits)
        ctx.count("ner:persons", persons)
        ctx.count("ner:surnames", surnames)
        pack.extend(rules)
    return pack, early, info, orgs


def _natasha_slots(ctx, views):
    """CPU slots for Natasha by the hw plan (one per worker process); no plan — one process."""
    try:
        n = sum(len(_ner.chunk_bounds(text)) for _rel, text in views)
        return hw.stage_plan(ctx, "natasha", hw.Workload(n_texts=n)).slots
    except Exception as e:  # noqa: BLE001 — the plan does not crash the stage: one process
        ctx.note(f"natasha: hardware plan not built ({type(e).__name__}) — one process")
        return None


def _engine_workers(ctx) -> int:
    """Engine processes by the hw plan (stage ``anonymize``; the engine itself takes no more than it
    has files and parts). Without fork (Windows) the engine runs sequentially anyway."""
    try:
        return max(1, len(hw.stage_plan(ctx, "anonymize", hw.Workload()).slots))
    except Exception as e:  # noqa: BLE001 — the plan does not crash the stage: one process
        ctx.note(f"anonymize: hardware plan not built ({type(e).__name__}) — one process")
        return 1


_DOC_SPLIT_CHARS = 64 << 10    # a document view longer than this goes to line windows over all the workers
_DOC_WINDOW_CHARS = 32 << 10   # view characters per window (~0.5–1 s of the full rule chain)


def _mask_document(eng, task) -> tuple:
    """One office/pdf document through the engine (in a worker of ``Engine.map`` or here) →
    ("read", exception name) | ("layer",) | ("same",) | ("written", changed lines) |
    ("write", exception name) | ("lines", view lines): the view is longer than ``split`` characters,
    the caller masks it in windows (``_split_documents``)."""
    path, kind, rel, split = task
    p = Path(path)
    try:
        lines = EA.document_lines(p, kind)
    except EA.DocumentError as e:
        return ("read", e.exc_name)
    if lines is None:
        return ("layer",)
    if split and sum(map(len, lines)) > split:
        return ("lines", lines)
    return _write_masked(p, kind, lines, _mask_window(eng, (rel, lines)))


def _mask_window(eng, task) -> list:
    """View lines of one document through the engine: a line does not depend on its neighbours."""
    rel, lines = task
    return [eng.mask_text(line, rel) if line else line for line in lines]


def _write_masked(p: Path, kind: str, lines: list, new: list) -> tuple:
    changed = sum(1 for a, b in zip(lines, new) if a != b)
    if not changed:
        return ("same",)
    try:
        EA.document_write(p, kind, new)
    except Exception as e:     # noqa: BLE001 — OSError, la-docs layer: same as a read error
        return ("write", getattr(e, "exc_name", type(e).__name__))
    return ("written", changed)


def _write_document(_eng, task) -> tuple:
    path, kind, lines, new = task
    return _write_masked(Path(path), kind, lines, new)


def _line_windows(lines: list, target: int) -> list:
    """[(start, end)] of contiguous line slices of about ``target`` characters (a longer line — alone)."""
    out, start, size = [], 0, 0
    for i, line in enumerate(lines):
        if size and size + len(line) > target:
            out.append((start, i))
            start, size = i, 0
        size += len(line)
    out.append((start, len(lines)))
    return out


def _split_documents(ctx, eng, docs: list, results: list, workers: int) -> None:
    """Documents read with the ("lines", …) result: their lines are masked in windows over all the
    workers (one big document is no longer the tail of the stage on one worker), then the documents are
    written in the pool; ``results`` get the result of ``_mask_document`` as from one pass."""
    big = [i for i, res in enumerate(results) if res is not None and res[0] == "lines"]
    if not big:
        return
    windows, owner = [], []
    for i in big:
        lines, rel = results[i][1], ctx.rel(docs[i][0])
        for a, b in _line_windows(lines, _DOC_WINDOW_CHARS):
            windows.append((rel, lines[a:b]))
            owner.append(i)
    order = sorted(range(len(windows)), key=lambda k: -sum(map(len, windows[k][1])))
    got = eng.map(_mask_window, [windows[k] for k in order], workers, label="anonymize:windows")
    masked: list = [None] * len(windows)
    for k, out in zip(order, got):
        masked[k] = out
    new: dict = {i: [] for i in big}
    for i, out in zip(owner, masked):
        if out is None or new[i] is None:      # the worker died on a window twice: as on the document
            new[i] = None
        else:
            new[i].extend(out)
    writes = [i for i in big if new[i] is not None]
    for i in big:
        if new[i] is None:
            results[i] = None
    got = eng.map(_write_document, [(str(docs[i][0]), docs[i][1], results[i][1], new[i]) for i in writes],
                  workers, label="anonymize:write")
    for i, res in zip(writes, got):
        results[i] = res


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _documents(ctx, eng, files, workers: int = 1) -> None:
    """Office/PDF views through the engine — in up to ``workers`` processes (``Engine.map``); a long view
    is masked in line windows over all the workers (``_split_documents``); the results go to ctx here in
    file order."""
    docs = [(p, kind) for p in files for kind in (EA.file_kind(p),) if kind in ("office", "pdf")]
    order = list(range(len(docs)))
    split = 0
    if workers > 1:
        # the biggest documents first: one started last is the tail of the stage on one worker
        order.sort(key=lambda i: -_size(docs[i][0]))
        split = _DOC_SPLIT_CHARS
    got = eng.map(_mask_document, [(str(docs[i][0]), docs[i][1], ctx.rel(docs[i][0]), split) for i in order],
                  workers)
    results: list = [None] * len(docs)
    for i, res in zip(order, got):
        results[i] = res
    _split_documents(ctx, eng, docs, results, workers)
    missing: set = set()
    for (p, kind), res in zip(docs, results):
        if res is None:                # the worker died on this document twice
            res = ("read", "worker died")
        if res[0] == "read":
            # the document layer failed (broken archive etc.): the file was not read, so not anonymized —
            # it stays as is with a mark; the path goes to the owner's report, notes have no path
            if not ctx.is_marked(p, UNPROCESSED):
                ctx.mark(p, UNPROCESSED, f"not anonymized: document ({kind}) not read")
            ctx.count("masked:documents_failed", 1)
            ctx.note(f"document ({kind}) not read by the la-docs layer ({res[1]}) — kept "
                     "as is, see the report")
        elif res[0] == "layer":
            if kind not in missing:
                missing.add(kind)
                ctx.note(f"layer {kind} (la-docs) unavailable: documents of this kind are not anonymized")
        elif res[0] == "write":
            ctx.mark(p, UNPROCESSED, f"not anonymized: document ({kind}) not written")
            ctx.count("masked:documents_failed", 1)
            ctx.note(f"document ({kind}) not written by the la-docs layer ({res[1]}) — kept as is, "
                     "see the report")
        elif res[0] == "written":
            ctx.count("masked:document_lines", res[1])
            ctx.count("masked:documents", 1)


def _identity_fields(ctx, eng, ros, files) -> None:
    from anonymizer import structured
    for p in files:
        if p.suffix.lower() != ".json" or not p.exists():
            continue
        rel = ctx.rel(p)
        try:
            if structured.detect(str(p), rel) is not None:
                continue
        except Exception:              # noqa: BLE001
            continue
        n = _roster.mask_identity_fields(p, eng, ros)
        if n:
            ctx.count("masked:identity_fields", n)


def _transcode(ctx, files) -> None:
    """UTF-16/32 with a BOM → UTF-8 without a BOM before the roster, NER and the engine (the engine
    reads UTF-8 only, and the output for the main pipeline is UTF-8). File record —
    ctx.data['file_marks'][rel]['transcoded_from'] (for the data owner's report), counter
    masked:transcoded."""
    marks = ctx.data.setdefault("file_marks", {})
    for p in files:
        codec = EA.transcode_wide_bom(p)
        if codec:
            marks.setdefault(ctx.rel(p), {})["transcoded_from"] = codec
            ctx.count("masked:transcoded", 1)


def run(ctx) -> None:
    cfg = ctx.config
    EA.set_salt(cfg.project.salt)          # minting is needed already when the rules are built
    _ner.guard_filename_words()            # «Техническое задание.docx» is not a surname
    files = [Path(p) for p in ctx.files()]
    _transcode(ctx, files)
    ros = _roster.collect(ctx.work, files, workers=_engine_workers(ctx))
    ctx.data["roster"] = _roster.to_data(ros)
    ctx.count("roster:persons", len(ros.persons))
    ctx.count("roster:handles", len(ros.handles))
    pack = _executor.build(cfg.executor)
    for k, v in pack.counts.items():
        ctx.count(f"executor:{k}", v)
    first, link_pack = _links.rules(cfg.links)
    EA.prefetch_views(ctx, files)          # document views (OCR of scans) in worker processes
    ner_pack, ner_early, _info, ner_orgs = names_layer(ctx, files)
    roster_early, roster_pack = _roster.rule_groups(ros)
    # contractor details go before the financial detector; roster pairs/initials, then name model
    # mentions and pairs go before the name detector; roster nicknames and surnames, NER surnames,
    # other domains — in the pack slot; after them the contractor name in a host/path (executor
    # glued: by then a foreign host is already a whole RESOURCE). Organizations the model found go
    # after the contractor's own rules (its token wins for its names) — the same slots
    workers = _engine_workers(ctx)
    eng = EA.Engine(cfg.project.salt, first=first + pack.first, before_orgs=pack.rules + ner_orgs.rules,
                    before_names=roster_early + ner_early,
                    pack=roster_pack + ner_pack + link_pack + pack.late + ner_orgs.late,
                    workers=workers)
    ctx.data["gate_controls"] = pack.controls + _roster.controls(ros)
    _documents(ctx, eng, files, workers)
    files = [p for p in files if p.exists()]
    # Discord/Slack identity fields go BEFORE the engine: otherwise the display_name/real_name field
    # rule mints a nickname as PERSON_NAME, while the same nickname in text is HANDLE (two tokens
    # for one value)
    _identity_fields(ctx, eng, ros, files)
    report = eng.run_tree(ctx.work)
    counts = eng.rule_counts(report)
    ctx.data["mask_rules"] = counts
    for rule, n in counts.items():
        ctx.count("masked:" + family_of(rule), n)
    ctx.count("masked:files_changed", report.files_changed)
    if report.fusions_split:
        ctx.count("masked:fusions_split", report.fusions_split)
    for why in _failed(report):
        ctx.count("masked:files_failed", 1)
        ctx.note(f"file not processed by the engine ({why})")  # no path: notes carry no personal data
    moves: list = []
    renamed = eng.rename_tree(ctx.work, moves, workers=workers, keep=frozenset(hw.EXPORT_MARKERS))
    ctx.marks_moved(moves)
    ctx.count("masked:renamed", len(renamed))


def _failed(report):
    """The engine's FILE_FAILED: reasons from the pseudo-rule examples (the path is not returned)."""
    return [ex.split(" -> ")[-1] for ex in report.examples.get("file_failed", ())]
