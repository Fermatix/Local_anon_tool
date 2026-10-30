"""Run output: the archive <name>_clean.zip, local_manifest.json for the main pipeline (only counters,
salt fingerprint, versions — no paths and no content) and report.html for the data owner (with paths
of removed files; not included in the archive).

The archive is deterministic: members in the stable order of ``ctx.files()``, the same fixed time and
permissions for all entries — the same input gives a byte-for-byte identical archive.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .context import category_key

ENGINE_COMMIT = "798bec5"           # anon_pipeline commit anonymizer/ was copied from
MANIFEST_NAME = "local_manifest.json"
REPORT_NAME = "report.html"
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)   # the earliest time zip allows — the same for all members
_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".tgz", ".tbz2", ".txz", ".tar",
                     ".7z", ".rar")


def input_stem(path: Path) -> str:
    """Input name without archive suffixes: "export.tar.gz" → "export"."""
    name = Path(path).name
    low = name.lower()
    for s in _ARCHIVE_SUFFIXES:
        if low.endswith(s) and len(name) > len(s):
            return name[: -len(s)]
    return name or "export"


def archive_name(ctx) -> str:
    want = (ctx.config.output.archive_name or "").strip()
    if want:
        return want if want.lower().endswith(".zip") else want + ".zip"
    return input_stem(ctx.input_path) + "_clean.zip"


def build_archive(ctx, dest: Path) -> int:
    """Deterministic zip of the working copy → dest (through a temporary file next to it). Returns
    the number of files."""
    tmp = dest.with_name(dest.name + ".part")
    n = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in ctx.files():
            if p.is_symlink():
                continue
            rel = ctx.rel(p)
            info = zipfile.ZipInfo(rel, date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 << 16)
            with open(p, "rb") as src, z.open(info, "w") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            n += 1
    os.replace(tmp, dest)
    return n


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain_value(v) -> bool:
    """A number, a flag or a short name without path separators (tier, device, version)."""
    if isinstance(v, (bool, int, float)):
        return True
    return isinstance(v, str) and len(v) <= 64 and not any(c in v for c in "/\\")


def compute(ctx) -> dict:
    """Tiers, devices and workers per stage, the CPU probe number, onnxruntime and ctranslate2
    versions (``hw.stage_plan`` → ``ctx.data["compute"]``). Only numbers and short names: anything
    else (a path, a nested structure) is dropped. No stage planned — ``{}``."""
    out: dict = {}
    for k, v in sorted((ctx.data.get("compute") or {}).items()):
        if isinstance(v, dict):
            v = {str(a): b for a, b in sorted(v.items()) if _plain_value(a) and _plain_value(b)}
        elif isinstance(v, (list, tuple)):
            v = [x for x in v if _plain_value(x)]
        elif not _plain_value(v):
            continue
        out[str(k)] = v
    return out


def manifest(ctx, ok: bool, archive: Path | None, n_files: int) -> dict:
    """Manifest for the main pipeline: versions, salt fingerprint, stages, counters, check
    results. No paths, no file names, no content."""
    failures = ctx.data.get("gate_failures") or []
    by_check = Counter(str(f.get("check", "?")) for f in failures)
    doc = {
        "format": "localanon-manifest/1",
        "localanon_version": __version__,
        "engine_commit": ENGINE_COMMIT,
        "salt_fingerprint": ctx.config.salt_fingerprint(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stages": list(ctx.stages_run),
        "ok": bool(ok),
        "gates": {"ok": not failures, "failures": len(failures),
                  "by_check": dict(sorted(by_check.items()))},
        "counters": {k: int(v) for k, v in sorted(ctx.counters.items())},
        "removed_by_category": ctx.removed_by_category(),
        "removed_borderline": sum(1 for r in ctx.removals if r.borderline),
        "marked_by_category": ctx.marked_by_category(),
        "notes": len(ctx.notes),
        "compute": compute(ctx),
        "archive": None,
    }
    if archive is not None:
        doc["archive"] = {"files": n_files, "bytes": archive.stat().st_size,
                          "sha256": _sha256(archive)}
    return doc


# --------------------------------------------------------------------------- report for the data owner

_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2em auto;max-width:72em;
padding:0 1em;color:#222;line-height:1.45}
h1{font-size:1.6em}h2{font-size:1.2em;margin-top:2em;border-bottom:1px solid #ccc}
table{border-collapse:collapse;width:100%;font-size:.95em}
th,td{border:1px solid #ddd;padding:.3em .5em;text-align:left;vertical-align:top;
word-break:break-word}th{background:#f3f3f3}
.ok{color:#1a7f37;font-weight:600}.bad{color:#b42318;font-weight:600}
.mark{background:#fff3cd;color:#7a4b00;font-weight:600;padding:0 .3em;border-radius:3px}
code,pre{background:#f6f8fa;padding:.1em .3em;border-radius:3px}
pre{padding:.6em;overflow:auto}.muted{color:#666}
"""


def _e(x) -> str:
    return html.escape(str(x), quote=True)


RESTORABLE = ("document", "unknown_binary")   # categories that [documents].restore brings back


def _restorable(category: str) -> bool:
    return category_key(category) in RESTORABLE


def _table(head: list, rows: list) -> str:
    if not rows:
        return '<p class="muted">none</p>'
    out = ["<table><thead><tr>"] + [f"<th>{_e(h)}</th>" for h in head] + ["</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _chats_html(chats) -> str:
    """Chat list from ctx.data['chats'] (written by the chat stage). Expected: a list of dicts with
    counters (chat name/id, messages total, kept, removed…) — all keys are shown as they are;
    a dict {chat: counters} and a list of strings are accepted too."""
    if not chats:
        return '<p class="muted">the chat stage did not run or there are no chats</p>'
    if isinstance(chats, dict):
        chats = [dict({"chat": k}, **(v if isinstance(v, dict) else {"value": v}))
                 for k, v in chats.items()]
    if not isinstance(chats, list):
        return f"<pre>{_e(chats)}</pre>"
    if all(isinstance(c, dict) for c in chats):
        keys: list = []
        for c in chats:
            for k in c:
                if k not in keys:
                    keys.append(k)
        rows = [[_e(c.get(k, "")) for k in keys] for c in chats]
        return _table(keys, rows)
    return "<ul>" + "".join(f"<li>{_e(c)}</li>" for c in chats) + "</ul>"


def _restore_html(removals) -> str:
    """“How to restore a file” by the owner's list (DESIGN §9b): which key brings back what."""
    restorable = [r for r in removals if _restorable(r.category)]
    example = next((r.rel for r in restorable if r.borderline),
                   restorable[0].rel if restorable else "folder/file.pdf")
    return "\n".join([
        "<h3>How to restore a file</h3>",
        "<p>Change the settings file (<code>localanon.toml</code>) and run the script again on the "
        "original export:</p><ul>",
        "<li><b>A document</b> (category <code>document:&lt;kind&gt;</code>) — add its path, as in the "
        "table, to <code>[documents].restore</code>:</li>",
        f"</ul><pre>[documents]\nrestore = [{_e(json.dumps(example, ensure_ascii=False))}]</pre><ul>",
        "<li><b>All documents of a kind</b> — remove the kind from "
        "<code>[documents].delete_kinds</code>; <code>borderline = \"keep\"</code> keeps doubtful "
        "matches (marked “check”).</li>",
        "<li><b>Messages</b> and the attachments removed with them (category "
        "<code>messages:&lt;category&gt;</code>), a chat emptied by the filter — remove the category "
        "from <code>[messages].drop_categories</code>, or turn the filter off with "
        "<code>[messages].filter = \"off\"</code>.</li>",
        "<li><b>Avatars and profile photos</b> (category <code>image</code>) — "
        "<code>[media].images = \"keep\"</code>.</li>",
        "<li><b>Audio and video</b> (category <code>media</code>) are removed after transcription: the "
        "transcript stays in the archive instead of the recording. A recording is not handed over.</li>",
        "<li><b>A chat</b> (category <code>chat</code>) removed by the settings — remove it from "
        "<code>[chats].exclude</code> (or add it to <code>[chats].include_only</code> if that list is "
        "set); a chat with no messages left after the filter — see “Messages”.</li>",
        "</ul><p>Unreadable files, archives that were not unpacked and files of unknown kinds are not "
        "removed: they stay in the archive, listed under “Kept with a mark”.</p>",
    ])


def report_html(ctx, ok: bool, archive: Path | None, n_files: int) -> str:
    cfg = ctx.config
    failures = ctx.data.get("gate_failures") or []
    removals = list(ctx.removals)
    borderline = [r for r in removals if r.borderline]
    parts = ["<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">",
             "<title>LocalAnonPipeline — report</title>", f"<style>{_CSS}</style></head><body>",
             "<h1>LocalAnonPipeline — pre-cleaning report</h1>",
             f'<p class="muted">localanon {_e(__version__)}, engine {_e(ENGINE_COMMIT)}, '
             f"project “{_e(cfg.project.name)}”, input: <code>{_e(ctx.input_path)}</code>, "
             f"{_e(datetime.now().strftime('%Y-%m-%d %H:%M'))}</p>"]
    if ok and archive is not None:
        passed = "Checks passed. " if not failures else ""
        parts.append(f'<p class="ok">{passed}The archive is built: <code>{_e(archive)}</code> '
                     f"({n_files} files).</p>")
        if failures:
            parts.append(f"<p>The checks at the end found {len(failures)} problems (section “Check "
                         "problems”). They do not stop the archive: the main pipeline finishes these "
                         "places, the manifest carries their count.</p>")
        parts.append(f"<p>Hand over only two files: <code>{_e(archive.name)}</code> and "
                     f"<code>{MANIFEST_NAME}</code>. This report stays with you: it contains paths and "
                     "reasons of removal.</p>")
    else:
        parts.append('<p class="bad">The archive is not built.</p>')
        if ctx.data.get("error"):
            parts.append(f"<p>The run was interrupted: <code>{_e(ctx.data['error'])}</code></p>")
    parts.append("<h2>Stages</h2>")
    parts.append("<p>" + (", ".join(f"<code>{_e(s)}</code>" for s in ctx.stages_run) or
                          '<span class="muted">no stage ran</span>') + "</p>")

    parts.append(f"<h2>Removed from the export ({len(removals)})</h2>")
    if borderline:
        parts.append(f'<p><span class="mark">check</span> — {len(borderline)} doubtful removals: the '
                     "signs are ambiguous, the file was removed to be safe.</p>")
    rows = []
    for r in removals:
        mark = ' <span class="mark">check</span>' if r.borderline else ""
        rows.append([f"<code>{_e(r.rel)}</code>", _e(r.category), _e(r.reason) + mark])
    parts.append(_table(["path", "category", "reason"], rows))
    parts.append(_restore_html(removals))

    marks = list(getattr(ctx, "marks", []))
    parts.append(f"<h2>Kept with a mark ({len(marks)})</h2>")
    if marks:
        parts.append("<p>These files stayed in the archive. The “not anonymized” mark "
                     "means the script could not read or process the file: it is passed as is and "
                     "our main pipeline finishes it. If a file must not be handed over, remove it "
                     "from the input and run the script again.</p>")
    parts.append(_table(["path", "category", "reason"],
                        [[f"<code>{_e(m.rel)}</code>", _e(m.category), _e(m.reason)]
                         for m in marks]))

    parts.append("<h2>Chats</h2>")
    parts.append(_chats_html(ctx.data.get("chats")))

    parts.append(f"<h2>Check problems ({len(failures)})</h2>")
    parts.append(_table(["file", "check", "detail"],
                        [[f"<code>{_e(f.get('rel', ''))}</code>", _e(f.get("check", "")),
                          _e(f.get("detail", ""))] for f in failures]))

    parts.append(f"<h2>Notes ({len(ctx.notes)})</h2>")
    parts.append("<ul>" + "".join(f"<li>{_e(n)}</li>" for n in ctx.notes) + "</ul>"
                 if ctx.notes else '<p class="muted">none</p>')

    parts.append("<h2>Counters</h2>")
    parts.append(_table(["key", "value"],
                        [[_e(k), _e(v)] for k, v in sorted(ctx.counters.items())]))
    parts.append("</body></html>")
    return "\n".join(parts)


def finish(ctx, ok: bool) -> None:
    """Build the output: if ok — archive + manifest + report; otherwise only the report (and a
    manifest with ok=false, so that the run is visible)."""
    out = Path(ctx.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = None
    n_files = 0
    if ok:
        archive = out / archive_name(ctx)
        n_files = build_archive(ctx, archive)
    doc = manifest(ctx, ok, archive, n_files)
    (out / MANIFEST_NAME).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    if ctx.config.output.report or not ok:
        (out / REPORT_NAME).write_text(report_html(ctx, ok, archive, n_files), encoding="utf-8")
