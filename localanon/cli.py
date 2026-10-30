"""Command line: ``python -m localanon init <config>`` and ``python -m localanon run …``.

Exit codes of ``run``: 0 — the archive is built; 1 — a stage failed (stage name and error in the
message, the report is written); 2 — config, input or argument error; 3 — structural checks failed
(no archive, the report is written).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from . import __version__
from . import config as cfgmod
from .context import Removal, RunContext
from . import stages


class UsageError(Exception):
    """A clear config/input error — printed as one line, exit code 2."""


def _err(msg: str) -> None:
    print(f"localanon: {msg}", file=sys.stderr, flush=True)


def cmd_init(ns) -> int:
    p = Path(ns.config)
    if p.exists() and not ns.force:
        _err(f"{p} already exists — not overwriting (--force overwrites it, the salt changes)")
        return 2
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(cfgmod.template(ns.name or ""), encoding="utf-8")
    print(f"config created: {p}")
    return 0


def _load_config(path: str):
    p = Path(path)
    if not p.is_file():
        raise UsageError(f"config not found: {p} (create it: localanon init {p})")
    try:
        return cfgmod.load(p)
    except ValueError as e:
        raise UsageError(f"config {p}: {e}")
    except Exception as e:  # tomllib.TOMLDecodeError and other parse errors
        raise UsageError(f"config {p} cannot be read: {type(e).__name__}: {e}")


def _copy_input(src: Path, work: Path, ctx: RunContext) -> None:
    """Copy the input to the working copy. Symlinks are neither copied nor followed: the file behind
    a link may lie outside the export."""
    import os
    skipped = 0
    for root, dirs, names in os.walk(src):
        dirs.sort()
        rel = Path(root).relative_to(src)
        (work / rel).mkdir(parents=True, exist_ok=True)
        keep = []
        for d in dirs:
            if (Path(root) / d).is_symlink():
                skipped += 1
            else:
                keep.append(d)
        dirs[:] = keep
        for n in sorted(names):
            p = Path(root) / n
            if p.is_symlink():
                skipped += 1
                continue
            shutil.copyfile(p, work / rel / n)
    if skipped:
        ctx.count("input:links_skipped", skipped)
        ctx.note(f"the input has {skipped} symbolic links — skipped (a link may point to a file "
                 "outside the export)")


def _unpack_input(src: Path, work: Path, ctx: RunContext) -> None:
    from .stages import inventory
    if inventory.archive_format(src.name) is None:
        raise UsageError(f"input {src}: neither a directory nor an archive of a known format "
                         "(zip, tar, tar.gz, tgz, 7z, rar)")
    try:
        # the input archive has no absolute size limit (an export with media can exceed 2 GiB):
        # the compression ratio, disk space and the check of what was actually written remain
        skipped = inventory.extract_archive(src, work, unlimited=True)
    except inventory.UnpackError as e:
        raise UsageError(f"input {src} not extracted: {e}")
    work.mkdir(parents=True, exist_ok=True)
    for name, why in skipped:
        ctx.removals.append(Removal("!" + name, "archive_member", f"not extracted: {why}", False))
        ctx.count("removed:archive_member")


def _stage_list(ns) -> list:
    if not ns.stages:
        return list(stages.ORDER)
    want = [s.strip() for s in ns.stages.split(",") if s.strip()]
    unknown = [s for s in want if s not in stages.ORDER]
    if unknown:
        raise UsageError(f"--stages: unknown stages {', '.join(unknown)}; "
                         f"known: {', '.join(stages.ORDER)}")
    return [s for s in stages.ORDER if s in want]      # order as in stages.ORDER


def cmd_run(ns) -> int:
    try:
        return _run(ns)
    except UsageError as e:
        _err(str(e))
        return 2


def _run(ns) -> int:
    cfg = _load_config(ns.config)
    src = Path(ns.input)
    if not src.exists():
        raise UsageError(f"input not found: {src}")
    out = Path(ns.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise UsageError(f"output directory {out}: {e.strerror or e}")
    plan = _stage_list(ns)
    if ns.tmp and not Path(ns.tmp).is_dir():
        raise UsageError(f"--tmp {ns.tmp}: no such directory")
    tmp = Path(tempfile.mkdtemp(prefix="localanon-", dir=ns.tmp))
    work = tmp / "work"
    rc = 1
    from . import viewcache
    try:
        work.mkdir()
        viewcache.enable(tmp / "views")    # document views shared by the stages and their workers
        ctx = RunContext(config=cfg, input_path=src, work=work, out=out)
        if src.is_dir():
            _copy_input(src, work, ctx)
        else:
            _unpack_input(src, work, ctx)
        shard = getattr(ns, "shard", "")
        if shard:
            from . import hw
            k, n = hw.parse_shard(shard)

            def warn(msg):
                ctx.note(msg)
                print(f"! {msg}", file=sys.stderr)
            ctx.data["shard"] = {"k": k, "n": n, **hw.select_shard(work, k, n, note=warn)}
        failed = None
        for name in plan:
            try:
                mod = stages.load(name)
                mod.run(ctx)
            except Exception as e:  # noqa: BLE001 — any stage error is reported with the stage name
                failed = (name, e)
                break
            ctx.stages_run.append(name)
        from . import pack
        if failed is not None:
            name, e = failed
            ctx.data["error"] = f"stage {name}: {type(e).__name__}: {e}"
            _err(f"stage \"{name}\" failed with {type(e).__name__}: {e}")
            if ns.verbose:
                traceback.print_exception(e, file=sys.stderr)
            rc = 1
        else:
            rc = 0
            if ctx.data.get("gate_failures"):
                # the checks at the end never stop the archive: their findings go to the report and
                # their count to the manifest, the main pipeline finishes these places
                n = len(ctx.data["gate_failures"])
                print(f"! checks found {n} problems — the archive is built anyway; details in "
                      f"{out / pack.REPORT_NAME}", file=sys.stderr, flush=True)
        try:
            pack.finish(ctx, ok=(rc == 0))
        except Exception as e:  # noqa: BLE001
            _err(f"packing failed: {type(e).__name__}: {e}")
            if ns.verbose:
                traceback.print_exception(e, file=sys.stderr)
            return 1
        if rc == 0:
            print(f"done: {out / pack.archive_name(ctx)}; manifest {out / pack.MANIFEST_NAME}; "
                  f"report {out / pack.REPORT_NAME}", flush=True)
        else:
            print(f"report: {out / pack.REPORT_NAME}", flush=True)
        return rc
    finally:
        viewcache.disable()
        if ns.keep_work:
            print(f"working copy kept: {work}", flush=True)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def cmd_hw(ns) -> int:
    from . import hw
    cfg = cfgmod.load(ns.config) if ns.config else cfgmod.Config()
    res = hw.detect()
    probes = hw.install_probes(res, res.notes.append) if ns.probe_cuda else {}
    pl = hw.plan(res, cfg)
    if ns.json:
        import json
        print(json.dumps({"resources": res.to_dict(), "plan": pl.to_dict(), **({"probes": probes} if probes else {})},
                         ensure_ascii=False, indent=2))
    else:
        print(hw.describe(res, pl))
        for what, st in probes.items():
            print(f"CUDA probe {what}: " + {"ok": "works", "failed": "FAILED (reason above)",
                                             "skip": "skipped (no NVIDIA card or package)"}[st])
    # --probe-cuda: exit code 1 if a probe failed — the install script then fails
    return 1 if "failed" in probes.values() else 0


def cmd_merge(ns) -> int:
    import zipfile
    from . import hw
    try:
        info = hw.merge_outputs([Path(d) for d in ns.dirs], Path(ns.out))
    except zipfile.BadZipFile:
        print("merge: shard archive is damaged — rebuild this shard", file=sys.stderr)
        return 2
    except ValueError as e:                  # merge texts have no paths; JSONDecodeError is a ValueError too
        print(e if str(e).startswith("merge:") else f"merge: shard manifest cannot be read ({type(e).__name__})",
              file=sys.stderr)
        return 2
    for n in info.get("notes", []):
        print(n, file=sys.stderr)
    print(f"shards merged: {info['shards']}, files in the archive: {info['files']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="localanon", description="Local pre-cleaning of exports")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init", help="create a config with a fresh salt")
    a.add_argument("config")
    a.add_argument("--name", default="")
    a.add_argument("--force", action="store_true")
    a.set_defaults(fn=cmd_init)
    r = sub.add_parser("run", help="clean an export")
    r.add_argument("--in", dest="input", required=True, help="export directory or archive")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--config", required=True)
    r.add_argument("--tmp", default=None, help="directory for the working copy (system default)")
    r.add_argument("--stages", default="", help="comma-separated — only these stages (debugging)")
    r.add_argument("--keep-work", action="store_true", help="do not delete the working copy")
    r.add_argument("--verbose", "-v", action="store_true", help="traceback on a stage error")
    r.add_argument("--shard", default="", help="k/N — only the k-th of N parts of the input (several machines)")
    r.set_defaults(fn=cmd_run)
    h = sub.add_parser("hw", help="detected hardware and the stage layout on it")
    h.add_argument("--config", default="")
    h.add_argument("--json", action="store_true")
    h.add_argument("--probe-cuda", action="store_true",
                   help="compute probes of onnxruntime and ctranslate2 on the GPU (after installation); failure — exit code 1")
    h.set_defaults(fn=cmd_hw)
    m = sub.add_parser("merge", help="merge shard outputs (run --shard) into one")
    m.add_argument("dirs", nargs="+", help="shard output directories")
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_merge)
    ns = ap.parse_args(argv)
    return ns.fn(ns)
