"""Command-line interface.

    python -m anonymizer scan   <path>              # dry run: what would be masked
    python -m anonymizer apply  <path>              # mask in place
    python -m anonymizer gate   <path> [--controls FILE]
    python -m anonymizer build  <path> <out.zip>    # package (excludes forbidden files)

Common: --config FILE  --confidence high,context[,low]  --detectors a,b,c
Runs fully general (pattern detectors) with no corpus knowledge; add
--known brands,clients for extra precision on a familiar corpus.
Salt for scan/apply: --salt-file <file with 64 hex chars> or the ANON_SALT variable.
"""
from __future__ import annotations

import argparse
import sys

from .config.default import load
from .engine.pipeline import Pipeline


def _die(msg):
    """argparse-style fatal error (the parser object is out of scope here)."""
    print(f"anonymizer: error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _cfg(args):
    cfg = load(args.config)
    if args.confidence:
        tiers = tuple(args.confidence.split(","))
        unknown = set(tiers) - {"high", "context", "low"}
        if unknown:
            _die(f"unknown confidence tiers: {', '.join(sorted(unknown))} "
                 f"(valid: high, context, low)")
        cfg["confidence"] = tiers
    if args.detectors:
        want = set(args.detectors.split(","))
        # only-these semantics turn everything unlisted OFF, so a typo'd name
        # must abort loudly — otherwise `apply` exits 0 having masked nothing.
        unknown = want - set(cfg["detectors"])
        if unknown:
            _die(f"unknown detectors: {', '.join(sorted(unknown))} "
                 f"(valid: {', '.join(cfg['detectors'])})")
        cfg["detectors"] = {k: (k in want) for k in cfg["detectors"]}
    if args.known:
        cfg["known_values"] = args.known.split(",")
    if getattr(args, "known_file", None):
        cfg["known_files"] = list(args.known_file)   # paths to external packs
    return cfg


SALT_ENV = "ANON_SALT"


def _read_salt(args) -> str:
    """Salt from --salt-file (a file with 64 hex chars) or the ANON_SALT environment variable.
    The value is never printed, neither in errors nor in logs."""
    import os
    path = getattr(args, "salt_file", None)
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError as e:
            _die(f"--salt-file {path!r}: {e.strerror or e}")
    salt = os.environ.get(SALT_ENV, "").strip()
    if not salt:
        _die(f"scan/apply need a salt: --salt-file <file with 64 hex chars> or the {SALT_ENV} variable")
    return salt


_STATE_TMP = None   # temporary state dir (wire_salt without state_dir): one per process


def wire_salt(salt_hex: str, mint_version: int = 2, state_dir=None) -> None:
    """Standalone engine salt setup (no conveyor): mint mints tokens, pseudo does HMAC
    linking; the state dir for pseudo is created when needed.
    Without ``state_dir`` a temporary dir is used: the previous one is removed on the next
    call, the last one at process exit (nothing is left in tmp)."""
    global _STATE_TMP
    import os
    import tempfile
    from .engine import mint as _mint
    from .engine import pseudo as _pseudo
    _mint.set_tenant(salt_hex, mint_version)
    tmp = None
    if state_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="anonymizer-state-")
        state_dir = tmp.name
    os.makedirs(os.path.join(state_dir, "state"), exist_ok=True)
    _pseudo.set_tenant(salt_hex, state_dir)
    if _STATE_TMP is not None:
        _STATE_TMP.cleanup()
    _STATE_TMP = tmp


def _wire_tenant(args):
    """The salt is required for scan/apply: without it mint dies IN THE MIDDLE of apply,
    leaving a partly masked corpus. Order: flags are validated first (_cfg), then the
    salt, so a typo in the flags is not hidden behind the salt requirement."""
    wire_salt(_read_salt(args), int(args.mint_version), getattr(args, "run", None))


def cmd_scan(args, apply=False):
    cfg = _cfg(args)      # 1) flags (fail loudly on a typo)
    _wire_tenant(args)    # 2) salt (fail loudly BEFORE the first write)
    rep = Pipeline(cfg).run(args.path, apply=apply)
    print(rep.render(dry=not apply))
    return 0


def cmd_gate(args):
    from .gates import structural, control
    ok_s, f = structural.run(args.path)
    print(structural.render(ok_s, f))
    ok_c = True
    if args.controls:
        ok_c, res = control.run(args.path, control.load_pack(args.controls))
        print(control.render(ok_c, res))
    return 0 if (ok_s and ok_c) else 1


def cmd_build(args):
    import os
    import zipfile
    import hashlib
    from .gates.structural import is_forbidden
    n = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(args.path):
            for f in fs:
                p = os.path.join(dp, f)
                relp = os.path.relpath(p, args.path)
                # test the WHOLE relative path (same oracle as the gate): a
                # forbidden name carried by a parent dir (surrogate_map/…)
                # must not slip into the release zip via its basename.
                if is_forbidden(relp):
                    continue
                z.write(p, relp)
                n += 1
    md5 = hashlib.md5(open(args.out, "rb").read()).hexdigest()
    print(f"packed {n} files -> {args.out}  md5={md5}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="anonymizer", description="Text-corpus anonymization pipeline")
    ap.add_argument("--config")
    ap.add_argument("--confidence")
    ap.add_argument("--detectors")
    ap.add_argument("--known")
    ap.add_argument("--run", help="state dir (state/ for linking); temporary by default")
    ap.add_argument("--salt-file", help="salt file (64 hex chars); otherwise the ANON_SALT variable")
    ap.add_argument("--mint-version", type=int, default=2, choices=(1, 2),
                    help="token minting version (default 2)")
    ap.add_argument("--known-file", action="append",
                    help="path to an external pack file (runs/<co>/packs/<co>.py); repeatable")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "apply"):
        p = sub.add_parser(name)
        p.add_argument("path")
    pg = sub.add_parser("gate")
    pg.add_argument("path")
    pg.add_argument("--controls")
    pb = sub.add_parser("build")
    pb.add_argument("path")
    pb.add_argument("out")

    args = ap.parse_args(argv)
    if args.cmd == "scan":
        return cmd_scan(args, apply=False)
    if args.cmd == "apply":
        return cmd_scan(args, apply=True)
    if args.cmd == "gate":
        return cmd_gate(args)
    if args.cmd == "build":
        return cmd_build(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
