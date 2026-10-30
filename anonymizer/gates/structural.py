"""Structural acceptance gate — runs after every pass; the idempotency oracle.

Fails on: broken JSONL, a mask token welded onto a letter (glue), forbidden
files in the tree, and JSON number/timestamp/hash fields clobbered by a token.
Returns (ok, findings). Non-zero BLOCKING findings block a release.

Robustness (the client may send any files):
  * a defect is only what was NOT in the client's source: the
    glue/field_type/broken counters per line of a work/ file are compared with
    the same file in <run>/raw/ (the path before the rename pass is unwound via
    state/rename_journal.jsonl). "The client had it before anonymization" → WARN
    pre_existing, not BLOCK; no source — everything counts as new (as before,
    coverage is not lost silently);
  * FORBIDDEN is split by meaning: pipeline service artifacts → BLOCK; raw
    client databases (*.db/*.sqlite) → WARN unshippable (not packed into the
    release by is_forbidden, do not block the round); client .log/.py/
    README/docs/state/… are ordinary content;
  * a file larger than ANON_MAX_FILE_BYTES (default 50 MB) is not read into
    memory whole, but is NOT skipped either (the two largest Telegram exports
    used to go into the release unchecked): the profile is computed as a stream,
    in chunks (_profile_stream — the same result as _profile over the whole
    text, memory ≈ chunk + margin). WARN skipped_large — only above
    ANON_GATE_STREAM_MAX_BYTES (default 4 GiB): there the limit is time, not
    memory; such a skip is printed loudly.
"""
from __future__ import annotations

import json
import os
import re

from ..engine.boundary import GLUE_RE
from ..engine.walk import (MAX_FILE_BYTES_ENV, SKIP_EXT,
                           iter_files, max_file_bytes, read_text)

# The shipping blacklist has TWO meanings (one regex used to confuse pipeline
# service artifacts with the NAMES of client files → a deterministic round BLOCK
# on logs/chat.log, attachments/bot.py, README.md, k8s/tenant.yaml,
# docs/state/, finance/Audit/, cache/app.sqlite):
#   1) pipeline service artifacts — cannot be in a release, they have no way into
#      the work/ tree (they live in run_dir); found → BLOCK:
#        state/ (entity_map/salt/decisions), audit/, audit_work/, tenant.yaml
#        (64-hex reversibility salt), approved.tsv / candidates.tsv (mask
#        literals = the personal data itself) — ONLY at the top level of the tree:
#        that is where they lie in run_dir (protection against mixing up "run
#        instead of work"), while nested docs/state/, k8s/tenant.yaml,
#        finance/Audit/ are client content;
#        entity_map* / surrogate_map* — ANYWHERE in the path: the whole
#        token→original key (de-anonymization), the cost of a false miss
#        is out of all proportion to the cost of a false ban;
#   2) a raw client database (*.db + sqlite sidecars -wal/-shm/-journal,
#      *.sqlite/.sqlite3) — a binary, its content is not anonymized → it cannot
#      go into a release (release._pack / cmd_build / the archive gate filter by
#      is_forbidden), but this is not a pipeline failure: the gate gives a WARN
#      unshippable with the name and does NOT block the round.
#   *.log / *.py / readme* are no longer forbidden: transcripts, dev-Discord
#   attachments, a repository README are shipped content; the pipeline keeps its
#   own logs and packs outside work/ (state/pipeline/, packs/).
_ARTIFACT_RX = (
    r"^(?:state|audit|audit_work)(?:/|$)|"
    r"^tenant\.ya?ml$|^approved\.tsv$|^candidates\.tsv$|"
    r"surrogate_map|entity_map"
)
_RAW_DB_RX = r"(?:\.db(?:-wal|-shm|-journal)?|\.sqlite3?)$"
PIPELINE_ARTIFACT = re.compile(_ARTIFACT_RX, re.I)
RAW_DB = re.compile(_RAW_DB_RX, re.I)
# The full packing blacklist: what must not get into the archive.
FORBIDDEN = re.compile(f"{_ARTIFACT_RX}|{_RAW_DB_RX}", re.I)
NUM_FIELD = re.compile(r'"(size|size_bytes|count|duration|width|height|bitrate|position)"\s*:\s*"?[A-ZА-Я][\w]*_(?:\d+|REDACTED)')
TS_FIELD = re.compile(r'"(created|updated|resolved|timestamp|edited|created_at|updated_at)"\s*:\s*"?[A-ZА-Я][\w]*_(?:REDACTED|\d+)')

# File size cap for reading into memory — the same env and the same parser as walk
# (walk.max_file_bytes; the name is kept for tests and messages).
MAX_BYTES_ENV = MAX_FILE_BYTES_ENV
# a file above the cap is checked as a stream; skipped only above this limit
STREAM_MAX_ENV = "ANON_GATE_STREAM_MAX_BYTES"
DEFAULT_STREAM_MAX = 4 * 1024 ** 3
# gate-only knobs: the check runs AFTER writing, they do not affect the output of
# computation units — the engine knob check (test_incremental.test_env_knobs_listed) takes them from here
GATE_KNOBS = (STREAM_MAX_ENV, "ANON_YO_FOLD",   # ANON_YO_FOLD — control gate (gates/control.py)
              "ANON_GATE_JSON_UNESCAPE")      # control gate over decoded JSON
_STREAM_CHUNK = 4 * 1024 * 1024    # chars per read
_STREAM_MARGIN = 64 * 1024         # margin before the buffer end: a match is not cut by a chunk boundary
_STREAM_BACK = 32                  # chars before the resume point (lookbehind, context)

_BLOCKING = ("broken_jsonl", "glue", "forbidden", "field_type")
_WARN_KEYS = ("pre_existing", "unshippable", "skipped_large")
_WARN_FILES = 20      # detailed WARN lines per run; then one summary


def _norm(rel: str) -> str:
    return rel.replace("\\", "/")


def is_pipeline_artifact(relpath: str) -> bool:
    """A pipeline service artifact by its PLACE in the tree (relpath from the root):
    the BLOCK class of the structural gate."""
    return bool(PIPELINE_ARTIFACT.search(_norm(relpath)))


def is_raw_db(relpath: str) -> bool:
    """A raw client database (*.db + sqlite sidecars, *.sqlite): not packed,
    but does not block the round (WARN unshippable)."""
    return bool(RAW_DB.search(_norm(relpath)))


def is_forbidden(relpath: str) -> bool:
    """True if ``relpath`` (RELATIVE to the tree root) must not ship. Matching
    the whole relative path — not the basename — is load-bearing: a forbidden
    name carried by a parent directory (``surrogate_map/part1.csv``) is the
    token→original re-identification key itself and must be caught. Shared
    helper so release._pack/cmd_build apply the exact same test before packing
    (= a pipeline service artifact OR a raw client database)."""
    return bool(FORBIDDEN.search(_norm(relpath)))


def _max_bytes() -> int:
    """Cap from ANON_MAX_FILE_BYTES (bytes) — parsed by walk.max_file_bytes:
    empty → default; <= 0 → 0, no cap (a file of any size in memory); not a number →
    a walk WARN (stderr, once per process) + default (the gate does not fail on a bad env)."""
    return max_file_bytes(nonpositive=0)


def _stream_max() -> int:
    """Streaming check limit from ANON_GATE_STREAM_MAX_BYTES (bytes); empty →
    default 4 GiB; <= 0 → no limit; not a number → WARN + default."""
    raw = os.environ.get(STREAM_MAX_ENV, "").strip()
    if not raw:
        return DEFAULT_STREAM_MAX
    try:
        return int(raw)
    except ValueError:
        print(f"structural: WARN {STREAM_MAX_ENV}={raw!r} is not a number — "
              f"default {DEFAULT_STREAM_MAX}", flush=True)
        return DEFAULT_STREAM_MAX


# --------------------------------------------------------------------------- #
# the client's source (before) to compare with the output (after)
# --------------------------------------------------------------------------- #
def _baseline_dir(root: str, baseline) -> str | None:
    """Source dir for before/after. baseline=None → auto: <run>/raw
    when root is <run>/work (the round.py signature does not change); '' →
    comparison off; otherwise an explicit dir. No dir → None (all defects
    count as new — coverage is not lost)."""
    if baseline is None:
        root = os.path.abspath(root)
        if os.path.basename(root) != "work":
            return None
        baseline = os.path.join(os.path.dirname(root), "raw")
    if not baseline or not os.path.isdir(baseline):
        return None
    return baseline


def _load_rename_journal(run_dir: str):
    """[(src, dst)] from state/rename_journal.jsonl in order; no journal →
    []; a broken record/unreadable file → None (mapping unknown)."""
    journal = os.path.join(run_dir, "state", "rename_journal.jsonl")
    if not os.path.isfile(journal):
        return []
    out = []
    try:
        with open(journal, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                src, dst = rec["src"], rec["dst"]
                if not isinstance(src, str) or not isinstance(dst, str):
                    return None
                out.append((_norm(src), _norm(dst)))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return out


def _reverse_rename_map(run_dir: str, base_dir: str):
    """{current rel in work/: source rel in raw/}: the journal is replayed
    FORWARD over the raw/ list — the same semantics as release._expected_rels /
    mine._rename_map (an exact file rename is O(1), a dir rename works as a
    PREFIX; duplicate records from rollback+retries are idempotent: the second
    pass no longer finds the prefix — unwinding the journal backwards broke on
    duplicates). No journal → {} (identity); broken → None."""
    journal = _load_rename_journal(run_dir)
    if not journal:
        return journal                      # None (broken) / [] (no journal)
    rev = {}
    for dp, _dns, fns in os.walk(base_dir):
        for fn in fns:
            r = _norm(os.path.relpath(os.path.join(dp, fn), base_dir))
            rev[r] = r                      # current → original
    for src, dst in journal:
        if src in rev:                      # exact file rename
            rev[dst] = rev.pop(src)
            continue
        pre = src + "/"
        for r in [r for r in rev if r.startswith(pre)]:
            rev[dst + "/" + r[len(pre):]] = rev.pop(r)
    return rev


def _baseline_profile(base_dir: str, run_dir: str, rel: str, rev_box: list,
                      cap: int, is_json: bool):
    """(source profile | None, reason it is missing). A large source (> cap)
    is streamed, like the output: otherwise its "as received" defects would count
    as new and give a false BLOCK. The rename map is built lazily once — only
    when the first file has defects."""
    if rev_box[0] is False:
        rev_box[0] = _reverse_rename_map(run_dir, base_dir)
    rev = rev_box[0]
    why = "no source in raw/"
    if rev is None:
        orig = rel                          # broken journal — try identity
        why += " (broken state/rename_journal.jsonl — the path before rename is not restored)"
    elif not rev:
        orig = rel                          # no journal — names did not change
    else:
        orig = rev.get(_norm(rel))
        if orig is None:
            return None, why + " (no original matches the file per rename_journal)"
    p = os.path.join(base_dir, orig)
    if not os.path.isfile(p):
        return None, why
    try:
        size = os.stat(p).st_size
    except OSError:
        return None, why
    if cap > 0 and size > cap:
        smax = _stream_max()
        if smax > 0 and size > smax:
            return None, (f"source {orig} is above the streaming check limit "
                          f"{STREAM_MAX_ENV}={smax} ({size} bytes)")
        prof = _profile_stream(p, is_json)
        if prof is None:
            return None, f"source {orig} is not readable"
        return prof, ""
    text = read_text(p)
    if text is None:
        return None, f"source {orig} is not readable"
    return _profile(text, is_json), ""


# --------------------------------------------------------------------------- #
# file defect profile: counters per line
# --------------------------------------------------------------------------- #
def _line_hits(rx, text: str, ctx) -> dict:
    """{line number: (hit count, context of the first)} in ONE pass over the
    text (the line number is counted incrementally, O(len(text)))."""
    hits, line, pos = {}, 1, 0
    for m in rx.finditer(text):
        line += text.count("\n", pos, m.start())
        pos = m.start()
        if line in hits:
            hits[line] = (hits[line][0] + 1, hits[line][1])
        else:
            hits[line] = (1, ctx(text, m))
    return hits


def _glue_ctx(text, m):
    return f"…{text[max(0, m.start() - 15):m.end() + 10]}…"


def _field_ctx(text, m):
    return m.group(0)[:50]


def _parses(s: str) -> bool:
    try:
        # a BOM before JSON is not a pipeline failure (the file came like that)
        json.loads(s.lstrip("﻿"))
        return True
    except Exception:
        return False


def _broken_lines(text: str) -> tuple[int, set]:
    """(JSON documents/lines checked, {broken line numbers}). First the whole
    file: a .json is usually ONE (pretty-printed) document, a line-by-line parse
    would fail every valid one on the opening '{'; a .jsonl of one pretty-printed
    document is valid too. If it does not parse — the line model (JSONL,
    also under a .json name)."""
    if _parses(text):
        return 1, set()
    broken, n = set(), 0
    for i, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        if not s:
            continue
        n += 1
        if not _parses(s):
            broken.add(i)
    return n, broken


def _profile(text: str, is_json: bool) -> dict:
    p = {"glue": _line_hits(GLUE_RE, text, _glue_ctx), "n_json": 0,
         "broken": set(), "field_type": {}, "lines": text.count("\n")}
    if is_json:
        p["n_json"], p["broken"] = _broken_lines(text)
        ft = _line_hits(NUM_FIELD, text, _field_ctx)
        for k, v in _line_hits(TS_FIELD, text, _field_ctx).items():
            ft[k] = (ft[k][0] + v[0], ft[k][1]) if k in ft else v
        p["field_type"] = ft
    return p


# --------------------------------------------------------------------------- #
# the same profile as a stream — files above the cap are not skipped
# --------------------------------------------------------------------------- #
def _iter_chunks(path):
    """File text in chunks of _STREAM_CHUNK chars — the same decoding as
    read_text (utf-8 + surrogateescape, newline='': CRLF as is). OSError/
    UnicodeError propagate (the caller treats it as read_text None)."""
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as fh:
        while True:
            s = fh.read(_STREAM_CHUNK)
            if not s:
                return
            yield s


class _RxStream:
    """_line_hits over the text in chunks. A match counts when ≥ _STREAM_MARGIN
    chars remain before the buffer end (or the file has ended): a match crossing a
    chunk boundary (\\s* in NUM_FIELD/TS_FIELD across a line break) is not cut.
    The next search starts at the end of the last match (like finditer over the
    whole text), _STREAM_BACK chars are kept before the resume point
    (GLUE_RE lookbehind, _glue_ctx context). The line number is incremental."""

    def __init__(self, rx, ctx):
        self.rx, self.ctx, self.hits = rx, ctx, {}
        self.buf, self.pos, self.lpos, self.line = "", 0, 0, 1

    def feed(self, chunk: str, final: bool = False) -> None:
        self.buf += chunk
        buf = self.buf
        limit = len(buf) if final else len(buf) - _STREAM_MARGIN
        if limit > self.pos:
            for m in self.rx.finditer(buf, self.pos):
                if m.start() >= limit:
                    break
                self.line += buf.count("\n", self.lpos, m.start())
                self.lpos = m.start()
                if self.line in self.hits:
                    n, c = self.hits[self.line]
                    self.hits[self.line] = (n + 1, c)
                else:
                    self.hits[self.line] = (1, self.ctx(buf, m))
                self.pos = m.end()
            self.pos = max(self.pos, limit)
        cut = self.pos - _STREAM_BACK
        if cut > 0:
            if self.lpos < cut:
                self.line += buf.count("\n", self.lpos, cut)
                self.lpos = cut
            self.buf = buf[cut:]
            self.pos -= cut
            self.lpos -= cut


# a JSON token with leading whitespace: 1 string, 2 number, 3 literal, 4 punctuation.
# Exactly what json.loads accepts (C scanner, strict): control chars in a
# string — no, leading zeros — no, NaN/Infinity/-Infinity — yes.
_JTOK = re.compile(
    r'[ \t\n\r]*(?:'
    r'("[^"\\\x00-\x1f]*(?:\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})[^"\\\x00-\x1f]*)*")'
    r'|(-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)'
    r'|(true|false|null|NaN|Infinity|-Infinity)'
    r'|([{}\[\]:,]))')
_JWS = re.compile(r'[ \t\n\r]*')
# an unclosed string that may still become valid (all chars allowed, an unfinished
# escape possible at the end): wait for data instead of declaring it invalid
_JPART_STR = re.compile(
    r'[ \t\n\r]*"[^"\\\x00-\x1f]*(?:\\(?:["\\/bfnrt]|u[0-9a-fA-F]{0,4})?[^"\\\x00-\x1f]*)*')
_JSON_MARGIN = 64      # a token at the very end of the buffer is not taken: "1" may become "1.5"


class _JsonStream:
    """A streaming analogue of _parses(text): json.loads(text.lstrip('\\ufeff')) without
    an exception ⇔ ok. A regex tokenizer + a stack automaton; exactly one top-level
    value. Buffer ≈ chunk + margin (+ the length of the longest string): at a
    position with no token we wait for data only if ahead there is whitespace, an
    unfinished string or a tail shorter than the margin — otherwise invalid at once.
    Nesting depth is not limited (json.loads on old Pythons fails with
    RecursionError at >~1000 levels — valid JSON is not broken because of that)."""

    def __init__(self):
        self.buf, self.pos, self.stack = "", 0, []
        self.state, self.ok, self.started = "V", True, False

    def _done(self) -> None:
        self.state = "D" if not self.stack else ("CO" if self.stack[-1] == "{" else "CA")

    def feed(self, chunk: str, final: bool = False) -> None:
        if not self.ok:
            return
        buf = self.buf[self.pos:] + chunk
        self.pos = 0
        if not self.started:
            buf = buf.lstrip("﻿")
            if not buf and not final:
                self.buf = ""
                return
            self.started = True
        self.buf = buf
        n = len(buf)
        while True:
            m = _JTOK.match(buf, self.pos)
            if m is None:
                if not final and not (_JWS.fullmatch(buf, self.pos)
                                      or _JPART_STR.fullmatch(buf, self.pos)
                                      or n - self.pos < _JSON_MARGIN):
                    self.ok = False          # no token can start at this position
                    return
                break
            if not final and m.end() + _JSON_MARGIN > n:
                break
            g, st = m.lastindex, self.state
            if g == 4:
                c = m.group(4)
                if c == "{" and st in ("V", "VA"):
                    self.stack.append("{")
                    self.state = "K"
                elif c == "[" and st in ("V", "VA"):
                    self.stack.append("[")
                    self.state = "VA"
                elif (c == "}" and st in ("K", "CO")) or (c == "]" and st in ("VA", "CA")):
                    self.stack.pop()
                    self._done()
                elif c == ":" and st == "C":
                    self.state = "V"
                elif c == "," and st == "CO":
                    self.state = "KR"
                elif c == "," and st == "CA":
                    self.state = "V"
                else:
                    self.ok = False
                    return
            elif g == 1 and st in ("K", "KR"):
                self.state = "C"
            elif st in ("V", "VA"):
                self._done()
            else:
                self.ok = False
                return
            self.pos = m.end()
        if final and (self.state != "D" or _JWS.fullmatch(buf, self.pos) is None):
            self.ok = False


def _broken_lines_stream(path: str) -> tuple[int, set]:
    """The line model of _broken_lines (JSONL) over the file in chunks: lines split
    by "\\n", like text.split("\\n")."""
    broken, n, i, tail = set(), 0, 0, ""
    for chunk in _iter_chunks(path):
        parts = (tail + chunk).split("\n")
        tail = parts.pop()
        for line in parts:
            i += 1
            s = line.strip()
            if s:
                n += 1
                if not _parses(s):
                    broken.add(i)
    i += 1
    s = tail.strip()
    if s:
        n += 1
        if not _parses(s):
            broken.add(i)
    return n, broken


def _profile_stream(path: str, is_json: bool):
    """_profile over the file in chunks: the same result, memory ≈ chunk +
    margin. None — the file is not readable (like read_text None)."""
    glue = _RxStream(GLUE_RE, _glue_ctx)
    num = _RxStream(NUM_FIELD, _field_ctx) if is_json else None
    ts = _RxStream(TS_FIELD, _field_ctx) if is_json else None
    jv = _JsonStream() if is_json else None
    feeds = [x for x in (glue, num, ts, jv) if x is not None]
    lines = 0
    try:
        for chunk in _iter_chunks(path):
            lines += chunk.count("\n")
            for f in feeds:
                f.feed(chunk)
        for f in feeds:
            f.feed("", final=True)
    except (OSError, UnicodeError):
        return None
    p = {"glue": glue.hits, "n_json": 0, "broken": set(), "field_type": {},
         "lines": lines}
    if is_json:
        if jv.ok:
            p["n_json"] = 1
        else:
            try:
                p["n_json"], p["broken"] = _broken_lines_stream(path)
            except (OSError, UnicodeError):
                return None
        ft = dict(num.hits)
        for k, v in ts.hits.items():
            ft[k] = (ft[k][0] + v[0], ft[k][1]) if k in ft else v
        p["field_type"] = ft
    return p


def _new_hits(after: dict, before, same_lines: bool) -> tuple[dict, int]:
    """(new {line: (increase, context)}, lines with pre-existing). before=None
    → everything is new. Same line count — compared per line (after > before on
    a line = a new defect); different (in theory — a rule with a line break) —
    roughly per file: more in total → everything new, otherwise everything old."""
    if before is None:
        return dict(after), 0
    if same_lines:
        new, pre = {}, 0
        for line, (n, ctx) in after.items():
            b = before.get(line, (0, ""))[0]
            if b:
                pre += 1
            if n > b:
                new[line] = (n - b, ctx)
        return new, pre
    ta = sum(n for n, _c in after.values())
    tb = sum(n for n, _c in before.values())
    if ta > tb:
        return dict(after), 0
    return {}, len(after)


def _new_broken(after: set, before, same_lines: bool) -> tuple[set, set]:
    """(new broken lines, pre-existing broken lines)."""
    if before is None:
        return set(after), set()
    if same_lines:
        return after - before, after & before
    if len(after) > len(before):
        return set(after), set()
    return set(), set(after)


def run(root: str, max_examples: int = 5, baseline=None) -> tuple[bool, dict]:
    """(ok, findings). Only the _BLOCKING keys block; _WARN_KEYS are WARN with
    the file name (pre_existing / unshippable / skipped_large), they do not block.
    baseline is the source dir (see _baseline_dir)."""
    findings = {k: [] for k in _BLOCKING + _WARN_KEYS}
    n_json = 0
    cap = _max_bytes()
    root_abs = os.path.abspath(root)
    run_dir = os.path.dirname(root_abs)
    base_dir = _baseline_dir(root, baseline)
    rev_box = [False]              # lazy: the rename map is needed only on defects
    pre_files = no_base_files = warned = streamed = 0

    def _warn(msg: str) -> None:
        nonlocal warned
        warned += 1
        if warned <= _WARN_FILES:
            print(f"structural: WARN {msg}", flush=True)

    for path, rel in iter_files(root, skip_ext=set()):
        # Walk WITHOUT skip_ext so a forbidden name (chat.db, entity_map.*) does
        # not slip through by extension; the name test itself uses only rel.
        if is_pipeline_artifact(rel):
            findings["forbidden"].append(rel)
            continue
        if is_raw_db(rel):
            findings["unshippable"].append(rel)
            _warn(f"{rel}: raw client database (.db/.sqlite) — content is not "
                  "anonymized, not packed into the release; does not block the round")
            continue
        # Content is gated ONLY for text files: read_text of a binary through
        # surrogateescape blows gigabytes of media up into UCS-4 + a GLUE_RE run over
        # the whole blob (100 MB mp4 → ~900 MB RSS/sec; on GB media — OOM/hours).
        if os.path.splitext(path)[1].lower() in SKIP_EXT:
            continue
        # The size cap BEFORE reading: a whole file in memory ×4 (surrogateescape +
        # split + copies) — a 500 MB json killed the gate with MemoryError/SIGKILL.
        # Above the cap — not a skip but a streaming profile (memory ≈ chunk);
        # a skip happens only above the streaming check limit, and loudly.
        try:
            size = os.stat(path).st_size
        except OSError:
            continue               # vanished between walk and stat — like read_text None
        is_json = path.endswith((".jsonl", ".json"))
        if cap > 0 and size > cap:
            smax = _stream_max()
            if smax > 0 and size > smax:
                findings["skipped_large"].append(f"{rel}: {size} bytes")
                print(f"structural: WARNING {rel}: {size} bytes > {STREAM_MAX_ENV}="
                      f"{smax} — content NOT checked (streaming check limit)",
                      flush=True)
                continue
            after = _profile_stream(path, is_json)
            if after is None:
                continue
            streamed += 1
            print(f"structural: {rel}: {size} bytes > {MAX_BYTES_ENV}={cap} — "
                  "checked as a stream", flush=True)
        else:
            text = read_text(path)
            if text is None:
                continue
            after = _profile(text, is_json)
        n_json += after["n_json"]
        if not (after["glue"] or after["broken"] or after["field_type"]):
            continue               # clean file — leave the source alone
        # before/after: a defect is only what was not in the client's source
        before, same = None, False
        if base_dir is not None:
            before, why = _baseline_profile(base_dir, run_dir, rel, rev_box, cap,
                                            is_json)
            if before is None:
                no_base_files += 1
                _warn(f"{rel}: {why} — before/after comparison unavailable, "
                      "defects count as new")
            else:
                same = before["lines"] == after["lines"]
        else:
            no_base_files += 1
        g_new, g_pre = _new_hits(after["glue"],
                                 before["glue"] if before else None, same)
        f_new, f_pre = _new_hits(after["field_type"],
                                 before["field_type"] if before else None, same)
        b_new, b_pre = _new_broken(after["broken"],
                                   before["broken"] if before else None, same)
        for line, (_n, ctx) in sorted(g_new.items()):
            if len(findings["glue"]) < max_examples:
                findings["glue"].append(f"{rel}:{line}: {ctx}")
        for line in sorted(b_new):
            if len(findings["broken_jsonl"]) < max_examples:
                findings["broken_jsonl"].append(f"{rel}:{line}")
        for line, (_n, ctx) in sorted(f_new.items()):
            if len(findings["field_type"]) < max_examples:
                findings["field_type"].append(f"{rel}:{line}: {ctx}")
        if g_pre or f_pre or b_pre:
            pre_files += 1
            note = f"{rel}: glue={g_pre} field_type={f_pre} broken={len(b_pre)}"
            if len(findings["pre_existing"]) < max_examples:
                findings["pre_existing"].append(note)
            _warn(f"pre-existing {note} (lines) — the client had it before "
                  "anonymization, does not block")
    if warned > _WARN_FILES:
        print(f"structural: WARN … {warned - _WARN_FILES} more WARN lines "
              "hidden (see findings)", flush=True)
    if base_dir is None and no_base_files and \
            os.path.basename(root_abs) == "work":
        print(f"structural: WARN no {os.path.join(run_dir, 'raw')} — "
              f"before/after comparison unavailable, defects in {no_base_files} "
              "files count as new", flush=True)
    ok = not any(findings[k] for k in _BLOCKING)
    findings["_json_checked"] = n_json
    findings["_pre_existing_files"] = pre_files
    findings["_no_baseline_files"] = no_base_files
    findings["_streamed_files"] = streamed
    return ok, findings


def render(ok: bool, findings: dict) -> str:
    lines = [("PASS" if ok else "FAIL") + f"  (json lines checked: {findings.get('_json_checked',0)})"]
    for k in _BLOCKING:
        v = findings.get(k) or []
        lines.append(f"  {k:<14} {len(v)}")
        for e in v[:3]:
            lines.append(f"       {e}")
    for k in _WARN_KEYS:
        v = findings.get(k) or []
        if not v:
            continue
        lines.append(f"  {k:<14} {len(v)}  (WARN, does not block)")
        for e in v[:3]:
            lines.append(f"       {e}")
    if findings.get("_streamed_files"):
        lines.append(f"  {'streamed':<14} {findings['_streamed_files']}  "
                     "(above ANON_MAX_FILE_BYTES — checked as a stream)")
    return "\n".join(lines)
