"""The orchestrator: load enabled detectors, walk a tree, apply rules, report.

Usage:
    from anonymizer.engine.pipeline import Pipeline
    from anonymizer.config.default import CONFIG
    rep = Pipeline(CONFIG).run("/path/to/docs", apply=False)
    print(rep.render(dry=True))
"""
from __future__ import annotations

import csv
import errno
import gc
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from typing import NamedTuple

from .formats import (_mask_json_line, csv_delimiter, csv_record_chunks,
                      mask_csv_cells, mask_values)
from .boundary import split_fusions
from .context import ctx_reach
from .viewmask import apply_view, json_key_edits
from .vnorm import ChainMap, fold_view, homoglyph_view, transport_view
from .report import Report
from .rule import PrefilterState, apply_rules
from .secret import is_secret_value, is_vault_context
from . import placeholders as P
from .walk import (StreamWriter, WriteError, fit_name, iter_files, iter_lines,
                   nul_text_files, read_text, write_bytes, write_text)
from .. import structured as _structured
# Mail (owner's decision item 3): .eml/.mbox letters — by the slots of
# anonymizer/formats/mail.py, assembled in the source encoding (_run_mail)
from .. import formats as _mailfmt

# detector modules in application order (most-specific / self-validating first)
DETECTOR_ORDER = [
    "secrets",     # signatures + label + entropy
    "encoding",    # decode-before-scan (base64/hex/percent)
    "discord_ids", # snowflake user-id + <@mention> (before financial: 17-19
                   # digits inside <@…> must not go to card/account)
    "financial",   # checksummed ids, bank, card, iban
    "contacts",    # phone/email/ip/mac/cookie
    "addresses",   # street/house/flat/city/postcode/coords/dob
    "orgs",        # opf+name / legal-suffix / quoted / domain / brand / crm
    "names",       # dict/morphology/patronymic/initials/diminutive/translit/slug/filename
    "special",     # opt-in: health/ethnicity person↔fact ties
    "linking",     # opt-in: HMAC-pseudonymise snowflake/thread ids, coarsen timestamps
]

# CRM-range company tokens (mint.crm_org): the ONLY 5-digit ORG tokens in v1
# (tag("ORG") mints 1000..9999, crm_org mints ORG_20000..99999, MOD["CRM"]) and
# the ONLY 8-digit ones in v2 (#91: tag("ORG") — 7 digits, crm_org —
# ORG_20000000..99999999, MOD_V2["CRM"]). Both forms — by tenant.yaml mint_version.
CRM_ORG_RE = re.compile(r"\bORG_[2-9](?:\d{4}|\d{7})\b")

# --- parallel pass over files ----------
# A call batch: round dry on 1205 files with a pack of 41k forms ran on ONE
# core of 24 (1 thread, 0 children) for ~40 minutes. Files are independent: token minting
# is a pure hash of the salt (mint.tag), rules have no state between files,
# writing is atomic per file (write_text/StreamWriter). A pool of forked processes
# inherits the already compiled Pipeline (rules + salt), chunks of the tree go
# in FILE ORDER (imap), reports are merged by Report.merge in the same order —
# counters and examples match the sequential run number for number.
# Default workers='auto' = all available cores (the owner: "always at the
# maximum"); workers=1 / no fork (non-POSIX) / few files = the old
# sequential path, byte for byte the same. An error in a worker surfaces
# (fail loudly).
#
# Robustness: multiprocessing.Pool.imap_unordered did NOT survive
# the death of a worker — SIGKILL from the OOM killer on one giant file → Pool
# silently respawns the process, the task is lost, imap_unordered waits for it FOREVER
# (bpo-22393): round dry "running" for days, chain_loop does not fire. Now
# the pool is run_pool on top of concurrent.futures.ProcessPoolExecutor (the same fork
# context, the same COW inheritance): a worker death = BrokenProcessPool →
# the suspects (in flight) are rerun one by one, the culprit = file_failed in
# the report (reason worker_died), the other files are finished on a fresh pool.
# MemoryError on one file — also file_failed (reason memory), not a crash
# of the round. Other exceptions (a broken pack rule) are still loud.
#
# Worker robustness:
#   * gc.freeze before fork (frozen_heap): the cyclic GC in every worker
#     walked ALL inherited objects (10k pack rules, dictionaries, AC
#     automata) and rewrote PyGC_Head → COW pages were copied into every
#     worker (measured by tools/bench_fork_rss.py). The automata are built in
#     the parent BEFORE the fork (prebuild_rules) — otherwise every worker built its own;
#   * worker cap by memory: n = min(cpus, MemAvailable // (64 MB + k ×
#     the largest in-memory file)), k = ANON_WORKER_MEM_K (default 25 —
#     measured, see _WORKER_MEM_K_DEFAULT), only with workers=auto; WARN with
#     the numbers;
#   * a watchdog time budget per file (ANON_FILE_TIMEOUT_S = base +
#     s/MB, SIGALRM in the worker and in the sequential branch) → file_failed
#     (reason timeout), the round lives on; the file on disk is not touched;
#   * faulthandler: SIGUSR1 → stacks to stderr (inherited by workers through
#     fork + again in the pool initializer), ANON_HANG_DUMP_S>0 → a dump to
#     <run>/state/hang.log every N s (install_hang_watch — cli/round).
_FORK_PIPE = None                 # Pipeline inherited by workers through fork
_PARALLEL_MIN_FILES = 8           # below this the pool overhead costs more than the gain
FILE_FAILED = "file_failed"       # pseudo-rule of the report: the file was NOT processed
# Cap on fixpoint passes for the projection of a structured file. In the projection
# thousands of short texts go in a row, and a context rule can drag
# the chain "replacement → context for a neighbour → replacement" longer than 3 passes (measured on
# a real export: up to 7 passes to stabilize per batch) — with cap 3
# the remainder surfaced on a repeated dry and broke the round idempotency oracle.
# The loop exits on the first pass without replacements, so the margin is free.
_STRUCT_FIXPOINT_PASSES = 16
# Ordinary files and stream lines had cap 3, while the chain "replacement → context for
# a neighbour → replacement" needs 4–5 (synthetic 3000 lines: 12209 → 352 → 14 → 2 → 0) — a remainder on
# a repeated dry, BLOCK idempotency and a round rollback. Now the cap is 16 for all paths (the loop exits on
# the first pass without replacements — the margin is free), fixpoint_capped is merged into Report.merge, WARN with
# the file name.
# The check pass of the fixpoint (the one that ends the loop with zero
# replacements) ran all rules over the text that the rules after the last one fired in the previous
# pass had already seen exactly like this and did not fire (rule k of the previous pass saw the text after
# rules 1..k-1; after the last fired j the text did not change any more — rules j+1..M saw
# the result of the pass). Now such a pass goes over rules 1..j, and the rest — only if among them
# at least one fired (the text is different — the rest whole, as before). The output, counters and the view log
# are the same; rule errors (rule_errors) too: a pass that added an error entry does not give
# the next one a tail skip (a repeated error on the same text would be counted the old way once more).


def _fixpoint_passes() -> int:
    return _STRUCT_FIXPOINT_PASSES
# Worker cap by memory: needed = _WORKER_MEM_BASE + k × the largest
# file. k — how many times the worker memory peak on a file exceeds the file. Measured by
# tools/bench_fork_rss.py (macOS, py3.14, 2–5 MB, Cyrillic/emoji/csv, PD
# dense and sparse, a pack of 10k rules and without): the peak of Python objects (tracemalloc)
# ≈ 10–11× REGARDLESS of PD density and the pack — str UCS-2 (Cyrillic ≈ 1×
# UTF-8 bytes; UCS-4 with emoji — 2×), a casefold copy for the AC prefilter,
# pattern.sub (a list of slices + join = 2×) on every rule with a match,
# text/new of two-three fixpoint passes; the RSS peak (what the OOM killer sees)
# ≈ 15–20× per MB (allocator slack: large str are not returned at once) plus
# ~25–30 MB fixed per worker (COW of touched rule pages on
# refcount writes + arenas). Measured by variant (RSS marginal MB/MB):
# Cyrillic 13–16, csv 18.5, UCS-4 (emoji in chats is the norm) 21; 25 — the worst
# with a ~20 % margin; base 64 MB — the fixed part (23–51 MB on a 10k pack)
# with a margin for a pack 4 times larger. To refine on the server:
# PYTHONPATH=. python3 tools/bench_fork_rss.py --only k. 0 = the cap is off.
_WORKER_MEM_K_DEFAULT = 25.0
_WORKER_MEM_BASE = 64 << 20
# Time budget per file. Precedents: e5 — 1 MB \w run ≈ 40 min in
# _EMAIL (a catastrophic regex), e8 — iban 3.6 s on 5000 SKU (legitimate).
# Legitimate engine speed: ~36 s/MB on one core BEFORE the AC prefilter
# (round dry 67 MB / 40 min with a pack of 41k forms), after it — many times faster;
# 120 s/MB = ×3 margin over the worst legitimate case, the base of 600 s covers small
# files under load (24 workers, swap). 1 MB → 12 min (e5 is caught).
_FILE_TIMEOUT_S_DEFAULT = 600.0
_FILE_TIMEOUT_S_PER_MB = 120.0
_HANG_LOG_NAME = "hang.log"       # <run>/state/hang.log — stack dumps
# --- a large structured file — in parts on different workers -----------
# The tg round (26 Telegram exports) was bounded by two files of 95/66 MB: each
# went through one worker, 77 min of computation and as much of writing on 2 cores of 24.
# The line-by-line layout already cuts a file into batches of BATCH_UNITS messages, and
# the rules see only the projection of their batch, so batches are independent
# pool tasks: first a PREPARATION task (structured.plan_parts: byte
# boundaries of batches and isolated values — exactly as the sequential
# pass), from its result — PART tasks (structured.mask_part). The reports
# of parts are merged in batch order, the output — by gluing pieces in
# file order; the output and the dry report are byte for byte as in the sequential pass.
# ANON_SPLIT=0 — turn off (the whole file, as before); ANON_SPLIT_MIN_BYTES
# — the file size threshold (default 1 MB); ANON_PART_BATCHES — batches per part.
_SPLIT_MIN_BYTES_DEFAULT = 1 << 20
_PART_BATCHES_DEFAULT = 1
# a large .jsonl of the stream branch (> stream_threshold) — in parts by lines:
# ANON_STREAM_PART_BYTES — the part size (default 8 MB), a part starts with
# a "warm-up" of ANON_STREAM_WARM_BYTES of lines before it (default 64 KB) — the tail
# of the rule context at the boundary; the parent checks it against the actual tail
# of the previous part and on a mismatch recomputes the part with the right tail
_STREAM_PART_BYTES_DEFAULT = 8 << 20
_STREAM_WARM_BYTES_DEFAULT = 64 << 10
# --- a large CSV/TSV — rules by windows of record batches ------------
# A client batch: four Google Docs batches (CSV 16–35 MB with lists of people)
# did not fit into the file budget (FILE_FAILED timeout, an empty release). The reason is
# not quadratic code, but the order "rule after rule over the whole text": every
# rule whose probe is ANYWHERE in the file scans the WHOLE file (str.find
# of start literals, re.sub), and on a large file almost all have probes: the pack
# echo20 — 84 thousand rules, on 0.35 million characters the prefilter leaves 11.6 thousand, on
# the whole file (20.6 million characters) — 59 thousand; a pass ≈ 59 thousand scans of 20 million
# characters (an estimate of ~1.6 h per pass on one core, 4–6 passes; echo20 cut-downs:
# 1.2 → 2.6 → 5.4 MB = 558 → 1116 → 2727 s, growing faster than the size). On a batch of
# 64 thousand characters hundreds to thousands of rules have probes, a pass is linear in the file size.
# A batch (core) — ≈ _CSV_CHUNK_CHARS characters of whole records (a line break outside
# quotes: a multiline cell is not cut). The rules go over a WINDOW: the core plus
# margins — neighbouring lines of the RAW text not shorter than _CSV_HALO_CHARS (and 8 context
# windows) on each side; only the core goes into the output and the report (the core lines in
# the output — by the count of «\n»: masks do not change it; if they did — _ChunkLines, the file
# goes whole the old way). So a rule at a seam sees its neighbours in the same intermediate
# state of the pass as on the whole text (context on the left, lookahead, a match
# across a line break, chains of passes) — the output matches the whole text as long as
# the influence of replacements does not reach past the margin (context windows are hundreds of characters). The tail of the output view
# of the previous batch (as with .jsonl stream lines) did not work here: on the echo06/echo20 cut-downs
# with batches of 8–16 thousand characters it gave a different rule at the seam (the context word in
# the tail is already covered by a token) — windows with margins on the same cut-downs are byte for byte.
# The core report is _CoreReport (only what starts in the core lines). Column
# passes (column rules, INN/OGRN, entropy secrets, CRM) go over the whole
# text, as before (they need the header; they are linear). A file shorter than _CSV_CHUNK_MIN goes
# whole, the old way (byte for byte).
# A large plain text (logs, .txt, no extension) goes by the same windows: without the column passes,
# batches end at line breaks (line_chunks). Only a flat text whose rules see the text itself
# (_plain_text: .json/.html/.xml views parse the whole file, CSV has its own batches, mail its slots).
# SML: three logs of 21 MB went whole through one worker each — 4 of 10 workers busy for 6+
# minutes at the end of the tree.
_CSV_CHUNK_CHARS = 64 << 10
_CSV_CHUNK_MIN = 1 << 20
_CSV_HALO_CHARS = 4 << 10


def line_chunks(text: str, target: int) -> list:
    """Boundaries of line batches of a plain text — [0, b1, …, len(text)]: a batch is at least
    ``target`` characters (the last one is the rest) and ends right after «\\n»."""
    n = len(text)
    if target <= 0 or n <= target:
        return [0, n]
    bounds, a = [0], 0
    while a + target < n:
        j = text.find("\n", a + target)
        if j < 0 or j + 1 >= n:
            break
        a = j + 1
        bounds.append(a)
    bounds.append(n)
    return bounds


def _is_csv(rel: str) -> bool:
    return rel.lower().endswith((".csv", ".tsv"))


def _plain_text(rel: str) -> bool:
    """A flat text for batch windows by lines: not CSV/TSV (record batches), not a file with a
    transport view or a structural profile (_VIEW_EXT: parsed whole), not a letter (slots)."""
    return (not rel.lower().endswith((".csv", ".tsv") + _VIEW_EXT)
            and _mailfmt.mail_kind(rel) is None)


class _ChunkLines(Exception):
    """The window masks changed the line count (a multiline rule — a PEM block in
    a cell) — the core cannot be cut by lines; the file goes whole the old way."""


def _nth_nl(s: str, k: int) -> int:
    """The position right after the k-th «\n» from the start (k = 0 — the start)."""
    pos = 0
    for _ in range(k):
        pos = s.index("\n", pos) + 1
    return pos


def _core_span(s: str, n0: int, n_r: int, last: bool) -> tuple:
    """[c0, c1) of the window core in the text ``s`` (any version: masks do not change the line count): after
    the ``n0``-th «\n» from the start — up to the end of the text (``last``) or up to the «\n» number ``n_r`` + 1
    from the end inclusive."""
    c0 = _nth_nl(s, n0)
    if last:
        return c0, len(s)
    pos = len(s)
    for _ in range(n_r + 1):
        pos = s.rindex("\n", 0, pos)
    return c0, pos + 1


def _halo_bounds(text: str, a: int, b: int, halo: int) -> tuple:
    """The batch window [a, b): margins — whole lines, not less than ``halo`` characters on each side
    (up to the start/end of the text)."""
    n = len(text)
    l = 0 if a - halo <= 0 else text.rfind("\n", 0, a - halo) + 1
    if b + halo >= n:
        return l, n
    j = text.find("\n", b + halo)
    return l, (n if j < 0 else j + 1)


class _CoreReport(Report):
    """The report of the CORE of a batch window (Report.part: without the context word cap). The window is the core with
    margins of neighbouring lines; the rules go over the whole window, and the report gets only what
    starts in the core lines: match records and rule errors (by the match start), glues (by
    the split position), view counters (by the slice of the core lines). The core lines in any text version —
    by the count of «\n» (_core_span). Every match is recorded by exactly one window — the one in whose core it
    started."""

    def __init__(self, n0: int, n_r: int, last: bool):
        super().__init__()
        self.ctx_cap = None
        self.n0, self.n_r, self.last = n0, n_r, last
        self._span = None              # (text version, c0, c1) — cached per version

    def _core(self, s: str) -> tuple:
        hit = self._span
        if hit is None or hit[0] is not s:
            hit = self._span = (s,) + _core_span(s, self.n0, self.n_r, self.last)
        return hit[1], hit[2]

    def keep_match(self, m) -> bool:
        c0, c1 = self._core(m.string)
        return c0 <= m.start() < c1

    def record(self, rule_name, before, after, match=None, why=None):
        if match is not None and not self.keep_match(match):
            return
        super().record(rule_name, before, after, match, why)

    def count_in_core(self, s: str, positions) -> int:
        c0, c1 = self._core(s)
        return sum(1 for p in positions if c0 <= p < c1)

    def core_slice(self, s: str) -> str:
        c0, c1 = _core_span(s, self.n0, self.n_r, self.last)
        return s[c0:c1]

    def done(self, base):
        """The window is done: drop the version cache (the report goes to the parent), the spans of rule errors go into
        the coordinates of the file text (the window start ``base``)."""
        self._span = None
        if base:
            for rec in self.rule_errors:
                a, b = rec["span"]
                rec["span"] = (a + base, b + base)
        return self


# registry preparation (minting canonical names with one chain call)
# by the pool in groups: gluing 2 729 tg names took 205 s (superlinear rules
# on a long text), in groups of 500 — 60 s sequentially; below the threshold
# of names — one call, as before
_NAMES_PAR_MIN = 200
_NAMES_GROUP = 100


def _rule_max_errors() -> int:
    """ANON_RULE_MAX_ERRORS (rule._max_errors), read in the PARENT.

    A part of a file counts the threshold over ITS OWN error list, the sequential
    pass — over the list of the file report (direct chain calls: single slots,
    rollback of isolated values, stream lines) and over the list of the call (a batch).
    The count of a part at any point ≤ the count of the file at the same point, so a part
    that failed on the threshold = the file would fail sequentially too; if none failed and
    the total errors ≤ the threshold — the sequential one did not fail either. What remains is "the total
    is above the threshold" — the file is redone whole the old way (exactly)."""
    from .rule import _max_errors
    return _max_errors()


def capture_key(rel: str) -> str:
    """The file name in the computation store — sha256 of relpath."""
    import hashlib
    return hashlib.sha256(rel.encode("utf-8", "surrogateescape")).hexdigest()


def capture_path(capture: str, rel: str) -> str:
    """The output path of file ``rel`` in the computation store: the name is sha256 of
    relpath (stable, without directories and exotic names)."""
    return os.path.join(capture, capture_key(rel))


# Environment switches that change the engine RESULT (the outcome
# FILE_FAILED, the processing path, detector decisions) — their values are stored in
# the metadata of store units; other values in the next computation →
# the increment is forbidden (a full computation). Scheduling and diagnostic switches (the number
# of workers, splitting into parts, dumps) do not affect the output — ENV_SCHED. A new
# ANON_* engine switch must go into one of the lists (test
# test_incremental.test_env_knobs_listed).
ENV_KNOBS = ("ANON_RULE_MAX_ERRORS", "ANON_FILE_TIMEOUT_S",
             "ANON_STRUCTURED_PROFILES", "ANON_PHONENUMBERS",
             "ANON_ID_CTX_GUARD", "ANON_MAX_FILE_BYTES",
             "ANON_MAX_STRUCT_BYTES", "ANON_PREFILTER", "ANON_PREFILTER_INCR",
             "ANON_PREFILTER_NEEDS", "ANON_PREFILTER_VERIFY",
             "ANON_AC_BACKEND",
             # the Russian trace dictionary changes the round rules
             "ANON_RU_TRACE", "ANON_RU_TRACE_CURRENCY", "ANON_RU_TRACE_LAWS",
             # dictionary entries turned off by the group owner's keep (group-keep) — change the round rules
             "ANON_RU_TRACE_DISABLED",
             # (the rollback switch "NUL in the head — silence" is removed —
             # the ENV_KNOBS set changed → a full increment computation once)
             # .pdf outside SKIP_EXT (the document view in work/ is read as text) — the layer
             # is active when ANON_PDF_PY is set (walk.pdf_inplace_active): it changes the walk set
             # (the config fingerprint must see a change of ANON_PDF_PY; the layer rollback switch is removed
             # — the ENV_KNOBS set changed, a full increment computation once)
             "ANON_PDF_PY")
# The rollback switches of detector and format rules
# were removed together with branch 0 (PASSPORT_LABELED, JSONL_SPANS, INN_ID_COLUMNS, TG_PEER_IDS,
# EMAIL_WIDE, IP_SURNAME, PAIR_DOUBLE_SURNAME, PHONE_PRETTY, VALUE_LINE_BREAKS, DOB_WIDE, PCT_URL,
# B64_PII, HANDLE_LONG, POSTCODE_TOKEN_TAIL, CSV_KEEP_EOL, CLIENT_TAIL, LONE_PW_NAMES,
# FIXPOINT_CAP16, RENAME_SUFFIX, CSV_CELL_SPANS, PACK_ORG_FIRST, FIXPOINT_TAIL_SKIP, VALUE_ONE_LINE,
# EMAIL_PUNYCODE, DOUBLE_SURNAME_SEED, REPORT_STATE_FIXPOINT, PHONE_TOKEN_GUARD, NAME_NBSP,
# ISSUE_KEYS, BARE_CC_DOMAINS, COMPOUND_LOGIN, VALUE_JSON_SAFE): the ENV_KNOBS set
# changed → the first round after the rollout computes the increment anew (a full computation, once).
# Second implementations and view layers were removed — OWN_TOKEN_STRICT,
# RU_TRACE_COLLOQ, PROBE_CLASS_FOLD, CTX_EDGE_LOOKBEHIND, ENGINE_HOMOGLYPH,
# HOMOGLYPH_AMBIGUOUS_STAGE2, TRANSPORT_VIEW, TRANSPORT_ENTITIES, ENGINE_FOLD_VIEW, MAIL_VIEW,
# JSONL_RAW_GUARD (and in ENV_SCHED — LOCAL_SUB, PROBE_CLASS_MERGE, HOMOGLYPH_LEAN,
# LOCAL_SUB_ANCHOR): the ENV_KNOBS set changed once more → one more full increment computation.
# Raw guard (production: two JSONL lines broken by the raw pass → deglue 0 →
# quarantine rc 9, a retry fails the same way). The safeguard: a JSONL line valid before the raw pass and
# invalid after it is repeated without the replacements that break it (the search — by one of the fired
# rules, then minimization of the set, then for a rule with several replacements on the line — only
# the breaking replacements: apply_rules(skip_match=…)), the other replacements stay; every removed replacement —
# Report.raw_guard (rule, file, line), the rule entries are subtracted from counts (the
# idempotency oracle, a repeated dry, will remove it again → 0), a WARN line to the output (_say_guard: one per
# line and rule; a stream in parts — at assembly). json.loads — only on lines that the raw
# pass changed; the cost is linear in the guarded lines. Stream and memory share one code
# (_raw_guard) with the same context tail (the output view before the safeguard). The rollback
# JSONL_RAW_GUARD=0 (a broken line went to the structural gate) is removed; line repair tests
# (tests/test_structural_line_repair.py) turn the safeguard off by substituting _raw_guard_text/_raw_guard.
_RAW_GUARD_TIER = "raw-guard-off"       # a stub tier: a rule from skip is not applied
_RAW_GUARD_EXT = (".jsonl", ".ndjson", ".json")


def _json_ok(s: str) -> bool:
    """The line is valid JSON (as _parses of the structural gate: a BOM in the head is not a breakage)."""
    try:
        json.loads(s.lstrip("﻿"))
        return True
    except Exception:  # noqa: BLE001 — any parse error
        return False
# Homoglyphs (owner's decision item 4): Latin lookalikes in a Cyrillic word (and vice versa) are folded
# only in the view for the rules; unmasked text in the output does not change.
# A two-sided token — lookalikes, ALL
# letters of both alphabets — with a Latin head (Latin, then Cyrillic: «comе» in «example.comе»,
# «xyом», the disputable «Coк») is folded into Cyrillic by the first stage of the homoglyph layer (the pair
# «Сoк Иван» matches, «Пeтров» is covered by a rule — owner's decision item 4); if it stayed unmasked —
# an additional stage by the raw view (vnorm.homoglyph_view(raw_ambiguous=True)): the rules see
# it as written, and a Latin pack rule with a Cyrillic tail «[а-яё]*» takes it, as before the homoglyph layer.
# A two-sided token with a Cyrillic head («Сoк», «тaк») is not a form of that rule and does not trigger the stage
# (there is no extra pass of rules over the file).
# Decisions on the stages of the homoglyph layer (the raw view stage, stage 2) — by the count of view tokens
# of the OUTPUT that apply_view builds in refold anyway, not by a second build of the file view and a second
# full pass of homoglyph_view over the view: the number of two-sided tokens with a Latin head and of Latin
# words with Cyrillic lookalikes (vnorm._fold_mixed, ``rev``) is counted in the same pass. No
# refold (the view did not change or the rules replaced nothing) — the input count; the output changed without refold
# (the view equals the text) — a pass over the output view. The stages and their order are the same, the output is byte for
# byte (tests/test_engine_speed.py).
# Between the transport view and the homoglyph layer — the layer
# vnorm.fold_view: invisible characters (ZWSP, soft hyphen, ZWJ/ZWNJ, U+2060, BOM) are removed, a letter +
# a combining mark are composed (NFD text → NFC), an accent over Cyrillic is removed; the PosMap map
# is composed with the transport one (vnorm.ChainMap), the output keeps the source form — invisible characters and accents
# outside replacements stay. The checks (control, residual, archive) fold the same (VIEW_FOLD),
# so «Рома́шка», NFD and «Васи​лёк» are a hit at the input and silence at the output.
# The view layer switches (ENGINE_HOMOGLYPH, HOMOGLYPH_AMBIGUOUS_STAGE2,
# HOMOGLYPH_LEAN, ENGINE_FOLD_VIEW, TRANSPORT_VIEW, TRANSPORT_ENTITIES,
# MAIL_VIEW) are removed — all layers are always on (ViewModes.ALL).


class ViewModes(NamedTuple):
    """View modes for the rules — all on
    (ALL); the field order = the layer order: transport view → fold_view → homoglyphs → the stage
    by the raw view for a two-sided token with a Latin head. The position map is a composition over
    the layers with a map (transport, fold_view; homoglyphs are 1:1). ``homoglyph_defer`` matters only with
    ``homoglyph``. In the engine the fields are read only by name; raw tuples of the former branches are accepted
    by one place — ``ViewModes.of``."""
    transport: bool
    fold_view: bool
    homoglyph: bool
    homoglyph_defer: bool

    @classmethod
    def of(cls, modes) -> "ViewModes":
        """ViewModes — as is; a pair (transport, homoglyphs) and a triple (transport,
        fold_view, homoglyphs) (test calls) — without the missing layers (False)."""
        if isinstance(modes, cls):
            return modes
        if len(modes) == 2:
            t_on, h_on = modes
            return cls(t_on, False, h_on, False)
        if len(modes) == 3:
            t_on, f_on, h_on = modes
            return cls(t_on, f_on, h_on, False)
        return cls(*modes)


# Engine modes — all layers (the layer switches are removed); transport is turned off only by the structured
# profile and the value pass (json.loads has already decoded)
ViewModes.ALL = ViewModes(True, True, True, True)

# Transport view (owner's decisions items 1 and 8): .json/.jsonl/.ndjson with \uXXXX and \/, .html/.htm/.xml with &#NNNN;,
# &#xHHHH;, &nbsp; — rules over the decoded view, substitution by the map into the source form
# (the ensure_ascii form stays). The homoglyph layer is built on top of the transport view.
# Memory: the view is a second copy of the text plus the PosMap map (arrays, a run of escapes
# of one width — one segment: for ensure_ascii O(words)); the rules, the log and refold — by the view;
# stage 2 of the homoglyph layer without reverse lookalikes does not build the transport view anew. The worker
# cap counts a JSON/HTML file with the _VIEW_MEM_FACTOR multiplier to k (see _cap_size): measured by
# tools/bench_transport_view.py on a 5 MB ensure_ascii JSONL (macOS, nice 15) — the tracemalloc peak
# ×1.6 and maxrss ×1.3 of the old path (before the fix ×12 and ×7: a map of 110 MB of lists and int on a 5 MB file);
# the multiplier 3 — a ×2 margin over the measurement; tests/test_transport_view_engine.py::test_view_memory_bounded
# keeps it not below the measurement.
_VIEW_EXT = (".json", ".jsonl", ".ndjson", ".html", ".htm", ".xml")
_VIEW_MEM_FACTOR = 3.0
# Mail: a letter holds raw bytes, slot texts (decoded: base64 ×3/4, headers
# are shorter), the slot view for the rules and the assembled output — measured ×3.7 of the file size; the multiplier 5 —
# a margin for another allocator/Python version (with 4 the margin was 7 %).
# tests/test_mail_connect.py::test_memory_bounded keeps the multiplier not below the measurement
_MAIL_MEM_FACTOR = 5.0


def _cap_size(size: int, rel: str) -> int:
    """File size for the worker memory cap: a file with a transport view (JSON/HTML)
    — with the _VIEW_MEM_FACTOR multiplier; a letter — with _MAIL_MEM_FACTOR; others — as is."""
    if rel.lower().endswith(_VIEW_EXT):
        return int(size * _VIEW_MEM_FACTOR)
    if _mailfmt.mail_kind(rel) is not None:
        return int(size * _MAIL_MEM_FACTOR)
    return size
ENV_SCHED = ("ANON_PIPE_WORKERS", "ANON_WORKER_MEM_K", "ANON_HANG_DUMP_S",
             "ANON_SPLIT", "ANON_SPLIT_MIN_BYTES", "ANON_PART_BATCHES",
             "ANON_STREAM_PART_BYTES", "ANON_STREAM_WARM_BYTES")


# International phone coverage depends on the phonenumbers package
# (requirements-optional): without it phone_intl_valid is silent, and «+44 20 7946 0958» goes out
# open. The presence of the package is in the switch snapshot (runs with different coverage are not mixed by the increment)
# and a WARN at engine start (Pipeline.__init__).
ENV_PHONENUMBERS_PKG = "#phonenumbers"
_PN_WARNED = False


def _env_snapshot() -> dict:
    from . import validators as _V
    snap = {k: os.environ.get(k) for k in ENV_KNOBS}
    snap[ENV_PHONENUMBERS_PKG] = "1" if _V.phone_intl_available() else "0"
    return snap


def _warn_phonenumbers(config) -> None:
    """WARN if the contacts detector is on but the phonenumbers package is missing (an explicit
    ANON_PHONENUMBERS=0 is the operator's choice, no WARN)."""
    global _PN_WARNED
    from . import validators as _V
    if _PN_WARNED or not config.get("detectors", {}).get("contacts", True) \
            or os.environ.get("ANON_PHONENUMBERS", "1") == "0" or _V.phone_intl_available():
        return
    _PN_WARNED = True                    # once per process
    print("pipeline: WARN the phonenumbers package is not installed — international numbers (except +7 and "
          "CIS) are not masked (phone_intl_valid is silent); pip install phonenumbers "
          "(requirements-optional.txt), ANON_PHONENUMBERS=0 silences this warning",
          flush=True)


def _write_piece(pieces_dir: str, text: str) -> tuple:
    """The output of a changed part goes to a temporary file in
    the run pieces directory (Pipeline._pieces_dir), the parent gets a path, not
    the text: the parent memory is O(1) per part, not O(file output) until the end
    of the pool. Assembly (_write_parts) reads and deletes the piece; the whole directory
    (with pieces of tasks replayed by the pool after a worker death) is deleted by
    Pipeline.run."""
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".anontmp", dir=pieces_dir)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(text.encode("utf-8", "surrogateescape"))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return ("file", tmp)


def _unit_names(files_meta: dict) -> list:
    """Names of all metadata units of the store (files, parts, iso)."""
    names = []
    for m in files_meta.values():
        if m.get("kind") == "parts":
            names += [pm["unit"] for pm in m["parts"]] + [m["iso"]]
        else:
            names.append(m["unit"])
    return names


@contextmanager
def _old_store(what: str):
    """Reading/copying the previous store after the sha check — a failure
    (the file was replaced/deleted during the run, a disk failure) = a clear
    SystemExit, not a traceback."""
    from . import units as _units
    try:
        yield
    except WriteError:
        raise                  # writing the destination (_write_parts) — a file failure
    except _units.READ_ERRORS as e:
        raise SystemExit(
            f"pipeline: {what} of the previous store — read/copy failure "
            f"after the sha check ({type(e).__name__}: {e}): the store was changed "
            "during the run or the disk failed; repeat the computation "
            "(ANON_ROUND_INCREMENTAL=0 — full)"
        ) from None


def _unit_reusable(m: dict, kind: str, dirty: set, max_err: int) -> bool:
    """A whole/stream file of the previous computation is usable as is: the same view
    of the unit, it has no trace of the struck-out rules, and its rule errors are not
    above the current threshold (otherwise the sequential pass could fail —
    ANON_RULE_MAX_ERRORS: n > threshold; ≤ 0 — any error is fatal)."""
    return (m.get("kind") == kind and kind != "parts"
            and not set(m["fired"]) & dirty
            and m.get("n_errors", 0) <= max(max_err, 0))


def _removed_rules(old: list, new: list):
    """Names of rules of the old pack that are missing from the new one, if the new one is
    exactly the old one with STRUCK-OUT rules (the same fingerprints in the same
    relative order: a rule sees the text after the previous ones, the order is
    part of the behaviour); otherwise None (added, changed, reordered or
    no fingerprint). Greedy subsequence matching is correct:
    by the name of a struck-out rule only the "dirtiness" of a unit is decided."""
    if any(d is None for _n, d in old) or any(d is None for _n, d in new):
        return None
    j, removed = 0, []
    for name, dg in old:
        if j < len(new) and new[j][1] == dg:
            j += 1
        else:
            removed.append(name)
    return removed if j == len(new) else None


from .pool import (available_cpus, pdeathsig_kill, register_sigusr1, run_pool,  # noqa: F401
                   _PR_SET_PDEATHSIG, _feed_new, _pool_init, _run_batch)


def resolve_workers(value) -> int:
    """Default 'auto' = ALL available cores (owner's decision 2026-09-10:
    "always at the maximum"); int → as is (floor 1; 1 = sequential);
    garbage = SystemExit (we do not guess silently)."""
    if value in (None, "", "auto"):
        return available_cpus()
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise SystemExit(f"pipeline: workers={value!r} — int or 'auto' is expected")
    return max(1, n)


from .rule import RuleErrorsExceeded            # ANON_RULE_MAX_ERRORS (2026-09-22)


def _split_eol(raw: str) -> tuple:
    """(body, line break) of a stream branch line: the EOL is split off BEFORE the rules —
    the in-memory branch works over text.split("\\n"), the rules must not see
    the end of a line. One helper for _run_stream and _spart_task."""
    for cand in ("\r\n", "\n", "\r"):
        if raw.endswith(cand):
            return raw[: -len(cand)], cand
    return raw, ""


def _file_failure(exc: BaseException) -> str | None:
    """The FILE_FAILED reason for an exception of ONE file or None (= systemic,
    goes out): MemoryError → memory (a giant csv whole in memory),
    RuleErrorsExceeded → rule_errors (the file exceeded ANON_RULE_MAX_ERRORS),
    walk.WriteError → write_error:<errno> (tmp/write/os.replace of this file:
    the name, permissions; the file on disk is not touched, tmp is removed). Anything else — a broken
    pack rule, a full disk etc. — is still loud."""
    if isinstance(exc, MemoryError):
        gc.collect()
        return "memory"
    if isinstance(exc, RuleErrorsExceeded):
        return "rule_errors"
    if isinstance(exc, FileTimeout):
        return "timeout"
    if isinstance(exc, WriteError):
        return "write_error:" + errno.errorcode.get(exc.errno, str(exc.errno))
    return None


class FileTimeout(BaseException):
    """SIGALRM expired — the file did not fit into ANON_FILE_TIMEOUT_S.
    BaseException: through any except Exception inside rules/formats
    (one-match robustness catches Exception on a match — the watchdog must pass through)."""


# failures of ONE file → FILE_FAILED, not a round crash (see _file_failure)
_FILE_FAILURES = (MemoryError, RuleErrorsExceeded, FileTimeout, WriteError)


def _fork_chunk(task):
    """The pool worker body. The task kind is the first element of the tuple:

      ("file", idx, path, rel) → ("file", idx, file Report, None) or
          ("file", idx, Report with rule_errors, reason) — a failure of ONE file
          from _file_failure does not fail the round: the file goes into the report as
          FILE_FAILED (see _run_parallel). Other exceptions (a broken
          pack rule — a systemic problem) go to the parent and surface
          as before. One file — one task (a test run
          2026-09-10: chunks of ~12 files left a tail of 5 workers on
          large transcripts with 19 idle). A file runs under
          the watchdog timer file_deadline (SIGALRM, the main thread of
          the worker) — did not fit the budget → FILE_FAILED(timeout), the worker
          is alive, the file on disk is not touched (the in-memory branch writes at the very
          end, the stream branch aborts tmp);
      ("prep", …) / ("part", …) — a large structured file in parts
          (Pipeline._prep_task / _part_task);
      ("names", …) — a group of names of registry preparation
          (Pipeline._names_task);
      ("cprep", …) / ("cpart", …) — a large CSV/TSV in batches of records
          (Pipeline._cprep_task / _cpart_task)."""
    kind = task[0]
    if kind == "prep":
        return _FORK_PIPE._prep_task(task)
    if kind == "part":
        return _FORK_PIPE._part_task(task)
    if kind == "spart":
        return _FORK_PIPE._spart_task(task)
    if kind == "cprep":
        return _FORK_PIPE._cprep_task(task)
    if kind == "cpart":
        return _FORK_PIPE._cpart_task(task)
    if kind == "names":
        return _FORK_PIPE._names_task(task)
    _kind, idx, path, rel = task
    sub = Report()
    try:
        with file_deadline(file_deadline_for(path, _FORK_PIPE._file_timeout)):
            _FORK_PIPE._run_file(path, rel, sub, _FORK_PIPE._fork_apply)
    except _FILE_FAILURES as exc:
        # rule_errors entries of the failed file (which rule, where)
        # go to the parent — otherwise rounds/rN/rule_errors.json would not see exactly
        # the errors that made the file failed; partial counters — do not
        # (the file is not written)
        errs = Report()
        errs.rule_errors = list(sub.rule_errors)
        return "file", idx, errs, _file_failure(exc)
    if _FORK_PIPE._capture is not None:          # the unit goes into the store
        _FORK_PIPE._write_unit(rel, "w", sub)
    return "file", idx, sub, None


def _unreadable_reason(path: str) -> str:
    """The reason why read_text returned None: an open/read error (unreadable:<error>),
    as in walk.read_text_safe; the file is readable — the failure was one-off (a race) or in decoding."""
    try:
        with open(path, "rb") as fh:
            fh.read(1)
    except OSError as exc:
        return f"unreadable:{exc.strerror or exc}"
    return "unreadable:reading failed (a one-off failure or decoding)"


def _record_failed(report: Report, rel: str, reason: str) -> None:
    """The file was NOT processed (the worker was killed by the OOM killer / MemoryError / the limit
    of rule errors): not silently — a WARN with the name + the pseudo-rule FILE_FAILED in
    the report (the counter and an example with relpath go into dry_report.json and the
    dry↔apply check), so that downstream can block EXACTLY this file, not
    the whole delivery."""
    print(f"pipeline: WARN file {rel!r} was NOT processed ({reason}) — skipped, "
          f"in the report {FILE_FAILED}", flush=True)
    report.record(FILE_FAILED, rel, reason)


def _note_csv_error(report: Report, rel: str, rule_id: str, e: Exception,
                    what: str = "") -> None:
    """A failure of the value pass over CSV/TSV on ONE file: csv.Error (a cell larger than
    the limit, CR in an unquoted field), formats.CsvStructureError (the dialect /
    columns did not match the header), an exception of value_fn of a column rule —
    WARN + an entry {rule_id, file, span, exc, msg} in report.rule_errors
    (rounds/rN/rule_errors.json), the pass is skipped, the file and the round live on.
    Before, csv.Error was swallowed in formats.mask_csv_cells —
    the column and entropy passes were lost for the whole file silently."""
    exc = "csv.Error" if type(e) is csv.Error else type(e).__name__
    rec = {"rule_id": rule_id, "file": rel, "span": (0, 0),
           "exc": exc, "msg": str(e)[:200]}
    report.rule_errors.append(rec)
    print(f"pipeline: WARN {what or rule_id} file {rel!r} — {exc}: "
          f"{rec['msg']}; skipped", flush=True)


# --- memory / time / worker diagnostics ------------------------------

def env_number(name: str, default, cast, what: str, label: str = "pipeline"):
    """An ANON_* number from env: empty → default; garbage → SystemExit (we do
    not guess silently). Read in the PARENT before the fork: SystemExit inside a worker
    = a worker death = BrokenProcessPool."""
    raw = os.environ.get(name, "")
    if raw.strip() == "":
        return default
    try:
        val = cast(raw)
        if isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))):
            raise ValueError("not a finite number")   # inf/nan: int(inf), setitimer(inf) fail
    except (TypeError, ValueError):
        raise SystemExit(f"{label}: {name}={raw!r} — {what} is expected")
    return val


def _mem_available_linux() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _mem_available_darwin() -> int | None:
    """macOS: vm_stat → (free + inactive + speculative + purgeable) × page —
    approximation of MemAvailable (the kernel gives inactive/purgeable away under pressure,
    like reclaimable cache in Linux). No vm_stat / could not parse → None."""
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    m = re.search(r"page size of (\d+) bytes", out)
    page = int(m.group(1)) if m else 4096
    pages = 0
    seen = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative",
                "Pages purgeable"):
        m = re.search(rf"^{key}:\s+(\d+)\.", out, re.M)
        if m:
            pages += int(m.group(1))
            seen += 1
    return pages * page if seen else None


def _mem_available_psutil() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except (OSError, ValueError, AttributeError):
        return None


def _mem_available_sysconf() -> int | None:
    """POSIX without /proc: SC_AVPHYS_PAGES × SC_PAGE_SIZE (free pages — rougher than
    MemAvailable, but better than nothing)."""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    if pages <= 0 or page <= 0:
        return None
    return pages * page


def mem_available() -> int | None:
    """Available memory (bytes): Linux — MemAvailable from /proc/meminfo,
    macOS — vm_stat (see _mem_available_darwin), otherwise psutil (if installed) or
    os.sysconf; nothing worked — None = "we do not know, we do not limit".
    A shared helper for pipeline and tools/ner_local."""
    got = None
    if os.path.exists("/proc/meminfo"):
        got = _mem_available_linux()
    elif sys.platform == "darwin":
        got = _mem_available_darwin()
    if got is None:
        got = _mem_available_psutil()
    if got is None:
        got = _mem_available_sysconf()
    return got


def fork_context():
    """The multiprocessing "fork" context or None where there is no fork (Windows): the engine pool
    inherits Pipeline through fork (copy-on-write), spawn does not replace it — the caller takes
    the single-process path, the output is byte for byte the same."""
    import multiprocessing as mp
    if not hasattr(os, "fork"):
        return None
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


def worker_mem_k() -> float:
    """ANON_WORKER_MEM_K → k (worker working set / file size);
    ≤ 0 = the cap is off."""
    return env_number("ANON_WORKER_MEM_K", _WORKER_MEM_K_DEFAULT, float,
                      "a number (0 = no memory cap)")


def cap_workers_by_memory(n: int, largest: int, label: str = "pipeline",
                          avail: int | None = None, k: float | None = None,
                          base: int = _WORKER_MEM_BASE) -> int:
    """n → min(n, MemAvailable // (base + k × largest)), not less than 1;
    largest — the largest file that a worker holds in memory whole
    (stream files do not count: O(line)). The cap fired → a WARN with the numbers.
    avail None = memory is unknown (not Linux/macOS) → no cap; k ≤ 0 →
    no cap."""
    if n <= 1 or largest <= 0:
        return n
    if k is None:
        k = worker_mem_k()
    if k <= 0:
        return n
    if avail is None:
        avail = mem_available()
    if avail is None:
        return n
    need = int(base + k * largest)
    cap = max(1, avail // max(1, need))
    if cap < n:
        print(f"{label}: WARN {cap} workers instead of {n} — available "
              f"{avail / 1e6:.0f} MB, a worker needs {need / 1e6:.0f} MB "
              f"({base / 1e6:.0f} MB + k={k:g} × the largest file "
              f"{largest / 1e6:.1f} MB; ANON_WORKER_MEM_K, 0 = no cap)",
              flush=True)
        return cap
    return n


def file_timeout_base() -> float:
    """ANON_FILE_TIMEOUT_S → the base budget per file, s (0 = off)."""
    return max(0.0, env_number("ANON_FILE_TIMEOUT_S", _FILE_TIMEOUT_S_DEFAULT,
                               float, "a number of seconds (0 = no timeout)"))


def file_deadline_for(path: str, base: float) -> float:
    """The file budget: base + _FILE_TIMEOUT_S_PER_MB × size (MB); base ≤ 0
    = 0 (no timer). The stream branch (.jsonl above the threshold) runs under the same
    timer — the budget grows with the size, it has plenty of margin."""
    if base <= 0:
        return 0.0
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return base + _FILE_TIMEOUT_S_PER_MB * size / 1e6


def _on_alarm(_signum, _frame):
    raise FileTimeout()


@contextmanager
def file_deadline(seconds: float):
    """ITIMER_REAL for seconds inside the block; expired → FileTimeout. Only
    the main thread of the process (a signal limitation) and only where setitimer exists
    (POSIX); seconds ≤ 0 = no timer. The previous handler is restored.
    The scheme — tools/ner_local._deadline."""
    import signal
    import threading
    if seconds <= 0 or not hasattr(signal, "setitimer") \
            or threading.current_thread() is not threading.main_thread():
        yield
        return
    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


_PARSE_BATCH = 1000         # rule patterns per pool task of _parse_ahead
_PARSE_MIN = 2000           # fewer patterns to parse — parsed here: a pool costs more
_PARSE_RULES: list | None = None


def _parse_task(span):
    from .rule import _analyze
    a, b = span
    return [((r.pattern.pattern, r.pattern.flags), _analyze(r.pattern)) for r in _PARSE_RULES[a:b]]


def _parse_ahead(rules, workers: int) -> None:
    """The sre parse of the rule patterns (rule._analyze: probes, start and anchor literals) in up to
    ``workers`` forked processes, the results go into the parse cache of this process, where rule_probes
    and rule_starts below find them. The parse is a pure function of (pattern, flags): the cache holds
    the same data as a parse here. A task whose worker died — parsed here later, on demand."""
    global _PARSE_RULES
    from .rule import _PARSE_CACHE
    todo, seen = [], set()
    for r in rules:
        p = getattr(r, "pattern", None)
        if not hasattr(p, "pattern") or hasattr(r, "_probes"):
            continue
        key = (p.pattern, p.flags)
        if key not in _PARSE_CACHE and key not in seen:
            seen.add(key)
            todo.append(r)
    ctx = fork_context() if workers > 1 and len(todo) >= _PARSE_MIN else None
    if ctx is None:
        return
    spans = [(i, min(i + _PARSE_BATCH, len(todo))) for i in range(0, len(todo), _PARSE_BATCH)]
    _PARSE_RULES = todo
    try:
        done, _dead = run_pool(_parse_task, spans, min(workers, len(spans)), ctx,
                               label="pipeline:parse", name=lambda sp: f"rules {sp[0]}-{sp[1]}")
    finally:
        _PARSE_RULES = None
    for i in range(len(spans)):
        for key, hit in done.get(i, ()):
            _PARSE_CACHE.setdefault(key, hit)


def prebuild_rules(rules, workers: int = 1) -> None:
    """Build in the PARENT everything that workers would otherwise build each on their own
    after fork: the prefilter AC automata (rule._rule_automata, cached by
    list identity) and rule probes. ``workers`` > 1 — the patterns are parsed in that many forked
    processes first (``_parse_ahead``)."""
    from .rule import _IDX_MIN, _PREFILTER_ON, _probe_index, _rule_automata
    _parse_ahead(rules, workers)
    if _PREFILTER_ON:
        _rule_automata(rules)
        if len(rules) >= _IDX_MIN:
            _probe_index(rules)          # the index of pass candidates
    # case folding tables and start literal automata (point
    # sub) — also in the parent, otherwise every worker would build them itself after fork
    from .rule import (_FIND_MAX, _fold_tables, _starts_automaton,
                       rule_starts)
    _fold_tables()
    for r in rules:
        if r.kind in ("regex", "token_remap"):
            st = rule_starts(r)
            if st is not None and len(st) > _FIND_MAX:
                _starts_automaton(r)


@contextmanager
def frozen_heap():
    """gc.collect + gc.freeze for the time of the forked process pool: the heap
    of the parent goes to the permanent generation, the cyclic GC of the workers does not
    walk it and does not rewrite PyGC_Head of inherited objects — the COW
    pages of rules/dictionaries/automata stay shared. After the pool —
    gc.unfreeze (the parent lives on as usual)."""
    gc.collect()
    frozen = hasattr(gc, "freeze")
    if frozen:
        gc.freeze()
    try:
        yield
    finally:
        if frozen:
            gc.unfreeze()


def hang_dump_s() -> float:
    """ANON_HANG_DUMP_S → the period of stack dumps to <run>/state/hang.log, s
    (0 = off, the default)."""
    return max(0.0, env_number("ANON_HANG_DUMP_S", 0.0, float,
                               "a number of seconds (0 = off)"))


_HANG = {"fh": None, "run": None, "stop": None}   # hang.log of the current stage + the thread stop event


def _hang_loop(fh, secs: float, stop, label: str) -> None:
    """The periodic dump thread: every secs — the stacks of all threads to
    hang.log through faulthandler.dump_traceback UNDER the GIL (other threads stand
    between bytecodes — the frames are consistent). NOT dump_traceback_later: its
    C watchdog reads the frames of other threads without the GIL and on Linux/3.11 with a pool and
    signals crashed the stage with SIGSEGV (server 2026-09-29). The cost: while the GIL is held by
    C code (a regex in re), the dump waits — for such a hang there is kill -USR1
    (the signal handler works inside C too) and ANON_FILE_TIMEOUT_S."""
    import faulthandler
    n = 0
    while not stop.wait(secs):
        n += 1
        try:
            fh.write(f"--- {label}: dump {n} (every {secs:g} s) ---\n")
            fh.flush()
            faulthandler.dump_traceback(fh, all_threads=True)
            fh.flush()
        except (OSError, ValueError):
            return                        # the file was closed on re-arm / exit


def install_hang_watch(run_dir: str | None = None, label: str = "stage"):
    """Diagnostics of a hung stage without root. Always — SIGUSR1 → stacks to
    stderr (`kill -USR1 <pid>`; pool workers too). With ANON_HANG_DUMP_S > 0
    and a known run_dir with state/ — the daemon thread _hang_loop writes the stacks of all
    threads to <run>/state/hang.log every N s (a header with pid/label/time;
    the dumps are numbered, without a clock). A repeated call on the same run_dir
    (cli.main → round.run_round) does not re-arm; a new run_dir — the previous
    thread stops, the file is closed. The thread does not survive fork —
    there is none in pool workers, nothing to clean. Returns: the hang.log path or None."""
    register_sigusr1()
    secs = hang_dump_s()
    if secs <= 0 or not run_dir:
        return None
    if _HANG["run"] == run_dir and _HANG["fh"] is not None:
        return _HANG["fh"].name
    state = os.path.join(run_dir, "state")
    if not os.path.isdir(state):
        return None                       # no workspace yet (init/a garbage --run)
    path = os.path.join(state, _HANG_LOG_NAME)
    try:
        fh = open(path, "a", encoding="utf-8")
    except OSError as exc:
        print(f"{label}: WARN hang.log was not opened ({exc}) — the periodic dump "
              "of stacks is off", file=sys.stderr, flush=True)
        return None
    import threading
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fh.write(f"=== {label} pid {os.getpid()} {now}: stack dump every "
             f"{secs:g} s (ANON_HANG_DUMP_S); stacks on demand — "
             f"kill -USR1 {os.getpid()} → stderr of the stage ===\n")
    fh.flush()
    old_fh, old_stop = _HANG["fh"], _HANG["stop"]
    if old_stop is not None:
        old_stop.set()                    # the previous thread exits on the nearest wait
    stop = threading.Event()
    th = threading.Thread(target=_hang_loop, args=(fh, secs, stop, label),
                          name="anon-hang-dump", daemon=True)
    th.start()
    _HANG["fh"], _HANG["run"], _HANG["stop"] = fh, run_dir, stop
    if old_fh is not None:
        old_fh.close()
    return path


class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        self.confidence = tuple(config.get("confidence", ("high", "context")))
        self.rule_tiers = dict(config.get("rule_tiers") or {})   # #76 P2-5
        # .jsonl files LARGER than the threshold are masked line by line (streaming) —
        # memory O(line), the output matches the in-memory path byte for byte.
        self.stream_threshold = config.get("stream_threshold_bytes",
                                           50 * 1024 * 1024)
        self.workers = resolve_workers(config.get("workers", "auto"))
        # Memory cap — only for auto (an explicit N is the operator's will,
        # like ANON_NER_WORKERS); the time budget per file is read HERE, in
        # the parent (garbage in env = SystemExit before the fork, not a worker death)
        self._workers_auto = config.get("workers", "auto") in (None, "", "auto")
        self._file_timeout = file_timeout_base()
        self._fork_apply = False
        # The store directory of computation outputs (run(capture=…)) and what
        # lies in it after the run: {relpath: output path}
        self._capture = None
        self.captured: dict = {}
        # a large structured file — in parts (see the module header)
        self._split_on = os.environ.get("ANON_SPLIT", "1").strip() != "0"
        self._split_min = env_number("ANON_SPLIT_MIN_BYTES",
                                     _SPLIT_MIN_BYTES_DEFAULT, int,
                                     "an integer number of bytes")
        self._part_batches = max(1, env_number(
            "ANON_PART_BATCHES", _PART_BATCHES_DEFAULT, int,
            "an integer number of batches"))
        self._stream_part = max(1, env_number(
            "ANON_STREAM_PART_BYTES", _STREAM_PART_BYTES_DEFAULT, int,
            "an integer number of bytes"))
        self._stream_warm = max(0, env_number(
            "ANON_STREAM_WARM_BYTES", _STREAM_WARM_BYTES_DEFAULT, int,
            "an integer number of bytes"))
        # Structured profiles (a Telegram export): a file with a format
        # signature is processed BY FIELDS (anonymizer/structured), not as
        # plain text. The switch — config["structured_profiles"]=False
        # or ANON_STRUCTURED_PROFILES=0 (a before/after measurement, an emergency rollback).
        self._structured_on = bool(config.get("structured_profiles", True)) \
            and os.environ.get("ANON_STRUCTURED_PROFILES", "1").strip() != "0"
        self._struct_cache: dict = {}     # path -> (profile | None, stream?)
        self._struct_reg = None           # the entity registry of the run
        self._prepared_tok = None         # tokens of registry preparation
        self._lazy_dumped = 0             # lazy tokens dumped
        self.units_meta = None            # units for the increment
        self.incremental = None           # the increment result of the run
        self._inc = None
        self.rules = []
        self.value_secret = config.get("detectors", {}).get("secrets", True)
        _warn_phonenumbers(config)
        self._financial_on = config.get("detectors", {}).get("financial", True)
        enabled = config.get("detectors", {})
        at_orgs = None      # the place of the orgs detector — multi-word pack brands are lifted here
        at_names = None     # the end of the orgs block (Latin brand rules — after domain*)
        for name in DETECTOR_ORDER:
            if name == "orgs":
                at_orgs = len(self.rules)
            if name == "names":
                at_names = len(self.rules)
            if not enabled.get(name, True):
                continue
            try:
                mod = importlib.import_module(f"anonymizer.detectors.{name}")
            except ModuleNotFoundError:
                # tolerate an absent module only when the detector rides the
                # implicit default; an EXPLICIT True in config must fail loudly
                # (silently dropping a requested detector = silent PII leak).
                if enabled.get(name) is True:
                    raise SystemExit(f"detector enabled in config but module "
                                     f"missing: anonymizer.detectors.{name}")
                continue
            self.rules.extend(getattr(mod, "RULES", []))
        if enabled.get("encoding", True):
            # without the contacts detector email_wide does not run — encoding does not yield
            # the punycode host to it (workers inherit through fork)
            from ..detectors import encoding as _enc
            _enc.EMAIL_WIDE_RULE_ACTIVE = bool(enabled.get("contacts", True))
        # optional known-values packs add precision on a familiar corpus.
        # a mistyped / unshipped pack must fail loudly: loading zero rules while
        # the operator believes the corpus is covered defeats the pack's purpose.
        at_pack = len(self.rules)
        for pack in config.get("known_values", []):
            try:
                mod = importlib.import_module(f"anonymizer.known_values.{pack}")
            except ModuleNotFoundError:
                raise SystemExit(f"unknown known_values pack: {pack!r} "
                                 f"(no module anonymizer.known_values.{pack})")
            self.rules.extend(getattr(mod, "RULES", []))
        # EXTERNAL pack files (--known-file): onboarding a company without editing
        # the tool sources — packgen puts runs/<co>/packs/<co>.py, and
        # the pipeline loads it by path. The same fail-loudly requirements as for
        # built-in packs: no file / no RULES / empty RULES = SystemExit.
        # The Russian trace (owner's decision 2026-10-03, all anonymization): the dictionary
        # known_values/ru_trace — "always mask" rules, bypassing verdicts and the gold set.
        # A config without the key (a manual dict of tests/API) = the default: on, env switches are honoured.
        from ..known_values import ru_trace as _rt
        rt_cfg = config.get("ru_trace")
        if not (isinstance(rt_cfg, dict) and set(rt_cfg) == {"enabled", "currency", "laws"}):
            rt_cfg = _rt.config_from_env(rt_cfg)
        for path in config.get("known_files", []):
            if not os.path.isfile(path):
                raise SystemExit(f"known-file not found: {path!r}")
            spec = importlib.util.spec_from_file_location(
                "anonymizer._known_file_" + os.path.basename(path).replace(".", "_"), path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            rules = getattr(mod, "RULES", None)
            # the mask literals of the pack are wholly covered by the dictionary (RU_TRACE_COVERED — packgen
            # does not build rules for them) — an empty RULES is honest while the dictionary is on
            covered = int(getattr(mod, "RU_TRACE_COVERED", 0) or 0)
            # literals that denote client roles are covered by an engine rule
            # (ENGINE_COVERED) — the ru_trace dictionary has nothing to do with it, turning it off does not fail the pack.
            # A mixed pack (a dictionary share AND an engine share) with the dictionary
            # turned off — SystemExit, as for a purely dictionary one: the dictionary literals are open
            engine = int(getattr(mod, "ENGINE_COVERED", 0) or 0)
            honest = (rules is not None and (covered > 0 or engine > 0)
                      and (covered == 0 or rt_cfg["enabled"]))
            if not rules and not honest:
                raise SystemExit(f"known-file {path!r} does not give a non-empty RULES — "
                                 "a pack without rules = a false sense of coverage"
                                 + (" (its literals were covered by the Russian trace dictionary, "
                                    "but the dictionary is off: ANON_RU_TRACE=0)" if covered else ""))
            self.rules.extend(rules)
        # Multi-word pack brand rules (brand_variation
        # marks them early: a space or a hyphen/dash in the form) — to the place of the orgs detector, before
        # the legal form rules and the names detector: otherwise «Директор Ромашкин Технолоджис» fell apart into
        # a PERSON_NAME of the surname and an open tail «Технолоджис», and «ООО Ромашкин Технолоджис» gave a second
        # ORG token. The relative order of the lifted rules is as in the pack. Behind stay
        # secrets/encoding/financial/contacts/addresses: a Latin brand inside an email still
        # goes to EMAIL_REDACTED before the brand. ambiguous:*/client:*/exact-tail are not lifted
        # (they have no early mark). The reordering is a full increment computation (
        # _removed_rules → None).
        # A Latin brand rule (early == "latin") is not placed before opf_name, but
        # after the last domain rule of orgs (domain*): otherwise the host label went under ORG, and
        # the remainder «.example.com»/«.ru» stayed open and the RESOURCE class was lost. A host with a brand in
        # the label — RESOURCE whole, as with 0; ahead stay quoted_name_orgctx/hashtag_brand/
        # homoglyph and the names detector (transliteration). Cyrillic ones (early == "cyr") — before opf_name.
        if at_orgs is not None:
            early = [r for r in self.rules[at_pack:] if getattr(r, "early", False)]
            if early:
                ids = {id(r) for r in early}
                rest = [r for r in self.rules if id(r) not in ids]
                cyr = [r for r in early if r.early != "latin"]
                lat = [r for r in early if r.early == "latin"]
                at_lat = at_orgs
                for i in range(at_orgs, at_names if at_names is not None else at_pack):
                    if rest[i].name.startswith("domain"):
                        at_lat = i + 1
                self.rules = rest[:at_orgs] + cyr + rest[at_orgs:at_lat] + lat + rest[at_lat:]
        # AFTER the packs: a person/organization from approved is masked by its own class and token
        # first (the surname «Смоленский» → PERSON_NAME of the same person, not GEO), the dictionary
        # takes the rest; packgen does not put geo from the dictionary into the pack (GEO_REDACTED — here).
        if rt_cfg["enabled"]:
            self.rules.extend(_rt.rules(currency=rt_cfg["currency"], laws=False))
            if rt_cfg["laws"]:
                # regulations — before the detectors (the legal form rule of the org detector would otherwise eat
                # the number «152» from «ГК РФ и 152-ФЗ»)
                self.rules[:0] = _rt.law_rules()
        # Glue post-rules "a token inside a snake_case identifier"
        # (HANDLE_1_god, North_ORG_2_South) — STRICTLY last, after
        # the packs: exactly the pack literals cut an identifier at «_». The switch
        # detectors.glue (on by default; CONFIG has no key — it is not a built-in PD
        # class, but a normalization of the result).
        if enabled.get("glue", True):
            from ..detectors import glue as _glue
            self.rules.extend(_glue.RULES)
        # pack:… names cannot be checked by config.load (external packs
        # are loaded here) — the same fail-loudly as for built-in rules
        names = {r.name for r in self.rules}
        unknown = sorted(k for k in self.rule_tiers
                         if k.startswith("pack:") and k not in names)
        if unknown:
            raise SystemExit(f"rule_tiers: unknown pack rules {unknown} — "
                             "the tier would be silently ignored")
        # CRM company-column masking (owner-gated, auto-detects company columns)
        self._mask_crm = config.get("owner", {}).get("mask_crm_columns", False) \
            and enabled.get("orgs", True)
        self._company_fn = None
        if self._mask_crm:
            from ..detectors.orgs import company_cell_fn
            # explicit columns (safe) when named; else auto-detect (aggressive)
            cols = config.get("owner", {}).get("crm_columns") or None
            self._company_fn = company_cell_fn(cols)

    # value-level entropy secret pass (format-aware, vault-gated by file OR key)
    def _secret_value_fn(self, vault_file):
        def fn(value, key=None):
            # zone/key — the evidence, hex 32/40/64 here is a key, not a digest
            if (vault_file or is_vault_context(key)) and is_secret_value(value, hash_ok=True):
                return P.SECRET
            return value
        return fn

    def _view_on(self) -> ViewModes:
        """View modes (ViewModes: transport, fold_view, homoglyphs, the stage by the raw view
        for a two-sided token with a Latin head) — all on (the switches are removed)."""
        return ViewModes.ALL

    @staticmethod
    def _build_view(text: str, rel, modes: ViewModes, cnt=None, both: bool = False, fcnt=None,
                    raw_ambiguous: bool = False, rev=None):
        """The text view for the rules: transport (a PosMap map or None), on top — the fold_view layer
        (invisible characters, NFC, accents; its own map, ChainMap composition), and on top — the homoglyph layer
        (1:1). ``view is text`` — nothing to decode and fold. ``fcnt`` — a list: the number
        of folded places of the fold_view layer (the view_fold report counter). ``both`` — the stage of the homoglyph
        layer: False — only Latin lookalikes in a Cyrillic word (owner's decision item 4),
        a Latin word with Cyrillic lookalikes («Petrоva», «Romаshka») stays mixed —
        the structural trigger names.translit_leet_name (the verify canary of the homoglyph class: without it
        «Petrоva» with a person cue stayed open); True — the second stage, full folding: the pack
        covers «Romаshka» with the clean spelling (the reverse case). ``raw_ambiguous``
        (HOMOGLYPH_AMBIGUOUS_STAGE2): the stage by the raw view — a two-sided token with
        a Latin head («comе», «xyом») stays as written, the rest is folded as in the first one.
        The fold_view and transport layers in this stage are the same as in the first one: a raw token is searched in the view
        without invisible characters and in NFC. ``rev`` — a list for counting
        Latin words with Cyrillic lookalikes (vnorm.homoglyph_view)."""
        modes = ViewModes.of(modes)
        t_on, f_on, h_on = modes.transport, modes.fold_view, modes.homoglyph
        view, pmap = transport_view(rel, text) if t_on else (text, None)
        if f_on:
            view, pm2 = fold_view(view)
            if pm2 is not None:
                if fcnt is not None:
                    fcnt.append(pm2.nseg())
                pmap = pm2 if pmap is None else ChainMap(pm2, pmap)
        if h_on:
            view = homoglyph_view(view, cnt, both=both, raw_ambiguous=raw_ambiguous, rev=rev)
        return view, pmap

    def _apply_view(self, text, view, pmap, report, rel, modes, both, raw_ambiguous=False,
                    view_rel=None, last=None, **kw):
        """``view_rel`` is the name for the transport view (see _masked); None means ``rel``. ``last``
        is a list [cnt, rev]: after each refold it holds the token count of the stage-1 view
        of the CURRENT output (the same pass that builds the view); no refold — stays as passed."""
        vrel = view_rel or rel

        def refold(t):
            if last is None:
                return self._build_view(t, vrel, modes, both=both, raw_ambiguous=raw_ambiguous)
            cnt, rev = [], []
            v, pm = self._build_view(t, vrel, modes, cnt, both=both, raw_ambiguous=raw_ambiguous,
                                     rev=rev)
            last[:] = [cnt, rev]
            return v, pm
        def capped(rounds: int) -> None:
            # cap on view rounds: the same counter and WARN as the cap on rule passes
            # (fixpoint_capped: the pipeline "clean" flag and the report see it)
            if hasattr(report, "fixpoint_capped"):
                report.fixpoint_capped += 1
            self._warn_view_capped(rel, rounds)
        return apply_view(
            text, view, pmap,
            lambda v, ed: self._apply_fixpoint(v, report, rel, edits=ed, **kw),
            refold=refold,
            audit=self._key_audit(report, rel) if pmap is not None else None,
            on_cap=capped)

    def _key_audit(self, report, rel):
        """Replacements whose source lies in a JSON key: a WARN once per file with the count (a
        full-name key is masked the same way as in a UTF-8 file today; open question to the owner)."""
        if rel is None or not rel.lower().endswith((".json", ".jsonl", ".ndjson")):
            return None

        def audit(log, src, pmap):
            n = json_key_edits(src, log, pmap)
            if not n:
                return
            seen = self.__dict__.setdefault("_key_warned", {})
            seen[rel] = seen.get(rel, 0) + n
            if seen[rel] == n:
                print(f"pipeline: WARN {rel}: replacements in JSON keys: {n} (the decoded key "
                      "is masked as in a UTF-8 file; transport view)", flush=True)
        return audit

    def _masked(self, text: str, report, rel, view_on: ViewModes, view_rel=None, **kw):
        """Rules over the view, substitution into the source → (new_text, new_view).
        ``view_on`` is ViewModes, see _view_on; ``kw`` are the _apply_fixpoint arguments of the
        call site (left_ctx, max_passes). No view needed (knobs at 0, nothing to decode or fold) —
        the old _apply_fixpoint call, new_view = new_text. ``view_rel`` is the name
        for the transport view only (html slot of a mail: ``rel + ".html"``);
        rules, the error log and the key audit see the real ``rel``."""
        view_on = ViewModes.of(view_on)
        vrel = view_rel or rel
        if any(view_on):
            cnt: list = []
            fcnt: list = []
            # last is [cnt, rev] of the current output view from refold; None in last means the
            # count is stale (output changed without refold) — a pass over the view, as before
            lean = view_on.homoglyph
            rev: list = []
            last = None
            # stage 1: transport + fold_view + Latin look-alikes in a Cyrillic word (both=False)
            view, pmap = self._build_view(text, vrel, view_on, cnt, both=False, fcnt=fcnt,
                                          rev=rev if lean else None)
            if lean:
                last = [cnt, rev]
            # window core report: view counters over the lines of its own core
            core = getattr(report, "core_slice", None)
            if view is not text:
                ccnt, cfcnt = cnt, fcnt
                if core is not None:
                    ccnt, cfcnt = [], []
                    self._build_view(core(text), vrel, view_on, ccnt, both=False, fcnt=cfcnt)
                if ccnt:
                    report.view_homoglyph = getattr(report, "view_homoglyph", 0) \
                        + sum(c[0] for c in ccnt)
                if cfcnt:
                    report.view_fold = getattr(report, "view_fold", 0) + sum(cfcnt)
                new, new_view, _n = self._apply_view(text, view, pmap, report, rel, view_on, False,
                                                     view_rel=vrel, last=last, **kw)
            else:
                new = self._apply_fixpoint(text, report, rel, **kw)
                new_view = new
                if lean and new != text:
                    last = None
            # stage over the raw view (HOMOGLYPH_AMBIGUOUS_STAGE2): an ambiguous
            # token with a Latin head («comе» in «example.comе», «xyом», «Coк» — all letters are
            # look-alikes in both alphabets) was folded to Cyrillic by stage 1 («Сoк Иван» —
            # the pair matched, «Пeтров» is covered by a rule); left unmasked, rules see it
            # raw, as before the view: a Latin pack rule with a Cyrillic tail «[а-яё]*» takes
            # «example.comе»/«xyом» (merge leaks). The extra rule pass over the file runs only
            # when such a token is in the input (stage-1 count, third number) and intact in the
            # output («Сoк»/«тaк» with a Cyrillic head do not trigger the stage). No folding
            # here — view_homoglyph does not grow; new_view further on is the stage-1 view (raw
            # tokens folded): the stream context tail and the stage-2 count are as without this stage.
            if view_on.homoglyph_defer and any(c[2] for c in cnt):
                # lean: no ambiguous tokens in the output view (refold count) — no stage without a
                # second view build; the count is stale or they exist — build and check, as before
                if last is None or any(c[2] for c in last[0]):
                    cnt_r: list = []
                    rev_r: list = []
                    view_r, pmap_r = self._build_view(new, vrel, view_on, cnt_r, both=False,
                                                      raw_ambiguous=True,
                                                      rev=rev_r if lean else None)
                    if lean:
                        last = [cnt_r, rev_r]
                    if any(c[2] for c in cnt_r):
                        if view_r is not new:
                            new, new_view, _n = self._apply_view(new, view_r, pmap_r, report, rel,
                                                                 view_on, False, raw_ambiguous=True,
                                                                 view_rel=vrel, last=last, **kw)
                        else:
                            new = self._apply_fixpoint(new, report, rel, **kw)
                            new_view = new
                            if lean and new != view_r:
                                last = None
                        new_view = homoglyph_view(new_view, both=False)
            # stage 2: a Latin word with Cyrillic look-alikes («Romаshka») — full folding
            # over the stage-1 output (rules already saw the mixed token: translit_leet_name); no such
            # tokens — no stage; the counter gets only them (Cyrillic words were counted above).
            # Whether they exist is decided by the stage-1 output view (new_view is the view of new): the
            # transport view and the whole-file map are rebuilt only when the stage is needed.
            # lean: their number comes from the same refold pass (vnorm ``rev``), no second pass.
            if view_on.homoglyph:
                if last is not None:
                    n_rev = sum(last[1])
                else:
                    cnt2: list = []
                    homoglyph_view(new_view, cnt2, both=True)
                    n_rev = sum(c[1] for c in cnt2)
                if n_rev:
                    if core is not None:
                        cnt2c: list = []
                        homoglyph_view(core(new_view), cnt2c, both=True)
                        n_rev = sum(c[1] for c in cnt2c)
                    report.view_homoglyph = getattr(report, "view_homoglyph", 0) + n_rev
                    view2, pmap2 = self._build_view(new, vrel, view_on, both=True)
                    new, new_view, _n = self._apply_view(new, view2, pmap2, report, rel, view_on,
                                                         True, view_rel=vrel, **kw)
            return new, new_view
        new = self._apply_fixpoint(text, report, rel, **kw)
        return new, new

    def _apply_fixpoint(self, text: str, report, rel: str,
                        left_ctx: str = "", max_passes=None, edits=None, skip=None,
                        skip_match=None) -> str:
        """Rules to a fixpoint (_rules_fixpoint) + splitting of token fusions.
        ``edits`` (viewmask.apply_view) is the log of replacement batches: apply_rules appends
        ("sub", […]) per rule, fusion splitting appends ("ins", [positions of «-»]).
        A rule with an incomplete boundary guard welds a token to the adjacent
        letter/digit («почтаEMAIL_REDACTED», an INN inside a link, a nickname before
        Cyrillic) — the structural gate blocks the round on any new
        fusion, and a Telegram export (414 thousand messages) always has
        such cases. They are split with the SAME GLUE_RE as the gate, right after the rules
        (boundary.split_fusions: «-» at the fusion point, both neighbours intact) — dry and
        apply run the same code, the diff signature and the idempotency oracle do not
        diverge. The split is context-free (any fusion in any
        text), so the stream branch and in-memory give the same output. After
        the split — a finishing rule pass: «-» may have opened a word boundary for
        another rule (cap of 3 rounds, as in the fixpoint). ``skip`` (nsjson, _raw_guard) are
        names of rules not applied in this call (repeating a line without them);
        ``skip_match`` is a callable (rule, m) -> bool: a single match stays as is
        (the breaking replacement is removed, not the whole rule)."""
        new = self._rules_fixpoint(text, report, rel, left_ctx, max_passes, edits, skip,
                                   skip_match)
        # window core report counts only fusions in the lines of its own core (by positions)
        core_n = getattr(report, "count_in_core", None)
        for _ in range(3):
            pos = [] if (edits is not None or core_n is not None) else None
            old, (new, n) = new, split_fusions(new, positions=pos)
            if not n:
                break
            if edits is not None:
                edits.append(("ins", pos))
            report.fusions_split = getattr(report, "fusions_split", 0) + (
                n if core_n is None else core_n(old, pos))
            new = self._rules_fixpoint(new, report, rel, left_ctx, max_passes, edits, skip,
                                       skip_match)
        else:
            pos = [] if (edits is not None or core_n is not None) else None
            old, (new, n) = new, split_fusions(new, positions=pos)  # round cap: the fusion is not passed to the gate
            if n:
                if edits is not None:
                    edits.append(("ins", pos))
                report.fusions_split = getattr(report, "fusions_split", 0) + (
                    n if core_n is None else core_n(old, pos))
        return new

    def _rules_fixpoint(self, text: str, report, rel: str,
                        left_ctx: str = "", max_passes=None, edits=None, skip=None,
                        skip_match=None) -> str:
        """apply_rules until stable (cap of 3 passes). A replacement may create
        context for the NEXT rule: the pack mints the middle segment
        of a dotted composite (org.telegram.android → org.ORG_6606.android),
        and the fused tail is caught by domain_display_vs_owner only on the second
        pass (oracle: "a repeated dry gave 11 replacements").
        Idempotent by construction: apply finished = repeated dry gives 0.
        A pathological rule ping-pong is stopped by the cap and finished by the oracle.
        ``left_ctx`` is the tail of previous lines of the stream branch (see apply_rules).
        ``max_passes`` is the pass cap (default 3 — the old behaviour; the path of
        structural profiles sets more, see _STRUCT_FIXPOINT_PASSES).
        ``skip`` (nsjson) are names of rules switched off for this call: a stub tier on top of
        rule_tiers (apply_rules skips the rule, the rule list and prefilter automata stay the same);
        ``skip_match`` skips a single match (apply_rules), None is the old path."""
        if max_passes is None:
            max_passes = _fixpoint_passes()
        rule_tiers = self.rule_tiers
        if skip:
            rule_tiers = dict(rule_tiers)
            rule_tiers.update((n, _RAW_GUARD_TIER) for n in skip)
        # probe presence (PrefilterState) lives across all passes —
        # the sets are monotone, passes 2/3 do not rescan the text
        st = PrefilterState()
        # trace of rules fired in a pass (indices): the next pass skips
        # rules after the last fired one if nothing fired before them
        # (apply_rules clean_stop). Only when rule errors are visible (the report's
        # rule_errors list): a pass with a new error gives no skip — the old path counted a repeated
        # error on the same text once more, and the ANON_RULE_MAX_ERRORS threshold must match.
        errs = getattr(report, "rule_errors", None)
        trace = [] if isinstance(errs, list) else None
        n_err = len(errs) if trace is not None else 0
        new = apply_rules(text, self.rules, report, rel,
                          confidence=self.confidence, left_ctx=left_ctx,
                          rule_tiers=rule_tiers, prefilter=st, edits=edits,
                          trace=trace, skip_match=skip_match)
        for _ in range(max(1, max_passes) - 1):
            if new == text:
                break
            stop = None
            if trace is not None:
                if len(errs) == n_err:
                    stop = (trace[-1] + 1) if trace else 0
                n_err = len(errs)
                trace.clear()
            text, new = new, apply_rules(new, self.rules, report, rel,
                                         confidence=self.confidence,
                                         left_ctx=left_ctx,
                                         rule_tiers=rule_tiers,
                                         prefilter=st, edits=edits,
                                         trace=trace, clean_stop=stop,
                                         skip_match=skip_match)
        else:
            # pass cap exhausted while the last pass still changed the text —
            # convergence not proven (name groups of the registry preparation
            # must converge, otherwise — the old single call)
            if new != text and hasattr(report, "fixpoint_capped"):
                report.fixpoint_capped += 1
                self._warn_capped(rel, max_passes)
        return new

    # ---- nsjson: guard of the raw pass over a JSONL line ----------------------------------
    def _raw_guard(self, body: str, new: str, report, rel: str, view_on, left_ctx: str,
                   line: int, say: bool = True) -> str:
        """Line ``body`` (valid JSON before the raw pass) became ``new`` (invalid JSON):
        find the replacements that break it and repeat the line without them. Search: the line
        runs through the same ``_masked`` with the same context tail into a draft report — the list
        of fired rules; first one at a time (one rule removed → the line is valid), otherwise all
        fired ones are removed (on a new breakage also the newly fired, cap of 3 rounds), then minimisation:
        a rule comes back if the line stays valid. Then refinement by REPLACEMENTS:
        for a culprit rule with several replacements on the line
        (matches by match text, ``skip_match`` of apply_rules) only the breaking ones are removed —
        first one at a time, otherwise all and then returned one by one; a second password in the same
        line that did not break JSON stays masked. Matches of one rule with the same text
        on a line are removed together (the key is the match text). Not repaired (pathology) —
        the line as it was, without replacements. First-pass records in ``report`` are replaced by
        the final ones (Report.swap_records: the difference, not everything); removed ones go to report.raw_guard,
        one record per removed replacement (a whole rule — by the number of its first-pass replacements). Printing
        the WARN about removed replacements is the caller's job (_say_guard: a chunked stream prints at assembly,
        not in the worker); ``say`` — whether to print pathology WARNs (in a worker only its own lines).
        The output is THE SAME for stream and memory: both branches call this with the same ``left_ctx`` (the
        output view tail BEFORE the guard) and the same ``body``."""
        part = lambda: Report.part(max_examples=report.max_examples,  # noqa: E731
                                   want_why=getattr(report, "want_why", True))

        def run(skip, keys=None, tap=None):
            rep = part()
            sm = None
            if keys or tap is not None:
                def sm(rule, m, keys=keys, tap=tap):
                    s = m.group(0)
                    if tap is not None:
                        lst = tap.setdefault(rule.name, [])
                        if s not in lst:
                            lst.append(s)
                    return bool(keys) and (rule.name, s) in keys
            out = self._masked(body, rep, rel, view_on, left_ctx=left_ctx,
                               skip=frozenset(skip) if skip else None, skip_match=sm)[0]
            return out, rep

        seen: dict = {}
        base, first = run((), tap=seen)
        if base != new and say:
            self._warn_guard(rel, f"line {line}: a single pass over the line differs from "
                                  "the pass over the file — the guard follows the single pass")
        fired = list(first.counts)
        order = {n: i for i, n in enumerate(fired)}
        by_order = lambda n: order.get(n, len(order))  # noqa: E731
        culprits, out, rep = None, None, None
        for name in fired:
            o, r = run({name})
            if _json_ok(o):
                culprits, out, rep = {name}, o, r
                break
        fixed = culprits is not None
        if culprits is None:
            culprits = set(fired)
            out, rep = run(culprits)
            for _ in range(3):
                if _json_ok(out):
                    break
                more = [n for n in rep.counts if n not in culprits]
                if not more:
                    break
                culprits |= set(more)
                out, rep = run(culprits)
            fixed = _json_ok(out)
            if fixed:
                for name in sorted(culprits, key=by_order):
                    o, r = run(culprits - {name})
                    if _json_ok(o):
                        culprits, out, rep = culprits - {name}, o, r
            else:
                culprits = set(fired) | set(rep.counts)
                out, rep = body, part()
                if say:
                    self._warn_guard(rel, f"line {line}: JSON is not repaired by removing replacements — "
                                          "the line is left without replacements")
        keys: set = set()            # (rule, match text) — removed single replacements
        if fixed:
            for name in sorted(culprits, key=by_order):
                if len(seen.get(name, ())) < 2:
                    continue             # one replacement of a rule on the line = the whole rule
                rest = culprits - {name}
                tap: dict = {}
                o, r = run(rest, keys, tap)
                if _json_ok(o):          # without the neighbours the line is intact even with this rule
                    culprits, out, rep = rest, o, r
                    continue
                vals = tap.get(name, [])
                if len(vals) < 2:
                    continue
                picked = None
                for v in vals:
                    o, r = run(rest, keys | {(name, v)})
                    if _json_ok(o):
                        picked, out_, rep_ = {(name, v)}, o, r
                        break
                if picked is None:
                    allv = {(name, v) for v in vals}
                    o, r = run(rest, keys | allv)
                    if not _json_ok(o):
                        continue         # pathology: the rule stays removed entirely
                    picked, out_, rep_ = allv, o, r
                    for v in vals:
                        cand = picked - {(name, v)}
                        if not cand:
                            break
                        o2, r2 = run(rest, keys | cand)
                        if _json_ok(o2):
                            picked, out_, rep_ = cand, o2, r2
                if len(picked) >= len(vals):
                    continue             # every replacement of the rule breaks — the whole rule
                culprits, keys, out, rep = rest, keys | picked, out_, rep_
        report.swap_records(first, rep)
        entries = [(name, max(1, len(seen.get(name, ())))) for name in culprits]
        for name, _v in keys:
            entries.append((name, 1))
        for name, n in sorted(entries, key=lambda e: (e[0], -e[1])):
            for _ in range(n):
                report.raw_guard.append({"rule": name, "file": rel, "line": line})
        return out

    def _say_guard(self, rel: str, entries) -> None:
        """A WARN line for each replacement removed by the guard (from Report.raw_guard records):
        one line per (line, rule), the replacement count as «×N». Called where the record
        is final: whole stream and memory — right away, chunked stream — at assembly of the parts
        (_assemble_stream; part warm-up and the parent's recount do not
        print)."""
        agg: dict = {}
        for e in entries:
            k = (e["line"], e["rule"])
            agg[k] = agg.get(k, 0) + 1
        for (line, name), n in agg.items():
            print(f"pipeline: WARN {rel}:{line or 'document'}: the raw pass broke JSON — removed "
                  f"replacement of rule {name}{f' ×{n}' if n > 1 else ''} (nsjson guard; "
                  "the remainder is seen by the control gate and residual)", flush=True)

    def _warn_guard(self, rel, msg: str) -> None:
        seen = self.__dict__.setdefault("_guard_warned", set())
        if (rel, msg) in seen:
            return
        seen.add((rel, msg))
        print(f"pipeline: WARN {rel}: JSONL guard: {msg}", flush=True)

    def _raw_guard_text(self, text: str, new: str, new_view: str, report, rel: str,
                        view_on) -> str:
        """In-memory branch: the line guard ``text`` → ``new`` (.jsonl/.ndjson/.json).
        A .json document valid as a whole before the pass and broken after it gets a document guard
        (line 0); otherwise line by line (JSONL under any name). json.loads only for changed
        lines. The line context tail is as in the stream: the output view before the guard ahead of the
        line start (new_view aligned by lines with new; not aligned — the view of previous output lines
        by the same _masked), the last _ctx_width characters."""
        n0 = len(report.raw_guard)
        out = self._raw_guard_lines(text, new, new_view, report, rel, view_on)
        self._say_guard(rel, report.raw_guard[n0:])
        return out

    def _raw_guard_lines(self, text: str, new: str, new_view: str, report, rel: str,
                         view_on) -> str:
        """Body of _raw_guard_text (no WARN printing): line by line or per document."""
        if rel.lower().endswith(".json") and _json_ok(text):
            if _json_ok(new):
                return new
            return self._raw_guard(text, new, report, rel, view_on, "", 0)
        src, out = text.split("\n"), new.split("\n")
        if len(src) != len(out):
            self._warn_guard(rel, f"line count changed ({len(src)} → {len(out)}) — "
                                  "a line guard is impossible")
            return new
        ctx_w = self._ctx_width()
        aligned = ctx_w and new_view.count("\n") == len(out) - 1
        vlines = None       # output view lines — one split per file
        changed = False
        for i, (a, b) in enumerate(zip(src, out)):
            if a == b:
                continue
            body, cr = (a[:-1], "\r") if a.endswith("\r") else (a, "")
            nb = b[:-1] if (cr and b.endswith("\r")) else b
            if _json_ok(nb) or not _json_ok(body):
                continue
            tail = ""
            if ctx_w and i:
                if aligned:
                    if vlines is None:
                        vlines = new_view.split("\n")
                    k = i
                    while k > 0 and len(tail) < ctx_w:
                        k -= 1
                        tail = vlines[k] + "\n" + tail
                    tail = tail[-ctx_w:]
                else:
                    k = i
                    while k > 0 and len(tail) < ctx_w:
                        k -= 1
                        pb, pcr = (out[k][:-1], "\r") if out[k].endswith("\r") else (out[k], "")
                        v = self._masked(pb, Report.part(want_why=False), rel, view_on)[1]
                        tail = v + pcr + "\n" + tail
                    tail = tail[-ctx_w:]
            fixed = self._raw_guard(body, nb, report, rel, view_on, tail, i + 1)
            if fixed != nb:
                out[i] = fixed + (cr if b.endswith("\r") else "")
                changed = True
        return "\n".join(out) if changed else new

    @staticmethod
    def _line_no_at(path: str, offset: int) -> int:
        """Number (1-based) of the line starting at byte ``offset``: newlines before it."""
        n = 1
        with open(path, "rb") as fh:
            left = offset
            while left > 0:
                chunk = fh.read(min(left, 1 << 20))
                if not chunk:
                    break
                n += chunk.count(b"\n")
                left -= len(chunk)
        return n

    def _warn_view_capped(self, rel, rounds: int) -> None:
        """The cap of view rounds (viewmask.apply_view) is exhausted, the output view did not converge —
        a WARN with the file name (once per file in the process)."""
        if rel is None:
            return
        seen = self.__dict__.setdefault("_view_capped_warned", set())
        if rel in seen:
            return
        seen.add(rel)
        print(f"pipeline: WARN {rel}: rule rounds over the view did not converge in {rounds} "
              "(fixpoint_capped) — the output view is not proven, a repeated dry will find the rest",
              flush=True)

    def _warn_capped(self, rel, passes: int) -> None:
        """Pass cap exhausted — a WARN with the file name (once per file in the process)."""
        if rel is None:
            return
        seen = self.__dict__.setdefault("_capped_warned", set())
        if rel in seen:
            return
        seen.add(rel)
        print(f"pipeline: WARN {rel}: the rule fixpoint did not converge in {passes} passes "
              "(fixpoint_capped) — a repeated dry will find the rest", flush=True)

    def _chunked(self, rel: str, text: str) -> bool:
        """The file's rules will run in batches — CSV/TSV (records) or a plain text (lines) no
        shorter than _CSV_CHUNK_MIN characters."""
        return len(text) >= _CSV_CHUNK_MIN and (_is_csv(rel) or _plain_text(rel))

    @staticmethod
    def _chunk_bounds(rel: str, text: str) -> list:
        """Batch boundaries: CSV/TSV — whole records (a quoted multiline cell is not cut), a plain text —
        lines."""
        if _is_csv(rel):
            return csv_record_chunks(text, _CSV_CHUNK_CHARS)
        return line_chunks(text, _CSV_CHUNK_CHARS)

    def _csv_halo(self) -> int:
        """Width of the batch window margin (characters on each side, see the _CSV_HALO_CHARS block)."""
        return max(_CSV_HALO_CHARS, 8 * self._ctx_width_cached())

    def _ctx_width_cached(self) -> int:
        """_ctx_width per rule list (iterating 84 thousand rules for every batch task is expensive)."""
        hit = self.__dict__.get("_ctxw")
        if hit is None or hit[0] is not self.rules:
            hit = self._ctxw = (self.rules, self._ctx_width())
        return hit[1]

    def _masked_window(self, w: str, n0: int, n_r: int, last: bool, base: int, rel,
                       view_on, rep):
        """Batch window: ``w`` is the left margin (``n0`` newlines), the core, the right margin (``n_r``
        newlines; ``last`` — no right margin, the core runs to the window end). Rules — the same
        ``_masked`` as for a whole text; the core output is the core lines of the output (masks do not change
        the line count; if it changed — _ChunkLines, the whole file). ``rep`` is the core _CoreReport. Returns
        the core output."""
        new, _nv = self._masked(w, rep, rel, view_on)
        if new.count("\n") != w.count("\n"):
            raise _ChunkLines()
        c0, c1 = _core_span(new, n0, n_r, last)
        rep.done(base)
        return new[c0:c1]

    def _masked_chunks(self, text: str, report, rel, view_on):
        """``_masked`` over record batch windows in order (the sequential path;
        batches on workers are _cprep_task/_cpart_task, same output and report). The core report is merged into
        ``report`` in batch order; the rule error threshold is per file, as for a whole text.
        Returns as ``_masked``: (output, output) — the CSV output view is not needed further."""
        bounds = self._chunk_bounds(rel, text)
        halo = self._csv_halo()
        limit = _rule_max_errors()
        acc = Report.part()            # window cores; into report on success (_ChunkLines — rollback)
        outs = []
        for a, b in zip(bounds, bounds[1:]):
            l, r = _halo_bounds(text, a, b, halo)
            rep = _CoreReport(text.count("\n", l, a), text.count("\n", b, r), b == r)
            try:
                outs.append(self._masked_window(text[l:r], rep.n0, rep.n_r, rep.last, l, rel,
                                                view_on, rep))
            except RuleErrorsExceeded:
                errs = acc.rule_errors + rep.done(l).rule_errors
            else:
                acc.merge(rep)
                errs = acc.rule_errors
                if not errs or len(report.rule_errors) + len(errs) <= limit:
                    continue
            # threshold per file (as for a whole text: column rules, then windows in order) —
            # the file report gets the first limit+1 errors, the file failed
            room = max(0, limit) + 1 - len(report.rule_errors)
            report.rule_errors.extend(errs[:max(0, room)])
            raise RuleErrorsExceeded(f"pipeline: file {rel!r}: more rule errors than "
                                     f"ANON_RULE_MAX_ERRORS={limit} — file failed")
        report.merge(acc)
        new = "".join(outs)
        return new, new

    # ---- large CSV/TSV in batches on different workers ------------
    # The sequential path (_masked_chunks) walks the windows in order on one core (16.8 MB:
    # ~10 min). Windows are independent — a PREPARATION task (_cprep_task: reading, column rules over
    # the whole text, batch and margin bounds, the text as one piece in the run's piece directory), and from its
    # result — WINDOW tasks (_cpart_task). The file report is column rules, then window cores in
    # order, then value passes over the whole text (_value_passes) — exactly as in _run_file;
    # output and report are byte for byte as in the sequential batch pass. The rule error threshold is per
    # file (as in _assemble_stream).
    def _csv_splittable(self, path: str, rel: str) -> bool:
        """A large CSV/TSV or plain text goes in batches on workers (the "batches or whole" decision is
        _chunked over the text in preparation; here — a cheap filter by size in bytes)."""
        if not self._split_on or not (_is_csv(rel) or _plain_text(rel)):
            return False
        try:
            return os.path.getsize(path) >= max(self._split_min, _CSV_CHUNK_MIN)
        except OSError:
            return False

    def _cprep_task(self, task):
        """Worker: preparation of a large CSV/TSV → ("cprep", idx, plan | None, reason | None,
        rule_errors of a failed preparation | None). The plan is (piece with the text after column
        rules, _write_chunks windows, column rules report). None without a reason — the whole file
        as the old task (below the threshold / a single batch)."""
        _k, idx, path, rel = task
        rep0 = Report.part()
        try:
            with file_deadline(file_deadline_for(path, self._file_timeout)):
                text = read_text(path)
                if text is None or not self._chunked(rel, text):
                    return "cprep", idx, None, None, None
                new = self._csv_prepass(text, rel, rep0) if _is_csv(rel) else text
                bounds = self._chunk_bounds(rel, new)
                if len(bounds) < 3:
                    return "cprep", idx, None, None, None
                piece, specs = self._write_chunks(new, bounds, self._csv_halo())
        except _FILE_FAILURES as exc:
            return "cprep", idx, None, _file_failure(exc), list(rep0.rule_errors)
        return "cprep", idx, (piece, specs, rep0), None, None

    def _write_chunks(self, text: str, bounds: list, halo: int):
        """The text as one piece in the piece directory (UTF-8, surrogateescape) → (piece,
        [(l, a, b, r, l in chars)] windows: piece bytes [a, b) are the core batch, [l, r) the window with margins;
        the last is the window start in text characters (rule error spans))."""
        import bisect
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".anontmp", dir=self._pieces_dir)
        offs = [0]                       # byte starts of batches (and the text end)
        try:
            with os.fdopen(fd, "wb") as fh:
                for a, b in zip(bounds, bounds[1:]):
                    data = text[a:b].encode("utf-8", "surrogateescape")
                    fh.write(data)
                    offs.append(offs[-1] + len(data))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

        def boff(pos):                   # byte offset of char pos: the batch + its prefix
            j = bisect.bisect_right(bounds, pos) - 1
            if j >= len(bounds) - 1:
                return offs[-1]
            return offs[j] + len(text[bounds[j]:pos].encode("utf-8", "surrogateescape"))

        specs = []
        for k, (a, b) in enumerate(zip(bounds, bounds[1:])):
            l, r = _halo_bounds(text, a, b, halo)
            specs.append((boff(l), offs[k], offs[k + 1], boff(r), l))
        return tmp, specs

    def _cpart_task(self, task):
        """Worker: a batch window of a large CSV/TSV → ("cpart", idx, k, _CoreReport | None, piece
        of the core output | None, reason | None, rule_errors of a failed window | None)."""
        _k, idx, k, path, rel, (piece, lb, ab, bb, rb, base) = task
        rep = _CoreReport(0, 0, bb == rb)
        try:
            with file_deadline(file_deadline_for(path, self._file_timeout)):
                with open(piece, "rb") as fh:
                    fh.seek(lb)
                    data = fh.read(rb - lb)
                rep.n0 = data.count(b"\n", 0, ab - lb)
                rep.n_r = data.count(b"\n", bb - lb)
                w = data.decode("utf-8", "surrogateescape")
                out = self._masked_window(w, rep.n0, rep.n_r, rep.last, base, rel,
                                          self._view_on(), rep)
        except _ChunkLines:
            return "cpart", idx, k, None, None, "lines", None
        except _FILE_FAILURES as exc:
            return ("cpart", idx, k, None, None, _file_failure(exc),
                    list(rep.done(base).rule_errors))
        return "cpart", idx, k, rep, _write_piece(self._pieces_dir, out), None, None

    def _assemble_csv(self, path: str, rel: str, idx: int, plan, got: dict,
                      apply: bool) -> Report:
        """Report and output of a large CSV/TSV from windows — as _run_file in batches (_masked_chunks).
        The rule error threshold accumulates per file (column rules, then windows): exceeded — the file
        failed with the first limit+1 errors. A window where masks changed the line count (_ChunkLines) —
        the whole file by the old path (as _run_file)."""
        piece, specs, rep0 = plan
        if any(got[k][5] == "lines" for k in got):
            # window masks changed the line count (a PEM block in a cell) — the whole file by the old path
            # (_run_file whole=True), here, in the parent, under the same file budget
            sub = Report()
            try:
                with file_deadline(file_deadline_for(path, self._file_timeout)):
                    self._run_file(path, rel, sub, apply, whole=True)
            except _FILE_FAILURES as exc:
                errs = Report()
                errs.rule_errors = list(sub.rule_errors)
                _record_failed(errs, rel, _file_failure(exc))
                return errs
            self._note_captured(path, rel, sub, apply)
            return sub
        limit = _rule_max_errors()
        file_rep = Report()
        file_rep.merge(rep0)
        errs = list(rep0.rule_errors)
        outs = []
        for k in range(len(specs)):
            res = got.get(k)
            if res is None:
                raise SystemExit(f"pipeline: file {rel!r} — batch {k} of {len(specs)} is missing: "
                                 "the pool lost a task (scheduler bug)")
            rep, out, err, ferrs = res[3], res[4], res[5], res[6]
            errs.extend(ferrs if err else rep.rule_errors)
            over = bool(errs) and len(errs) > limit
            if err or over:
                sub = Report()
                if err == "rule_errors" or (not err and over):
                    sub.rule_errors = errs[:max(0, limit) + 1]
                    err = "rule_errors"
                else:
                    sub.rule_errors = errs
                _record_failed(sub, rel, err)
                return sub
            file_rep.merge(rep)
            outs.append(out)
        parts = []
        for o in outs:
            with open(o[1], "rb") as fh:
                parts.append(fh.read().decode("utf-8", "surrogateescape"))
            os.unlink(o[1])
        new = self._value_passes(path, rel, "".join(parts), file_rep)
        if new != read_text(path):
            file_rep.files_changed += 1
            dest = self._dest(path, rel, apply)
            if dest is not None:
                try:
                    write_text(dest, new)
                except WriteError as exc:   # as for a whole file in a worker
                    sub = Report()
                    sub.rule_errors = list(file_rep.rule_errors)
                    _record_failed(sub, rel, _file_failure(exc))
                    return sub
                self._note_captured(path, rel, file_rep, apply)
        return file_rep

    def _streamed(self, path: str, rel: str) -> bool:
        """A .jsonl above the threshold goes as a line stream (memory
        O(line)) — a single criterion for _run_file and the worker cap.
        A .jsonl of a structural profile (YouTrack export) is not the stream branch: it
        goes by fields (also streamed, see _run_file)."""
        if os.path.splitext(rel)[1].lower() not in (".jsonl", ".ndjson"):
            return False
        if self._structured_profile(path, rel)[0] is not None:
            return False
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1            # read_text itself will drop an unreadable file
        return size > self.stream_threshold

    # ---- structural profiles -------------------------------------
    def _structured_profile(self, path: str, rel: str):
        """(profile | None, line stream?) of a file — by the content
        signature, cached per path (the file head is read once).
        .json are documents (Telegram), .jsonl/.ndjson are records (YouTrack)."""
        if not self._structured_on \
                or not rel.lower().endswith((".json", ".jsonl", ".ndjson")):
            return None, False
        hit = self._struct_cache.get(path)
        if hit is None:
            hit = self._struct_cache[path] = _structured.detect_layout(path, rel)
        return hit

    def _struct_chain(self):
        """The run's rule chain for minting entity tokens: the same
        fixpoint as for any text, writes nothing to the report."""
        return lambda text: self._apply_fixpoint(text, Report(want_why=False),
                                                 None)

    def _prepare_structured(self, root: str, files) -> None:
        """Before the walk (and before forking workers): structural files exist → the tree's entity
        registry + batch minting of tokens for canonical names."""
        self._struct_reg = None
        self._prepared_tok = None
        self._lazy_dumped = 0
        if not self._structured_on:
            return
        if not any(self._structured_profile(p, r)[0] is not None
                   for p, r in files):
            return
        reg = _structured.registry_for_tree(root, self.workers)
        # name minting by a pool in groups (provably the same output as
        # a single call — see _prepare_names_parallel; otherwise a single call)
        reg.prepare(self._struct_chain(),
                    outs=self._prepare_names_parallel(reg.prepare_names()))
        self._struct_reg = reg
        self._prepared_tok = dict(reg._tok)   # comparison with the previous computation

    def _struct_file_chain(self, rel: str):
        """Free-text rule chain of a structural file: through the look-alike view
        (Telegram values are already decoded — no transport view needed); registry minting
        (_struct_chain) goes without the view: the entity seed does not depend on the knob."""
        view_on = self._view_on()._replace(transport=False)   # json.loads of the profile already decoded it
        html_on = self._view_on()

        def chain(text, rep):
            return self._masked(text, rep, rel, view_on,
                                max_passes=_STRUCT_FIXPOINT_PASSES)[0]

        def html(text, rep):
            # slot in an HTML record (YouTrack wikifiedDescription/textPreview): rules over the HTML view
            # (&#NNN;, &nbsp; — «Имя&nbsp;Фамилия»), substitution into the record — as for an .html file
            return self._masked(text, rep, rel, html_on, view_rel=rel + ".html",
                                max_passes=_STRUCT_FIXPOINT_PASSES)[0]
        chain.html = html
        return chain

    def _struct_secret_fn(self, rel: str):
        if not self.value_secret:
            return None
        return self._secret_value_fn(is_vault_context(os.path.basename(rel)))

    def _run_structured(self, path: str, rel: str, report: Report,
                        apply: bool, prof, dest=None) -> None:
        """A structural profile file: identity and typed entities
        go by fields, free text through the same rule fixpoint and the same
        value secret pass as a regular file. ``dest`` is where to write
        (None on apply — in place)."""
        reg = self._struct_reg
        if reg is None:
            # direct _run_file call without run(): the registry from the file itself
            reg = _structured.build_registry([(path, rel)])
            reg.prepare(self._struct_chain())
        if dest is None and apply:
            dest = path
        if _structured.mask_file(
                path, rel, prof, reg, self._struct_file_chain(rel),
                report, dest is not None, secret_fn=self._struct_secret_fn(rel),
                dest=dest):
            report.files_changed += 1

    def _dest(self, path: str, rel: str, apply: bool):
        """Where to write the file result: apply — in place; a computation with
        a store (run(capture=…)) — into the store; otherwise None (dry
        writes nothing)."""
        if apply:
            return path
        if self._capture is not None:
            return capture_path(self._capture, rel)
        return None

    def _run_file(self, path: str, rel: str, report: Report, apply: bool,
                  whole: bool = False) -> None:
        """One file: a stream for a large .jsonl, otherwise in-memory passes
        (rules to a fixpoint → value secrets → CRM columns) + an atomic
        write on apply (on a computation with a store — there as well, into the
        store). Exactly the old body of the run() loop."""
        dest = self._dest(path, rel, apply)
        # a large .jsonl is not read whole — a line stream with
        # the same pass order and the same Report counters.
        if self._streamed(path, rel):
            self._run_stream(path, rel, report, apply, dest=dest)
            return
        # a .eml/.mbox mail — by slots; parsing did not converge —
        # the flat path below, as before the wiring
        kind = _mailfmt.mail_kind(rel)
        if kind is not None and self._run_mail(path, rel, report, kind, dest):
            return
        # a structural profile file — by fields, not as flat text
        prof = self._structured_profile(path, rel)[0]
        if prof is not None:
            self._run_structured(path, rel, report, apply, prof, dest=dest)
            return
        text = read_text(path)
        if text is None:
            # read failure (EIO/ESTALE, deletion in a race, permissions) —
            # not silence: FILE_FAILED with the reason, the round excludes the file from the output
            # (file_failed_excluded, limitations), as for NUL files, without a stop
            _record_failed(report, rel, _unreadable_reason(path))
            return
        new = text
        # column rules (kind="csv_column") go FIRST on .csv/.tsv:
        # each cell is seen whole → the same value = one token
        # regardless of the line context (a regex pass first would mark
        # «Иван» next to the header NAME_CTX in the top lines and break
        # the column consistency). Regex rules further on see the token and
        # skip it (TOKEN guards); cells outside entity columns — as
        # before, only by literal rules.
        if rel.lower().endswith((".csv", ".tsv")):
            new = self._csv_prepass(new, rel, report)
        # rules over the view, substitution into the source record; value passes below
        # go over the source record (JSON stays valid, escapes intact)
        view_on = self._view_on()
        src = new
        if not whole and self._chunked(rel, new):
            # a large CSV/TSV or plain text — rules in batch windows (see _masked_chunks);
            # masks changed the line count (a PEM block in a cell) — whole, as before (``whole`` means
            # whole right away: assembly of worker batches already hit this, _assemble_csv)
            try:
                new, new_view = self._masked_chunks(new, report, rel, view_on)
            except _ChunkLines:
                new, new_view = self._masked(new, report, rel, view_on)
        else:
            new, new_view = self._masked(new, report, rel, view_on)
        # nsjson: repeat a JSONL line broken by the raw pass without the breaking replacements
        if new != src and rel.lower().endswith(_RAW_GUARD_EXT):
            new = self._raw_guard_text(src, new, new_view, report, rel, view_on)
        new = self._value_passes(path, rel, new, report)
        if new != text:
            report.files_changed += 1
            if dest is not None:
                write_text(dest, new)

    def _value_passes(self, path: str, rel: str, new: str, report: Report) -> str:
        """Value passes after the rules (INN/OGRN columns, entropy secrets, CRM columns) —
        the old tail of _run_file; the CSV assembly from batches (_assemble_csv) calls it too."""
        if rel.lower().endswith((".csv", ".tsv")):
            new = self._apply_id_columns(new, rel, report)
        if self.value_secret:
            # the zone is the file NAME, not the directory: relpath «доступы/…»
            # or «tokens/…» (a client channel/directory) marked the whole
            # subdirectory as a secret zone — entire «пароли» threads went to SECRET_REDACTED
            # because of one word in the path. JSON/CSV keys are gated separately
            # (is_vault_context(key) in _secret_value_fn) — as before.
            vault = is_vault_context(os.path.basename(rel))
            # in JSONL/CSV the value-fn also inspects nothing extra; vault by
            # filename is a conservative gate (whole file marked secret-zone)
            try:
                new2 = mask_values(path, new, self._secret_value_fn(vault))
            except csv.Error as e:          # not silently: WARN + rule_errors
                _note_csv_error(report, rel, "entropy_secret", e,
                                "entropy pass entropy_secret")
                new2 = new
            if new2 != new:
                for _ in range(new2.count(P.SECRET) - new.count(P.SECRET)):
                    report.record("entropy_secret", "<value>", P.SECRET)
                new = new2
        # CRM/registry company-name COLUMN masking (owner-gated, CSV only)
        if self._mask_crm and rel.lower().endswith((".csv", ".tsv")):
            try:                            # auto-detects company columns
                new3 = mask_csv_cells(new, self._company_fn,
                                      delimiter=csv_delimiter(rel, new))
            except csv.Error as e:          # not silently: WARN + rule_errors
                _note_csv_error(report, rel, "company_column", e,
                                "CRM columns company_column")
                new3 = new
            if new3 != new:
                # count real CRM-token insertions (ORG_20000..99999), not a
                # substring prefix — the report is the apply/no-apply oracle
                # and must reflect EVERY masked run. Pre-existing CRM-range
                # tokens appear on both sides and cancel; floor of 1 keeps a
                # pure csv-quoting rewrite visible in the report.
                delta = len(CRM_ORG_RE.findall(new3)) - len(CRM_ORG_RE.findall(new))
                for _ in range(max(1, delta)):
                    report.record("company_column", "<cell>", "ORG_*")
                new = new3
        return new

    def _run_mail(self, path: str, rel: str, report: Report, kind: str, dest) -> bool:
        """Owner's decision: a .eml/.mbox/.mbx mail — bytes → slots
        (formats/mail.parse: headers with decoded encoded-words, text/* bodies from
        QP/base64 in the part charset, the mbox envelope) → each slot through the same ``_masked`` as
        flat text (views on top; text/html — with the hint ``rel + ".html"``,
        so that the transport view expands ``&#…;``) → ``doc.render`` builds the file in the original
        encoding (QP → QP, base64 → base64, Q/B word in the same charset; unchanged slots and
        everything between them — byte for byte) → a byte write (write_bytes, tmp + publish).
        Value secrets (mask_values) and CRM columns are not called for mail: it is not CSV/JSON.
        Attachments (``in_view=False``, not text/*) are not scanned — the counter
        ``report.mail_attachments`` and a WARN; ``doc.defects`` — a WARN with a count (once per file).
        Returns False when parsing did not converge (MailParseError and any parse error): a WARN,
        the caller takes the flat path, as before the wiring; the file is not lost."""
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError:
            return False                   # flat path: read_text will return None
        try:
            doc = _mailfmt.parse(raw, kind)
        except Exception as exc:           # noqa: BLE001 — one file, not the round
            self._warn_mail(rel, "mail not parsed "
                            f"({'spans diverged' if isinstance(exc, _mailfmt.MailParseError) else 'parsing failed'}: "
                            f"{exc}) — masking it as flat text")
            return False
        if doc.defects:
            self._warn_mail(rel, f"parsed incompletely: {len(doc.defects)} defects "
                            f"(first: {doc.defects[0]})")
        parts = doc.parts
        n_att = sum(1 for p in (parts if kind == "eml" else [p for _i, p in parts])
                    if not p.in_view and p.reason in ("attachment", "binary"))
        if n_att:
            report.mail_attachments = getattr(report, "mail_attachments", 0) + n_att
            self._warn_mail(rel, f"attachments not scanned: {n_att} (not text/* or "
                            "Content-Disposition: attachment — they do not enter the view and stay as is)")
        view_on = self._view_on()
        texts: dict = {}
        for s in doc.slots:
            # the .html hint is for the transport view only (the rule error
            # log and the per-file error cap get the real rel, not «a.eml.html»)
            view_rel = rel + ".html" if s.kind == "html" else None
            new, _view = self._masked(s.text, report, rel, view_on, view_rel=view_rel)
            if new != s.text:
                texts[s.key] = new
        if not texts:
            return True
        out = doc.render(texts)
        if out != raw:
            report.files_changed += 1
            if dest is not None:
                write_bytes(dest, out)
        return True

    def _warn_mail(self, rel: str, msg: str) -> None:
        """A mail WARN — once per (file, message) in the process."""
        seen = self.__dict__.setdefault("_mail_warned", set())
        key = (rel, msg[:40])
        if key in seen:
            return
        seen.add(key)
        print(f"pipeline: WARN {rel}: {msg}", flush=True)

    def _apply_id_columns(self, text: str, rel: str, report: Report) -> str:
        """A cell of a column with an id header (ИНН, INN, ОГРН, Tax ID) —
        INN/OGRN by checksum, even if it looks like unix time (the inn rule expects an id word in the same
        line, while the label is in the header). After the fixpoint: valid non-epochs are already tokens.
        The financial detector is off — the pass is silent."""
        from ..detectors import financial as _fin
        if not self._financial_on:
            return text
        # the tier takes rule_tiers and self.confidence into account, as in
        # apply_rules and column rules: the owner's decision to lower inn/ogrn silently did not
        # take effect; a rule removed from the set (blocked, round._apply_blocked) does not mask the
        # column either. The report gets the actual token (an OGRN cell was recorded as INN_REDACTED).
        by_name = {r.name: r for r in self.rules}
        allowed = {}
        for name, tok in (("inn", P.INN), ("ogrn", P.OGRN)):
            r = by_name.get(name)
            allowed[tok] = r is not None \
                and self.rule_tiers.get(name, r.confidence) in self.confidence
        if not any(allowed.values()):
            return text
        toks: list = []
        heads: dict = {}            # header → id_header: a few headers, a call per cell

        def fn(cell, header=None):
            hit = heads.get(header)
            if hit is None:
                hit = heads[header] = _fin.id_header(header)
            if not hit:
                return cell
            out = _fin.id_cell_mask(cell)
            if out == cell:
                return cell
            tok = P.OGRN if P.OGRN in out else P.INN
            if not allowed[tok]:
                return cell
            toks.append(tok)
            return out

        try:
            new = mask_csv_cells(text, fn, delimiter=csv_delimiter(rel, text))
        except csv.Error as e:          # not silently: WARN + rule_errors
            _note_csv_error(report, rel, "inn_id_column", e, "INN/OGRN columns inn_id_column")
            return text
        for tok in toks:
            report.record("inn_id_column", "<cell>", tok)
        return new

    def _csv_prepass(self, text: str, rel: str, report: Report) -> str:
        """Passes over .csv/.tsv BEFORE the rules (over the whole text; the CSV batch path calls the same in
        batch preparation): (1) the tabular registry profile (structured.tabular) —
        cells of personal-data columns become whole tokens of the text path; (2) column rules of the pack.
        The profile switch is shared with structural profiles
        (config structured_profiles / ANON_STRUCTURED_PROFILES). A profile exception on
        a file — WARN + rule_errors, the file goes the old way (as column rules)."""
        if self._structured_on:
            try:
                prof = _structured.tabular.profile_text(rel, text)
                if prof is not None and self._mask_crm:
                    # CRM columns (owner-gated mask_crm, _value_passes): the organisation column is
                    # an isolated crm_org range, as before; the profile does not take it
                    prof.kinds = {i: k for i, k in prof.kinds.items()
                                  if k != _structured.tabular.ORG}
                    if not prof.kinds:
                        prof = None
                if prof is not None:
                    text = _structured.tabular.mask_text(text, prof, report)
            except Exception as e:      # noqa: BLE001 — one file, not the round
                _note_csv_error(report, rel, "tabular", e, "tabular registry profile")
        return self._apply_csv_columns(text, rel, report)

    def _apply_csv_columns(self, text: str, rel: str, report: Report) -> str:
        """Column rules (orgs.csv_column_rule) over .csv/.tsv — BEFORE
        the regex passes (see _run_file): the same discipline as apply_rules
        — the confidence gate, scope by relpath, blocked rules are already removed from
        self.rules (round._apply_blocked). The report counter = the number of CHANGED
        cells (the dry↔apply oracle and the magnitude: packgen writes the expectation = the number
        of column cells ×3). A value_fn exception on a file — WARN + an entry in
        report.rule_errors + the rule is skipped, not a round crash;
        the same for csv.Error / CsvStructureError: the delimiter is chosen by
        extension (csv_delimiter: .tsv tab, .csv sniffed), the rule columns not
        found in the header — also WARN + rule_errors, not silence."""
        for rule in self.rules:
            # the tier takes rule_tiers into account, as in apply_rules (the owner's decision
            # to lower a column rule silently did not take effect)
            if rule.kind != "csv_column" \
                    or self.rule_tiers.get(rule.name, rule.confidence) not in self.confidence:
                continue
            if rule.scope is not None and rel not in rule.scope:
                continue
            value_fn = getattr(rule, "value_fn", None)
            if value_fn is None:
                continue
            n = [0]

            def fn(cell, header=None, _vf=value_fn, _n=n):
                out = _vf(cell, header)
                if out != cell:
                    _n[0] += 1
                return out

            try:
                new = mask_csv_cells(text, fn, getattr(rule, "columns", None),
                                     delimiter=csv_delimiter(rel, text))
            except Exception as e:      # noqa: BLE001 — one file, not the round
                _note_csv_error(report, rel, rule.name, e,
                                f"column rule {rule.name}")
                continue
            if new != text:
                hint = getattr(rule, "family", "csv_column")
                for _ in range(max(1, n[0])):
                    report.record(rule.name, "<cell>", f"<{hint}>")
                text = new
        return text

    def _run_files(self, files, report: Report, apply: bool) -> None:
        """The sequential path (workers=1 / few files / no fork). Each
        file goes into its own sub-report, merged by Report.merge on completion (as in
        the pool, number for number): a file failure (including a timeout in the middle
        of the rules) leaves no partial counters in the common report for a file
        that is not on disk; only its rule_errors remain."""
        # rule parsing and prefilter automata happen before the first file's budget (a pack
        # of 84 thousand rules is parsed in ~3 min; that is the cost of the process, not of a file)
        if files:
            prebuild_rules(self.rules)
        for path, rel in files:
            sub = Report()
            try:
                with file_deadline(file_deadline_for(path, self._file_timeout)):
                    self._run_file(path, rel, sub, apply)
            except _FILE_FAILURES as exc:
                # the same discipline as the pool: a failure of one file — FILE_FAILED
                errs = Report()
                errs.rule_errors = list(sub.rule_errors)
                _record_failed(errs, rel, _file_failure(exc))
                sub = errs
            else:
                self._note_captured(path, rel, sub, apply)
            report.merge(sub)

    def _note_captured(self, path: str, rel: str, sub: Report,
                       apply: bool) -> None:
        """The file changed during a computation with a store — its output lies in
        the store (the file is written exactly when files_changed)."""
        if not apply and self._capture is not None and sub.files_changed:
            self.captured[rel] = capture_path(self._capture, rel)

    # ---- a large structural file in parts -------------------------
    def _splittable(self, path: str, rel: str) -> bool:
        """The file goes in parts: a structural profile with a line layout of no
        less than ANON_SPLIT_MIN_BYTES. How many batches it has is decided by the preparation
        (one batch — the whole file, by the old path)."""
        if not (self._split_on and self._structured_on):
            return False
        prof, by_lines = self._structured_profile(path, rel)
        if prof is None or not by_lines:
            return False
        try:
            return os.path.getsize(path) >= self._split_min
        except OSError:
            return False

    def _stream_splittable(self, path: str, rel: str) -> bool:
        """A large .jsonl of the stream branch (> stream_threshold) goes in parts
        by lines: the line pass is independent of neighbouring lines, except
        the context tail (see _spart_task)."""
        if not self._split_on or not self._streamed(path, rel):
            return False
        try:
            return os.path.getsize(path) >= max(self._split_min,
                                                2 * self._stream_part)
        except OSError:
            return False

    @staticmethod
    def _line_start_at(fh, pos: int, size: int, back: bool = False) -> int:
        """Line start: the first after a \\n byte in [pos-1, …) (back=False)
        or the last not after pos (back=True); after \\n in newline=''
        mode it is always a line start (\\r\\n is one separator)."""
        if pos <= 0:
            return 0
        if pos >= size:
            return size
        if not back:
            fh.seek(pos - 1)
            at = pos - 1
            while True:
                chunk = fh.read(1 << 16)
                if not chunk:
                    return size
                i = chunk.find(b"\n")
                if i >= 0:
                    return at + i + 1
                at += len(chunk)
        hi = pos
        while hi > 0:
            lo = max(0, hi - (1 << 16))
            fh.seek(lo)
            chunk = fh.read(hi - lo)
            i = chunk.rfind(b"\n")
            if i >= 0:
                return lo + i + 1
            hi = lo
        return 0

    def _stream_specs(self, path: str) -> list:
        """Parts of a large .jsonl: [(start, end, warm-up start)] at
        line boundaries, ≈ ANON_STREAM_PART_BYTES each."""
        size = os.path.getsize(path)
        bounds = [0]
        with open(path, "rb") as fh:
            t = self._stream_part
            while t < size:
                b = self._line_start_at(fh, t, size)
                if b >= size:
                    break
                if b > bounds[-1]:
                    bounds.append(b)
                t = max(b, t) + self._stream_part
            bounds.append(size)
            specs = []
            for a, b in zip(bounds, bounds[1:]):
                warm = self._line_start_at(fh, a - self._stream_warm, size,
                                           back=True) if a else 0
                specs.append((a, b, min(warm, a)))
        return specs

    def _ctx_width(self) -> int:
        """Width of the stream branch context tail (_run_stream, _spart_task):
        max window of context rules plus the ctx regex lookbehind past the left window edge
        (ctx_reach) — the gate at the line start sees in the tail the same as in the
        whole text."""
        return max([r.window + ctx_reach(r.requires_context)
                    for r in self.rules if r.requires_context], default=0)

    def _spart_task(self, task, apply=None):
        """Worker (and the parent on a recount): a part of a large .jsonl by
        lines [off0, off1) → ("spart", idx, k, Report.part | None,
        changed?, output piece ("file", path) | None, reason | None,
        the tail at the part input, the tail at the output, rule_errors of a failed part).
        Line by line exactly as _run_stream. The context tail at the input: init_tail
        is given (a parent recount — exact) or lines [warm, off0)
        run into a draft report from an empty tail ("warm-up"): on
        long JSONL lines the tail at the part start comes out the same, the parent
        checks it."""
        _k, idx, k, path, rel, spec = task
        off0, off1, warm, init_tail = spec
        if apply is None:
            apply = self._fork_apply
        dest = self._dest(path, rel, apply)
        want = dest is not None
        rep = Report.part()
        scratch = Report.part()
        ctx_w = self._ctx_width()
        vault = is_vault_context(os.path.basename(rel))
        secret_fn = self._secret_value_fn(vault) if self.value_secret else None
        tail = "" if init_tail is None else init_tail
        start_at = off0 if init_tail is not None else warm
        out = [] if want else None
        changed = False
        assumed = None
        view_on = self._view_on()
        line0 = None        # line number of start_at — from the file once per part
        n_seen = 0
        from ..structured.engine import _ByteLines, _open_text
        try:
            with file_deadline(file_deadline_for(path, self._file_timeout)):
                fh = _open_text(path, start_at)
                try:
                    lines = _ByteLines(fh, start_at)
                    for raw in lines:
                        n_seen += 1
                        start = lines.pos - lines.last
                        if start >= off1:
                            if start != off1:
                                raise SystemExit(
                                    f"pipeline: part [{off0}, {off1}) "
                                    f"{rel!r}: line at {start} — the part "
                                    "boundary is not at a line start")
                            break
                        own = start >= off0
                        if own and assumed is None:
                            assumed = tail
                        target = rep if own else scratch
                        body, eol = _split_eol(raw)
                        # the context tail is from the masked VIEW (as the rules of the
                        # in-memory branch see it), the output is in the source record
                        new, new_view = self._masked(body, target, rel, view_on,
                                                     left_ctx=tail)
                        # nsjson: as in _run_stream; the line number comes from the read start
                        # offset once per part, then by a counter;
                        # the WARN about removed ones is printed at assembly (_assemble_stream), not here: warm-up
                        # and a parent recount printed them again
                        if new != body and not _json_ok(new) and _json_ok(body):
                            if line0 is None:
                                line0 = self._line_no_at(path, start_at)
                            new = self._raw_guard(body, new, target, rel, view_on, tail,
                                                  line0 + n_seen - 1, say=own)
                        if ctx_w:
                            tail = (tail + new_view + eol)[-ctx_w:]
                        if secret_fn is not None:
                            new2 = _mask_json_line(new, secret_fn)
                            if new2 != new:
                                for _ in range(new2.count(P.SECRET)
                                               - new.count(P.SECRET)):
                                    target.record("entropy_secret", "<value>",
                                                  P.SECRET)
                                new = new2
                        if own:
                            if new != body:
                                changed = True
                            if out is not None:
                                out.append(new + eol)
                finally:
                    fh.close()
        except _FILE_FAILURES as exc:
            return ("spart", idx, k, None, False, None, _file_failure(exc),
                    assumed, tail, list(rep.rule_errors))
        if assumed is None:
            assumed = tail
        piece = _write_piece(self._pieces_dir, "".join(out)) \
            if (out is not None and changed) else None
        return "spart", idx, k, rep, changed, piece, None, assumed, tail, None

    def _assemble_stream(self, path: str, rel: str, idx: int, specs: list,
                         got: dict, apply: bool) -> Report:
        """The report of a large .jsonl from parts — as _run_stream on the whole file.
        A part started with a wrong context tail (the warm-up did not give
        the actual tail of the previous part) is recounted here with the correct
        tail. The rule error threshold of the stream branch accumulates per file: exceeded
        on the way — the file failed with the first limit+1 errors (exactly where
        the sequential pass would fail)."""
        limit = _rule_max_errors()
        true_tail = ""
        done, errs = [], []
        for k, spec in enumerate(specs):
            res = got[k]
            if res[7] != true_tail:
                res = self._spart_task(("spart", idx, k, path, rel,
                                        (spec[0], spec[1], spec[0], true_tail)),
                                       apply=apply)
            rep, changed, text, err, _assumed, tail, ferrs = res[3:10]
            errs.extend(ferrs if err else rep.rule_errors)
            # the threshold is as in rule._note_error: checked when an error is recorded
            # (n > limit), a file without errors does not fail even with limit < 0
            over = bool(errs) and len(errs) > limit
            if err or over:
                sub = Report()
                if err == "rule_errors" or (not err and over):
                    sub.rule_errors = errs[:max(0, limit) + 1]
                    err = "rule_errors"
                else:
                    sub.rule_errors = errs
                _record_failed(sub, rel, err)
                return sub
            done.append((rep, changed, text))
            self._say_guard(rel, rep.raw_guard)     # nsjson: the final part — print
            true_tail = tail
        file_rep = Report()
        for rep, _ch, _t in done:
            file_rep.merge(rep)
        if any(ch for _r, ch, _t in done):
            file_rep.files_changed += 1
            dest = self._dest(path, rel, apply)
            if dest is not None:
                try:
                    self._write_parts(path, dest,
                                      [(s[0], s[1]) for s in specs],
                                      [t for _r, _ch, t in done])
                except WriteError as exc:   # as for a whole file in a worker
                    # the file's rule_errors travel as in _run_stream on the whole file
                    # (_run_files/_fork_chunk) — the report in parts == that one
                    sub = Report()
                    sub.rule_errors = list(file_rep.rule_errors)
                    _record_failed(sub, rel, _file_failure(exc))
                    return sub
                self._note_captured(path, rel, file_rep, apply)
        return file_rep

    def _prep_task(self, task):
        """Worker: preparing a file for parts (structured.plan_parts) →
        ("prep", idx, PartPlan | None, failure reason | None). A preparation failure
        does not decide the file's fate: the parent gives it whole by the old path."""
        _k, idx, path, rel = task
        prof = self._structured_profile(path, rel)[0]
        try:
            with file_deadline(file_deadline_for(path, self._file_timeout)):
                plan = _structured.plan_parts(path, prof, self._struct_reg,
                                              self._struct_file_chain(rel))
        except _FILE_FAILURES as exc:
            return "prep", idx, None, _file_failure(exc)
        self._dump_lazy()                # lazy tokens of the plan
        return "prep", idx, plan, None

    def _part_task(self, task):
        """Worker: a file part (structured.mask_part) → ("part", idx, k,
        Report.part | None, changed?, output piece ("file", path) | None,
        reason | None). The output is written only for a CHANGED part on
        write/store — into a temporary piece of the run's piece directory
        (_write_piece; the parent takes an unchanged piece as bytes from
        the source). LayoutError in the middle of a part (an edit did not find its line) —
        "layout": the parent redoes the whole file (there the fallback to DOM, as
        before). The time budget is as for a whole file (the watchdog is not stricter
        than before)."""
        _k, idx, k, path, rel, part = task
        prof = self._structured_profile(path, rel)[0]
        rep = Report.part()
        dest = self._dest(path, rel, self._fork_apply)
        want = dest is not None
        try:
            with file_deadline(file_deadline_for(path, self._file_timeout)):
                changed, text = _structured.mask_part(
                    path, prof, self._struct_reg, self._struct_file_chain(rel),
                    rep, self._struct_secret_fn(rel), part, want_text=want)
        except _structured.LayoutError:
            return "part", idx, k, None, False, None, "layout"
        except _FILE_FAILURES as exc:
            return "part", idx, k, None, False, None, _file_failure(exc)
        if self._capture is not None:            # the unit goes into the store
            self._write_unit(rel, f"p{k}", rep)
        piece = _write_piece(self._pieces_dir, text) \
            if (changed and text is not None) else None
        return "part", idx, k, rep, changed, piece, None

    @staticmethod
    def _copy_range(src, out, start: int, length: int, what: str) -> None:
        src.seek(start)
        left = length
        while left > 0:
            chunk = src.read(min(left, 1 << 20))
            if not chunk:
                raise SystemExit(f"pipeline: {what} is shorter than expected — the file "
                                 "changed during the run")
            out.write(chunk)
            left -= len(chunk)

    def _write_parts(self, path: str, dest: str, specs: list, outs: list):
        """Assemble the file from parts in file order and publish it atomically to
        ``dest`` (tmp + os.replace, as write_text): a changed part is its
        output piece ("file", path; _write_piece) or the output (str),
        an unchanged one is the source bytes of its range (None), a part from
        the previous computation is ("old", path, offset, length) bytes
        of the previous output. Pieces are removed on failure as well. Returns [(offset,
        length)] of the pieces in the assembled file (for the next increment).
        A failure of tmp, of writing to it (EIO, EFBIG, …) and of os.replace of the destination is
        walk.WriteError; a failure reading the source/pieces is raised as is."""
        from .walk import WriteSide, _mk_tmp, publish
        fd, tmp = _mk_tmp(dest)
        offs, pos = [], 0
        try:
            out = WriteSide(os.fdopen(fd, "wb"))
            with out, open(path, "rb") as src:
                for (off0, off1), piece in zip(specs, outs):
                    if isinstance(piece, str):
                        data = piece.encode("utf-8", "surrogateescape")
                        out.write(data)
                        n = len(data)
                    elif piece is None:
                        n = off1 - off0
                        self._copy_range(src, out, off0, n,
                                         f"source {os.path.basename(path)!r}")
                    elif piece[0] == "file":
                        n = os.path.getsize(piece[1])
                        with open(piece[1], "rb") as pf:
                            self._copy_range(pf, out, 0, n, "part output piece")
                    else:
                        _tag, old_path, o, n = piece
                        with _old_store("file output"), \
                                open(old_path, "rb") as old:
                            self._copy_range(old, out, o, n,
                                             "output of the previous computation")
                    offs.append((pos, n))
                    pos += n
            publish(tmp, dest)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        finally:                           # piece disk space — as the assembly goes
            for piece in outs:
                if isinstance(piece, tuple) and piece[0] == "file" \
                        and os.path.exists(piece[1]):
                    os.unlink(piece[1])
        return offs

    def _open_pieces(self, root: str, apply: bool) -> None:
        """The directory of part output pieces per run — in the computation
        store (capture) or a hidden .anontmp directory at the tree root on
        write (where the tmp of an atomic write lives; the tree walk skips such files,
        release treats them as binary). Pipeline.run removes it
        entirely — together with pieces of tasks replayed by the pool."""
        import tempfile
        base = self._capture if not apply else root
        self._pieces_dir = tempfile.mkdtemp(prefix=".anon_parts.",
                                            suffix=".anontmp", dir=base)

    # ---- incremental computation ------------------------------------
    def _inc_off(self, why: str):
        print(f"pipeline: incremental run impossible — {why}; full computation",
              flush=True)
        self.incremental = {"full": why}
        return None

    def _rule_digests(self) -> list:
        from .rule import rule_digest
        return [[r.name, rule_digest(r)] for r in self.rules]

    def _incremental_check(self, prev: dict):
        """Whether units of the previous computation of the same tree can be taken
        (round._incremental_prev already checked the tree, code, config, other
        packs and encoding targets). Rules: the new pack = the old one with struck-out
        rules (behaviour fingerprints, order). A rule with no trace in a unit
        (no record, no error) does not change it when struck out: at each step
        of the chain the text is the same. Chains OUTSIDE the unit report — registry name
        minting (preparation and lazy tokens) — are checked by result:
        all tokens must match. Encoding targets removed → units where the encoding layer fired
        are "dirty". Returns: a plan {dir, files,
        dirty, removed} or None (full computation, the reason is in
        self.incremental)."""
        from . import units as _units
        um = (prev.get("index") or {}).get("units") or {}
        old_d = um.get("rule_digests")
        if not old_d:
            return self._inc_off("the previous store has no rule fingerprints")
        env_old, env_new = um.get("env"), _env_snapshot()
        if env_old != env_new:
            diff = sorted(k for k in (*ENV_KNOBS, ENV_PHONENUMBERS_PKG)
                          if (env_old or {}).get(k) != env_new.get(k))
            return self._inc_off("engine environment knobs changed: "
                                 + (", ".join(diff) or "no snapshot"))
        self._digests = self._rule_digests()
        removed = _removed_rules(old_d, self._digests)
        if removed is None:
            return self._inc_off("the pack changed not only by striking out "
                                 "rules (added, changed, reordered "
                                 "or a rule without a fingerprint)")
        bad = self._inc_integrity(prev, um)
        if bad:
            return self._inc_off("the previous store is corrupt or incomplete "
                                 f"({bad}: sha256 mismatch)")
        dirty = set(removed)
        if prev.get("targets_removed"):
            from ..detectors import encoding as _enc
            dirty |= {r.name for r in _enc.RULES}
        reg = self._struct_reg
        regst = None
        if "registry" in um["unit_sha"]:
            try:
                regst = _units.read_state(
                    _units.unit_path(prev["dir"], "registry"))
            except _units.READ_ERRORS as e:
                return self._inc_off("registry tokens of the previous computation are not "
                                     f"readable ({type(e).__name__})")
        if reg is not None:
            if regst is None:
                return self._inc_off("no registry tokens of the previous computation")
            if self._prepared_tok != regst["prepared"]:
                return self._inc_off("registry tokens (name minting) changed")
            for canon, tok in regst["lazy"].items():
                if reg._canon_token(canon) != tok:
                    return self._inc_off("a lazy registry token changed")
        elif regst and (regst.get("prepared") or regst.get("lazy")):
            return self._inc_off("the registry of the previous computation does not match")
        return {"dir": prev["dir"], "files": um.get("files") or {},
                "dirty": dirty, "removed": len(removed)}

    @staticmethod
    def _inc_integrity(prev: dict, um: dict):
        """Everything the incremental run may take from the previous
        store is checked by sha256 BEFORE the first task: units (reports
        of files and parts, iso, registry — sha in the unit metadata) and outputs
        of changed files (sha in the store's index.json, prev["index"]
        ["files"]: {relpath: [name, sha256]}). Parts copy bytes of the previous
        output by offset — corruption of even one byte would go into the new output. Returns:
        what is corrupt (a string) or None. The metadata (index.json) is checked by round
        with its own sha (_store_index)."""
        from . import units as _units
        shas, files = um.get("unit_sha"), um.get("files")
        if not isinstance(shas, dict) or not isinstance(files, dict):
            return "no unit sha"
        outs = (prev.get("index") or {}).get("files") or {}
        try:
            names = _unit_names(files)
            if "registry" in shas:
                names.append("registry")
            for name in names:
                sha = _units.file_sha(_units.unit_path(prev["dir"], name))
                if sha is None or sha != shas.get(name):
                    return f"unit {name}"
            for rel, m in files.items():
                ch = any(pm["changed"] for pm in m["parts"]) \
                    if m.get("kind") == "parts" else m["changed"]
                if not ch:
                    continue
                key, ent = capture_key(rel), outs.get(rel)
                if not ent or ent[0] != key or _units.file_sha(
                        os.path.join(prev["dir"], key)) != ent[1]:
                    return f"output {rel!r}"
        except (KeyError, TypeError, AttributeError, IndexError):
            return "unit metadata"
        return None

    def _write_unit(self, rel: str, suffix: str, rep: Report) -> None:
        """Worker/parent: the unit report goes into units/ of the store."""
        from . import units as _units
        _units.write_unit(self._capture, f"{capture_key(rel)}.{suffix}", rep)
        self._dump_lazy()

    def _dump_lazy(self) -> None:
        """Lazy registry tokens of this process go into units/lazy.<pid> (the
        parent collects them: the next incremental run checks them all)."""
        reg = self._struct_reg
        if reg is None or self._capture is None \
                or len(reg._lazy) <= self._lazy_dumped:
            return
        from . import units as _units
        _units.write_state(_units.unit_path(self._capture,
                                            f"lazy.{os.getpid()}"),
                           dict(reg._lazy))
        self._lazy_dumped = len(reg._lazy)

    def _finish_units(self, files_meta: dict) -> None:
        """Parent after the pool: lazy tokens of workers + its own → units/
        registry, unit metadata → self.units_meta (round puts them into
        the store's index.json)."""
        from . import units as _units
        udir = os.path.join(self._capture, _units.UNITS)
        reg = self._struct_reg
        if reg is not None:
            lazy = {}
            if os.path.isdir(udir):
                for name in sorted(os.listdir(udir)):
                    if name.startswith("lazy.") and name.endswith(".json.gz"):
                        p = os.path.join(udir, name)
                        lazy.update(_units.read_state(p))
                        os.remove(p)
            lazy.update(reg._lazy)
            _units.write_state(_units.unit_path(self._capture, "registry"),
                               {"prepared": self._prepared_tok or {},
                                "lazy": lazy})
        if getattr(self, "_digests", None) is None:
            self._digests = self._rule_digests()
        # sha of each unit — the next incremental run checks them before the first task
        unit_sha = {}
        for name in _unit_names(files_meta) + (
                ["registry"] if reg is not None else []):
            sha = _units.file_sha(_units.unit_path(self._capture, name))
            if sha is None:
                raise SystemExit(f"pipeline: unit {name} of the computation store "
                                 "was not written (bug)")
            unit_sha[name] = sha
        self.units_meta = {"version": 1, "rule_digests": self._digests,
                           "env": _env_snapshot(), "unit_sha": unit_sha,
                           "files": files_meta}

    @staticmethod
    def _iso_state(plan) -> dict:
        """Batch bounds and isolated values of the preparation in JSON form
        (comparison with the old state from the store)."""
        st = {"starts": plan.starts, "iso_map": plan.iso_map,
              "iso_reps": {str(b): r.to_state()
                           for b, r in sorted(plan.iso_reps.items())}}
        return json.loads(json.dumps(st, ensure_ascii=True))

    def _reusable_parts(self, rel: str, plan, parts: list) -> dict:
        """Parts of the file from the previous computation {k: metadata}: the same
        part bounds, the same batches and isolated values (preparation
        with the new rules), and no trace of struck-out rules in the part."""
        from . import units as _units
        inc = self._inc
        m = inc["files"].get(rel) if inc else None
        if not m or m.get("kind") != "parts" \
                or len(m.get("parts") or ()) != len(parts):
            return {}
        if any((pm["off0"], pm["off1"]) != (p[0], p[1])
               for pm, p in zip(m["parts"], parts)):
            return {}
        try:
            old_iso = _units.read_state(_units.unit_path(inc["dir"], m["iso"]))
        except _units.READ_ERRORS:
            return {}                      # file parts — from scratch
        if old_iso != self._iso_state(plan):
            return {}
        return {k: pm for k, pm in enumerate(m["parts"])
                if not set(pm["fired"]) & inc["dirty"]}

    def _run_parallel(self, files, report: Report, apply: bool) -> bool:
        """A pool of fork processes over tree chunks in file order; False — the pool
        is unavailable (no fork), the caller goes sequentially. A
        large structural file is a preparation task, and from its result —
        part tasks (feed); the file is assembled from parts in the parent. On
        an incremental run (self._inc), units without a trace of struck-out rules come
        from the previous store, without tasks."""
        global _FORK_PIPE
        ctx = fork_context()
        if ctx is None:
            return False
        from . import units as _units
        # One file — one task, LARGE ONES FIRST (LPT): the pool tail is no
        # longer than one file. Results arrive in arbitrary order
        # (run_pool/as-completed), accumulate by idx and are merged into report STRICTLY
        # in file order — counters/examples match the sequential path.
        def _size(p):
            try:
                return os.path.getsize(p)
            except OSError:
                return 0
        inc = self._inc
        capture = self._capture if not apply else None
        split = {i for i, (p, r) in enumerate(files) if self._splittable(p, r)}
        sspecs: dict = {}                  # idx -> parts of a large .jsonl
        for i, (p, r) in enumerate(files):
            if i not in split and self._stream_splittable(p, r):
                specs = self._stream_specs(p)
                if len(specs) >= 2:
                    sspecs[i] = specs
        # a large CSV/TSV — preparation, then record batches (_cprep_task)
        csplit = {i for i, (p, r) in enumerate(files)
                  if i not in split and i not in sspecs and self._csv_splittable(p, r)}

        def kind_now(i):
            return "parts" if i in split else (
                "stream" if i in sspecs else "whole")

        # whole and stream files of the previous computation without a trace of struck-out
        reuse_file: dict = {}
        max_err = _rule_max_errors()
        if inc:
            for i, (p, r) in enumerate(files):
                m = inc["files"].get(r)
                if m and _unit_reusable(m, kind_now(i), inc["dirty"], max_err):
                    reuse_file[i] = m
        # preparations first (short ones, followed by parts — the longest
        # tasks), then parts of large .jsonl, then whole files in descending
        # size
        tasks = sorted((("prep", i, p, r) for i, (p, r) in enumerate(files)
                        if i in split), key=lambda t: -_size(t[2]))
        tasks += sorted((("cprep", i, p, r) for i, (p, r) in enumerate(files)
                         if i in csplit and i not in reuse_file), key=lambda t: -_size(t[2]))
        for i, specs in sorted(sspecs.items()):
            if i in reuse_file:
                continue
            p, r = files[i]
            tasks += [("spart", i, k, p, r, s + (None,))
                      for k, s in enumerate(specs)]
        tasks += sorted((("file", i, p, r) for i, (p, r) in enumerate(files)
                         if i not in split and i not in sspecs and i not in csplit
                         and i not in reuse_file),
                        key=lambda t: -_size(t[2]))
        n = self.workers if (split or sspecs or csplit) \
            else min(self.workers, max(1, len(tasks)))
        # worker cap by memory — from the largest file the
        # worker holds whole (a stream .jsonl is O(line), not counted); only
        # with workers=auto (an explicit N is the operator's will, as in ner_local)
        if self._workers_auto:
            # a structural file with a line layout also goes as a stream
            # (memory O(message batch)) — not part of the cap
            # a file with a transport view (JSON/HTML) holds a second copy of the text, the map and
            # the output view (refold) — counted with the _VIEW_MEM_FACTOR multiplier (measured, see _cap_size)
            largest = max((_cap_size(_size(p), r)
                           for p, r in files
                           if not self._streamed(p, r)
                           and not self._structured_profile(p, r)[1]),
                          default=0)
            n = cap_workers_by_memory(n, largest, label="pipeline")
        specs_of: dict = {}                # idx -> [(off0, off1)] of parts
        plans: dict = {}                   # idx -> PartPlan (preparation)
        reuse_parts: dict = {}             # idx -> {k: part metadata}
        redo: set = set()                  # the file is redone whole
        done_parts: dict = {}              # idx -> [part results]

        def errs_over(idx, got) -> bool:
            # the rule error threshold of the sequential pass accumulates over
            # the FILE report: parts that did not fail by their own count do not prove
            # that the file would not fail (see _rule_max_errors) — such a file goes whole
            if max_err <= 0:
                return False
            total = sum(len(r[3].rule_errors) for r in got) + sum(
                pm["n_errors"] for pm in reuse_parts.get(idx, {}).values())
            return total > max_err

        cplans: dict = {}                  # idx -> batch plan of a large CSV/TSV

        def feed(_i, res):
            kind, idx = res[0], res[1]
            path, rel = files[idx]
            if kind == "cprep":
                plan, err = res[2], res[3]
                if err is not None:        # preparation failure — file_failed at merge
                    return ()
                if plan is None:           # below the threshold / one batch — whole
                    return [("file", idx, path, rel)]
                cplans[idx] = plan
                return [("cpart", idx, k, path, rel, (plan[0],) + spec)
                        for k, spec in enumerate(plan[1])]
            if kind == "prep":
                plan = res[2]
                # JSONL profile: batches are small (tens of records) — a part is no
                # shorter than ANON_STREAM_PART_BYTES, as a part of the .jsonl stream branch
                jl = _structured.is_jsonl(self._structured_profile(path, rel)[0])
                parts = plan.parts(self._part_batches,
                                   self._stream_part if jl else 0) \
                    if plan is not None else []
                if len(parts) < 2:         # one batch / preparation failure
                    return [("file", idx, path, rel)]
                specs_of[idx] = [(p[0], p[1]) for p in parts]
                plans[idx] = plan
                reuse = self._reusable_parts(rel, plan, parts) if inc else {}
                reuse_parts[idx] = reuse
                new = [("part", idx, k, path, rel, p + (plan.iso_map,))
                       for k, p in enumerate(parts) if k not in reuse]
                if not new and errs_over(idx, []):
                    redo.add(idx)
                    return [("file", idx, path, rel)]
                return new
            if kind != "part" or idx in redo:
                return ()
            if res[6] == "layout":
                redo.add(idx)
                return [("file", idx, path, rel)]
            got = done_parts.setdefault(idx, [])
            got.append(res)
            need = len(specs_of[idx]) - len(reuse_parts.get(idx, {}))
            if len(got) == need and not any(r[6] for r in got) \
                    and errs_over(idx, got):
                redo.add(idx)
                return [("file", idx, path, rel)]
            return ()

        # CSV batches need the piece directory even in dry without a store (the batch text goes
        # to workers as a piece; the output is needed for value passes and the files_changed count)
        if (split or sspecs) and (apply or capture is not None) \
                or any(i not in reuse_file for i in csplit):
            self._open_pieces(self._root, apply)
        self._fork_apply = apply
        _FORK_PIPE = self
        # prefilter automata — in the parent BEFORE the fork (otherwise each worker
        # would build its own), then the heap is frozen for the pool's lifetime: the GC
        # of workers does not rewrite PyGC_Head of inherited objects, the COW
        # pages of rules/dictionaries/automata stay shared (bench_fork_rss)
        prebuild_rules(self.rules, n)
        try:
            with frozen_heap():
                results, dead = run_pool(
                    _fork_chunk, tasks, n, ctx, label="pipeline",
                    name=lambda t: repr(t[3] if t[0] not in ("part", "spart", "cpart")
                                        else f"{t[4]}#{t[2]}"),
                    feed=feed if (split or csplit) else None)
        finally:
            _FORK_PIPE = None
            self._fork_apply = False
        whole: dict = {}                   # idx -> (Report, reason | None)
        parts: dict = {}                   # idx -> {k: (rep, changed, text, err)}
        sparts: dict = {}                  # idx -> {k: spart result}
        cpreps: dict = {}                  # idx -> cprep result
        cparts: dict = {}                  # idx -> {k: cpart result}
        failed: dict = {}                  # idx -> reason (file_failed)
        for res in results.values():
            kind, idx = res[0], res[1]
            if kind == "file":
                whole[idx] = (res[2], res[3])
            elif kind == "part":
                parts.setdefault(idx, {})[res[2]] = res[3:]
            elif kind == "spart":
                sparts.setdefault(idx, {})[res[2]] = res
            elif kind == "cprep":
                cpreps[idx] = res
            elif kind == "cpart":
                cparts.setdefault(idx, {})[res[2]] = res
        for k in dead:
            t = tasks[k]
            if t[0] == "file":
                whole[t[1]] = (None, "worker_died")
            else:
                failed.setdefault(t[1], "worker_died")
        meta: dict = {}                    # units of the new store
        reused = units_total = 0

        def copy_unit(name):
            os.makedirs(os.path.join(capture, _units.UNITS), exist_ok=True)
            with _old_store(f"unit {name}"):
                shutil.copyfile(_units.unit_path(inc["dir"], name),
                                _units.unit_path(capture, name))

        def old_unit(name):
            with _old_store(f"unit {name}"):
                return _units.read_unit(inc["dir"], name)

        # merge STRICTLY in file order; failed goes through _record_failed into
        # a separate sub-report, so that FILE_FAILED examples come in the same order
        for idx in range(len(files)):
            path, rel = files[idx]
            key = capture_key(rel)
            if idx in reuse_file:          # file from the previous computation
                m = reuse_file[idx]
                sub = old_unit(m["unit"])
                if m["changed"]:
                    dst = capture_path(capture, rel)
                    with _old_store(f"output {rel!r}"):
                        shutil.copyfile(os.path.join(inc["dir"], key), dst)
                    self.captured[rel] = dst
                copy_unit(m["unit"])
                meta[rel] = m
                reused += 1
                units_total += 1
                report.merge(sub)
                continue
            if idx in whole:               # whole (including a redo after layout)
                sub, err = whole[idx]
                if err is None:
                    self._note_captured(path, rel, sub, apply)
                    if capture is not None:
                        meta[rel] = {"kind": "whole", "unit": f"{key}.w",
                                     "fired": _units.fired(sub),
                                     "changed": bool(sub.files_changed),
                                     "n_errors": len(sub.rule_errors)}
                        units_total += 1
                else:
                    sub = sub or Report()  # only the file's rule_errors
                    _record_failed(sub, rel, err)
                report.merge(sub)
                continue
            if idx in failed:
                sub = Report()
                _record_failed(sub, rel, failed[idx])
                report.merge(sub)
                continue
            if idx in csplit:              # batches of a large CSV/TSV
                pr = cpreps.get(idx)
                if pr is None:
                    raise SystemExit(f"pipeline: file {rel!r} — batch preparation is missing: the pool "
                                     "lost a task (scheduler bug)")
                if pr[3] is not None:      # preparation failure (timeout/memory)
                    sub = Report()
                    sub.rule_errors = list(pr[4] or ())
                    _record_failed(sub, rel, pr[3])
                    report.merge(sub)
                    continue
                sub = self._assemble_csv(path, rel, idx, cplans[idx], cparts.get(idx, {}),
                                         apply)
                if capture is not None and not sub.counts.get(FILE_FAILED):
                    # the unit is as for a whole file (kind «whole»: the incremental run takes the file whole)
                    self._write_unit(rel, "w", sub)
                    meta[rel] = {"kind": "whole", "unit": f"{key}.w",
                                 "fired": _units.fired(sub),
                                 "changed": bool(sub.files_changed),
                                 "n_errors": len(sub.rule_errors)}
                    units_total += 1
                report.merge(sub)
                continue
            if idx in sspecs:
                got = sparts.get(idx, {})
                if len(got) != len(sspecs[idx]):
                    raise SystemExit(
                        f"pipeline: file {rel!r} — {len(got)} parts of "
                        f"{len(sspecs[idx])}: the pool lost a task (scheduler "
                        "bug)")
                sub = self._assemble_stream(path, rel, idx, sspecs[idx], got,
                                            apply)
                if capture is not None and not sub.counts.get(FILE_FAILED):
                    self._write_unit(rel, "s", sub)
                    meta[rel] = {"kind": "stream", "unit": f"{key}.s",
                                 "fired": _units.fired(sub),
                                 "changed": bool(sub.files_changed),
                                 "n_errors": len(sub.rule_errors)}
                    units_total += 1
                report.merge(sub)
                continue
            got = parts.get(idx, {})
            specs = specs_of.get(idx)
            reuse = reuse_parts.get(idx, {})
            if specs is None or len(got) + len(reuse) != len(specs) \
                    or set(got) & set(reuse):
                raise SystemExit(
                    f"pipeline: file {rel!r} — {len(got)}+"
                    f"{len(reuse)} parts of {len(specs or ())}: the pool lost "
                    "a task (scheduler bug)")
            errs = [got[k][3] for k in range(len(specs))
                    if k in got and got[k][3]]
            if errs:
                # the sequential pass would stop at the first failure: the file
                # report is not merged (as for a whole structural file —
                # partial counters and batch rule_errors do not arrive)
                sub = Report()
                _record_failed(sub, rel, errs[0])
                report.merge(sub)
                continue
            # the file report = parts in batch order (as sub in mask_file),
            # then the file task report (as sub in _fork_chunk)
            file_rep = Report()
            changed = False
            outs, pmeta = [], []
            old_out = os.path.join(inc["dir"], key) if reuse else None
            for k in range(len(specs)):
                if k in reuse:             # part from the previous computation
                    pm = reuse[k]
                    rep = old_unit(pm["unit"])
                    ch = pm["changed"]
                    outs.append(("old", old_out, pm["out_off"], pm["out_len"])
                                if ch else None)
                    copy_unit(pm["unit"])
                    reused += 1
                else:
                    rep, ch, text = got[k][0], got[k][1], got[k][2]
                    outs.append(text if ch else None)
                file_rep.merge(rep)
                changed = changed or ch
                pmeta.append({"unit": f"{key}.p{k}", "off0": specs[k][0],
                              "off1": specs[k][1],
                              "fired": _units.fired(rep), "changed": ch,
                              "n_errors": len(rep.rule_errors),
                              "out_off": None, "out_len": None})
            units_total += len(specs)
            sub = Report()
            sub.merge(file_rep)
            if changed:
                sub.files_changed += 1
                dest = self._dest(path, rel, apply)
                if dest is not None:
                    try:
                        offs = self._write_parts(path, dest, specs, outs)
                    except WriteError as exc:   # as for a whole file in a worker
                        # the file report is not merged — as for structured.
                        # mask_file on the whole file (its report only on a successful
                        # write), the report in parts == the sequential one
                        sub = Report()
                        _record_failed(sub, rel, _file_failure(exc))
                        report.merge(sub)
                        continue
                    self._note_captured(path, rel, sub, apply)
                    for pm, (o, ln), piece in zip(pmeta, offs, outs):
                        if piece is not None:
                            pm["out_off"], pm["out_len"] = o, ln
            if capture is not None:
                _units.write_state(_units.unit_path(capture, f"{key}.iso"),
                                   self._iso_state(plans[idx]))
                meta[rel] = {"kind": "parts", "iso": f"{key}.iso",
                             "parts": pmeta}
            report.merge(sub)
        if capture is not None:
            self._finish_units(meta)
            if inc:
                self.incremental = {"units": units_total, "reused": reused,
                                    "removed": inc["removed"]}
        return True

    # ---- registry preparation by a pool over name groups ------------------
    def _names_task(self, task):
        """Worker: a group of registry names in one chain call (as
        Registry.prepare) → ("names", number, outputs, clean?, fallback?).
        Clean means the fixpoint converged without the pass cap and without fusion splitting:
        then the output of each name is its own fixed point and does not depend on
        the group composition. A rule ate the separator — fallback to a call
        per name (exactly the fallback path of prepare)."""
        _k, gi, names = task
        from ..structured.entities import _ISO
        rep = Report(want_why=False)
        outs = self._apply_fixpoint(_ISO.join(names), rep, None).split(_ISO)
        if len(outs) == len(names):
            clean = not rep.fixpoint_capped and not rep.fusions_split
            return "names", gi, outs, clean, False
        chain = self._struct_chain()
        return "names", gi, [chain(n) for n in names], True, True

    def _prepare_names_parallel(self, names: list):
        """Chain outputs for registry names by a pool of groups — or None (then
        Registry.prepare makes the old SINGLE call). Equality to a single call
        is proven, not assumed: all groups are clean (_names_task),
        and the joints of adjacent groups (the last name + the first of the next, with the same
        separator) split in two with the same outputs — a rule that ate the
        separator between groups would otherwise stay invisible. A fallback of one
        group to a per-name call is allowed: then a single call would also fall back
        to whole names, and for clean groups the output = the output of the name alone."""
        global _FORK_PIPE
        if self.workers <= 1 or len(names) < _NAMES_PAR_MIN:
            return None
        ctx = fork_context()
        if ctx is None:
            return None
        groups = [names[i:i + _NAMES_GROUP]
                  for i in range(0, len(names), _NAMES_GROUP)]
        tasks = [("names", gi, g) for gi, g in enumerate(groups)]
        # joints: pairs (the last name of a group, the first of the next)
        tasks += [("names", len(groups) + j, [groups[j][-1], groups[j + 1][0]])
                  for j in range(len(groups) - 1)]
        _FORK_PIPE = self
        prebuild_rules(self.rules, min(self.workers, len(tasks)))
        try:
            with frozen_heap():
                results, dead = run_pool(_fork_chunk, tasks,
                                         min(self.workers, len(tasks)), ctx,
                                         label="pipeline:names",
                                         name=lambda t: f"names#{t[1]}")
        finally:
            _FORK_PIPE = None
        if dead:
            return None
        by = {res[1]: res for res in results.values()}
        outs: list = []
        for gi in range(len(groups)):
            _k, _gi, o, clean, _fb = by[gi]
            if not clean:
                return None
            outs.extend(o)
        fell_back = any(by[gi][4] for gi in range(len(groups)))
        if not fell_back:
            for j in range(len(groups) - 1):
                _k, _gi, o, clean, fb = by[len(groups) + j]
                if fb or not clean or o != [by[j][2][-1], by[j + 1][2][0]]:
                    return None
        return outs

    def run(self, root: str, apply: bool = False, capture=None,
            prev=None) -> Report:
        """Tree run. ``capture`` is the store directory: with
        apply=False the output of EACH changed file is written there
        (capture_path), the tree is not touched; what was written is in self.captured;
        units for the next incremental run are in self.units_meta (pool).
        ``prev`` is the previous store of the same tree {"dir", "index",
        "targets_removed"} (round._incremental_prev): units without a trace of
        removed rules are taken from there; the outcome is in self.incremental."""
        report = Report()
        files = list(iter_files(root))
        # text files with NUL in the head are dropped by the sniffer and the walk does not see them; they go to the report
        # as FILE_FAILED(binary-content) (not processed, untouched on disk), not silently
        nul_failed = list(nul_text_files(root))
        self._capture = capture
        self.captured = {}
        self.units_meta = None
        self.incremental = None
        self._inc = None
        self._digests = None
        self._root = root
        self._pieces_dir = None            # directory of part output pieces
        if capture is not None:
            os.makedirs(capture, exist_ok=True)
        try:
            self._prepare_structured(root, files)
            if capture is not None and prev is not None and not apply:
                self._inc = self._incremental_check(prev)
            done = False
            if self.workers > 1 and (
                    len(files) >= _PARALLEL_MIN_FILES
                    or any(self._splittable(p, r)
                           or self._stream_splittable(p, r)
                           or self._csv_splittable(p, r)
                           for p, r in files)):
                done = self._run_parallel(files, report, apply)
            if not done:
                if self._inc is not None:
                    self.incremental = {"full": "sequential run"}
                    self._inc = None
                self._run_files(files, report, apply)
        finally:
            if self._pieces_dir is not None:
                shutil.rmtree(self._pieces_dir, ignore_errors=True)
                self._pieces_dir = None
            self._capture = None
            self._inc = None
        for _p, rel in nul_failed:
            _record_failed(report, rel, "binary-content")
        # the full number of such files: FILE_FAILED examples are capped (30,
        # shared with timeout/memory), and NUL files are recorded last — the round (_note_nul_text,
        # _oracle_root) counts them by this field, not by the examples. With the knob at 0 — 0
        report.nul_text = len(nul_failed)
        # optional: mask names/brands/doc-ids that survive in FILE/DIR NAMES.
        # off by default; the content pass never touches this leak class.
        if self.config.get("rename_files", False):
            self._rename_names(root, report, apply)
        return report

    def _run_stream(self, path: str, rel: str, report: Report, apply: bool,
                    dest=None) -> None:
        """Line-by-line processing of a large .jsonl: the file is never
        held in memory whole. The pass order on EACH line is the same as
        in the in-memory branch on the whole text (apply_rules, then value secrets
        through ``_mask_json_line``), so the output byte for byte and the Report
        counters number for number match it. Writing is a stream into a temporary
        file; os.replace ONLY if there was at least one replacement: a noop file is
        untouched (bytes/mtime intact). A dry run (apply=False) does not even create tmp.
        ``dest`` is where to write (None on apply — in place).
        """
        try:
            lines = iter_lines(path)
        except OSError as exc:
            # mirror of read_text: an unreadable file is FILE_FAILED, not silence
            _record_failed(report, rel, f"unreadable:{exc.strerror or exc}")
            return
        vault = is_vault_context(os.path.basename(rel))   # the zone is the file, not the directory
        if dest is None and apply:
            dest = path
        writer = StreamWriter(dest) if dest is not None else None
        changed = False
        # The rule context window is NOT cut at the line boundary
        # (a major finding): the in-memory branch sees the previous line's label through "\n" inside
        # window, the stream carries the tail of already processed lines (the masked
        # view — as late passes of the in-memory branch see it; the labels themselves
        # are not touched by masks, so the gate is equivalent). The tail length is the max
        # window over context rules (+ lookbehind past the window edge, see
        # _ctx_width); without such rules no tail is accumulated at all.
        ctx_w = self._ctx_width()
        tail = ""
        view_on = self._view_on()
        line_no = 0
        try:
            for raw in lines:
                line_no += 1
                body, eol = _split_eol(raw)      # EOL is split off BEFORE the rules
                new, new_view = self._masked(body, report, rel, view_on, left_ctx=tail)
                # nsjson: the guard uses the context tail BEFORE this line; the tail further on comes from the
                # output view before the guard (in memory the tail is taken the same way)
                if new != body and not _json_ok(new) and _json_ok(body):
                    n0 = len(report.raw_guard)
                    new = self._raw_guard(body, new, report, rel, view_on, tail, line_no)
                    self._say_guard(rel, report.raw_guard[n0:])
                if ctx_w:
                    # the tail is BEFORE value secrets: in-memory runs mask_values
                    # after ALL rules, i.e. the rules see the pre-secret text;
                    # the tail comes from the masked VIEW, as in memory
                    tail = (tail + new_view + eol)[-ctx_w:]
                if self.value_secret:
                    new2 = _mask_json_line(new, self._secret_value_fn(vault))
                    if new2 != new:
                        for _ in range(new2.count(P.SECRET) - new.count(P.SECRET)):
                            report.record("entropy_secret", "<value>", P.SECRET)
                        new = new2
                if new != body:
                    changed = True
                if writer is not None:
                    writer.write(new + eol)
        except BaseException:
            # an error in the middle of reading/rules: a half-written tmp is not published
            # and not left on disk; the error itself propagates (loudly)
            if writer is not None:
                writer.abort()
            raise
        if writer is not None:
            writer.commit(changed)
        if changed:
            report.files_changed += 1

    def _rename_names(self, root: str, report: Report, apply: bool) -> None:
        """Plan + (optionally) apply file/dir-name renames via the filenames
        detector. The plan is collected FIRST over a full walk; renames then run
        top-to-bottom (files, then dirs bottom-up) so a parent is never renamed
        out from under a not-yet-renamed child. Dry-run aware: when ``apply`` is
        False nothing on disk changes — the plan is only recorded in the report.
        A masked name longer than NAME_MAX bytes is shortened by walk.fit_name
        (deterministic, hash tail; a second pass leaves it as is) — the report
        records the fitted name.
        """
        try:
            from ..detectors import filenames as fn
        except ModuleNotFoundError:
            return
        # Two or more targets coincided («Отчет_Иванова.pdf»,
        # «Отчет_Иванову.pdf», «Отчет_Иванов.pdf» → one «Отчет_PERSON_NAME_….pdf») or the target is already
        # on disk — before, one file was renamed, the rest silently kept the full name, and the report
        # wrote rename_name for all. The collision is resolved by a _2/_3… suffix (walk.suffix_name, as
        # round._suffix_dst) with a WARN; on apply only performed pairs go to the report.
        from .walk import suffix_name
        plan, taken = [], set()
        for src, dst in fn.plan_renames(root, keep_taken=True):
            is_dir = os.path.isdir(src)

            def fit(path, _d=is_dir):
                n = os.path.basename(path)
                return os.path.join(os.path.dirname(path), fit_name(
                    n, is_dir=_d, stable=lambda x: fn.mask_name(x, _d) == x))
            full, dst = dst, fit(dst)
            if dst in taken or os.path.exists(dst):
                dst = suffix_name(full, taken, is_dir, fit)
                print(f"pipeline: WARN rename: name collision — {os.path.basename(src)!r} → "
                      f"{os.path.basename(dst)!r} (suffix)", flush=True)
            taken.add(dst)
            plan.append((src, dst))
        done = plan
        if apply and plan:
            done = []
            fn.apply_renames(plan, apply=True, done_pairs=done)
            if len(done) != len(plan):
                print(f"pipeline: WARN rename: {len(done)} of {len(plan)} "
                      "renames done", flush=True)
        for src, dst in done:
            report.record("rename_name", os.path.basename(src), os.path.basename(dst))
