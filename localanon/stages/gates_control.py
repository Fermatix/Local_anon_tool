"""Stage "gates_control".

Control checks: no value from [executor], no form of the people roster and no host from
[links].delete_hosts (and prefixes) is left in text, JSON (the transport view; a Telegram export —
the projection of free-text fields), Office/PDF document views and paths. Folding is the same as
in the engine's control gate (anonymizer/engine/vnorm.fold_text: ё/е, whitespace, dashes,
invisible characters, NFC, Latin↔Cyrillic look-alikes), hit boundaries are boundary.L/R
(gates.control.bounded_hits): the gate finds exactly what a rule had to mask. Roster forms are
case-sensitive (like the rules: «Сокол»/«СОКОЛ» — yes, «сокол» — no; a single homonym surname at the
start of a sentence — no, in a path — yes: roster.homonym_masked), contractor values and nicknames
are case-insensitive (a common-word term is case-sensitive, like its rule; lowercase only in a
host/path, executor.glued; a name that is an SQL keyword: the ALL CAPS form in a query is not a
hit, executor.sql_position — like the rule; ALL CAPS of a short term — outside a host/path,
executor.short_upper_hit; any case in quotes and after a legal form — by the rule's regexes,
executor.quoted_rx/opf_rx). INN/OGRN — by a regex with spaces between digits (executor.id_rx), like
the rule.

The file text is the one the engine masked (engine_adapter.engine_view: UTF-8 BOM, strict UTF-8,
otherwise UTF-8 + surrogateescape; anonymize re-encoded UTF-16/32 with a BOM to UTF-8, such text
left over is an unreadable_encoding violation). A file not in UTF-8, or one the document layer
failed on, is a violation (kind unreadable_encoding / unreadable_document): nobody checked it.
UTF-8 with rare broken places (a single byte or a cut character, ≤1 % and ≤64) is text and is
checked as usual; two broken bytes in a row (a name in cp1251/KOI8-R/cp866) — unreadable_encoding.
UTF-16 and UTF-32 without a BOM in a file with any extension, except binaries with a signature and
non-text data (the engine took it for a binary and did not mask it;
engine_adapter.utf16_without_bom, utf32_without_bom) — unreadable_encoding.

A file with the ``ctx.mark(..., "unprocessed", …)`` mark ("not anonymized": broken, unreadable, no
LibreOffice/OCR) is not checked by content and does not fail the run: it is listed in the data
owner's report, the main pipeline finishes it. Its path is checked like any other.

Violation → ctx.data['gate_failures']: {rel, check: content|path, kind: value kind, n} — without
the value itself. The files are checked in forked processes over the CPU slots of the hw plan
(engine_adapter.fork_map: contiguous slices, the biggest first); the result is that of one pass.
The forms come from the anonymize stage (ctx.data['gate_controls']); if it did not run, they are
rebuilt from the config and ctx.data['roster'] (the salt is set here too: the stage can run on its
own).
"""
from __future__ import annotations

import re
from pathlib import Path

from anonymizer.engine.ac import Automaton
from anonymizer.engine.boundary import CL, CR
from anonymizer.engine.vnorm import fold_text
from anonymizer.gates.control import bounded_hits

from .. import engine_adapter as EA
from .. import executor as _executor
from .. import links as _links
from .. import roster as _roster
from ..context import UNPROCESSED, category_key

NAME = "gates_control"
_CYR_TAIL_RX = re.compile(r"[а-яё]+$")


def _fold(s: str, case: bool) -> str:
    return fold_text(s, case=case, yo=True, homoglyph=True)[0]


_CL_RX = re.compile(CL)


def _left_bounded(rx, hay: str):
    """CL + rx matches, like finditer: re tries a regex starting with CL (lookbehind) at every
    position, while one starting with a literal searches fast; the left boundary is checked after."""
    pos = 0
    while True:
        m = rx.search(hay, pos)
        if m is None:
            return
        if _CL_RX.match(hay, m.start()):
            yield m
            pos = m.end()
        else:
            pos = m.start() + 1


class Checker:
    """Automata over the control forms (case-sensitive and not) plus regexes: details with
    spaces, single homonym surnames (a hit not at the start of a sentence), removal hosts."""

    def __init__(self, controls, hosts, prefixes):
        self.groups = {True: {}, False: {}}
        ids, homonyms = [], []
        glued: dict = {}
        glued_cased: dict = {}
        sql: dict = {_executor.SQL_CASED: {}, _executor.SQL_ANY: {}}
        # ALL CAPS of a short term (key as is), in quotes and after a legal form (key in lowercase)
        ctx_forms: dict = {_executor.SHORT_UPPER: {}, _executor.QUOTED: {}, _executor.OPF: {}}
        for form, kind, cased in controls:
            form = " ".join(str(form).split())
            if not form:
                continue
            if kind == _executor.KIND_INN:
                ids.append(re.sub(r"\D", "", form) or form)
                continue
            if cased in ctx_forms:
                key = _fold(form, case=cased != _executor.SHORT_UPPER)
                ctx_forms[cased].setdefault(key, set()).add(kind)
                continue
            if cased == _executor.GLUED:
                glued.setdefault(_fold(form, case=True), set()).add(kind)
                continue
            if cased == _executor.GLUED_CASED:
                glued_cased.setdefault(_fold(form, case=False), set()).add(kind)
                continue
            if cased in sql:
                key = _fold(form, case=False)
                key = key if cased == _executor.SQL_CASED else key.lower()
                sql[cased].setdefault(key, set()).add(kind)
                continue
            if kind == _roster.KIND_HOMONYM:
                homonyms.append(form)
                continue
            key = _fold(form, case=not cased)
            self.groups[cased].setdefault(key, set()).add(kind)
        self.ac = {c: (Automaton(list(g)) if g else None) for c, g in self.groups.items()}
        self.id_rx = _executor.id_rx(ids) if ids else None
        self.homonym_rx = None
        if homonyms:
            alt = "|".join(re.escape(_fold(f, case=False))
                           for f in sorted(set(homonyms), key=len, reverse=True))
            self.homonym_rx = re.compile(CL + r"(" + alt + r")" + CR)
        # a lowercase name that is a common word (a dev service word — only Capitalized/ALL CAPS):
        # a hit only in a host/path (executor.glued — like the rule)
        self.glued = {True: glued, False: glued_cased}
        self.glued_rx = {fold: self._glued_rx(forms) for fold, forms in self.glued.items()}
        # a name that is an SQL keyword: a hit, except the ALL CAPS form in a query (executor.sql_position)
        self.sql = sql
        self.sql_rx = {}
        for mode, forms in sql.items():
            if forms:
                alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
                flags = re.IGNORECASE if mode == _executor.SQL_ANY else 0
                self.sql_rx[mode] = re.compile(CL + r"(" + alt + r")" + CR, flags)
        # forms with context — by the same regex and predicate as the rule (executor: quoted_rx,
        # opf_rx, short_upper_hit); the case ending of a Latin term — like term_pattern
        self.ctx_forms = ctx_forms
        self.ctx_rx = {}
        for mode, forms in ctx_forms.items():
            if not forms:
                continue
            tail = "" if mode == _executor.SHORT_UPPER else r"[а-яё]*"
            alt = "|".join(re.escape(f) + (tail if f.isascii() else "")
                           for f in sorted(forms, key=len, reverse=True))
            if mode == _executor.SHORT_UPPER:          # CL separately (_left_bounded)
                rx = re.compile(r"(" + alt + r")" + CR)
            elif mode == _executor.QUOTED:
                rx = _executor.quoted_rx(alt, 0)
            else:
                rx = _executor.opf_rx(alt, 0)          # hay is folded to lowercase
            self.ctx_rx[mode] = rx
        self.host_rx = _links.host_residual_rx(hosts, prefixes)

    @staticmethod
    def _glued_rx(forms):
        if not forms:
            return None
        alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
        return re.compile(r"(?:(?<=\\[ntrbfv])|(?<![^\W\d_]))(" + alt + r")(?![^\W\d_])")

    def hits(self, text: str, *, path: bool = False, rel: str | None = None):
        """Counter: value kind → number of hits in the text; ``path`` — the text is a file path (file
        and directory names are a host/path context and not a sentence start); ``rel`` — the file
        name: in a code file ASCII quotes are not name quotes (executor.code_file, quoted_hit)."""
        out: dict = {}
        code = bool(rel) and _executor.code_file(rel)
        # fold the text once per case (seconds on a 100 MB log); lowercase first
        for fold in (True, False):
            hay = None
            if self.glued_rx[fold] is not None:
                hay = hay or _fold(text, case=fold)
                for m in self.glued_rx[fold].finditer(hay):
                    if _executor.glued(hay, m.start(1), m.end(1), path=path):
                        for kind in self.glued[fold][m.group(1)]:
                            out[kind] = out.get(kind, 0) + 1
            for mode, rx in (self.sql_rx.items() if not fold else ()):
                hay = hay or _fold(text, case=fold)
                for m in rx.finditer(hay):
                    if _executor.sql_position(hay, m.start(1), m.end(1)):
                        continue
                    key = m.group(1) if mode == _executor.SQL_CASED else m.group(1).lower()
                    for kind in self.sql[mode][key]:
                        out[kind] = out.get(kind, 0) + 1
            for mode, rx in self.ctx_rx.items():
                upper = mode == _executor.SHORT_UPPER
                if upper == fold:
                    continue
                hay = hay or _fold(text, case=fold)
                group = 2 if mode == _executor.OPF else 1
                for m in (_left_bounded(rx, hay) if upper else rx.finditer(hay)):
                    s, e = m.start(group), m.end(group)
                    if upper and not _executor.short_upper_hit(hay, s, e):
                        continue
                    if mode == _executor.QUOTED and not _executor.quoted_hit(hay, s, e, code=code):
                        continue
                    forms, key = self.ctx_forms[mode], m.group(group)
                    if key not in forms:            # a Latin form with a Russian ending («devа»)
                        key = _CYR_TAIL_RX.sub("", key)
                    for kind in forms.get(key, ()):
                        out[kind] = out.get(kind, 0) + 1
            ac = self.ac[not fold]
            if ac is not None:
                hay = hay or _fold(text, case=fold)
                for key in bounded_hits(ac, hay):
                    for kind in self.groups[not fold][key]:
                        out[kind] = out.get(kind, 0) + 1
            if self.homonym_rx is not None and not fold:
                hay = hay or _fold(text, case=fold)
                n = sum(1 for m in self.homonym_rx.finditer(hay)
                        if _roster.homonym_masked(hay, m.start(1), path=path))
                if n:
                    out[_roster.KIND_PERSON] = out.get(_roster.KIND_PERSON, 0) + n
            if self.id_rx is not None and fold:
                hay = hay or _fold(text, case=fold)
                n = len(self.id_rx.findall(hay))
                if n:
                    out[_executor.KIND_INN] = out.get(_executor.KIND_INN, 0) + n
        if self.host_rx is not None:
            n = len(self.host_rx.findall(text))
            if n:
                out["links.delete_host"] = out.get("links.delete_host", 0) + n
        return out


def _controls(ctx) -> list:
    controls = ctx.data.get("gate_controls")
    if controls is not None:
        return list(controls)
    pack = _executor.build(ctx.config.executor)
    ros = _roster.from_data(ctx.data.get("roster") or {})
    return pack.controls + _roster.controls(ros)


_CHUNK_BYTES = 4 << 20      # files per pool task: contiguous, about this many bytes (a bigger file — alone)
_CHUNK_FILES = 200


def _chunks(files: list) -> list:
    """Contiguous slices of the file list (the results are put together in file order)."""
    out, cur, size = [], [], 0
    for p in files:
        try:
            n = p.stat().st_size
        except OSError:
            n = 0
        if cur and (size + n > _CHUNK_BYTES or len(cur) >= _CHUNK_FILES):
            out.append((cur, size))
            cur, size = [], 0
        cur.append(p)
        size += n
    if cur:
        out.append((cur, size))
    return out


def _check_chunk(state, chunk) -> tuple:
    """Checks of one slice of files (in a worker of ``fork_map`` or here) → (notes, [(rel, kind, content
    hits | None, path hits)]): content hits — sorted (value kind, n); None — the file is "not anonymized",
    its content is not checked."""
    checker, root, unprocessed = state
    paths, _size = chunk
    notes: list = []
    out = []
    for _p, rel, kind, text in EA.iter_views(root, paths, notes=notes.append):
        if rel in unprocessed or kind in (EA.KIND_UNREADABLE_ENCODING, EA.KIND_UNREADABLE_DOCUMENT):
            content = None
        else:
            content = sorted(checker.hits(text, rel=rel).items())
        path = sorted(checker.hits(rel.replace("/", " / "), path=True, rel=rel).items())
        out.append((rel, kind, content, path))
    return notes, out


def _workers(ctx) -> int:
    from .anonymize import _engine_workers
    return _engine_workers(ctx)


def run(ctx) -> None:
    cfg = ctx.config
    EA.set_salt(cfg.project.salt)          # the stage also runs on its own: the rules mint tokens
    checker = Checker(_controls(ctx), cfg.links.delete_hosts, cfg.links.delete_host_prefixes)
    failures = ctx.data.setdefault("gate_failures", [])
    files = [Path(p) for p in ctx.files()]
    n_files = 0
    unprocessed = {m.cur or m.rel for m in getattr(ctx, "marks", ())
                   if category_key(m.category) == UNPROCESSED}
    skipped = 0
    EA.prefetch_views(ctx, files)          # document views rewritten by anonymize — in worker processes
    # the files in forked processes (the automata are inherited, only hit counters come back); the
    # results — in file order, the notes ("layer unavailable") — once, as from one pass
    got = EA.fork_map(_check_chunk, _chunks(files), _workers(ctx), state=(checker, ctx.work, unprocessed),
                      weight=lambda c: c[1], label="gates_control")
    noted: set = set()
    for notes, records in got:
        for msg in notes:
            if msg not in noted:
                noted.add(msg)
                ctx.note(msg)
        for rel, kind, content, path in records:
            n_files += 1
            if rel in unprocessed:
                # kept with the "not anonymized" mark: it is in the owner's report, we finish it on our
                # side; the content is not checked (the path is: anonymize renamed it)
                skipped += 1
            elif kind in (EA.KIND_UNREADABLE_ENCODING, EA.KIND_UNREADABLE_DOCUMENT):
                failures.append({"rel": rel, "check": "content", "kind": kind, "n": 1})
                ctx.count("gate:failures", 1)
                ctx.count("gate:" + kind, 1)
            else:
                for vkind, n in content:
                    failures.append({"rel": rel, "check": "content", "kind": vkind, "n": n})
                    ctx.count("gate:failures", n)
            for vkind, n in path:
                failures.append({"rel": rel, "check": "path", "kind": vkind, "n": n})
                ctx.count("gate:failures", n)
    ctx.count("gate:files_checked", n_files)
    if skipped:
        ctx.count("gate:skipped_unprocessed", skipped)
    if failures:
        ctx.note(f"control checks: {len(failures)} violations — the archive is built anyway, "
                 "see the report")
