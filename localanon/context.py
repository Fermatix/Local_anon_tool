"""Run context: working copy of the input, config, record of removals, marks and counters.

The input never changes: stages work on the copy in ``work``. Everything removed is recorded
through ``remove``, everything kept as is with a mark (not anonymized, doubtful) — through ``mark``.
From these come the report for the data owner (with paths, stays with the owner) and the manifest
for the main pipeline (only counters by category, no paths).
"""
from __future__ import annotations

import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config


_CATEGORY_KEY = re.compile(r"[A-Za-z0-9_-]{1,40}")


def category_key(category: str) -> str:
    """Category key for the manifest and counters: the part before the first ":", only
    ``[A-Za-z0-9_-]`` (otherwise ``other``) — so that a chat or file name that got into a category
    does not reach the manifest."""
    head = str(category).split(":", 1)[0].strip()
    return head if _CATEGORY_KEY.fullmatch(head) else "other"


@dataclass
class Removal:
    rel: str                 # path relative to the input root ('/' as separator)
    category: str            # image | document:<kind> | media | chat | message | personal_section | …
    reason: str              # short, without file content
    borderline: bool = False # doubtful case — shown in the report with the "check" mark


@dataclass
class Mark:
    rel: str                 # path relative to the input root ('/' as separator)
    category: str            # unprocessed | document:<kind> | image | media | archive | …
    reason: str              # short, without file content
    cur: str = ""            # current path in the working copy (after anonymize renames);
                             # empty — same as rel; rel stays original: the owner finds the file by it


# Category of the "file not anonymized" mark: the control gate skips it (our main pipeline finishes it).
UNPROCESSED = "unprocessed"


@dataclass
class RunContext:
    config: Config
    input_path: Path
    work: Path               # working copy of the input; stages edit it in place
    out: Path                # output directory: archive, manifest, report
    removals: list[Removal] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)  # kept with a mark (file untouched)
    counters: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)   # warnings for the report (no personal data)
    data: dict = field(default_factory=dict)         # exchange between stages (roster, chat map…)
    stages_run: list[str] = field(default_factory=list)  # names of stages that ran (cli → manifest)

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.work.resolve()).as_posix()

    def files(self):
        """All files of the working copy in a stable order."""
        out = []
        for root, dirs, names in os.walk(self.work):
            dirs.sort()
            for n in sorted(names):
                out.append(Path(root) / n)
        return out

    def remove(self, path: Path, category: str, reason: str, borderline: bool = False) -> None:
        """Remove a file (or directory) from the working copy and record it. The path and reason go
        to the owner's report; the manifest and counters get only the category key before the first
        ":" (``document:invoice`` → ``document``), so the kind may follow ":", but a chat or file
        name must not be in the category."""
        p = Path(path)
        rel = self.rel(p)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
        self.removals.append(Removal(rel, category, reason, borderline))
        self.counters["removed:" + category_key(category)] += 1

    def mark(self, path: Path, category: str, reason: str) -> None:
        """Keep a file as is and record a mark: the path and reason go only to the owner's report,
        the manifest and counters get the category key (``marked:<key>``). The ``unprocessed``
        category means "not anonymized": the control gate skips such a file."""
        p = Path(path)
        try:
            rel = self.rel(p)
        except ValueError:
            rel = str(path)
        self.marks.append(Mark(rel, category, reason))
        self.counters["marked:" + category_key(category)] += 1

    def is_marked(self, path: Path, category: str | None = None) -> bool:
        try:
            rel = self.rel(Path(path))
        except ValueError:
            return False
        return any((m.cur or m.rel) == rel
                   and (category is None or category_key(m.category) == category)
                   for m in self.marks)

    def marks_moved(self, moves) -> None:
        """Renames in the working copy [(old absolute path, new)] in the order they were done
        (directory contents first, then the directory) → current paths of the marks."""
        for old, new in moves:
            try:
                o, n = self.rel(Path(old).parent), self.rel(Path(new).parent)
            except ValueError:
                continue
            o = (o + "/" if o != "." else "") + Path(old).name
            n = (n + "/" if n != "." else "") + Path(new).name
            for m in self.marks:
                cur = m.cur or m.rel
                if cur == o or cur.startswith(o + "/"):
                    m.cur = n + cur[len(o):]

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] += n

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def removed_by_category(self) -> dict:
        """{category key: count} over removals — for the manifest: the category is cut at the first
        ":" and holds nothing like a path or name (see category_key)."""
        out: Counter = Counter()
        for r in self.removals:
            out[category_key(r.category)] += 1
        return dict(sorted(out.items()))

    def marked_by_category(self) -> dict:
        """{category key: count} over marks — for the manifest, like removed_by_category."""
        out: Counter = Counter()
        for m in self.marks:
            out[category_key(m.category)] += 1
        return dict(sorted(out.items()))
