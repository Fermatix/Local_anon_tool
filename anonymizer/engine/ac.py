"""Aho-Corasick: one pass over the text for ANY number of literal patterns.

Why: naive literal passes (`for c in controls: c in text` in the control
gate, batches of regex alternations in residual) are O(files × literals); on a full
corpus with thousands of control forms that is minutes to hours. The automaton gives O(text + hits)
regardless of the number of patterns. Pure stdlib (~100 lines) — zero dependencies
are kept; pyahocorasick is an opt-in C backend with the same interface (below).

Semantics: SUBSTRING search (like `in`). Word boundaries/case endings are the
caller's concern (a post-check around a hit with a local regex — the classic hybrid
"AC prefilter + exact confirmation").

    ac = Automaton(["Иванов", "Ромашка"])          # build O(sum of lengths)
    for start, end, pat in ac.iter(text): ...      # one pass
    ac.found(text) -> set(patterns)                # the common case of gates

Casefold mode: Automaton(pats, casefold=True) folds the patterns and searches in
text.casefold() — positions are returned IN THE CASEFOLDED TEXT (irrelevant for presence
gates; for positional needs the caller does text.casefold() once
and lives in its coordinates).

Backend: with pyahocorasick installed (requirements-optional) a C automaton
works inside — the same contract: ALL occurrences of ALL patterns (nested
and overlapping — through output links, not iter_long), positions are str indexes
(code points: a non-BMP character and a surrogate of a surrogateescape string from walk take
one position in both backends; the unicode build of pyahocorasick stores UCS-4),
the longest pattern first at the same end position. Casefold is folded here,
before add_word. Without the package — pure Python as before, with a one-time WARN to stderr
(like natasha in mine: the core has zero dependencies, the accelerator is a mark); the WARN comes at
import (the parent imports engine before fork → one line per run, not per
worker: the flag is inherited) or at the first build. Measured: found() ×4.7,
apply_rules with 6,000 literal rules 29.6 s → 6.8 s.
ANON_AC_BACKEND=py|c forces the choice (tests run both; py silences the
WARN; c without the package is a SystemExit: the accelerator was requested but is missing).
"""
from __future__ import annotations

import os
import sys
from collections import deque

try:
    import ahocorasick as _pyac          # C accelerator, opt-in (requirements-optional)
except ImportError:                      # pragma: no cover
    _pyac = None

BACKEND_ENV = "ANON_AC_BACKEND"
_BACKENDS = ("py", "c")
_warned = False


def _env_backend():
    return os.environ.get(BACKEND_ENV, "").strip().lower() or None


def _warn_missing() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    print("ac: WARN pyahocorasick is not installed — Aho-Corasick in pure Python "
          "(apply/control gate/residual ~4-5 times slower); "
          "pip install pyahocorasick (requirements-optional.txt); "
          f"{BACKEND_ENV}=py silences this warning",
          file=sys.stderr, flush=True)


def resolve_backend(backend=None) -> str:
    """'c' | 'py': explicit argument > ANON_AC_BACKEND > auto (c with the package, otherwise
    py + WARN once per process). Garbage / c without the package — SystemExit."""
    if backend is None:
        backend = _env_backend()
    if backend is None:
        if _pyac is not None:
            return "c"
        _warn_missing()
        return "py"
    if backend not in _BACKENDS:
        raise SystemExit(f"ac: бэкенд {backend!r} ({BACKEND_ENV}) — ожидается py|c")
    if backend == "c" and _pyac is None:
        raise SystemExit(f"ac: backend c ({BACKEND_ENV}), but pyahocorasick is not "
                         f"installed — pip install pyahocorasick or use py")
    return backend


if _pyac is None and _env_backend() != "py":
    _warn_missing()               # once per run: before pipeline forks its workers


class Automaton:
    __slots__ = ("_goto", "_fail", "_out", "_c", "casefold", "backend")

    def __init__(self, patterns, casefold: bool = False, backend=None):
        pats = set()
        for p in patterns:
            if not isinstance(p, str) or not p:
                raise SystemExit(f"ac.Automaton: a pattern must be a non-empty string, "
                                 f"got {p!r}")
            pats.add(p.casefold() if casefold else p)
        self.casefold = casefold
        self.backend = resolve_backend(backend)
        self._goto = self._fail = self._out = self._c = None
        if self.backend == "c":
            self._build_c(pats)
        else:
            self._build_py(pats)

    def _build_c(self, pats):
        # without words make_automaton leaves a trie and iter raises AttributeError —
        # an empty pattern set = an automaton without hits (_c is None)
        if not pats:
            return
        a = _pyac.Automaton()
        for pat in sorted(pats):          # sorted — deterministic build
            a.add_word(pat, pat)          # value = pattern: start = end - len + 1
        a.make_automaton()
        self._c = a

    def _build_py(self, pats):
        # goto: list of dict char->state; out: patterns ending in the state
        self._goto = [{}]
        self._out = [()]
        for pat in sorted(pats):          # sorted — deterministic build
            s = 0
            for ch in pat:
                nxt = self._goto[s].get(ch)
                if nxt is None:
                    self._goto.append({})
                    self._out.append(())
                    nxt = len(self._goto) - 1
                    self._goto[s][ch] = nxt
                s = nxt
            self._out[s] = self._out[s] + (pat,)
        # BFS fail links
        self._fail = [0] * len(self._goto)
        q = deque()
        for ch, s in self._goto[0].items():
            q.append(s)                    # fail of the root's children = root
        while q:
            s = q.popleft()
            for ch, t in self._goto[s].items():
                q.append(t)
                f = self._fail[s]
                while f and ch not in self._goto[f]:
                    f = self._fail[f]
                self._fail[t] = self._goto[f].get(ch, 0)
                if self._fail[t] == t:     # a root child pointed to itself
                    self._fail[t] = 0
                if self._out[self._fail[t]]:
                    self._out[t] = self._out[t] + self._out[self._fail[t]]

    def iter(self, text: str):
        """All occurrences: (start, end, pattern); end is exclusive."""
        if self.casefold:
            text = text.casefold()
        if self.backend == "c":
            if self._c is not None:
                # pyahocorasick.iter -> (index of the LAST character, value)
                for end1, pat in self._c.iter(text):
                    yield end1 - len(pat) + 1, end1 + 1, pat
            return
        s = 0
        goto, fail, out = self._goto, self._fail, self._out
        for i, ch in enumerate(text):
            while s and ch not in goto[s]:
                s = fail[s]
            s = goto[s].get(ch, 0)
            if out[s]:
                for pat in out[s]:
                    yield i - len(pat) + 1, i + 1, pat

    def found(self, text: str) -> set:
        """Which patterns occur (presence, no positions) — the gate mode."""
        if self.backend == "c":
            if self._c is None:
                return set()
            if self.casefold:
                text = text.casefold()
            return {pat for _, pat in self._c.iter(text)}
        return {pat for _, _, pat in self.iter(text)}
