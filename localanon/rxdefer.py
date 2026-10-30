"""Regexes of the run's rules compiled in worker processes.

``re.compile`` is almost all pure Python: the parse (``re._parser``) and the code generation
(``re._compiler._code``); the pattern object is then built from that code by the C ``_sre.compile``. The
rule builders with thousands of regexes (roster.person_rules, executor._org_rule) return ``Deferred`` —
compiled on first use, exactly as ``re.compile`` would have done; ``compile_all`` compiles a batch of them
before the engine starts: the parse and the code in forked workers (engine_adapter.fork_map), the pattern
objects here from the same code — equal to ``re.compile`` (``Pattern ==`` compares the pattern, the flags
and the code). A rule built and dropped (a pair rule of a surname without a given name) is never compiled."""
from __future__ import annotations

import re

try:                                    # the re internals of CPython (3.11+)
    import _sre
    from re import _compiler, _parser
except ImportError:                     # pragma: no cover — another implementation: compiled here
    _sre = _compiler = _parser = None

_BATCH = 64                 # patterns per pool task
_POOL_MIN = 256             # fewer deferred patterns — compiled here


class Deferred:
    """A regex compiled on first use; any attribute of ``re.Pattern`` is that of the compiled one."""

    __slots__ = ("source", "flags_in", "_rx")

    def __init__(self, pattern: str, flags: int = 0):
        self.source, self.flags_in, self._rx = pattern, int(flags), None

    def compiled(self) -> re.Pattern:
        if self._rx is None:
            self._rx = re.compile(self.source, self.flags_in)
        return self._rx

    def __getattr__(self, name):
        if name in Deferred.__slots__:          # not set yet (a copy being built): no recursion
            raise AttributeError(name)
        return getattr(self.compiled(), name)

    def __repr__(self) -> str:
        return f"Deferred({self.source!r}, {self.flags_in})"


def compile(pattern: str, flags: int = 0) -> Deferred:  # noqa: A001 — the re.compile of the builders
    return Deferred(pattern, flags)


def _codes(_state, batch: list) -> list:
    """(flags, code, groups, groupindex, indexgroup) of each (pattern, flags) — the arguments ``re`` gives
    ``_sre.compile``; None — the pattern does not compile (re.compile raises the same error later, here)."""
    out = []
    for pattern, flags in batch:
        try:
            p = _parser.parse(pattern, flags)
            code = [int(c) for c in _compiler._code(p, flags)]     # opcodes are int subclasses, not picklable
        except Exception:               # noqa: BLE001
            out.append(None)
            continue
        groupindex = dict(p.state.groupdict)
        indexgroup = [None] * p.state.groups
        for k, i in groupindex.items():
            indexgroup[i] = k
        out.append((flags | p.state.flags, code, p.state.groups - 1, groupindex, tuple(indexgroup)))
    return out


def compile_all(deferred: list, workers: int) -> int:
    """Compiles the not yet compiled ``Deferred`` of the list over ``workers`` forked processes (fewer
    than ``_POOL_MIN``, one worker, no fork — they stay compiled on first use); returns how many patterns
    the pool compiled."""
    todo = [d for d in deferred if isinstance(d, Deferred) and d._rx is None]
    if workers < 2 or len(todo) < _POOL_MIN or _sre is None:
        return 0
    from . import engine_adapter as EA
    keys = list(dict.fromkeys((d.source, d.flags_in) for d in todo))
    batches = [keys[i:i + _BATCH] for i in range(0, len(keys), _BATCH)]
    got = EA.fork_map(_codes, batches, workers, weight=lambda b: sum(len(p) for p, _f in b),
                      label="rules:compile")
    made: dict = {}
    for batch, outs in zip(batches, got):
        for key, res in zip(batch, outs):
            if res is not None:
                made[key] = _sre.compile(key[0], *res)
    first = next(iter(made), None)
    if first is None or made[first] != re.compile(*first):    # another re: compiled here, on first use
        return 0
    for d in todo:
        rx = made.get((d.source, d.flags_in))
        if rx is not None:
            d._rx = rx
    return len(made)


def resolve(rules: list, workers: int = 1) -> None:
    """The rules' ``Deferred`` patterns → compiled ``re.Pattern`` (a batch over ``workers`` processes,
    the rest here): the engine sees only real patterns."""
    compile_all([r.pattern for r in rules], workers)
    for r in rules:
        if isinstance(r.pattern, Deferred):
            r.pattern = r.pattern.compiled()
