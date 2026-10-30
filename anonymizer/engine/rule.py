"""The Rule schema and the core rule-application loop.

Every detector module exposes an ordered ``RULES: list[Rule]``. A Rule is a
declarative record; all I/O (walking, DRY, counting, format dispatch) lives in
the engine, so detectors stay pure and testable.

Ordering discipline (encoded by list order): most-specific/self-validating
first, longest literals first, geo-forms before person-forms, full pairs before
halves. The engine applies rules top-to-bottom; each sees the text left by the
previous one.
"""
from __future__ import annotations

import hashlib
import heapq
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .context import ctx_reach, has_context, left_hit, right_hit
from . import ac

try:                                   # 3.11+: sre_parse moved to re._parser
    from re import _parser as _sre
except ImportError:                    # pragma: no cover
    import sre_parse as _sre           # type: ignore

# --- PREFILTER BY REQUIRED LITERALS --------------------------------------------
# The alpha pack is ~10 thousand separate regexes; apply_rules ran EACH over the whole text
# of a file (10k × len(text) per pass, 2–3 passes) — about a minute per transcript, while
# a given file contains dozens of literals out of thousands. The prefilter:
# from the PARSED regex (re._parser, not a string heuristic) a set of
# probes is derived — literal substrings, at least one of which MUST be present
# in the text for the regex to match. No probe in the text → the regex
# cannot match → the rule is skipped without scanning. Any node for
# which a required literal cannot be proven (a character class, ANY, an empty
# alternative, backref, an unknown op) → None → the rule is NOT filtered.
# The output is byte for byte the same (only rules that would have found nothing
# anyway are skipped). ANON_PREFILTER=0 — turn off; ANON_PREFILTER_VERIFY=1 —
# self-check mode: every skip is confirmed by pattern.search(text) is
# None, a mismatch = SystemExit (tests, oracle).
#
# Incremental prefilter. Before, after EACH rule that changed
# the text, the sets of present probes were reset and the next rule with
# probes rescanned the WHOLE text (a study: 364 full scans =
# 87 % of time on the production model). Now repl accumulates replacement spans, and after sub
# the automaton runs only over WINDOWS around the replaced parts of the new text
# (± max probe length): a new probe can appear only crossing a replaced
# part, so the window sees it; a vanished probe stays in the set —
# that is over-inclusion, it only cancels a rule skip, the output does not change.
# The sets are monotonic, so PrefilterState is carried across the passes
# of the fixpoint (pipeline._apply_fixpoint) without rebuilding. ANON_PREFILTER_INCR=0
# — the old full recount (bench tools/bench_prefilter.py, parity tests).
# The superset has a cost (bench profile): a probe that vanished under a token
# keeps the rule in work — an extra sub of a literal rule with a lookbehind
# prefix costs ~15 ms on 593K (271 extra subs = +2 s, more than the saved
# scans on the C backend). So before sub a probe is confirmed by an exact `in`
# over the text (a casefold cache per text version) — 0.05–0.2 ms per probe; for
# sets of ≤ _EXACT_MAX probes this is cheaper than one sub, large sets (the name
# gazetteer, TLD) stay on one automaton.
#
# Rule.needs. Rules without derivable probes (53 of 112 built-in:
# the pattern starts with a lookbehind, a class, `\d{10}`) ran a full re pass
# on every fixpoint pass. A detector declares a cheap NECESSARY condition
# of a REPLACEMENT: a string/tuple of strings (at least one must be in the text — they go into the
# same automata as probes, incrementally) or an re.Pattern (search once per
# text version; True is monotonic, False is reset on a replacement). The condition
# is checked over left_ctx+text: the requires_context gate in the stream sees
# the tail of previous lines, so needs=<ctx-regex> is legal for it too. No
# condition in the text ⇒ the rule cannot replace anything ⇒ sub is not called.
# VERIFY: a skip by needs is confirmed by running the rule — 0 replacements, otherwise
# SystemExit. ANON_PREFILTER_NEEDS=0 — turn needs off (bench/tests).
_PREFILTER_ON = os.environ.get("ANON_PREFILTER", "1") != "0"
_PREFILTER_VERIFY = os.environ.get("ANON_PREFILTER_VERIFY", "0") == "1"
_INCR_ON = os.environ.get("ANON_PREFILTER_INCR", "1") != "0"
_NEEDS_ON = os.environ.get("ANON_PREFILTER_NEEDS", "1") != "0"
_PROBE_MIN = 3          # shorter — almost always in the text, the filter is useless
_EXACT_MAX = 16         # a probe set no larger than this — confirmed by an exact `in` before sub
# counters for the bench/tests (cheap; not reset automatically)
STATS = {"probe_skip": 0, "exact_skip": 0, "needs_skip": 0,
         "scan_full": 0, "scan_win": 0, "local_sub": 0, "full_sub": 0}
# re-needs recheck full / by windows (separately: STATS is checked against the baseline by tests)
NEEDS_STATS = {"needs_full": 0, "needs_win": 0}


# --- [её] CLASSES AND THE WHITESPACE CLASS IN PROBES ------------------------------
# The packgen pack writes «е»/«ё» after the third letter of a form as the class [её] (ANON_YO_FOLD),
# and the space of a multi-word form as the class [^\S\n]+ (ANON_PACKGEN_WS_CLASS). The class
# cut the literal from which probes and start literals are derived: «Сем[её]нов»
# → probe «сем», «Сергей[^\S\n]+Петров» → one part of the pair; the rule ran over
# batches without the name (a study: point subs on apply ×6.9, apply
# 286 → 406 s). A rule whose parse met such a class
# gets probes and start literals in its own "alphabet" (y): a class of letters
# equal after ё→е folding (for an IGNORECASE rule — after _fold) reads as
# that letter; a class of horizontal spaces only ([^\S\n], [ \t], a repeat
# min ≥ 1 — [^\S\n]+) reads as one space in a probe (_ws_squeeze — only in
# the probe itself; see below). Soundness: a regex match ⇒ the folded text has
# the folded probe (the class takes only characters that folding turns into the same letter /
# into a run). The pattern, the pack and rule_digest do not change. VERIFY (ANON_PREFILTER_VERIFY=1)
# checks every skip and point sub.
# y-ALPHABET WITHOUT LIVE TEXT COPIES. The bench profile (synthetic "like
# Telegram", a batch of ~1.3 million characters, thousands of pack hits per pass): the cost of the classes is not
# the folds themselves, but two extra "live" y-alphabets in the fold_cache of apply_rules and second windows
# of the increment with folding of space runs. Therefore:
#  — start literals of ALL IGNORECASE rules are folded with _yo_ci (as for y-rules), and
#    are searched in one alphabet 1 — fold_cache holds one text fold (_yo_ci); for
#    a rule without classes with «ё» in a literal there are slightly more candidates (match decides) — the output
#    is the same; case-sensitive y-rules (a handful) — alphabet 2;
#  — y-probes without folding space runs: a multi-word probe («сергей петров») is a
#    conjunction of parts (each must be in the text, the same automata and the same presence
#    sets), a one-word one — as before; the text for y-probes is only ё→е;
#  — increment windows are shared by all four alphabets (± max probe width).
# Soundness: a conjunction of parts is weaker than an exact probe (a superset), point
# sub candidates are a superset; VERIFY checks every skip and point sub. Probes and start
# literals seen by rule_probes/rule_starts (reporting, tests) are folded.
# The PROBE_CLASS_FOLD and PROBE_CLASS_MERGE switches are removed
# together with the old paths (_advance_y, _walk_left/_walk_right, folding of text runs).
_HWS_RUN = re.compile(r"[^\S\n]{2,}|[^\S \n]")     # a run of ≥ 2 or a single non-space


def _ws_squeeze(s: str) -> str:
    """Every run of horizontal whitespace characters ([^\\S\\n]) → one space."""
    return _HWS_RUN.sub(" ", s)


def _yo_raw(s: str) -> str:
    """ё→е, Ё→Е (the case-sensitive y-alphabet): per character, the same length."""
    return s.replace("ё", "е").replace("Ё", "Е")


def _yo_ci(s: str) -> str:
    """_fold + ё→е (the IGNORECASE y-alphabet): per character, the same length."""
    return _fold(s).replace("ё", "е")


def _hws_class(av) -> bool:
    """An IN node takes only horizontal spaces: [^\\S\\n…] or a set of
    space literals without a line break."""
    items = list(av)
    if items and items[0][0] is _sre.NEGATE:
        rest = items[1:]
        notspace = any(o is _sre.CATEGORY and a is _sre.CATEGORY_NOT_SPACE
                       for o, a in rest)
        nl = any((o is _sre.LITERAL and a == 10)
                 or (o is _sre.RANGE and a[0] <= 10 <= a[1]) for o, a in rest)
        return notspace and nl
    return bool(items) and all(o is _sre.LITERAL and a != 10 and chr(a).isspace()
                               for o, a in items)


class _ClassReader:
    """Reading character classes in the parse (see the block above). ``used`` —
    the rule got into the y-alphabet: at least one class was read."""
    __slots__ = ("ci", "ws", "used")

    def __init__(self, ci: bool, ws: bool):
        self.ci, self.ws, self.used = ci, ws, False

    def char(self, av):
        """IN node → a letter (a class of letters equal after ё→е folding) | " " (a class
        of horizontal spaces, for probes only) | None."""
        if av and all(o is _sre.LITERAL for o, _a in av):
            fold = _yo_ci if self.ci else _yo_raw
            got = {fold(chr(a)) for _o, a in av}
            if len(got) == 1:
                c = got.pop()
                if len(c) == 1 and fold(c) == c:
                    self.used = True
                    return c
        if self.ws and _hws_class(av):
            self.used = True
            return " "
        return None

    def ws_run(self, av) -> bool:
        """A repeat min ≥ 1 of one space node (a literal or a class) — one space of a probe."""
        if not self.ws:
            return False
        lo, _hi, sub = av
        sub = list(sub)
        if lo < 1 or len(sub) != 1:
            return False
        op, a = sub[0]
        ok = (op is _sre.LITERAL and a != 10 and chr(a).isspace()) or \
            (op is _sre.IN and _hws_class(a))
        if ok:
            self.used = True
        return ok


def _leading_literal(seq, rd=None) -> str:
    """Literal prefix of an sre sequence (empty if it does not start
    with a literal). Contiguous with what stands left of the sequence.
    ``rd`` — _ClassReader: a read class is a literal too."""
    out = []
    for op, av in seq:
        if op is _sre.LITERAL:
            out.append(chr(av))
        elif rd is not None and op is _sre.IN and (c := rd.char(av)) is not None:
            out.append(c)
        elif rd is not None and op in _REPEAT_OPS and rd.ws_run(av):
            out.append(" ")
        else:
            break
    return "".join(out)


_REPEAT_OPS = tuple(o for o in (_sre.MAX_REPEAT, _sre.MIN_REPEAT,
                                getattr(_sre, "POSSESSIVE_REPEAT", None)) if o is not None)


def _alt_probes(seq, ci: bool, min_len: int = _PROBE_MIN, rd=None):
    """List of probes for a sequence of sre nodes (one of the probes must
    be in the text) or None (cannot be proven). ci — case-insensitive.

    Candidates: (a) the longest required literal piece of the top
    level; (b) for a branch standing RIGHT after a literal piece (sre
    factors out the common prefix of an alternation: «Ne|Nea» → «Ne» + BRANCH(«», «a»)),
    — the full variants "prefix + literal start of the branch" (contiguous,
    hence sound; a call batch: the probe «ne» is in any text,
    while «Mariya»/«Smirnova» are not). The most selective candidate is taken
    (the maximum length of the shortest probe).

    ``min_len`` — the length threshold of a SINGLE probe (truncation by usefulness, not by
    soundness). For an alternation BRANCH the threshold is 2: a short branch («Ян»
    in the name gazetteer, «ИП» among legal forms, «AC|SK» twilio) used to zero the probe
    set of the whole rule; the selectivity of the set is decided by the top level — a set
    of many probes is accepted, a single short probe still is not.
    A one-letter branch («7|8» phone_ru, «г|с|д» city) zeroes the set
    now too: such a probe is in any text, and each of its hits is work
    for the automaton.

    ``rd`` — _ClassReader: a read class continues the literal
    piece (as a letter or a space) instead of cutting it."""
    chunks: list = []
    cur: list = []
    candidates: list = []        # probe sets from branching required nodes

    def flush():
        nonlocal cur
        if cur:
            chunks.append("".join(cur))
            cur = []

    def _branch_alts(alts):
        """(full probes "prefix + literal start of the branch" — contiguous and
        non-empty with a non-empty prefix; probes of branches without a prefix — None if
        at least one branch has no required literal)."""
        prefix = "".join(cur)
        full = [prefix + _leading_literal(alt, rd) for alt in alts] if prefix else None
        inner: list = []
        for alt in alts:
            pr = _alt_probes(alt, ci, 2, rd)
            if not pr:
                inner = None
                break
            inner.extend(pr)
        return full, inner

    for op, av in seq:
        if op is _sre.LITERAL:
            cur.append(chr(av))
        elif rd is not None and op is _sre.IN and (c := rd.char(av)) is not None:
            cur.append(c)            # [её] — a letter, [^\S\n] — a space
        elif rd is not None and op in _REPEAT_OPS and rd.ws_run(av):
            cur.append(" ")          # [^\S\n]+ — a space
        elif op is _sre.SUBPATTERN:
            add_flags, sub = av[1], av[-1]   # av = (group, add, del, p)
            if add_flags & re.IGNORECASE and not ci:
                flush()                      # a local (?i) — not proven
                continue
            # a group of one branch is transparent: its literal start
            # is contiguous with the current piece
            if len(sub) == 1 and sub[0][0] is _sre.BRANCH:
                full, inner = _branch_alts(sub[0][1][1])
            else:
                full, inner = _branch_alts([sub])
            flush()
            if full:
                candidates.append(full)
            if inner:
                candidates.append(inner)
        elif op is _sre.BRANCH:
            full, inner = _branch_alts(av[1])
            flush()
            if full:
                candidates.append(full)
            if inner:
                candidates.append(inner)
        elif op in (_sre.MAX_REPEAT, _sre.MIN_REPEAT) or \
                (hasattr(_sre, "POSSESSIVE_REPEAT")
                 and op is _sre.POSSESSIVE_REPEAT):
            lo, _hi, sub = av
            flush()
            if lo >= 1:
                sub_pr = _alt_probes(sub, ci, _PROBE_MIN, rd)
                if sub_pr:
                    candidates.append(sub_pr)
        elif hasattr(_sre, "ATOMIC_GROUP") and op is _sre.ATOMIC_GROUP:
            flush()
            sub_pr = _alt_probes(av, ci, _PROBE_MIN, rd)
            if sub_pr:
                candidates.append(sub_pr)
        elif op in (_sre.IN, _sre.ANY, _sre.NOT_LITERAL, _sre.CATEGORY,
                    _sre.RANGE, _sre.AT, _sre.ASSERT, _sre.ASSERT_NOT,
                    _sre.GROUPREF, _sre.GROUPREF_EXISTS):
            flush()                  # a character/position without a known literal
        else:
            return None              # an unknown node — not filtered
    flush()
    best = max(chunks, key=len) if chunks else ""
    if len(best) >= min_len:
        candidates.append([best])
    good = [c for c in candidates if c and all(c)]
    if not good:
        return None
    # the most selective set: the one whose shortest probe is longer than all others
    pick = max(good, key=lambda c: (min(len(p) for p in c), -len(c)))
    if min(len(p) for p in pick) < min_len and len(pick) == 1:
        return None
    return [_fold(p) if ci else p for p in pick]


# Parsing a pattern is a pure function (pattern, flags); a per-process cache: the round
# oracle and registry preparation build the pipeline anew (new Rule objects from
# the same pack) — without a cache 43 thousand patterns of the tg pack would be parsed again
# (~9 s per build). Pool workers inherit the cache from the parent (prebuild_rules).
_PARSE_CACHE: dict = {}


def _analyze(pattern):
    """(probes | None, start literals | None, probes in the y-alphabet, start
    literals in the y-alphabet) of a pattern — one sre parse (y — a class was read
    in the parse, ё→е and space folding, see the block above)."""
    key = (pattern.pattern, pattern.flags)
    hit = _PARSE_CACHE.get(key)
    if hit is not None:
        return hit
    probes = None
    parsed = None
    ci = bool(pattern.flags & re.IGNORECASE)
    rd = _ClassReader(ci, ws=True)
    try:
        parsed = _sre.parse(pattern.pattern, pattern.flags)
        probes = _alt_probes(list(parsed), ci, _PROBE_MIN, rd)
        if probes is not None and ci:
            probes = [_fold(p) for p in probes]
    except Exception:               # noqa: BLE001 — could not parse: not filtered
        probes = None
    py = rd.used
    pf = probes                 # search structure (parts of y-probes)
    if probes is not None and py:
        probes = [_ws_squeeze(_yo_ci(p) if ci else _yo_raw(p)) for p in probes]
        pf = _probe_parts(probes)
    rs = _ClassReader(ci, ws=False)
    starts = _starts_of(parsed, ci, rs) if parsed is not None else None
    # no start literal — anchor literals of the tail (literals, off, y) | None
    anch = _anchor_of(parsed, ci) if (starts is None and parsed is not None) else None
    hit = (probes, starts, py, rs.used, pf, anch)
    _PARSE_CACHE[key] = hit
    return hit


def _probe_parts(probes):
    """y-probes for search without folding space runs — a tuple
    of alternatives, an alternative is a tuple of parts (words of the folded probe of length ≥ 2; each
    must be in the text). An alternative without parts (only short words) — absence cannot be
    proven: None (the rule is not filtered by probes, as with probes=None)."""
    alts = []
    for p in probes:
        parts = tuple(w for w in p.split(" ") if len(w) >= 2)
        if not parts:
            return None
        alts.append(parts)
    return tuple(alts)


def rule_probes(rule):
    """Probes of a rule (cached on the object): None — the rule is not filtered."""
    try:
        return rule._probes
    except AttributeError:
        pass
    probes, starts, py, sy, pf, anch = _analyze(rule.pattern)
    rule._probes = probes
    rule._probes_ci = bool(rule.pattern.flags & re.IGNORECASE)
    rule._probes_y = py             # probes are in the y-alphabet (ё→е and space folding)
    # what the automata search for — y-probes by parts (a tuple of tuples) or the probes themselves;
    # None — not filtered by probes
    rule._probes_f = pf
    if not hasattr(rule, "_starts"):
        # the same parse — without a second parse; local_sub=False — a full sub
        local = getattr(rule, "local_sub", True)
        rule._starts = starts if local else None
        rule._starts_ci = rule._probes_ci
        rule._starts_y = sy         # start literals are in the ё→е fold
        # no start literals — anchor literals of the tail (literals, off,
        # literals in the ё→е fold) | None; rule_starts does not show them (start ones — as before)
        rule._anchor = anch if (local and starts is None) else None
    return probes


def rule_anchor(rule):
    """Anchor literals of a rule: (literals, off, y) — any
    match contains one of the literals no further than off characters from the start; None — none (there are
    start literals, the head is unbounded, local_sub=False)."""
    try:
        return rule._anchor
    except AttributeError:
        pass
    rule_probes(rule)
    return rule._anchor


def _local_lits(rule):
    """(point-sub literals, off, literals in the ё→е fold): start ones (off 0) or anchor ones;
    (None, 0, False) — the point path is unavailable."""
    st = rule_starts(rule)
    if st is not None:
        return st, 0, getattr(rule, "_starts_y", False)
    a = rule._anchor
    if a is not None:
        return a
    return None, 0, False


@dataclass
class Rule:
    name: str
    pattern: re.Pattern
    mask: Callable[[re.Match], str]          # match -> replacement string
    value_group: int = 0                      # group holding the sensitive value
    validator: Optional[Callable[[str], bool]] = None   # True => real => mask
    requires_context: Optional[re.Pattern] = None
    window: int = 120
    ctx_right: int = 0                        # right context window within the line; 0 = left only
    guard: Optional[Callable[[re.Match], bool]] = None   # True => leave untouched
    confidence: str = "high"                  # high | context | low
    scope: Optional[set] = None               # rel-paths this rule is limited to
    kind: str = "regex"                       # regex | value | csv_column | token_remap
    note: str = ""
    # a rule match may legitimately contain a line break (a PEM block).
    # None = derive from the pattern (rule_spans_lines), True/False — explicitly.
    multiline: Optional[bool] = None
    # a cheap NECESSARY replacement condition (see the prefilter block above):
    # str / tuple of str — at least one substring must be in the text (for
    # a rule with global IGNORECASE the comparison is by casefold, otherwise exact —
    # declare a local (?i:…) as a regex); re.Pattern — search over
    # left_ctx+text (a regex necessity or the rule's requires_context).
    # None — no condition, the rule is not filtered by needs (as before).
    needs: Optional[object] = None
    # point sub (see the "POINT SUB" block) — the output is the same, so it is not part of
    # the behaviour fingerprint (rule_digest). False — only a full sub:
    # for a rule whose mask/guard/validator reads m.pos/m.endpos (on the point
    # path m.pos is the candidate position, not 0).
    local_sub: bool = True

    def _value(self, m: re.Match) -> str:
        # fail loudly: a value_group outside the pattern geometry is
        # rule damage (typical for an external packgen pack with value_group=2 on
        # a 1-group template). A silent fallback to m.group(0) fed the validator/
        # requires_context/seed the WRONG span — a silent recall loss without diagno-
        # stics. Let a bad index fail the run with the rule name.
        try:
            return m.group(self.value_group) if self.value_group else m.group(0)
        except (IndexError, error) as e:  # noqa
            raise SystemExit(f"rule {self.name}: value_group={self.value_group} "
                             f"outside the pattern geometry ({e})")


error = re.error


def rule_needs(rule):
    """Normalized Rule.needs (cached on the object): None — no condition;
    ("s", frozenset(strings), ci) — needs strings (casefold with ci: they go into
    the same automata as probes); ("re", pattern) — a regex. Garbage in needs —
    SystemExit (fail loudly: a silent None = a loss of speed-up without a trace)."""
    try:
        return rule._needs
    except AttributeError:
        pass
    nd = getattr(rule, "needs", None)
    out = None
    if nd is not None:
        if isinstance(nd, re.Pattern):
            out = ("re", nd)
        else:
            items = (nd,) if isinstance(nd, str) else nd
            try:
                items = tuple(items)
            except TypeError:
                items = ()
            if not items or not all(isinstance(s, str) and s for s in items):
                raise SystemExit(
                    f"rule {rule.name}: needs — a non-empty string, a tuple "
                    f"of non-empty strings or re.Pattern, got {nd!r}")
            ci = bool(rule.pattern.flags & re.IGNORECASE)
            out = ("s", frozenset(_fold(s) if ci else s for s in items), ci)
    rule._needs = out
    return out


_AC_CACHE = {}          # id(rules) -> (rules, ac_ci, ac_raw, max_ci, max_raw); the last list


def _rule_automata(rules):
    """Two Aho-Corasick automata from the probes of ALL rules: a single scan
    instead of per-rule `p in text` (9.4k rules × pass → 87% of time). ci probes
    (already folded by _fold in rule_probes) are searched in _fold(text) — casefold + «ı→i»,
    the way re compares under IGNORECASE, raw ones — in the text:
    the probe sets match exactly, without over-inclusion. Rules without probes
    (rule_probes=None) do NOT go into the automata and are NOT filtered. needs strings
    go into the same automata. Returns: (ac_ci, ac_raw, max_ci,
    max_raw) — the maximum probe lengths set the width of increment windows.
    Cached by list identity (self.rules is stable → built once per
    run)."""
    hit = _AC_CACHE.get(id(rules))
    if hit is not None and hit[0] is rules:
        return hit[1:5]
    ci, raw = set(), set()
    yraw, yci = set(), set()        # probes of the y-alphabets (folded text)
    for r in rules:
        p = rule_probes(r)
        if p is not None and r._probes_f is not None:
            pf = r._probes_f
            if r._probes_y:
                if pf is not p:          # parts of y-probes
                    pf = [w for alt in pf for w in alt]
                (yci if r._probes_ci else yraw).update(pf)
            else:
                (ci if r._probes_ci else raw).update(p)
        nd = rule_needs(r)
        if nd is not None and nd[0] == "s":
            (ci if nd[2] else raw).update(nd[1])
    ac_ci = ac.Automaton(sorted(ci)) if ci else None
    ac_raw = ac.Automaton(sorted(raw)) if raw else None
    yent = None
    if yraw or yci:
        yent = (ac.Automaton(sorted(yraw)) if yraw else None,
                ac.Automaton(sorted(yci)) if yci else None,
                max(map(len, yraw | yci)))
    ent = (rules, ac_ci, ac_raw,
           max(map(len, ci), default=0), max(map(len, raw), default=0), yent)
    _AC_CACHE.clear()
    _AC_CACHE[id(rules)] = ent
    return ent[1:5]


def _rule_automata_y(rules):
    """(y-raw automaton | None, y-ci automaton | None, max y-probe length) |
    None — no y-probes. The same cache as _rule_automata."""
    _rule_automata(rules)
    return _AC_CACHE[id(rules)][5]


# --- INVERTED PROBE INDEX — ITERATING ONLY CANDIDATES -------------------------
# A pack of a Google Docs export — 84 thousand rules (dictionary brands + the pack); apply_rules
# iterated over ALL of them on every call, though on a short batch only a hundred have probes: the check
# "the probe is absent" (_absent: isdisjoint over the presence set) — 2–3 µs per rule, i.e.
# ~0.25 s per call at any text size; on a file cut into batches of records
# (pipeline._masked_chunks) this became the main cost. The index: probe → rule numbers (by
# probe alphabet, as in _absent/_absent_parts), plus the numbers of rules that probes do not
# filter out (no probes derived). The candidates of a call are these rules and the rules of at least one
# present probe; a probe that first appeared after a replacement (_advance) adds its
# rules further down the list. The order is by ascending number, the loop body is the same (tier, scope,
# _absent with an exact `in`, needs, point/full sub): a rule outside the candidates is exactly one
# for which _absent would return True (none of its probes is in the presence set at the moment
# of the check), i.e. the output, report, trace and clean_stop are the same. The only difference is diagnostics:
# STATS["probe_skip"] does not count rules outside the candidates; presence sets are computed
# at once for all alphabets that have probes (before — on first need; a superset, like
# the increment windows). Not enabled: fewer rules than _IDX_MIN (iteration is cheaper), VERIFY (checking
# every skip), ANON_PREFILTER_INCR=0 (the sets are reset after a replacement).
# The threshold is 4096 rules: the built-in set (~1.8 thousand) is iterated in ~3 ms per call, while the index on it is
# ~0.8 MB of objects per rule list (tests/test_streaming.py measures the stream peak together with building
# the prefilter automata); client packs are 10–85 thousand rules, there iteration is 25–250 ms per call.
_IDX_MIN = 4096
_IDX_CACHE = {}         # id(rules) -> (rules, always, by); the last list, like _AC_CACHE


def _probe_index(rules):
    """(always, by): always — numbers of regex/token_remap rules that probes do not filter out
    (_probes_f is None), ascending; by[alpha] — {probe: [rule numbers]} by the alphabet
    of the check (_absent: ci + 2·y; _absent_parts — parts of y-probes, ci + 2)."""
    hit = _IDX_CACHE.get(id(rules))
    if hit is not None and hit[0] is rules:
        return hit[1], hit[2]
    always: list = []
    by = ({}, {}, {}, {})
    for i, r in enumerate(rules):
        if r.kind not in ("regex", "token_remap"):
            continue                    # apply_rules skips them (value/csv_column)
        p = rule_probes(r)
        pf = r._probes_f
        if pf is None:
            always.append(i)
            continue
        if pf is p:
            d, items = by[r._probes_ci + 2 * r._probes_y], p
        else:
            d, items = by[r._probes_ci + 2], {w for alt in pf for w in alt}
        for s in set(items):
            d.setdefault(s, []).append(i)
    by = tuple({s: tuple(v) for s, v in d.items()} for d in by)   # without spare list capacity
    _IDX_CACHE.clear()
    _IDX_CACHE[id(rules)] = (rules, always, by)
    return always, by


class _Cands:
    """Iterating the candidates of a pass (see the block above): (number, rule) by ascending number —
    a merge of ``always`` and a heap of rules of present probes; ``grow`` — a probe first appeared
    in the set (_advance): its rules after the current one become candidates."""
    __slots__ = ("rules", "always", "ai", "heap", "seen", "by", "cur")

    def __init__(self, rules, always, by, present):
        self.rules, self.always, self.by = rules, always, by
        self.ai = 0
        self.cur = -1
        seen = bytearray(len(rules))
        heap = []
        for alpha, got in present:
            d = by[alpha]
            for p in got:
                for i in d.get(p, ()):
                    if not seen[i]:
                        seen[i] = 1
                        heap.append(i)
        heapq.heapify(heap)
        self.heap, self.seen = heap, seen

    def __iter__(self):
        return self

    def __next__(self):
        h, al, ai = self.heap, self.always, self.ai
        a = al[ai] if ai < len(al) else None
        if h and (a is None or h[0] < a):
            i = heapq.heappop(h)
        elif a is not None:
            i = a
            self.ai = ai + 1
        else:
            raise StopIteration
        self.cur = i
        return i, self.rules[i]

    def grow(self, alpha, probes) -> None:
        d, seen, cur, h = self.by[alpha], self.seen, self.cur, self.heap
        for p in probes:
            for i in d.get(p, ()):
                if i > cur and not seen[i]:
                    seen[i] = 1
                    heapq.heappush(h, i)


def _candidates(rules, st, acs, text, left_ctx):
    """_Cands for an apply_rules call | None — the old iteration over all rules."""
    if len(rules) < _IDX_MIN or _PREFILTER_VERIFY or not _INCR_ON:
        return None
    always, by = _probe_index(rules)
    present = [(a, _present(st, a, acs, text, left_ctx)) for a in range(4) if by[a]]
    return _Cands(rules, always, by, present)


class PrefilterState:
    """Prefilter state of ONE text: sets of present
    probes (a superset of the true presence — monotonic), the re-needs cache,
    the text for which the state is current. Created in apply_rules or
    once in _apply_fixpoint and passed to every pass (prefilter=st):
    passes 2/3 do not rescan the text. The binding to the text is checked —
    a state from another text = SystemExit (otherwise rules are silently skipped)."""
    __slots__ = ("raw", "ci", "needs", "text", "full", "low", "rawy", "ciy", "yhay",
                 "flog", "nfrom", "fspans", "fabs", "wcache")

    def __init__(self):
        self.raw = None         # set of raw-automaton probes | None (not scanned yet)
        self.ci = None          # set of ci-automaton probes (casefold) | None
        self.needs = {}         # re-needs -> bool; False is reset on a replacement
        self.text = None        # the text after the last apply_rules with this state
        self.full = None        # left_ctx + text (lazily; text itself is without left_ctx)
        self.low = None         # _fold(full) (lazily, per text version)
        self.rawy = None        # set of y-raw-automaton probes | None
        self.ciy = None         # set of y-ci-automaton probes | None
        self.yhay = None        # [y-raw, y-ci] folded full (lazily, per version)
        # the log of replacements since the re-needs were
        # confirmed — [spans of the fired rule] in firing order (coordinates
        # of the text version BEFORE that rule); nfrom[pat] — the log length at the last
        # confirmation of False for this pattern (see _needs_ok)
        self.flog = []
        self.nfrom = {}         # pat -> (len(flog), fspans) at the confirmation of False
        self.fspans = 0         # total replacements in the log (from the start / since the last reset)
        self.fabs = []          # fabs[j] — sum of max(len(out), len(in)) of log replacements up to j inclusive
        self.wcache = None      # ((len(flog), start), windows) — windows of the current text version


# probe alphabets: 0 — raw, 1 — ci (as before); 2/3 — y-raw/y-ci
_ALPHA_ATTR = ("raw", "ci", "rawy", "ciy")


def _haystack(st, ci, text, left_ctx):
    """left_ctx+text (casefold with ci) — cached per text version: one casefold
    per replacement instead of one per rule. ``ci`` — alphabet 0/1 (False/True) or
    2/3 (y-raw/y-ci — ё→е folding and folding of space runs)."""
    if st.full is None:
        st.full = left_ctx + text if left_ctx else text
        st.low = None
        st.yhay = None
    if ci >= 2:
        if st.yhay is None:
            st.yhay = [None, None]
        h = st.yhay[ci - 2]
        if h is None:                   # only ё→е, space runs are not folded
            h = (_haystack(st, True, text, left_ctx).replace("ё", "е")
                 if ci == 3 else _yo_raw(st.full))
            st.yhay[ci - 2] = h
        return h
    if not ci:
        return st.full
    if st.low is None:
        st.low = _fold(st.full)     # casefold + «ı→i» — the way re compares
    return st.low


def _present(st, ci, acs, text, left_ctx):
    """The set of probes present in left_ctx+text (lazily, one full
    scan per text and alphabet; after that — only increment windows). ``acs`` —
    automata by alphabet (raw, ci, y-raw, y-ci)."""
    attr = _ALPHA_ATTR[ci]
    got = getattr(st, attr)
    if got is None:
        STATS["scan_full"] += 1
        a = acs[ci]
        got = a.found(_haystack(st, ci, text, left_ctx)) if a is not None else set()
        setattr(st, attr, got)
    return got


def _absent(st, items, ci, acs, text, left_ctx, exact=True) -> bool:
    """None of the strings ``items`` is in left_ctx+text: first the automaton
    (a superset, O(1)), then for a small set — an exact `in` over the
    cached text (removes the over-inclusion of the superset before sub).
    ``exact=False`` — the automaton only: the rule goes by point sub,
    which searches for occurrences itself (an exact `in` would be a second text scan)."""
    present = _present(st, ci, acs, text, left_ctx)
    if present.isdisjoint(items):
        return True
    if not exact or len(items) > _EXACT_MAX:
        return False
    hay = _haystack(st, ci, text, left_ctx)
    for p in items:
        if p in hay:
            return False
    STATS["exact_skip"] += 1
    return True


def _absent_parts(st, alts, ci, acs, text, left_ctx, exact=True) -> bool:
    """No alternative of a y-probe (a tuple of parts) can be in the
    text — every alternative has at least one part missing from the presence set; for
    a small set — also an exact `in` of each part over the folded text (as _absent)."""
    present = _present(st, ci, acs, text, left_ctx)
    live = [alt for alt in alts if all(p in present for p in alt)]
    if not live:
        return True
    if not exact or len(alts) > _EXACT_MAX:
        return False
    hay = _haystack(st, ci, text, left_ctx)
    for alt in live:
        if all(p in hay for p in alt):
            return False
    STATS["exact_skip"] += 1
    return True


# --- WINDOWED RECHECK OF re-needs ---------
# A re-needs with value False was reset on EVERY replacement (_advance), and the next
# pass of rules searched each such pattern again over the whole text: on a Telegram batch
# (~1 MB, ~70 rules with regex-needs, thousands of pack hits per pass) this is 20 % of the time
# of dry (bench profile: _needs_ok 7.7 s of 36). A pattern match with
# bounded width W (and bounded assertions A) in the new text that was not
# in the old one must cross a window around a replacement: outside the windows the text is the same as it was, and the search
# there already gave False. So False stays False, and on a check the pattern is searched only in
# the windows of the replacements accumulated since the confirmation (the st.flog log: spans of each
# fired rule; window coordinates are translated through the spans of later rules —
# a shift and, inside a replaced piece, extension to its edges). The window width is W + 2A + 2
# (W — the maximum width of the pattern, A — of assertions; +2 for \b/^/$ at the edges); a pattern with
# unbounded width (\s*, .+) or more than _NEEDS_WIN_SPANS replacements since the
# confirmation — a full search, as before (more windows cost more than one search). search(full, pos, endpos): ^/\b/lookbehind
# see the real neighbours (not a slice), $ and lookahead at the right edge of a window may give a false
# True — that only cancels a skip (the rule runs and decides itself), the output is the same. The superset
# is sound in the same direction as the probe windows. The fallback to resetting False on every replacement is removed.
_NEEDS_WIN_SPANS = 4096     # more replacements since the confirmation — a full search
_NEEDS_LOG_CAP = 4096       # a longer log — reset (False → full search, as before)
_NEEDS_WIDE = 1 << 16       # a pattern/assertion width not less than this — unbounded
_NEEDS_REACH: dict = {}     # pattern -> window width | None (unbounded)


def _assert_width(seq) -> int:
    """Maximum width of assertions (lookbehind/lookahead) in an sre parse tree."""
    best = 0
    for op, av in seq:
        if op in (_sre.ASSERT, _sre.ASSERT_NOT):
            sub = av[1]
            best = max(best, sub.getwidth()[1], _assert_width(list(sub)))
        elif op is _sre.SUBPATTERN:
            best = max(best, _assert_width(list(av[-1])))
        elif op is _sre.BRANCH:
            for alt in av[1]:
                best = max(best, _assert_width(list(alt)))
        elif op in _REPEAT_OPS:
            best = max(best, _assert_width(list(av[2])))
        elif op is _sre.GROUPREF_EXISTS:
            for part in av[1:]:
                if part is not None:
                    best = max(best, _assert_width(list(part)))
        elif hasattr(_sre, "ATOMIC_GROUP") and op is _sre.ATOMIC_GROUP:
            best = max(best, _assert_width(list(av)))
        if best >= _NEEDS_WIDE:
            return best
    return best


def _needs_reach(pat):
    """Width of the re-needs pattern recheck window (W + 2A + 2) | None — unbounded."""
    try:
        return _NEEDS_REACH[pat]
    except KeyError:
        pass
    reach = None
    try:
        parsed = _sre.parse(pat.pattern, pat.flags)
        w = parsed.getwidth()[1]
        a = _assert_width(list(parsed))
        if w < _NEEDS_WIDE and a < _NEEDS_WIDE:
            reach = w + 2 * a + 2
    except Exception:               # noqa: BLE001 — could not parse: a full search
        reach = None
    _NEEDS_REACH[pat] = reach
    return reach


def _needs_windows(st, start) -> list:
    """Windows of the replacements of the log flog[start:] in the coordinates of the CURRENT text, merged and
    ascending. A replacement window [a, b) from version j is extended by D_j — the sum of
    max(len(out), len(in)) of all replacements AFTER j: no position of version j shifts
    by more than D_j (a replacement wholly to the left shifts by |the length difference|, an edge inside
    a replaced piece — no further than its edge), so the window is only wider (sound, O(1) per
    replacement). Cached by (log length, start)."""
    key = (len(st.flog), start)
    if st.wcache is not None and st.wcache[0] == key:
        return st.wcache[1]
    flog, fabs = st.flog, st.fabs
    total = fabs[-1] if fabs else 0
    wins = []
    for j in range(start, len(flog)):
        D = total - fabs[j]
        d = 0
        for s, e, k in flog[j]:
            wins.append((s + d - D, s + d + k + D))
            d += k - (e - s)
    wins.sort()
    out = []
    for a, b in wins:
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1][1] = b
        else:
            out.append([a, b])
    st.wcache = (key, out)
    return out


def _needs_ok(st, pat, text, left_ctx) -> bool:
    """re-needs: search over left_ctx+text once per text version. False with a windowed
    recheck over the replacements since the confirmation (see the block above)."""
    ok = st.needs.get(pat)
    if ok is None:
        ok = pat.search(_haystack(st, False, text, left_ctx)) is not None
        st.needs[pat] = ok
        if not ok:
            st.nfrom[pat] = (len(st.flog), st.fspans)
        return ok
    if ok:
        return ok
    start, cnt0 = st.nfrom.get(pat, (0, 0))
    n = len(st.flog)
    if start >= n:
        return False                    # no replacements since the confirmation
    hay = _haystack(st, False, text, left_ctx)
    reach = _needs_reach(pat)
    wins = None
    if reach is not None and st.fspans - cnt0 <= _NEEDS_WIN_SPANS:
        # windows — from the earliest confirmation among the waiting patterns (a superset;
        # one computation per text version for all), longer than half the text — a full search
        start_min = min(v[0] for v in st.nfrom.values())
        wins = _needs_windows(st, start_min)
        if sum(b - a for a, b in wins) + 2 * reach * len(wins) > len(hay) // 2:
            wins = None
    if wins is None:
        NEEDS_STATS["needs_full"] += 1
        ok = pat.search(hay) is not None
    else:
        NEEDS_STATS["needs_win"] += 1
        off = len(hay) - len(text)      # len(left_ctx)
        ok = False
        for a, b in wins:
            lo = max(0, off + a - reach)
            hi = min(len(hay), off + b + reach)
            if lo < hi and pat.search(hay, lo, hi) is not None:
                ok = True
                break
    st.needs[pat] = ok
    if ok:
        st.nfrom.pop(pat, None)
    else:
        st.nfrom[pat] = (n, st.fspans)
    return ok


def _advance(st, spans, new, left_ctx, acs, width, ywidth=0, cands=None):
    """The text changed (spans — (start, end, len(out)) of the replacements of one rule
    in the coordinates of the OLD text, in order): probes that appeared in the new
    text are searched only in windows ± width around the replaced parts of the NEW
    text (the coordinate shift accumulates per replacement). Windows are merged; a window at
    the start of the text is extended with the tail of left_ctx (ctx-needs of the stream branch).
    Vanished probes are not removed from the set (over-inclusion, sound).
    re-needs with value False are stale (a replacement could add digits/
    a token), True stays: a superset only cancels a skip.
    y-alphabets — the same windows of width max(width, ``ywidth``).
    ``cands`` (_Cands) — probes that first got into the set are handed to
    the iteration of the pass candidates (their rules further down the list become candidates)."""
    st.full = st.low = None
    st.yhay = None
    # False stays: the recheck is over the log windows (see _needs_ok); the log
    # overflowed — reset to the old path (False → full search)
    st.wcache = None
    if len(st.flog) >= _NEEDS_LOG_CAP:
        st.flog = []
        st.fabs = []
        st.nfrom = {}
        st.fspans = 0
        if st.needs:
            st.needs = {k: v for k, v in st.needs.items() if v}
    if st.nfrom:                    # there are False values that need windows
        st.flog.append(tuple(spans))
        st.fspans += len(spans)
        st.fabs.append((st.fabs[-1] if st.fabs else 0)
                       + sum(max(k, e - s) for s, e, k in spans))
    elif st.needs:
        st.needs = {k: v for k, v in st.needs.items() if v}
    if st.raw is None and st.ci is None and st.rawy is None and st.ciy is None:
        return
    if not _INCR_ON:                    # the old behaviour: a full recount
        st.raw = st.ci = None
        st.rawy = st.ciy = None
        return
    ywin = st.rawy is not None or st.ciy is not None
    if st.raw is None and st.ci is None and not ywin:
        return
    if ywin:
        # y-probes without folding runs — the same windows (± the maximum of the widths), the window is folded
        # ё→е for the y-automata
        width = max(width, ywidth)
    ac_ci, ac_raw = acs[1], acs[0]
    wins = []
    delta = 0
    for s, e, n in spans:
        ns = s + delta
        a, b = ns - width, ns + n + width
        if wins and a <= wins[-1][1]:
            if b > wins[-1][1]:
                wins[-1][1] = b
        else:
            wins.append([a, b])
        delta += n - (e - s)
    def add(alpha, got):
        cur = getattr(st, _ALPHA_ATTR[alpha])
        if cands is not None:
            got = got - cur
            if not got:
                return
            cands.grow(alpha, got)
        cur |= got

    for a, b in wins:
        win = new[max(a, 0):b]
        if a < 0 and left_ctx:
            win = left_ctx[a:] + win
        STATS["scan_win"] += 1
        if st.raw is not None and ac_raw is not None:
            add(0, ac_raw.found(win))
        low = None
        if st.ci is not None and ac_ci is not None:
            low = _fold(win)
            add(1, ac_ci.found(low))
        if ywin:
            if st.rawy is not None and acs[2] is not None:
                add(2, acs[2].found(_yo_raw(win)))
            if st.ciy is not None and acs[3] is not None:
                add(3, acs[3].found((low if low is not None else _fold(win)).replace("ё", "е")))


# --- POINT SUB (start literals) ---------------------------------
# The prefilter decides "whether to run the rule", but if a probe is in the text, sub
# scans the WHOLE text of the batch. Pack rules (person-ci/-latin, exact-tail,
# the ru_trace dictionary) start with a boundary assertion (lookbehind), so
# re cannot jump by the first character and tries a match at every position:
# ~6 ms per batch of 200–270K, thousands of rules per pass, 9 of 10 such subs change
# nothing (the probe is there, the word is not: «ivan» inside «ivanov»; a measurement on a
# tg export: 77 % of the computation time — re.sub).
# Point sub: from the PARSED regex a set of START literals is derived —
# after the leading zero-width nodes (anchors, lookahead/lookbehind, a branch
# of assertions only — the boundary.L boundary) a match MUST start with
# one of them. So a match can start only at an occurrence of a start
# literal: candidates are found with str.find (C speed), at each one —
# pattern.match(text, pos) (assertions see the whole text: pos ≠ a slice). The order
# of candidates is ascending, positions taken by a previous match are skipped — exactly
# the semantics of re.sub left to right without overlaps: positions outside the candidates
# cannot match, and the first successful one among the candidates is the same that
# the scanner would find. The minimum pattern width is ≥ 1 (empty matches are not served).
# repl is called on the same matches in the same order; the match object differs
# only in m.pos (the search start position) — the engine, detectors and packs do not
# read it (checked with grep; new rules that read m.pos/endpos must
# turn off the point path — do not do that).
# IGNORECASE: candidates are searched in the FOLDED text (_fold, see below) —
# the folding matches how re itself compares, character for character.
# ANON_PREFILTER_VERIFY=1 — every point sub is checked against the spans of
# pattern.finditer, a mismatch = SystemExit. A rule for which the point path
# does not fit (reads m.pos/m.endpos) declares Rule.local_sub=False — for
# it the sub is always full (the ANON_LOCAL_SUB switch is removed, the point path is the base).
_LOCAL_MIN = 512        # a shorter text — a full sub is cheaper than searching candidates
_LOCAL_CAP = 50_000     # more start literals — a full sub
_FIND_MAX = 32          # more literals — candidates by an automaton (one pass), otherwise str.find
_ZW_OPS = (_sre.AT, _sre.ASSERT, _sre.ASSERT_NOT)
# A rule without a start literal, but with a head of BOUNDED width before
# the literal (brand:*:glue of the pack: an optional glue tail `([0-9A-Za-z_]{0,24})` before
# the stem) — "anchor" literals: a match must contain one of them no further than off characters
# from its start. The point path tries match at positions [q − off, q] of each occurrence q
# (ascending, without repeats) — exactly the positions where finditer could start a match;
# other positions cannot give a match (the head is never wider than off). The output is the same
# (ANON_PREFILTER_VERIFY checks against the finditer spans).
_ANCHOR_MAX = 64        # a wider head — no anchor is built (a full sub)

# --- CASE FOLDING EXACTLY AS re.IGNORECASE ----------------
# The prefilter (probes, needs, AC windows) and point sub search for rule literals in
# the folded text. The folding must treat as equal exactly the characters that
# re treats as equal (a str pattern, IGNORECASE without LOCALE): a text character equals
# a pattern literal ⟺ their simple lower cases (_sre.unicode_tolower) lie
# in one class of re._casefix._EXTRA_CASES (i/ı, s/ſ, µ/μ, σ/ς, в/ᲀ …) — this is exactly
# how re compiles (re._compiler: tolower = _sre.unicode_tolower,
# fixes = _EXTRA_CASES; checked on 3.11 and 3.14). str.casefold differs
# from this: «İ» → «i̇» (2 characters — the probe «sidorova» was not found in
# «SİDOROVA», a rule with «İ» in a literal did not find «I»), «ß» → «ss», ligatures
# expand — the length changes and positions drift apart.
# The folding is per character 1:1 (the length does NOT change): a character → the representative
# of its class. The representative is the common one-character casefold of the class members,
# so for all characters except the "problematic" P (casefold ≠ representative:
# İ, ı, ß, ẞ, ligatures, ŉ … — ~105 on 3.14) it matches str.casefold:
# a text without P is folded by the C path of casefold (~0.6 ms per 250K), with them —
# translate by a table (~4 ms). Both branches give the same result (a test).
# The tables are built lazily once per process (≈0.1–0.3 s; prebuild_rules — in
# the parent before the fork) for the CURRENT interpreter: the Unicode versions of 3.11 and 3.14
# differ, a ready table from another machine must not be used. No re internals
# (another Python version) — SystemExit: silently folding "somehow" = a leak.
_FOLD_TAB = None        # (regex of P characters, translate table)
_FOLD_BLOCK = 512       # a block of code points: blocks without case are skipped whole


def _fold_tables():
    global _FOLD_TAB
    if _FOLD_TAB is not None:
        return _FOLD_TAB
    import sys as _sys
    try:
        import _sre as _sre_c
        from re import _casefix
        tolower = _sre_c.unicode_tolower
        extra = _casefix._EXTRA_CASES
    except (ImportError, AttributeError) as e:
        raise SystemExit(
            f"rule: case folding as in re.IGNORECASE is unavailable — no "
            f"_sre.unicode_tolower / re._casefix._EXTRA_CASES ({e}); Python "
            f"{_sys.version.split()[0]} is not verified (verified 3.11–3.14): "
            f"update the folding in anonymizer/engine/rule.py")
    extra_cps = set(extra)
    for v in extra.values():
        extra_cps.update(v)
    extra_blocks = {cp // _FOLD_BLOCK for cp in extra_cps}
    groups = {}             # class key (min of simple lower cases) -> members
    for base in range(0, 0x110000, _FOLD_BLOCK):
        if 0xD800 <= base < 0xE000:
            continue        # surrogates: blocks are aligned, D800–DFFF whole
        end = min(base + _FOLD_BLOCK, 0x110000)
        blk = "".join(map(chr, range(base, end)))
        if base // _FOLD_BLOCK not in extra_blocks and blk.casefold() == blk \
                and blk.lower() == blk:
            continue        # the block has no case and no casefold expansions
        for cp in range(base, end):
            lo = tolower(cp)
            key = min((lo,) + tuple(extra[lo])) if lo in extra else lo
            if key != cp or cp in extra_cps or chr(cp).casefold() != chr(cp):
                g = groups.get(key)
                if g is None:
                    groups[key] = g = {key}
                g.add(cp)
    table = {}
    prob = set()
    for key, mem in groups.items():
        votes = {}
        for m in mem:
            f = chr(m).casefold()
            if len(f) == 1:
                votes[f] = votes.get(f, 0) + 1
        rep = min(votes, key=lambda f: (-votes[f], f)) if votes else chr(key)
        for m in mem:
            if chr(m) != rep:
                table[m] = rep
            if chr(m).casefold() != rep:
                prob.add(m)
    pre = re.compile("[" + "".join(re.escape(chr(c)) for c in sorted(prob)) + "]") \
        if prob else re.compile(r"(?!)")
    _FOLD_TAB = (pre, table)
    return _FOLD_TAB


def _fold(s: str) -> str:
    """Case folding as in re.IGNORECASE: per character 1:1, the same length (see the block above).
    One function for the text, probes, needs strings and start literals."""
    pre, table = _FOLD_TAB if _FOLD_TAB is not None else _fold_tables()
    if pre.search(s) is None:
        return s.casefold()
    return s.translate(table)


# text folding for searching start literals by the rule alphabet: 0 — the text as
# is, 1 — IGNORECASE, 2/3 — y-raw / y-ci (all per character, the same length)
_START_FOLD = (None, _fold, _yo_raw, _yo_ci)


def _zero_width(seq) -> bool:
    """An sre sequence consumes no characters (only assertions/anchors,
    branches and groups of them)."""
    for op, av in seq:
        if op in _ZW_OPS:
            continue
        if op is _sre.BRANCH and all(_zero_width(list(a)) for a in av[1]):
            continue
        if op is _sre.SUBPATTERN and _zero_width(list(av[-1])):
            continue
        return False
    return True


def _start_lits(seq, rd=None):
    """Literal prefixes one of which a match of the sequence must start with
    (after the leading zero-width nodes), or None.
    ``rd`` — _ClassReader without spaces: the class [её] is a letter."""
    i = 0
    while i < len(seq) and _zero_width([seq[i]]):
        i += 1

    def lit(op, av):
        if op is _sre.LITERAL:
            return chr(av)
        if rd is not None and op is _sre.IN:
            return rd.char(av)
        return None

    def extend(prefs, seq, j):
        # (prefixes | None) — extended with literals while the nodes are literal; after
        # a branch (prefixes of different lengths) — stop, no further extension
        while j < len(seq):
            op, av = seq[j]
            c = lit(op, av) if op is _sre.LITERAL or op is _sre.IN else None
            if c is not None:
                prefs = [p + c for p in prefs]
            elif op is _sre.SUBPATTERN:
                if av[1] or av[2]:          # local flags (?i:…) — not proven
                    return prefs
                inner = list(av[-1])
                if len(inner) == 1 and inner[0][0] is _sre.BRANCH:
                    return extend(prefs, inner, 0)
                # a group without a branch in the middle: its literal start
                # is contiguous; what follows inside the group is not continued
                sub = extend(prefs, inner, 0)
                if sub is None:
                    return None
                if inner and all(op2 is _sre.LITERAL or
                                 (op2 is _sre.IN and lit(op2, av2) is not None)
                                 for op2, av2 in inner):
                    prefs = sub              # the whole group is a literal — go on
                else:
                    return sub
            elif op is _sre.BRANCH:
                out = []
                for alt in av[1]:
                    pp = extend(list(prefs), list(alt), 0)
                    if pp is None:
                        return None
                    out.extend(pp)
                    if len(out) > _LOCAL_CAP:
                        return None
                return out
            else:
                return prefs
            j += 1
        return prefs

    prefs = extend([""], list(seq), i)
    if not prefs or any(len(p) < 2 for p in prefs):
        return None
    return prefs


def _minimal_prefixes(prefs) -> tuple:
    """Drop prefixes that start with another prefix of the set (their occurrences
    are already covered): «ivan» covers «ivanov», «ivan ivanov»."""
    keep = []
    for p in sorted(set(prefs), key=len):
        if not any(p.startswith(k) for k in keep):
            keep.append(p)
    return tuple(keep)


def _starts_of(parsed, ci: bool, rd=None):
    """Start literals of a parsed pattern (fold with IGNORECASE) or None.
    ``rd``: a class was read (rd.used) — the literals are in the ё→е fold."""
    try:
        if parsed.getwidth()[0] < 1:
            return None             # the point path does not serve empty matches
        sl = _start_lits(list(parsed), rd)
        if not sl:
            return None
        return _starts_alphabet(sl, ci, rd)
    except Exception:               # noqa: BLE001 — could not parse: a full sub
        return None


def _starts_alphabet(sl, ci: bool, rd=None):
    """Literals ``sl`` in the search alphabet of the rule (see _starts_of) or None."""
    if rd is not None and rd.used:
        out = _minimal_prefixes((_yo_ci if ci else _yo_raw)(p) for p in sl)
    elif ci:
        # all IGNORECASE rules search for starts in one alphabet _yo_ci (the text is folded ё→е):
        # a literal with «ё» — also ё→е (more candidates, the same output)
        out = _minimal_prefixes(_yo_ci(p) for p in sl)
    else:
        out = _minimal_prefixes(sl)
    if not out or len(out) > _LOCAL_CAP or any(len(p) < 2 for p in out):
        return None
    return out


def _anchor_of(parsed, ci: bool):
    """(see the _ANCHOR_MAX block): anchor literals of a pattern without
    a start literal — the start literals of the tail after a head of bounded width.
    (literals in the search alphabet, off, literals in the ё→е fold) or None: the head is wider than _ANCHOR_MAX,
    empty matches, still no literal."""
    try:
        if parsed.getwidth()[0] < 1:
            return None             # the point path does not serve empty matches
        seq = list(parsed)
        off = 0
        for i, node in enumerate(seq):
            if node[0] in _ZW_OPS:
                continue            # assertions/anchors consume no characters
            if off:
                rd = _ClassReader(ci, ws=False)
                sl = _start_lits(seq[i:], rd)
                if sl:
                    out = _starts_alphabet(sl, ci, rd)
                    if out is None:
                        return None
                    return out, off, rd.used
            w = _sre.SubPattern(parsed.state, [node]).getwidth()[1]
            if w > _ANCHOR_MAX or off + w > _ANCHOR_MAX:
                return None
            off += w
        return None
    except Exception:               # noqa: BLE001 — could not parse: a full sub
        return None


def rule_starts(rule):
    """Start literals of a rule (cached on the object, fold with IGNORECASE) or
    None — point sub is unavailable for the rule (a full sub, as before).
    Usually already computed by rule_probes (one pattern parse, in the parent before
    the fork — pipeline.prebuild_rules)."""
    try:
        return rule._starts
    except AttributeError:
        pass
    rule_probes(rule)
    return rule._starts


def _starts_automaton(rule):
    """An Aho–Corasick automaton over the start literals of a rule (cached on the object) —
    for rules with many literals (_FIND_MAX): one pass over the text instead of
    searching for each literal."""
    try:
        return rule._starts_ac
    except AttributeError:
        pass
    rule._starts_ac = ac.Automaton(rule._starts)
    return rule._starts_ac


def _anchor_automaton(rule, lits):
    """An automaton over the anchor literals of a rule (cached on the object, as _starts_automaton)."""
    try:
        return rule._anchor_ac
    except AttributeError:
        pass
    rule._anchor_ac = ac.Automaton(lits)
    return rule._anchor_ac


def _local_matches(rule, text, starts, folded, off: int = 0):
    """Matches of a rule left to right without overlaps — only at occurrences
    of start literals (see the block above). ``folded`` — _fold(text) for
    an IGNORECASE rule (the same length as text), otherwise None. ``off`` > 0 — the literals
    are anchor ones: match is tried at positions [q − off, q] of each occurrence q,
    ascending and without repeats, from the end position of the found match — as finditer."""
    hay = folded if folded is not None else text
    cand = []
    if len(starts) > _FIND_MAX:
        auto = _starts_automaton(rule) if not off else _anchor_automaton(rule, starts)
        cand = [s for s, _e, _p in auto.iter(hay)]
    else:
        find = hay.find
        for p in starts:
            i = find(p)
            while i >= 0:
                cand.append(i)
                i = find(p, i + 1)
    if not cand:
        return []
    cand.sort()
    match = rule.pattern.match
    out = []
    if off:
        tried = 0               # positions < tried are checked or lie inside a found match
        for q in cand:
            s = max(q - off, tried)
            while s <= q:
                m = match(text, s)
                if m is None:
                    s += 1
                    continue
                out.append(m)
                e = m.end()
                s = e if e > s else s + 1   # no empty matches (_anchor_of): a guard against a loop
            if s > tried:
                tried = s
        return out
    last = -1
    end = 0
    for s in cand:
        if s < end or s == last:
            continue
        last = s
        m = match(text, s)
        if m is not None:
            out.append(m)
            end = m.end()
    return out


# --- ROBUSTNESS OF ONE MATCH -----------------------------
# A non-re.error exception from validator/guard/mask on ONE match (typically:
# value_group of a pack rule on an optional group → m.group(N) is None →
# mint._h(None) → AttributeError; validator(None)) flew as a raw traceback
# without a file name, killed the pool worker and the whole round; diagnostics — a binary
# search over the corpus. Now: WARN (rule/file/span/exception), the match
# is SKIPPED (the source text stays, no record is written), an entry goes into the list
# errors; more than ANON_RULE_MAX_ERRORS (default 50; 0 = any error is fatal)
# errors per file → RuleErrorsExceeded with the file name (Exception, not
# SystemExit: a pool of forked processes carries it to the parent, SystemExit kills
# the worker and hangs the pool). re.error from mask/guard/validator — still
# a broken rule = SystemExit (see apply_rules).
_MSG_CAP = 200          # tail of the exception message in WARN/the entry


class RuleErrorsExceeded(RuntimeError):
    """The file exceeded ANON_RULE_MAX_ERRORS rule errors — it is marked failed."""


def _max_errors() -> int:
    raw = os.environ.get("ANON_RULE_MAX_ERRORS", "50")
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"rule: ANON_RULE_MAX_ERRORS={raw!r} — an integer is expected")


def env_on(name: str) -> bool:
    """An engine 0/1 switch (env; empty/«1» — on, «0» — off, anything else — SystemExit).
    The module calls it at import — a bad value fails in the parent, before the worker pool
    (SystemExit in a worker would hang the pool, see above)."""
    raw = os.environ.get(name, "").strip()
    if raw in ("", "1"):
        return True
    if raw == "0":
        return False
    raise SystemExit(f"{name}={raw!r}: 0 or 1 is expected")


def no_guard(m) -> bool:
    """A no-op guard: for these rules the guard was only a rollback switch (removed together with
    branch 0); the guard stays, because the why block of dry_report.json and the guard aggregate (ok, not none)
    of such rules are part of the output, byte for byte as before."""
    return False


def _note_error(errors: list, rule, m, e, relpath) -> None:
    """One bad match: WARN + an entry {rule_id, file, span, exc, msg} in
    ``errors``; the threshold is counted over the entries of THIS file (the list may be
    shared by a report/run)."""
    a, b = m.span()
    rec = {"rule_id": rule.name, "file": relpath, "span": (a, b),
           "exc": type(e).__name__, "msg": str(e)[:_MSG_CAP]}
    errors.append(rec)
    print(f"rule: WARN rule {rule.name} file {relpath or '?'} span {a}-{b} "
          f"— {rec['exc']}: {rec['msg']}", flush=True)
    limit = _max_errors()
    n = sum(1 for x in errors if x.get("file") == relpath)
    if n > limit:
        raise RuleErrorsExceeded(
            f"rule: file {relpath or '?'}: {n} rule errors > "
            f"ANON_RULE_MAX_ERRORS={limit} — file failed (last: rule "
            f"{rule.name} span {a}-{b} {rec['exc']}: {rec['msg']})") from e


# --- A MASK DOES NOT CHANGE THE LINE COUNT -----------------------------
# A rule with `\s+` inside value_group=0 matches across "\n": «Moscow\nPetrov» →
# one PERSON token — two CSV/TSV records merged, the content between them
# vanished; the structural gate on a line count mismatch only switches to
# a rough comparison and does not block. The primary protection is the patterns ([^\S\r\n]
# instead of \s); this one is a safeguard in apply_rules: a mask that changes the count of "\n"/"\r"
# in a match is skipped as a bad match (WARN + rule_errors, see safe), unless
# the rule DECLARES multiline — Rule.multiline=True or (by
# default) derived from the parsed pattern: a DOTALL dot, a class with an explicit
# "\n" / a range through it / a category+its negation pair ([\s\S]),
# a literal "\n" (pem_key: `[\s\S]{0,20000}?` between BEGIN/END). A bare
# `\s` (a class of one CATEGORY_SPACE) and negative classes ([^\d\n], [^"])
# do NOT declare multiline — exactly the source of merges.


class MaskChangesLineCount(ValueError):
    """The mask changed the number of line breaks in the match — the match is skipped."""


def keep_line_breaks(token: str, src: str) -> str:
    """A replacement that keeps the line breaks of the source piece — the token + the same «\\r»/«\\n»
    in order. The line/record count does not change (safeguard, _check_line_count), and a value
    broken by a line break («+7 916\\n123-45-67», «Г.В.\\nПлеханова») is masked whole,
    instead of being skipped with WARN (a tg round: 23 skips — 20 phone_ru, 2 initials_surname,
    1 special_ethnicity_person). Without line breaks in src — exactly token (the output does not change)."""
    return token + "".join(ch for ch in src if ch in "\r\n")


class MaskEatsEscape(ValueError):
    """The match starts/ends with a backslash — in JSONL that is half of an escape
    («\\"», «\\n»): the replacement would eat the escape and break the JSON. The match is skipped."""


def _check_escape_edges(rule, m, out: str) -> None:
    """A safeguard (smoke test 2026-09-29: a pack rule by a literal with a trailing «\\»
    → the line stopped being JSON → BLOCK of the structural gate): a match with an edge
    backslash is not replaced unless the replacement itself keeps it."""
    src = m.group(0)
    if not src:
        return
    eats_left = src[0] == "\\" and not out.startswith("\\")
    eats_right = src[-1] == "\\" and not out.endswith("\\")
    if eats_left or eats_right:
        raise MaskEatsEscape(
            "a match with an edge backslash (JSONL escape) — the replacement would eat "
            "the escape and break the JSON; a literal/pattern must not include «\\» "
            "at the edge")


def _declares_newline(items, dotall: bool) -> bool:
    """A walk over the parsed pattern (re._parser): is there a node that
    EXPLICITLY allows a line break (see the block above). Lookaround consumes no
    text — it is not entered."""
    for op, av in items:
        if op is _sre.LITERAL:
            if av == 10:
                return True
        elif op is _sre.IN:
            if any(o is _sre.NEGATE for o, _ in av):
                continue
            cats = {a for o, a in av if o is _sre.CATEGORY}
            if any((o is _sre.LITERAL and a == 10)
                   or (o is _sre.RANGE and a[0] <= 10 <= a[1]) for o, a in av):
                return True
            for pos, neg in ((_sre.CATEGORY_SPACE, _sre.CATEGORY_NOT_SPACE),
                             (_sre.CATEGORY_DIGIT, _sre.CATEGORY_NOT_DIGIT),
                             (_sre.CATEGORY_WORD, _sre.CATEGORY_NOT_WORD)):
                if pos in cats and neg in cats:
                    return True
        elif op is _sre.ANY:
            if dotall:
                return True
        elif op is _sre.SUBPATTERN:
            _g, add, dele, sub = av
            inner = (dotall or bool(add & re.DOTALL)) and not (dele & re.DOTALL)
            if _declares_newline(sub, inner):
                return True
        elif op is _sre.BRANCH:
            if any(_declares_newline(alt, dotall) for alt in av[1]):
                return True
        elif op in (_sre.MAX_REPEAT, _sre.MIN_REPEAT) or \
                (hasattr(_sre, "POSSESSIVE_REPEAT")
                 and op is _sre.POSSESSIVE_REPEAT):
            if _declares_newline(av[2], dotall):
                return True
        elif hasattr(_sre, "ATOMIC_GROUP") and op is _sre.ATOMIC_GROUP:
            if _declares_newline(av, dotall):
                return True
        elif op is _sre.GROUPREF_EXISTS:
            _g, yes, no = av
            if _declares_newline(yes, dotall) or (no and _declares_newline(no, dotall)):
                return True
    return False


def rule_spans_lines(rule) -> bool:
    """The rule declares a multiline match (cached on the object): Rule.multiline,
    otherwise derived from the pattern; an unparsable pattern is treated as declared (do not
    break someone else's rule with the safeguard)."""
    try:
        return rule._multiline
    except AttributeError:
        pass
    ml = getattr(rule, "multiline", None)
    if ml is None:
        try:
            flags = rule.pattern.flags
            ml = _declares_newline(list(_sre.parse(rule.pattern.pattern, flags)),
                                   bool(flags & re.DOTALL))
        except Exception:           # noqa: BLE001 — could not parse: no safeguard
            ml = True
    rule._multiline = bool(ml)
    return rule._multiline


def _check_line_count(rule, m, out: str) -> None:
    """Safeguard: the count of "\n"/"\r" in the replacement == in the match, otherwise
    MaskChangesLineCount (→ safe: WARN + rule_errors + the match is skipped)."""
    src = m.group(0)
    n0 = src.count("\n") + src.count("\r")
    n1 = out.count("\n") + out.count("\r")
    if n0 != n1 and not rule_spans_lines(rule):
        raise MaskChangesLineCount(
            f"the mask changes the number of line breaks {n0} → {n1} (merging "
            f"lines/records): use [^\\S\\r\\n] instead of \\s in the pattern, or "
            f"Rule.multiline=True for a legitimately multiline rule")


# --- "WHY MASKED" (Presidio decision_process) --------------------
# Every mask carries the facts of the decision: the rule, the tier, the validator/guard result,
# the fired context word and its offset. Collected ONLY for
# masked matches (rejected ones are free, as before) and ONLY when
# the report asks (report.want_why; Report has it on by default — dry/apply;
# someone else's report object without the attribute = off, record is called with the old
# signature). The why format (dict): {"rule": str, "tier": high|context|low,
# "validator": True|None (False never happens — the match is rejected), "guard":
# False|None (True never happens), "ctx_word": str|None — the text of the requires_context
# match in the window, "ctx_pos": int|None — the offset of the word start from
# the start of the value (negative; relative, not absolute — the stream and
# memory see one window and give one number)}.


def _context_hit(text, start, ctx, window, want_why, right=0, end=None):
    """The context gate as before — context.has_context with POSITIONAL
    arguments (the context signature is not touched) — plus, for "why", the
    word itself and its offset: a second search with the same compiled regex over the same
    slice, and only when the gate passed and the report asks for why. ``right``/``end``
    (Rule.ctx_right) — the right window [end, end+right) WITHIN the
    line (context.right_segment); a word on the right gives a POSITIVE offset
    from the start of the value."""
    ok = has_context(text, start, ctx, window, right=right, end=end)
    if not ok or not want_why:
        return ok, None, None
    hit = left_hit(text, start, ctx, window)
    if hit is not None:
        return (ok,) + hit
    if right > 0:                        # the right window fired
        e = start if end is None else end
        hit = right_hit(text, e, ctx, right)
        if hit is not None:
            return ok, hit[0], e - start + hit[1]
    return ok, None, None                # the same regex, the same slice — never happens


def _core_edits(text, spans, outs):
    """A batch of rule replacements for the view log (viewmask.EditLog) — only the differing CORE
    of each replacement: the common prefix and suffix of the match ``text[s:e]`` and ``out`` (the label that the rule
    re-emits in the replacement: «паспорт PASSPORT_REDACTED», «ИП PERSON_NAME_1», «адрес: POSTCODE_…») do
    not go into the log. The view output is the same, and in the source these characters stay byte for byte: an escaped
    entry (`\\u043f\\u0430…` of an ensure_ascii file is not expanded into «паспорт»), a lookalike in the label
    («паcпорт» with a Latin «c» — not fixed, owner's decision item 4). A replacement without a text change does not
    go into the log. ``spans`` — (start, end, len(out)) in the coordinates of ``text`` (the version BEFORE the rule)."""
    batch = []
    for (s, e, _n), o in zip(spans, outs):
        m = text[s:e]
        lim = min(len(m), len(o))
        k = 0
        while k < lim and m[k] == o[k]:
            k += 1
        lim -= k
        j = 0
        while j < lim and m[-1 - j] == o[-1 - j]:
            j += 1
        if k == len(m) == len(o):
            continue
        batch.append((s + k, e - j, o[k:len(o) - j]))
    return batch


def apply_rules(text, rules, report, relpath=None, confidence=("high", "context", "low"),
                left_ctx="", errors=None, rule_tiers=None, prefilter=None, edits=None,
                trace=None, clean_stop=None, skip_match=None):
    """Run ``rules`` over ``text``; return the transformed text. Records every
    replacement in ``report``. Respects per-rule scope, confidence gate,
    validator, context window and guard.

    ``skip_match`` (nsjson, pipeline._raw_guard) — a callable ``(rule, m) -> bool`` for each
    match that reached the mask (validator, context and guard passed): True — the match stays as
    is, without a record and without a span (a replacement removed by the JSONL line safeguard). None — the old path.

    ``trace`` — a list: the index of each rule that changed the text in this
    call is appended (in ``rules`` order). ``clean_stop`` — the index from which the remaining rules
    are skipped if no rule BEFORE it in this call changed the text: the check pass
    of the fixpoint (pipeline._rules_fixpoint) — the rules after the last one that fired in the previous pass
    have already seen exactly this text and did not fire; if at least one fired — the text is different, the rest runs
    whole, as before. None — the old path.

    ``edits`` (viewmask) — a list: for each rule that changed the text,
    ``("sub", [(start, end, out), …])`` is appended — its replacements in the coordinates of the text version BEFORE that rule,
    ascending, without overlaps (the order of ``pattern.sub``/``_local_matches``). None — do not
    collect (the old path).

    ``left_ctx`` — the tail of the PREVIOUS text for the stream branch: the
    in-memory path sees the context label of the previous line
    through "\\n" inside the ``rule.window`` window, the line-by-line stream must see
    the same tail — otherwise context-gated PD at the start of a line goes out
    unmasked and the byte oracle stream==memory breaks.

    ``errors`` — a list for entries about bad matches (see _note_error); None →
    ``report.rule_errors`` if the report has such a list, otherwise a local one
    (the errors are visible only in WARN). Returns — still only the text.

    ``rule_tiers`` — CONFIG["rule_tiers"]: {rule_name: tier}
    on top of Rule.confidence; None/{} = rule tiers as they are.

    ``prefilter`` — PrefilterState for carrying probe presence between
    fixpoint passes over ONE text: the state is bound to the text
    that the previous apply_rules with it returned (another text = SystemExit).
    None — a local state per call."""
    if errors is None:
        errors = getattr(report, "rule_errors", None)
    if not isinstance(errors, list):    # someone else's report with __getattr__ → not a list
        errors = []
    # why is collected only when the report asks (Report.want_why); someone else's
    # object without the attribute — the old record signature (without kwargs)
    want_why = getattr(report, "want_why", False) is True   # not a truthy stub
    # the report of a window core (pipeline._CoreReport) — an error of a match outside the core is written by the window
    # in whose core the match started (the match is skipped anyway, as with any error)
    keep_err = getattr(report, "keep_match", None)
    rule_tiers = rule_tiers or {}
    # a single AC prefilter; st.raw/st.ci — which probes are really in
    # the text (lazily; after a replacement they are extended by windows, not reset).
    st = prefilter
    if _PREFILTER_ON:
        ac_ci, ac_raw, max_ci, max_raw = _rule_automata(rules)
        width = max(max_ci, max_raw)
        yac = _rule_automata_y(rules)          # y-alphabets | None
        acs = (ac_raw, ac_ci) + ((yac[0], yac[1]) if yac else (None, None))
        ywidth = yac[2] if yac else 0
        if st is None:
            st = PrefilterState()
        elif st.text is not None and st.text is not text and st.text != text:
            raise SystemExit(
                "prefilter: PrefilterState was passed with another text — "
                "the state is bound to the text that the previous "
                "apply_rules with it returned (fixpoint: pass the returned text)")
    # (text version, text folding | None) by the alphabets of start literals:
    # 1 — _fold (IGNORECASE), 2/3 — ё→е case-sensitive / _fold + ё→е
    fold_cache = [None, (None, None), (None, None), (None, None)]
    # alphabet 1 — one _yo_ci folding for all IGNORECASE rules (the one of y-ci,
    # _START_FOLD[3]); y-ci rules search for starts in it too
    sfold = (None, _START_FOLD[3], _START_FOLD[2], _START_FOLD[3])
    fired = False               # at least one rule of this call changed the text
    # a large rule list — iterating only the candidates (see _Cands)
    cands = _candidates(rules, st, acs, text, left_ctx) if _PREFILTER_ON else None
    for i, rule in (enumerate(rules) if cands is None else cands):
        if clean_stop is not None and i >= clean_stop and not fired:
            break               # the rest has already seen exactly this text and did not fire
        if rule.kind not in ("regex", "token_remap"):
            continue  # value/csv_column rules are driven by engine.formats
        tier = rule_tiers.get(rule.name, rule.confidence)   # #76 P2-5
        if tier not in confidence:
            continue
        if rule.scope is not None and (relpath is None or relpath not in rule.scope):
            continue
        verify_needs = False
        loc = None                  # start literals for point sub
        loff = 0                    # offset of anchor literals (0 — start ones)
        lsy = False
        if _PREFILTER_ON:
            probes = rule_probes(rule)
            if len(text) >= _LOCAL_MIN:
                loc, loff, lsy = _local_lits(rule)
            pf = rule._probes_f
            if pf is not None:
                if pf is probes:
                    gone = _absent(st, probes, rule._probes_ci + 2 * rule._probes_y, acs,
                                   text, left_ctx, exact=loc is None)
                else:                   # parts of a y-probe
                    gone = _absent_parts(st, pf, rule._probes_ci + 2, acs,
                                         text, left_ctx, exact=loc is None)
                if gone:
                    STATS["probe_skip"] += 1
                    if _PREFILTER_VERIFY and rule.pattern.search(text):
                        raise SystemExit(
                            f"prefilter: rule {rule.name} was skipped, but "
                            f"pattern.search found a match — the probes "
                            f"{probes!r} are wrong (bug in _alt_probes/AC)")
                    continue
            nd = rule_needs(rule) if _NEEDS_ON else None
            if nd is not None:
                if nd[0] == "s":
                    ok = not _absent(st, nd[1], int(nd[2]), acs, text, left_ctx)
                else:
                    ok = _needs_ok(st, nd[1], text, left_ctx)
                if not ok:
                    STATS["needs_skip"] += 1
                    if _PREFILTER_VERIFY:
                        verify_needs = True     # run it and make sure: 0 replacements
                    else:
                        continue
        spans = []                  # (start, end, len(out)) of the real replacements
        outs = []                   # out of each replacement — for the view log (edits)

        def repl(m, rule=rule, tier=tier, spans=spans, outs=outs):
            val = rule._value(m)
            v_ok = None                  # facts of the decision (None = no stage)
            if rule.validator:
                v_ok = bool(rule.validator(val))
                if not v_ok:
                    return m.group(0)
            ctx_word = ctx_pos = None
            if rule.requires_context:
                if left_ctx and m.start() < rule.window + ctx_reach(rule.requires_context):
                    # the window (with lookbehind past its left edge —
                    # context.left_hit) hits the start of the line:
                    # extend it with the tail of previous lines — exactly what
                    # the in-memory branch sees in the whole text (label
                    # tokens are not touched by masks, so the final tail
                    # is equivalent to the intermediate one)
                    hay = left_ctx + m.string[: m.start()]
                    hit = left_hit(hay, len(hay), rule.requires_context, rule.window)
                    ok = hit is not None
                    if ok and want_why:
                        ctx_word, ctx_pos = hit
                    if not ok and rule.ctx_right:
                        # the right window — within the line (the stream sees only it);
                        # lookbehind at its left edge sees the value and, as
                        # in memory, the tail of previous lines
                        o = len(left_ctx)
                        ok, ctx_word, ctx_pos = _context_hit(
                            left_ctx + m.string,
                            o + m.start(), rule.requires_context, 0,
                            want_why, right=rule.ctx_right, end=o + m.end())
                else:
                    ok, ctx_word, ctx_pos = _context_hit(
                        m.string, m.start(), rule.requires_context,
                        rule.window, want_why, right=rule.ctx_right,
                        end=m.end())
                if not ok:
                    return m.group(0)
            g_hit = None
            if rule.guard:
                g_hit = bool(rule.guard(m))
                if g_hit:
                    return m.group(0)
            if skip_match is not None and skip_match(rule, m):
                return m.group(0)
            out = rule.mask(m)
            if out == m.group(0):
                return out
            _check_line_count(rule, m, out)     # do not merge lines
            _check_escape_edges(rule, m, out)   # do not eat a JSONL escape
            if want_why:
                report.record(rule.name, m.group(0), out, m, why={
                    "rule": rule.name,
                    "tier": tier,
                    "validator": v_ok, "guard": g_hit,
                    "ctx_word": ctx_word, "ctx_pos": ctx_pos})
            else:
                report.record(rule.name, m.group(0), out, m)
            spans.append((m.start(), m.end(), len(out)))
            if edits is not None:
                outs.append(out)
            return out

        def safe(m, rule=rule, repl=repl):
            """One bad match → WARN + skip + an entry in errors;
            re.error passes through — a broken rule fails below."""
            try:
                return repl(m)
            except re.error:
                raise
            except Exception as e:   # noqa: BLE001 — one match, not the round
                if keep_err is None or keep_err(m):
                    _note_error(errors, rule, m, e, relpath)
                return m.group(0)

        # fail loudly: re.error from mask/guard/validator in the middle of
        # .sub is a broken rule (the patterns are precompiled; the real
        # source is the dynamic mask/guard of external packgen packs). A silent
        # `continue` returned the SOURCE text of the rule (PD survived), but record()
        # for the matches applied before the failure stayed in the report — the Report
        # diverged from the text actually produced. Let it fail loudly.
        # Other exceptions of ONE match are suppressed by safe (see RuleErrorsExceeded).
        new_fold = None             # {alphabet: folding of the new text, glued from pieces}
        try:
            # point sub by start literals (see the block above); otherwise —
            # a full sub, as before
            if _PREFILTER_ON:
                starts = loc
            elif len(text) >= _LOCAL_MIN:
                starts, loff, lsy = _local_lits(rule)
            else:
                starts = None
            # foldings of the CURRENT text version (if any) — for the rule to search
            # in its alphabet and for gluing the foldings of the new version by any point rule
            live = [(j, fold_cache[j][1]) for j in (1, 2, 3)
                    if fold_cache[j][0] is text and fold_cache[j][1] is not None]
            folded = None
            sk = 0
            if starts is not None:
                sk = rule._starts_ci + 2 * bool(lsy)
                if rule._starts_ci:
                    sk = 1
            if sk:
                if fold_cache[sk][0] is not text:
                    f = sfold[sk](text)
                    f = f if len(f) == len(text) else None
                    fold_cache[sk] = (text, f)
                    if f is not None:
                        live.append((sk, f))
                folded = fold_cache[sk][1]
                if folded is None:
                    # safeguard: the folding is per character 1:1 (the same length) — never gets here
                    starts = None
            if starts is not None:
                STATS["local_sub"] += 1
                ms = _local_matches(rule, text, starts, folded, loff)
                if _PREFILTER_VERIFY:
                    exp = [m.span() for m in rule.pattern.finditer(text)]
                    got = [m.span() for m in ms]
                    if exp != got:
                        raise SystemExit(
                            f"local-sub: rule {rule.name}: point matches {got[:5]}… "
                            f"≠ finditer {exp[:5]}… — {'anchor' if loff else 'start'} "
                            f"literals {starts[:5]!r} (off {loff}) are wrong "
                            f"(bug in {'_anchor_of' if loff else '_start_lits'})")
                if ms:
                    parts = []
                    fparts = {j: [] for j, _c in live}
                    last = 0
                    for m in ms:
                        s, e = m.start(), m.end()
                        out = safe(m)
                        parts.append(text[last:s])
                        parts.append(out)
                        for j, cur in live:
                            fp = fparts.get(j)
                            if fp is None:
                                continue
                            fo = sfold[j](out)
                            if len(fo) == len(out):
                                fp.append(cur[last:s])
                                fp.append(fo)
                            else:
                                del fparts[j]   # the folding of the replacement changed length — recount
                        last = e
                    parts.append(text[last:])
                    new = "".join(parts)
                    new_fold = {}
                    for j, cur in live:
                        fp = fparts.get(j)
                        if fp is not None:
                            fp.append(cur[last:])
                            new_fold[j] = "".join(fp)
                else:
                    new = text
            else:
                STATS["full_sub"] += 1
                new = rule.pattern.sub(safe, text)
        except re.error as e:
            raise SystemExit(f"rule {rule.name}: application failed ({e})")
        if verify_needs and spans:
            raise SystemExit(
                f"prefilter: rule {rule.name} was skipped by needs "
                f"{rule.needs!r}, but made {len(spans)} replacements — needs is not "
                f"a necessary condition (detector bug)")
        if spans:
            # the text changed — probe presence is extended by windows around
            # the replacements: an IGNORECASE rule that depends on a token
            # minted EARLIER in the same pass (special_*,
            # handle_label) sees it through the window (VERIFY
            # failed on special_slur_person with a stale probe)
            if _PREFILTER_ON:
                _advance(st, spans, new, left_ctx, acs, width, ywidth, cands)
            if edits is not None:
                edits.append(("sub", _core_edits(text, spans, outs)))
            fired = True
            if trace is not None:
                trace.append(i)
            text = new
            # the folding of the new version — glued from pieces (without a full recount)
            for j in (1, 2, 3):
                f = new_fold.get(j) if new_fold else None
                fold_cache[j] = (new, f) if f is not None else (None, None)
    if _PREFILTER_ON:
        st.text = text
    return text


# ---- rule BEHAVIOUR fingerprint ----------------------------------------
# The incremental round computation compares the rules of the old and the new pack: the
# same rule = the same fingerprint. The fingerprint is sha256 of everything that affects
# matches and replacements: the name, the pattern and flags, the Rule fields (window, context, tier,
# scope, value_group, multiline, needs…) and the mask/guard/validator functions —
# bytecode, constants (recursively), names, default arguments,
# closure contents and the PACK MODULE GLOBALS that the code refers to
# (the list of forms in the pack file). Line numbers and the file name are not part of the fingerprint:
# moving a rule within the pack file is not a change. The engine/detector code is held by
# code_sha256 of the computation fingerprint (its equality is a condition of the increment), so
# functions from engine modules are included by name and code, without their globals.
# An unknown object in a closure → None ("do not know" = the rule changed).
import types as _types


class _Undigestible(Exception):
    pass


def _dg(obj, h, depth: int = 0) -> None:
    if depth > 40:
        raise _Undigestible("depth")
    t = type(obj)
    if obj is None or obj is Ellipsis or t in (bool, int, float, complex):
        h.update(f"{t.__name__}:{obj!r};".encode())
        return
    if t is slice:                       # slice constants in co_consts (3.12+)
        h.update(b"slice")
        _dg((obj.start, obj.stop, obj.step), h, depth + 1)
        return
    if t is str:
        h.update(b"s" + str(len(obj)).encode() + b":"
                 + obj.encode("utf-8", "surrogatepass") + b";")
        return
    if t is bytes:
        h.update(b"b" + str(len(obj)).encode() + b":" + obj + b";")
        return
    if t in (tuple, list):
        h.update(f"{t.__name__}[{len(obj)}".encode())
        for x in obj:
            _dg(x, h, depth + 1)
        h.update(b"]")
        return
    if t in (set, frozenset):
        subs = []
        for x in obj:
            hh = hashlib.sha256()
            _dg(x, hh, depth + 1)
            subs.append(hh.hexdigest())
        h.update(f"{t.__name__}{{{','.join(sorted(subs))}}}".encode())
        return
    if t is dict:
        subs = []
        for k, v in obj.items():
            hh = hashlib.sha256()
            _dg(k, hh, depth + 1)
            _dg(v, hh, depth + 1)
            subs.append(hh.hexdigest())
        h.update(f"dict{{{','.join(sorted(subs))}}}".encode())
        return
    if isinstance(obj, re.Pattern):
        h.update(b"re:")
        _dg(obj.pattern, h, depth + 1)
        h.update(f"{obj.flags};".encode())
        return
    if isinstance(obj, _types.CodeType):
        h.update(b"code:" + obj.co_code)
        for attr in ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount",
                     "co_flags", "co_names", "co_varnames", "co_freevars",
                     "co_cellvars"):
            _dg(getattr(obj, attr, None), h, depth + 1)
        _dg(getattr(obj, "co_exceptiontable", b""), h, depth + 1)
        _dg(tuple(obj.co_consts), h, depth + 1)
        return
    if isinstance(obj, _types.FunctionType):
        h.update(f"fn:{obj.__module__}.{obj.__qualname__};".encode())
        _dg(obj.__code__, h, depth + 1)
        _dg(obj.__defaults__, h, depth + 1)
        _dg(obj.__kwdefaults__, h, depth + 1)
        for cell in obj.__closure__ or ():
            try:
                _dg(cell.cell_contents, h, depth + 1)
            except ValueError:          # an empty cell
                h.update(b"cell:empty;")
        g = obj.__globals__
        if str(g.get("__name__", "")).startswith("anonymizer._known_file_"):
            # globals of the PACK module (forms, tables) — rule data
            for name in sorted(_code_names(obj.__code__)):
                if name in g:
                    h.update(f"g:{name}=".encode())
                    val = g[name]
                    if isinstance(val, _types.ModuleType):
                        h.update(f"mod:{val.__name__};".encode())
                    else:
                        _dg(val, h, depth + 1)
        return
    if isinstance(obj, _types.ModuleType):
        h.update(f"mod:{obj.__name__};".encode())
        return
    if isinstance(obj, (_types.BuiltinFunctionType, _types.BuiltinMethodType)):
        h.update(f"builtin:{getattr(obj, '__module__', '')}."
                 f"{obj.__qualname__};".encode())
        self_ = getattr(obj, "__self__", None)
        if self_ is not None and not isinstance(self_, _types.ModuleType):
            _dg(self_, h, depth + 1)
        return
    if isinstance(obj, _types.MethodType):
        _dg(obj.__func__, h, depth + 1)
        _dg(obj.__self__, h, depth + 1)
        return
    if isinstance(obj, Rule):
        _dg(_rule_fields(obj), h, depth + 1)
        return
    raise _Undigestible(t.__name__)


def _code_names(code) -> set:
    names = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, _types.CodeType):
            names |= _code_names(c)
    return names


def _rule_fields(rule) -> tuple:
    return (rule.name, rule.pattern, rule.mask, rule.value_group,
            rule.validator, rule.requires_context, rule.window,
            rule.ctx_right, rule.guard, rule.confidence,
            sorted(rule.scope) if rule.scope is not None else None,
            rule.kind, rule.multiline, rule.needs,
            # column rules: the cell function and the column one
            getattr(rule, "value_fn", None), getattr(rule, "columns", None),
            getattr(rule, "family", None))


def rule_digest(rule):
    """sha256 of the rule behaviour (see the block above) or None — the closure has
    an object that cannot be fingerprinted honestly (the rule is treated as
    changed)."""
    h = hashlib.sha256()
    try:
        _dg(_rule_fields(rule), h)
    except _Undigestible:
        return None
    return h.hexdigest()


# ---- convenience factories ---------------------------------------------------
def regex_rule(name, pattern, prefix=None, redacted=None, *, seed=None, keep=(),
               value_group=0, validator=None, requires_context=None, window=120,
               ctx_right=0, guard=None, confidence="high", scope=None, flags=0,
               note="", needs=None):
    """Build a Rule that replaces the match (or its ``value_group``) with either
    a minted ``prefix`` token (seeded via ``seed(match)``) or a fixed
    ``redacted`` placeholder, optionally re-emitting ``keep`` groups verbatim
    (label / legal form / « » / # / trailing ``.ORG_123``). ``needs`` — as in
    Rule (packgen pack files are built by this factory)."""
    from . import mint as _mint

    rx = re.compile(pattern, flags) if isinstance(pattern, str) else pattern

    def _seed(m):
        if seed:
            return seed(m)
        return m.group(value_group) if value_group else m.group(0)

    def _mask(m):
        if redacted is not None:
            token = redacted
        else:
            token = _mint.tag(prefix, _seed(m)) if prefix != "PERSON_NAME" else _mint.pname(_seed(m))
        if not keep:
            return token
        parts = []
        for g in keep:
            try:
                parts.append((m.start(g), m.group(g)))
            except (IndexError, re.error):
                pass
        # rebuild: kept groups in their positions + token where the value was
        # (simple case: keep is the label/prefix, value_group is the value)
        if value_group:
            vs, ve = m.span(value_group)
            base = m.group(0)
            off = m.start(0)
            return base[: vs - off] + token + base[ve - off:]
        return "".join(p[1] for p in parts) + token

    return Rule(
        name=name, pattern=rx, mask=_mask, value_group=value_group,
        validator=validator, requires_context=requires_context, window=window,
        ctx_right=ctx_right, guard=guard, confidence=confidence, scope=scope,
        note=note, needs=needs,
    )
