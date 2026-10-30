"""Reporting harness: per-rule counters + capped before/after examples.

Encodes the project's #1 discipline: if a rule's count is an order of magnitude
over what you expected, that's not a find, it's corruption — read the examples,
don't apply. ``Report`` makes counts + samples visible in dry mode.

"Why masked" (after Presidio decision_process): record(..., why=dict)
takes the facts of the match decision (see rule.py: rule/tier/validator/guard/
ctx_word/ctx_pos) and stores them compactly — a per-rule aggregate ``why_stats`` +
why examples aligned with ``examples`` by index. Serialization is
``why_json()``: a block for the OPTIONAL top-level key "why" of dry_report.json
(next to "rules", not inside — the signoff/htmlview contract reads only
"rules"); None when there is no why at all — reports without masks stay
byte for byte the same.
"""
from __future__ import annotations

from collections import Counter, defaultdict

WHY_TOP = 10          # top N context words of a rule in serialization
_WHY_WORDS_CAP = 64   # cap of distinct context words per rule (the rest → «…»)
_WHY_OTHER = "…"
def _tri(v, ok_when) -> str:
    """bool|None → aggregate bucket: none (no such step) | ok | fail.
    validator: ok on True; guard: ok on False (did not fire = let through)."""
    if v is None:
        return "none"
    return "ok" if bool(v) is ok_when else "fail"


def _ctx_count(counter: Counter, word: str, n: int,
               cap: int | None = _WHY_WORDS_CAP) -> None:
    """Counter of context words with a cap on distinct keys: a new word over the cap
    goes to «…». Insertion order is kept, so merging pieces in file
    order gives the same key set as a sequential run.
    ``cap=None`` — no cap (a report of a PART of a file: see Report.part)."""
    if cap is None or word in counter or len(counter) < cap:
        counter[word] += n
    else:
        counter[_WHY_OTHER] += n


class Report:
    def __init__(self, max_examples: int = 30, want_why: bool = True):
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.files_changed = 0
        self.rule_errors = []     # skipped matches of apply_rules (see rule._note_error)
        # token↔letter/digit fusions split by the engine after the rules
        # (boundary.split_fusions). Not a replacement and not a rule: not part of counts/signature,
        # visible in the round output and the manifest.
        self.fusions_split = 0
        # (viewmask): mixed-script tokens folded in the view for rules
        # (vnorm.homoglyph_view). Not a replacement and not a rule: not part of counts/signature,
        # a diagnostic of the folding volume (risk of extra masking).
        self.view_homoglyph = 0
        # (vnorm.fold_view): places folded in the view for rules (removed invisibles, NFC,
        # stripped stress marks). Like view_homoglyph: not in counts/signature; in the store — only when non-zero.
        self.view_fold = 0
        # (formats/mail): mail attachments not scanned by the engine (not text/* or
        # Content-Disposition: attachment — they do not go into the view). Like view_fold: not in counts/signature;
        # in the store — only when non-zero.
        self.mail_attachments = 0
        self.max_examples = max_examples
        # want_why — apply_rules collects "why" only when the report
        # asks (default on: dry/apply); why_stats — a per-rule aggregate
        # {tier, validator: Counter(ok/fail/none), guard: Counter(…),
        # ctx_words: Counter(casefolded word → n)}; why_examples — a list of
        # why-dict|None aligned with examples[rule] by index
        self.want_why = want_why
        self.why_stats = {}
        self.why_examples = defaultdict(list)
        # cap of distinct context words (None — no cap, a part report)
        self.ctx_cap = _WHY_WORDS_CAP
        # fixpoint calls that hit the pass cap (did not converge) —
        # a caller diagnostic, not part of merge/signature
        self.fixpoint_capped = 0
        # rules with a trace in a DISCARDED pass (rollback of a structured
        # file projection: a rule ate a separator, the layout did not converge).
        # They did affect the result — without them the pass would have succeeded and the output/report
        # would differ — but they are not in counts/rule_errors. Incremental mode
        # counts the unit as touched by them (trace). Not part of the signature.
        self.shadow = set()
        # nsjson (JSONL_RAW_GUARD): raw-pass replacements removed by the safeguard from a JSONL
        # line they broke: [{rule, file, line}] (line 1-based; 0 — a .json document).
        # The rule's own records are subtracted from counts/examples (swap_records) — counts reflect
        # the output, the idempotence oracle does not see the removed replacement. Not a replacement and not a rule:
        # not part of the signature; in the store — only when non-empty.
        self.raw_guard: list = []

    @classmethod
    def part(cls, max_examples: int = 30, want_why: bool = True) -> "Report":
        """Report of a PART of a file (batches of one file on different workers).
        A sequential run accumulates a rule's context words in the file report
        with the _WHY_WORDS_CAP cap: the fate of a new word (a key or «…») is decided
        by the counter state at the moment of its FIRST appearance. A part report with
        its own cap would decide it earlier («…» for a word that is already a key in the
        file) — merging parts would diverge from a sequential run.
        Without a cap a part keeps all words in order of first appearance with
        full sums — merging into the file report (with the cap) reproduces the
        sequential counter exactly. Examples (the first max_examples per
        rule) merge exactly anyway: a part's examples are a prefix of its records."""
        rep = cls(max_examples=max_examples, want_why=want_why)
        rep.ctx_cap = None
        return rep

    def record(self, rule_name, before, after, match=None, why=None):
        self.counts[rule_name] += 1
        ex = self.examples[rule_name]
        if len(ex) < self.max_examples:
            if match is not None:
                # the CLEANED value is marked ⟦…⟧ (readability of the view:
                # "what exactly is hidden"); the context is wide — the view folds
                # the line and unfolds it on click
                st, en, s = match.start(), match.end(), match.string
                pre = s[max(0, st - 160):st].replace("\n", " ")
                val = s[st:en].replace("\n", " ")
                post = s[en:en + 160].replace("\n", " ")
                ex.append(f"…{pre}⟦{val}⟧{post}…  ->  {after}")
            else:
                ex.append(f"{before!r} -> {after}")
            # a None placeholder keeps alignment with examples (pseudo-rules
            # and calls without why)
            self.why_examples[rule_name].append(
                self._why_example(why, before) if why else None)
        if why:
            self._why_add(rule_name, why)

    @staticmethod
    def _why_example(why: dict, before) -> dict:
        val = before.replace("\n", " ") if isinstance(before, str) else str(before)
        return {"val": val, "validator": why.get("validator"),
                "guard": why.get("guard"), "ctx_word": why.get("ctx_word"),
                "ctx_pos": why.get("ctx_pos")}

    def _why_add(self, rule_name, why: dict) -> None:
        st = self.why_stats.get(rule_name)
        if st is None:
            st = self.why_stats[rule_name] = {
                "tier": why.get("tier"), "validator": Counter(),
                "guard": Counter(), "ctx_words": Counter()}
        st["validator"][_tri(why.get("validator"), True)] += 1
        st["guard"][_tri(why.get("guard"), False)] += 1
        w = why.get("ctx_word")
        if w is not None:
            _ctx_count(st["ctx_words"], str(w).casefold(), 1,
                       getattr(self, "ctx_cap", _WHY_WORDS_CAP))

    def merge(self, other: "Report") -> None:
        """Merge the report of a tree piece (parallel Pipeline.run).
        Pieces come in file order, so counters and the first max_examples
        examples match a sequential run number for number and
        line for line (the dry report/diff signature/fingerprint do not diverge)."""
        self.counts.update(other.counts)
        other_why_ex = getattr(other, "why_examples", {})
        for name, exs in other.examples.items():
            mine = self.examples[name]
            room = self.max_examples - len(mine)
            if room > 0:
                mine.extend(exs[:room])
                # why examples travel in the same window — index alignment
                # is kept (a foreign/old object without why → placeholders)
                ow = other_why_ex.get(name)
                if ow is None or len(ow) != len(exs):
                    ow = [None] * len(exs)
                self.why_examples[name].extend(ow[:room])
        for name, st in getattr(other, "why_stats", {}).items():
            mine_st = self.why_stats.get(name)
            if mine_st is None:
                mine_st = self.why_stats[name] = {
                    "tier": st["tier"], "validator": Counter(),
                    "guard": Counter(), "ctx_words": Counter()}
            mine_st["validator"].update(st["validator"])
            mine_st["guard"].update(st["guard"])
            cap = getattr(self, "ctx_cap", _WHY_WORDS_CAP)
            for w, n in st["ctx_words"].items():   # insertion order of the piece
                _ctx_count(mine_st["ctx_words"], w, n, cap)
        self.files_changed += other.files_changed
        self.rule_errors.extend(getattr(other, "rule_errors", ()))
        self.fusions_split += getattr(other, "fusions_split", 0)
        self.view_homoglyph += getattr(other, "view_homoglyph", 0)
        self.view_fold += getattr(other, "view_fold", 0)
        self.mail_attachments += getattr(other, "mail_attachments", 0)
        # the "fixpoint did not converge" flag used to be lost when merging file/part reports
        self.fixpoint_capped += getattr(other, "fixpoint_capped", 0)
        self.shadow.update(getattr(other, "shadow", ()))
        self.raw_guard.extend(getattr(other, "raw_guard", ()))

    def swap_records(self, old: "Report", new: "Report") -> None:
        """nsjson: replace the line records made by pass ``old`` (the line repeated as it was) with
        the records of pass ``new`` (the line without breaking replacements) — by DIFFERENCE, not wholesale: a rule
        that fired the same way in both passes is not touched (its examples stay in their
        places — stream and memory give one report). Subtracted: counts by the difference, examples by
        line (first occurrence; beyond the example cap there is nothing to subtract), why examples by the same
        index, why_stats counters by the why of the subtracted examples. Added: as record."""
        names = list(old.counts) + [n for n in new.counts if n not in old.counts]
        for name in names:
            n_old, n_new = old.counts.get(name, 0), new.counts.get(name, 0)
            old_keys = [_ex_key(e) for e in old.examples.get(name, [])]
            new_keys = [_ex_key(e) for e in new.examples.get(name, [])]
            if n_old == n_new and old_keys == new_keys:
                continue
            # a multiset of examples by key (value + token; the context differs for a line and
            # for a file): old minus new — subtract, new minus old — add
            drop = list(old_keys)
            add = []
            for ex in new.examples.get(name, []):
                k = _ex_key(ex)
                if k in drop:
                    drop.remove(k)
                else:
                    add.append(ex)
            if n_old > n_new:
                self.counts[name] -= n_old - n_new
                if self.counts[name] <= 0:
                    del self.counts[name]
            elif n_new > n_old:
                self.counts[name] += n_new - n_old
            mine = self.examples.get(name)
            mine_why = self.why_examples.get(name)
            old_why = old.why_examples.get(name, [])
            for key in drop:
                i = old_keys.index(key) if key in old_keys else -1
                why = old_why[i] if 0 <= i < len(old_why) else None
                if why:
                    self._why_sub(name, why)
                if mine:
                    js = [j for j, e in enumerate(mine) if _ex_key(e) == key]
                    if js:
                        j = js[0]
                        del mine[j]
                        if mine_why is not None and j < len(mine_why):
                            del mine_why[j]
            new_why = new.why_examples.get(name, [])
            for ex in add:
                try:
                    i = new.examples[name].index(ex)
                except ValueError:
                    i = -1
                why = new_why[i] if 0 <= i < len(new_why) else None
                exs = self.examples[name]
                if len(exs) < self.max_examples:
                    exs.append(ex)
                    self.why_examples[name].append(
                        {"val": why["val"], "validator": why["validator"],
                         "guard": why["guard"], "ctx_word": why["ctx_word"],
                         "ctx_pos": why["ctx_pos"]} if why else None)
                if why:
                    st = new.why_stats.get(name)
                    self._why_add(name, {"tier": st["tier"] if st else None,
                                         "validator": why["validator"], "guard": why["guard"],
                                         "ctx_word": why["ctx_word"]})
            if name in self.why_stats and name not in self.counts:
                del self.why_stats[name]
                self.why_examples.pop(name, None)
                self.examples.pop(name, None)

    def _why_sub(self, rule_name, why: dict) -> None:
        st = self.why_stats.get(rule_name)
        if st is None:
            return
        for key, val in (("validator", _tri(why.get("validator"), True)),
                         ("guard", _tri(why.get("guard"), False))):
            st[key][val] -= 1
            if st[key][val] <= 0:
                del st[key][val]
        w = why.get("ctx_word")
        if w is not None:
            w = str(w).casefold()
            if w not in st["ctx_words"]:
                w = _WHY_OTHER
            if w in st["ctx_words"]:
                st["ctx_words"][w] -= 1
                if st["ctx_words"][w] <= 0:
                    del st["ctx_words"][w]

    def trace(self) -> set:
        """Rules that left a trace in the report — a record (counter),
        a match error or the shadow of a discarded pass."""
        names = set(self.counts)
        names.update(e.get("rule_id") for e in self.rule_errors
                     if e.get("rule_id"))
        names.update(self.shadow)
        return names

    def shadow_of(self, discarded: "Report") -> None:
        """A pass with report ``discarded`` was discarded (rollback) — its trace
        goes into the shadow of this report."""
        self.shadow.update(discarded.trace())

    # ---- report state for the computation store --------------------------------
    def to_state(self) -> dict:
        """Full report state (JSON-compatible; the insertion order of
        counters and context words is kept — cap decisions depend on it
        when merging). from_state(to_state()) merges into another report exactly
        like the original object."""
        st = {
            "max_examples": self.max_examples, "want_why": self.want_why,
            "ctx_cap": getattr(self, "ctx_cap", _WHY_WORDS_CAP),
            "counts": [[k, n] for k, n in self.counts.items()],
            "examples": [[k, list(v)] for k, v in self.examples.items()],
            "why_examples": [[k, list(v)] for k, v in self.why_examples.items()],
            "why_stats": [[k, {"tier": st["tier"],
                               "validator": list(st["validator"].items()),
                               "guard": list(st["guard"].items()),
                               "ctx_words": list(st["ctx_words"].items())}]
                          for k, st in self.why_stats.items()],
            "files_changed": self.files_changed,
            "rule_errors": [dict(e) for e in self.rule_errors],
            "fusions_split": self.fusions_split,
            "view_homoglyph": self.view_homoglyph,
            **({"view_fold": self.view_fold} if self.view_fold else {}),
            **({"mail_attachments": self.mail_attachments} if self.mail_attachments else {}),
            **({"raw_guard": [dict(e) for e in self.raw_guard]} if self.raw_guard else {}),
            "shadow": sorted(self.shadow),
        }
        # The report of a store unit (units.py) used to lose the "fixpoint did not
        # converge" flag — in an incremental run the WARN and fixpoint_capped of a
        # reused unit were zero (from_state
        # also reads a store without the key).
        st["fixpoint_capped"] = self.fixpoint_capped
        return st

    @classmethod
    def from_state(cls, st: dict) -> "Report":
        rep = cls(max_examples=st["max_examples"], want_why=st["want_why"])
        rep.ctx_cap = st["ctx_cap"]
        for k, n in st["counts"]:
            rep.counts[k] = n
        for k, v in st["examples"]:
            rep.examples[k] = list(v)
        for k, v in st["why_examples"]:
            rep.why_examples[k] = list(v)
        for k, s in st["why_stats"]:
            rep.why_stats[k] = {"tier": s["tier"],
                                "validator": Counter(dict(s["validator"])),
                                "guard": Counter(dict(s["guard"])),
                                "ctx_words": Counter(dict(s["ctx_words"]))}
        rep.files_changed = st["files_changed"]
        rep.rule_errors = [dict(e) for e in st["rule_errors"]]
        rep.fusions_split = st["fusions_split"]
        rep.view_homoglyph = st.get("view_homoglyph", 0)   # older stores have no key
        rep.fixpoint_capped = st.get("fixpoint_capped", 0)  # older stores have no key
        rep.view_fold = st.get("view_fold", 0)             # key only when non-zero
        rep.mail_attachments = st.get("mail_attachments", 0)   # key only when non-zero
        rep.raw_guard = [dict(e) for e in st.get("raw_guard", ())]   # key only when non-empty
        rep.shadow = set(st["shadow"])
        return rep

    @property
    def total(self):
        return sum(self.counts.values())

    def why_json(self, text_fn=None, limit=None):
        """The "why" block for dry_report.json, or None if there is no why at all
        (old reports and reports without masks stay byte for byte the same). Format:
            {rule: {"tier": high|context|low,
                    "validator": {"ok": n, "fail": n, "none": n},
                    "guard":     {"ok": n, "fail": n, "none": n},
                    "ctx_words": [[word, n], …]  # top-WHY_TOP by (-n, word)
                    "examples":  [why|null, …]}}  # aligned with rules[rule].examples
            why = {"val": value, "validator": true|null, "guard": false|null,
                   "ctx_word": str|null, "ctx_pos": int|null}
        Determinism: rules and words are sorted, examples are in run
        order. text_fn — string conversion for humans (round._human_text:
        surrogates → U+FFFD); limit — the example cap (like [:30] of "rules")."""
        if not self.why_stats:
            return None
        fn = text_fn or (lambda s: s)
        out = {}
        for name in sorted(self.why_stats):
            st = self.why_stats[name]
            words = sorted(st["ctx_words"].items(),
                           key=lambda kv: (-kv[1], kv[0]))[:WHY_TOP]
            exs = list(self.why_examples.get(name, ()))
            if limit is not None:
                exs = exs[:limit]
            out[name] = {
                "tier": st["tier"],
                "validator": {k: st["validator"].get(k, 0)
                              for k in ("ok", "fail", "none")},
                "guard": {k: st["guard"].get(k, 0)
                          for k in ("ok", "fail", "none")},
                "ctx_words": [[fn(w), n] for w, n in words],
                "examples": [None if w is None else {
                    "val": fn(w["val"]), "validator": w["validator"],
                    "guard": w["guard"],
                    "ctx_word": (None if w["ctx_word"] is None
                                 else fn(w["ctx_word"])),
                    "ctx_pos": w["ctx_pos"]} for w in exs],
            }
        return out

    def render(self, dry=True) -> str:
        head = "DRY" if dry else "APPLIED"
        lines = [f"{head} — {self.total} replacements in {self.files_changed} files"]
        for name, n in self.counts.most_common():
            lines.append(f"  {name:<28} {n}")
            for e in self.examples[name][:3]:
                lines.append(f"       {e}")
        return "\n".join(lines)


def _ex_key(example) -> str:
    """nsjson (Report.swap_records): example key without context — «⟦value⟧  ->  token» (the
    record for a whole file and for one line have different contexts, the value and token are the same); an example
    of the old format («'v' -> TOKEN») — as is."""
    s = str(example)
    a, b = s.find("⟦"), s.rfind("⟧")
    if a < 0 or b < a:
        return s
    tok = s.rsplit("  ->  ", 1)
    return s[a:b + 1] + "  ->  " + (tok[1] if len(tok) == 2 else "")


def _example_token(example) -> str | None:
    """Replacement token from a dry_report example line («…⟦v⟧…  ->  TOKEN» or
    the old «'v' -> TOKEN»); None if there is no separator."""
    s = str(example)
    for sep in ("  ->  ", " -> "):
        if sep in s:
            return s.rsplit(sep, 1)[1]
    return None


def why_index(doc, by: str = "val") -> dict:
    """Loaded dry_report.json → {key: {rule, tier, validator, guard,
    ctx_word, ctx_pos}} from why examples. by="val" — key is the value
    (a join with a candidate literal: verdict_dataset/llm_verdict), by="token" —
    key is the replacement token from the aligned "rules" example (release_audit: why
    PERSON_NAME_12 stands in a zone). An old report without "why" / a broken block → {}
    (backward compatibility, not fail loudly: the reason is enrichment, not a
    contract). Determinism: rules are sorted, the first key wins."""
    if not isinstance(doc, dict):
        return {}
    why = doc.get("why")
    rules = doc.get("rules")
    if not isinstance(why, dict):
        return {}
    out: dict = {}
    for name in sorted(why):
        blk = why[name]
        if not isinstance(blk, dict):
            continue
        exs = blk.get("examples")
        if not isinstance(exs, list):
            continue
        rule_exs = (rules.get(name, {}).get("examples", [])
                    if isinstance(rules, dict)
                    and isinstance(rules.get(name), dict) else [])
        for i, w in enumerate(exs):
            if not isinstance(w, dict):
                continue
            if by == "token":
                key = _example_token(rule_exs[i]) if i < len(rule_exs) else None
            else:
                key = w.get("val")
            if not isinstance(key, str) or not key or key in out:
                continue
            out[key] = {"rule": name, "tier": blk.get("tier"),
                        "validator": w.get("validator"),
                        "guard": w.get("guard"),
                        "ctx_word": w.get("ctx_word"),
                        "ctx_pos": w.get("ctx_pos")}
    return out
