"""Control-value acceptance gate.

Scans the tree (content + filenames) for a pluggable pack of exact leak strings
that a fix pass was supposed to remove. The strings are corpus-specific, so the
pack is supplied by the caller (a file, one literal per line) — the *mechanism*
is general, the literals are not.
"""
from __future__ import annotations

import os
import re

from ..engine.boundary import L as _BL, R as _BR
from ..engine.walk import iter_files, max_file_bytes, nul_text_files, read_text

# Hit boundaries are THE SAME bricks as in the pack rules (boundary.L / boundary.R):
# the gate must find exactly what the rule was obliged to mask.
# A multi-word form >12 chars used to be searched as a bare substring while the rule
# masks on word boundaries → a superstring in the corpus ("Сергей Иванович" for the
# control "Сергей Иванов", "Michael Browning" for "Michael Brown") = an endless
# gate_control BLOCK with no way out. match(s, pos): the lookbehind sees chars
# BEFORE pos — the L semantics (JSONL escapes \n \t \r \b \f \v \uXXXX on the left
# = a line boundary, a `\` on the left is not a boundary) carry over one to one.
_L_RE = re.compile(_BL)
_R_RE = re.compile(_BR)

# A whole file in memory → a large one is scanned in chunks (memory O(chunk)),
# threshold ANON_MAX_FILE_BYTES (bytes; default 50 MB; parsed by
# walk.max_file_bytes). _CHUNK is the chunk size (a test lowers it to check the seams).
_CHUNK = 4 * 1024 * 1024
# result keys that are not control forms (a control line cannot start with '#':
# load_pack treats it as a comment) — they do not block
PATHS_KEY = "#paths"
SKIPPED_KEY = "#skipped"

# "ё" and "е" are one letter. The control gate, residual and the archive gate fold
# ё→е (Ё→Е) on both sides with a 1:1 replacement — positions and word boundaries stay the same;
# the pack (packgen) catches both spellings. ANON_YO_FOLD=0 — compare as is. The knob is read
# in one place — vnorm.yo_fold_on.
from ..engine.vnorm import YO_FOLD_ENV, yo_fold_on  # noqa: E402,F401
_YO = str.maketrans("ёЁ", "еЕ")


def yo_fold(s: str) -> str:
    """ё→е, Ё→Е — a char-for-char replacement (length and positions do not change)."""
    return s.translate(_YO)


# The round control gate, residual and the archive gate compare by VIEW — vnorm.fold_text
# on both sides (forms and text): ё→е (per ANON_YO_FOLD), horizontal whitespace and its runs →
# a space, dashes U+2010–U+2015/U+2212 → "-", invisible chars removed, NFC, stress marks over
# Cyrillic removed, Latin↔Cyrillic look-alikes. The folding is exactly what the pack
# (classes [её], [^\S\n]+, dashes — ANON_PACKGEN_WS_CLASS/DASH_CLASS) or the engine (fold_view,
# homoglyph_view) is guaranteed to remove: a form the gate finds on input is masked on output,
# and an "endless BLOCK" is impossible. A line break is not folded (by design).
# The VIEW_FOLD / VIEW_HOMOGLYPH knobs (the old ё→е 1:1-only comparison; folding without
# look-alikes) were removed together with the old paths in the control gate, residual, release and round.


def view_fold_fn(case: bool = False):
    """The check view folding with a map: ``f(s) -> (folded, PosMap|None)``. ``case`` —
    casefold (residual, archive)."""
    from ..engine.vnorm import fold_text
    yo = yo_fold_on()
    return lambda s: fold_text(s, case=case, yo=yo, homoglyph=True)


# In .json/.jsonl from json.dumps(ensure_ascii=True), PHP and JS, Cyrillic is written
# as \uXXXX and "/" as "\/"; the gate read the raw text and did not see that form. With
# ANON_GATE_JSON_UNESCAPE=1 a JSON file is scanned decoded: \uXXXX → the char, "\/" →
# "/"; other escapes ("\n", "\"", "\\") stay — the boundary.L word boundary sees them as before.
# A hit is counted once (the decoded view instead of the raw one).
# The view is vnorm.transport_view (the same as for the engine rules: surrogate pairs,
# "\" parity, and HTML — &#NNNN;/&#xHHHH;/&nbsp; in .html/.htm/.xml). The gate knob defaults to
# the view, as in the engine (the engine knob TRANSPORT_VIEW was removed, always the view): the
# engine masks decoded values and substitutes them into the original record, so a form in escapes
# on output is as much a leak as plain text, and an "endless BLOCK" is impossible (a repeated dry
# run over the output is empty). Set explicitly — taken as set: =0 — raw text (diagnostics).
# A large file (streamed in chunks) also goes by view, the view is built per chunk with an overlap
# (_stream_hits). A structured profile (a line-by-line Telegram export with \uXXXX) — the view on
# top of the structured.gate_text projection.
JSON_UNESCAPE_ENV = "ANON_GATE_JSON_UNESCAPE"
_VIEW_EXT = (".json", ".jsonl", ".ndjson", ".html", ".htm", ".xml")
# chunk tail that the next chunk may read differently: an unfinished or finished JSON escape
# (\, \u, \u04…, a high surrogate waiting for its pair) and an HTML entity (&, &#, &#x41,
# &nbs, &#1050 without ";")
_TAIL_JSON = re.compile(r"\\(?:u(?:[0-9a-fA-F]{0,3}|[0-9a-fA-F]{4}(?:\\(?:u[0-9a-fA-F]{0,3})?)?)?)?\Z")
_TAIL_HTML = re.compile(r"&(?:#(?:[xX]?[0-9a-fA-F]{0,7})?|n(?:b(?:s(?:p)?)?)?)?\Z")
# A numeric entity in .txt (without &nbsp;) and in JSON — together with the escape
# ("\" parity only for the "\" branch, see _pending_len). "\Z" holds both branches of the
# alternation — otherwise a "\" not at the tail ("\"" inside a JSON string) would be held in
# pending for up to 16 chars (same hits, extra delay)
_TAIL_ENT = re.compile(r"&(?:#(?:[xX]?[0-9a-fA-F]{0,7})?)?\Z")
_TAIL_JSON_ENT = re.compile("(?:" + _TAIL_JSON.pattern[:-2] + "|" + _TAIL_ENT.pattern[:-2] + r")\Z")


def json_unescape_on() -> bool:
    raw = os.environ.get(JSON_UNESCAPE_ENV, "").strip()
    if raw:
        return raw == "1"
    return True


def json_unescape_view(text: str, rel: str | None = None) -> str:
    """Text of the file ``rel`` in the transport view (vnorm.transport_view: \\uXXXX with surrogate
    pairs and "\\/" in JSON, numeric entities and &nbsp; in HTML; other escapes as is). ``rel`` None —
    as .json."""
    from ..engine.vnorm import transport_view
    return transport_view(rel if rel is not None else "x.json", text)[0]


def _view_rel(rel: str):
    """rel of a file the gate reads by view, otherwise None. A .txt with numeric entities is
    also read by view (vnorm.transport_rel_on)."""
    if not json_unescape_on():
        return None
    if rel.lower().endswith(_VIEW_EXT):
        return rel
    from ..engine.vnorm import transport_rel_on
    return rel if transport_rel_on(rel) else None


def load_pack(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip() and not ln.startswith("#")]


def _bounded_iter(ac, s: str):
    """(start, end, form) — Aho-Corasick hits that pass the word boundaries
    boundary.L/R (shared with the pack rules) — see bounded_hits."""
    for st, en, c in ac.iter(s):
        # '_' is a word char FOR THE GATE (stricter than the rule's R, which lets
        # '_' through): the gate is only more lenient than the pack — what the rule
        # masked is gone from the output, and "сыр_x" would be welded to the token
        # by the rule; the structural glue gate catches that, not the control gate
        if (st and s[st - 1] == "_") or not _L_RE.match(s, st):
            continue
        if (en < len(s) and s[en] == "_") or not _R_RE.match(s, en):
            continue
        yield st, en, c


def bounded_hits(ac, s: str):
    """Aho-Corasick hits with a SINGLE discipline for control targets — the
    boundary.L/R word boundaries FOR ALL forms, of any length and word count: pack
    rules mask only a whole word (L/CL … R/CR), so the gate must count the same
    thing as a word. A bare substring fires inside words ("Roma" in "Romashka",
    "ани" in "компании"; a stem inside a longer word; "Сергей Иванов" inside
    "Сергей Иванович"), and the rule does not mask it → an endless BLOCK. TOKENS
    glued to neighbours are the structural glue gate's job."""
    for _st, _en, c in _bounded_iter(ac, s):
        yield c


def _max_bytes() -> int:
    """Streaming scan threshold — parsed by walk.max_file_bytes: empty → default;
    <= 0 → 0, every non-empty file is streamed; not a number → a walk WARN (stderr,
    once per process) + default."""
    return max_file_bytes(nonpositive=0)


def _stream_hits(path: str, ac, maxlen: int, fold=None, view_rel=None, fold_map=None,
                 accept=None):
    """Hits over the file IN CHUNKS, without the whole text in memory — byte-for-byte
    the same as bounded_hits over the whole text (an oracle in the test). The chunk
    overlap W = the longest form + 6 (the L lookbehind is up to 6 chars: ``\\uXXXX``):
    a hit ending inside the overlap (en < keep) was already counted by the previous
    chunk; a hit that runs into the chunk end (en == len(buf)) before EOF is deferred —
    its right boundary is not visible yet, in the next chunk it ends exactly at keep.

    With ``view_rel`` set — the same algorithm IN VIEW COORDINATES: the overlap ov is
    the VIEW tail of the previous chunk (W view chars), the chunk is decoded
    (transport_view) and appended to it; between chunks only an unfinished escape is
    carried over raw (_TAIL_*: "\\u04", "\\ud83d\\ude0", a finished "\\uXXXX" — a
    high surrogate waiting for its pair, "&#10", "&nbs"; with "\\" parity — "\\\\u04"
    is not an escape), the next chunk reads it, at EOF it stays raw. ov + the chunk
    view is a continuous piece of the whole file's view (no cut escapes or position
    shifts), so keep and the deferred hit work as on the raw path. The 1:1 (ё/е)
    folding goes on top of the view.

    With ``fold_map`` set — the vnorm.fold_text folding over the buffer changes the
    length (whitespace runs, invisible chars, NFC): hits are searched in the folded
    buffer, ``keep`` moves into its coordinates as the folded length of the overlap
    (_keep_folded: folding is local, and the folding of ``ov`` is a prefix of the
    folding of ``ov + chunk`` everywhere except the last char — and a hit ending on
    it was deferred by the previous chunk), a deferred hit is ``en == len(folded)``;
    the overlap is taken raw — the buffer piece under the last W folded chars, but
    not longer than the cap 4·maxlen + 6 (memory O(chunk)): if the last W folded
    chars hold more than 3·maxlen invisible and extra whitespace chars, a form on the
    seam may be missed or counted twice.

    ``accept`` (ANON_ROUND_CONTROL_POLICY, round._control_policy) — a hit filter
    ``accept(group_key, buffer, start, end) -> bool`` in the coordinates of the buffer
    the automaton ran over (folded with fold/fold_map); False — the hit is not counted.
    None — the old path, byte for byte."""
    W = maxlen + 6
    W_RAW = 4 * maxlen + 6
    view = None
    if view_rel is not None:
        from ..engine.vnorm import transport_view
        html = view_rel.lower().endswith((".html", ".htm", ".xml"))
        json_ = view_rel.lower().endswith((".json", ".jsonl", ".ndjson"))
        if html:
            tail_rx, parity = _TAIL_HTML, False
        elif json_:
            tail_rx, parity = _TAIL_JSON_ENT, True       # an entity on the chunk seam
        else:
            tail_rx, parity = _TAIL_ENT, False
        view = lambda s: transport_view(view_rel, s)[0]      # noqa: E731
    ov, pending = "", ""
    with open(path, encoding="utf-8", errors="surrogateescape",
              newline="") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            eof = not chunk
            if view is None:
                buf, keep = ov + chunk, len(ov)
            else:
                raw = pending + chunk
                n = 0 if eof else _pending_len(raw, tail_rx, parity)
                pending = raw[len(raw) - n:] if n else ""
                buf, keep = ov + view(raw[:len(raw) - n]), len(ov)
            if fold_map is not None:
                folded, m = fold_map(buf)
                keep_f = _keep_folded(fold_map, ov)
                for st, en, c in _bounded_iter(ac, folded):
                    if en < keep_f or (en == len(folded) and not eof):
                        continue
                    if accept is not None and not accept(c, folded, st, en):
                        continue
                    yield c
                if eof:
                    return
                cut = max(0, len(folded) - W)
                src_cut = cut if m is None else m.start(cut)
                ov = buf[max(src_cut, len(buf) - W_RAW):]
                continue
            hay = fold(buf) if fold else buf
            for st, en, c in _bounded_iter(ac, hay):
                if en < keep or (en == len(buf) and not eof):
                    continue
                if accept is not None and not accept(c, hay, st, en):
                    continue
                yield c
            if eof:
                return
            ov = buf[-W:]


def _keep_folded(fold_map, ov: str) -> int:
    """``keep`` of the stream in folded coordinates — the folded length of the overlap ``ov``
    (hits ending before it were counted by the previous chunk). Not the view position under
    len(ov) (vnorm.view_pos): a run of only invisible chars at the end of the previous chunk
    gave nothing, and with a space from the new chunk it becomes a view space — a position
    "after the run" would skip the deferred hit before it."""
    return len(fold_map(ov)[0]) if ov else 0


def _pending_len(raw: str, rx, parity: bool) -> int:
    """Length of the ``raw`` tail that the next chunk may read differently (an unfinished
    or finished escape/entity — _TAIL_*); 0 — no tail. ``parity`` — in JSON a "\\"
    before the found "\\" escapes it (an odd run of "\\" on the left is not an escape)."""
    pos = max(0, len(raw) - 16)
    while True:
        m = rx.search(raw, pos)
        if m is None:
            return 0
        i = m.start()
        if parity and raw[i] == "\\":        # the "&#…" entity branch has no "\" parity
            k = 0
            while i - k - 1 >= 0 and raw[i - k - 1] == "\\":
                k += 1
            if k % 2:                    # "\\u04": an escaped backslash — look for a shorter tail
                pos = i + 1
                continue
        return len(raw) - i


def run(root: str, controls: list[str],
        path_controls=None, accept=None) -> tuple[bool, dict]:
    """(ok, residual). residual: form → hit count (content + paths); plus the
    service keys PATHS_KEY (form → hits ONLY in rel paths, WARN, not BLOCK) and
    SKIPPED_KEY (rel paths that could not be read). ok=False only from blocking
    hits: in content — always; in paths — only for forms from ``path_controls``
    (those the round rename pass can fix: org/domain literals of its org_map).
    Other categories in paths (a handle, a name in
    "Direct Messages - ivan_petrov [123].json", a dir "ipetrov") are not fixed by
    the rename pass — a BLOCK for them could not be cleared: WARN + a record.

    ``accept`` (ANON_ROUND_CONTROL_POLICY, round._control_policy: the case policy of
    the control_case.tsv pack) — a hit filter IN CONTENT: ``accept(form, text, start,
    end) -> bool`` in the coordinates of the text the automaton ran over (the file
    view, folded with ANON_YO_FOLD); False — the hit is not counted (neither in
    residual nor in blocking). Hits in paths skip the filter (as in residual: the
    policy is about text). None — the old path, byte for byte."""
    # Aho-Corasick instead of a naive `c in text` × N: the gate's profile is "literals
    # are ABSENT" (a cleanliness check), where a naive loop scans every file to the
    # end for every literal — O(files×literals×size). The automaton makes one pass
    # per file for any number of controls (bench: 20K literals × 5 MB: naive 38.5s →
    # AC 0.18s) and, unlike a regex alternation, reports ALL nested literals
    # ("Иванов" inside "Иванова") — the counters are exact.
    from ..engine.ac import Automaton
    hits = {c: 0 for c in controls}
    path_warn: dict = {}
    skipped: list = []
    blocking = 0
    path_controls = set(path_controls or ())
    maxlen = max((len(c) for c in controls), default=0)
    cap = _max_bytes()
    # ё/е: the automaton runs over folded forms, the text is folded; a hit of a folded form is a
    # hit of every control form with that folding ("Фёдоров" and "Федоров" are one). A group is an
    # ordered set: a repeated line in the control file is one form, as in the automaton.
    # The vnorm.fold_text folding on both sides — forms and text; hits are only counted,
    # the map is not needed in memory, the stream (fold_map) needs it (keep in folded coordinates)
    fold_map = view_fold_fn(case=False)
    fold = lambda s: fold_map(s)[0]              # noqa: E731
    groups: dict = {}
    for c in controls:
        groups.setdefault(fold(c) if fold else c, {})[c] = None
    ac = Automaton(list(groups)) if controls else None

    def _hits(s):
        for k in bounded_hits(ac, fold(s) if fold else s):
            yield from groups[k]

    def _text_hits(s):
        """Hits in content: with ``accept`` — with positions and a per-form filter, otherwise _hits."""
        if accept is None:
            yield from _hits(s)
            return
        hay = fold(s) if fold else s
        for st, en, k in _bounded_iter(ac, hay):
            for c in groups[k]:
                if accept(c, hay, st, en):
                    yield c

    def _stream_accept(k, hay, st, en):
        # the stream yields the group key (the folded form); the hit counts if the filter
        # passes at least one form of the group — below all forms of the group are counted, as before
        return any(accept(c, hay, st, en) for c in groups[k])

    for path, rel in iter_files(root):
        if ac is None:
            break
        for c in _hits(rel):
            hits[c] += 1
            if c in path_controls:
                blocking += 1
            else:
                path_warn[c] = path_warn.get(c, 0) + 1
        # A file of a structured profile (a Telegram export) is checked by its
        # free-text fields — the profile does not touch service fields, and a
        # literal matching their value would be a BLOCK that cannot be cleared
        # (structured.gate_text). A parse error — the file goes the usual way.
        # A JSONL profile (a YouTrack export): the gate reads the RAW record — as
        # without the profile (above the cap — streamed by _stream_hits): no field
        # value, service ones included, bypasses the gate; the residue is cleaned by fix-engine
        try:
            from .. import structured
            view = None if structured.is_jsonl_file(path, rel) \
                else structured.gate_text(path, rel)
        except Exception as e:           # noqa: BLE001 — the gate must not crash
            print(f"gate_control: WARN {rel}: profile not parsed ({e}) — "
                  "scanning as plain text", flush=True)
            view = None
        if view is not None:
            # the line-by-line layout of a Telegram export gives raw lines with \uXXXX —
            # the projection goes by view too
            vr = _view_rel(rel)
            if vr is not None:
                view = json_unescape_view(view, vr)
            for c in _text_hits(view):
                hits[c] += 1
                blocking += 1
            continue
        try:
            size = os.stat(path).st_size
        except OSError:
            size = -1                    # read_text itself sorts out an unreadable file
        # A .eml/.mbox mail — the formats.mail_text view (decoded headers and
        # bodies), as in the engine; if parsing fails — the plain scan below (WARN in mail_gate_text).
        # Above the cap — a streamed plain scan, as for any file (the whole mail is not kept in memory)
        if size <= cap:
            try:
                from ..formats import mail_gate_text
                view = mail_gate_text(path, rel)
            except (OSError, UnicodeError) as e:
                print(f"gate_control: WARN {rel}: not read ({e}) — skipped", flush=True)
                skipped.append(rel)
                continue
            if view is not None:
                for c in _text_hits(view):
                    hits[c] += 1
                    blocking += 1
                continue
        if size > cap:
            print(f"gate_control: WARN {rel}: {size} bytes > "
                  f"ANON_MAX_FILE_BYTES={cap} — streaming scan in chunks",
                  flush=True)
            try:
                for k in _stream_hits(path, ac, maxlen, fold, view_rel=_view_rel(rel),
                                      fold_map=fold_map,
                                      accept=None if accept is None else _stream_accept):
                    for c in groups[k]:
                        hits[c] += 1
                        blocking += 1
            except (OSError, UnicodeError) as e:
                print(f"gate_control: WARN {rel}: not read ({e}) — "
                      f"skipped", flush=True)
                skipped.append(rel)
            continue
        text = read_text(path)
        if text is None:
            if size >= 0:                # stat worked, reading did not — record it
                skipped.append(rel)
            continue
        vr = _view_rel(rel)
        if vr is not None:
            text = json_unescape_view(text, vr)
        for c in _text_hits(text):
            hits[c] += 1
            blocking += 1
    # A text file with NUL in its head was sniffed out by the walk — the gate did not read it;
    # not silently: into SKIPPED_KEY (WARN in render), like any unread file
    for _p, rel in nul_text_files(root):
        print(f"gate_control: WARN {rel}: NUL in the head of a text file — not read "
              "(binary sniff), skipped", flush=True)
        skipped.append(rel)
    residual = {c: n for c, n in hits.items() if n}
    if path_warn:
        print("gate_control: WARN control forms in paths that the rename pass "
              "does not fix (not BLOCK): "
              + ", ".join(f"{c!r}={n}" for c, n in path_warn.items()),
              flush=True)
        residual[PATHS_KEY] = path_warn
    if skipped:
        residual[SKIPPED_KEY] = skipped
    return (blocking == 0), residual


def render(ok: bool, residual: dict) -> str:
    # blocking hits of a form = all its hits minus WARN hits in paths
    paths = residual.get(PATHS_KEY) or {}
    forms = ", ".join(f"{c!r}={n - paths.get(c, 0)}" for c, n in residual.items()
                      if not c.startswith("#") and n - paths.get(c, 0) > 0)
    extra = ""
    if residual.get(PATHS_KEY):
        extra += "; paths (WARN): " + ", ".join(
            f"{c!r}={n}" for c, n in residual[PATHS_KEY].items())
    if residual.get(SKIPPED_KEY):
        extra += f"; skipped (WARN): {len(residual[SKIPPED_KEY])} file(s)"
    if ok:
        return "PASS — 0 blocking control residuals" + extra
    return "FAIL — residuals: " + forms + extra
