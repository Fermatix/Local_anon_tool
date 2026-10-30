"""Filename / directory-name detector — the leak class the content pipeline ignores.

The content pass only ever masks *inside* a file. But a personal surname, a
first-last slug, a Google-Doc-ID or a plaintext catalog label can survive in the
FILE or DIRECTORY NAME itself (``CV-Ivan-Petrov.pdf``, ``Отчет_Иванова.md``,
``doc__1AbCdEfGh2345XyZ.md``, ``discord/Виджеты/``). This module masks only the
name-bearing part of each such name and keeps everything structural — the
extension, a trailing hash, already-placed ``*_NNN`` tokens, and every separator
(``.`` ``_`` ``-`` ``__`` space).

Design mirrors the rest of the engine:

  * one entity -> one token: surnames / first-last slugs seed ``mint.pname`` on a
    canonical (lowercase, de-slugged) key; Google-Doc-IDs seed ``mint.tag("DOC_ID")``;
    catalog labels seed ``mint.tag("ORG")`` — so a name in a filename collapses to
    the SAME token the content pass would have minted for it.
  * segment discipline: a name is split on separators, every segment classified,
    and only the sensitive segments are rewritten in place (r17_filenames /
    fixpath2 primitive). Extensions and hash segments are never touched.
  * FP-safety paramount (the tree is full of look-alikes: ``ivanhoe``, ``petroleum``,
    ``Alexandria``, ``report.md``, ``Dublin``): a pure-token filename is skipped
    whole; a STOP-list of legit words that merely END in a surname suffix
    (``login``, ``domain``, ``admin``, ``Berlin``, ``plugin`` …) is never masked;
    a bare Latin stem needs a NAME SIGNAL: an unambiguous translit family
    ending (-ov/-ev/-ova/-enko/-sky …) stands on its own, but the short
    ``-in/-yn`` tail (``plain``, ``checkin``, ``mountain`` — 2300+ English
    words, an open class no stop-list closes) only counts next to a known
    given name in the SAME name (``Ivan_Ilyin.pdf``), mirroring
    ``names._lat_pair_guard``; a Doc-ID must be mixed-case AND contain a digit
    AND not already be a token.
  * rename discipline (r15_catalogs / fixpath2): the plan is collected first over
    a full ``os.walk``; the engine then renames files, then dirs BOTTOM-UP, so a
    parent dir is never renamed out from under a child mid-walk. Dry-run aware.

Exposes both shapes the pipeline can consume:
    ``mask_name(name, is_dir=False) -> str``   the pure name-transform
    ``plan_renames(root) -> list[(old_abs, new_abs)]``   the walk + plan
"""
from __future__ import annotations

import os
import re

from ..engine import mint
from ..engine.boundary import _PREFIX_ALT, TOKEN_RE
from ..engine.gazetteer import EN_FIRST_NAMES, RU_FIRST_NAMES
from ..engine.vocab import COMMON_EN, SURNAME_ENDINGS, TECH_TERMS, TRANSLIT
from .names import _canon_lat, _lat_first_key
from . import orgs as _orgs

# --------------------------------------------------------------------------- #
# stop-lists — legit words that must never be masked
# --------------------------------------------------------------------------- #

# Words that END in a productive surname suffix (…ин/…ов/…ко/…off/…ман …) but are
# ordinary vocabulary, not people. Matched case-insensitively, whole-segment.
_SUFFIX_STOP = {
    # -in / -ин words
    "login", "admin", "domain", "plugin", "margin", "origin", "begin", "join",
    "main", "coin", "chain", "brain", "again", "berlin", "dublin", "kremlin",
    "bulletin", "min", "linkedin", "gitin", "bitcoin", "protein", "vitamin",
    "checkin", "signin", "builtin", "stdin", "mixin", "blockchain", "toolchain",
    "keychain",
    "мин", "магазин", "бюллетень",
    # -ov / -off words
    "info", "video", "photo", "logo", "demo", "repo", "memo", "promo", "typo",
    "off", "takeoff", "backoff", "kickoff", "payoff", "handoff", "logoff",
    "cutoff", "dropoff", "spinoff", "signoff", "layoff", "runoff", "shutoff",
    "knockoff", "setoff", "tradeoff", "playoff", "standoff", "liftoff",
    "sendoff", "turnoff", "writeoff", "oneoff", "improv",
    # -ev / -ova / -eva / -ina words (English homographs of translit endings)
    "prev", "abbrev", "elev", "kiev", "anova", "supernova", "cordova", "geneva",
    "china", "retina", "stamina", "angina", "ballerina", "cantina", "alumina",
    "patina", "latina", "carolina", "argentina",
    # -sky words
    "risky", "husky", "whisky", "pesky", "frisky", "dusky", "musky", "bluesky",
    # -ko / -ук / -ец words
    "echo", "gecko", "disco", "django", "mango", "turbo", "banco",
    # -man words
    "human", "roman", "german", "woman", "batman", "postman", "husman",
    "freeman", "layman", "salesman", "iron", "roadmap",
    # -ich / -ых
    "which", "rich", "switch", "match", "search", "branch", "launch",
    # generic report/doc words that could look name-ish
    "readme", "changelog", "license", "manifest", "index", "config",
    "settings", "package", "release", "backup", "archive", "draft",
    "final", "version", "summary", "report", "notes", "meeting", "agenda",
    "invoice", "receipt", "budget", "roadmap", "overview", "template",
    "отчет", "отчёт", "договор", "смета", "план", "заметки", "итоги",
    "презентация", "документ", "справка", "приказ", "протокол",
}

# Segments that are structural, never a name.
_KEEP_WORDS = {
    "cv", "nda", "resume", "doc", "docs", "file", "files", "attachment",
    "img", "image", "screenshot", "screen", "photo", "pic", "final", "draft",
    "v1", "v2", "v3", "copy", "new", "old", "tmp", "temp", "backup",
    "frontend", "backend", "fullstack", "account", "user", "id", "pm",
    "intern", "study", "guide", "edu", "conf", "hr", "pr", "ai", "css", "money",
    # typical file-name stems («plain.txt» went PERSON_NAME_… — the stem
    # ends in «-in»); also shield the FIRST word of a hyphen slug, so
    # ``export-ivanov`` masks only the surname (one entity -> one token)
    "plain", "export", "log", "sample", "todo", "contract", "list", "table",
    "chart", "audio", "dump", "output", "input", "result", "results", "stats",
    "stat", "meta", "dist", "src", "lib", "bin", "etc",
}
_KEEP_WORDS |= {w.lower() for w in COMMON_EN} | {w.lower() for w in TECH_TERMS}

# Common Latin transliterated given names + their -ov/-in/-enko surname families
# are what we WANT to catch; these are handled by morphology below, not a list.

# --------------------------------------------------------------------------- #
# segment classifiers
# --------------------------------------------------------------------------- #

# Any already-placed mask token (PERSON_NAME_1, ORG_12, DOC_ID_34, *_REDACTED).
_IS_TOKEN = re.compile(r"^(?:" + TOKEN_RE.pattern + r")$")

# A hex-ish hash chunk (md5/sha fragment) — kept verbatim (identifies the file).
_IS_HASH = re.compile(r"^[0-9a-f]{16,}$")

# Cyrillic surname stem + productive family ending + case tail (Иванова, Петров).
_SURN_TAIL = r"(?:а|у|е|ы|ой|ою|ым|ом|ых|им|его|ого|ему|ому|)"
_CYR_SURNAME = re.compile(
    r"^([А-ЯЁ][а-яё]+" + SURNAME_ENDINGS + _SURN_TAIL + r")$")

# Latin transliterated surname family endings (translit of -ов/-ев/-ин/-ский/…).
_LAT_SURNAME = re.compile(
    r"(?i)^([a-z]{2,}(?:ov|ev|in|yn|sky|skiy|skaya|tsky|enko|uk|yuk|chuk|"
    r"yan|dze|shvili|off|ova|eva|ina))$")

# A first-last slug: Ivan-Petrov / maria-smirnova / Oleg-Sidorov.
# Two alpha words joined by '-', the SECOND looking like a surname; an optional
# trailing digit disambiguator (…547) is folded into the same person's key.
_FIRSTLAST = re.compile(
    r"(?i)^([a-zа-яё]{2,})-([a-zа-яё]{2,})(\d{0,4})$")

# Google-Doc / Sheet ID: 25-50 chars of [A-Za-z0-9_-], MIXED case AND a digit
# (that mix is what distinguishes a real ID from a slug or a token). fixpath2.
_DOCID = re.compile(r"^[A-Za-z0-9_-]{25,50}$")

# Family endings that are AMBIGUOUS on a bare Latin stem: a dictionary scan
# (/usr/share/dict/words) finds 2300+ English words in «-in» (plain, mountain,
# captain, checkin …) against a handful for every other ending (cutoff, risky,
# china — all in _SUFFIX_STOP). Same split names._lat_pair_guard makes.
_LAT_WEAK_END = ("in", "yn")


def _lat_key(seg: str) -> str:
    """Canon key of a name segment for the given-name cue — the same as
    ``names._lat_first_key``: TRANSLIT over Cyrillic → x→ks → canon →
    squeeze after y→i (Дмитрий ``dmitrii`` / Dmitry ``dmitri``)."""
    return _lat_first_key(seg)


# Canon keys of known given names (EN gazetteer + translit RU gazetteer).
_GIVEN_KEYS = frozenset(_lat_key(w) for w in EN_FIRST_NAMES | RU_FIRST_NAMES)


def _given_names(pieces) -> frozenset:
    """Lowercased segments of ONE name that are known given names — the cue a
    «-in/-yn» stem needs (``Ivan_Ilyin.pdf``, ``Sergey-Ilyin.pdf``)."""
    return frozenset(p.lower() for p in pieces if p and _lat_key(p) in _GIVEN_KEYS)


def _lat_surname(seg: str, cue=frozenset()) -> bool:
    """A bare Latin segment is a transliterated surname when it carries a
    family ending, is not a dictionary / tech / file word, and — for the
    ambiguous «-in/-yn» tail — ANOTHER segment of the name is a given name."""
    low = seg.lower()
    if not _LAT_SURNAME.match(seg) or low in _KEEP_WORDS or low in _SUFFIX_STOP:
        return False
    if low.endswith(_LAT_WEAK_END) and not (cue - {low}):
        return False
    return True


def _has_lower(s):
    return bool(re.search(r"[a-z]", s))


def _has_upper(s):
    return bool(re.search(r"[A-Z]", s))


def _has_digit(s):
    return bool(re.search(r"\d", s))


def _is_docid(seg: str) -> bool:
    """A Google-Doc-ID: 25-50 chars, mixed case, has a digit, not a token/word."""
    if not _DOCID.fullmatch(seg):
        return False
    if not (_has_lower(seg) and _has_upper(seg) and _has_digit(seg)):
        return False
    if _IS_TOKEN.match(seg):
        return False
    # a plain SCREAMING_SNAKE or a token-prefixed id is not a doc id
    if re.match(r"^(?:PERSON|ORG|USER|GEO|EVENT|PHONE|EMAIL|DOC_ID|RESOURCE|"
                r"HANDLE|DATE|IP|SECRET)", seg):
        return False
    return True


# --- Google id with separators INSIDE -----------------------------------------
# The Google id alphabet is base64url: "_" and "-" occur inside the id itself (in
# a gdocs corpus 3,656 of 4,032 names with a 44-char id). mask_name splits a name on
# "_" and "-" BEFORE classification, so such an id fell apart into pieces: a piece
# of ≥ 25 chars went under DOC_ID while the tail stayed ("…DOC_ID_NNN_<tail>"),
# or all pieces were shorter than 25 — and the whole id stayed in the file name
# (audit findings). Now, before the pieces are classified, a WINDOW of
# adjacent pieces joined by a single "_" (a hyphen lives inside a piece) is searched
# that looks like a Google id as a whole. So that a CamelCase name with digits
# ("MyProjectV2_FinalReportQ3_DraftV1") does not become an id, a window is accepted only
# at the EXACT length of a Google id (44 — Docs/Sheets/Slides, 33 and 28 — Drive files),
# with three character classes and "randomness" (frequent class changes between adjacent
# characters). An id without separators is still taken by _is_docid (25–50).
_GID_LEN = (44, 33, 28)
_GID_RUN = re.compile(r"^[A-Za-z0-9_-]+$")
_MINT_IN = re.compile(
    r"(?:PERSON_NAME|PERSON|ORG|USER_ID|USER|HANDLE|RESOURCE|DOC_ID)_\d"
    r"|_REDACTED")
_RAND_MIN = 0.30
_RAND_MIN_STRUCT = 0.20     # a window at the structural place of an id ("…__<id>")
_INNER_DIGITS = re.compile(r"(?<=[A-Za-z])\d+(?=[A-Za-z])")


def _cls(ch: str) -> str:
    return "d" if ch.isdigit() else ("u" if ch.isupper() else "l")


def _rand_rate(s: str) -> float:
    """Share of adjacent alphanumeric pairs with a class change (lower/
    upper/digit); separators break a pair. A random base64url id has
    ~0.5–0.6, a snake/camel name ("Meeting_notes_2024_Q3") — 0.1–0.25."""
    pairs = diff = 0
    prev = None
    for ch in s:
        if not ch.isalnum():
            prev = None
            continue
        c = _cls(ch)
        if prev is not None:
            pairs += 1
            diff += c != prev
        prev = c
    return diff / pairs if pairs else 0.0


def _is_gid_run(run: str, structural: bool = False, lens=_GID_LEN) -> bool:
    """A Google id with "_"/"-" inside: exact length, three character classes,
    randomness, no minted token inside.

    ``structural`` — the window has strong evidence: it stands at the id's place
    by the export layout ("<title>__<id>", "<title> _ <id>") or one
    of its pieces ITSELF passes _is_docid (≥ 25 chars — the old logic minted
    DOC_ID on it anyway, leaving the rest of the id next to it). Then a
    soft randomness threshold is enough. Without evidence (the id is welded to the title
    by a single "_", all pieces are short) digits INSIDE a letter
    context ("r5z", "w69N") are also required: a CamelCase name has digits at word edges
    ("ProjectV2_", "Q3_"), and "MyProjectV2_FinalReportQ3_DraftV1" (exactly 33
    chars, frequent case changes) would otherwise be taken for a Drive id. The criterion
    cannot be made strict for ALL positions: 74 real ids of the corpus have 0–1 such
    digits (digits stand at separators and edges)."""
    if len(run) not in lens or not _GID_RUN.match(run):
        return False
    if "_" not in run and "-" not in run:
        return False                      # no separators — _is_docid's job
    if not (_has_lower(run) and _has_upper(run) and _has_digit(run)):
        return False
    if _MINT_IN.search(run) or not run[0].isalnum() or not run[-1].isalnum():
        return False
    rate = _rand_rate(run)
    if structural:
        return rate >= _RAND_MIN_STRUCT
    need = 2 if len(run) >= 40 else 1
    return len(_INNER_DIGITS.findall(run)) >= need and rate >= _RAND_MIN


_UNDERSCORES = re.compile(r"^_+$")


# a token in a name — ONLY by known minting prefixes (boundary._PREFIX_ALT, like
# OWN_TOKEN_RE): the wide TOKEN_RE recognizes a "token" in any
# "CAPITALS_digits", and an upper-case transliterated surname "CV_PETROV_2023" stayed
# open in all trees, while a Google id piece "…ecV_4_…" broke the id window (half
# the id open, a repeated mask_name gave a replacement). The "_digits" tail (up to 8 times) does not
# capture the digit head of the next piece ("PERSON_NAME_1_1AbC…<Google id>"
# — the token is only "PERSON_NAME_1", the id stays a piece and goes under DOC_ID)
_TOKEN_IN = re.compile(
    r"(?<![A-Za-z0-9])(?:" + _PREFIX_ALT + r")_(?:\d{1,32}(?:_\d{1,32}){0,7}|REDACTED)"
    r"(?![A-Za-zА-Яа-яЁё0-9])")


def _has_token(piece: str) -> bool:
    """A piece is an already placed token or contains one ("PERSON_NAME_1234567",
    "«PERSON_NAME_1234567»", "DOC_ID_12-abc"): an id window does not start at it and
    does not extend through it."""
    return _TOKEN_IN.search(piece) is not None


def _split_keep_tokens(stem: str) -> list:
    """_SEP_SPLIT.split, but an already placed token stays ONE piece. Splitting on
    "_" cut "PERSON_NAME_1234567" into "PERSON", "NAME", "1234567", and a Google id
    window started at "NAME_1234567_<random piece>" (Telegram,
    an attachment name after the first pass) was folded into DOC_ID — mask_name on
    its own output gave a replacement (the round idempotence oracle) and broke the
    person token. Even indices are pieces, odd ones separators, as in
    _SEP_SPLIT.split; without tokens — the same list."""
    parts = _SEP_SPLIT.split(stem)
    spans = [m.span() for m in _TOKEN_IN.finditer(stem)]
    if not spans:
        return parts
    ends, pos = [], 0
    for part in parts:
        pos += len(part)
        ends.append(pos)

    def piece_at(off: int) -> int:
        return next(k for k, e in enumerate(ends) if e > off)

    # a token starts with a capital and ends with a digit/REDACTED — always in pieces,
    # not in separators: joining pieces k1..k2 keeps the alternation
    glue = [False] * len(parts)
    for a, b in spans:
        for k in range(piece_at(a), piece_at(b - 1)):
            glue[k] = True
    out, buf = [], ""
    for k, part in enumerate(parts):
        buf += part
        if not glue[k]:
            out.append(buf)
            buf = ""
    return out


def _has_docid_piece(chunk: str) -> bool:
    """A piece (or its part between hyphens) itself passes _is_docid."""
    return _is_docid(chunk) or any(_is_docid(p) for p in chunk.split("-"))


def _fold_pass(parts: list, structural_pass: bool, lens=_GID_LEN):
    out, folded, i, n = [], False, 0, len(parts)
    while i < n:
        if i % 2 == 1:                    # separator
            out.append(parts[i])
            i += 1
            continue
        sep = parts[i - 1] if i > 0 else ""
        # the id's place by the export layout: "…__<id>", "….__<id>", "… _ <id>"
        structural = "__" in sep or (sep.strip() == "_" and sep != "_")
        if structural != structural_pass or _has_token(parts[i]):
            out.append(parts[i])
            i += 1
            continue
        strong = structural or _has_docid_piece(parts[i])
        joined, best, j = parts[i], None, i
        if _is_gid_run(joined, strong, lens):
            best = (i, joined)
        while j + 2 < n and len(joined) < max(lens):
            nsep = parts[j + 1]
            # from a structural start an id may contain "__" inside; from
            # an ordinary one — only a single "_": otherwise a window started in the TITLE
            # stepped over "__" and took the start of the id ("АВ__1r5z…" of exactly 28
            # chars was taken for a Drive id, the tail of the real id stayed)
            if not (_UNDERSCORES.match(nsep) if structural else nsep == "_"):
                break
            if _has_token(parts[j + 2]):
                break
            joined += nsep + parts[j + 2]
            j += 2
            strong = strong or _has_docid_piece(parts[j])
            if _is_gid_run(joined, strong, lens):
                best = (j, joined)
        if best is not None:
            out.append(mint.tag("DOC_ID", best[1]))
            folded = True
            i = best[0] + 1
        else:
            out.append(parts[i])
            i += 1
    return out, folded


def _fold_gid_runs(parts: list):
    """Fold windows "piece(_piece)*" that look like a Google id as a whole into one
    DOC_ID token. ``parts`` is the result of _SEP_SPLIT.split (even — pieces,
    odd — separators); a hyphen lives inside a piece. Three passes in
    decreasing confidence: (1) windows from a STRUCTURAL start (after "__" — there the id
    stands by the export layout) of exactly 44 — Docs/Sheets ids, the vast
    majority; (2) a structural start, lengths 33/28 (Drive files); (3) other
    starts (the id welded to the title by a single "_"). The order matters: a window
    started in the last word of the title («Отчёт _ План__<id>») would otherwise collect
    33 chars from the word and the start of the id before the real id got its 44.
    Returns: (new parts, whether anything was folded)."""
    parts, f1 = _fold_pass(parts, True, _GID_LEN[:1])
    parts, f2 = _fold_pass(parts, True, _GID_LEN[1:])
    parts, f3 = _fold_pass(parts, False)
    return parts, f1 or f2 or f3


# An id tail/head welded to an ALREADY placed DOC_ID token ("DOC_ID_NNN_<tail>",
# "<head>_DOC_ID_NNN-<tail>" — names renamed by older versions): a random
# alphanumeric piece of 4–24 chars separated by "_"/"-" from the token is the rest of the
# same id and goes under its own DOC_ID (the name stays unique).
_DOCID_TOK = re.compile(r"(DOC_ID_\d+)")
_TAIL_AFTER = re.compile(r"^([_-])([A-Za-z0-9]{4,24})(?![A-Za-z0-9])")
_HEAD_BEFORE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9]{4,24})([_-])$")


def _id_fragment(piece: str) -> bool:
    classes = {_cls(c) for c in piece}
    return len(classes) >= 2 and _rand_rate(piece) >= _RAND_MIN \
        and piece.lower() not in _KEEP_WORDS and piece.lower() not in _SUFFIX_STOP


def _mask_docid_tails(stem: str) -> str:
    """Up to a fixpoint (≤ 4 passes): a minted remainder itself becomes a
    DOC_ID token, and the next id piece ("…_DOC_ID_N_DOC_ID_M_<one more piece>")
    is picked up by the next pass."""
    for _ in range(4):
        segs = _DOCID_TOK.split(stem)
        if len(segs) < 3:
            return stem
        for k in range(1, len(segs), 2):      # odd — DOC_ID tokens
            after = _TAIL_AFTER.match(segs[k + 1])
            if after and _id_fragment(after.group(2)):
                segs[k + 1] = (after.group(1) + mint.tag("DOC_ID", after.group(2))
                               + segs[k + 1][after.end():])
            before = _HEAD_BEFORE.search(segs[k - 1])
            if before and _id_fragment(before.group(1)):
                segs[k - 1] = (segs[k - 1][:before.start()]
                               + mint.tag("DOC_ID", before.group(1)) + before.group(2))
        new = "".join(segs)
        if new == stem:
            break
        stem = new
    return stem


def _canon_name(seg: str) -> str:
    """Canonical person-key for a slug segment: lowercased, hyphens->spaces,
    trailing digit-disambiguator dropped (so ``maria-smirnova547`` and
    ``maria-smirnova`` collapse to one token)."""
    s = seg.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\d+", "", s)
    return " ".join(s.split())


def _classify_segment(seg: str, is_dir: bool, cue=frozenset()):
    """Return (kind, token) for a single name segment, or (None, None) to keep it.

    kinds: 'docid' -> DOC_ID token, 'name' -> pname token, 'label' -> ORG token.
    ``cue`` — given-name segments of the whole name (see :func:`_given_names`).
    """
    if not seg:
        return None, None
    low = seg.lower()
    # keep structural / dictionary / already-token / hash segments verbatim
    if _IS_TOKEN.match(seg) or _IS_HASH.match(seg):
        return None, None
    if low in _KEEP_WORDS or low in _SUFFIX_STOP:
        return None, None

    # 1) Google-Doc-ID (most specific — self-identifying by shape)
    if _is_docid(seg):
        return "docid", mint.tag("DOC_ID", seg)

    # 2) first-last person slug (Ivan-Petrov, maria-smirnova547)
    m = _FIRSTLAST.match(seg)
    if m and m.group(1).lower() not in _KEEP_WORDS | _SUFFIX_STOP:
        second = m.group(2)
        # the second word must look like a surname (RU cyr or translit family)
        if _CYR_SURNAME.match(second.capitalize()) or _lat_surname(second, cue):
            return "name", mint.pname(_canon_name(seg))

    # 3) a bare Cyrillic surname (Иванова, Петров) — case-tail folded to nominative
    mc = _CYR_SURNAME.match(seg)
    if mc:
        # seed on the de-inflected nominative stem so all cases collapse
        return "name", mint.pname(_denflect_cyr(seg))

    # 4) a bare transliterated surname (Petrov, Smirnov, Sidorov)
    if _lat_surname(seg, cue):
        return "name", mint.pname(low)

    return None, None


# Cyrillic case-tail stripping to a nominative-ish key (иванова/иванову -> иванов).
_CYR_STRIP = re.compile(
    r"(ой|ою|ым|ом|ых|им|его|ого|ему|ому|а|у|е|ы)$")


def _denflect_cyr(seg: str) -> str:
    base = seg.lower()
    # keep productive family endings intact; strip only the trailing case vowel
    for end in ("ский", "цкий", "ской", "цкой", "енко", "ов", "ев", "ёв", "ин",
                "ын", "ук", "юк", "чук", "ян", "дзе", "off", "ман"):
        idx = base.find(end)
        if idx >= 0:
            head = base[:idx + len(end)]
            return head
    return _CYR_STRIP.sub("", base)


# --------------------------------------------------------------------------- #
# name-level transform (segment-aware, separators kept)
# --------------------------------------------------------------------------- #

# Split a name into segments + the separators BETWEEN them, preserving both so we
# can rebuild the name exactly. We split on the "hard" separators (__  _  .  space)
# but KEEP '-' inside a segment, because a hyphen joins a first-last person slug
# (``Ivan-Petrov``) that must be classified as ONE entity. A '-'-joined
# chunk that is NOT a person slug is split further and classified piece-by-piece.
_SEP_SPLIT = re.compile(r"(__+|[_. ]+)")
_HYPHEN_SPLIT = re.compile(r"(-+)")


def _mask_chunk(chunk: str, is_dir: bool, cue=frozenset()):
    """Classify one hard-separator-delimited chunk (may contain hyphens).

    Returns (new_chunk, changed). A ``first-last`` person slug is masked as one
    token; otherwise the chunk is split on '-' and each piece classified, so
    ``Ivan-Petrov`` masks the surname half while a lone ``report`` stays put.
    """
    # whole-chunk person slug: Ivan-Petrov / maria-smirnova547
    if "-" in chunk:
        m = _FIRSTLAST.match(chunk)
        if m and m.group(1).lower() not in _KEEP_WORDS | _SUFFIX_STOP:
            second = m.group(2)
            # the second word is checked like a bare stem (dictionary stop +
            # «-in» cue): ``auto-login`` / ``user-signin`` are not a person
            if _CYR_SURNAME.match(second.capitalize()) or _lat_surname(second, cue):
                return mint.pname(_canon_name(chunk)), True
        # not a clean first-last slug -> classify each hyphen piece independently
        pieces = _HYPHEN_SPLIT.split(chunk)
        out, changed = [], False
        for j, piece in enumerate(pieces):
            if j % 2 == 1:                     # a '-' separator, kept verbatim
                out.append(piece)
                continue
            _, token = _classify_segment(piece, is_dir, cue)
            if token is not None and token != piece:
                out.append(token)
                changed = True
            else:
                out.append(piece)
        return "".join(out), changed
    # no hyphen: a single segment
    _, token = _classify_segment(chunk, is_dir, cue)
    if token is not None and token != chunk:
        return token, True
    return chunk, False


def mask_name(name: str, is_dir: bool = False) -> str:
    """Mask the sensitive segments of one file/dir NAME, keeping the extension,
    any trailing hash, existing tokens and every separator intact.

    Returns the (possibly identical) new name. A name that is purely tokens /
    structural words is returned unchanged.
    """
    if not name or name in (".", ".."):
        return name

    # peel a real file extension off a FILE name (not a dir) so we never touch it
    ext = ""
    stem = name
    if not is_dir:
        root, dot_ext = os.path.splitext(name)
        # only treat a short alpha-num suffix as an extension (".pdf", ".jsonl"),
        # never a doc-id fragment or a version like ".ORG_32693"
        if dot_ext and re.fullmatch(r"\.[A-Za-z][A-Za-z0-9]{0,7}", dot_ext) \
                and not _IS_TOKEN.match(dot_ext[1:]):
            stem, ext = root, dot_ext

    # id remainders around an already placed DOC_ID token, then a Google id with
    # separators inside — whole, BEFORE per-segment classification
    stem0 = stem
    # someone else's client placeholder («Отчёт_Исполнитель_12_v2», «Папка Клиент_5») — the same
    # ORG token as in text (orgs.CLIENT_ROLE_WORDS); before splitting into segments: "_" between the
    # word and the number is part of the designation, not a separator
    stem = _orgs.sub_client_roles(stem, names=True)
    stem = _mask_docid_tails(stem)
    parts, folded = _fold_gid_runs(_split_keep_tokens(stem))
    # given-name cue for the ambiguous «-in/-yn» stems: every alpha piece of
    # the name (hyphen pieces too) that is a known given name
    cue = _given_names(p for part in parts[::2] for p in part.split("-"))
    changed = folded or stem != stem0
    out = []
    for i, part in enumerate(parts):
        # odd indices are hard separators (kept verbatim)
        if i % 2 == 1:
            out.append(part)
            continue
        new_chunk, ch = _mask_chunk(part, is_dir, cue)
        out.append(new_chunk)
        changed = changed or ch

    if not changed:
        return name
    # an id remainder next to a DOC_ID token minted per segment in THIS call
    # (an id of non-standard length: the window did not fold, the long piece went under a token)
    return _mask_docid_tails("".join(out)) + ext


# --------------------------------------------------------------------------- #
# rename planning  (walk once, plan; caller applies bottom-up)
# --------------------------------------------------------------------------- #
def plan_renames(root: str, keep_taken: bool = False):
    """Walk ``root`` and return an ordered ``[(old_abs, new_abs), …]`` rename plan.

    Files come first, then directories BOTTOM-UP (deepest first), so a caller can
    ``os.rename`` the list top-to-bottom without ever renaming a parent out from
    under a not-yet-renamed child. Never renames during the walk. Pure-inspection:
    this function does NO filesystem mutation.
    """
    root = os.path.abspath(root)
    file_plan = []
    dir_plan = []
    # Guard (tests/test_gaps_detectors.py): the raw
    # API did not resolve target collisions itself — «Акт Иванов.pdf» and «Акт Иванова.pdf» → one target,
    # apply_renames silently skipped the second one, the name with the surname stayed. Now a target taken
    # by the plan or on disk gets a _2/_3 suffix (walk.suffix_name, as in Pipeline). keep_taken
    # (Pipeline resolves it itself, truncating the name) — no suffix.
    from ..engine.walk import suffix_name
    suffix = not keep_taken
    taken: set = set()
    # topdown=False -> children before parents (bottom-up dirs)
    for dp, dns, fns in os.walk(root, topdown=False):
        for fn in fns:
            nf = mask_name(fn, is_dir=False)
            if nf != fn:
                src = os.path.join(dp, fn)
                dst = os.path.join(dp, nf)
                if suffix and (dst in taken or os.path.exists(dst)):
                    dst = suffix_name(dst, taken, False, lambda p: p)
                if keep_taken or suffix or not os.path.exists(dst):
                    taken.add(dst)
                    file_plan.append((src, dst))
        for dn in dns:
            nd = mask_name(dn, is_dir=True)
            if nd != dn:
                src = os.path.join(dp, dn)
                dst = os.path.join(dp, nd)
                if suffix and (dst in taken or os.path.exists(dst)):
                    dst = suffix_name(dst, taken, True, lambda p: p)
                if keep_taken or suffix or not os.path.exists(dst):
                    taken.add(dst)
                    dir_plan.append((src, dst))
    # files first, then dirs already deepest-first from topdown=False
    return file_plan + dir_plan


def apply_renames(plan, apply: bool = False, done_pairs=None):
    """Execute a rename ``plan`` (from :func:`plan_renames`). No-op unless
    ``apply`` is True. Returns the count of renames actually performed; missing
    sources / existing targets are skipped silently (idempotent re-runs).
    ``done_pairs`` (a list) — performed pairs are written here (the report
    covers only what was done)."""
    done = 0
    if not apply:
        return done
    for src, dst in plan:
        if not os.path.exists(src) or os.path.exists(dst):
            continue
        try:
            os.rename(src, dst)
            done += 1
            if done_pairs is not None:
                done_pairs.append((src, dst))
        except OSError:
            pass
    return done
