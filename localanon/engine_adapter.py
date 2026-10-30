"""Adapter of the anonymizer/ engine for local pre-cleaning.

What it does:
  * sets the ``[project].salt`` salt in ``anonymizer.engine.mint`` (mint version 2 — as in new
    main pipeline workspaces: 7-digit tokens; ``mint.set_tenant``);
  * builds the engine ``Pipeline`` in single-process mode (Windows: no fork), without the
    "Russian trace" dictionary (its data is not shipped) and with extra rules in four slots: link
    removal and contractor registration numbers go FIRST (before the detectors: otherwise the
    engine's ``domain`` turns ``docs.google.com`` into a RESOURCE token, and financial mints part of
    an INN with a space as DOC_ID), contractor names — before the organization detector (as the
    engine lifts multi-word pack brands: the legal entity "ООО «Ромашка Групп»" gets the contractor
    token before ``opf_name`` mints its own), roster pairs and initials — before the name detector
    (exact metadata literals take the whole person before ``bare_surname_dict``), the rest (roster
    handles and surnames, NER, other domains) — in the pack slot, after the detectors but before the
    gluing post-rules;
  * masks a tree (``run_tree``: text, JSON/JSONL through structured profiles, CSV through the table
    profile — all as in the main pipeline), one string (``mask_text``: document views, file
    names) and renames files and directories (``rename_tree``);
  * transcodes UTF-16/32 text with BOM to UTF-8 without BOM before the engine (``transcode_wide_bom``);
  * gives text views of the tree files for NER and the checks (``iter_views``): the text is read the
    way the engine masks it (``engine_view``: UTF-8 BOM, strict UTF-8, otherwise UTF-8 +
    surrogateescape — "sparsely broken" or unreadable; UTF-16/32 with BOM — not transcoded); an
    undecoded file and a file the document layer failed on are given with a special ``kind`` — the
    gate counts them as violations, anonymize removes a broken document.

The engine is not modified. From the private Pipeline API it uses ``_masked`` (rules by view with
substitution into the source) and ``_view_on`` — a proposal for the engine: expose them as
``Pipeline.mask_text``.
"""
from __future__ import annotations

import bisect
import codecs
import importlib.util
import os
import re
import zlib
from pathlib import Path

# without pyahocorasick the engine prints a WARN in every process; pure Python is a normal mode
# of the local run (ANON_AC_BACKEND=py silences the warning, the output is the same)
if importlib.util.find_spec("ahocorasick") is None:
    os.environ.setdefault("ANON_AC_BACKEND", "py")

from anonymizer.config.default import load as _load_engine_config  # noqa: E402
from anonymizer.engine import mint  # noqa: E402
from anonymizer.engine.pipeline import Pipeline  # noqa: E402
from anonymizer.engine.report import Report  # noqa: E402
from anonymizer.engine.walk import (SKIP_EXT, fit_name, has_text_bom, skip_reason,  # noqa: E402
                                    suffix_name)

MINT_VERSION = 2
# the engine's decode_bytes mark: decoding failed, bytes kept as surrogates — rules and the gate
# do not see such text (cp1251/UTF-16 without BOM etc.); the file view counts as unreadable
UNREADABLE_NOTE = "surrogateescape"
# UTF-8 with sparse broken spots (no more than SPARSE_MAX_BAD and SPARSE_MAX_SHARE of the bytes, one
# spot — in any file) is read if every spot is single — a byte or a truncated multi-byte
# character — surrounded by valid UTF-8 (_sparse_broken)
SPARSE_NOTE = "utf-8-sparse-broken"
SPARSE_MAX_BAD = 64
SPARSE_MAX_SHARE = 0.01
# UTF-16 and UTF-32 without BOM in a file with any extension — an unreadable view (utf16_without_bom,
# utf32_without_bom); a binary with a signature (offset, bytes) is not checked
BINARY_MAGIC = (
    (0, b"\x89PNG"), (0, b"PK\x03\x04"), (0, b"PK\x05\x06"), (0, b"PK\x07\x08"), (0, b"%PDF-"),
    (0, b"SQLite format 3\x00"), (0, b"\x1f\x8b"), (0, b"7z\xbc\xaf\x27\x1c"), (0, b"Rar!\x1a\x07"),
    (0, b"\xff\xd8\xff"), (0, b"GIF8"), (0, b"\x7fELF"), (0, b"MZ"),
    (0, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"), (0, b"RIFF"), (0, b"OggS"), (0, b"fLaC"),
    (0, b"ID3"), (0, b"wOFF"), (0, b"wOF2"), (0, b"OTTO"), (0, b"\x00\x01\x00\x00"),
    (0, b"\x00\x00\x01\x00"), (0, b"BZh"), (0, b"\xfd7zXZ\x00"), (0, b"\x28\xb5\x2f\xfd"),
    (0, b"\xca\xfe\xba\xbe"), (0, b"\xfe\xed\xfa"), (0, b"\xce\xfa\xed\xfe"), (0, b"\xcf\xfa\xed\xfe"),
    (0, b"\x00asm"), (0, b"\x93NUMPY"), (0, b"\x89HDF"), (0, b"PAR1"), (0, b"\x1a\x45\xdf\xa3"),
    (0, b"8BPS"), (4, b"ftyp"), (0, b"II*\x00"), (0, b"MM\x00*"), (0, b"BM"),
)
UTF16_ZERO_SHARE = 0.2
UTF32_ZERO_SHARE = 0.9
WIDE_MIN_TEXT = 0.8
# word structure of the decode (_texty): spaces and newlines ≥ WIDE_MIN_SPACE chars, mean word
# length ≤ WIDE_MAX_WORD, letters and digits ≥ WIDE_MIN_ALNUM; without spaces — all letters
# (≥ WIDE_DENSE_LETTERS); otherwise (minified JSON, CSV, a log with hashes) — letters and digits ≥
# WIDE_MIN_ALNUM and the head shorter than WIDE_SHORT chars or zlib-compressible to ≤ WIDE_MAX_BITS bits
# per char (text has repeated keys, separators, names; uint16 in a narrow band is noise ≈ 7 bits per char)
WIDE_MIN_SPACE = 0.05
WIDE_MAX_WORD = 20
WIDE_MIN_ALNUM = 0.4
WIDE_DENSE_LETTERS = 0.95
WIDE_SHORT = 1024
WIDE_MAX_BITS = 6.5
UTF16_HEAD_BYTES = 65536
# the engine reads UTF-16 and UTF-32 with BOM as UTF-8 + surrogateescape and does not mask them: anonymize
# transcodes such a file to UTF-8 without BOM before the engine (transcode_wide_bom), then the engine,
# the gate and NER read the UTF-8 copy; a BOM file that fails to decode or has NUL and control chars
# in the head (WIDE_HEAD_CHARS) is a binary (PCM with the first sample −2 = FE FF), not text
WIDE_BOMS = (("utf-32-le", b"\xff\xfe\x00\x00"), ("utf-32-be", b"\x00\x00\xfe\xff"),
             ("utf-16-le", b"\xff\xfe"), ("utf-16-be", b"\xfe\xff"))
WIDE_HEAD_CHARS = 32768
WIDE_BOM_NOTE = "wide-bom"
KIND_UNREADABLE_ENCODING = "unreadable_encoding"
KIND_UNREADABLE_DOCUMENT = "unreadable_document"


class DocumentError(Exception):
    """The document layer failed on a file (broken docx/pdf): kind and exception name —
    no path and no content."""

    def __init__(self, kind: str, exc: BaseException):
        super().__init__(f"{kind}: {type(exc).__name__}")
        self.kind = kind
        self.exc_name = type(exc).__name__

OFFICE_EXT = (".docx", ".xlsx", ".pptx")
PDF_EXT = (".pdf",)
# file extension kept on rename (like filenames.mask_name)
_EXT_RX = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,7}$")
_TOKEN_EXT = re.compile(r"^[A-Z][A-Z0-9_]*_\d+$")


def set_salt(salt: str) -> None:
    """Data owner's salt → engine minting. The salt value is never printed."""
    mint.set_tenant(salt, MINT_VERSION)


def engine_config(workers: int = 1) -> dict:
    """Engine config for the local run: ``workers`` processes (1 — sequential; more — the engine's
    fork pool, the output does not depend on the number; without fork the engine is sequential),
    profiles on, the Russian trace dictionary off (its data is not shipped), known_values packs empty."""
    cfg = _load_engine_config()
    cfg["workers"] = max(1, int(workers))
    cfg["known_values"] = []
    cfg["known_files"] = []
    cfg["ru_trace"] = {"enabled": False, "currency": False, "laws": False}
    cfg["structured_profiles"] = True
    cfg["rename_files"] = False
    return cfg


_MAP = None              # (engine, fn) of Engine.map, inherited by its forked workers
_FORK = None             # (fn, state) of fork_map, inherited by its forked workers


def _map_task(task):
    eng, fn = _MAP
    return fn(eng, task)


def _fork_task(task):
    fn, state = _FORK
    return fn(state, task)


_NAME_BATCH = 200          # file/directory names per pool task of rename_tree


def _mask_names(eng, batch: list) -> list:
    """Masked names of [(name, is_dir)] (in a worker of ``fork_map`` or here)."""
    from anonymizer.detectors import filenames as fn
    return [eng.mask_name(name, is_dir, fn) for name, is_dir in batch]


def fork_map(fn, tasks: list, workers: int, *, state=None, weight=None, label: str = "pool") -> list:
    """``[fn(state, task) for task in tasks]`` in up to ``workers`` forked processes: ``state`` (automata,
    rules) is inherited copy-on-write, only the tasks and the results are pickled. With ``weight`` the heavy
    tasks go out first (the one started last is the tail). Without fork, with one worker or one task — here.
    Results in task order; a task whose worker died twice (the engine pool retries it once alone) — here."""
    global _FORK
    from anonymizer.engine import pipeline as _pl
    ctx = _pl.fork_context() if workers > 1 and len(tasks) > 1 else None
    if ctx is None:
        return [fn(state, t) for t in tasks]
    from anonymizer.engine.pool import run_pool
    order = list(range(len(tasks)))
    if weight is not None:
        order.sort(key=lambda i: -weight(tasks[i]))
    _FORK = (fn, state)
    try:
        done, _dead = run_pool(_fork_task, [tasks[i] for i in order], min(workers, len(tasks)), ctx,
                               label=label, name=lambda _t: "a task")
    finally:
        _FORK = None
    out: list = [None] * len(tasks)
    for k, i in enumerate(order):
        out[i] = done[k] if k in done else fn(state, tasks[i])
    return out


class Engine:
    """Engine with the salt and extra rules.

    ``first`` — rules before all detectors (link removal, contractor registration numbers),
    ``before_orgs`` — before the organization detector (contractor), ``before_names`` — before the
    name detector (roster pairs and initials: exact metadata literals must take the whole person
    before bare_surname_dict/dict_first_name split the pair into two tokens),
    ``pack`` — in the pack slot (roster handles and surnames, NER, other domains).
    """

    def __init__(self, salt: str, *, first=(), before_orgs=(), before_names=(), pack=(), workers: int = 1):
        set_salt(salt)
        self.pipe = Pipeline(engine_config(workers))
        self._place_rules(list(first), list(before_orgs), list(before_names), list(pack))
        from . import rxdefer
        rxdefer.resolve(self.pipe.rules, workers)     # deferred rule regexes: a batch over the workers
        self.report = Report(want_why=False)

    def _place_rules(self, first: list, before_orgs: list, before_names: list,
                     pack: list) -> None:
        from anonymizer.detectors import glue as _glue
        from anonymizer.detectors import names as _names
        from anonymizer.detectors import orgs as _orgs
        rules = list(self.pipe.rules)
        n = len(rules)

        def first_of(mod):
            ids = {id(r) for r in mod.RULES}
            return next((i for i, r in enumerate(rules) if id(r) in ids), n)

        at_orgs, at_names, at_glue = first_of(_orgs), first_of(_names), first_of(_glue)
        if at_glue < at_names:
            at_glue = n
        if at_names < at_orgs:
            at_names = at_orgs
        # inserts from the end — indexes earlier in the list stay valid
        rules = rules[:at_glue] + pack + rules[at_glue:]
        rules = rules[:at_names] + before_names + rules[at_names:]
        rules = rules[:at_orgs] + before_orgs + rules[at_orgs:]
        self.pipe.rules = first + rules

    # ---- text --------------------------------------------------------------
    def mask_text(self, text: str, rel: str = "text.txt") -> str:
        """One string/text through the same rule chain as the file ``rel`` (view by extension)."""
        if not text:
            return text
        return self.pipe._masked(text, self.report, rel, self.pipe._view_on())[0]

    def map(self, fn, tasks: list, workers: int = 1, label: str = "anonymize") -> list:
        """``[fn(self, task) for task in tasks]`` in up to ``workers`` forked processes: the engine
        with its built rules is inherited copy-on-write, as in the engine's own pool (``run_tree``).
        Without fork (Windows), with one worker or one task — in this process. Results in task
        order; a task whose worker died twice (the engine pool retries it once alone) — None."""
        global _MAP
        from anonymizer.engine import pipeline as _pl
        ctx = _pl.fork_context() if workers > 1 and len(tasks) > 1 else None
        if ctx is None:
            return [fn(self, t) for t in tasks]
        from anonymizer.engine.pool import run_pool
        _pl.prebuild_rules(self.pipe.rules, min(workers, len(tasks)))
        _MAP = (self, fn)
        try:
            with _pl.frozen_heap():
                done, _dead = run_pool(_map_task, list(tasks), min(workers, len(tasks)), ctx,
                                       label=label, name=lambda _t: "a document")
        finally:
            _MAP = None
        return [done.get(i) for i in range(len(tasks))]

    def chain(self):
        """``chain(text) -> text`` for minting canonical names (like Pipeline._struct_chain)."""
        return lambda text: self.pipe._apply_fixpoint(text, Report(want_why=False), None)

    # ---- tree --------------------------------------------------------------
    def run_tree(self, root: Path) -> Report:
        """Mask a tree in place (text, JSON/JSONL profiles, CSV). Returns the engine report
        (counters by rule; examples are not given out)."""
        return self.pipe.run(str(root), apply=True)

    def rename_tree(self, root: Path, moves: list | None = None, workers: int = 1,
                    keep: frozenset = frozenset()) -> list:
        """File and directory names as text: the name stem goes through the rule chain (the extension
        stays), then the engine's file name detector; collisions get a _2/_3 suffix, long names are
        truncated (walk.fit_name). Returns [(old name, new name)] without paths; ``moves`` receives
        [(old path, new path)] in rename order (for ctx.mark marks: RunContext.marks_moved).
        File names in ``keep`` (lower case) stay as they are: an export is found by its marker file
        (Telegram ``result.json``), a common word of the name can be an organization of the model.

        The names are masked first, in up to ``workers`` forked processes (a masked name depends on the
        name alone); the renames go after, here, in the walk order — the result is that of one pass.
        The walk is listed up front: bottom-up, a directory is listed before anything in it is renamed."""
        root = Path(root)
        walk = [(Path(dp), sorted(fns), sorted(dns)) for dp, dns, fns in os.walk(root, topdown=False)]
        names = sorted({(n, False) for _dp, fns, _dns in walk for n in fns if n.lower() not in keep}
                       | {(n, True) for _dp, _fns, dns in walk for n in dns})
        batches = [names[i:i + _NAME_BATCH] for i in range(0, len(names), _NAME_BATCH)]
        if workers > 1 and len(batches) > 1:
            from anonymizer.engine import pipeline as _pl
            _pl.prebuild_rules(self.pipe.rules, workers)   # automata in the parent: the workers inherit them
        masked = {}
        for batch, out in zip(batches, fork_map(_mask_names, batches, workers, state=self,
                                                label="rename")):
            masked.update(zip(batch, out))
        done = []
        for dp, fns, dns in walk:
            taken: set = set()
            for name in fns:
                self._rename_one(dp, name, masked.get((name, False), name), False, taken, done, moves)
            for name in dns:
                self._rename_one(dp, name, masked[name, True], True, taken, done, moves)
        return done

    def mask_name(self, name: str, is_dir: bool, fn=None) -> str:
        if fn is None:
            from anonymizer.detectors import filenames as fn
        stem, ext = name, ""
        if not is_dir:
            m = _EXT_RX.search(name)
            if m and m.start() > 0 and not _TOKEN_EXT.match(name[m.start() + 1:]):
                stem, ext = name[:m.start()], name[m.start():]
        from .roster import path_context
        with path_context():           # file/directory name: context rules know it is a path
            new = self.mask_text(stem, rel=name) if stem else stem
        new = new.replace("/", "-").replace("\\", "-").replace("\n", " ").strip() or stem
        new = fn.mask_name(new + ext, is_dir=is_dir)
        if is_dir and new != name:
            new = new.rstrip(". ") or name
        return new

    def _rename_one(self, dp: Path, name: str, new: str, is_dir: bool, taken: set, done: list,
                    moves: list | None = None) -> None:
        if new == name:
            return
        new = fit_name(new, is_dir=is_dir)
        dst = dp / new
        if str(dst) in taken or dst.exists():
            dst = Path(suffix_name(str(dp / new), taken, is_dir, lambda p: p))
        taken.add(str(dst))
        try:
            os.rename(dp / name, dst)
        except OSError:
            return
        done.append((name, dst.name))
        if moves is not None:
            moves.append((dp / name, dst))

    # ---- views -------------------------------------------------------------
    def rule_counts(self, report: Report) -> dict:
        return {k: int(v) for k, v in report.counts.items() if v}


def file_kind(path: Path) -> str:
    """text | office | pdf | binary — by extension and sniffing, as the engine does."""
    ext = path.suffix.lower()
    if ext in OFFICE_EXT:
        return "office"
    if ext in PDF_EXT:
        return "pdf"
    if ext in SKIP_EXT:
        return "binary"
    try:
        with open(path, "rb") as fh:
            head = fh.read(8192)
    except OSError:
        return "binary"
    from anonymizer.engine.walk import is_binary_bytes
    return "binary" if is_binary_bytes(head) else "text"


def utf16_without_bom(path: Path) -> bool:
    """Any extension (or none), no binary signature (BINARY_MAGIC) and no BOM, zeros
    > UTF16_ZERO_SHARE at even or odd positions of the head, at the other parity — no more than
    a twentieth of them, and the head decodes to text in this byte order (_texty: at least
    WIDE_MIN_TEXT text chars) — looks like UTF-16 without BOM: the engine treats such a file as
    binary and does not mask it, the gate — an unreadable_encoding violation. The high byte of
    Cyrillic (U+04xx) is 0x04 and counts together with zero: in Russian text zeros come only from
    spaces. 16-bit data (quiet PCM, sensor readings, depth, 16-bit TIFF) gives as many zeros in the
    high bytes but does not decode to text."""
    head = _plain_head(path)
    if head is None:
        return False
    even, odd = (half.count(0) + half.count(4) for half in (head[0::2], head[1::2]))
    major, minor = max(even, odd), min(even, odd)
    if major <= UTF16_ZERO_SHARE * (len(head) // 2) or minor * 20 > major:
        return False
    return _texty(head[: len(head) & ~1].decode("utf-16-le" if odd >= even else "utf-16-be",
                                                 "replace"))


def utf32_without_bom(path: Path) -> bool:
    """UTF-32 without BOM: in every four bytes of the head the two high ones (LE — positions 2 and 3,
    BE — 0 and 1) are zero in at least UTF32_ZERO_SHARE of the quads, and the head decodes to text
    in this byte order (_texty). UTF-32 zeros sit at both parities — the UTF-16 sign does not see
    it; the engine treats the file as binary, the gate — an unreadable_encoding violation."""
    head = _plain_head(path)
    if head is None or len(head) < 4:
        return False
    head = head[: len(head) & ~3]
    quads = len(head) // 4
    for codec, hi in (("utf-32-le", (2, 3)), ("utf-32-be", (0, 1))):
        zero = sum(1 for k in range(0, len(head), 4) if head[k + hi[0]] == 0 == head[k + hi[1]])
        if zero >= UTF32_ZERO_SHARE * quads:
            return _texty(head.decode(codec, "replace"))
    return False


def _plain_head(path: Path):
    """File head without BOM and binary signature (None — do not check)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(UTF16_HEAD_BYTES)
    except OSError:
        return None
    if not head or has_text_bom(head[:4]) or _binary_magic(head):
        return None
    return head


# a text char: printable ASCII and whitespace, Cyrillic, no-break space, «», °, general punctuation,
# currencies, "№" (Latin with diacritics is a small share in text); quiet PCM (U+0000–U+012C and
# U+FED4–U+FFFF), uint16 ≤ 2048 (U+0000–U+07FF: control, Latin-1, IPA, Greek, Armenian,
# Hebrew…) and 16-bit TIFF give less than half text chars
_TEXT_CHAR_RX = re.compile(r"[\t\n\r\x20-\x7e\u00a0\u00ab\u00b0\u00bb\u0400-\u052f"
                           r"\u2000-\u206f\u20a0-\u20cf\u2116]")


_SPACE_RX = re.compile(r"[ \t\n\r ]+")
_LETTER_RX = re.compile(r"[A-Za-zА-Яа-яЁё]")
_ALNUM_RX = re.compile(r"[A-Za-zА-Яа-яЁё0-9]")


def _texty(text: str) -> bool:
    """The head decode is text: text chars ≥ WIDE_MIN_TEXT and a word structure — spaces and
    newlines ≥ WIDE_MIN_SPACE, mean word length ≤ WIDE_MAX_WORD, letters and digits ≥
    WIDE_MIN_ALNUM, or (text without spaces) Latin and Cyrillic letters ≥ WIDE_DENSE_LETTERS,
    or (minified JSON, CSV, a log with hashes) letters and digits ≥ WIDE_MIN_ALNUM and the head
    shorter than WIDE_SHORT or zlib-compressible to ≤ WIDE_MAX_BITS bits per char (head ≤ 32 768 chars).
    uint16 from the ASCII (≈ 50–125) and Cyrillic (U+0400–U+052F) bands gives many text chars, but
    ~1 % spaces and ≤ 60 % letters, and does not compress (≈ 7 bits per char); the 32–47 band is
    spaces and punctuation without letters."""
    if not text or len(_TEXT_CHAR_RX.findall(text)) < WIDE_MIN_TEXT * len(text):
        return False
    space = sum(m.end() - m.start() for m in _SPACE_RX.finditer(text))
    words = len(_SPACE_RX.split(text.strip(" \t\n\r ")))
    if (space >= WIDE_MIN_SPACE * len(text) and len(text) - space <= WIDE_MAX_WORD * words
            and len(_ALNUM_RX.findall(text)) >= WIDE_MIN_ALNUM * len(text)):
        return True
    if len(_LETTER_RX.findall(text)) >= WIDE_DENSE_LETTERS * len(text):
        return True
    if len(_ALNUM_RX.findall(text)) < WIDE_MIN_ALNUM * len(text):
        return False
    if len(text) < WIDE_SHORT:
        return True
    return 8 * len(zlib.compress(text.encode("utf-8"), 9)) / len(text) <= WIDE_MAX_BITS


def _binary_magic(head: bytes) -> bool:
    return any(head[off:off + len(magic)] == magic for off, magic in BINARY_MAGIC)


_CTRL_CHAR_RX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def wide_bom(head: bytes) -> str:
    """UTF-16/UTF-32 codec by the BOM at the data start ('' — no such BOM)."""
    return next((codec for codec, bom in WIDE_BOMS if head.startswith(bom)), "")


def wide_bom_text(data: bytes):
    """Text of data with a UTF-16/32 BOM (without the BOM itself) or None — not text: no BOM, decoding
    failed or the head has NUL and control chars other than \\t \\r \\n (16-bit data whose first
    bytes happen to match a BOM)."""
    codec = wide_bom(data[:4])
    if not codec:
        return None
    bom = next(b for c, b in WIDE_BOMS if c == codec)
    try:
        text = data[len(bom):].decode(codec)
    except UnicodeDecodeError:
        return None
    if _CTRL_CHAR_RX.search(text, 0, WIDE_HEAD_CHARS):
        return None
    return text


def transcode_wide_bom(path: Path) -> str:
    """A UTF-16/32 text file with BOM → UTF-8 without BOM in place (newlines as they were): the
    engine reads only UTF-8, the main pipeline too. Returns the source encoding ('' — the file is not
    transcoded: not text, no BOM, walk.skip_reason)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return ""
    codec = wide_bom(head)
    if not codec or skip_reason(str(path)):
        return ""
    try:
        text = wide_bom_text(Path(path).read_bytes())
        if text is None:
            return ""
        Path(path).write_bytes(text.encode("utf-8"))
    except OSError:
        return ""
    return codec


def document_lines(path: Path, kind: str):
    """View lines of an Office/PDF document through the document layer; None — the layer is unavailable
    (NotImplementedError). Any other layer exception (broken archive, OSError) → DocumentError:
    the file is not read, the caller decides (anonymize removes it, the gate — a violation)."""
    try:
        if kind == "office":
            from . import office
            return list(office.extract_view(path))
        if kind == "pdf":
            from . import pdf
            return list(pdf.extract_view(path))
    except NotImplementedError:
        return None
    except Exception as e:          # noqa: BLE001 — the document layer must not fail the stage
        raise DocumentError(kind, e) from e
    return None


def document_write(path: Path, kind: str, lines: list) -> None:
    """Write a document view; any layer exception (OSError, broken build) → DocumentError."""
    try:
        if kind == "office":
            from . import office
            office.apply_view(path, lines)
        elif kind == "pdf":
            from . import pdf
            pdf.rebuild(path, lines)
    except Exception as e:          # noqa: BLE001
        raise DocumentError(kind, e) from e


def read_view(path: Path):
    """(text, note) — the file text exactly as the engine masks it (engine_view);
    text None — do not read the file (walk.skip_reason: extension, size, binary). Both the gate and
    NER read this view: there is no second decoding ladder (charset_normalizer, cp1251 heuristic)
    — transcoding the input to UTF-8 is done by the inventory stage, and the engine gets its
    result."""
    reason = skip_reason(str(path))
    if reason:
        return None, reason
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return None, f"unreadable:{exc.strerror or exc}"
    return engine_view(data)


def engine_view(data: bytes):
    """(text, note) of bytes as the engine sees them: BOM — decode by BOM (note "bom"), strict UTF-8 —
    "strict"; otherwise — like the engine (walk.read_text): UTF-8 + surrogateescape. Such text is
    "sparsely broken" (SPARSE_NOTE: the engine masks it as text and writes the bytes back as they
    were) if the broken spots are rare and single (_sparse_broken), otherwise — UNREADABLE_NOTE (a
    gate violation). UTF-16/32 BOM: text — WIDE_BOM_NOTE (the file is not transcoded, the engine did
    not mask it — a gate violation), not text (wide_bom_text) — (None, "binary-content")."""
    if wide_bom(data[:4]):
        text = wide_bom_text(data)
        return (None, "binary-content") if text is None else (text, WIDE_BOM_NOTE)
    bom = has_text_bom(data[:4])
    if bom:
        try:
            return data.decode(bom), "bom"
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8"), "strict"
    except UnicodeDecodeError:
        pass
    text = data.decode("utf-8", "surrogateescape")
    return text, SPARSE_NOTE if _sparse_broken(text) else UNREADABLE_NOTE


def _broken_spans(data: bytes) -> list:
    """Broken UTF-8 spots — [(start, end)] in bytes, as the decoder marks them: a truncated
    multi-byte sequence (a lead byte and an incomplete continuation tail) is one spot, any other
    byte that does not form UTF-8 is a one-byte spot."""
    spans = []

    def rec(e):
        spans.append((e.start, e.end))
        return "", e.end

    codecs.register_error("localanon.broken_spans", rec)
    data.decode("utf-8", "localanon.broken_spans")
    return spans


def _sparse_broken(text: str) -> bool:
    # a broken spot is one character: a single byte or a truncated multi-byte character (a log line cut
    # by a byte limit in the middle of "—", an emoji, CJK); one spot is allowed even in a short file:
    # in an 8-bit encoding it is one letter — not a name
    data = text.encode("utf-8", "surrogateescape")
    spans = _broken_spans(data)
    if len(spans) > SPARSE_MAX_BAD or len(spans) > max(1, SPARSE_MAX_SHARE * len(data)):
        return False
    # the share does not look at what the broken spots are — a name in cp1251/KOI8-R/cp866 inside a
    # UTF-8 log is 7–10 bytes in a row, each a separate spot. A word (a run of non-ASCII bytes) with a
    # broken spot is allowed only if it has one spot and the other characters are Latin, Greek,
    # Cyrillic, punctuation (_SPARSE_NEIGHBOUR_RX; cp866 "Юрий" = 9E + E0A8A9 — one spot and U+0A29
    # Gurmukhi). Symbols, box drawing, CJK, emoji (_SPARSE_EXOTIC_RX) may neighbour only a truncated
    # character or a byte ≥ C0 ("✅\xff", "中\xe6\x96"): a single continuation byte 80–BF is a cp866
    # letter ("Ахад" = 80 · U+5824, "Атом" = 80 · U+23AC). A truncated character (lead byte C2–F4) is
    # the end of a line cut by a byte limit, so it sits at the word end; in the middle of a word these
    # are cp866 letters ("рахим" = E0A0 · U+5869 …). A word with an exotic neighbour that reads in
    # cp866 as a Cyrillic word in one case — "Хвощ", "хвощ", "ХВОЩ" (no Ё: emoji F0 9F… in cp866 are
    # "ЁЯ…") — is a cp866 word ("хвощ" = U+58AE · E9). CJK, kana, box drawing, symbols in cp866 are a
    # mix of cases and pseudographics: lead byte E2–E9 — lowercase т–щ, continuations 80–BF —
    # uppercase, lowercase, box drawing mixed. The same for a word that is entirely a truncated
    # character of two or three bytes: "ян" = EF AD, "Ёж" = F0 A6, "те" = E2 A5; truncated "—", "№",
    # emoji — "тА", "тД", "ЁЯ". A real log cut is justified by a character with the same start
    # elsewhere in the text (a truncated CJK — a neighbouring ideograph)
    starts = [a for a, _b in spans]
    seen = None
    for m in _NON_ASCII_BYTES_RX.finditer(data):
        lo = bisect.bisect_left(starts, m.start())
        hi = bisect.bisect_left(starts, m.end())
        if hi - lo == 0:
            continue
        if hi - lo > 1:
            return False
        a, b = spans[lo]
        rest = data[m.start():a].decode("utf-8") + data[b:m.end()].decode("utf-8")
        if 0xC2 <= data[a] <= 0xF4 and b != m.end():
            return False
        exotic = data[a] >= 0xC0
        if not all(_SPARSE_NEIGHBOUR_RX.match(ch) or (exotic and _SPARSE_EXOTIC_RX.match(ch))
                   for ch in rest):
            return False
        cut = 0xC2 <= data[a] and b - a >= 2
        odd = not all(_SPARSE_NEIGHBOUR_RX.match(ch) for ch in rest)
        if (odd or (cut and not rest)) and _CYR_WORD_RX.fullmatch(m.group().decode("cp866")):
            if seen is None:
                seen = {ch.encode("utf-8")[:k] for ch in set(text) if "\x80" <= ch and
                        not "\udc80" <= ch <= "\udcff" for k in (2, 3)}
            if not (cut and data[a:b] in seen):
                return False
    return True


_NON_ASCII_BYTES_RX = re.compile(rb"[\x80-\xff]+")
_CYR_WORD_RX = re.compile(r"[А-ЯЁ]?[а-яё]+|[А-Я]+")


_SPARSE_NEIGHBOUR_RX = re.compile(r"[\u0080-\u052f\u1e00-\u1eff\u2000-\u206f\u20a0-\u20cf"
                                  r"\u2100-\u214f]")
# other scripts are not neighbours: cp866 bytes form them (80 · E0A0A1 "Араб" — U+0821)
_SPARSE_EXOTIC_RX = re.compile(
    r"[\u2150-\u23ff\u2460-\u27bf\u2b00-\u2bff\u3000-\u30ff\u4e00-\u9fff\uac00-\ud7af"
    r"\ufe00-\ufe0f\uff00-\uffef\U0001f000-\U0001faff]")


def structured_text(path: Path, rel: str, text: str, *, ner: bool):
    """Projection of the free text of a structured profile file (Telegram) or None."""
    from anonymizer import structured
    try:
        prof = structured.detect(str(path), rel)
        if prof is None:
            return None
        if ner:
            return structured.projection(text.lstrip("﻿"), prof, ner=True)
        return structured.gate_text(str(path), rel)
    except Exception:          # noqa: BLE001 — the profile is not parsed: as plain text
        return None


def text_view(path: Path, rel: str, *, ner: bool = False):
    """File text as the rules see it: JSON/HTML — the transport view (\\uXXXX decoded),
    a Telegram export — a projection of the free text fields. None — not readable or not
    decoded (see read_view)."""
    text, note = read_view(path)
    if text is None or note in (UNREADABLE_NOTE, WIDE_BOM_NOTE):
        return None
    return _view_of(path, rel, text, ner=ner)


def _view_of(path: Path, rel: str, text: str, *, ner: bool) -> str:
    st = structured_text(path, rel, text, ner=ner)
    if st is not None:
        return st
    from anonymizer.engine.vnorm import transport_view
    return transport_view(rel, text)[0]


def prefetch_views(ctx, files) -> int:
    """Document views (office/pdf) of ``files`` parsed ahead in worker processes — one per CPU slot
    of the hw plan (stage ``documents``), the heavy ones first — into the run's view cache
    (``viewcache``): a stage that then reads them one by one (``iter_views``, the masking) takes them
    from there. Without the cache, with fewer than two documents or one slot — nothing: the stage
    reads them itself. → the number of documents given to the workers."""
    from . import viewcache
    if viewcache.directory() is None:
        return 0
    docs = [(str(p), kind) for p in files for kind in (file_kind(Path(p)),) if kind in ("office", "pdf")]
    if len(docs) < 2:
        return 0
    try:
        from . import hw
        slots = hw.stage_plan(ctx, "documents", hw.Workload(n_documents=len(docs))).slots
    except Exception as e:  # noqa: BLE001 — the plan does not crash the stage: it reads them itself
        ctx.note(f"documents: hardware plan not built ({type(e).__name__}) — views read in one process")
        return 0
    if len(slots) < 2:
        return 0
    from . import devpool
    devpool.run(_prefetch_batch, docs, slots, batch_size=1, weight=_doc_size, ctx=ctx)
    return len(docs)


def _doc_size(item) -> int:
    try:
        return os.path.getsize(item[0])
    except OSError:
        return 0


def _prefetch_batch(_state, batch: list) -> list:
    for path, kind in batch:
        try:
            document_lines(Path(path), kind)
        except Exception:  # noqa: BLE001 — the stage reads it again and handles the error itself
            pass
    return [None] * len(batch)


def iter_views(root: Path, files, *, ner: bool = False, notes=None):
    """(path, rel, kind, text) over the tree files: text, document views (office/pdf — through the
    document layer; no layer — a note in ``notes`` once). Binaries are skipped. A file that is not
    decoded or stayed UTF-16/32 with BOM (not transcoded: the engine did not mask it) —
    kind KIND_UNREADABLE_ENCODING — or one the document layer failed on
    (KIND_UNREADABLE_DOCUMENT) is given with empty text — the caller decides (gate: a violation)."""
    root = Path(root)
    missing: set = set()
    for p in files:
        p = Path(p)
        rel = p.relative_to(root).as_posix()
        kind = file_kind(p)
        if utf16_without_bom(p) or utf32_without_bom(p):
            yield p, rel, KIND_UNREADABLE_ENCODING, ""
            continue
        if kind == "binary":
            continue
        if kind in ("office", "pdf"):
            try:
                lines = document_lines(p, kind)
            except DocumentError:
                yield p, rel, KIND_UNREADABLE_DOCUMENT, ""
                continue
            if lines is None:
                if notes is not None and kind not in missing:
                    missing.add(kind)
                    notes(f"{kind} layer is unavailable: document views cannot be read")
                continue
            yield p, rel, kind, "\n".join(lines)
            continue
        text, note = read_view(p)
        if text is None:
            continue
        if note in (UNREADABLE_NOTE, WIDE_BOM_NOTE):
            yield p, rel, KIND_UNREADABLE_ENCODING, ""
            continue
        yield p, rel, "text", _view_of(p, rel, text, ner=ner)
