"""File walking + safe I/O.

``errors='surrogateescape'`` is mandatory: the corpus contains byte sequences
that are not valid UTF-8, and without it a chunk of files can't be read/written
round-trip. Writes are atomic (tmp + replace) and only happen when content
actually changed and we're not in DRY mode.
"""
from __future__ import annotations

import codecs
import errno
import hashlib
import os
import re
import sys
import tempfile

SKIP_EXT = {
    ".zip", ".db", ".gz", ".tgz", ".zst", ".bin", ".pkl",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg",
    ".pdf", ".mp3", ".mp4", ".mov", ".wav", ".m4a",
    # other audio/video (the transcribe stage transcribes them; the media file
    # itself is binary: it is not read as text and is not handed to the client)
    ".ogg", ".oga", ".opus", ".flac", ".aac", ".wma", ".amr",
    ".mkv", ".avi", ".webm", ".m4v", ".3gp",
    ".woff", ".woff2", ".ttf", ".otf",
    ".xlsx", ".docx", ".pptx",
    # robustness: the client throws in anything — archives, media, office files,
    # binaries, DB/ML dumps are not read as text; .anontmp is an unfinished
    # tmp of a killed writer (_mk_tmp promised skipping by extension)
    ".7z", ".rar", ".tar", ".bz2", ".xz", ".lz4", ".lzma", ".jar", ".war",
    ".apk", ".dmg", ".iso",
    ".bmp", ".tif", ".tiff", ".heic", ".heif", ".psd", ".ai",
    ".doc", ".xls", ".ppt", ".odt", ".ods", ".odp", ".msg", ".pst", ".ost",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".class", ".pyc", ".pyo",
    ".wasm", ".lnk",
    ".wmv", ".flv", ".eot", ".ttc", ".cur",
    ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".parquet", ".npy", ".npz",
    ".h5", ".hdf5", ".pt", ".pth", ".onnx", ".safetensors", ".pickle",
    ".joblib",
    # agreed with release: .rtf is "text", but Cyrillic there is
    # \'e8 escapes: the round "scans", masks do not fire → must not be
    # handed out. Deliberately NOT here: .ts (TypeScript), .key (PEM), .raw, .img
    ".tbz", ".tbz2", ".txz", ".lz", ".cab", ".msi", ".lib", ".pyd", ".swf",
    ".rtf", ".pages", ".numbers", ".one", ".db3", ".mdb", ".accdb", ".avro",
    ".orc", ".feather", ".ckpt", ".avif", ".eps", ".cr2", ".nef",
    ".aiff", ".aif", ".mpg", ".mpeg",
    ".anontmp",
}   # no repeats in the literal (tests/test_eng1012_simplify.py)
TEXT_EXT = {".txt", ".md", ".csv", ".tsv", ".jsonl", ".json", ".html", ".htm", ".log", ".yaml", ".yml", ".xml"}

# PDF (owner's decision 14.10.2026: documents are anonymized in place, the file stays .pdf).
# The PDF layer (tools/pdf_layer.py + worker tools/pdf_worker.py in its own venv, $ANON_PDF_PY): intake
# puts a text "view" of the document into work/<name>.pdf (all text places, one unit — one
# line), rounds, gates and the final cleanup work with it as text, release builds a new PDF from the
# anonymized view. So .pdf leaves SKIP_EXT when the layer is active: when
# ANON_PDF_PY is set (no venv interpreter — PDFs are skipped as before: .pdf is binary, it does not go
# into the archive; intake prints one line and writes state/limitations.md). The layer rollback switch
# is removed. Read at import (like the other engine switches); pdf_refresh() recomputes after
# an environment change (tests).
PDF_PY_ENV = "ANON_PDF_PY"
PDF_MAGIC = b"%PDF-"


def pdf_binary_head(ext: str, head: bytes) -> bool:
    """A .pdf file with a %PDF- head while the layer is active is binary:
    in work/ it is a binary copy of a NOT parsed document (the layer puts the view only on success;
    a view starting with %PDF- is treated by the layer as a document failure — fail-closed), in raw/ —
    the original. The NUL sniff does not catch it (an ASCII head: uncompressed objects, XMP at the start), and the round,
    mine, residual, verify would read PDF syntax as text. Layer not active (.pdf in
    SKIP_EXT) — False: the old path byte for byte."""
    return ext == ".pdf" and ".pdf" not in SKIP_EXT and head.startswith(PDF_MAGIC)


def pdf_inplace_active() -> bool:
    """The PDF layer is active: the venv interpreter ANON_PDF_PY is set (the path to the executable is not
    checked here — skipping with a reason is decided by tools/pdf_layer)."""
    return bool(os.environ.get(PDF_PY_ENV, "").strip())


def pdf_refresh() -> bool:
    """Bring SKIP_EXT in line with the current environment (the set is edited in place: importers of
    `from walk import SKIP_EXT` see the same). Returns whether the layer is active."""
    active = pdf_inplace_active()
    if active:
        SKIP_EXT.discard(".pdf")
    else:
        SKIP_EXT.add(".pdf")
    return active


pdf_refresh()

# --- a single policy for reading client text ----------------------------------
#
# One client file must not bring a stage down: a size cap (ANON_MAX_FILE_BYTES,
# os.stat BEFORE reading), a binary sniff (NUL / share of control bytes in the first 8 KB),
# and a decoding ladder: utf-8-sig strict (the BOM is stripped; a UTF-16/32 BOM —
# the matching codec) → charset_normalizer (if the module exists; only
# encodings from _CN_ALLOWED, otherwise the step is skipped) → cp1251
# heuristic (share of Cyrillic among letters) → utf-8/surrogateescape as the last
# (bytes are not lost, but the text is marked note='surrogateescape').
MAX_FILE_BYTES_ENV = "ANON_MAX_FILE_BYTES"
MAX_FILE_BYTES_DEFAULT = 50 * 1024 * 1024
SNIFF_BYTES = 8192
_CTRL_RATIO = 0.30          # share of control bytes in the head → binary
_CP1251_MIN_CYR = 0.30      # cp1251 heuristic: share of Cyrillic among letters
_CN_MAX_CHAOS = 0.10        # charset_normalizer: "garbage" threshold of a match
_CTRL_BYTES = bytes(b for b in range(0x20) if b not in (9, 10, 13)) + b"\x7f"
_CYR_RX = re.compile(r"[А-Яа-яЁё]")
_LET_RX = re.compile(r"[^\W\d_]")
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    print(f"walk: WARN {msg}", file=sys.stderr, flush=True)


def max_file_bytes(nonpositive=None) -> int:
    """File size cap from ANON_MAX_FILE_BYTES (default 50 MB) — the ONLY
    parser of the switch (gates call the same one, they have no parsers of their own); garbage in
    env = the default with a one-time WARN (not SystemExit: the cap is a safeguard).
    A value <= 0: for the engine — the same garbage (default + WARN); if ``nonpositive``
    is given — return it: gates pass 0 = "cap off" (the structural gate
    reads a file of any size in memory, the control gate scans any
    non-empty file as a stream — their old meaning of zero, pinned by tests)."""
    raw = os.environ.get(MAX_FILE_BYTES_ENV, "").strip()
    if not raw:
        return MAX_FILE_BYTES_DEFAULT
    try:
        n = int(raw)
    except ValueError:
        n = None
    if n is not None and n <= 0 and nonpositive is not None:
        return nonpositive
    if n is None or n <= 0:
        _warn_once("max_file_bytes",
                   f"{MAX_FILE_BYTES_ENV}={raw!r} — not a positive integer; "
                   f"using the default {MAX_FILE_BYTES_DEFAULT}")
        return MAX_FILE_BYTES_DEFAULT
    return n


# A structured profile file (a Telegram export — one result.json for
# the whole account, up to 99.5 MB on real data) is not cut by the common cap: the engine
# reads it as a stream by fields (anonymizer/structured), memory O(a batch of
# messages). Its own ceiling is ANON_MAX_STRUCT_BYTES (default 1 GB): stages
# that read the whole file (intake, mine, verify, residual) hold it in memory.
MAX_STRUCT_BYTES_ENV = "ANON_MAX_STRUCT_BYTES"
MAX_STRUCT_BYTES_DEFAULT = 1024 * 1024 * 1024


def max_struct_bytes() -> int:
    """Size cap of a structured profile file from ANON_MAX_STRUCT_BYTES
    (default 1 GB); garbage in env = the default with a one-time WARN."""
    raw = os.environ.get(MAX_STRUCT_BYTES_ENV, "").strip()
    if not raw:
        return MAX_STRUCT_BYTES_DEFAULT
    try:
        n = int(raw)
        if n <= 0:
            raise ValueError
        return n
    except ValueError:
        _warn_once("max_struct_bytes",
                   f"{MAX_STRUCT_BYTES_ENV}={raw!r} — not a positive integer; "
                   f"using the default {MAX_STRUCT_BYTES_DEFAULT}")
        return MAX_STRUCT_BYTES_DEFAULT


def structured_oversize_ok(path, rel=None, size=None) -> bool:
    """The file is over the common cap, but it is a structured profile export not
    over ANON_MAX_STRUCT_BYTES → it may be read. Profiles are off
    (ANON_STRUCTURED_PROFILES=0) or the signature is not recognized → False (common cap).
    The import is lazy: structured itself imports the engine."""
    if os.environ.get("ANON_STRUCTURED_PROFILES", "1").strip() == "0":
        return False
    try:
        if size is None:
            size = os.stat(path).st_size
        if size > max_struct_bytes():
            return False
        from .. import structured
        return structured.detect(path, rel) is not None
    except Exception:              # noqa: BLE001 — a safeguard, not logic
        return False


def has_text_bom(head: bytes) -> str:
    """Codec name by the BOM at the start of the data ('' — no BOM)."""
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return "utf-32"
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    return ""


# --- exact original form of the text --------------------------------------------
#
# The client gets files IN THE ORIGINAL encoding (owner's decision 2026-09-22:
# cp1251 → cp1251, UTF-16 with BOM → UTF-16 with the same BOM, UTF-8 with BOM → with BOM;
# Excel does not read UTF-8 CSV without a BOM). intake records the form in the ledger
# (bom_name / newline_style), release encodes back (encode_text).
# Check order: utf-32-le before utf-16-le — the common prefix FF FE.
BOM_BYTES = (
    ("utf-8-sig", b"\xef\xbb\xbf"),
    ("utf-32-le", b"\xff\xfe\x00\x00"), ("utf-32-be", b"\x00\x00\xfe\xff"),
    ("utf-16-le", b"\xff\xfe"), ("utf-16-be", b"\xfe\xff"),
)


def bom_name(head: bytes) -> str:
    """Exact BOM name by the byte prefix: utf-8-sig | utf-16-le | utf-16-be |
    utf-32-le | utf-32-be | '' (none). Unlike has_text_bom (the CODEC name:
    utf-16/utf-32 read the byte order from the BOM themselves) — the form that release
    returns to the client byte for byte."""
    for name, pref in BOM_BYTES:
        if head.startswith(pref):
            return name
    return ""


def newline_style(text: str) -> str:
    """Line break style: crlf | lf | cr | mixed | none. Transcoding does not touch
    EOL — the ledger records the style, release checks it before handing out."""
    crlf = text.count("\r\n")
    kinds = [k for k, n in (("crlf", crlf), ("lf", text.count("\n") - crlf),
                            ("cr", text.count("\r") - crlf)) if n]
    if not kinds:
        return "none"
    return kinds[0] if len(kinds) == 1 else "mixed"


def encode_text(text: str, encoding: str, bom: str = "") -> bytes:
    """The reverse of decode_bytes: text → bytes of the original form. bom (a name
    from bom_name) sets both the codec without the BOM and the byte prefix; without a BOM —
    text.encode(encoding). Always strict: an unrepresentable character (a mask introduced
    a character outside cp1251) = UnicodeEncodeError, the policy is the caller's."""
    if bom:
        codec = "utf-8" if bom == "utf-8-sig" else bom
        return dict(BOM_BYTES)[bom] + text.encode(codec)
    return text.encode(encoding)


def is_binary_bytes(head: bytes) -> bool:
    """Binary sniff by the file head: a UTF-16/32 BOM — text (NUL is legal there);
    otherwise NUL or > 30% control bytes → binary."""
    if not head:
        return False
    if has_text_bom(head):
        return False
    if b"\x00" in head:
        return True
    ctrl = sum(head.count(b) for b in _CTRL_BYTES)
    return ctrl > _CTRL_RATIO * len(head)


def skip_reason(path, rel=None, *, max_bytes=None) -> str:
    """Why the file is NOT to be read as text (a contract shared with release):
    '' — text; 'ext .x' — an extension from SKIP_EXT (by rel if given, otherwise
    by path); 'too_large' — size > ANON_MAX_FILE_BYTES (os.stat BEFORE reading;
    a structured profile export — up to ANON_MAX_STRUCT_BYTES);
    'binary-content' — the head sniff (is_binary_bytes); 'unreadable:<error>'."""
    ext = os.path.splitext(rel if rel is not None else path)[1].lower()
    if ext in SKIP_EXT:
        return f"ext {ext}"
    cap = max_file_bytes() if max_bytes is None else max_bytes
    try:
        size = os.stat(path).st_size
        if size > cap and not structured_oversize_ok(path, rel, size):
            return "too_large"
        with open(path, "rb") as fh:
            head = fh.read(SNIFF_BYTES)
    except OSError as exc:
        return f"unreadable:{exc.strerror or exc}"
    if is_binary_bytes(head) or pdf_binary_head(ext, head):
        return "binary-content"
    return ""


def _charset_normalizer():
    try:
        import charset_normalizer
        return charset_normalizer
    except ImportError:
        _warn_once("charset_normalizer",
                   "charset_normalizer is not installed — encoding detection of "
                   "non-UTF8 text uses only the cp1251 heuristic "
                   "(pip install charset_normalizer)")
        return None


# Counters of the decoding ladder — in one regex pass (C), not in a per-character loop
# of Python and a list of one-letter strings: _LET_RX.findall on a 144 MB registry is ~100 million strings in
# memory, generators over characters — minutes per every non-UTF8 archive member (the release archive gate
# decodes a restored cp1251 member with the ladder). The predicates are the same: _CTRL_RX — a character with
# ord < 0x20 outside \t\n\r; _BAD_LET_RX — a letter (_LET_RX) with ord ≥ 0x80 outside Cyrillic _CYR_RX.
_CTRL_RX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_NON_LET_RX = re.compile(r"[\W\d_]+")
_NON_CYR_RX = re.compile(r"[^А-Яа-яЁё]+")          # по строке из одних букв
_BAD_LET_RX = re.compile(r"[^\x00-\x7fА-Яа-яЁё]")   # по строке из одних букв


def _letters(text: str) -> str:
    """Letters of text (characters under _LET_RX) as one string, in text order."""
    return _NON_LET_RX.sub("", text)


def _cp1251_readable(text: str) -> bool:
    let = _letters(text)
    if not let:
        return False
    cyr = len(_NON_CYR_RX.sub("", let))         # every Cyrillic letter is a letter: the count over let is the same
    if cyr < _CP1251_MIN_CYR * len(let):
        return False
    ctrl = len(_CTRL_RX.findall(text))
    return ctrl <= 0.01 * len(text)


def _mostly_ru_en(text: str) -> bool:
    """Sanity filter of a charset_normalizer match: the corpus is Russian-English —
    a decode where >10% of letters are outside Latin/Cyrillic (Hebrew/Greek on a short
    koi8 sample) is rejected in favour of the cp1251 heuristic."""
    let = _letters(text)
    if not let:
        return True
    ok = len(let) - len(_BAD_LET_RX.findall(let))
    return ok >= 0.9 * len(let)


# Encodings the ladder accepts from charset_normalizer (names — codecs.lookup().name):
# the corpus is Russian-English, everything else (big5, cp1253, shift_jis, cp1250, koi8-u…) on a short
# sample is a false match; such a step is skipped, next is the cp1251 heuristic.
_CN_ALLOWED = frozenset({
    "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be",
    "cp1251", "cp866", "koi8-r", "cp1252", "iso8859-1", "mac-cyrillic",
})


def _cn_allowed(encoding) -> bool:
    try:
        return codecs.lookup(str(encoding)).name in _CN_ALLOWED
    except LookupError:
        return False


def decode_bytes(data: bytes):
    """Decoding ladder -> (text, encoding, note). encoding — the name
    of the original encoding ('utf-8' = the bytes are already UTF-8 without a BOM: no copy needed);
    note: 'bom' | 'strict' | 'charset_normalizer' | 'heuristic' |
    'surrogateescape' (the last step: decoding failed, bytes are kept as
    surrogates — the text is only partly readable). A charset_normalizer match
    is accepted only for encodings from _CN_ALLOWED (encoding — its name,
    as before); the others go to the next step."""
    bom = has_text_bom(data[:4])
    if bom:
        try:
            return data.decode(bom), bom, "bom"
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8"), "utf-8", "strict"
    except UnicodeDecodeError:
        pass
    cn = _charset_normalizer()
    if cn is not None:
        try:
            best = cn.from_bytes(data).best()
        except Exception as exc:                     # a library, not a stage
            _warn_once("cn_fail", f"charset_normalizer failed: {exc!r}")
            best = None
        if best is not None and best.chaos <= _CN_MAX_CHAOS \
                and _cn_allowed(best.encoding):
            try:
                text = data.decode(best.encoding)
            except (UnicodeDecodeError, LookupError):
                text = None
            if text is not None and _mostly_ru_en(text):
                return text, best.encoding, "charset_normalizer"
    try:
        text = data.decode("cp1251")
    except UnicodeDecodeError:
        text = None
    if text is not None and _cp1251_readable(text):
        return text, "cp1251", "heuristic"
    return data.decode("utf-8", "surrogateescape"), "utf-8", "surrogateescape"


def read_text_safe(path, *, max_bytes=None):
    """The single point for reading client text: (text, encoding, note).
    text=None — do not read the file: note ∈ 'ext .x' | too_large | binary-content |
    unreadable:<error> (skip_reason: the size is checked by os.stat BEFORE reading,
    binary — by the head, the whole file is not read). Line breaks are not
    translated (like read_text newline='')."""
    reason = skip_reason(path, max_bytes=max_bytes)
    if reason:
        return None, "", reason
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return None, "", f"unreadable:{exc.strerror or exc}"
    return decode_bytes(data)


def safe_str(s: str) -> str:
    """A string for print/json without UnicodeEncodeError: surrogates (non-UTF8
    bytes of names from os.fsdecode) → the escaped form \\udcXX."""
    try:
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:
        return s.encode("utf-8", "backslashreplace").decode("utf-8")


def _segment_safe(seg: str) -> str:
    try:
        seg.encode("utf-8")
        return seg
    except UnicodeEncodeError:
        pass
    raw = seg.encode("utf-8", "surrogateescape")
    best = None
    # cp866 (OEM: a Windows zip without the UTF-8 flag) and cp1251 cover the same
    # bytes — pick the decode where non-ASCII characters are almost all Cyrillic
    for enc in ("cp866", "cp1251"):
        try:
            cand = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if not cand.isprintable():
            continue
        hi = [ch for ch in cand if ord(ch) > 0x7F]
        score = (sum(1 for ch in hi if _CYR_RX.match(ch)) / len(hi)) if hi else 0.0
        if score >= 0.8 and (best is None or score > best[0]):
            best = (score, cand)
    if best is not None:
        return best[1]
    return "".join(ch if ord(ch) < 0xD800 or ord(ch) > 0xDFFF
                   else f"_x{ord(ch) - 0xDC00:02x}" for ch in seg)


def safe_name(rel: str) -> str:
    """A safe name for a relpath with surrogates (a Windows zip with cp866/cp1251):
    per segment — the bytes as cp1251/cp866 if that is readable Cyrillic, otherwise
    a hex suffix _xNN for every bad byte. A valid name is returned as
    is."""
    try:
        rel.encode("utf-8")
        return rel
    except UnicodeEncodeError:
        return "/".join(_segment_safe(seg) for seg in rel.split("/"))


def iter_files(root, skip_ext=SKIP_EXT, sniff=None):
    """Yield (abspath, relpath) for every non-binary file under root.

    sniff=None: the binary head sniff (is_binary_bytes) is on when the
    extension filter is active; ``skip_ext=set()`` ("give me everything", structural: forbidden
    names) turns the sniff off too. sniff=True/False — explicitly."""
    root = os.path.abspath(root)
    if sniff is None:
        sniff = bool(skip_ext)
    for dp, _, fs in os.walk(root):
        for f in fs:
            ext = os.path.splitext(f)[1].lower()
            if ext in skip_ext:
                continue
            p = os.path.join(dp, f)
            if sniff:
                try:
                    with open(p, "rb") as fh:
                        head = fh.read(SNIFF_BYTES)
                except OSError:
                    head = b""       # read_text itself filters out an unreadable file
                if is_binary_bytes(head) or pdf_binary_head(ext, head):
                    continue
            yield p, os.path.relpath(p, root)


# A file that is text by extension with NUL in the first
# SNIFF_BYTES is treated as binary by the sniff, and iter_files silently dropped it from the walk — neither FILE_FAILED
# in the engine nor SKIPPED in the control gate (a silent leak: a file with personal data was not read or masked,
# and the run report did not have it). Now nul_text_files returns such files separately: the engine
# writes them as FILE_FAILED(binary-content) (BLOCK file_failed of the round until a waiver, a record in
# state/limitations.md), the control gate — into SKIPPED_KEY. The sniff, walk and reading are the same
# (skip_reason — «binary-content», the file is not read or touched). Real binaries
# (the extension is not a text one, or the head is "garbage" even without NUL) — silently, as before.
# (The emergency rollback switch to the old silence is removed.)
_NUL_TEXT_EXT = TEXT_EXT | {".eml", ".mbox", ".mbx"}


_UTF16_NUL_SHARE = 0.10     # share of code units with a 0x00 high byte (space, digit, EOL)


def _utf16_like(head: bytes) -> bool:
    """The head looks like UTF-16 without a BOM (PowerShell export, bcp, Excel "Unicode text"):
    NUL bytes lie at offsets of one parity (the high byte of a code unit — for a space, digit,
    punctuation and EOL it is 0x00 in any language; the opposite parity is allowed within 5 %), there are at
    least _UTF16_NUL_SHARE of the number of units, and decoding in this byte order gives text (no more
    control and broken characters than _CTRL_RATIO). Without this, Cyrillic in UTF-16 (0x04 0x1x) after stripping
    NUL for is_binary_bytes is solid control bytes, a "real binary": the criterion would depend
    on the language of the text (Latin in UTF-16 was caught, Cyrillic was not)."""
    even, odd = head[0::2].count(0), head[1::2].count(0)
    major, minor = max(even, odd), min(even, odd)
    if minor * 20 > major or major < _UTF16_NUL_SHARE * (len(head) // 2):
        return False
    text = head[: len(head) & ~1].decode("utf-16-le" if odd >= even else "utf-16-be",
                                         "replace")
    bad = sum(1 for ch in text
              if (ord(ch) < 0x20 and ch not in "\t\n\r") or ch in "\x7f�")
    return bad <= _CTRL_RATIO * len(text)


def is_nul_text(head: bytes) -> bool:
    """The head is text in everything except NUL: there is NUL, no UTF-16/32 BOM (with a BOM NUL is legal —
    such a file is text anyway), and either the binary sniff is silent without the NUL bytes (Latin in UTF-16,
    NUL inside ordinary text), or the head looks like UTF-16 without a BOM (_utf16_like — Cyrillic
    and any non-Latin text)."""
    return (b"\x00" in head and not has_text_bom(head)
            and (not is_binary_bytes(head.replace(b"\x00", b"")) or _utf16_like(head)))


def nul_text_files(root, skip_ext=SKIP_EXT):
    """(abspath, relpath) of files with a text extension (TEXT_EXT, mail) that iter_files
    filtered out by the sniff because of NUL in the head (is_nul_text). The order is as in iter_files."""
    root = os.path.abspath(root)
    for dp, _, fs in os.walk(root):
        for f in fs:
            ext = os.path.splitext(f)[1].lower()
            if ext in skip_ext or ext not in _NUL_TEXT_EXT:
                continue
            p = os.path.join(dp, f)
            try:
                with open(p, "rb") as fh:
                    head = fh.read(SNIFF_BYTES)
            except OSError:
                continue
            if is_nul_text(head):
                yield p, os.path.relpath(p, root)


def read_text(path):
    # newline='' is required: the default universal-
    # newlines mode silently normalizes CRLF→LF, the in-memory branch diverges from
    # the stream (iter_lines/StreamWriter keep EOL as is) and the
    # stream==memory byte oracle breaks on CRLF files.
    try:
        with open(path, encoding="utf-8", errors="surrogateescape",
                  newline="") as fh:
            return fh.read()
    except (OSError, UnicodeError):
        return None


# File/directory name limit in BYTES: Linux NAME_MAX (ext4/xfs/btrfs) is
# 255 bytes of UTF-8, Cyrillic is 2 bytes per letter. macOS counts up to 255
# characters, so the limit is a constant, not pathconf: the same name on
# any machine (determinism of the rename journal).
NAME_MAX_BYTES = 255
_TMP_PREFIX = ".anon."
_TMP_SUFFIX = ".anontmp"
# fit_name: the tail is a space + 16 hex of sha256 of the full name; an extension longer than
# _FIT_EXT_MAX bytes is not an extension but part of the stem; truncation by word —
# back to the last of _FIT_BREAK (the separators of filenames.mask_name).
# The tail separator is a space, not «_»: through a single «_» mask_name glues
# neighbouring pieces into a Google id window (_fold_gid_runs: «Meeting_ABC_<16 hex>» —
# exactly 28 characters → DOC_ID) and takes «_<hex>» for the rest of the id after a DOC_ID
# token (_mask_docid_tails) — a repeated rename pass would rename again
_FIT_HASH_HEX = 16
_FIT_EXT_MAX = 20
_FIT_BREAK = "_. -"
_FIT_SEP = " "
# errno "all writing stopped" (disk full, quota, read-only FS) — not
# a failure of one file: raised as is, loudly
_SYSTEMIC_WRITE = frozenset(
    e for e in (getattr(errno, n, None) for n in ("ENOSPC", "EDQUOT", "EROFS"))
    if e is not None)


class WriteError(OSError):
    """Write failure of ONE file: tmp next to it, writing to tmp, os.replace
    (name, permissions, directory). Pipeline puts the file into FILE_FAILED
    (write_error:<errno>), the pass goes on. A subclass of OSError — the old
    ``except OSError`` catch it as before."""


def _write_failed(e: OSError) -> OSError:
    """OSError of a write → WriteError; systemic (_SYSTEMIC_WRITE) and already
    wrapped — as is."""
    if isinstance(e, WriteError) or e.errno in _SYSTEMIC_WRITE:
        return e
    if e.errno is None:
        return WriteError(*e.args)
    return WriteError(e.errno, e.strerror, e.filename)


class WriteSide:
    """An open atomic-write tmp filled by the caller (assembling
    a file from parts): a write/close failure (EIO, EFBIG, …) — WriteError, as in
    write_text. Reading the sources goes around it — their failure is not a "write failure"."""

    def __init__(self, fh):
        self._fh = fh

    def write(self, data):
        try:
            return self._fh.write(data)
        except OSError as e:
            raise _write_failed(e)

    def close(self):
        try:
            self._fh.close()
        except OSError as e:
            raise _write_failed(e)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _name_bytes(name: str) -> bytes:
    return os.fsencode(name)


def suffix_name(full: str, taken, is_dir: bool, fit) -> str:
    """Separate a colliding rename target with a deterministic suffix _2/_3… before the extension (file) or at the end
    (directory) (as tools/round._suffix_dst); ``full`` — the full path of the masked target before truncation, ``fit(path)`` — the path
    after name truncation (walk.fit_name). The number grows while the path is taken in ``taken`` or on
    disk."""
    base, ext = (full, "") if is_dir else os.path.splitext(full)
    i = 2
    while True:
        cand = fit(f"{base}_{i}{ext}")
        if cand not in taken and not os.path.exists(cand):
            return cand
        i += 1


def fit_name(name: str, is_dir: bool = False,
             limit: int = NAME_MAX_BYTES, stable=None) -> str:
    """A file/directory name within ``limit`` BYTES. Within the limit — as is.
    Longer: the stem is truncated at a character boundary (a word stump — back to
    the last separator if no more than half is lost), the file extension
    (up to _FIT_EXT_MAX bytes) is kept, before it — a space and 16 hex of
    sha256 of the FULL name. Deterministic; long names with a common start
    do not merge; the tail [0-9a-f]{16} is a separate hash segment that
    filenames.mask_name leaves as is.
    ``stable(name) -> bool`` — the caller's masking (rename pass) leaves
    the name as is. If not (the stem stump itself looks maskable) — the stem
    backs off one more word, down to empty (hash + extension): a repeated pass
    over the truncated name is a noop. Not called for names within the limit."""
    raw = _name_bytes(name)
    if len(raw) <= limit:
        return name
    stem, ext = name, ""
    if not is_dir:
        root, dot_ext = os.path.splitext(name)
        if dot_ext and root and len(_name_bytes(dot_ext)) <= _FIT_EXT_MAX:
            stem, ext = root, dot_ext
    tail = hashlib.sha256(raw).hexdigest()[:_FIT_HASH_HEX] + ext
    budget = limit - len(_name_bytes(_FIT_SEP + tail))
    used = n = 0
    for ch in stem:
        w = len(_name_bytes(ch))
        if used + w > budget:
            break
        used += w
        n += 1
    head = stem[:n]
    if n < len(stem) and stem[n] not in _FIT_BREAK:
        cut = max(head.rfind(c) for c in _FIT_BREAK)
        if cut >= n // 2:
            head = head[:cut]
    head = head.rstrip(_FIT_BREAK)
    while True:
        fit = head + _FIT_SEP + tail if head else tail
        if not head or stable is None or stable(fit):
            return fit
        cut = max(head.rfind(c) for c in _FIT_BREAK)
        head = head[:max(cut, 0)].rstrip(_FIT_BREAK)


def _mk_tmp(path):
    """A unique tmp in the destination directory (constitution: mkstemp, NOT path+'.tmp').
    The fixed name ``path+'.anontmp'`` brought down the second writer: two applies on
    one tree wrote into ONE tmp — the second truncated the first one's unfinished tmp,
    and its os.replace failed with a raw FileNotFoundError after someone else's commit
    (the same class already fixed in residual/llm_recheck). The ``.anontmp`` suffix
    is kept (uniqueness in the middle of the name) so that stale-tmp/`_no_tmp`
    recognize it and iter_files can skip it by extension. The name is SHORT and does not
    depend on the file name (``.anon.<rnd>.anontmp``): the old
    ``<basename>.<rnd>.anontmp`` went beyond NAME_MAX for a name of 239 bytes —
    ENAMETOOLONG brought down the whole engine pass on Linux (macOS counts the limit in
    characters, it did not reproduce there). A creation failure is WriteError."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        return tempfile.mkstemp(prefix=_TMP_PREFIX, suffix=_TMP_SUFFIX, dir=d)
    except OSError as e:
        raise _write_failed(e)


def publish(tmp, path):
    """os.replace of the finished tmp into place; a failure is WriteError (as in _mk_tmp)."""
    try:
        os.replace(tmp, path)
    except OSError as e:
        raise _write_failed(e)


def write_text(path, text):
    fd, tmp = _mk_tmp(path)
    # newline='' symmetric to read_text/StreamWriter: EOLs are written without translation
    try:
        try:
            with os.fdopen(fd, "w", encoding="utf-8",
                           errors="surrogateescape", newline="") as fh:
                fh.write(text)
        except OSError as e:
            raise _write_failed(e)
        publish(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_bytes(path, data: bytes):
    """Atomic write of bytes (an assembled mail in the original encoding) —
    the same tmp + publish as write_text; a failure is WriteError, the tmp is removed."""
    fd, tmp = _mk_tmp(path)
    try:
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
        except OSError as e:
            raise _write_failed(e)
        publish(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def iter_lines(path):
    """A line reader for streaming large files: memory O(line),
    the file is never read whole.

    ``newline=''`` — universal splitting on \\n / \\r\\n / \\r, but the EOL
    stays IN the line untouched: unchanged lines go back to disk
    byte for byte (without normalizing line ends). An OPEN error surfaces
    right from this call (not at the first next), so that the caller can tell
    "the file is not readable — skip it, like read_text" from an error in the middle of reading
    (that one surfaces from the iteration and must abort the write, not publish
    a truncated result)."""
    fh = open(path, encoding="utf-8", errors="surrogateescape", newline="")

    def gen():
        with fh:
            yield from fh

    return gen()


class StreamWriter:
    """Atomic streaming write: the same contract as write_text
    (a unique short tmp ``.anon.<rnd>.anontmp`` in the destination directory +
    os.replace), but line by line, without the whole text in memory. ``commit(changed=
    True)`` publishes the result atomically; ``commit(changed=False)`` / ``abort()``
    remove the tmp — a noop file is untouched byte for byte (mtime/inode intact),
    an exception leaves no garbage and never replaces the original
    with a half-written file. A unique tmp (mkstemp, NOT a fixed one) — so that
    a second concurrent writer does not truncate someone else's tmp and does not publish
    an interleave / fail with a raw FileNotFoundError (see _mk_tmp). A write failure is
    WriteError (a commit failure removes the tmp itself)."""

    def __init__(self, path):
        self.path = path
        fd, self.tmp = _mk_tmp(path)
        # newline='' symmetric to iter_lines: EOLs are written as is, without translation
        self._fh = os.fdopen(fd, "w", encoding="utf-8",
                             errors="surrogateescape", newline="")

    def write(self, chunk):
        try:
            self._fh.write(chunk)
        except OSError as e:
            raise _write_failed(e)

    def commit(self, changed):
        """Close the tmp and either atomically replace the original (there were replacements), or
        remove the tmp without touching the original (noop)."""
        try:
            self._fh.close()
            if changed:
                os.replace(self.tmp, self.path)
            else:
                os.remove(self.tmp)
        except OSError as e:
            try:
                os.remove(self.tmp)
            except OSError:
                pass
            raise _write_failed(e)

    def abort(self):
        """Emergency cleanup: close and delete the tmp, the original is not touched."""
        try:
            self._fh.close()
        finally:
            try:
                os.remove(self.tmp)
            except OSError:
                pass
