"""Stage "inventory".

Inventory of the working copy: the kind of each file by the extension table (the table is taken
from anon_pipeline conveyor/filereg.py; Telegram stickers — by the exporter's file name), size.
Archives are unpacked in place into a ``<archive>.files/`` directory (like tools/unpack.py of the
main pipeline); the archive itself goes after a full unpack (its content is next to it); nested
archives — down to ``MAX_DEPTH``. zip, tar, tar.gz/tgz, tar.bz2/tbz2, tar.xz/txz, single
.gz/.bz2/.xz — stdlib; 7z — via ``py7zr``, rar — via ``rarfile`` (if the library is installed).

Data owner's decision of 27.10: an archive that is not unpacked (no unpacker, broken,
password-protected, a "bomb", deeper than ``MAX_DEPTH``) is not removed — it stays as is with the
mark ``ctx.mark(arc, 'unprocessed:archive', 'not anonymized: not unpacked: <reason>')``; a partly
unpacked one (a member was not extracted or repeats a name) also stays next to its unpacked files
with this mark. Links, devices and members with an unsafe path are never extracted — an
``archive_member`` record in the removal log.

Protection: members leaving the directory (``..``, absolute paths, a ``C:`` drive), links
(symlink/hardlink), devices are not unpacked at all; a "bomb" — the total declared size is over
``MAX_TOTAL_BYTES``, the compression ratio is over ``MAX_RATIO`` (from ``RATIO_FLOOR`` of size),
more than ``MAX_MEMBERS`` members, or the actual size on disk went over the limit during
unpacking (we do not trust declared sizes) — the archive is not unpacked and stays with a mark.
Free disk space is checked by the listing and by actual writes (``DISK_RESERVE`` kept spare). The
input archive (``--in``, cli) has no absolute limit — only the ratio and disk space.

Text files (kinds text/structured by the table and files with an extension outside the table)
not in UTF-8 are re-encoded to UTF-8 in place: otherwise the UTF-8 gate blocks the archive, and
the engine reads cp1251 as garbage and full names leak into the output. Detection: strict UTF-8 →
as is; a BOM (UTF-8/UTF-16/UTF-32) → by the BOM, if it reads without errors; otherwise cp1251,
cp866 or koi8-r — by how plausible the Russian text is (``_ru_score``: words of Russian letters,
share of lowercase, frequent letters and bigrams; box drawing is a frame, typography is not
letters). Few letters, close scores or no plausible candidate — the file is not touched (the gate
will show it): damage is worse than refusal, a damaged full name can no longer be masked. The
report gets a note with the number of files, without names; counter ``files:recoded``.

A symbolic link in the working copy is removed (``link``): anything outside the input may be
behind it.

A ``.json``/``.jsonl``/``.ndjson`` that does not parse already in the input gets the
``input_broken:<check>`` mark (gates_struct.mark_input_broken): it is processed like any file —
masked as text — and the structural check at the end skips it, the breakage is not ours.

Result — ``ctx.data['inventory']``: a list of ``{rel, kind, size}`` for all files after
unpacking; counters ``files:<kind>`` and ``archives:*`` — in ``ctx.counters``.
"""
from __future__ import annotations

import codecs
import errno
import lzma
import math
import os
import re
import shutil
import tarfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from ..context import Removal

NAME = "inventory"

KINDS = ("text", "structured", "office", "pdf", "image", "sticker", "svg", "audio", "video",
         "archive", "binary")

# extension → file kind (the conveyor/filereg.py table)
_KIND_EXT: dict = {}
for _k, _exts in {
    "text": (".txt", ".md", ".html", ".htm", ".log", ".yaml", ".yml", ".xml", ".ini", ".cfg",
             ".conf", ".rst", ".eml", ".mbox", ".vcf", ".ics", ".srt", ".vtt"),
    "structured": (".json", ".jsonl", ".ndjson", ".csv", ".tsv"),
    "office": (".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt", ".odt", ".ods", ".odp", ".rtf",
               ".pages", ".numbers", ".one", ".msg"),
    "pdf": (".pdf",),
    "image": (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif",
              ".psd", ".ai", ".avif", ".ico", ".eps", ".cr2", ".nef"),
    "sticker": (".tgs",),
    "svg": (".svg",),
    "audio": (".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac", ".wma", ".amr",
              ".aiff", ".aif"),
    "video": (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".3gp", ".wmv", ".flv", ".mpg",
              ".mpeg"),
    "archive": (".zip", ".gz", ".tgz", ".zst", ".7z", ".rar", ".tar", ".bz2", ".xz", ".lz4",
                ".lzma", ".jar", ".war", ".tbz", ".tbz2", ".txz", ".lz", ".cab", ".dmg", ".iso"),
}.items():
    for _e in _exts:
        _KIND_EXT[_e] = _k
_BINARY_EXT = {
    ".db", ".bin", ".pkl", ".woff", ".woff2", ".ttf", ".otf", ".apk", ".exe", ".dll", ".so",
    ".dylib", ".o", ".a", ".class", ".pyc", ".pyo", ".wasm", ".lnk", ".eot", ".ttc", ".cur",
    ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".parquet", ".npy", ".npz", ".h5", ".hdf5",
    ".pt", ".pth", ".onnx", ".safetensors", ".pickle", ".joblib", ".msi", ".lib", ".pyd", ".swf",
    ".db3", ".mdb", ".accdb", ".avro", ".orc", ".feather", ".ckpt", ".anontmp", ".pst", ".ost",
}
for _e in _BINARY_EXT:
    _KIND_EXT[_e] = "binary"

# Telegram exporter names: a sticker and its preview — no user data
_STICKER_RX = re.compile(r"^sticker_\d+(?:@[\d_-]+)?\.\w{1,5}(?:_thumb\.jpg)?$")
_WIN_DRIVE = re.compile(r"^[A-Za-z]:")

OUT_SUFFIX = ".files"
MAX_DEPTH = 3                        # nested archives: deeper ones are kept with a mark
MAX_TOTAL_BYTES = 2 << 30            # total unpacked size of one archive (nested included)
MAX_RATIO = 100                      # unpacked / archive size
RATIO_FLOOR = 64 << 20               # the ratio is checked only from this size
MAX_MEMBERS = 10000
_CHUNK = 1 << 20

# zip/tar — stdlib; a single compressed stream — the module from the table; 7z/rar — optional packages
_TAR_EXT = (".tar", ".tgz", ".tbz", ".tbz2", ".txz", ".tar.gz", ".tar.bz2", ".tar.xz")
_STREAM_EXT = {".gz": "gzip", ".bz2": "bz2", ".xz": "lzma", ".lzma": "lzma"}
_ZIP_EXT = (".zip", ".jar", ".war")
# member/stream read errors: broken data (zlib/lzma are not OSError subclasses), a cut file,
# bz2/gzip raise OSError/EOFError/ValueError
_READ_ERRORS = (OSError, EOFError, ValueError, RuntimeError, zlib.error, lzma.LZMAError)


def kind_of(name: str) -> str:
    """File kind by name: a Telegram sticker — by the exporter's name, otherwise the extension
    table; an extension outside the table is text (the main pipeline's intake reads it so; see
    known_ext)."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    if _STICKER_RX.match(base):
        return "sticker"
    ext = os.path.splitext(base)[1].lower()
    return _KIND_EXT.get(ext, "text")


def known_ext(name: str) -> bool:
    """The extension is in the kind table (or it is a Telegram sticker). A file outside the table
    has an unknown kind: gates_struct keeps it only if it reads as UTF-8."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    if _STICKER_RX.match(base):
        return True
    return os.path.splitext(base)[1].lower() in _KIND_EXT


def archive_format(name: str) -> str | None:
    """zip | tar | stream:<module> | 7z | rar | None (not an archive or a format with no unpacker)."""
    low = name.lower()
    if low.endswith(_ZIP_EXT):
        return "zip"
    if low.endswith(_TAR_EXT):
        return "tar"
    ext = os.path.splitext(low)[1]
    if ext in _STREAM_EXT:
        return "stream:" + _STREAM_EXT[ext]
    if ext == ".7z":
        return "7z"
    if ext == ".rar":
        return "rar"
    return None


def is_archive(name: str) -> bool:
    return kind_of(name) == "archive"


_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {
    f"LPT{i}" for i in range(1, 10)}


def safe_member(path: str) -> bool:
    """The member path is safe: relative, without "..", "." and empty segments, not a Windows
    drive, without ":" in a segment (an NTFS alternate stream ``a:b``) and without reserved Windows
    names (``CON``, ``NUL``, ``aux.txt`` — on Windows these are the console/null, not a file). One
    rule on all platforms, so the output does not depend on the OS."""
    p = path.replace("\\", "/")
    if not p or p.startswith("/") or _WIN_DRIVE.match(p):
        return False
    for s in p.split("/"):
        if s in ("", ".", "..") or ":" in s:
            return False
        if s.split(".", 1)[0].rstrip(" ").upper() in _WIN_RESERVED:
            return False
    return True


_LAT = re.compile(r"[A-Za-z]")
_RU_RUN = re.compile(r"[а-яА-ЯёЁ]+")
_GLUED = re.compile(r"[A-Za-z][а-яА-ЯёЁ]|[а-яА-ЯёЁ][A-Za-z]")


def _cp437_name(name: str) -> bool:
    """The name after cp866 looks like real cp437 (a Western archiver): no Cyrillic at all
    («╔═══╗ box.txt») or it is glued into Latin words and there is less of it than Latin in the
    name stem («cafВ rВsumВ», «GrБсe MБller»). A Russian name with Latin after a space or a sign
    («Отчёт v2», «IMG_Скан») is cp866."""
    stem = os.path.splitext(name.replace("\\", "/").rsplit("/", 1)[-1])[0]
    cyr = sum(len(m.group()) for m in _RU_RUN.finditer(name))
    if not cyr:
        return True
    return bool(_GLUED.search(name)) and cyr < len(_LAT.findall(stem))


# a name we trust as UTF-8 without the flag: Latin (with Latin-1 Supplement and Extended),
# Cyrillic, digits, space and common punctuation. A short Russian cp866 name can also be valid
# UTF-8 (р–я = E0–EF and Ё/ё = F0/F1 lead 3–4-byte characters): «сад» → «ᠤ», «Ёлка» → «𫪠»
_PLAIN_NAME = re.compile(r"[\x20-\x7e -ɏḀ-ỿЀ-ӿ‐-…№]*")
_RU_NAME = re.compile(r"[\x20-\x7eЀ-ӿ№]*")
_RU_CASE = re.compile(r"[А-ЯЁ]?[а-яё]*|[А-ЯЁ]+")
# CJK, Hangul, kana (with fullwidth forms and ideographs beyond U+FFFF): a three-letter cp866
# word starting with х–щ, ы, ь is one such character by bytes («表» = «шби»). Rare ideographs of
# Extension A (U+3400–4DBF) and compatibility (U+F900–FAFF) do not count: these are Russian words
# starting with у/ф/я («уже» = «㦥», «фон» = «䮭», «яда» = «鸞»)
_CJK = re.compile("[\u1100-\u11ff\u2e80-\u2fdf\u3000-\u33ff\u4e00-\u9fff"
                  "\ua960-\ua97f\uac00-\ud7ff\ufe30-\ufe4f\uff00-\uffef"
                  "\U00020000-\U0003ffff]")
_TOKEN = re.compile(r"[^\W_]+")       # a name word: letters and digits in a row («第1章» is one)


def _plain_name(name: str) -> bool:
    return bool(_PLAIN_NAME.fullmatch(name))


def _ru_name(name: str) -> bool:
    """The name after cp866 is plausible Russian: only ASCII, Cyrillic and "№", Cyrillic is
    present, words are lowercase, ALL CAPS or Capitalized («сад», «ТЗ», «Ёлка»; UTF-8 «日本語» in
    cp866 is «цЧецЬмшкЮ») and not glued into Latin words (see ``_cp437_name``)."""
    if not _RU_NAME.fullmatch(name) or not _RU_RUN.search(name):
        return False
    if not all(_RU_CASE.fullmatch(m.group()) for m in _RU_RUN.finditer(name)):
        return False
    return not _cp437_name(name)


def _cjk_name(name: str) -> bool:
    """All non-ASCII in the name is CJK, Hangul or kana («表.xlsx», «第1章.txt», «보고서.hwp»)."""
    wide = [ch for ch in name if ch >= "\x80"]
    return bool(wide) and all(_CJK.match(ch) for ch in wide)


def _ru_name_strong(name: str) -> bool:
    """The cp866 name has a Russian word of 4+ letters («Ёлка») or two words with Cyrillic
    («сон/фон»; «фон 2» — no: «2» has no Cyrillic). «чмм1чла» («第1章») is one word."""
    if any(len(m.group()) >= 4 for m in _RU_RUN.finditer(name)):
        return True
    return sum(1 for t in _TOKEN.findall(name) if _RU_RUN.search(t)) >= 2


def zip_member_name(info: zipfile.ZipInfo) -> str:
    """Name of a zip member. With the UTF-8 flag (bit 11) — as is. Without the flag zipfile
    decodes the name bytes as cp437 (Cyrillic → «ÄΓτ±Γ»): we get the bytes back and try strict
    UTF-8, then cp866 (DOS/Windows archivers) — otherwise the engine will not recognize a full
    name in the file name. Valid UTF-8 is taken at once only if it has Latin, Cyrillic, digits and
    punctuation (``_PLAIN_NAME``); with other characters («ᠤ.txt», but also «资料.txt») — cp866 if
    it gives a plausible Russian name (``_ru_name``: «сад.txt»), otherwise UTF-8. If the UTF-8 is
    all CJK/Hangul/kana (``_cjk_name``), cp866 also needs a word of 4+ letters or a second word
    (``_ru_name_strong``): otherwise «表.xlsx» would become «шби.xlsx». Trade-off: a three-letter
    cp866 name starting with х–щ («чай.txt», «шаг.txt») is one CJK character by bytes and stays so
    («砩.txt»); with у/ф («уже», «фон») it is an Extension A character, which does not count, and
    the name is Russian. If after cp866 there is no Cyrillic or it is only glued into Latin words
    («cafВ»), this is real cp437 («café») and the name stays as zipfile read it."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:
        return info.filename
    try:
        utf = raw.decode("utf-8")
    except UnicodeDecodeError:
        utf = None
    if utf is not None and _plain_name(utf):
        return utf
    name = raw.decode("cp866", errors="replace")
    if utf is not None:
        if not _ru_name(name) or _cjk_name(utf) and not _ru_name_strong(name):
            return utf
        return name
    if _cp437_name(name):
        return info.filename
    return name


class UnpackError(Exception):
    """The archive is not unpacked: a short reason, without content (goes to the data owner's
    report)."""


DISK_RESERVE = 256 << 20             # free disk space we keep spare:
DISK_RESERVE_MAX = 2 << 30           # max(256 MiB, 2 % of the disk), but at most 2 GiB — otherwise
#                                      a big nearly full disk could not unpack even 5 bytes
NO_SPACE_HINT = ("not enough disk space to unpack — unpack the input yourself and pass the "
                 "directory (--in <directory>) or set --tmp on a disk with enough room")


def free_bytes(path: Path) -> int | None:
    """Free space on the disk of ``path`` (or its nearest existing parent) minus the reserve
    ``min(max(DISK_RESERVE, 2 % of the disk), DISK_RESERVE_MAX)``; None — cannot be found out."""
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        du = shutil.disk_usage(p)
    except OSError:
        return None
    reserve = min(max(DISK_RESERVE, du.total // 50), DISK_RESERVE_MAX)
    return max(0, du.free - reserve)


class _Budget:
    """Unpacked size budget for a top-level archive (nested ones share it): the absolute limit
    ``total`` (None — no limit: the ``--in`` input archive), the compression ratio from the archive
    size and free disk space (by the listing and by actual writes)."""

    def __init__(self, total: int | None, archive_size: int, dest: Path | None = None):
        self.total = total
        self.cap = total
        if archive_size > 0:
            ratio_cap = max(RATIO_FLOOR, MAX_RATIO * archive_size)
            self.cap = ratio_cap if total is None else min(total, ratio_cap)
        self.free = free_bytes(dest) if dest is not None else None
        self.used = 0

    def take(self, n: int) -> None:
        self.used += n
        if self.cap is not None and self.used > self.cap:
            raise UnpackError(f"looks like an archive bomb: more than {self.cap} bytes unpacked")
        if self.free is not None and self.used > self.free:
            raise UnpackError(NO_SPACE_HINT)


def _check_listing(sizes: list, archive_size: int, budget: _Budget) -> None:
    if len(sizes) > MAX_MEMBERS:
        raise UnpackError(f"{len(sizes)} members > {MAX_MEMBERS}")
    total = sum(sizes)
    if budget.total is not None and total + budget.used > budget.total:
        raise UnpackError(f"declared size {total} bytes is over the limit {budget.total}")
    if total > RATIO_FLOOR and archive_size > 0 and total > MAX_RATIO * archive_size:
        raise UnpackError(f"declared size {total} bytes > {MAX_RATIO}× the archive size")
    if budget.free is not None and total + budget.used > budget.free:
        raise UnpackError(f"declared size {total} bytes, free {budget.free} — "
                          + NO_SPACE_HINT)


def _copy_limited(src, dst_path: Path, budget: _Budget) -> int:
    """Stream → file in chunks, counting the budget while writing (we do not trust the declared
    size). A read failure in the middle of a member removes the half-written file and re-raises."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with open(dst_path, "wb") as out:
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                n += len(chunk)
                budget.take(len(chunk))
                out.write(chunk)
    except BaseException as e:
        try:
            dst_path.unlink()
        except OSError:
            pass
        if isinstance(e, OSError) and e.errno == errno.ENOSPC:
            raise UnpackError(NO_SPACE_HINT) from e
        raise
    return n


def _unpack_zip(path: Path, dest: Path, budget: _Budget, skipped: list) -> None:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError, ValueError) as e:
        raise UnpackError(f"broken zip ({type(e).__name__})")
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        _check_listing([i.file_size for i in infos], path.stat().st_size, budget)
        seen: set = set()
        for info in infos:
            name = zip_member_name(info)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                skipped.append((name, "символическая ссылка"))
                continue
            if not safe_member(name):
                skipped.append((name, "опасный путь"))
                continue
            key = name.replace("\\", "/")
            if key in seen:                  # a second member with the same name (also after
                skipped.append((name, "duplicate name"))   # re-encoding) does not overwrite
                continue
            seen.add(key)
            if info.flag_bits & 0x1:
                raise UnpackError("password-protected members")
            try:
                with zf.open(info) as src:
                    _copy_limited(src, dest / PurePosixPath(name.replace("\\", "/")), budget)
            except UnpackError:
                raise
            except (zipfile.BadZipFile, *_READ_ERRORS) as e:
                if "password" in str(e).lower() or "encrypted" in str(e).lower():
                    raise UnpackError("password-protected members")
                skipped.append((name, f"not extracted ({type(e).__name__})"))


def _unpack_tar(path: Path, dest: Path, budget: _Budget, skipped: list) -> None:
    try:
        tf = tarfile.open(path, mode="r:*")
    except (tarfile.TarError, *_READ_ERRORS) as e:
        raise UnpackError(f"broken tar ({type(e).__name__})")
    with tf:
        members = []
        total = 0
        while True:
            try:
                m = tf.next()
            except (tarfile.TarError, *_READ_ERRORS) as e:
                raise UnpackError(f"broken tar ({type(e).__name__})")
            if m is None:
                break
            if m.isdir():
                continue
            members.append(m)
            total += max(0, m.size)
            if len(members) > MAX_MEMBERS or (budget.total is not None
                                              and total + budget.used > budget.total):
                break
        _check_listing([max(0, m.size) for m in members], path.stat().st_size, budget)
        for m in members:
            name = m.name
            if m.issym() or m.islnk():
                skipped.append((name, "ссылка"))
                continue
            if not m.isfile():
                skipped.append((name, "не файл"))
                continue
            if not safe_member(name):
                skipped.append((name, "опасный путь"))
                continue
            try:
                src = tf.extractfile(m)
            except (tarfile.TarError, *_READ_ERRORS) as e:
                skipped.append((name, f"not extracted ({type(e).__name__})"))
                continue
            if src is None:
                skipped.append((name, "not extracted"))
                continue
            with src:
                try:
                    _copy_limited(src, dest / PurePosixPath(name.replace("\\", "/")), budget)
                except UnpackError:
                    raise
                except (tarfile.TarError, *_READ_ERRORS) as e:
                    skipped.append((name, f"not extracted ({type(e).__name__})"))


def _unpack_stream(path: Path, dest: Path, budget: _Budget, module: str) -> None:
    """A single compressed stream (.gz/.bz2/.xz without tar): one member named as the archive
    without the extension."""
    import importlib
    mod = importlib.import_module(module)
    inner = path.name[: -len(path.suffix)] or "data"
    if not safe_member(inner):
        raise UnpackError("unsafe member name")
    try:
        with mod.open(path, "rb") as src:
            _copy_limited(src, dest / inner, budget)
    except UnpackError:
        raise
    except _READ_ERRORS as e:
        raise UnpackError(f"broken stream ({type(e).__name__})")


def _unpack_7z(path: Path, dest: Path, budget: _Budget, skipped: list) -> None:
    try:
        import py7zr
    except ImportError:
        raise UnpackError("py7zr library is missing (pip install py7zr)")
    try:
        z = py7zr.SevenZipFile(path, mode="r")
    except Exception as e:  # noqa: BLE001 — py7zr raises its own and system errors
        if "password" in str(e).lower():
            raise UnpackError("password-protected members")
        raise UnpackError(f"broken 7z ({type(e).__name__})")
    with z:
        if getattr(z, "needs_password", lambda: False)():
            raise UnpackError("password-protected members")
        infos = [i for i in z.list() if not i.is_directory]
        _check_listing([i.uncompressed for i in infos], path.stat().st_size, budget)
        names = []
        for i in infos:
            if not safe_member(i.filename):
                skipped.append((i.filename, "опасный путь"))
                continue
            names.append(i.filename)
        try:
            data = z.read(targets=names) if names else {}
        except Exception as e:  # noqa: BLE001
            raise UnpackError(f"broken 7z ({type(e).__name__})")
        for name, bio in data.items():
            if not safe_member(name):
                continue
            _copy_limited(bio, dest / PurePosixPath(name.replace("\\", "/")), budget)


def _unpack_rar(path: Path, dest: Path, budget: _Budget, skipped: list) -> None:
    try:
        import rarfile
    except ImportError:
        raise UnpackError("rarfile library is missing (pip install rarfile and unrar/bsdtar)")
    try:
        rf = rarfile.RarFile(str(path))
    except Exception as e:  # noqa: BLE001
        raise UnpackError(f"broken rar ({type(e).__name__})")
    with rf:
        if rf.needs_password():
            raise UnpackError("password-protected members")
        infos = [i for i in rf.infolist() if not i.is_dir()]
        _check_listing([i.file_size for i in infos], path.stat().st_size, budget)
        for i in infos:
            name = i.filename
            if getattr(i, "is_symlink", lambda: False)():
                skipped.append((name, "символическая ссылка"))
                continue
            if not safe_member(name):
                skipped.append((name, "опасный путь"))
                continue
            try:
                with rf.open(i) as src:
                    _copy_limited(src, dest / PurePosixPath(name.replace("\\", "/")), budget)
            except UnpackError:
                raise
            except Exception as e:  # noqa: BLE001
                skipped.append((name, f"not extracted ({type(e).__name__})"))


def extract_archive(path: Path, dest: Path, budget: _Budget | None = None, *,
                    unlimited: bool = False) -> list:
    """Unpack the archive ``path`` into ``dest`` (the directory is created). Returns a list of
    ``(member name, reason)`` of skipped members. ``UnpackError`` — the archive is not unpacked as
    a whole (broken, password, bomb, no space, no library); dest is cleaned up then.
    ``unlimited`` — no absolute size limit (the ``--in`` input archive): the compression ratio
    and free disk space still apply."""
    fmt = archive_format(path.name)
    if fmt is None:
        raise UnpackError("no unpacker for this format")
    if budget is None:
        budget = _Budget(None if unlimited else MAX_TOTAL_BYTES, path.stat().st_size, dest)
    skipped: list = []
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if fmt == "zip":
            _unpack_zip(path, dest, budget, skipped)
        elif fmt == "tar":
            _unpack_tar(path, dest, budget, skipped)
        elif fmt.startswith("stream:"):
            _unpack_stream(path, dest, budget, fmt.split(":", 1)[1])
        elif fmt == "7z":
            _unpack_7z(path, dest, budget, skipped)
        elif fmt == "rar":
            _unpack_rar(path, dest, budget, skipped)
    except UnpackError:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    if not any(p.is_file() for p in dest.rglob("*")):
        shutil.rmtree(dest, ignore_errors=True)      # do not leave an empty unpack directory
    return skipped


# ------------------------------------------------------------------ text file encoding

_SAMPLE = 1 << 20                    # bytes used to pick an 8-bit encoding
_CANDIDATES = ("cp1251", "cp866", "koi8-r")
# Thresholds of Russian text plausibility for a candidate (all on letters inside words of ≥ 2
# Russian letters: a single letter between digits, spaces and ASCII punctuation is no evidence:
# cp1251 nbsp/dash look like «а»/«Ч» in cp866).
_MIN_LETTERS = 4                     # fewer letters — too little evidence, do not touch
_CYR_SHARE = 0.7                     # Russian letters among non-ASCII characters without frame and typography
_LOWER_SHARE = 0.5                   # lowercase among letters of words not in ALL CAPS (koi8-r
#                                      read as cp1251 flips the case: «йЧБО рЕФТПЧ»)
_TOP_SHARE = 0.4                     # frequent letters among letters (≈ 0.74 in Russian text)
_BIGRAM_SHARE = 0.25                 # frequent bigrams among bigrams inside words («мммм» of a frame)
_MIXED_SHARE = 0.1                   # letters in words "Cyrillic right next to another script"
_MIN_LOGP = -4.8                     # mean log2 p(letter) by Russian letter frequencies: Russian
#                                      text ≈ −4.0…−4.7 bits, a wrong code page ≤ −4.8…−6.8
_MARGIN = 0.3                        # gap (bits) between the best and the second candidate, otherwise do not touch
_TOP_LETTERS = frozenset("оеаинтсрвлкм")
# letter frequencies of Russian text, % (ё counts as е)
_RU_FREQ = {"о": 10.97, "е": 8.49, "а": 8.01, "и": 7.35, "н": 6.70, "т": 6.26, "с": 5.47,
            "р": 4.73, "в": 4.54, "л": 4.40, "к": 3.49, "м": 3.21, "д": 2.98, "п": 2.81,
            "у": 2.62, "я": 2.01, "ы": 1.90, "ь": 1.74, "г": 1.70, "з": 1.65, "б": 1.59,
            "ч": 1.44, "й": 1.21, "х": 0.97, "ж": 0.94, "ш": 0.73, "ю": 0.64, "ц": 0.48,
            "щ": 0.36, "э": 0.32, "ф": 0.26, "ъ": 0.04}
_RU_LOGP = {k: math.log2(v / 100) for k, v in _RU_FREQ.items()}
_FREQ_BIGRAMS = frozenset("""
ст но то на ен ов ни ра во ко ро ан ер не ос ре ль ол по пр ом ет ел од ал те го ли ем ле ин
ва ск ес ор ка ат ис ед ар та ве ти ны ла он ит ри об ло ия мо де за ме ей ых да ас ам бы ки
их ег ок ак ие ми ча чт аз из ую ир ур ав ну ду ят ус им ты от ру ут ог ич вы вс вн ыл ил нн
сс ые ый ой ая ее ью ья ьн зн дн мн рн сн тн чн шн жн зд зв ск сп св см сл тв тр др бр гр кр
вр щи щу ща ше ши ша же жи жа жд це ци ца чи че чу чк хо ха хи юд юч ют ющ яз ял ян ям яв яс
ях эт эк эл эн бл бо бе ба би бу гд гл ги ге га гу ди до ды дл дв дс зы зо зи зе йн йс йт кт
кл кн ку кс ке лу лы лю ля лк мы му мп мб нт нд нк нц нг нс ня ню нь оп ос оц оч ош ощ ою оя
ож оз ои ой ол пи пе пу пы пл пп ры ря рь рд рг рк рм рт рх сь ся си се су сы сх сч тк тс ть
тя ту уд уж уз ук ул ум ун уп уч уш фи фо фе фа фу хн цы чь шк шл щн ыв ым ын ыр ыс ыт ыш ьк
ьм ьс ьт ьш юр юс ящ яю вл вт вп вк вз вд вш гн гр жу жк зм зл зу зя йк йд йо кв кр кц ки ля
лг лж лс лт мл нз нр ня пт пс рв рп рс рш рю ря сн ст сг сж ск сл см ср сф сх сц ти тл тн тм
тх ты ун ус фр фт хв хр хт цв ча чт шт ыг ыд ыз ык ыл ьб ьг ьд ьз ьч эр эс юб юл юн ют юч яд
як яр ят ян ям
""".split())
_RU = re.compile(r"[а-яА-ЯёЁ]")
_TEXT_LETTER = re.compile(r"[A-Za-zа-яА-ЯёЁ]")
_WORD = re.compile(r"[^\W\d_]+")      # a word: letters of any script in a row
_RUNE = re.compile(r"([^\W\d_])\1{2,}")   # a run of ≥ 3 equal letters: frame «═» → «ННН»
_BOX = re.compile(r"[─-▟]")  # DOS box drawing — a table frame, not letters
# DOS frame lines and fills: none of these cp866 bytes is a Russian letter in cp1251 except
# «Н» (═), «Д» (─) and «Ы» (█) — Russian ALL CAPS cp1251 read as cp866 has few of them
_BOX_LINES = frozenset("═─│║░▒▓█")
_BOX_SHARE = 0.5                     # share of box drawing among non-ASCII cp866 and of lines in it:
#                                      above it the file is a DOS frame, not text in another encoding
_TYPO = frozenset(" –—…«»•“”‘’№°±·")   # typography that fits Russian text
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_XML_DECL_ENC = re.compile(rb'(<\?xml[^>]*?encoding=["\'])([^"\']+)(["\'])', re.I)
_BOMS = ((codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF32_LE, "utf-32"),
         (codecs.BOM_UTF32_BE, "utf-32"), (codecs.BOM_UTF16_LE, "utf-16"),
         (codecs.BOM_UTF16_BE, "utf-16"))


def _is_strict_utf8(path: Path) -> bool:
    dec = codecs.getincrementaldecoder("utf-8")()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_CHUNK)
                if not chunk:
                    dec.decode(b"", final=True)
                    return True
                dec.decode(chunk)
    except (UnicodeDecodeError, OSError):
        return False


def _ru_score(text: str) -> float | None:
    """Plausibility of Russian text — the mean log2 p(letter) by Russian letter frequencies (bits,
    the closer to zero the better), or None — does not look like Russian text.

    Only words of Russian letters of length 2+ count: a single letter between digits and ASCII
    is no evidence (cp1251 nbsp and dash in cp866 are «а» and «Ч»); a word of one repeated letter,
    a word of two different letters in ALL CAPS or with a repeat, and a run of ≥ 3 equal letters
    are a frame (cp866 «═» via cp1251 is «НННН», «══╤══» is «ННСНН», via koi8-r «ммсмм»; also «ТЗ»,
    «мама») and count neither as letters nor as bigrams. Box drawing is a frame, typography (nbsp,
    dashes, quotes, №) is not letters; other non-ASCII characters are "foreign" (their share is
    limited by ``_CYR_SHARE``). A word with Russian and foreign letters side by side («GrьЯe») counts
    against. Refusal on few letters, flipped case (share of lowercase in words not typed in ALL
    CAPS: abbreviations «ООО», «ИНН» do not count), rare letters or rare bigrams."""
    if _CTRL.search(text):
        return None
    ru_all = odd = 0
    for ch in text:
        if ch < "\x80" or ch in _TYPO or _BOX.match(ch):
            continue
        if _RU.match(ch):
            ru_all += 1
        else:
            odd += 1
    n = cased = lower = top = mixed = bigrams = freq = 0
    logp = 0.0
    for w in _WORD.findall(_RUNE.sub(" ", text)):
        k = len(set(w))
        if k == 1 or k == 2 and (w.isupper() or len(w) > 2):
            continue                                # «нн», «ННСНН», «ммсмм» — a separator
        ru = len(_RU.findall(w))
        if ru != len(w):
            if ru:
                mixed += len(w)
            continue
        if len(w) < 2:
            continue
        n += len(w)
        low = w.lower().replace("ё", "е")
        if not w.isupper():
            cased += len(w)
            lower += sum(1 for ch in w if ch.islower())
        top += sum(1 for ch in low if ch in _TOP_LETTERS)
        logp += sum(_RU_LOGP[ch] for ch in low)
        for a, b in zip(low, low[1:]):
            bigrams += 1
            freq += (a + b) in _FREQ_BIGRAMS
    if n < _MIN_LETTERS or mixed > _MIXED_SHARE * n:
        return None
    if ru_all / (ru_all + odd) < _CYR_SHARE:
        return None
    if cased and lower / cased < _LOWER_SHARE:
        return None
    if top / n < _TOP_SHARE or freq / bigrams < _BIGRAM_SHARE:
        return None
    score = logp / n
    return score if score >= _MIN_LOGP else None


def _bom_text_ok(head: bytes, enc: str) -> bool:
    """The start of a file with a BOM reads with the codec without errors, without control
    characters, and has at least one Latin or Russian letter: a binary file that happens to start
    with FF FE is not the text «Ā»."""
    try:
        text = codecs.getincrementaldecoder(enc)().decode(head, final=False)
    except UnicodeDecodeError:
        return False
    return not _CTRL.search(text) and bool(_TEXT_LETTER.search(text))


def _dos_frame(text: str) -> bool:
    """The cp866 reading is a DOS frame: box drawing is over ``_BOX_SHARE`` of non-ASCII
    characters, and lines and fills (``_BOX_LINES``) are over ``_BOX_SHARE`` of box drawing.
    Russian ALL CAPS cp1251 in cp866 is all box drawing too, but junctions and corners («╬╥╫»), not
    lines; lowercase koi8-r too, and exactly half («да» — «─┴», «Ёё» — «│г») is not a frame."""
    wide = box = lines = 0
    for ch in text:
        if ch < "\x80":
            continue
        wide += 1
        if _BOX.match(ch):
            box += 1
            lines += ch in _BOX_LINES
    return box > _BOX_SHARE * wide and lines > _BOX_SHARE * box


def detect_text_encoding(path: Path) -> str | None:
    """Encoding of a text file if it is NOT plain UTF-8 and can be detected with confidence:
    'utf-8-sig' (UTF-8 with a BOM), 'utf-16'/'utf-32' (by the BOM), 'cp1251', 'cp866', 'koi8-r'.
    None — the file is already UTF-8 without a BOM, or the encoding is not recognized (must not be
    touched). Of the 8-bit ones the candidate with the best ``_ru_score`` wins, and only if the
    second is at least ``_MARGIN`` behind. A DOS frame (``_dos_frame``) without Russian text in the
    cp866 reading is not re-encoded at all: junctions «╤╪┼» via cp1251 are «ЙННС», «ННШНН», frequent
    letters."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(_SAMPLE)
    except OSError:
        return None
    for bom, enc in _BOMS:                      # UTF-32 before UTF-16: FF FE 00 00
        if head.startswith(bom):
            return enc if _bom_text_ok(head, enc) else None
    if _is_strict_utf8(path):
        return None
    if b"\0" in head:
        return None
    scored = []
    for enc in _CANDIDATES:
        s = _ru_score(head.decode(enc, errors="replace"))
        if s is not None:
            scored.append((s, enc))
    if not scored:
        return None
    if all(enc != "cp866" for _, enc in scored) and _dos_frame(head.decode("cp866", "replace")):
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] - scored[1][0] < _MARGIN:
        return None
    return scored[0][1]


def recode_to_utf8(path: Path, enc: str) -> None:
    """Rewrite the file to UTF-8 without a BOM as a stream (via a temp file next to it +
    os.replace); the encoding in the XML declaration becomes utf-8."""
    tmp = path.with_name(path.name + ".recode.anontmp")
    dec = codecs.getincrementaldecoder(enc)(errors="replace")
    first = True
    try:
        with open(path, "rb") as src, open(tmp, "wb") as dst:
            while True:
                chunk = src.read(_CHUNK)
                text = dec.decode(chunk, final=not chunk)
                data = text.encode("utf-8")
                if first:
                    first = False
                    data = data.removeprefix(codecs.BOM_UTF8)   # in case the decoder emits a BOM
                    if path.suffix.lower() == ".xml":
                        data = _XML_DECL_ENC.sub(rb"\1utf-8\3", data, count=1)
                dst.write(data)
                if not chunk:
                    break
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


_RECODE_BATCH = 16                   # files per pool task of recode_texts


def _recode_batch(_state, paths: list) -> list:
    """Detection and re-encoding of a slice of files (in a worker of ``fork_map`` or here) → per file
    the encoding it was re-encoded from, or None (UTF-8 already, not recognized, or the rewrite failed —
    the UTF-8 gate will show the file in the report)."""
    out = []
    for p in paths:
        enc = detect_text_encoding(p)
        if enc is not None:
            try:
                recode_to_utf8(p, enc)
            except (OSError, UnicodeError, LookupError):
                enc = None
        out.append(enc)
    return out


def _batch_bytes(paths: list) -> int:
    n = 0
    for p in paths:
        try:
            n += p.stat().st_size
        except OSError:
            pass
    return n


def recode_texts(ctx) -> None:
    """All text files not in UTF-8 → UTF-8; a note without names. These are the kinds
    text/structured by the table and files with an extension outside the table (``kind_of`` gives
    them text): ``.sql`` in UTF-16 with a BOM from SSMS, ``.ps1`` from PowerShell ISE, ``.py`` in
    cp1251 are text and must not be removed as "binary of unknown kind"; when in doubt the detector
    leaves the file alone, and gates_struct handles it with a mark, as before. The files are read and
    rewritten in forked processes over the CPU slots of the hw plan (each file on its own)."""
    from .. import engine_adapter as EA
    from .anonymize import _engine_workers
    files = [p for p in ctx.files() if not p.is_symlink() and kind_of(p.name) in ("text", "structured")]
    batches = [files[i:i + _RECODE_BATCH] for i in range(0, len(files), _RECODE_BATCH)]
    workers = _engine_workers(ctx) if len(batches) > 1 else 1
    by_enc: dict = {}
    for encs in EA.fork_map(_recode_batch, batches, workers, weight=_batch_bytes, label="inventory"):
        for enc in encs:
            if enc is None:
                continue
            by_enc[enc] = by_enc.get(enc, 0) + 1
            ctx.count("files:recoded")
            ctx.count("files:recoded:" + enc)
    if by_enc:
        total = sum(by_enc.values())
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_enc.items()))
        ctx.note(f"text files not in UTF-8 re-encoded to UTF-8: {total} ({parts})")


def _out_dir(path: Path) -> Path:
    """``<archive>.files``; if such a directory is already in the input — a free name with a
    suffix (``<archive>.files-2``, ``-3`` …), so the unpacked files do not mix with other content."""
    base = path.with_name(path.name + OUT_SUFFIX)
    if not base.exists():
        return base
    n = 2
    while True:
        cand = path.with_name(f"{path.name}{OUT_SUFFIX}-{n}")
        if not cand.exists():
            return cand
        n += 1


# Skipped members that must never be extracted: links, devices, leaving the directory.
SECURITY_SKIPS = frozenset(("символическая ссылка", "ссылка", "не файл", "опасный путь"))
ARCHIVE_MARK = "unprocessed:archive"


def _keep_archive(ctx, arc: Path, why: str, counter: str) -> None:
    """The archive is not unpacked (or only partly): it stays as is with the "not anonymized" mark."""
    ctx.mark(arc, ARCHIVE_MARK, f"not anonymized: {why}"[:300])
    ctx.count(counter)


def unpack_tree(ctx, root: Path, depth: int, budget: _Budget | None) -> None:
    """Archives under ``root`` (a directory already unpacked from an archive of depth depth-1): each
    is unpacked into ``<archive>.files/`` and goes away; its content — recursively down to MAX_DEPTH.
    An archive not unpacked (or only partly unpacked) stays with a mark."""
    archives = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()
                      and is_archive(p.name))
    for arc in archives:
        if not arc.exists():
            continue
        rel = ctx.rel(arc)
        if archive_format(arc.name) is None:
            _keep_archive(ctx, arc, "not unpacked: no unpacker for this format", "archives:unsupported")
            continue
        if depth > MAX_DEPTH:
            _keep_archive(ctx, arc, f"not unpacked: nested deeper than {MAX_DEPTH}",
                          "archives:too_deep")
            continue
        dest = _out_dir(arc)
        b = budget if budget is not None else _Budget(MAX_TOTAL_BYTES, arc.stat().st_size, dest)
        try:
            skipped = extract_archive(arc, dest, b)
        except UnpackError as e:
            _keep_archive(ctx, arc, f"not unpacked: {e}", "archives:failed")
            continue
        lost = []
        for name, why in skipped:
            if why not in SECURITY_SKIPS:
                lost.append(why)         # the content was not extracted — it stays in the archive itself
                continue
            # the member is not on disk (the path could lead outside) — record it in the log directly:
            # "archive!member", as in the main pipeline's log database
            ctx.removals.append(Removal(f"{rel}!{name}", "archive_member",
                                        f"не извлечён: {why}", False))
            ctx.count("removed:archive_member")
        ctx.count("archives:unpacked")
        if lost:
            _keep_archive(ctx, arc, f"partly unpacked: members not extracted — {len(lost)} "
                                    f"({', '.join(sorted(set(lost)))}); the archive is kept next "
                                    "to the unpacked files", "archives:partial")
        else:
            arc.unlink()
        unpack_tree(ctx, dest, depth + 1, b)


def run(ctx) -> None:
    unpack_tree(ctx, ctx.work, 1, None)
    recode_texts(ctx)
    inv = []
    for p in ctx.files():
        if p.is_symlink():
            ctx.remove(p, "link", "symbolic link in the working copy")
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        k = kind_of(p.name)
        inv.append({"rel": ctx.rel(p), "kind": k, "size": size})
        ctx.count("files:" + k)
    ctx.data["inventory"] = inv
    ctx.count("files:total", len(inv))
    from .gates_struct import mark_input_broken     # here: gates_struct imports this module
    mark_input_broken(ctx)
