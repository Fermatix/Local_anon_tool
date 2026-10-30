"""PDF: document view and rebuild from text only.

The PDF worker of the engine's source pipeline, moved in-process, without a separate venv and
worker: PyMuPDF is imported lazily (extra ``pdf``); without it ``available()`` is False and the
documents stage keeps the PDF with a mark.

* ``extract(path)`` — layout and view: page lines (a line is one unit) and form fields in place
  on the page among the lines (y, x), then author/text/subject of annotations, link URIs, bookmarks,
  Info metadata fields; a page without a text layer (scan) — tesseract OCR through MuPDF if a
  tessdata directory is found (``tessdata_dir``), otherwise the page counts as unread
  (``ocr_missing_pages``).
* ``extract_view(path)`` — view lines only.
* ``rebuild(path, lines)`` — a NEW PDF in place from text only: pages of the same size, lines
  written with the built-in PyMuPDF font with Cyrillic (Droid Sans Fallback, subset), links,
  annotations, bookmarks and metadata — from the view; nothing is copied from the original — no
  images, attachments, XMP or fonts. ``extract_view`` of the rebuilt file gives the same lines.
* ``strip_band_images(path)`` — header/footer logos of a kept PDF: a picture (same pixels) placed
  in the top or bottom band of most pages is redacted by its bbox (pixels only: text and line art
  stay); the text of every page must stay the same, otherwise the file is left as is.
"""
from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

META_KEYS = ("title", "author", "subject", "keywords", "creator", "producer")
MIN_CHARS = 20                   # fewer letters/digits on the pages — "almost no text"
OCR_LANG = "rus+eng"
OCR_DPI = 200
FONT_NAME = "cjk"                # built-in MuPDF font with Cyrillic (Droid Sans Fallback)
_OCR_PAGE_MIN_CHARS = 5          # fewer letters/digits on a page — no text layer (scan)
_MIN_FONT = 3.0
BAND = 0.2                       # top/bottom share of the page height for header/footer logos
_TESSDATA_CANDIDATES = (
    "/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tesseract-ocr/4.00/tessdata",
    "/usr/share/tessdata", "/usr/local/share/tessdata", "/opt/homebrew/share/tessdata",
)


class PdfError(Exception):
    """Document not processed: does not open, password-protected, not a PDF, rebuild failed."""


def _fitz():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        try:
            import fitz
            return fitz
        except ImportError:
            return None


def available() -> bool:
    """PyMuPDF is installed."""
    return _fitz() is not None


def tessdata_dir() -> str:
    """tessdata directory for OCR: TESSDATA_PREFIX, standard Linux/macOS paths, Windows
    (Program Files\\Tesseract-OCR\\tessdata). Empty — OCR is unavailable."""
    cands = [os.environ.get("TESSDATA_PREFIX", "")]
    if os.name == "nt":
        for key in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(key, "")
            if base:
                cands.append(str(Path(base) / "Tesseract-OCR" / "tessdata"))
                cands.append(str(Path(base) / "Programs" / "Tesseract-OCR" / "tessdata"))
    cands.extend(_TESSDATA_CANDIDATES)
    for c in cands:
        if c and Path(c).is_dir() and any(Path(c).glob("*.traineddata")):
            return c
    return ""


def ocr_available() -> bool:
    return available() and bool(tessdata_dir())


def one_line(s) -> str:
    """A text unit is one view line: newlines and NUL → space."""
    if s is None:
        return ""
    s = str(s)
    return s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\x00", " ")


def _letters(s: str) -> int:
    return sum(1 for ch in s if ch.isalnum())


@dataclass
class Extraction:
    lines: list[str] = field(default_factory=list)      # view: line i ↔ items[i]
    items: list[dict] = field(default_factory=list)     # layout of the units (no text)
    pages: list[dict] = field(default_factory=list)     # {w, h, rot}
    goto: list[dict] = field(default_factory=list)      # internal links
    chars: int = 0                                      # letters/digits on the pages
    ocr_pages: int = 0                                  # pages recognized
    ocr_missing_pages: int = 0                          # scans without OCR (no tessdata / failure)
    ocr_why: str = ""
    images: int = 0
    attachments: int = 0
    xmp: bool = False

    @property
    def text_poor(self) -> bool:
        return self.chars < MIN_CHARS


# ── extraction ───────────────────────────────────────────────────────────────

def _page_lines(fz, page, textpage=None) -> list[dict]:
    """Page lines from get_text("dict") in the coordinates of the rotated page (page.rect)."""
    m = page.rotation_matrix
    d = page.get_text("dict", textpage=textpage) if textpage is not None else page.get_text("dict")
    out = []
    for b in d.get("blocks", ()):
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", ()):
            spans = ln.get("spans") or []
            text = one_line("".join(s.get("text", "") for s in spans))
            if not text.strip():
                continue
            size = max((float(s.get("size") or 0) for s in spans), default=0.0) or 10.0
            ox, oy = spans[0].get("origin") or (ln["bbox"][0], ln["bbox"][3])
            pt = fz.Point(ox, oy) * m
            r = fz.Rect(ln["bbox"]) * m
            out.append({"x": round(pt.x, 2), "y": round(pt.y, 2), "x1": round(r.x1, 2),
                        "s": round(size, 2), "t": text})
    return out


def _field_baseline(r) -> tuple[float, float]:
    """(y, x) of the line the field becomes on rebuild (render: text at the lower third of the box)."""
    h = max(float(r[3]) - float(r[1]), 4.0)
    return round(float(r[1]) + h * 0.75, 2), round(float(r[0]) + 1, 2)


def _place_fields(page_items: list[dict], fields: list[dict]) -> list[dict]:
    """Insert form fields among the page lines in place (y, x): before the first line that is
    lower/further right than the field; the order of the lines themselves (stream order) stays."""
    if not fields:
        return page_items
    out = list(page_items)
    for f in sorted(fields, key=lambda f: _field_baseline(f["r"])):
        fy, fx = _field_baseline(f["r"])
        idx = len(out)
        for i, it in enumerate(out):
            if (float(it["y"]), float(it["x"])) > (fy, fx) if it["k"] == "line" else \
                    _field_baseline(it["r"]) > (fy, fx):
                idx = i
                break
        out.insert(idx, f)
    return out


def _page_has_graphics(page) -> bool:
    try:
        if page.get_images(full=True):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return bool(page.get_drawings())
    except Exception:  # noqa: BLE001
        return False


def _open(fz, path):
    try:
        doc = fz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        raise PdfError(f"does not open: {exc.__class__.__name__}")
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise PdfError("password-protected (non-empty password)")
        if not doc.is_pdf:
            raise PdfError("not a PDF")
    except PdfError:
        doc.close()
        raise
    return doc


# The same unchanged file is read by the documents stage, the name layer, the masking and the
# rebuild — with OCR of scans each time. The memo keeps the result by (path, size, mtime) within the
# process; a rewritten file has another key. Callers get a copy, the memo stays as read. Across
# processes and past the memo size the run's view cache (viewcache, by the file bytes) answers.
_MEMO: OrderedDict = OrderedDict()
_MEMO_MAX = 64


def _memo_key(path, ocr: bool):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_size, st.st_mtime_ns, bool(ocr))


def extract(path, ocr: bool = True) -> Extraction:
    """Layout and view of a document. Does not open / password-protected / not a PDF — PdfError.
    An unchanged file is read once per process (memo by path, size and mtime) and once per run
    (the view cache by the bytes, when the run set it)."""
    key = _memo_key(path, ocr)
    if key is not None and key in _MEMO:
        _MEMO.move_to_end(key)
        return copy.deepcopy(_MEMO[key])
    ex = _extract_cached(path, ocr)
    if key is not None:
        _MEMO[key] = copy.deepcopy(ex)
        while len(_MEMO) > _MEMO_MAX:
            _MEMO.popitem(last=False)
    return ex


def _extract_cached(path, ocr: bool) -> Extraction:
    from . import viewcache
    if viewcache.directory() is None:
        return _extract(path, ocr)
    try:
        data = Path(path).read_bytes()
    except OSError:
        return _extract(path, ocr)                  # the read error comes from _extract
    tess = tessdata_dir() if ocr else ""
    k = viewcache.key("pdf", data, bool(ocr), tess)
    got = viewcache.get(k)
    if isinstance(got, Extraction):
        return got
    digest = hashlib.sha256(data).digest()
    ex = _extract(path, ocr, page_key=lambda pno: _page_key(digest, pno, tess))
    viewcache.put(k, ex)
    return ex


def _page_key(digest: bytes, pno: int, tess: str) -> str:
    """View cache key of the OCR lines of one page: the file bytes (``digest``), the page, the OCR settings."""
    from . import viewcache
    return viewcache.key("pdf-ocr-page", digest, pno, OCR_LANG, OCR_DPI, tess)


def _needs_ocr(page, lines: list[dict]) -> bool:
    """A page without a text layer (fewer letters than _OCR_PAGE_MIN_CHARS) and with graphics — a scan."""
    return sum(_letters(ln["t"]) for ln in lines) < _OCR_PAGE_MIN_CHARS and _page_has_graphics(page)


def _ocr_lines(fz, page, tess: str) -> list[dict]:
    tp = page.get_textpage_ocr(language=OCR_LANG, dpi=OCR_DPI, full=True, tessdata=tess)
    try:
        return _page_lines(fz, page, textpage=tp)
    finally:
        del tp


def page_count(path) -> int:
    """Number of pages; 0 — not read (no PyMuPDF, does not open, not a PDF)."""
    fz = _fitz()
    if fz is None:
        return 0
    try:
        doc = _open(fz, path)
    except PdfError:
        return 0
    try:
        return doc.page_count
    finally:
        doc.close()


def ocr_page(path, pno: int) -> bool:
    """OCR of one scanned page into the run's view cache under the key the full read looks up
    (``extract``): a long scan is recognized page by page in several worker processes. True — the page
    is in the cache (recognized now or before); False — no cache, no OCR, the page has a text layer or
    is not read (the full read decides then)."""
    from . import viewcache
    tess = tessdata_dir()
    fz = _fitz()
    if viewcache.directory() is None or not tess or fz is None:
        return False
    try:
        data = Path(path).read_bytes()
    except OSError:
        return False
    k = _page_key(hashlib.sha256(data).digest(), pno, tess)
    if isinstance(viewcache.get(k), list):
        return True
    try:
        doc = _open(fz, path)
    except PdfError:
        return False
    try:
        if not 0 <= pno < doc.page_count:
            return False
        page = doc[pno]
        if not _needs_ocr(page, _page_lines(fz, page)):
            return False
        lines = _ocr_lines(fz, page, tess)
    except Exception:  # noqa: BLE001 — the full read meets the same failure and records it
        return False
    finally:
        doc.close()
    viewcache.put(k, lines)
    return True


def _extract(path, ocr: bool = True, page_key=None) -> Extraction:
    from . import viewcache
    fz = _fitz()
    if fz is None:
        raise PdfError("PyMuPDF is not installed")
    doc = _open(fz, path)
    ex = Extraction()
    try:
        items: list[dict] = []
        tess = tessdata_dir() if ocr else ""
        for pno, page in enumerate(doc):
            ex.pages.append({"w": round(page.rect.width, 2), "h": round(page.rect.height, 2),
                             "rot": page.rotation})
            try:
                ex.images += len(page.get_images(full=True))
            except Exception:  # noqa: BLE001
                pass
            lines = _page_lines(fz, page)
            if ocr and _needs_ocr(page, lines):
                if not tess:
                    ex.ocr_why = ex.ocr_why or "tessdata not found (tesseract, TESSDATA_PREFIX)"
                    ex.ocr_missing_pages += 1
                else:
                    # a page recognized ahead in another process (ocr_page) comes from the view cache
                    k = page_key(pno) if page_key is not None else None
                    ocr_lines = viewcache.get(k) if k is not None else None
                    if not isinstance(ocr_lines, list):
                        try:
                            ocr_lines = _ocr_lines(fz, page, tess)
                        except Exception as exc:  # noqa: BLE001
                            ex.ocr_why = ex.ocr_why or f"OCR: {exc.__class__.__name__}"
                            ex.ocr_missing_pages += 1
                            ocr_lines = []
                        else:
                            if k is not None:
                                viewcache.put(k, ocr_lines)
                    if ocr_lines:
                        lines = lines + ocr_lines
                        ex.ocr_pages += 1
            page_items = [{"k": "line", "p": pno, "x": ln["x"], "y": ln["y"], "x1": ln["x1"],
                           "s": ln["s"], "t": ln["t"]} for ln in lines]
            # form fields — in place on the page among the lines (y, x), before annotations and
            # links: after rebuild the field becomes a text line in the same place, so the view stays
            fields = []
            for w in page.widgets():
                r = [round(v, 2) for v in w.rect]
                name, val = one_line(w.field_name), one_line(w.field_value)
                t = f"{name}: {val}" if name and val else (name or val)
                if t.strip():
                    fields.append({"k": "field", "p": pno, "r": r, "t": t})
            items.extend(_place_fields(page_items, fields))
            for ai, a in enumerate(page.annots()):
                info = a.info or {}
                r = [round(v, 2) for v in a.rect]
                for f in ("title", "content", "subject"):
                    t = one_line(info.get(f))
                    if t.strip():
                        items.append({"k": "annot", "p": pno, "r": r, "g": ai, "f": f, "t": t})
            for lk in page.get_links():
                r = [round(v, 2) for v in lk["from"]]
                uri = one_line(lk.get("uri") or lk.get("file"))
                if uri.strip():
                    items.append({"k": "link", "p": pno, "r": r, "t": uri})
                elif lk.get("kind") == fz.LINK_GOTO and lk.get("page") is not None:
                    ex.goto.append({"p": pno, "r": r, "to": int(lk["page"])})
        for lvl, title, pg in doc.get_toc(simple=True):
            t = one_line(title)
            if t.strip():
                items.append({"k": "toc", "lvl": int(lvl), "p": int(pg), "t": t})
        meta = doc.metadata or {}
        for key in META_KEYS:
            t = one_line(meta.get(key))
            if t.strip():
                items.append({"k": "meta", "key": key, "t": t})
        try:
            ex.xmp = bool(doc.get_xml_metadata())
        except Exception:  # noqa: BLE001
            ex.xmp = False
        try:
            ex.attachments = doc.embfile_count()
        except Exception:  # noqa: BLE001
            ex.attachments = 0
        ex.chars = sum(_letters(it["t"]) for it in items if it["k"] == "line")
        ex.lines = [it["t"] for it in items]
        ex.items = [{k: v for k, v in it.items() if k != "t"} for it in items]
        return ex
    finally:
        doc.close()


def extract_view(path) -> list[str]:
    """Text view of a PDF (page lines, links, metadata); a scan — OCR if tesseract is present."""
    return extract(path).lines


# ── rebuild ──────────────────────────────────────────────────────────────────

def _fit_size(font, text: str, size: float, width: float) -> float:
    if width <= 0 or not text:
        return max(size, _MIN_FONT)
    w = font.text_length(text, fontsize=size)
    if w > width * 1.02 and w > 0:
        size = size * width / w
    return max(size, _MIN_FONT)


def render(layout: Extraction, lines: list[str], out_path) -> None:
    """A new PDF from the layout and view lines (len(lines) == len(layout.items))."""
    fz = _fitz()
    if fz is None:
        raise PdfError("PyMuPDF is not installed")
    items = layout.items
    if len(lines) != len(items):
        raise PdfError(f"number of view lines ({len(lines)}) differs from the number of layout units ({len(items)})")
    font = fz.Font(FONT_NAME)
    out = fz.open()
    for pg in layout.pages:
        out.new_page(width=float(pg["w"]), height=float(pg["h"]))
    if out.page_count == 0:
        out.new_page()
    npages = list(range(out.page_count))
    writers = [fz.TextWriter(out[i].rect) for i in npages]
    annots: dict = {}
    toc = []
    meta = {k: "" for k in META_KEYS}
    for it, t in zip(items, lines):
        t = one_line(t)
        k = it["k"]
        if k == "line":
            p = min(max(int(it["p"]), 0), len(npages) - 1)
            size = _fit_size(font, t, float(it.get("s") or 10.0),
                             float(it.get("x1", 0)) - float(it["x"]))
            if t.strip():
                writers[p].append((float(it["x"]), float(it["y"])), t, font=font, fontsize=size)
        elif k == "field":
            p = min(max(int(it["p"]), 0), len(npages) - 1)
            r = it["r"]
            h = max(float(r[3]) - float(r[1]), 4.0)
            size = _fit_size(font, t, min(10.0, h * 0.8), float(r[2]) - float(r[0]) - 2)
            if t.strip():
                writers[p].append((float(r[0]) + 1, float(r[1]) + h * 0.75), t, font=font,
                                  fontsize=size)
        elif k == "annot":
            annots.setdefault((int(it["p"]), int(it["g"])), {"r": it["r"]})[it["f"]] = t
        elif k == "link":
            p = min(max(int(it["p"]), 0), len(npages) - 1)
            if t.strip():
                out[p].insert_link({"kind": fz.LINK_URI, "from": fz.Rect(it["r"]), "uri": t})
        elif k == "toc":
            toc.append([int(it["lvl"]), t, min(max(int(it["p"]), 1), len(npages))])
        elif k == "meta":
            meta[it["key"]] = t
    for p, tw in zip(npages, writers):
        try:
            tw.write_text(out[p])
        except Exception as exc:  # noqa: BLE001
            raise PdfError(f"could not write the text: {exc.__class__.__name__}")
    for (pno, _g), a in sorted(annots.items()):
        p = min(max(pno, 0), len(npages) - 1)
        r = a["r"]
        an = out[p].add_text_annot(fz.Point(float(r[0]), float(r[1])), a.get("content", ""))
        an.set_info(title=a.get("title", ""), content=a.get("content", ""),
                    subject=a.get("subject", ""))
        an.update()
    for g in layout.goto:
        p = min(max(int(g["p"]), 0), len(npages) - 1)
        to = min(max(int(g.get("to", 0)), 0), len(npages) - 1)
        try:
            out[p].insert_link({"kind": fz.LINK_GOTO, "from": fz.Rect(g["r"]), "page": to})
        except Exception:  # noqa: BLE001
            pass
    if toc:
        fixed, prev = [], 0
        for lvl, t, pg in toc:
            lvl = max(1, min(lvl, prev + 1))
            fixed.append([lvl, t, pg])
            prev = lvl
        try:
            out.set_toc(fixed)
        except Exception as exc:  # noqa: BLE001
            raise PdfError(f"bookmarks: {exc.__class__.__name__}")
    meta["creationDate"] = ""
    meta["modDate"] = ""
    out.set_metadata(meta)
    try:
        out.subset_fonts()
    except Exception:  # noqa: BLE001 — the font is embedded whole
        pass
    out.save(str(out_path), garbage=4, deflate=True, clean=True, no_new_id=True)
    out.close()


def verify_clean(path) -> list[str]:
    """Problems of the rebuilt PDF: images, attachments, XMP. Empty — clean."""
    fz = _fitz()
    chk = fz.open(str(path))
    try:
        problems = []
        n_img = sum(len(p.get_images(full=True)) for p in chk)
        if n_img:
            problems.append(f"images: {n_img}")
        if chk.embfile_count():
            problems.append(f"attachments: {chk.embfile_count()}")
        if chk.get_xml_metadata():
            problems.append("XMP metadata")
        return problems
    finally:
        chk.close()


def rebuild(path, lines: list[str], layout: Extraction | None = None) -> None:
    """Rebuild a PDF in place from the view text only (without images, fonts and attachments of
    the original). layout — the layout from extract(path) if already there (otherwise read again)."""
    p = Path(path)
    if layout is None:
        layout = extract(p)
    fd, tmp = tempfile.mkstemp(prefix=".pdf.", suffix=".tmp", dir=str(p.parent))
    os.close(fd)
    try:
        render(layout, list(lines), tmp)
        problems = verify_clean(tmp)
        if problems:
            raise PdfError("left in the rebuilt file: " + ", ".join(problems))
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _band_hits(doc, band: float) -> dict:
    """{image digest: [(page number, bbox)]} for pictures lying fully in the top or bottom band."""
    hits: dict = {}
    for page in doc:
        h = float(page.rect.height)
        for info in page.get_image_info(hashes=True):
            bbox = info.get("bbox")
            digest = info.get("digest")
            if not bbox or not digest:
                continue
            y0, y1 = float(bbox[1]), float(bbox[3])
            if y1 <= h * band or y0 >= h * (1.0 - band):
                hits.setdefault(bytes(digest), []).append((page.number, tuple(bbox)))
    return hits


def strip_band_images(path, band: float = BAND) -> int:
    """Redact pictures repeated in the header/footer band of most pages (more than half, at least
    two pages); → how many placements were redacted. The text of each page is compared before and
    after: a difference — PdfError, file unchanged."""
    fz = _fitz()
    if fz is None:
        raise PdfError("PyMuPDF is not installed")
    p = Path(path)
    doc = _open(fz, p)
    try:
        n_pages = doc.page_count
        if n_pages < 2:
            return 0
        targets: dict = {}
        for digest, places in _band_hits(doc, band).items():
            pages = {pno for pno, _ in places}
            if len(pages) >= 2 and len(pages) * 2 > n_pages:
                for pno, bbox in places:
                    targets.setdefault(pno, []).append(bbox)
        if not targets:
            return 0
        before = [page.get_text("text") for page in doc]
        count = 0
        for pno, boxes in sorted(targets.items()):
            page = doc[pno]
            for bbox in boxes:
                page.add_redact_annot(fz.Rect(bbox))
                count += 1
            try:
                page.apply_redactions(images=fz.PDF_REDACT_IMAGE_PIXELS,
                                      graphics=fz.PDF_REDACT_LINE_ART_NONE,
                                      text=fz.PDF_REDACT_TEXT_NONE)
            except Exception as exc:  # noqa: BLE001 — any MuPDF failure leaves the file as is
                raise PdfError(f"redaction failed: {exc.__class__.__name__}")
        fd, tmp = tempfile.mkstemp(prefix=".pdf.", suffix=".tmp", dir=str(p.parent))
        os.close(fd)
        try:
            doc.save(tmp, garbage=3, deflate=True)
            chk = fz.open(tmp)
            try:
                after = [page.get_text("text") for page in chk]
            finally:
                chk.close()
            if after != before:
                raise PdfError("page text changed after redaction")
            doc.close()
            doc = None
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return count
    finally:
        if doc is not None:
            doc.close()
