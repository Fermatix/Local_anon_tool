"""Stage "documents".

Removed are the document kinds of ``[documents].delete_kinds`` (by default the owner's list of
27.10 evening, see below); everything else stays and is anonymized by the anonymize stage, the
main pipeline finishes what is uncertain.

For each document (docx, doc, odt, rtf, xlsx, xls, ods, pptx, ppt, odp, OOXML templates/add-ins/
slides/Visio, Word 2003 XML, pdf, txt, md, html):

1. text — ``office.view_lines`` / ``pdf.extract``; old formats (doc, xls, ppt, rtf, odt, ods,
   odp, Word 2003 XML) and OOXML extensions with foreign content go through LibreOffice with an
   explicit import filter by content (``office.import_filter``), if it is found
   (``office.soffice_bin``); text (CSV, TSV, an HTML "to Excel" export) under .xls/.doc is handled
   as a text file; Excel 2003 XML and flat ODF under .xml — by node text; .xml is a document only by
   ``office.xml2003_kind`` or a flat ODF root, parts of an unpacked OOXML are not touched;
2. the "papers" classifier (``docclass``): a kind from ``[documents].delete_kinds`` →
   ``ctx.remove(p, 'document:<kind>', '<features>', borderline)``; a doubtful match with
   ``borderline = "keep"`` and any other kind (waybill, claim, guarantee letter…) stay —
   ``ctx.mark(p, 'document:<kind>', …)``: the kind is in the report;
   paths from ``[documents].restore`` are not classified;
3. the rest of the content is not changed: body pictures and attachments stay, the PDF is not
   rebuilt here (text anonymization is the anonymize stage); only header pictures go (below).

Owner's decision of 27.10 (evening, DESIGN.md §9b): the list of the task is back — contract,
supplementary agreement, power of attorney, act, invoice and UPD are removed
(``[documents].delete_kinds``), a doubtful match is removed flagged (``borderline = "delete"``).
Pictures of headers, footers and slide masters of a kept OOXML document (executor logos) are cut
(``office.strip_hf_images``; header text and body pictures stay, the document is checked); kept
PDFs lose pictures repeated in the top or bottom band of most pages (``pdf.strip_band_images``);
both are switched by ``[documents].header_images`` (``delete`` | ``keep``). Counters: ``documents:header_pictures``,
``documents:header_pictures_docs``; a failure leaves the file as is, marked ``image:header``.

A document not read for any reason (broken, unrecognized, no LibreOffice, a scan without OCR,
no PyMuPDF, a PDF error) is not removed: it stays as it was with the mark
``ctx.mark(p, 'unprocessed', 'not anonymized: <reason>')`` — the path and reason go to the data
owner's report, the control gate does not fail on such a file. An old format is also marked after
classification: anonymize reads the document view of OOXML only. ``ctx.count`` holds counters only.

For txt/md/html a confident removal needs a hint in the file name (a title in text is weaker than
in a document; this does not apply to text under .xls/.doc — that is an office-format document),
and HTML chat exports (Telegram, Discord) are not classified at all —
that is correspondence, handled by the messages stage.

Documents go to worker processes, one per CPU slot of the hw plan (stage ``documents``), the heavy
ones first; what a worker does to ctx comes back as records and is applied in file order, so the
removals, marks, counters and notes are those of one process. The views it parses (OCR included)
stay in the run's view cache (``viewcache``) for the later stages.
"""
from __future__ import annotations

import codecs
import fnmatch
import functools
import html as htmlmod
import re
from pathlib import Path

from .. import docclass, office, pdf
from ..context import UNPROCESSED, category_key

NAME = "documents"

TEXT_EXT = frozenset((".txt", ".md", ".markdown", ".html", ".htm", ".xhtml"))
HTML_EXT = frozenset((".html", ".htm", ".xhtml"))
PDF_EXT = frozenset((".pdf",))

_HTML_TAG_RX = re.compile(r"<(?!%)[^>]+>")              # <%= … %> (ERB/JSP) is not a tag
_CHAT_HTML_RX = re.compile(r'class="(message|chatlog|page_header|chatlog__message)', re.I)


_CHARSET_RX = re.compile(rb"charset\s*=\s*[\"']?([A-Za-z0-9_.:-]+)", re.I)
# HTML — by the decoded text (UTF-16 is not visible in raw bytes)
_HTML_DOC_RX = re.compile(r"<(!doctype\s+html|html|head|body|table)[\s>]", re.I)


def _utf16_nobom(raw: bytes) -> str | None:
    """UTF-16 codec without a BOM (Office "Web Page"/"Unicode Text"): there are zero bytes, and the
    head reads as UTF-16 almost entirely as Latin, Cyrillic and punctuation; otherwise None."""
    head = raw[:4096]
    head = head[:len(head) // 2 * 2]
    if len(head) < 4 or b"\x00" not in head:
        return None
    for enc in ("utf-16-le", "utf-16-be"):
        try:
            s = head.decode(enc)
        except UnicodeDecodeError:
            continue
        ok = sum(1 for ch in s if ch in "\t\n\r\x0c" or " " <= ch <= "\u04ff" or "\u2000" <= ch <= "\u206f"
                 or ch in "№₽€")
        if ok * 100 >= len(s) * 98:
            return enc
    return None


def _decode_text(raw: bytes) -> tuple[str, str, bytes]:
    """(text, codec, BOM) of a text file: BOM → UTF-8/UTF-16; otherwise UTF-8; if that fails — the
    declared charset (<meta charset=…>), otherwise cp1251. The file is written back with the same
    codec and BOM, so the charset declaration and readers match the bytes."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "replace"), "utf-8", b"\xef\xbb\xbf"
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", "replace"), "utf-16-le", b"\xff\xfe"
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", "replace"), "utf-16-be", b"\xfe\xff"
    enc16 = _utf16_nobom(raw)
    if enc16:
        return raw.decode(enc16, "replace"), enc16, b""
    try:
        return raw.decode("utf-8"), "utf-8", b""
    except UnicodeDecodeError:
        pass
    m = _CHARSET_RX.search(raw[:4096])
    enc = m.group(1).decode("ascii", "replace").lower() if m else "cp1251"
    try:
        codecs.lookup(enc)
    except LookupError:
        enc = "cp1251"
    if enc.replace("-", "").replace("_", "") in ("utf8", "utf16", "utf16le", "utf16be", "utf32"):
        enc = "cp1251"                                       # declared UTF, bytes are not UTF: as is
    return raw.decode(enc, "replace"), enc, b""


_HTML_CODE_RX = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)
_HTML_BREAK_RX = re.compile(r"</(?:tr|p|div|li|h[1-6])\s*>|<br\b[^>]*>", re.I)
_HTML_CELL_RX = re.compile(r"</t[dh]\s*>", re.I)


def _html_text(s: str) -> str:
    """HTML text: a table row, paragraph, <br>, block, list item, heading start a new line, a cell
    is separated by a tab (a title like «Счёт на оплату № …» stays on its own line). ERB/JSP template
    inserts ``<%= … %>`` are not tags: they stay in the text, the classifier treats them as
    placeholders."""
    s = _HTML_CELL_RX.sub("\t", _HTML_BREAK_RX.sub("\n", _HTML_CODE_RX.sub(" ", s)))
    return htmlmod.unescape(_HTML_TAG_RX.sub(" ", s))


_MIME_RX = re.compile(rb"^\s*(MIME-Version\s*:|Content-Type\s*:\s*multipart/)", re.I | re.M)


def _looks_text(raw: bytes) -> bool:
    """Text (CSV, TSV, HTML) under an old Office extension: UTF-16 with or without a BOM, or no zero
    bytes and almost no control bytes in the head; MHT (MIME, pictures inside base64) is not text."""
    head = raw[:4096]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")) or _utf16_nobom(head):
        return True
    if b"\x00" in head or _MIME_RX.search(head):
        return False
    ctrl = sum(1 for b in head if b < 0x20 and b not in b"\t\n\r\x0c")
    return ctrl * 100 <= len(head)


_XML_BREAK_RX = re.compile(r"</(?:[\w.-]+:)?(?:p|h|Cell|table-cell|Row|table-row|title)>"
                           r"|<(?:[\w.-]+:)?line-break\b[^>]*>", re.I)
_XML_SPACE_RX = re.compile(r"<(?:[\w.-]+:)?(?:s|tab)\b[^>]*/>")
_XML_SKIP_RX = re.compile(r"<!--.*?-->|<\?.*?\?>|<!DOCTYPE[^>]*>", re.S)


def _xml_text(xml: str) -> str:
    """Text of Excel 2003 XML / flat ODF: a line per paragraph, cell, heading; no markup."""
    s = _XML_BREAK_RX.sub("\n", _XML_SKIP_RX.sub("", xml))
    s = htmlmod.unescape(re.sub(r"<[^>]+>", "", _XML_SPACE_RX.sub(" ", s)))
    return "\n".join(x.strip() for x in s.splitlines() if x.strip())


def _restore_patterns(cfg) -> list[str]:
    out = []
    for x in cfg.restore or []:
        x = str(x).replace("\\", "/").strip().lstrip("./")
        if x:
            out.append(x)
    return out


def _is_restored(rel: str, patterns: list[str]) -> bool:
    low = rel.lower()
    for pat in patterns:
        if rel == pat or low == pat.lower():
            return True
        if pat.endswith("/") and low.startswith(pat.lower()):
            return True
        if fnmatch.fnmatchcase(rel, pat) or fnmatch.fnmatchcase(low, pat.lower()):
            return True
    return False


class _Stage:
    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg = ctx.config.documents
        self.delete_kinds = set(self.cfg.delete_kinds or [])
        self.restore = _restore_patterns(self.cfg)
        self.soffice = office.soffice_bin()
        self.pdf_ok = pdf.available()
        self._noted: set[str] = set()

    def unprocessed(self, p: Path, reason: str) -> None:
        """The document was not read or cannot be anonymized: it stays as it was, a mark in the report."""
        self.ctx.mark(p, UNPROCESSED, "not anonymized: " + reason)
        self.ctx.count("documents:unprocessed")

    def note_once(self, key: str, msg: str) -> None:
        if key not in self._noted:
            self._noted.add(key)
            self.ctx.note("documents: " + msg)

    # --- classification -------------------------------------------------------------------------
    def classify_remove(self, p: Path, text: str, restored: bool, weak_title: bool = False) -> bool:
        """A kind from delete_kinds → remove (True); other kinds and a doubtful contract with
        borderline = "keep" stay with a kind mark in the report."""
        ctx = self.ctx
        if restored:
            ctx.count("documents:restored")
            return False
        d = docclass.classify_detail(text, p.name)
        if d.kind is None:
            return False
        borderline = d.borderline
        if weak_title and "filename" not in d.cues:
            borderline = True
        reason = ("doubtful, " if borderline else "") + f"score {d.score:.2f}: " + ", ".join(d.features[:6])
        if d.kind not in self.delete_kinds:
            ctx.count("documents:kind_not_deleted")
            ctx.mark(p, f"document:{d.kind}", "kept and anonymized, this kind is not removed; " + reason)
            return False
        if borderline and self.cfg.borderline == "keep":
            ctx.count("documents:borderline_kept")
            ctx.mark(p, f"document:{d.kind}", "check: looks like a removed kind, kept "
                                              "([documents].borderline = keep); " + reason)
            return False
        ctx.remove(p, f"document:{d.kind}", reason, borderline=borderline)
        return True

    # --- Office ------------------------------------------------------------------------------------
    def office(self, p: Path, kind: str, native: bool, restored: bool) -> None:
        ctx = self.ctx
        data = p.read_bytes()
        foreign = office.sniff_foreign(data)
        # non-OOXML content (OLE, RTF, HTML, ODF, Office 2003 XML) goes through LibreOffice there and
        # back; OOXML under any extension (.doc included) is cleaned directly
        legacy = foreign is not None
        if legacy and not native and foreign in ("", "html") and _looks_text(data):
            # a CSV/TSV/HTML "to Excel" export under .xls/.doc is text, not old Office
            ctx.count("documents:text_as_office")
            self.text(p, p.suffix.lower(), restored, office_ext=True)
            return
        if legacy:
            if foreign == "":
                self.unprocessed(p, "content not recognized (not OOXML, OLE, RTF, HTML, "
                                    "ODF or Office XML)")
                return
            if self.soffice is None:
                self.note_once("soffice", "LibreOffice not found: documents in old formats (doc, xls, ppt, "
                                          "rtf, odt, ods, odp, Office 2003 XML) are kept with a mark")
                self.unprocessed(p, "old format, cannot be read without LibreOffice")
                return
            # an explicit import filter by content: junk under an OLE header and a password-protected
            # document are not "converted" by a text filter but fail → a mark
            infilter = office.import_filter(foreign, kind)
            if infilter is None:
                self.unprocessed(p, f"LibreOffice does not read this content ({foreign}) as {kind}")
                return
            try:
                container = office.to_ooxml(p, kind, infilter)
            except office.OfficeError as exc:
                self.unprocessed(p, f"LibreOffice did not convert to {kind}: {exc}")
                return
            ctx.count("documents:legacy_converted")
        else:
            container = data
        try:
            lines = office.view_lines(container)
        except office.OfficeError as exc:
            self.unprocessed(p, f"not readable as OOXML: {exc}")
            return
        if self.classify_remove(p, "\n".join(lines), restored):
            return
        if legacy:
            # anonymize reads the document view of OOXML only: an old format stays as it was
            self.unprocessed(p, f"old format ({p.suffix}), text is not anonymized locally")
            return
        if not restored:
            self.header_pictures(p, office.strip_hf_images, office.OfficeError)

    def header_pictures(self, p: Path, strip, error) -> None:
        """Logos: pictures of headers, footers and slide masters go (§9b), the rest stays."""
        ctx = self.ctx
        if self.cfg.header_images != "delete":
            return
        try:
            n = strip(p)
        except error as exc:
            ctx.count("documents:header_pictures_failed")
            ctx.mark(p, "image:header", f"check: header pictures (logos) not removed — {exc}")
            return
        if n:
            ctx.count("documents:header_pictures", n)
            ctx.count("documents:header_pictures_docs")

    # --- PDF -----------------------------------------------------------------------------------------
    def pdf(self, p: Path, restored: bool) -> None:
        ctx = self.ctx
        if not self.pdf_ok:
            self.note_once("pymupdf", "PyMuPDF is not installed: all PDFs are kept as is with a mark "
                                      "(run.sh / run.ps1 installs it; by hand: "
                                      ".venv/bin/python -m pip install -e .)")
            self.unprocessed(p, "PyMuPDF is not installed — cannot read the PDF")
            return
        try:
            ex = pdf.extract(p)
        except pdf.PdfError as exc:
            self.unprocessed(p, f"PDF: {exc}")
            return
        if ex.ocr_pages:
            ctx.count("documents:ocr_pages", ex.ocr_pages)
        if ex.ocr_missing_pages:
            if ex.text_poor:
                self.note_once("ocr", "OCR unavailable (tesseract/tessdata not found): scans without a text "
                                      "layer are kept as is with a mark")
                self.unprocessed(p, "scan without a text layer, OCR unavailable (no tesseract): "
                                    + ex.ocr_why)
                return
            self.note_once("ocr_partial", "OCR unavailable: scanned pages in mixed PDFs are not anonymized")
            ctx.count("documents:pdf_pages_unread", ex.ocr_missing_pages)
            if not self.classify_remove(p, "\n".join(ex.lines), restored):
                self.unprocessed(p, f"scanned pages without OCR: {ex.ocr_missing_pages}")
            return
        if self.classify_remove(p, "\n".join(ex.lines), restored):
            return
        ctx.count("documents:pdf_kept")
        if not restored:
            self.header_pictures(p, pdf.strip_band_images, pdf.PdfError)

    # --- text ------------------------------------------------------------------------------------------
    def text(self, p: Path, ext: str, restored: bool, office_ext: bool = False) -> None:
        """txt/md/html; office_ext — text under .xls/.doc: the decoded text decides whether it is HTML."""
        raw = p.read_bytes()
        if b"\x00" in raw[:4096] and not raw.startswith((b"\xff\xfe", b"\xfe\xff")) and not _utf16_nobom(raw):
            return                                              # a binary under a text extension
        text = _decode_text(raw)[0]
        if office_ext:
            is_html = _HTML_DOC_RX.search(text[:4096]) is not None
        else:
            is_html = ext in HTML_EXT
        plain = _html_text(text) if is_html else text
        chat_export = is_html and len(_CHAT_HTML_RX.findall(text[:200000])) >= 3
        # under .xls/.doc it is an office-format document, not a note: a title in the text is a full hint
        if not chat_export:
            self.classify_remove(p, plain, restored, weak_title=not office_ext)

    # --- XML documents without a container (Excel 2003 XML, flat ODF) ----------------------------
    def flat_xml(self, p: Path, kind: str, restored: bool) -> None:
        """Excel 2003 XML and flat ODF: classification by node text, a "non-paper" stays as is
        (flat ODF embedded pictures too: we do not cut them)."""
        ctx = self.ctx
        try:
            xml = office._decode_part(p.read_bytes(), p.name)[0]
        except office.OfficeError as exc:
            self.unprocessed(p, f"XML not readable: {exc}")
            return
        if self.classify_remove(p, _xml_text(xml), restored):
            return
        ctx.count("documents:xml_kept")

    # --- walk ------------------------------------------------------------------------------------------
    @staticmethod
    def _office_xml_kind(p: Path) -> str | None:
        """.xml: Word/Excel 2003 XML (office.xml2003_kind) → family (docx|xlsx|pptx), flat ODF → "fodf",
        None — plain XML (parts of an unpacked OOXML included)."""
        with open(p, "rb") as fh:
            head = fh.read(4096)
        foreign = office.sniff_foreign(head)
        if foreign == "xml":
            return office.xml2003_kind(head)
        return "fodf" if foreign == "fodf" else None

    def jobs(self) -> list[tuple]:
        """Documents of the working copy in ``ctx.files()`` order: (path, kind_info, office_xml, restored)."""
        ctx = self.ctx
        out = []
        for p in ctx.files():
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            kind_info = office.kind_of(p)
            office_xml = False
            if kind_info is None and ext == ".xml":
                office_xml = self._office_xml_kind(p)
                if office_xml is None:
                    continue                                        # plain XML is not a document
            elif kind_info is None and ext not in TEXT_EXT and ext not in PDF_EXT:
                continue
            out.append((p, kind_info, office_xml, _is_restored(ctx.rel(p), self.restore)))
        return out

    def one(self, p: Path, kind_info, office_xml, restored: bool) -> None:
        ctx = self.ctx
        ctx.count("documents:seen")
        try:
            if office_xml:
                if office_xml in ("xlsx", "fodf"):
                    self.flat_xml(p, office_xml, restored)
                else:
                    self.office(p, office_xml, False, restored)
            elif kind_info is not None:
                self.office(p, kind_info[0], kind_info[1], restored)
            elif p.suffix.lower() in PDF_EXT:
                self.pdf(p, restored)
            else:
                self.text(p, p.suffix.lower(), restored)
        except Exception as exc:  # noqa: BLE001 — one document does not crash the run
            if p.exists() and not ctx.is_marked(p, UNPROCESSED):
                self.unprocessed(p, f"not processed: {type(exc).__name__}")
            ctx.count("documents:errors")

    def run(self) -> None:
        jobs = self.jobs()
        slots = _slots(self.ctx, len(jobs))
        if not slots or len(slots) < 2:
            for job in jobs:
                self.one(*job)
            return
        # documents in worker processes (one per CPU slot of the plan, the heavy ones first); the
        # effects on ctx come back as records and are applied here in file order — the same marks,
        # removals, counters and notes as in one process
        from .. import devpool
        ctx = self.ctx
        if self.pdf_ok:
            _ocr_ahead(ctx, [p for p, kind_info, office_xml, _r in jobs
                             if kind_info is None and not office_xml and p.suffix.lower() in PDF_EXT], slots)
        items = [(str(p), kind_info, office_xml, restored, ctx.is_marked(p, UNPROCESSED))
                 for p, kind_info, office_xml, restored in jobs]
        results = devpool.run(_worker_batch, items, slots, batch_size=1, weight=_size, ctx=ctx,
                              init=functools.partial(_worker_init, ctx.config, str(ctx.work)))
        for events in results:
            for ev in events:
                _replay(self, ev)


def _slots(ctx, n_docs: int):
    """CPU slots of the hw plan (stage ``documents``); fewer than two documents or no plan — none
    (the stage runs in its own process)."""
    if n_docs < 2:
        return None
    try:
        from .. import hw
        return hw.stage_plan(ctx, "documents", hw.Workload(n_documents=n_docs)).slots
    except Exception as e:  # noqa: BLE001 — the plan does not crash the stage: one process
        ctx.note(f"documents: hardware plan not built ({type(e).__name__}) — one process")
        return None


def _size(item) -> int:
    try:
        return Path(item[0]).stat().st_size
    except OSError:
        return 0


def _ocr_ahead(ctx, pdfs: list, slots) -> int:
    """Pages of multi-page PDFs are recognized ahead, one page per task over all the slots
    (``pdf.ocr_page`` → the run's view cache): a long scan does not keep one worker busy while the
    others idle — its read in the documents pool takes the pages from the cache. Without the view
    cache or OCR — nothing (each document does its OCR itself). → pages put in the cache."""
    from .. import devpool, viewcache
    if viewcache.directory() is None or not pdf.ocr_available():
        return 0
    pages = [(str(p), pno) for p in pdfs for n in (pdf.page_count(p),) if n > 1 for pno in range(n)]
    if len(pages) < 2:
        return 0
    return sum(1 for ok in devpool.run(_ocr_batch, pages, slots, batch_size=1, ctx=ctx) if ok)


def _ocr_batch(_state, batch: list) -> list:
    return [pdf.ocr_page(path, pno) for path, pno in batch]


class _Recorder:
    """RunContext of a worker process: what the stage does to ctx is recorded (``events``) and
    applied by the parent (``_replay``); files are not removed here — the parent does that."""

    def __init__(self, config, work: str):
        self.config = config
        self.work = Path(work)
        self.events: list = []
        self._marked: set = set()

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.work.resolve()).as_posix()

    def count(self, key: str, n: int = 1) -> None:
        self.events.append(("count", key, n))

    def note(self, msg: str) -> None:
        self.events.append(("note", msg))

    def mark(self, path: Path, category: str, reason: str) -> None:
        self.events.append(("mark", str(path), category, reason))
        self._marked.add(category_key(category))

    def remove(self, path: Path, category: str, reason: str, borderline: bool = False) -> None:
        self.events.append(("remove", str(path), category, reason, borderline))

    def is_marked(self, path: Path, category: str | None = None) -> bool:
        return bool(self._marked) if category is None else category in self._marked


class _WorkerStage(_Stage):
    def note_once(self, key: str, msg: str) -> None:
        self.ctx.events.append(("note_once", key, msg))     # once per run — decided by the parent


def _worker_init(config, work: str, _slot):
    return _WorkerStage(_Recorder(config, work))


def _worker_batch(stage: _WorkerStage, batch: list) -> list:
    out = []
    for path, kind_info, office_xml, restored, unprocessed in batch:
        rec = stage.ctx
        rec.events = []
        rec._marked = {UNPROCESSED} if unprocessed else set()
        stage.one(Path(path), tuple(kind_info) if kind_info else None, office_xml, restored)
        out.append(rec.events)
    return out


def _replay(stage: _Stage, ev: tuple) -> None:
    ctx = stage.ctx
    op = ev[0]
    if op == "count":
        ctx.count(ev[1], ev[2])
    elif op == "note":
        ctx.note(ev[1])
    elif op == "note_once":
        stage.note_once(ev[1], ev[2])
    elif op == "mark":
        ctx.mark(Path(ev[1]), ev[2], ev[3])
    elif op == "remove":
        ctx.remove(Path(ev[1]), ev[2], ev[3], ev[4])


def run(ctx) -> None:
    _Stage(ctx).run()
