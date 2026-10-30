"""Office documents: Word / Excel / PowerPoint in place.

The core of the main pipeline's ``tools/office.py`` without its pipeline, gates and pool:

* ``extract_view(path)`` — a text view of an OOXML container: one line per text unit
  (a Word paragraph with formatting runs glued, a sharedStrings string and an inline Excel string,
  a PowerPoint shape paragraph, headers and footers, footnotes, comments and their authors, docProps
  properties, external hyperlinks, sheet names, slide notes, alternative text…);
* ``apply_view(path, lines)`` — the changed view back: edits are spread over the source text
  nodes char by char (difflib), everything else stays byte for byte; an unchanged view leaves the
  document alone;
* ``strip_media(path)`` — cut out pictures and embedded objects: ``media/``, ``embeddings/``,
  ``activeX/`` parts, the thumbnail, embedded fonts, the digital signature
  (``_xmlsignatures/``: the signature picture and the signer certificate), any part referenced by
  a ``…/image``, ``…/oleObject``, ``…/package`` relationship, and any part with a picture
  signature (``sniff_image``) — whatever its name and directory; picture markup (``w:drawing``,
  ``w:pict`` with ``v:imagedata``, ``p:pic``, Excel drawing anchors, background, OLE, ``a:blip``),
  relationships to removed parts and references to those relationships, ``[Content_Types].xml``
  entries. The document stays valid: all XML parses, no references to relationships without a
  target (``check``).

XML is not reserialized by a parser (ElementTree loses namespace declarations referenced by
``mc:Ignorable`` — Word then sees the file as broken): parts are edited char by char by spans,
the rest is unchanged. Standard library only; LibreOffice is an external command for old
formats (``to_ooxml`` / ``from_ooxml``), if found (``soffice_bin``), with an explicit import filter
by content (``import_filter``): junk under an OLE header and a password-protected document are not
read by the "text" filter but fail. Word 2003 XML — ``xml2003_kind`` / ``XML2003_EXPORT``.
"""
from __future__ import annotations

import difflib
import io
import os
import posixpath
import re
import shutil
import signal
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

# extension → OOXML kind; native — the container is already OOXML (strip_bytes does not depend on the
# format: vsdx, templates, add-ins and single slides are cleaned the same way)
NATIVE = {".docx": "docx", ".docm": "docx", ".dotx": "docx", ".dotm": "docx",
          ".xlsx": "xlsx", ".xlsm": "xlsx", ".xltx": "xlsx", ".xltm": "xlsx", ".xlam": "xlsx",
          ".pptx": "pptx", ".pptm": "pptx", ".potx": "pptx", ".potm": "pptx", ".ppsx": "pptx",
          ".ppsm": "pptx", ".sldx": "pptx", ".sldm": "pptx",
          ".vsdx": "vsdx", ".vsdm": "vsdx"}
LEGACY = {".doc": "docx", ".dot": "docx", ".rtf": "docx", ".odt": "docx", ".wps": "docx",
          ".xls": "xlsx", ".xlt": "xlsx", ".ods": "xlsx",
          ".ppt": "pptx", ".pps": "pptx", ".pot": "pptx", ".odp": "pptx"}
OFFICE_EXT = frozenset(NATIVE) | frozenset(LEGACY)
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
SOFFICE_ENV = "LOCALANON_SOFFICE"          # path to soffice or 0 — do not use
SOFFICE_TIMEOUT_S = 300
# LibreOffice import filters by (sniff_foreign content, kind): an explicit filter instead of
# autodetection — OLE with junk or under a password is not "converted" by the text filter but fails
IMPORT_FILTERS = {
    ("ole", "docx"): "MS Word 97", ("ole", "xlsx"): "MS Excel 97", ("ole", "pptx"): "MS PowerPoint 97",
    ("rtf", "docx"): "Rich Text Format",
    ("html", "docx"): "HTML (StarWriter)", ("html", "xlsx"): "HTML (StarCalc)",
    ("odf", "docx"): "writer8", ("odf", "xlsx"): "calc8", ("odf", "pptx"): "impress8",
    ("xml", "docx"): "MS Word 2003 XML", ("xml", "xlsx"): "MS Excel 2003 XML",
}
# Word 2003 XML (.xml extension, <?mso-application progid=…?>): the export format back.
# LibreOffice only reads Excel 2003 XML (there is an import filter, no export one) — the documents
# stage removes such a file with a mark
XML2003_EXPORT = {"docx": "xml:MS Word 2003 XML"}
# newlines inside a text node (Alt+Enter in a cell) are CR in the view: view units are
# separated by LF; back — LF
_NL_IN_UNIT = "\r"


class OfficeError(Exception):
    """The document is not processed: broken, under a password, not OOXML, LibreOffice did not convert it."""


def kind_of(path) -> tuple[str, bool] | None:
    """(OOXML kind, native) by extension or None — not an office document."""
    ext = Path(str(path)).suffix.lower()
    if ext in NATIVE:
        return NATIVE[ext], True
    if ext in LEGACY:
        return LEGACY[ext], False
    return None


# ============================================================================================
# XML: tag scanner by spans (no reserialization)
# ============================================================================================
_TOK = re.compile(r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<!DOCTYPE[^>]*>"
                  r"|<(/?)([^\s/>]+)([^>]*?)(/?)>", re.S)
_ATTR = re.compile(r"""([^\s=/]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_XML_SPACE = re.compile(r"""\sxml:space\s*=\s*["']preserve["']""")
_ENT = re.compile(r"&(#x[0-9A-Fa-f]+|#[0-9]+|amp|lt|gt|quot|apos);")
_NAMED = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}


def _unescape(raw: str) -> str:
    def sub(m):
        e = m.group(1)
        if e[0] == "#":
            try:
                return chr(int(e[2:], 16) if e[1] in "xX" else int(e[1:]))
            except (ValueError, OverflowError):
                return m.group(0)
        return _NAMED[e]
    return _ENT.sub(sub, raw) if "&" in raw else raw


def _escape_text(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_attr(s: str, quote: str) -> str:
    s = _escape_text(s)
    return s.replace('"', "&quot;") if quote == '"' else s.replace("'", "&apos;")


def _local(name: str) -> str:
    return name.rsplit(":", 1)[-1]


def _parse_attrs(s: str) -> dict:
    return {m.group(1): (m.group(2) if m.group(2) is not None else m.group(3))
            for m in _ATTR.finditer(s)}


# --- policy: what goes into the view -------------------------------------------------------
# groups (a view unit = all text pieces of a group as one stream): a Word paragraph, a DrawingML
# paragraph (PowerPoint, charts, SmartArt, text boxes), a sharedStrings string / inline string /
# Excel comment, Visio shape text
_GROUPS = frozenset(("w:p", "a:p", "si", "is", "text", "Text", "r:t"))
# separators inside a group are fixed pieces (not edited, they keep word boundaries)
_FIXED = {"w:tab": "\t", "w:br": " ", "w:cr": " ", "a:br": " ", "w:noBreakHyphen": "-",
          "w:softHyphen": "", "w:ptab": "\t", "w:sym": " "}
# the text of these elements is deleted revisions (its own stream in the group)
_DEL = frozenset(("w:delText", "w:delInstrText"))
# field codes — each a separate unit (HYPERLINK "mailto:…", MERGEFIELD Name)
_INSTR = frozenset(("w:instrText", "w:delInstrText"))
# the text of these elements is not taken into the view: drawing anchor geometry, binary data,
# application name and version (app.xml)
_SKIP_TEXT_PREFIX = ("xdr:", "cdr:")
_SKIP_TEXT = frozenset(("w:binData", "pkg:binaryData", "vt:blob", "vt:stream", "vt:cf",
                        "vt:storage", "vt:ostorage", "vt:ostream", "w:docPartGallery",
                        "Application", "AppVersion"))
# VML shapes whose id/ID is the shape name (v:shapetype is not here: its id is the type)
_VML_SHAPES = frozenset(("v:shape", "v:rect", "v:oval", "v:roundrect", "v:line", "v:polyline",
                         "v:arc", "v:curve", "v:group", "v:image"))
# attribute values in the view: (attribute name, elements | None — any)
_ATTR_PICK = {
    "w:author": None, "w:initials": None,          # Word comments and revisions
    "w15:author": None, "w15:userId": None,        # people.xml
    "author": None, "initials": None,              # commentAuthors PowerPoint
    "descr": None, "title": None,                  # alternative text of pictures
    "displayName": None, "userId": None,           # persons.xml Excel
    "name": frozenset(("sheet", "definedName", "cacheField", "pivotField", "connection", "person",
                       "table", "tableColumn", "p:cmAuthor", "cmAuthor", "pivotTableDefinition",
                       "queryTable", "externalBook", "sheetName", "cm:author",
                       "wp:docPr", "p:cNvPr", "pic:cNvPr", "xdr:cNvPr", "cdr:cNvPr", "a:cNvPr",
                       "wps:cNvPr", "dgm:cNvPr", "c:cNvPr", "dsp:cNvPr", "wpg:cNvPr",
                       "property", "p188:author", "dataField", "p:custShow", "p14:section",
                       "p:tag", "p:cSld", "worksheetSource")),
    "string": frozenset(("v:textpath",)),           # WordArt / watermark text (VML)
    "href": None, "o:href": None,                   # VML: shape link
    "prompt": frozenset(("dataValidation",)),       # Excel: data validation prompt and error
    "promptTitle": frozenset(("dataValidation",)),
    "error": frozenset(("dataValidation",)),
    "errorTitle": frozenset(("dataValidation",)),
    "dataCaption": frozenset(("pivotTableDefinition",)),
    "rowHeaderCaption": frozenset(("pivotTableDefinition",)),
    "colHeaderCaption": frozenset(("pivotTableDefinition",)),
    "grandTotalCaption": frozenset(("pivotTableDefinition",)),
    "missingCaption": frozenset(("pivotTableDefinition",)),
    "errorCaption": frozenset(("pivotTableDefinition",)),
    "ddeTopic": frozenset(("ddeLink",)),
    "w:docLocation": frozenset(("w:hyperlink",)),
    "codeName": frozenset(("sheetPr", "workbookPr")),
    "v": frozenset(("s", "x")),                    # pivot cache: string items
    "connection": frozenset(("dbPr", "olapPr", "webPr", "textPr", "connection")),
    "command": frozenset(("dbPr", "olapPr")),
    "sourceFile": frozenset(("connection", "textPr")),
    "description": frozenset(("connection", "definedName", "table")),
    "url": frozenset(("webPr", "x15ac:absPath", "absPath")),
    "Target": frozenset(("Relationship",)),        # only TargetMode="External"
    "refreshedBy": frozenset(("pivotCacheDefinition",)),
    "id": _VML_SHAPES, "ID": _VML_SHAPES,
    "alt": None,                                    # VML: alternative text of any shape
    "src": frozenset(("v:imagedata", "v:fill", "v:stroke", "v:image")),
    "o:suggestedsigner": None, "o:suggestedsigner2": None, "o:suggestedsigneremail": None,
    "o:signinginstructions": None, "o:althref": None, "o:title": None,
    "comment": frozenset(("definedName",)),
    "sheet": frozenset(("worksheetSource",)),
    "val": frozenset(("sheetName", "filter", "customFilter", "p:tag")),
    "lastModifiedBy": None, "creator": None,
    "w:instr": frozenset(("w:fldSimple",)),
    "w:tooltip": frozenset(("w:hyperlink",)),
    "w:anchor": frozenset(("w:hyperlink",)),
    "w:name": frozenset(("w:bookmarkStart", "w:docVar")),
    "w:val": frozenset(("w:tblCaption", "w:tblDescription", "w:docVar", "w:alias", "w:description",
                        "w:udl", "w:query", "w:table", "w:mappedName", "w:default", "w:listEntry",
                        "w:statusText", "w:helpText")),
    "w:displayText": frozenset(("w:listItem",)),
    "w:value": frozenset(("w:listItem",)),
    "display": frozenset(("hyperlink",)),
    "tooltip": frozenset(("hyperlink", "a:hlinkClick", "a:hlinkHover", "a:hlinkMouseOver")),
    "location": frozenset(("hyperlink",)),
}
# xml:space="preserve" is set only on elements that have it in the schema: Word <w:t>/<w:delText>
# and Excel <t>; DrawingML <a:t> has no attributes
_NO_XML_SPACE_PREFIX = ("a:",)
# Excel headers and footers: codes &L &C &R &P &N &D &T &F &A &G &"font,style" &NN &KRRGGBB are fixed
_HF_ELEMS = frozenset(("oddHeader", "oddFooter", "evenHeader", "evenFooter", "firstHeader",
                       "firstFooter"))
_HF_CODE = re.compile(r'&amp;(?:"[^"]*"|K[0-9A-Fa-f]{6}|\d+|[A-Za-z&])')
# in Excel sheets <v> is a cell value: only string cells are taken (t=str|inlineStr|e)
_CELL_STR_TYPES = frozenset(("str", "inlineStr", "e"))


class _Unit:
    __slots__ = ("segs", "first")

    def __init__(self, first: int):
        self.segs = []        # ((start, end, kind, meta), xml): kind text|attr|fixed
        self.first = first


def extract_units(xml: str, part: str) -> list:
    """View units of an XML part: a list of lists of segments (start, end, kind, meta) in document
    order. kind: text — a text node (span of the raw escaped text, meta — (start tag span, element
    name)); attr — an attribute value (meta — the quote); fixed — a fixed piece (meta — the string)."""
    is_sheet = part.startswith("xl/worksheets/") or part.startswith("xl/externalLinks/")
    is_rels = part.endswith(".rels")
    is_pivot = "pivotCache" in part
    stack = []                 # (name, attr_s, start_tag_span, group_id)
    units = []
    streams = {}               # (group_id, stream) -> _Unit
    gid = 0
    pos = 0
    skip_depth = 0

    def group_ctx():
        for name, _attrs, _span, g in reversed(stack):
            if g is not None:
                return g, name
        return None, None

    def in_del():
        return any(name in ("w:del", "w:moveFrom") for name, _a, _s, _g in stack)

    def unit_for(key, start):
        u = streams.get(key)
        if u is None:
            u = _Unit(start)
            streams[key] = u
            units.append(u)
        return u

    for m in _TOK.finditer(xml):
        ms, me = m.span()
        if ms > pos:
            raw = xml[pos:ms]
            if stack and not skip_depth and raw.strip():
                name, attrs, span, _g = stack[-1]
                ok = not name.startswith(_SKIP_TEXT_PREFIX) and name not in _SKIP_TEXT
                if ok and is_sheet and name == "v":
                    ok = len(stack) > 1 and stack[-2][0] in ("c", "cell") \
                        and _parse_attrs(stack[-2][1]).get("t") in _CELL_STR_TYPES
                if ok and is_rels:
                    ok = False
                if ok:
                    g, _gname = group_ctx()
                    if g is None or name in _INSTR:
                        u = _Unit(pos)
                        units.append(u)
                    else:
                        u = unit_for((g, "del" if (name in _DEL or in_del()) else "main"), pos)
                    if name in _HF_ELEMS:
                        at = pos
                        for cm in _HF_CODE.finditer(raw):
                            cs, ce = pos + cm.start(), pos + cm.end()
                            if cs > at:
                                u.segs.append(((at, cs, "text", (span, name)), xml))
                            u.segs.append(((cs, ce, "fixed", " "), xml))
                            at = ce
                        if at < ms:
                            u.segs.append(((at, ms, "text", (span, name)), xml))
                    else:
                        u.segs.append(((pos, ms, "text", (span, name)), xml))
        pos = me
        closing, name, attr_s, selfclose = m.groups()
        if name is None:
            continue                                   # comment, CDATA, PI, DOCTYPE
        if closing:
            while stack:
                popped = stack.pop()
                if popped[0] in _SKIP_TEXT:
                    skip_depth = max(0, skip_depth - 1)
                if popped[0] == name:
                    break
            continue
        attrs = None
        if attr_s:
            base = ms + 1 + len(name)
            for am in _ATTR.finditer(attr_s):
                an, v2, v3 = am.groups()
                elems = _ATTR_PICK.get(an, "no")
                if elems == "no" or (elems is not None and name not in elems):
                    continue
                if an == "v" and not is_pivot:
                    continue
                if an == "Target":
                    if attrs is None:
                        attrs = _parse_attrs(attr_s)
                    if attrs.get("TargetMode") != "External":
                        continue
                val = v2 if v2 is not None else v3
                if not val.strip():
                    continue
                q = '"' if v2 is not None else "'"
                vs = base + am.start(2 if q == '"' else 3)
                u = _Unit(vs)
                units.append(u)
                u.segs.append(((vs, vs + len(val), "attr", q), xml))
        if name in _FIXED:
            g, _gn = group_ctx()
            if g is not None and not skip_depth:
                u = unit_for((g, "del" if in_del() else "main"), ms)
                u.segs.append(((ms, ms, "fixed", _FIXED[name]), xml))
        if selfclose:
            continue
        g = None
        if name in _GROUPS:
            gid += 1
            g = gid
        if name in _SKIP_TEXT:
            skip_depth += 1
        stack.append((name, attr_s, (ms, me), g))
    units.sort(key=lambda u: u.first)
    return [u.segs for u in units if any(k != "fixed" for (_s, _e, k, _m), _x in u.segs)]


def unit_text(segs) -> str:
    return "".join(m if k == "fixed" else _unescape(xml[s:e]) for (s, e, k, m), xml in segs)


def _distribute(segs, old: str, new: str) -> list:
    """Spread the new unit text over the editable segments: equal pieces go to their places,
    replacements/inserts — to the segment where the source span started (for a fixed one — the
    neighbouring editable one). Returns the new text of each segment (None for fixed)."""
    if len(segs) == 1:
        return [new]
    owner = []
    editable = [i for i, ((_s, _e, k, _m), _x) in enumerate(segs) if k != "fixed"]
    for i, ((s, e, k, m), xml) in enumerate(segs):
        ln = len(m) if k == "fixed" else len(_unescape(xml[s:e]))
        owner.extend([i] * ln)
    out = [None if k == "fixed" else [] for (_s, _e, k, _m), _x in segs]

    def near(i):
        if segs[i][0][2] != "fixed":
            return i
        after = [j for j in editable if j > i]
        return after[0] if after else editable[-1]

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                if out[owner[k]] is not None:
                    out[owner[k]].append(new[j1 + k - i1])
            continue
        if i1 < len(owner):
            idx = next((owner[k] for k in range(i1, i2) if segs[owner[k]][0][2] != "fixed"), None)
            if idx is None:
                idx = near(owner[i1])
        else:
            idx = editable[-1]
        out[idx].append(new[j1:j2])
    return ["".join(x) if x is not None else None for x in out]


def apply_units(xml: str, units: list, lines: list) -> str:
    """Apply view lines to a part: only changed units, edits by spans."""
    edits = []
    for segs, line in zip(units, lines):
        old = unit_text(segs)
        new = line.replace(_NL_IN_UNIT, "\n")
        if new == old:
            continue
        texts = _distribute(segs, old, new)
        for ((s, e, k, m), _x), t in zip(segs, texts):
            if k == "fixed":
                continue
            if t == _unescape(xml[s:e]):
                continue
            if k == "attr":
                edits.append((s, e, _escape_attr(t, m)))
            else:
                edits.append((s, e, _escape_text(t)))
                span, name = m
                if t and (t[0].isspace() or t[-1].isspace()) and _local(name) in ("t", "delText") \
                        and not name.startswith(_NO_XML_SPACE_PREFIX) \
                        and not _XML_SPACE.search(xml[span[0]:span[1]]):
                    ins = span[1] - 1
                    edits.append((ins, ins, ' xml:space="preserve"'))
    if not edits:
        return xml
    edits.sort(key=lambda t: (t[0], t[1]))
    out, pos = [], 0
    for s, e, rep in edits:
        if s < pos:
            raise OfficeError("overlapping XML edits")
        out.append(xml[pos:s])
        out.append(rep)
        pos = e
    out.append(xml[pos:])
    return "".join(out)


# ============================================================================================
# container: parts, encodings, view
# ============================================================================================
_XML_PART_RX = re.compile(r"\.(xml|rels)$", re.I)
_VML_RX = re.compile(r"\.vml$", re.I)
# parts cut out whole: pictures, attachments, ActiveX, thumbnail, embedded fonts,
# digital signature (_xmlsignatures: the signature picture and the certificate with the signer's name)
_MEDIA_RX = re.compile(r"(^|/)(media|embeddings|activeX|fonts)/|^docProps/thumbnail\.|\.bin$|^_xmlsignatures/"
                       r"|\.(emf|wmf|odttf|fntdata|ttf|otf|png|jpe?g|jfif|jp2|jpx|j2k|gif|bmp|dib|tiff?|svg|svgz|webp"
                       r"|ico|cur|pct|pict|pcz|eps|wdp|hdp|heic|heif|avif|tga|psd|xcf|sigs)$", re.I)
# relationships whose targets are cut out with the part, whatever its name and directory
_DROP_REL_TYPE = re.compile(r"/(image|oleObject|package|audio|video|media|hdphoto|digital-signature/(origin|signature))$")
_SNIFF_HEAD = 4096


def sniff_image(head: bytes) -> str | None:
    """Picture kind by the first bytes (png, jpeg, gif, webp, tiff, psd, ico, heic/avif, bmp, emf,
    wmf, eps, xcf, jpeg2000, svg) or None. Shared by the images stage and Office container cleaning."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if head.startswith(b"8BPS"):
        return "psd"
    if head.startswith(b"\x00\x00\x01\x00") or head.startswith(b"\x00\x00\x02\x00"):
        # ico/cur: image count 1..64, then a plausible directory entry
        n = int.from_bytes(head[4:6], "little")
        if 1 <= n <= 64 and len(head) >= 22 and head[9] == 0:
            return "ico"
    if len(head) >= 12 and head[4:8] == b"ftyp" and head[8:12] in (
            b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"avif", b"avis"):
        return "heic" if head[8:12] not in (b"avif", b"avis") else "avif"
    if head.startswith(b"BM") and len(head) >= 30:
        size = int.from_bytes(head[2:6], "little")
        off = int.from_bytes(head[10:14], "little")
        dib = int.from_bytes(head[14:18], "little")
        if head[6:10] == b"\x00\x00\x00\x00" and dib in (12, 40, 52, 56, 108, 124) and 26 <= off <= size:
            return "bmp"
    if head.startswith(b"\x01\x00\x00\x00") and head[40:44] == b" EMF":
        return "emf"
    if head.startswith(b"\xd7\xcd\xc6\x9a") or head.startswith(b"\x01\x00\x09\x00"):
        return "wmf"
    if head.startswith(b"%!PS-Adobe") and b"EPSF" in head[:64]:
        return "eps"
    if head.startswith(b"\xc5\xd0\xd3\xc6"):
        return "eps"
    if head.startswith(b"gimp xcf"):
        return "xcf"
    if head.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n"):
        return "jpeg2000"
    # SVG is text: an <svg …> root at the start (after the prolog/comments)
    text = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if text.startswith(b"<"):
        t = re.sub(rb"<\?.*?\?>|<!--.*?-->|<!DOCTYPE[^>]*>", b"", text[:_SNIFF_HEAD], flags=re.S).lstrip()
        if re.match(rb"<svg[\s>]", t, re.I):
            return "svg"
    return None
_BOMS = ((b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe\x00\x00", "utf-32-le"),
         (b"\x00\x00\xfe\xff", "utf-32-be"), (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"))
_XML_DECL_ENC = re.compile(rb"""^\s*<\?xml[^>]*?encoding\s*=\s*["']([A-Za-z0-9._-]+)["']""")


def _decode_part(raw: bytes, name: str) -> tuple:
    """(text, codec, BOM) of an XML part: by BOM; without BOM — UTF-16 by zeros in the head, otherwise
    UTF-8, and if that fails — by the encoding= declaration (checked: the text starts with "<")."""
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            try:
                return raw[len(bom):].decode(enc), enc, bom
            except UnicodeDecodeError:
                raise OfficeError(f"{name}: XML with BOM {enc} cannot be read")
    if len(raw) >= 2 and raw[0] == 0 and raw[1] != 0:
        enc = "utf-16-be"
    elif len(raw) >= 2 and raw[1] == 0 and raw[0] != 0:
        enc = "utf-16-le"
    else:
        try:
            return raw.decode("utf-8"), "utf-8", b""
        except UnicodeDecodeError:
            pass
        m = _XML_DECL_ENC.match(raw[:256])
        enc = m.group(1).decode("ascii").lower() if m else "utf-8"
        if enc in ("utf8", "utf-8") or enc.startswith(("utf-16", "utf16", "utf-32", "utf32",
                                                        "ucs-2", "ucs2", "ucs-4", "ucs4")):
            raise OfficeError(f"{name}: XML is not in utf-8 (declared {enc}, no BOM and no UTF-16/32 zeros)")
    try:
        text = raw.decode(enc)
    except (UnicodeDecodeError, LookupError):
        raise OfficeError(f"{name}: XML is not in {enc}")
    if text and not text.lstrip().startswith("<"):
        raise OfficeError(f"{name}: XML is not in {enc} (in this encoding the text does not start with '<')")
    return text, enc, b""


def _encode_part(text: str, codec: tuple, name: str) -> bytes:
    enc, bom = codec
    try:
        return bom + text.encode(enc)
    except UnicodeEncodeError as exc:
        raise OfficeError(f"{name}: a character outside the part encoding {enc}: {exc.reason}")


def _is_xml_part(name: str) -> bool:
    return bool(_XML_PART_RX.search(name)) and name != "[Content_Types].xml"


def _is_media(name: str) -> bool:
    return bool(_MEDIA_RX.search(name))


def sniff_foreign(data: bytes):
    """Kind of content that is not OOXML: "ole" (old format / under a password), "rtf", "html",
    "odf" (OpenDocument zip), "xml" (Word/Excel 2003 XML: xml2003_kind), "fodf" (flat ODF: root
    office:document) or "" — not recognized; None — OOXML zip."""
    head = data[:4096]
    if head.startswith(_OLE_MAGIC):
        return "ole"
    if data[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = set(z.namelist())
                if "[Content_Types].xml" in names:
                    return None
                if "mimetype" in names and z.read("mimetype").startswith(
                        b"application/vnd.oasis.opendocument"):
                    return "odf"
        except (zipfile.BadZipFile, OSError, KeyError):
            pass
        return None
    if b"{\\rtf" in head[:1024]:
        return "rtf"
    if re.search(rb"<(!doctype\s+html|html|head|body)[\s>]", head, re.I):
        return "html"
    if xml2003_kind(head):
        return "xml"
    if _xml_root(head) == ("document", _ODF_OFFICE_NS):
        return "fodf"
    return ""


def open_container(data: bytes) -> zipfile.ZipFile:
    if data[:8] == _OLE_MAGIC:
        raise OfficeError("not OOXML (OLE container: old format or a password-protected document)")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise OfficeError(f"cannot be read as zip: {exc}")
    if "[Content_Types].xml" not in z.namelist():
        raise OfficeError("not OOXML: no [Content_Types].xml")
    return z


def container_units(z: zipfile.ZipFile) -> list:
    """[(part name, xml, units, codec)] over all XML parts in zip order."""
    out = []
    for info in z.infolist():
        name = info.filename
        if info.is_dir() or not _is_xml_part(name):
            continue
        xml, enc, bom = _decode_part(z.read(name), name)
        units = extract_units(xml, name)
        if units:
            out.append((name, xml, units, (enc, bom)))
    return out


def _unit_line(segs) -> str:
    return unit_text(segs).replace("\r\n", _NL_IN_UNIT).replace("\n", _NL_IN_UNIT) \
        .replace("\r", _NL_IN_UNIT)


def view_lines(data: bytes) -> list[str]:
    """View lines from the bytes of an OOXML container (once per run for the same bytes, when the
    run set the view cache)."""
    from . import viewcache
    k = viewcache.key("ooxml", data) if viewcache.directory() is not None else None
    if k is not None:
        got = viewcache.get(k)
        if isinstance(got, list):
            return got
    z = open_container(data)
    out = [_unit_line(segs) for _n, _x, units, _c in container_units(z) for segs in units]
    if k is not None:
        viewcache.put(k, out)
    return out


def extract_view(path) -> list[str]:
    """Text view of a document: one line per text unit (like the main pipeline's tools/office.py)."""
    return view_lines(Path(path).read_bytes())


def _rezip(z: zipfile.ZipFile, replace: dict, drop: set = frozenset()) -> bytes:
    """Rebuild the container: parts in replace get new bytes, drop — thrown out, the rest — as is."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zo:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or name in drop:
                continue
            body = replace[name] if name in replace else z.read(name)
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zo.writestr(zi, body)
    return buf.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".office.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def apply_bytes(data: bytes, lines: list[str]) -> bytes | None:
    """Container bytes with the view applied or None — the view did not change."""
    z = open_container(data)
    parts = container_units(z)
    expect = sum(len(u) for _n, _x, u, _c in parts)
    if len(lines) != expect:
        raise OfficeError(f"view does not match the document: {len(lines)} lines, {expect} units")
    new_xml = {}
    i = 0
    for name, xml, units, codec in parts:
        chunk = lines[i:i + len(units)]
        i += len(units)
        out = apply_units(xml, units, chunk)
        if out != xml:
            new_xml[name] = _encode_part(out, codec, name)
    if not new_xml:
        return None
    return _rezip(z, new_xml)


def apply_view(path, lines: list[str]) -> None:
    """Write the changed view back into the document in place, keeping the formatting.
    An unchanged view leaves the file alone (byte for byte)."""
    p = Path(path)
    out = apply_bytes(p.read_bytes(), list(lines))
    if out is not None:
        _write_atomic(p, out)


# ============================================================================================
# media removal
# ============================================================================================
def _element_spans(xml: str, names: frozenset) -> list[tuple[int, int, str]]:
    """Spans (start, end, name) of the outer elements with names from names; nested ones of the same
    name are inside the outer one."""
    out = []
    stack = []                                   # (name, start)
    open_at = None                               # (start, stack depth, name)
    for m in _TOK.finditer(xml):
        closing, name, _attr_s, selfclose = m.groups()
        if name is None:
            continue
        if closing:
            while stack:
                n, _s = stack.pop()
                if n == name:
                    if open_at is not None and open_at[1] == len(stack):
                        out.append((open_at[0], m.end(), open_at[2]))
                        open_at = None
                    break
            continue
        if selfclose:
            if open_at is None and name in names:
                out.append((m.start(), m.end(), name))
            continue
        if open_at is None and name in names:
            open_at = (m.start(), len(stack), name)
        stack.append((name, m.start()))
    return out


def _cut(xml: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return xml
    out, pos = [], 0
    for s, e in sorted(spans):
        if s < pos:
            continue
        out.append(xml[pos:s])
        pos = e
    out.append(xml[pos:])
    return "".join(out)


# container elements of pictures and objects: removed whole if there is a reference to a
# picture/object inside (_PIC_INSIDE); background, OLE, ActiveX, sheet background picture — always
_PIC_HOLDERS = frozenset((
    "w:drawing", "w:pict", "w:object", "mc:AlternateContent",       # Word
    "w:background",
    "p:pic", "p:graphicFrame", "p:bg",                                # PowerPoint
    "xdr:twoCellAnchor", "xdr:oneCellAnchor", "xdr:absoluteAnchor",   # Excel drawings
    "cdr:relSizeAnchor", "cdr:absSizeAnchor",                         # shapes on charts
    "a:buBlip",                                                       # picture bullet
))
_PIC_INSIDE = re.compile(r"<(a:blip|v:imagedata|o:OLEObject|p:oleObj|w:control|pic:pic|xdr:pic|cdr:pic"
                         r"|a:buBlip|v:fill[^>]*\s(r:id|o:relid|src)=|v:stroke[^>]*\s(r:id|o:relid|src)=)")
_ALWAYS_CUT = frozenset(("picture", "oleObjects", "controls", "legacyDrawingHF",
                         "p:embeddedFontLst", "c:externalData", "w:embedRegular", "w:embedBold",
                         "w:embedItalic", "w:embedBoldItalic", "a:blip", "v:imagedata", "o:OLEObject",
                         "p:oleObj", "w:control"))
_BLIPFILL = frozenset(("a:blipFill",))
_VML_PIC_SHAPES = frozenset(("v:shape", "v:rect", "v:image", "v:group"))
_RID_ATTR = re.compile(r"""\s([\w.-]+:(?:id|embed|link|pict|href|dm|lo|qs|cs))\s*=\s*["'](rId[^"']*|[^"']*)["']""")
_RID_VALUE = re.compile(r"^rId\d+$")
_RELS_RX = re.compile(r"<Relationship\b[^>]*/>|<Relationship\b[^>]*>.*?</Relationship>", re.S)
_OVERRIDE_RX = re.compile(r"<Override\b[^>]*/>|<Override\b[^>]*>.*?</Override>", re.S)


def _strip_part_xml(xml: str, name: str) -> tuple[str, int]:
    """Picture and object markup out of an XML part; → (xml, how many elements were cut out)."""
    n = 0
    spans = []
    for s, e, el in _element_spans(xml, _PIC_HOLDERS):
        body = xml[s:e]
        if el in ("w:background", "p:bg"):
            cut = bool(re.search(r"\s(r:id|r:embed|o:relid|src)=", body)) or "<a:blip" in body
        else:
            cut = bool(_PIC_INSIDE.search(body))
        if cut:
            spans.append((s, e))
    n += len(spans)
    xml = _cut(xml, spans)
    # picture fill of shapes, tables, theme → no fill
    spans = [(s, e) for s, e, _el in _element_spans(xml, _BLIPFILL)]
    if spans:
        n += len(spans)
        out, pos = [], 0
        for s, e in spans:
            out.append(xml[pos:s])
            out.append("<a:noFill/>")
            pos = e
        out.append(xml[pos:])
        xml = "".join(out)
    # VML parts and leftovers: shapes with a picture, single references
    if _VML_RX.search(name):
        spans = [(s, e) for s, e, _el in _element_spans(xml, _VML_PIC_SHAPES)
                 if _PIC_INSIDE.search(xml[s:e])]
        n += len(spans)
        xml = _cut(xml, spans)
    spans = [(s, e) for s, e, _el in _element_spans(xml, _ALWAYS_CUT)]
    n += len(spans)
    xml = _cut(xml, spans)
    return xml, n


def _rels_dir(part: str) -> str:
    """Directory against which Target values in the .rels of part are resolved."""
    d = posixpath.dirname(part)
    return d[:-len("/_rels")] if d.endswith("_rels") else ""


def _resolve_target(rels_name: str, target: str) -> str:
    base = _rels_dir(rels_name)
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(posixpath.join(base, target) if base else target)


def _clean_rels(xml: str, rels_name: str, present: set) -> tuple[str, set, set]:
    """Relationships without a target (internal Target not among present) and relationships of the
    removed types (picture, OLE, embedded package, signature) — out. → (xml, kept Ids, removed Ids)."""
    keep_ids, drop_ids = set(), set()
    spans = []
    for m in _RELS_RX.finditer(xml):
        attrs = _parse_attrs(m.group(0))
        rid = attrs.get("Id", "")
        target = attrs.get("Target", "")
        if _DROP_REL_TYPE.search(attrs.get("Type", "")) and attrs.get("TargetMode") != "External":
            drop_ids.add(rid)
            spans.append(m.span())
            continue
        if attrs.get("TargetMode") == "External" or not target:
            keep_ids.add(rid)
            continue
        if _resolve_target(rels_name, target) in present:
            keep_ids.add(rid)
        else:
            drop_ids.add(rid)
            spans.append(m.span())
    return _cut(xml, spans), keep_ids, drop_ids


def _rel_targets_to_drop(z: zipfile.ZipFile, names: list) -> set:
    """Parts referenced by relationships of the removed types (…/image, …/oleObject, …/package,
    …/audio, …/video, …/media, signature) — whatever the part name and directory."""
    out = set()
    for n in names:
        if not n.endswith(".rels"):
            continue
        try:
            xml = _decode_part(z.read(n), n)[0]
        except OfficeError:
            continue
        for m in _RELS_RX.finditer(xml):
            a = _parse_attrs(m.group(0))
            if _DROP_REL_TYPE.search(a.get("Type", "")) and a.get("TargetMode") != "External" and a.get("Target"):
                out.add(_resolve_target(n, a["Target"]))
    return out


def _image_parts(z: zipfile.ZipFile, names: list) -> set:
    """Parts with a picture signature, whatever their extension and directory (XML parts and
    [Content_Types].xml are not checked)."""
    out = set()
    for n in names:
        if _is_xml_part(n) or n == "[Content_Types].xml" or _VML_RX.search(n):
            continue
        with z.open(n) as fh:
            head = fh.read(_SNIFF_HEAD)
        if sniff_image(head):
            out.add(n)
    return out


_DEFAULT_RX = re.compile(r"<Default\b[^>]*/>|<Default\b[^>]*>.*?</Default>", re.S)
_KEEP_DEFAULT_EXT = frozenset(("rels", "xml"))


def _clean_content_types(ct: str, present: set) -> str:
    """[Content_Types].xml: Override for missing parts and Default for extensions no part has any
    more (except rels and xml) — out."""
    spans = []
    for m in _OVERRIDE_RX.finditer(ct):
        pn = _parse_attrs(m.group(0)).get("PartName", "").lstrip("/")
        if pn and pn not in present:
            spans.append(m.span())
    exts = {posixpath.splitext(n)[1].lstrip(".").lower() for n in present}
    for m in _DEFAULT_RX.finditer(ct):
        ext = _parse_attrs(m.group(0)).get("Extension", "").lower()
        if ext and ext not in _KEEP_DEFAULT_EXT and ext not in exts:
            spans.append(m.span())
    return _cut(ct, spans)


def _rels_for(part: str) -> str:
    d, b = posixpath.split(part)
    return posixpath.join(d, "_rels", b + ".rels") if d else posixpath.join("_rels", b + ".rels")


def _dangling(xml: str, ids: set) -> list[tuple[int, int, str]]:
    """r:* attributes with a value outside ids: (attribute start, end, value)."""
    out = []
    for m in _RID_ATTR.finditer(xml):
        v = m.group(2)
        prefix, local = m.group(1).split(":", 1)
        if v and v not in ids and (prefix == "r" or _RID_VALUE.match(v)
                                   or local in ("embed", "link", "pict")):
            out.append((m.start(), m.end(), v))
    return out


def _cut_dangling(xml: str, ids: set) -> tuple[str, int]:
    """Elements referencing relationships without a target — out (the element whose start tag has the reference)."""
    n = 0
    for _ in range(10000):
        bad = _dangling(xml, ids)
        if not bad:
            break
        at = bad[0][0]
        # the start tag containing the attribute
        lt = xml.rfind("<", 0, at)
        m = _TOK.match(xml, lt)
        if m is None or m.group(2) is None:
            break
        name = m.group(2)
        if m.group(4):                                 # self-closing
            xml = xml[:lt] + xml[m.end():]
        else:
            spans = [(s, e) for s, e, _el in _element_spans(xml[lt:], frozenset((name,)))]
            if not spans or spans[0][0] != 0:
                break
            xml = xml[:lt] + xml[lt + spans[0][1]:]
        n += 1
    return xml, n


def strip_bytes(data: bytes) -> tuple[bytes, int]:
    """(container bytes without media, how many parts and elements were cut out)."""
    z = open_container(data)
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    drop = {n for n in names if _is_media(n)}
    drop |= _rel_targets_to_drop(z, names) & set(names)
    drop |= _image_parts(z, [n for n in names if n not in drop])
    present = set(names) - drop
    cut = len(drop)
    new = {}
    xml_parts = {}
    for n in names:
        if n in drop or n == "[Content_Types].xml":
            continue
        if _is_xml_part(n) and not n.endswith(".rels") or _VML_RX.search(n):
            xml, enc, bom = _decode_part(z.read(n), n)
            out, k = _strip_part_xml(xml, n)
            cut += k
            xml_parts[n] = (out, (enc, bom), out != xml)
    # relationships: without a target — out; then references to removed relationships — out of the parts
    rel_ids: dict = {}
    for n in names:
        if n in drop or not n.endswith(".rels"):
            continue
        xml, enc, bom = _decode_part(z.read(n), n)
        out, keep, dropped = _clean_rels(xml, n, present)
        cut += len(dropped)
        rel_ids[n] = keep
        if out != xml:
            new[n] = _encode_part(out, (enc, bom), n)
    for n, (xml, codec, changed) in xml_parts.items():
        ids = rel_ids.get(_rels_for(n), set())
        out, k = _cut_dangling(xml, ids)
        cut += k
        if changed or k:
            new[n] = _encode_part(out, codec, n)
    # [Content_Types].xml: Override for removed parts and Default without parts — out
    ct_name = "[Content_Types].xml"
    ct, enc, bom = _decode_part(z.read(ct_name), ct_name)
    ct_out = _clean_content_types(ct, present)
    if ct_out != ct:
        new[ct_name] = _encode_part(ct_out, (enc, bom), ct_name)
    return _rezip(z, new, drop), cut


def check_bytes(data: bytes, media: bool = True) -> list[str]:
    """Container problems: media parts, XML that does not parse, relationships without a target,
    references to relationships without a target. Empty — the document is intact. ``media=False`` — integrity only (pictures may stay)."""
    problems = []
    try:
        z = open_container(data)
    except OfficeError as exc:
        return [str(exc)]
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    present = set(names)
    if media:
        for n in names:
            if _is_media(n):
                problems.append(f"media part left: {n}")
        for n in _image_parts(z, [n for n in names if not _is_media(n)]):
            problems.append(f"part with a picture signature left: {n}")
    rel_ids = {}
    for n in names:
        if _is_xml_part(n) or n == "[Content_Types].xml" or _VML_RX.search(n):
            raw = z.read(n)
            try:
                xml = _decode_part(raw, n)[0]
                ET.fromstring(re.sub(r"^\s*<\?xml[^>]*\?>", "", xml, count=1).encode("utf-8"))
            except (OfficeError, ET.ParseError) as exc:
                problems.append(f"XML does not parse: {n}: {exc}")
                continue
            if n.endswith(".rels"):
                ids = set()
                for m in _RELS_RX.finditer(xml):
                    a = _parse_attrs(m.group(0))
                    if media and a.get("TargetMode") != "External" and _DROP_REL_TYPE.search(a.get("Type", "")):
                        problems.append(f"relationship to a picture/object/signature left: {n}: {a.get('Id')} → {a.get('Target')}")
                    elif a.get("TargetMode") != "External" and a.get("Target") and \
                            _resolve_target(n, a["Target"]) not in present:
                        problems.append(f"relationship without a target: {n}: {a.get('Id')} → {a['Target']}")
                    ids.add(a.get("Id", ""))
                rel_ids[n] = ids
    for n in names:
        if (_is_xml_part(n) and not n.endswith(".rels")) or _VML_RX.search(n):
            xml = _decode_part(z.read(n), n)[0]
            ids = rel_ids.get(_rels_for(n), set())
            for _s, _e, v in _dangling(xml, ids):
                problems.append(f"reference to a relationship without a target: {n}: {v}")
    return problems


def strip_media(path) -> int:
    """Cut media and embedded objects out of docx/xlsx/pptx in place; → how many parts were removed.
    The result is checked (check_bytes); problems — OfficeError, the file is unchanged."""
    p = Path(path)
    data = p.read_bytes()
    out, cut = strip_bytes(data)
    problems = check_bytes(out)
    if problems:
        raise OfficeError("document is not intact after media removal: " + "; ".join(problems[:3]))
    if out != data:
        _write_atomic(p, out)
    return cut


def check(path) -> list[str]:
    return check_bytes(Path(path).read_bytes())


# ============================================================================================
# pictures of headers, footers and slide masters (executor logos)
# ============================================================================================
# parts reached by these relationship types are page furniture: Word headers and footers,
# PowerPoint slide masters and layouts (and notes/handout masters)
_HF_REL_TYPE = re.compile(r"/(header|footer|slideMaster|slideLayout|notesMaster|handoutMaster)$")
# Excel header/footer codes; "&&" is a literal ampersand, "&G" is the picture placeholder
_HF_PIC_CODE = re.compile(r'&amp;(?:&amp;|"[^"]*"|K[0-9A-Fa-f]{6}|\d+|[A-Za-z])')
_HF_TEXT = frozenset(_HF_ELEMS | {"x:" + e for e in _HF_ELEMS})
_LEGACY_HF = frozenset(("legacyDrawingHF", "x:legacyDrawingHF"))


def _rel_entries(xml: str):
    for m in _RELS_RX.finditer(xml):
        yield m.span(), _parse_attrs(m.group(0))


def _drop_rels(xml: str, rels_name: str, pick) -> tuple[str, set, set]:
    """Relationships for which ``pick(attrs)`` holds — out. → (xml, kept Ids, internal targets
    of the dropped ones)."""
    keep, targets, spans = set(), set(), []
    for span, a in _rel_entries(xml):
        if pick(a):
            spans.append(span)
            if a.get("TargetMode") != "External" and a.get("Target"):
                targets.add(_resolve_target(rels_name, a["Target"]))
        else:
            keep.add(a.get("Id", ""))
    return _cut(xml, spans), keep, targets


def _drop_hf_pic_codes(xml: str) -> str:
    """``&G`` (header picture) out of Excel header/footer strings; other codes and text stay."""
    out, pos = [], 0
    for s, e, _el in _element_spans(xml, _HF_TEXT):
        body = xml[s:e]
        body = _HF_PIC_CODE.sub(lambda m: "" if m.group(0) == "&amp;G" else m.group(0), body)
        out.append(xml[pos:s])
        out.append(body)
        pos = e
    out.append(xml[pos:])
    return "".join(out)


def strip_hf_bytes(data: bytes) -> tuple[bytes, int]:
    """Pictures of headers, footers and slide masters out of an OOXML container; text of the
    headers and pictures of the body stay. → (bytes, pictures removed).

    * Word ``header*``/``footer*`` and PowerPoint masters/layouts (found by relationship type, not
      by name): picture markup (``w:drawing``, ``w:pict``, ``p:pic``, picture background…) and
      their image/OLE relationships;
    * Excel: ``legacyDrawingHF`` of a sheet, the VML part it points to (header pictures only) and
      the ``&G`` codes of the header strings;
    * a media part goes only when no remaining relationship points to it (a body picture may
      share it); ``[Content_Types].xml`` follows.
    """
    z = open_container(data)
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    present = set(names)
    rels = {n: _decode_part(z.read(n), n) for n in names if n.endswith(".rels")}
    rel_xml = {n: v[0] for n, v in rels.items()}
    hf_parts = set()
    for n, xml in rel_xml.items():
        for _span, a in _rel_entries(xml):
            if a.get("TargetMode") != "External" and a.get("Target") and _HF_REL_TYPE.search(a.get("Type", "")):
                t = _resolve_target(n, a["Target"])
                if t in present and _is_xml_part(t):
                    hf_parts.add(t)
    parts: dict = {}                       # part → (xml, codec) changed
    drop: set = set()
    candidates: set = set()                # media targets of dropped relationships
    cut = 0

    def part_xml(n):
        if n in parts:
            return parts[n][0]
        return _decode_part(z.read(n), n)[0]

    def set_part(n, xml):
        codec = parts[n][1] if n in parts else _decode_part(z.read(n), n)[1:]
        parts[n] = (xml, codec)

    def set_rels(n, xml):
        rel_xml[n] = xml
        set_part(n, xml)

    # Excel sheets: legacyDrawingHF → VML part with the header pictures
    for n in names:
        if n.endswith(".rels") or not _is_xml_part(n) or n in hf_parts:
            continue
        if b"legacyDrawingHF" not in z.read(n):
            continue
        xml = part_xml(n)
        spans = _element_spans(xml, _LEGACY_HF)
        if not spans:
            continue
        ids = set()
        for s, e, _el in spans:
            m = re.search(r"""\sr:id\s*=\s*["']([^"']+)["']""", xml[s:e])
            if m:
                ids.add(m.group(1))
        xml = _drop_hf_pic_codes(_cut(xml, [(s, e) for s, e, _el in spans]))
        set_part(n, xml)
        cut += len(spans)
        rn = _rels_for(n)
        if rn in rel_xml:
            out, _keep, targets = _drop_rels(rel_xml[rn], rn, lambda a: a.get("Id") in ids)
            set_rels(rn, out)
            for t in targets & present:
                if _VML_RX.search(t):
                    drop.add(t)
                    vr = _rels_for(t)
                    if vr in rel_xml:
                        drop.add(vr)
                        for _span, a in _rel_entries(rel_xml[vr]):
                            if a.get("TargetMode") != "External" and a.get("Target"):
                                candidates.add(_resolve_target(vr, a["Target"]))
                else:
                    candidates.add(t)
    # Word headers/footers, PowerPoint masters and layouts
    for n in sorted(hf_parts):
        xml = part_xml(n)
        out, k = _strip_part_xml(xml, n)
        rn = _rels_for(n)
        dropped = False
        if rn in rel_xml:
            rx, keep, targets = _drop_rels(
                rel_xml[rn], rn, lambda a: a.get("TargetMode") != "External"
                and bool(_DROP_REL_TYPE.search(a.get("Type", ""))))
            if targets or rx != rel_xml[rn]:
                set_rels(rn, rx)
                candidates |= targets
                dropped = True
                out, k2 = _cut_dangling(out, keep)
                k += k2
        if k or dropped:
            cut += k
            set_part(n, out)
    for n in drop:
        parts.pop(n, None)
    # media still referenced by a remaining relationship stays
    referenced = set()
    for n, xml in rel_xml.items():
        if n in drop:
            continue
        for _span, a in _rel_entries(xml):
            if a.get("TargetMode") != "External" and a.get("Target"):
                referenced.add(_resolve_target(n, a["Target"]))
    drop |= {t for t in candidates & present if t not in referenced and not _is_xml_part(t)}
    if not parts and not drop:
        return data, 0
    ct_name = "[Content_Types].xml"
    new = {n: _encode_part(xml, codec, n) for n, (xml, codec) in parts.items()}
    if ct_name in present:
        ct, enc, bom = _decode_part(z.read(ct_name), ct_name)
        ct_out = _clean_content_types(ct, present - drop)
        if ct_out != ct:
            new[ct_name] = _encode_part(ct_out, (enc, bom), ct_name)
    return _rezip(z, new, drop), cut


def strip_hf_images(path) -> int:
    """Pictures of headers, footers and slide masters out of docx/xlsx/pptx in place; → how many.
    The result is checked (``check_bytes(media=False)``); a problem — OfficeError, file unchanged."""
    p = Path(path)
    data = p.read_bytes()
    out, cut = strip_hf_bytes(data)
    if out == data:
        return 0
    problems = check_bytes(out, media=False)
    if problems:
        raise OfficeError("document broken after removing header pictures: " + "; ".join(problems[:3]))
    _write_atomic(p, out)
    return cut


# ============================================================================================
# LibreOffice: old formats
# ============================================================================================
def soffice_bin() -> str | None:
    """Path to soffice or None: the LOCALANON_SOFFICE variable (path | 0), PATH, the standard
    directories of Windows (Program Files), Linux, macOS."""
    raw = os.environ.get(SOFFICE_ENV, "").strip()
    if raw == "0":
        return None
    if raw:
        return raw if Path(raw).is_file() else None
    for name in ("soffice", "soffice.exe", "libreoffice"):
        hit = shutil.which(name)
        if hit:
            return hit
    cands: list[Path] = []
    if os.name == "nt":
        for key in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            base = os.environ.get(key, "")
            if base:
                cands.extend(Path(base).glob("LibreOffice*/program/soffice.exe"))
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            cands.extend(Path(local).glob("Programs/LibreOffice*/program/soffice.exe"))
    else:
        cands.extend(Path(x) for x in ("/usr/bin/soffice", "/usr/lib/libreoffice/program/soffice",
                                       "/usr/lib64/libreoffice/program/soffice",
                                       "/snap/bin/libreoffice",
                                       "/Applications/LibreOffice.app/Contents/MacOS/soffice"))
        cands.extend(Path("/opt").glob("libreoffice*/program/soffice"))
    for c in cands:
        if c.is_file():
            return str(c)
    return None


def import_filter(foreign: str, kind: str) -> str | None:
    """LibreOffice import filter by content (sniff_foreign) and document kind or None —
    LibreOffice does not read such content in this kind (PowerPoint 2003 XML, unrecognized)."""
    return IMPORT_FILTERS.get((foreign or "", kind))


_PROGID_RX = re.compile(rb"<\?mso-application\s+progid\s*=\s*[\"'](Word|Excel|PowerPoint)\.", re.I)
# document root after the prolog (declaration, PI, comments, DOCTYPE)
_XML_ROOT_RX = re.compile(rb"^(?:\xef\xbb\xbf)?\s*(?:(?:<\?.*?\?>|<!--.*?-->|<!DOCTYPE[^>]*>)\s*)*"
                          rb"<([A-Za-z_][\w.-]*)(?::([A-Za-z_][\w.-]*))?([^>]*)>", re.S)
_ODF_OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_ROOT_2003 = {("wordDocument", "http://schemas.microsoft.com/office/word/2003/wordml"): "docx",
              ("Workbook", "urn:schemas-microsoft-com:office:spreadsheet"): "xlsx"}


def _xml_root(head: bytes) -> tuple[str, str] | None:
    """(local name, namespace) of the XML root by the header or None — not XML."""
    m = _XML_ROOT_RX.match(head)
    if not m:
        return None
    prefix, local = (m.group(1), m.group(2)) if m.group(2) else (None, m.group(1))
    attr = b"xmlns:" + prefix if prefix else b"xmlns"
    ns = re.search(rb"\s" + re.escape(attr) + rb"\s*=\s*[\"']([^\"']*)[\"']", m.group(3))
    return local.decode("ascii"), ns.group(1).decode("utf-8", "replace") if ns else ""


def xml2003_kind(head: bytes) -> str | None:
    """Word/Excel 2003 XML document kind by the header: docx | xlsx | pptx | None (not Office XML).
    Signs — only <?mso-application progid=…?> or a w:wordDocument root (…/word/2003/wordml)
    or ss:Workbook (urn:schemas-microsoft-com:office:spreadsheet); a mention of
    urn:schemas-microsoft-com:office:* is not one: every Word 2007+ part declares it (document.xml,
    header*.xml, settings.xml)."""
    m = _PROGID_RX.search(head)
    if m:
        return {b"word": "docx", b"excel": "xlsx", b"powerpoint": "pptx"}[m.group(1).lower()]
    return _ROOT_2003.get(_xml_root(head))


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and all its descendants: soffice is a wrapper, the child soffice.bin does the
    conversion, and proc.kill() left it orphaned. On POSIX — the whole group (the process runs in
    its own session), on Windows — taskkill /T over the tree."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait()


def _run_soffice(cmd: list[str], timeout: int) -> None:
    """cmd in its own process group; past the timeout the group is killed, TimeoutExpired."""
    if os.name == "nt":
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        group = {"start_new_session": True}
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, **group)
    try:
        proc.wait(timeout=timeout)
    except BaseException:
        _kill_tree(proc)
        raise


def soffice_convert(src, fmt: str, timeout: int = SOFFICE_TIMEOUT_S, infilter: str | None = None) -> bytes:
    """File → format fmt (docx|xlsx|pptx|doc|xls|ppt|rtf|odt|ods|odp|pdf|"xml:MS Word 2003 XML")
    through LibreOffice; → the result bytes. infilter — an explicit import filter (without it
    LibreOffice picks one itself and reads junk with the text filter). The profile is in a temporary
    directory, the file is copied under a simple name."""
    binp = soffice_bin()
    if binp is None:
        raise OfficeError("LibreOffice (soffice) not found")
    src = Path(src)
    with tempfile.TemporaryDirectory(prefix="localanon_lo_", ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        indir, outdir, profile = tmp / "in", tmp / "out", tmp / "profile"
        indir.mkdir()
        outdir.mkdir()
        safe = indir / ("doc" + src.suffix.lower())
        shutil.copyfile(src, safe)
        cmd = [binp, "--headless", "--norestore", "--nologo", "--nolockcheck",
               "-env:UserInstallation=" + profile.resolve().as_uri()]
        if infilter:
            cmd.append("--infilter=" + infilter)
        cmd += ["--convert-to", fmt, "--outdir", str(outdir), str(safe)]
        try:
            _run_soffice(cmd, timeout)
        except subprocess.TimeoutExpired:
            raise OfficeError(f"LibreOffice did not finish in {timeout} s")
        except OSError as exc:
            raise OfficeError(f"LibreOffice failed to start: {exc.strerror or exc}")
        out = outdir / ("doc." + fmt.split(":")[0])
        if not out.is_file() or out.stat().st_size == 0:
            raise OfficeError(f"LibreOffice did not convert to {fmt} (broken file, password, format)")
        return out.read_bytes()


def to_ooxml(path, kind: str, infilter: str | None = None) -> bytes:
    """Old format → OOXML container of kind (docx|xlsx|pptx) through LibreOffice; infilter — an
    explicit import filter (import_filter), so that junk and a password-protected document are not
    read by the text filter."""
    data = soffice_convert(path, kind, infilter=infilter)
    open_container(data)
    return data


def from_ooxml(data: bytes, ext: str, kind: str | None = None) -> bytes:
    """OOXML container → old format ext (.doc, .xls, .rtf, …; .xml — Word/Excel 2003 XML by
    kind) through LibreOffice."""
    ext = ext.lower()
    if ext == ".xml":
        if kind not in XML2003_EXPORT:
            raise OfficeError(f"no Office 2003 XML format for kind {kind}")
        fmt = XML2003_EXPORT[kind]
    else:
        kind = LEGACY.get(ext) or NATIVE.get(ext)
        if kind is None:
            raise OfficeError(f"unknown extension {ext}")
        fmt = ext.lstrip(".")
    with tempfile.TemporaryDirectory(prefix="localanon_lo_", ignore_cleanup_errors=True) as tmp:
        p = Path(tmp) / ("doc." + kind)
        p.write_bytes(data)
        return soffice_convert(p, fmt)
