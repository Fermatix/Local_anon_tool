"""Mail ``.eml`` / ``.mbox``: parsing into parts for masking and assembling
back in the original encoding.

Finding: quoted-printable in the body and ``=?utf-8?Q?…?=`` / ``=?utf-8?B?…?=``
in headers were never decoded — names in a mail were invisible to detectors; and
what the encoding detector did catch, it broke: a B-encoded From header whole
into ``=?utf-8?B?ENCODED_REDACTED?=``, a base64 body whole into SECRET_REDACTED.
Owner's decision: decode for masking and assemble back in the SAME encoding
(QP → QP, base64 → base64, Q/B word → the same word with the same encoding
and charset); signatures (DKIM/S-MIME) are invalid after masking anyway.

Model. A mail is bytes; parsing gives:
  * ``slots`` — the "view" for masking: one slot per header (the value with
    encoded-words decoded and folding removed), per body of each text part
    (text/* — decoded from QP/base64 in the part's charset) and for the mbox
    ``From `` envelope line; attachments (not text/*, or
    ``Content-Disposition: attachment``) do not go into the view — only metadata
    in ``parts``;
  * ``render(texts)`` — assembly: a slot whose text did not change comes out of
    the original bytes as it was; a changed one is re-encoded into the encoding
    and charset of its part (a header — chunk by chunk: encoded-word into an
    encoded-word of the same form, literal into literal); everything between
    slots (header names, MIME boundaries, preambles, attachments, order) — byte
    for byte.

Why spans and not ``email.generator``: the stdlib generator is not byte-exact
even without edits — it rewrites CRLF to LF (policy.linesep), refolds long
headers, loses the mbox ``From ``. So the standard ``email`` here is only
responsible for header semantics (Content-Type, boundary, charset, file name —
``BytesHeaderParser`` with compat32, which rewrites nothing), while the
positions of all parts are computed over bytes by the module itself.
The property "parse → assemble without edits = source" holds by construction:
an unchanged slot comes out as a slice of the original bytes, gaps as slices.

Not supported (the part stays as is and does not get into the view, the reason
is in ``parts[i].reason`` and ``defects``): uuencode/x-* CTE, non-text
attachments, message/rfc822 under base64/QP, multipart without a single boundary
line, an unknown charset (including the name of a non-text codec: base64,
hex, rot13) is read as UTF-8 and re-encoded to UTF-8 under the old label
(only for a CHANGED part). Bytes outside the part's charset (a utf-8 label on
cp1251 bytes, an 8-bit header without encoded-word) stay in ``Slot.text`` as
surrogates U+DC80..DCFF (surrogateescape) — detectors do not see them, there is
a ``defects`` record by the slot key; write the view out (log, file) with
``errors="surrogateescape"``, otherwise ``encode("utf-8")`` fails; on assembly
the surrogates turn back into the same bytes. HTML entities ``&#…;`` in text/html
are not expanded (a separate step). RFC 2231 parameters (filename*=)
are read for metadata but not masked. For a CHANGED part: a character outside
the BODY charset is written as "?" (the engine mask is ASCII, a theoretical case;
an encoded-word carries its charset in the word itself, and such a word goes to
utf-8); a group of adjacent encoded-words with different charsets is re-encoded
run by run, each in its own charset; a changed group is folded — one word per
line (mail EOL + space); mixed line breaks in a QP body are normalized to the
part's line break; a line break written by the mask into a header or envelope
becomes a space; edge whitespace of a header value is not preserved; the
trailing line break of a body (the part boundary, in mbox — before the next
mail's "From ") is preserved: if the mask removed it, it is restored.

Broken input goes into the view and into ``defects`` (the caller decides whether
to take the file): a header block line without "Name:" and without indentation
(a line break without indentation, export garbage) — slot ``kind="header"``,
``header=None``, key ``./[n]``; stdlib semantics (Content-Type, charset,
boundary) are read from "Name: …" lines and take effect after such a line. A BOM
before the mail is outside the view and stays. An mbox not starting with "From "
(BOM, whitespace, a file with one mail without an envelope): a non-whitespace
prefix before the first "From " line is a mail without an envelope with a
defect; whitespace is a gap.

Interface for the integration package::

    kind = kind_for_path(path)            # "eml" | "mbox" | None
    doc = parse(raw, kind)                # Mail | Mbox
    for s in doc.slots: ...               # Slot(key, kind, text, part, header, encoding, charset)
    out = doc.render({s.key: new_text})   # bytes; without an argument == raw
    out = mask_bytes(raw, kind, fn)       # fn(Slot) -> str — in one call
    doc.defects                           # list[str]: what was not fully parsed

Slot keys: ``./Subject[0]`` (a root header, ordinal among headers of the same
name), ``./[0]`` (a nameless header block line), ``1.2/body`` (a part body by
path as in email.walk), ``From_`` (mbox envelope); in mbox — prefixed with the
mail number ``3:…``.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import difflib
import email.header
import email.parser
import email.policy
import os
import re
from dataclasses import dataclass, field

__all__ = ["Mail", "Mbox", "Slot", "Part", "MailParseError", "kind_for_path",
           "parse", "parse_eml", "parse_mbox", "mask_bytes",
           "STRUCTURAL_HEADERS"]

# headers that never go into the view: MIME markup, not data
STRUCTURAL_HEADERS = frozenset({
    "mime-version", "content-transfer-encoding", "content-id",
    "content-length", "content-md5"})
# Content-Type of a container (carries boundary) is markup too; Content-Type
# of a leaf (charset, legacy name=) goes into the view

_TEXTUAL_MESSAGE = frozenset({"message/delivery-status",
                              "message/disposition-notification"})
_NESTED = frozenset({"message/rfc822", "message/global"})
_TRANSPARENT_CTE = frozenset({"", "7bit", "8bit", "binary"})
_CTE_OK = _TRANSPARENT_CTE | {"quoted-printable", "base64"}
_UTF8_ALIASES = frozenset({"", "utf-8", "utf8", "us-ascii", "ascii",
                           "unknown-8bit", "x-unknown", "default"})

_BOM = b"\xef\xbb\xbf"
_EW = re.compile(rb"=\?([^?\s]+)\?([QqBb])\?([^?\s]*)\?=")
_FOLD = re.compile(rb"\r?\n")
_WS_ONLY = re.compile(rb"[ \t\r\n]*\Z")
_FIELD_NAME = re.compile(rb"[!-9;-~]+")
_Q_SAFE = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                    b"0123456789!*+-/")
_CRLF_RE = re.compile(r"\r\n|\r|\n")
_SURROGATE = re.compile("[\udc80-\udcff]")     # bytes not parsed during decoding


class MailParseError(ValueError):
    """Parsing did not add up (spans overlapped) — the caller leaves the file as is."""


# --------------------------------------------------------------------------- #
# codecs
# --------------------------------------------------------------------------- #
def _enc_errors(exc):
    """surrogateescape for bytes not parsed during decoding (U+DC80..DCFF),
    "?" for characters missing from the part's charset (the mask is ASCII, a rare
    case: non-ASCII text written by the mask into a koi8-r part)."""
    if isinstance(exc, UnicodeEncodeError):
        out = bytearray()
        for ch in exc.object[exc.start:exc.end]:
            o = ord(ch)
            if 0xDC80 <= o <= 0xDCFF:
                out.append(o - 0xDC00)
            else:
                out += b"?"
        return bytes(out), exc.end
    raise exc


codecs.register_error("anon_mail", _enc_errors)


def _codec(charset) -> tuple[str, bool]:
    """Python codec name for a mail charset → (codec, whether it is known). Unknown
    or empty — utf-8 (ASCII-compatible; unparsed bytes go into
    surrogateescape and come back as the same bytes)."""
    if isinstance(charset, bytes):
        charset = charset.decode("ascii", "replace")
    cs = (charset or "").split("*", 1)[0].strip().strip('"').lower()
    if cs in _UTF8_ALIASES:
        return "utf-8", True
    try:
        info = codecs.lookup(cs)
    except LookupError:
        return "utf-8", False
    # codecs.lookup also finds bytes→bytes / str→str codecs (base64, hex,
    # rot13, zlib, quopri): a charset with such a name is not text, .decode() on
    # it fails with LookupError; treat it as unknown
    if not getattr(info, "_is_text_encoding", True):
        return "utf-8", False
    return info.name, True


def _decode(bs: bytes, codec: str) -> str:
    return bs.decode(codec, "surrogateescape")


def _encode(text: str, codec: str) -> bytes:
    return text.encode(codec, "anon_mail")


def _decode_ew(enc: bytes, payload: bytes):
    """Encoded-word bytes or None (does not decode → stays a literal)."""
    if enc in (b"B", b"b"):
        s = payload.rstrip(b"=")
        if len(s) % 4 == 1:
            return None
        try:
            return base64.b64decode(s + b"=" * (-len(s) % 4), validate=True)
        except (binascii.Error, ValueError):
            return None
    try:
        return binascii.a2b_qp(payload, header=True)
    except (binascii.Error, ValueError):
        return None


def _q_encode(data: bytes) -> bytes:
    out = bytearray()
    for b in data:
        if b == 0x20:
            out += b"_"
        elif b in _Q_SAFE:
            out.append(b)
        else:
            out += b"=%02X" % b
    return bytes(out)


def _encodable(text: str, codec: str) -> bool:
    """Whether the text is representable in the codec (surrogates as bytes, as on assembly)."""
    try:
        text.encode(codec, "surrogateescape")
    except UnicodeEncodeError:
        return False
    return True


def _encode_words(text: str, charset: bytes, enc: bytes, codec: str,
                  eol: bytes = b"\r\n") -> bytes:
    """Text → a chain of encoded-words of the same form (charset and Q/B as in
    the source), each ≤ 75 chars, split at character boundaries; words joined
    by folding (mail EOL + space): a line ≤ 76 bytes, like the stdlib
    generator — RFC 5322 requires ≤ 998 (MUST) and recommends ≤ 78."""
    head = b"=?" + charset + b"?" + enc + b"?"
    budget = max(75 - len(head) - 2, 8)
    is_b = enc in (b"B", b"b")
    words: list[bytes] = []
    cur = bytearray()
    for ch in text:
        cb = _encode(ch, codec)
        if is_b:
            new_len = ((len(cur) + len(cb) + 2) // 3) * 4
        else:
            new_len = len(_q_encode(bytes(cur) + cb))
        if cur and new_len > budget:
            words.append(bytes(cur))
            cur = bytearray()
        cur += cb
    if cur or not words:
        words.append(bytes(cur))
    if is_b:
        payloads = [base64.b64encode(w) for w in words]
    else:
        payloads = [_q_encode(w) for w in words]
    return (eol + b" ").join(head + p + b"?=" for p in payloads)


# --------------------------------------------------------------------------- #
# public structures
# --------------------------------------------------------------------------- #
@dataclass
class Slot:
    """A view unit: text for masking and the map back."""
    key: str
    kind: str            # header | text | html | envelope
    text: str
    part: str            # part path: "" — root, "1.2" — as in email.walk
    header: str | None   # header name (kind == header); None — a header block
    #                      line without "Name:" (key ``./[n]``, see defects)
    encoding: str        # Q | B | mixed | raw | ascii | quoted-printable | base64 | 7bit | 8bit | binary
    charset: str | None


@dataclass
class Part:
    """Part metadata (leaf or container); attachments only here."""
    path: str
    content_type: str
    charset: str | None
    encoding: str
    disposition: str | None
    filename: str | None
    size: int
    in_view: bool
    reason: str = ""     # why not in the view: attachment / binary / multipart / …


# --------------------------------------------------------------------------- #
# assembly segments
# --------------------------------------------------------------------------- #
class _Seg:
    """A mutable range of the original bytes [start, end) with the view text."""
    __slots__ = ("start", "end", "text", "slot")

    def render(self, buf: bytes, new: str) -> bytes:  # pragma: no cover - abstract
        raise NotImplementedError


class _EnvSeg(_Seg):
    """mbox envelope: bytes after ``From `` up to the end of the line."""

    def __init__(self, start, end, text):
        self.start, self.end, self.text, self.slot = start, end, text, None

    def render(self, buf, new):
        if new == self.text:
            return buf[self.start:self.end]
        return _encode(_CRLF_RE.sub(" ", new), "utf-8")


class _Chunk:
    __slots__ = ("kind", "start", "end", "view", "charset", "enc", "codec", "lead")

    def __init__(self, kind, start, end, view, charset=None, enc=None, codec=None, lead=0):
        self.kind, self.start, self.end, self.view = kind, start, end, view
        self.charset, self.enc, self.codec = charset, enc, codec
        self.lead = lead      # ew: whitespace bytes (incl. folding) before the first word


class _HdrSeg(_Seg):
    """A header value: chunks (runs of encoded-words of one charset, and literals)
    in source order; the changed text is spread over the chunks by a
    diff map, and each changed chunk is re-encoded in its own form."""

    def __init__(self, start, end, chunks, eol=b"\r\n"):
        self.start, self.end, self.chunks, self.slot = start, end, chunks, None
        self.eol = eol
        self.text = "".join(c.view for c in chunks)

    def render(self, buf, new):
        if new == self.text:
            return buf[self.start:self.end]
        new = _CRLF_RE.sub(" ", new)
        views = _remap([c.view for c in self.chunks], self.text, new)
        self._keep_ws_edges(views)
        out = bytearray()
        for c, v in zip(self.chunks, views):
            if v == c.view:
                out += buf[c.start:c.end]
            elif c.kind == "ew":
                if v:
                    charset, codec = c.charset, c.codec
                    if not _encodable(v, codec):
                        # text not representable in the word's charset (the mask
                        # wrote Cyrillic into a latin-1 word): the word goes to utf-8,
                        # not "?" — the charset is in the word itself
                        charset, codec = b"utf-8", "utf-8"
                    out += buf[c.start:c.start + c.lead]
                    out += _encode_words(v, charset, c.enc, codec, self.eol)
            else:
                out += _encode(v, "utf-8")
        return bytes(out)

    def _keep_ws_edges(self, views: list[str]) -> None:
        """A literal that bordered an encoded-word through whitespace must, after
        the diff map, still start/end with whitespace (RFC 2047: a word is
        separated by whitespace): a non-whitespace edge moves into the neighbouring
        word ("MASK_" + "83" → "MASK_83" in the word). A word that became empty
        disappears — nothing is shifted into it ("<EMAIL_REDACTED>" stays a
        literal). The concatenation does not change."""
        cs = self.chunks
        for i, c in enumerate(cs):
            if c.kind != "lit":
                continue
            v = views[i]
            if i > 0 and cs[i - 1].kind == "ew" and views[i - 1] \
                    and c.view[:1].isspace():
                j = 0
                while j < len(v) and not v[j].isspace():
                    j += 1
                if j:
                    views[i - 1] += v[:j]
                    v = v[j:]
            if i + 1 < len(cs) and cs[i + 1].kind == "ew" and views[i + 1] \
                    and c.view[-1:].isspace():
                j = len(v)
                while j > 0 and not v[j - 1].isspace():
                    j -= 1
                if j < len(v):
                    views[i + 1] = v[j:] + views[i + 1]
                    v = v[:j]
            views[i] = v


class _BodySeg(_Seg):
    """Body of a text part: QP/base64/transparent, the part's charset."""

    def __init__(self, start, end, text, cte, codec, eol):
        self.start, self.end, self.text, self.slot = start, end, text, None
        self.cte, self.codec, self.eol = cte, codec, eol

    def render(self, buf, new):
        if new == self.text:
            return buf[self.start:self.end]
        if self.text.endswith(("\n", "\r")) and not new.endswith(("\n", "\r")):
            # the trailing line break of the body is the part boundary: in an mbox
            # without a blank line between mails it separates the body from the next
            # "From "; a mask that removed it (whitespace trimming) would glue mails — restore it
            new += "\r\n" if self.text.endswith("\r\n") else self.text[-1]
        data = _encode(new, self.codec)
        if self.cte == "quoted-printable":
            out = binascii.b2a_qp(data, istext=True)
            if self.eol == b"\r\n":
                out = out.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            return out
        if self.cte == "base64":
            out = base64.encodebytes(data).replace(b"\n", self.eol)
            raw = buf[self.start:self.end]
            if raw and not raw.endswith((b"\n", b"\r")):
                out = out[:-len(self.eol)] if out.endswith(self.eol) else out
            return out
        return data


def _remap(views: list[str], old: str, new: str) -> list[str]:
    """Spread the changed text over the chunks: old position → new position by
    the matching difflib blocks; a replacement that falls on a chunk joint goes
    to the chunk where it started. The concatenation of the result == new always."""
    # pos[i] — the new position for old position i; set once (by the first
    # block that sees it): otherwise an insertion at the start was lost — the equal
    # after it moved pos[0] past the inserted text. The edges are fixed:
    # pos[0] = 0, pos[len] = len(new) — the concatenation of chunks == new.
    pos: list = [None] * (len(old) + 1)
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2 + 1):
                if pos[i] is None:
                    pos[i] = j1 + (i - i1)
        else:
            if pos[i1] is None:
                pos[i1] = j1
            for i in range(i1 + 1, i2 + 1):
                pos[i] = j2
    pos[0] = 0
    pos[len(old)] = len(new)
    out, a = [], 0
    for v in views:
        b = a + len(v)
        out.append(new[pos[a]:pos[b]])
        a = b
    return out


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _iter_lines(buf: bytes, start: int, end: int):
    """(line start, content end without EOL, line end with EOL)."""
    pos = start
    while pos < end:
        nl = buf.find(b"\n", pos, end)
        if nl < 0:
            yield pos, end, end
            return
        ce = nl - 1 if nl > pos and buf[nl - 1] == 0x0D else nl
        yield pos, ce, nl + 1
        pos = nl + 1


def _eol_of(buf, start, end) -> bytes:
    """Line break of block [start, end) by its first line; no line break — LF."""
    nl = buf.find(b"\n", start, end)
    if nl < 0:
        return b"\n"
    return b"\r\n" if nl > start and buf[nl - 1] == 0x0D else b"\n"


def _header_block(buf, start, end):
    """→ (end of the header block, body start): a blank line is the separator."""
    for ls, ce, le in _iter_lines(buf, start, end):
        if ce == ls:
            return ls, le
    return end, end


class _Field:
    __slots__ = ("name", "start", "vstart", "end", "cend")

    def __init__(self, name, start, vstart, end, cend):
        self.name, self.start, self.vstart = name, start, vstart
        self.end, self.cend = end, cend       # end — after EOL, cend — before EOL


def _parse_fields(buf, start, hdr_end) -> list[_Field]:
    fields: list[_Field] = []
    for ls, ce, le in _iter_lines(buf, start, hdr_end):
        line = buf[ls:ce]
        if line[:1] in (b" ", b"\t") and fields:
            fields[-1].end, fields[-1].cend = le, ce
            continue
        m = _FIELD_NAME.match(line)
        if m and line[m.end():m.end() + 1] == b":":
            fields.append(_Field(line[:m.end()].decode("ascii"), ls,
                                 ls + m.end() + 1, le, ce))
        else:
            fields.append(_Field(None, ls, ls, le, ce))     # garbage line — as is
    return fields


def _header_chunks(buf, v0, v1) -> list[_Chunk]:
    """Header value [v0, v1) → chunks: groups of adjacent encoded-words
    (whitespace between them drops out of the view per RFC 2047), one chunk per
    run of one charset — the run's bytes are joined before decoding (a character
    split across words stays whole), on assembly the run is re-encoded in its
    charset; and literals (folding removed: EOL dropped, the continuation space stays)."""
    v = buf[v0:v1]
    toks = []
    pos = 0
    for m in _EW.finditer(v):
        dec = _decode_ew(m.group(2), m.group(3))
        if dec is None:
            continue
        if m.start() > pos:
            toks.append(("lit", pos, m.start(), None))
        toks.append(("ew", m.start(), m.end(), (m.group(1), m.group(2), dec)))
        pos = m.end()
    if pos < len(v):
        toks.append(("lit", pos, len(v), None))
    groups: list = []                 # ["ew", [(s, e, cs, enc, dec)]] | ["lit", s, e]
    held = None                       # whitespace literal after an ew group
    for kind, s, e, extra in toks:
        if kind == "ew":
            if groups and groups[-1][0] == "ew" and (held is not None or
                                                      groups[-1][1][-1][1] == s):
                groups[-1][1].append((s, e) + extra)
            else:
                if held is not None:
                    groups.append(["lit", held[0], held[1]])
                groups.append(["ew", [(s, e) + extra]])
            held = None
            continue
        if held is not None:
            groups.append(["lit", held[0], held[1]])
            held = None
        if groups and groups[-1][0] == "ew" and _WS_ONLY.match(v, s, e):
            held = (s, e)
        else:
            groups.append(["lit", s, e])
    if held is not None:
        groups.append(["lit", held[0], held[1]])
    chunks: list[_Chunk] = []
    for g in groups:
        if g[0] == "lit":
            raw = _FOLD.sub(b"", v[g[1]:g[2]])
            chunks.append(_Chunk("lit", v0 + g[1], v0 + g[2],
                                 _decode(raw, "utf-8")))
            continue
        # runs of one charset: [key, charset, enc, start of the first word, end
        # of the last, bytes]; the run's charset and Q/B come from its first word
        runs: list = []
        for s, e, cs, enc, dec in g[1]:
            key = cs.split(b"*", 1)[0].lower()
            if runs and runs[-1][0] == key:
                runs[-1][4] = e
                runs[-1][5] += dec
            else:
                runs.append([key, cs, enc, s, e, bytearray(dec)])
        prev_end = runs[0][3]
        for _key, cs, enc, s, e, data in runs:
            codec, _ok = _codec(cs)
            # whitespace (folding) between runs goes at the start of the next chunk: on
            # re-encoding it comes out before the words, as it was
            chunks.append(_Chunk("ew", v0 + prev_end, v0 + e, _decode(bytes(data), codec),
                                 cs, enc, codec, lead=s - prev_end))
            prev_end = e
    return chunks


def _cte_of(msg) -> str:
    cte = (msg.get("Content-Transfer-Encoding") or "").split(";", 1)[0]
    return cte.strip().strip('"').strip().lower()


def _filename(msg):
    name = msg.get_filename()
    if name and "=?" in name:
        try:
            name = str(email.header.make_header(email.header.decode_header(name)))
        except Exception:  # noqa: BLE001 — metadata, do not fail
            pass
    return name


class _Builder:
    """Parsing state of one mail: segments, slots, parts, defects."""

    def __init__(self, buf: bytes, attachments: bool):
        self.buf = buf
        self.attachments = attachments
        self.segs: list[_Seg] = []
        self.slots: list[Slot] = []
        self.parts: list[Part] = []
        self.defects: list[str] = []

    def _add(self, seg: _Seg, slot: Slot) -> None:
        seg.slot = slot
        self.segs.append(seg)
        self.slots.append(slot)
        if _SURROGATE.search(slot.text):
            # bytes outside the part's charset (a utf-8 label on cp1251 bytes, an 8-bit
            # header without encoded-word): in the view as surrogateescape surrogates,
            # detectors do not see such text; the caller decides by the defect
            self.defects.append(f"{slot.key}: bytes outside charset — in the view as surrogates "
                                "U+DC80..DCFF (surrogateescape), text not parsed")

    # ---- a mail (possibly with an envelope line) ----
    def message(self, start: int, end: int, path: str = "") -> None:
        buf = self.buf
        if buf.startswith(_BOM, start, end):
            # an export BOM before the first line: outside the view (a gap, byte for
            # byte), otherwise the first header / envelope would become a garbage line
            self.defects.append(f"{path or '.'}: BOM before the mail — outside the view, kept")
            start += len(_BOM)
        if buf[start:start + 5] == b"From ":
            ls, ce, le = next(_iter_lines(buf, start, end))
            text = _decode(buf[ls + 5:ce], "utf-8")
            self._add(_EnvSeg(ls + 5, ce, text),
                      Slot("From_", "envelope", text, path, None, "raw", None))
            start = le
        self.entity(start, end, path, "text/plain")

    # ---- a MIME entity: headers + body ----
    def entity(self, start: int, end: int, path: str, default_ct: str) -> None:
        buf = self.buf
        hdr_end, body_start = _header_block(buf, start, end)
        fields = _parse_fields(buf, start, hdr_end)
        # stdlib reads header semantics only from "Name: …" lines (with their
        # continuations): on a garbage line its parser ends the headers,
        # and a Content-Type after it would be lost
        msg = email.parser.BytesHeaderParser(policy=email.policy.compat32) \
            .parsebytes(b"".join(buf[f.start:f.end] for f in fields if f.name is not None))
        msg.set_default_type(default_ct)
        ct = msg.get_content_type()
        multipart = ct.startswith("multipart/")
        cte = _cte_of(msg)
        charset = msg.get_content_charset()
        disposition = msg.get_content_disposition()
        eol = _eol_of(buf, start, hdr_end)       # for folding changed headers
        counts: dict[str, int] = {}
        junk = 0
        for f in fields:
            v0, v1 = f.vstart, f.cend
            while v0 < v1 and buf[v0] in (0x20, 0x09):
                v0 += 1
            while v1 > v0 and buf[v1 - 1] in (0x20, 0x09):
                v1 -= 1
            if f.name is None:
                # a header block line without "Name:" and without indentation (a line
                # break without indentation, export garbage): stdlib starts the body
                # from it; here it is in the view as a nameless header (header=None),
                # otherwise the text in it would bypass the mask
                key = f"{path or '.'}/[{junk}]"
                junk += 1
                self.defects.append(f"{key}: header line without \"Name:\" — " + (
                    "whitespace, outside the view" if v0 == v1 else "in the view as is"))
            else:
                low = f.name.lower()
                n = counts.get(low, 0)
                counts[low] = n + 1
                if low in STRUCTURAL_HEADERS:
                    continue
                if low == "content-type" and (multipart or msg.get_param("name") is None):
                    continue              # markup; into the view only with legacy name=
                key = f"{path or '.'}/{f.name}[{n}]"
            if v0 == v1:
                continue
            chunks = _header_chunks(buf, v0, v1)
            seg = _HdrSeg(v0, v1, chunks, eol)
            encs = [c.enc.upper().decode("ascii") for c in chunks if c.kind == "ew"]
            if not encs:
                enc = "raw" if any(b > 0x7E for b in buf[v0:v1]) else "ascii"
                cs = None
            else:
                enc = encs[0] if len(set(encs)) == 1 else "mixed"
                cs = next(c.charset for c in chunks if c.kind == "ew") \
                    .decode("ascii", "replace")
            self._add(seg, Slot(key, "header", seg.text, path, f.name, enc, cs))
        size = end - body_start
        if multipart:
            boundary = msg.get_boundary()
            if boundary is None:
                self._opaque(path, ct, charset, cte, disposition, msg, size,
                             "multipart without boundary")
                return
            if not self.multipart(body_start, end, path,
                                  boundary.encode("ascii", "replace"),
                                  ct == "multipart/digest"):
                self._opaque(path, ct, charset, cte, disposition, msg, size,
                             "multipart without boundary lines")
                return
            self.parts.append(Part(path, ct, charset, cte or "7bit", disposition,
                                   None, size, False, "multipart"))
            return
        if ct in _NESTED:
            if cte not in _TRANSPARENT_CTE:
                self._opaque(path, ct, charset, cte, disposition, msg, size,
                             f"nested mail under {cte}")
                return
            self.parts.append(Part(path, ct, charset, cte or "7bit", disposition,
                                   _filename(msg), size, False, "message"))
            self.entity(body_start, end, path + ".1" if path else "1", "text/plain")
            return
        textual = ct.startswith("text/") or ct in _TEXTUAL_MESSAGE
        if not textual:
            self._opaque(path, ct, charset, cte, disposition, msg, size, "binary")
            return
        if disposition == "attachment" and not self.attachments:
            self._opaque(path, ct, charset, cte, disposition, msg, size, "attachment")
            return
        if cte not in _CTE_OK:
            self._opaque(path, ct, charset, cte, disposition, msg, size,
                         f"encoding {cte}")
            return
        raw = buf[body_start:end]
        data = _decode_body(raw, cte)
        if data is None:
            self._opaque(path, ct, charset, cte, disposition, msg, size,
                         f"{cte} does not decode")
            return
        codec, known = _codec(charset)
        if not known:
            self.defects.append(f"{path or '.'}: charset {charset!r} is unknown — "
                                "read as utf-8")
        if cte in _TRANSPARENT_CTE and codec != "utf-8" and not data.isascii():
            # intake re-encodes the work/ copy of an 8-bit mail that is not UTF-8
            # wholly into UTF-8, while the "charset=windows-1251" header stays — a body
            # under a transparent CTE was read with the header's codec and gave
            # garbage (a name invisible to rules and gates). Strict UTF-8 first,
            # like the walk.decode_bytes ladder: non-ASCII cp1251/koi8-r practically
            # never reads as strict UTF-8 (consecutive bytes ≥ 0xC0 are not continuations),
            # so the original bytes (raw/, mine) go by the part's charset, as before.
            # Assembly uses the same codec it was read with (work/ stays UTF-8;
            # release restores the original form from the intake ledger).
            try:
                text = data.decode("utf-8")
                codec = "utf-8"
            except UnicodeDecodeError:
                text = _decode(data, codec)
        else:
            text = _decode(data, codec)
        kind = "html" if ct == "text/html" else "text"
        self.parts.append(Part(path, ct, charset, cte or "7bit", disposition,
                               _filename(msg), size, True))
        if not raw:
            return
        eol = b"\r\n" if b"\r\n" in raw else b"\n"
        self._add(_BodySeg(body_start, end, text, cte, codec, eol),
                  Slot(f"{path or '.'}/body", kind, text, path, None,
                       cte or "7bit", charset))

    def _opaque(self, path, ct, charset, cte, disposition, msg, size, reason):
        self.parts.append(Part(path, ct, charset, cte or "7bit", disposition,
                               _filename(msg), size, False, reason))
        if reason not in ("attachment", "binary", "multipart"):
            self.defects.append(f"{path or '.'}: {reason} — part not in the view")

    # ---- multipart: parts between boundary lines ----
    def multipart(self, start, end, path, boundary, digest) -> bool:
        buf = self.buf
        dash = b"--" + boundary
        delims = []                       # (line_start, line_end, closing)
        for ls, ce, le in _iter_lines(buf, start, end):
            if not buf.startswith(dash, ls, ce):
                continue
            rest = buf[ls + len(dash):ce]
            closing = rest.startswith(b"--")
            if closing:
                rest = rest[2:]
            if rest.strip(b" \t"):
                continue
            delims.append((ls, le, closing))
            if closing:
                break
        if not delims:
            return False
        default_ct = "message/rfc822" if digest else "text/plain"
        n = 0
        for i, (ls, le, closing) in enumerate(delims):
            if closing:
                break
            if i + 1 < len(delims):
                nxt = delims[i + 1][0]
                if buf[nxt - 2:nxt] == b"\r\n":
                    pend = nxt - 2
                elif buf[nxt - 1:nxt] == b"\n":
                    pend = nxt - 1
                else:
                    pend = nxt
                pend = max(pend, le)
            else:
                pend = end                # no closing boundary — up to the end
            n += 1
            self.entity(le, pend, f"{path}.{n}" if path else str(n), default_ct)
        return True


def _decode_body(raw: bytes, cte: str):
    if cte == "quoted-printable":
        try:
            return binascii.a2b_qp(raw)
        except (binascii.Error, ValueError):
            return None
    if cte == "base64":
        # foreign characters = not base64 (b64decode without validate silently swallows
        # them); broken padding is repaired
        s = re.sub(rb"\s", b"", raw)
        if s and not re.fullmatch(rb"[A-Za-z0-9+/]+=*", s):
            return None
        try:
            return base64.b64decode(s, validate=True)
        except (binascii.Error, ValueError):
            pass
        s = s.rstrip(b"=")
        if len(s) % 4 == 1:
            s = s[:-1]
        try:
            return base64.b64decode(s + b"=" * (-len(s) % 4), validate=True)
        except (binascii.Error, ValueError):
            return None
    return raw


def _check_segs(segs: list[_Seg], n: int) -> None:
    pos = 0
    for s in segs:
        if not (pos <= s.start <= s.end <= n):
            raise MailParseError(f"spans diverged: {pos} > [{s.start}, {s.end}) at {n}")
        pos = s.end


# --------------------------------------------------------------------------- #
# public interface
# --------------------------------------------------------------------------- #
class Mail:
    """One mail: ``slots`` — the view, ``render(texts)`` — assembly, ``parts`` —
    part metadata (attachments only here), ``defects`` — what was not parsed."""

    def __init__(self, raw: bytes, attachments: bool = False, _range=None):
        self.raw = raw
        start, end = _range or (0, len(raw))
        b = _Builder(raw, attachments)
        b.message(start, end)
        _check_segs(b.segs, len(raw))
        self._segs = b.segs
        self._range = (start, end)
        self.slots: list[Slot] = b.slots
        self.parts: list[Part] = b.parts
        self.defects: list[str] = b.defects

    def render(self, texts=None) -> bytes:
        """Assembly: ``texts`` — {slot key: new text} (no key or the same
        text — the slot from the original bytes). Without an argument — the source."""
        texts = texts or {}
        buf, (pos, end) = self.raw, self._range
        out = bytearray()
        for seg in self._segs:
            out += buf[pos:seg.start]
            out += seg.render(buf, texts.get(seg.slot.key, seg.text))
            pos = seg.end
        out += buf[pos:end]
        return bytes(out)

    def mask(self, fn) -> bytes:
        """``fn(slot) -> str`` for each slot; assembly."""
        return self.render({s.key: fn(s) for s in self.slots})


class Mbox:
    """A sequence of mbox mails: ``From `` separators and blank lines
    between mails are preserved; slot keys are ``<mail number>:<key>``."""

    def __init__(self, raw: bytes, attachments: bool = False):
        self.raw = raw
        self.messages: list[Mail] = []
        self._gaps: list[tuple[int, int]] = []     # bytes between mails
        self._defects: list[str] = []              # file defects, not mail ones
        starts = [0] if raw.startswith((b"From ", _BOM + b"From ")) else []
        pos = 0
        while True:
            i = raw.find(b"\nFrom ", pos)
            if i < 0:
                break
            starts.append(i + 1)
            pos = i + 1
        first = starts[0] if starts else len(raw)
        no_envelope = False
        if first > 0:
            head = raw[:first]
            bom = head.startswith(_BOM)
            if head[len(_BOM) if bom else 0:].strip(b" \t\r\n"):
                # not whitespace before the first "From " line: a file with one mail
                # without an envelope, whitespace/garbage before the first From — otherwise
                # these bytes would go to the output bypassing the view; parse it as a mail
                # (Mail.message strips the BOM), a defect
                starts.insert(0, 0)
                no_envelope = True
            elif bom:
                self._defects.append("BOM before mbox — outside the view, kept")
        prev = 0
        for i, s in enumerate(starts):
            self._gaps.append((prev, s))
            e = starts[i + 1] if i + 1 < len(starts) else len(raw)
            # a blank line before the next mail is a separator, not body
            if i + 1 < len(starts):
                if raw[e - 2:e] == b"\n\n":
                    e -= 1
                elif raw[e - 3:e] == b"\n\r\n":
                    e -= 2
                e = max(e, s)
            m = Mail(raw, attachments, (s, e))
            if i == 0 and no_envelope:
                m.defects.insert(0, "mail without a \"From \" envelope line — parsed as a mail")
            self.messages.append(m)
            prev = e
        self._gaps.append((prev, len(raw)))

    @property
    def slots(self) -> list[Slot]:
        out = []
        for i, m in enumerate(self.messages):
            for s in m.slots:
                out.append(Slot(f"{i}:{s.key}", s.kind, s.text, s.part, s.header,
                                s.encoding, s.charset))
        return out

    @property
    def parts(self) -> list[tuple[int, Part]]:
        return [(i, p) for i, m in enumerate(self.messages) for p in m.parts]

    @property
    def defects(self) -> list[str]:
        return self._defects + [f"{i}:{d}" for i, m in enumerate(self.messages)
                                for d in m.defects]

    def render(self, texts=None) -> bytes:
        per: dict[int, dict] = {}
        for k, v in (texts or {}).items():
            i, _, key = k.partition(":")
            per.setdefault(int(i), {})[key] = v
        out = bytearray()
        for i, m in enumerate(self.messages):
            g0, g1 = self._gaps[i]
            out += self.raw[g0:g1]
            out += m.render(per.get(i))
        g0, g1 = self._gaps[-1]
        out += self.raw[g0:g1]
        return bytes(out)

    def mask(self, fn) -> bytes:
        return self.render({s.key: fn(s) for s in self.slots})


def kind_for_path(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".eml":
        return "eml"
    if ext in (".mbox", ".mbx"):
        return "mbox"
    return None


def parse_eml(raw: bytes, attachments: bool = False) -> Mail:
    return Mail(raw, attachments)


def parse_mbox(raw: bytes, attachments: bool = False) -> Mbox:
    return Mbox(raw, attachments)


def parse(raw: bytes, kind: str, attachments: bool = False):
    if kind == "eml":
        return parse_eml(raw, attachments)
    if kind == "mbox":
        return parse_mbox(raw, attachments)
    raise ValueError(f"unknown mail kind {kind!r}")


def mask_bytes(raw: bytes, kind: str, fn, attachments: bool = False) -> bytes:
    """Parse → ``fn(slot) -> str`` per slot → assemble. A parsing exception
    (MailParseError) propagates: the caller decides whether to leave the file as is."""
    return parse(raw, kind, attachments).mask(fn)
