"""Container formats: a file is split into a text "view" for masking and
assembled back in its original encoding.

Unlike ``engine/formats.py`` (value passes over JSON/CSV leaves inside the
engine, the text is already UTF-8), these are formats whose text sits under a
transport encoding (quoted-printable, base64, RFC 2047 in headers) and is not
visible to detectors without decoding. The first format is ``mail``
(.eml / .mbox / .mbx).

Wiring: the engine (``Pipeline._run_mail``) masks every mail slot with the same
``_masked`` as plain text and assembles the file with ``doc.render`` in the
original encoding; the round control gate, residual, the archive gate and mine
read the mail as a "view" (``mail_text`` — slot texts joined by newlines). The
``MAIL_VIEW`` knob (``0`` — mail as plain text) has been removed. Intake is
unchanged.
"""
from __future__ import annotations

from .mail import MailParseError, kind_for_path, parse

__all__ = ["mail_kind", "mail_text", "mail_gate_text"]

_WARNED: set = set()        # rel paths that already got a parse WARN (once per file per process)


def mail_kind(rel: str):
    """``"eml"`` | ``"mbox"`` for mail, otherwise None — the single place that decides
    "this file is mail" for the engine, gates and mine."""
    return kind_for_path(rel)


HTML_HINT = "part.html"     # hint for the transport view of an html slot (the extension decides)


def _slot_line(s) -> str:
    text = s.text
    if s.kind == "header" and s.header is not None:
        text = f"{s.header}: {text}"
    elif s.kind == "html":
        # the engine masks an html slot through the transport view (.html hint:
        # &#…;/&nbsp; expanded); checks see the slot the same way, otherwise a name
        # written with entities in the HTML part would not exist for them
        from ..engine.vnorm import transport_view
        text = transport_view(HTML_HINT, text)[0]
    return text


def mail_text(raw: bytes, kind: str) -> str:
    """The mail as checks see it (control gate, residual, archive gate, mine):
    slot texts joined by ``"\\n"``, a header as ``"Name: value"`` (encoded-words
    decoded, folding removed), a text/* body decoded from QP/base64 in the part
    charset, text/html in the transport view (``vnorm.transport_view`` with the
    ``.html`` hint, as in the engine), the mbox envelope as is; attachments are
    not in the view. Bytes outside the part charset (surrogateescape surrogates
    in the slot) become U+FFFD, as with a plain read with ``errors="replace"``:
    the view goes into reports and logs without ``surrogateescape``.
    A parse error (``MailParseError`` and others) propagates: the caller reads
    the file as plain text."""
    doc = parse(raw, kind)
    view = "\n".join(_slot_line(s) for s in doc.slots)
    return view.encode("utf-8", "replace").decode("utf-8")


def mail_gate_text(path, rel, text=None):
    """The mail text AS RESIDUE CHECKS SEE IT (round control gate, residual),
    modelled on ``structured.gate_text``: ``None`` means the file is not mail (by
    ``rel``, else by ``path``) — read it as usual. ``text`` is unused (mail is
    bytes; the release archive member reads ``mail_text`` itself). A parse error
    gives ``None`` + a WARN once per file: the gate falls back to plain text.
    A read error (OSError) propagates, as on the plain path."""
    kind = mail_kind(rel if rel is not None else path)
    if kind is None:
        return None
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        return mail_text(raw, kind)
    except Exception as exc:               # noqa: BLE001 — a mail parse error must not break the gate
        key = rel if rel is not None else path
        if key not in _WARNED:
            _WARNED.add(key)
            why = "spans diverged" if isinstance(exc, MailParseError) else "parse failed"
            print(f"mail: WARN {key}: mail not parsed ({why}: {exc}) — "
                  "checking as plain text", flush=True)
        return None
