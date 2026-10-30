"""Stage order. Each stage is a module with ``NAME`` and ``run(ctx: RunContext) -> None``.

Module owners (agent packages, see docs/DESIGN.md):
  inventory, gates_struct  — la-core
  images, documents        — la-docs
  exports, messages        — la-chat
  transcribe               — la-asr
  anonymize, gates_control — la-mask
Order matters: images and "papers" go before the exports are parsed; transcription goes before the
filter and anonymization (the transcript passes both); checks go last.
"""
from __future__ import annotations

import importlib

ORDER = [
    "inventory",
    "images",
    "documents",
    "exports",
    "transcribe",
    "messages",
    "anonymize",
    "gates_struct",
    "gates_control",
]


def load(name: str):
    return importlib.import_module(f"{__name__}.{name}")
