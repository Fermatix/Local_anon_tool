"""gliner-x-large on top of the name model (``[ner].gliner``; extra ``gliner`` = gliner + torch).

Installed with ``python -m localanon.fetch --gliner``: a snapshot of ``knowledgator/gliner-x-large`` in
the models directory (``<models>/gliner-x-large``, localanon/fetch.py) with "whitespace" word splitting
instead of stanza — stanza downloads its own models from the network on first start. During a run the
model is loaded only from this directory (``local_files_only``, ``HF_HUB_OFFLINE``), without network.

"person" spans are merged with the name model spans (localanon/stages/anonymize.py). The
"organization" label is asked together with "person" so that company names do not end up as people.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

LABELS = ["person", "organization"]
MAX_WORDS = 200                   # piece of a line: gliner cuts input by words (max_len 384 tokens)
BATCH = 8
_WORDS = re.compile(r"\S+")
_MODEL: dict = {}


def missing_reason(root: str | os.PathLike | None = None) -> str:
    """Why gliner will not start (for ``ctx.note``); empty — everything is in place."""
    try:
        import gliner  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return ("no gliner or torch package (.venv/bin/python -m pip install -e \".[gliner]\" in the repository "
                "folder; Windows: .venv\\Scripts\\python.exe)")
    from . import fetch
    if fetch.gliner_path(root) is None:
        return f"model not downloaded to {fetch.models_dir(root)} (python -m localanon.fetch --gliner)"
    return ""


def _device(device: str) -> str:
    import torch
    d = (device or "auto").strip().lower()
    if d == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return "cuda:0" if d == "cuda" else d


def _load(path: Path, device: str):
    key = (str(path), device)
    if key not in _MODEL:
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[var] = "1"
        from gliner import GLiNER
        m = GLiNER.from_pretrained(str(path), local_files_only=True)
        if device != "cpu":
            m = m.to(device)             # fp32, as in the benchmarks
        _MODEL.clear()
        _MODEL[key] = m.eval()
    return _MODEL[key]


def pieces(text: str, max_words: int = MAX_WORDS) -> list[tuple[int, int]]:
    """Pieces [a, b) of a line, at most ``max_words`` words each (split by whitespace)."""
    ws = [(m.start(), m.end()) for m in _WORDS.finditer(text)]
    return [(ws[i][0], ws[min(i + max_words, len(ws)) - 1][1]) for i in range(0, len(ws), max_words)]


def spans(texts: list[str], threshold: float = 0.5, device: str = "auto", root=None,
          batch_size: int = BATCH, info: dict | None = None) -> list[list[tuple[int, int]]]:
    """Spans of "person" per line. No package or model — FileNotFoundError/ImportError (the caller
    writes a note from ``missing_reason``)."""
    import torch
    from . import fetch
    path = fetch.gliner_path(root)
    if path is None:
        raise FileNotFoundError("gliner-x-large is not downloaded")
    dev = _device(device)
    t0 = time.perf_counter()
    m = _load(path, dev)
    if info is not None:
        info.update(gliner_device=dev, gliner_load_s=round(time.perf_counter() - t0, 2))
    jobs = [(i, a, texts[i][a:b]) for i, t in enumerate(texts) for a, b in pieces(t)]
    jobs.sort(key=lambda j: len(j[2]))
    out: list[list[tuple[int, int]]] = [[] for _ in texts]
    run = getattr(m, "inference", None) or m.batch_predict_entities
    with torch.no_grad():
        for k in range(0, len(jobs), 64):
            chunk = jobs[k:k + 64]
            ents = run([j[2] for j in chunk], LABELS, threshold=threshold, batch_size=batch_size)
            for (i, a, _t), es in zip(chunk, ents):
                out[i].extend((a + e["start"], a + e["end"]) for e in es if e["label"] == "person")
    for sp in out:
        sp.sort()
    return out
