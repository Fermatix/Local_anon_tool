"""Hygiene of a candidate literal cut from RAW text.

NER and harvesters read raw .jsonl: an entity span may cross a JSONL escape
(«Acme\\nCEO», «Иван Ин\\n Мария Петрова») or catch the tail of an escape
(«…School\\» before «"»). An escape is a line/field boundary, not part of the entity:
a pack rule on such a literal eats the escape (breaks JSON or glues
lines), and a literal «\\n» gives thousands of false hits in the verify residue.

ONE function for all candidate sources: mine.Harvest.add, ner_local,
ner_rubert. Hygiene used to live only in Harvest.add, while NER layers write
candidates through their own _merge_candidates around it — a smoke run again saw
«Imperial College Business School\\» in candidates.tsv.

The escape set is as in engine/boundary._ESC: \\n \\t \\r \\b \\f \\v and \\uXXXX.
The module lives in anonymizer/engine/ — the directory is part of the code fingerprint of the
mine stamp (a hygiene change = recomputed candidates).
"""
import re

JSONL_ESC = re.compile(r"\\(?:[ntrbfv]|u[0-9A-Fa-f]{4})")


def literal_parts(literal: str) -> list:
    """Literal → list of clean parts: split on JSONL escapes, strip
    edge backslashes and spaces; empty parts (a literal of escapes/slashes
    only) disappear. A literal without «\\» is returned as is
    (fast path, order and content unchanged)."""
    if "\\" not in literal:
        return [literal]
    out = []
    for part in JSONL_ESC.split(literal):
        part = part.strip().strip("\\").strip()
        if part:
            out.append(part)
    return out
