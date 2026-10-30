"""Structured format profiles: an export is processed field by field.

Profile = signature + document spine + a "path → action" table. The engine
(engine.py) is shared; the profiles are Telegram Desktop (telegram.py, a .json document) and
YouTrack (youtrack.py, JSONL: line = issue). A new format is one more module
with a ``PROFILE`` object (``sniff(head)`` for .json, ``sniff_jsonl(head)`` for
.jsonl/.ndjson → profile | None) in the ``PROFILES`` tuple.

Pipeline entry points:
    detect(path, rel) / detect_text(rel, head)  — the file profile or None;
    mask_file(...)                              — Pipeline._run_file;
    plan_parts(...) / mask_part(...)            — a file in parts on different
                                                  workers (Pipeline);
    projection(text, profile, ner=…)            — tools/mine, tools/ner_local;
    projection_file(path, profile, ner=…)       — the same, streamed (JSONL);

    text_lines(path)                            — audit L0 (tools/audit_zones);
    gate_text(path, rel, text)                  — residue checks: control
                                                  gate, residual, archive gate;
    raw_chunks(lines)                           — raw text in chunks (gates
                                                  on a JSONL profile);
    registry_for_tree(root)                     — the run's entity registry.

CSV/TSV table registries (tabular.py) — a profile by columns, not by
JSON fields: profile_text / mask_text / blank_text; Pipeline calls it before
the rules on .csv/.tsv, mine and NER blank the cells of PII columns.
"""
from . import tabular, telegram, youtrack
from .engine import (STATS, LayoutError, PartPlan, apply_edits,
                     build_registry, detect, detect_layout, detect_text,
                     distribute, gate_text, is_jsonl, is_jsonl_file,
                     jsonl_profile_of, mask_file, mask_part, raw_chunks,
                     plan_parts, profiles_enabled, projection,
                     projection_file, registry_for_tree, text_lines)
from .entities import Registry, canonical, mirror_seed, person_parts

PROFILES = (telegram.PROFILE, youtrack.PROFILE)

# mining cohort: the run's known literals from identity fields
ROSTER_COHORT = "tg-roster"

__all__ = ["PROFILES", "ROSTER_COHORT", "STATS", "LayoutError", "PartPlan",
           "Registry", "apply_edits", "build_registry", "canonical", "detect",
           "detect_layout", "detect_text", "distribute", "gate_text",
           "is_jsonl_file", "raw_chunks",
           "is_jsonl", "jsonl_profile_of", "mask_file", "mask_part",
           "mirror_seed",
           "person_parts", "plan_parts", "profiles_enabled", "projection",
           "projection_file", "registry_for_tree", "tabular", "text_lines"]
