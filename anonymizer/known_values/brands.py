"""EXAMPLE optional known-values pack.

The general pipeline works with NO corpus knowledge. When you repeatedly process
a *familiar* corpus, you can pin its exact brand / client names here so all their
spelling/ASR/typo variants collapse to one canonical token with high precision.
Enable with:  --known brands   (or config["known_values"] = ["brands"]).

This is just a template — replace with your own entities. It reuses the general
``orgs.brand_variation`` factory, so no bespoke regex logic lives here, only data.
"""
from __future__ import annotations

from ..detectors.orgs import brand_variation

# Each entry (spec dict consumed by orgs.brand_variation):
#   token_seed:    canonical key -> mint.tag('ORG', token_seed); one token for all forms
#   latin:         exact case-insensitive whole-word literals
#   cyr_stems:     Cyrillic inflected stems (masked with CL..CR + [а-яё]*)
#   require_substr: a discriminating substring a Cyrillic stem MUST contain (FP-guard)
#   caps_only:     if True, latin literals match only in the exact case given
BRANDS = [
    {
        "token_seed": "acme",
        "latin": ["Acme", "AcmeCorp", "Acme-Corp"],
        "cyr_stems": ["акмэ", "акме"],
        "require_substr": None,
    },
    # add your real brands here, e.g.:
    # {"token_seed": "acme", "latin": ["Acme", "AcmeSoft"],
    #  "cyr_stems": ["акме", "акми"], "require_substr": "акм"},
]

RULES = []
for b in BRANDS:
    RULES.extend(brand_variation(b))
