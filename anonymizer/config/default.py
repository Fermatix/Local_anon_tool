"""Default pipeline configuration.

- ``detectors``   : which detector modules run.
- ``confidence``  : which rule-confidence tiers fire. Default ("high","context")
                    is the safe production setting — "low" rules (bare entropy,
                    global common-surname sweeps) over-mask and are opt-in.
- ``owner``       : owner-decision toggles. Every accepted key is either
                    consulted by the engine (mask_crm_columns / crm_columns) or
                    wired/validated by ``load()``: mask_snowflake_ids switches
                    the ``linking`` detector on; mask_generic_first_names rides
                    the "low" confidence tier (its real switch). A key that
                    would be silently ignored is REJECTED — see _OWNER_KEYS.
- ``known_values``: optional corpus-specific literal packs (empty by default →
                    the pipeline is fully general and works on unfamiliar docs).
- ``stream_threshold_bytes``: .jsonl files LARGER than this are masked line by
                    line (streaming, E2-05/06) instead of whole-file in memory;
                    output and Report counters are identical either way.
- ``rule_tiers``  : per-rule tier override ``{rule_name: "high"|"context"|"low"}``
                    (P2-5). Wins over ``Rule.confidence`` in apply_rules, so one
                    rule can be promoted/demoted without touching a detector.
                    An unknown rule name or tier fails fast in ``load()``.
"""
CONFIG = {
    "detectors": {
        "secrets": True,
        "encoding": True,
        "discord_ids": True,  # snowflake user-id → USER_ID
        "financial": True,
        "contacts": True,
        "addresses": True,
        "orgs": True,
        "names": True,
        "special": False,   # health/ethnicity ties — opt in explicitly
        "linking": False,   # HMAC-pseudonymise snowflake/thread ids — opt in explicitly
    },
    "confidence": ("high", "context"),   # add "low" for maximum recall (more FP)
    "owner": {
        "mask_generic_first_names": False,  # bare common-first-name sweep. The
                                         # real switch is the "low" tier (the
                                         # sweep is names.bare_first_name_low,
                                         # a 'low' rule; the engine has no
                                         # per-rule gate) — load() rejects True
                                         # without "low" in confidence.
        "mask_snowflake_ids": False,     # structural linkage ids. True switches
                                         # the ``linking`` detector ON (keyed
                                         # HMAC pseudonyms, same id shape) —
                                         # wired in load(), never a no-op.
        "mask_crm_columns": False,       # CRM/registry company COLUMN masking — opt-in.
                                         # auto-detect over-masks ambiguous headers
                                         # (e.g. "Name" = product, not company),
                                         # so name the real columns explicitly:
        "crm_columns": [],               # e.g. ["Company","Counterparty"] — empty =>
                                         #      auto-detect (aggressive; unfamiliar docs)
    },
    "known_values": [],   # e.g. ["brands", "clients"]
    # Russian trace (owner decision: every Russian trace is masked, with no exceptions, in
    # ALL anonymization): the dictionary anonymizer/known_values/ru_trace.py — Russian
    # companies/services/tools, geography, government bodies, universities — "always mask"
    # rules that bypass verdicts and gold. Env overrides: ANON_RU_TRACE=0 (emergency off),
    # ANON_RU_TRACE_CURRENCY=1 (rouble amounts), ANON_RU_TRACE_LAWS=1 (Russian laws, GOST).
    "ru_trace": {"enabled": True, "currency": False, "laws": False},
    # .jsonl LARGER than this threshold (in bytes) is processed line by line
    # (streaming): memory O(line) instead of O(file); output and Report counters
    # are byte-for-byte the same as in the in-memory branch. 0 = stream every
    # non-empty .jsonl; a huge value = streaming effectively off.
    "stream_threshold_bytes": 50 * 1024 * 1024,
    # per-rule tier {rule_name: "high"|"context"|"low"} on top of
    # Rule.confidence (e.g. {"phone_intl": "high"}). An unknown name or tier is
    # a SystemExit in load() (_finalize), like owner keys / known_values.
    "rule_tiers": {},
}

# ---- owner-key contract ------------------------------------------------------
# Accepted owner keys. Each one demonstrably DOES something — it is consulted by
# the engine or wired/validated in ``_finalize`` below. A silently ignored owner
# toggle is a missed-masking trap (the owner believes a data class is handled
# while nothing runs), so ``load()`` rejects everything else: the same
# fail-loudly discipline as the unknown-pack / missing-detector checks in the
# pipeline.
_OWNER_KEYS = frozenset({
    "mask_generic_first_names",   # validated: rides the "low" confidence tier
    "mask_snowflake_ids",         # wired: True forces detectors.linking on
    "mask_public_figures",        # validated: True only (no fame exemption exists,
                                  #   matching names are always masked anyway)
    "mask_crm_columns",           # consulted by Pipeline (CSV company columns)
    "crm_columns",                # consulted by Pipeline (explicit column names)
})
# Retired keys that never did anything -> the real behaviour, so an old config
# fails fast with a pointer instead of silently shipping data the owner
# believed was handled.
_OWNER_RETIRED = {
    "mask_salaries":
        "no salary detector exists; commercial figures are never masked — "
        "remove the key",
    "mask_oss_handles":
        "handles are ALWAYS masked when the names detector runs (handle_at / "
        "handle_label / linkedin_slug rules); there is no OSS-handle exemption "
        "to toggle — remove the key",
}


_TIERS = ("high", "context", "low")


def _known_rule_names(detectors, packs):
    """Rule names of every built-in detector listed under ``detectors`` (enabled
    or not: a tier for a rule of a disabled detector is not a config error) plus
    the ``known_values`` packs. Imported lazily — only when ``rule_tiers`` is
    non-empty; a missing module is skipped here (the Pipeline reports it)."""
    import importlib
    names = set()
    mods = [f"anonymizer.detectors.{d}" for d in detectors] + \
           [f"anonymizer.known_values.{p}" for p in packs]
    for mod_name in mods:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        names.update(r.name for r in getattr(mod, "RULES", []))
    return names


def _ru_trace_rule_names(cfg):
    rt = cfg.get("ru_trace") or {}
    if not rt.get("enabled"):
        return set()
    from ..known_values import ru_trace as _rt
    return {r.name for r in _rt.rules(currency=rt.get("currency"), laws=rt.get("laws"))}


def _check_rule_tiers(cfg):
    """``rule_tiers`` (P2-5) fails fast like the owner block: a mistyped rule
    name or tier would be silently ignored — the owner believes a rule is
    promoted/demoted while nothing changes. External pack rules (``pack:…``,
    loaded from ``known_files`` by the Pipeline) cannot be checked here."""
    tiers = cfg.get("rule_tiers") or {}
    if not isinstance(tiers, dict):
        raise SystemExit("config: rule_tiers must be a dict {rule_name: tier}")
    bad = {k: v for k, v in tiers.items() if v not in _TIERS}
    if bad:
        raise SystemExit(f"config: rule_tiers: unknown tier(s) {bad}; valid: "
                         f"{', '.join(_TIERS)}")
    if tiers:
        known = _known_rule_names(cfg.get("detectors") or {},
                                  cfg.get("known_values") or []) | _ru_trace_rule_names(cfg)
        unknown = sorted(k for k in tiers
                         if k not in known and not k.startswith("pack:"))
        if unknown:
            raise SystemExit(f"config: rule_tiers: unknown rule(s) {unknown} "
                             f"(the tier would be silently ignored — a "
                             f"missed-masking trap); known rules: "
                             f"{sorted(known)}")
    return dict(tiers)


def _finalize(cfg):
    """Validate the owner block and wire its toggles into their real switches.

    Runs on EVERY config ``load()`` returns (default or user-merged), so an
    owner decision can never be a silent no-op:

    * unknown / retired keys under ``owner`` fail fast (a typo here means the
      owner believes a data class is handled while nothing runs);
    * ``mask_snowflake_ids: True`` forces the ``linking`` detector on — that
      detector IS the snowflake/thread-id switch (keyed HMAC pseudonyms, same
      shape/length, reply-chains survive);
    * ``mask_generic_first_names: True`` requires the "low" confidence tier:
      the bare first-name sweep (``names.bare_first_name_low``) is a 'low'
      rule and the engine has no per-rule gate, so without "low" the toggle
      would silently mask nothing;
    * ``mask_public_figures``: no fame exemption is implemented — the name
      rules mask every matching name regardless — so only ``True`` (which
      matches actual behaviour) is accepted; ``False`` promises an exemption
      nothing implements.
    """
    owner = dict(cfg.get("owner") or {})
    for key in owner:
        if key in _OWNER_RETIRED:
            raise SystemExit(f"config: owner.{key} is retired: "
                             f"{_OWNER_RETIRED[key]}")
        if key not in _OWNER_KEYS:
            raise SystemExit(f"config: unknown owner key {key!r} (it would be "
                             f"silently ignored — a missed-masking trap); "
                             f"known keys: {sorted(_OWNER_KEYS)}")
    if owner.get("mask_public_figures") is False:
        raise SystemExit(
            "config: owner.mask_public_figures=False asks for a public-figure "
            "exemption that is not implemented — matching names are always "
            "masked. Remove the key (True matches actual behaviour).")
    if owner.get("mask_generic_first_names") and \
            "low" not in tuple(cfg.get("confidence", ())):
        raise SystemExit(
            "config: owner.mask_generic_first_names=True needs the 'low' "
            "confidence tier — the bare first-name sweep "
            "(names.bare_first_name_low) is a 'low' rule, so without it the "
            "toggle silently masks nothing. Add \"low\" to confidence.")
    cfg["owner"] = owner
    # wire the snowflake toggle into its real switch. The sub-dict is copied
    # first so the module-level CONFIG is never mutated through an alias.
    detectors = dict(cfg.get("detectors") or {})
    if owner.get("mask_snowflake_ids"):
        detectors["linking"] = True
    cfg["detectors"] = detectors
    # Russian trace: env knobs are resolved HERE so the final value gets into the
    # round's config_sha256 (env bypassing the config would falsely reuse a result)
    from ..known_values import ru_trace as _rt
    cfg["ru_trace"] = _rt.config_from_env(cfg.get("ru_trace"))
    cfg["rule_tiers"] = _check_rule_tiers(cfg)     # P2-5: fail fast on typos
    return cfg


def load(path=None):
    """Load config from a python/JSON file, else return the default. Every
    returned config is validated + wired by ``_finalize`` (owner toggles)."""
    if not path:
        return _finalize(dict(CONFIG))
    import json
    import os
    if path.endswith(".json"):
        with open(path) as fh:
            user = json.load(fh)
    else:  # a python module exposing CONFIG
        import importlib.util
        spec = importlib.util.spec_from_file_location("usercfg", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        user = getattr(mod, "CONFIG", {})
    merged = dict(CONFIG)
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return _finalize(merged)
