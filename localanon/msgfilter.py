"""Message filter: a category for every message.

[messages].filter:
  auto  — the model (localanon/msgmodel.py) if ``msgmodel.available(model_dir)``, otherwise rules;
  model — the model; no model — rules and a note in the report;
  rules — rules only (localanon/msgrules.py);
  off   — everything is ``other`` (nothing is removed).
Empty text is ``other`` without classification.

Decision with the model: probabilities ``softmax(logits / T)`` with the temperature T of the
tier (``calibration`` in index.json), the "delete" probability ``p_drop`` — their sum over
[messages].drop_categories. ``p_drop >= threshold`` — the most likely drop category, otherwise the
most likely of the rest (the message stays). Rules are not consulted when the model works.
[messages].threshold: ``"auto"`` — the threshold of the tier (``calibration.threshold``, chosen on
val: wrongly deleted ``dev`` <= 10 %, the highest delete recall; per-category ``thresholds`` if the
tier has them), without calibration — ``FALLBACK_THRESHOLD``; a number in (0, 1) — that threshold
on ``p_drop``. ``0.5`` — the default of earlier versions, where it meant "argmax probability >= 0.5,
else rules" — is read as ``"auto"`` with a note (``LEGACY_THRESHOLD``).
"Doubtful" — kept messages with ``p_drop`` in ``[threshold - DOUBT_BAND, threshold)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import msgrules
from .config import MSG_CATEGORIES

BATCH = 64
FALLBACK_THRESHOLD = 0.5      # tier without calibration (models exported by older versions)
LEGACY_THRESHOLD = 0.5        # [messages].threshold of the config template in older versions
DOUBT_BAND = 0.15


@dataclass
class Msg:
    text: str
    context: list[str] = field(default_factory=list)   # 1–3 previous messages of the same chat
    chat_kind: str = ""                                 # personal_chat, private_group, channel, …


def _note(notes, msg: str) -> None:
    if notes is not None and msg not in notes:
        notes.append(msg)


def model_available(config, notes=None) -> bool:
    """Whether the filter model is available for the current config (never raises)."""
    mode = config.messages.filter
    if mode not in ("auto", "model"):
        return False
    model_dir = config.messages.model_dir or ""
    try:
        from . import msgmodel
        ok = bool(msgmodel.available(model_dir))
    except Exception as e:            # stub or broken install — fall back to rules
        ok = False
        if model_dir or mode == "model":
            _note(notes, f"message filter: model unavailable ({type(e).__name__}), using rules")
    if not ok and mode == "model":
        _note(notes, "message filter: [messages].filter = model, but there is no model — using rules")
    return ok


def model_plan(ctx, n_texts: int):
    """Plan of the msgmodel stage from the hw layout (as ``localanon hw``): tier, devices, threads, batches;
    precision of the CPU tier — from the export manifest (``msgmodel.cpu_file``), on a GPU — the fp16
    file. Without a GPU — CPU and a note. No layout — None (the old path: CPU, ``model.onnx``)."""
    from . import hw
    try:
        sp = hw.stage_plan(ctx, "msgmodel", hw.Workload(n_messages=n_texts))
        res = hw.resources(ctx)
    except Exception as e:  # noqa: BLE001 — the layout must not break the stage
        _note(ctx.notes, f"messages: hardware layout not built ({type(e).__name__}) — filter on CPU")
        return None
    if not sp.on_gpu:
        why = ("set in the config" if (ctx.config.compute.devices or "auto").strip().lower() == "cpu"
               else "no GPU found" if not res.gpus
               else "the GPU did not fit (the reason is in the hw: msgmodel note)")
        _note(ctx.notes, f"messages: filter model on CPU — {why}")     # the stage calls classify for every chat
    return sp


def threshold_setting(config, notes=None):
    """[messages].threshold → "auto" or a float; the legacy 0.5 → "auto" with a note."""
    raw = config.messages.threshold
    if isinstance(raw, str):
        return "auto"
    thr = float(raw)
    if thr == LEGACY_THRESHOLD:
        _note(notes, "messages: [messages].threshold = 0.5 is the default of earlier versions — read as "
                     "\"auto\" (the threshold of the model tier); for exactly this threshold write 0.501")
        return "auto"
    return thr


def decision(config, cal: dict, notes=None) -> tuple[float, float, dict]:
    """→ (temperature, threshold on p_drop, per-category thresholds) from the config and the tier
    calibration."""
    temp = cal.get("temperature")
    temp = float(temp) if isinstance(temp, (int, float)) and temp > 0 else 1.0
    setting = threshold_setting(config, notes)
    if setting != "auto":
        return temp, setting, {}
    thr = cal.get("threshold")
    if not isinstance(thr, (int, float)) or not 0 < thr < 1:
        _note(notes, f"messages: the filter model has no calibrated threshold — using {FALLBACK_THRESHOLD}")
        return temp, FALLBACK_THRESHOLD, {}
    per = cal.get("thresholds") or {}
    per = {k: float(v) for k, v in per.items() if isinstance(v, (int, float)) and 0 < v < 1}
    return temp, float(thr), per


def decide(probs: dict, drop: list[str], thr: float, per: dict | None = None) -> tuple[str, float]:
    """One message: → (category, p_drop). ``per`` — threshold of the most likely drop category."""
    keep = [c for c in MSG_CATEGORIES if c not in drop]
    pd = sum(float(probs.get(c, 0.0)) for c in drop)
    top_drop = max(drop, key=lambda c: probs.get(c, 0.0)) if drop else ""
    t = (per or {}).get(top_drop, thr)
    if drop and pd >= t:
        return top_drop, pd
    return (max(keep, key=lambda c: probs.get(c, 0.0)) if keep else "other"), pd


def classify(msgs: list[Msg], config, notes: list[str] | None = None, ctx=None,
             doubt: list | None = None) -> list[str]:
    """→ a category from config.MSG_CATEGORIES for every message. The model (localanon/msgmodel.py)
    if [messages].filter allows it and the model exists, otherwise rules (localanon/msgrules.py).
    ``notes`` — where to write warnings (ctx.notes); optional. ``ctx`` — the run context: the
    model runs by the hw layout (``model_plan``); without it — CPU, as before.
    ``doubt`` — if a list is given, it gets one bool per message: kept by the model with p_drop in
    the doubtful band (always False for rules)."""
    n = len(msgs)
    mode = config.messages.filter
    if doubt is not None:
        doubt[:] = [False] * n
    if mode == "off" or n == 0:
        return ["other"] * n
    out: list[str | None] = [None] * n
    idx = [i for i, m in enumerate(msgs) if isinstance(m.text, str) and m.text.strip()]
    idx_set = set(idx)
    for i in range(n):
        if i not in idx_set:
            out[i] = "other"
    if idx and model_available(config, notes):
        from . import msgmodel
        drop = [c for c in MSG_CATEGORIES if c in set(config.messages.drop_categories)]
        plan = model_plan(ctx, len(idx)) if ctx is not None else None
        tier = plan.model if plan is not None and plan.model and plan.model != "default" else None
        temp, thr, per = decision(config, msgmodel.calibration(config.messages.model_dir, tier), notes)
        # with a plan — one call: predict itself spreads batches, slots and processes over the GPUs
        step = len(idx) if plan is not None else BATCH
        try:
            for start in range(0, len(idx), step):
                chunk = idx[start:start + step]
                kw = {"plan": plan, "ctx": ctx} if plan is not None else {}
                preds = msgmodel.predict_proba(config.messages.model_dir,
                                               [msgs[i].text for i in chunk],
                                               [list(msgs[i].context) for i in chunk], temperature=temp, **kw)
                for i, probs in zip(chunk, preds):
                    if not isinstance(probs, dict):
                        continue
                    cat, pd = decide(probs, drop, thr, per)
                    out[i] = cat
                    if doubt is not None and cat not in drop and thr - DOUBT_BAND <= pd:
                        doubt[i] = True
        except Exception as e:
            _note(notes, f"message filter: model error ({type(e).__name__}), using rules")
            out = [("other" if i not in idx_set else None) for i in range(n)]
            if doubt is not None:
                doubt[:] = [False] * n
    for i in idx:
        if out[i] is None:
            out[i] = msgrules.classify_text(msgs[i].text, msgs[i].context)
    return out  # type: ignore[return-value]
