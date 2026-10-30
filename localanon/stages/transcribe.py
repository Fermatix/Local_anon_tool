"""Stage "transcribe".

Audio (mp3, m4a, ogg/oga/opus — Telegram voice messages, wav, flac, aac, wma, amr, aiff) and video
(mp4 — Telegram video notes too, mov, mkv, webm, avi, m4v, 3gp, wmv, flv, mpg) are transcribed by
``localanon/asr.py`` into ``<stem>.transcript.txt`` next to the file (lines
"[HH:MM:SS-HH:MM:SS] SPEAKER_00: utterance"); the media file itself is removed from the working
copy: ``ctx.remove(p, 'media', 'transcribed')``. The transcript then passes the message filter and
anonymization as plain text.

Owner's decision of 27.10 (evening, DESIGN.md §9b) returns this v1 behaviour: voice can't be
anonymized locally, so media never reaches the archive. A file whose transcription failed is
removed flagged ``borderline`` ("check it"); when transcription is unavailable for the whole run
(no faster-whisper or model, ``transcribe = false``) every media file is removed with the reason
and one ``ctx.note`` for the run, without the flag (as in v1).

Model, device, precision and worker threads follow the ``localanon.hw`` plan (``asr_plan``, as in
``localanon hw``): one worker per slot (``cpu`` or ``cuda:N``); without a GPU — CPU and a note.

Media file outcomes:
  * there is speech — the transcript is next to it, the file is removed ("transcribed"); the pair
    "media rel → transcript rel" goes to ``ctx.data['transcripts']``;
  * no speech or no audio stream — the file is removed ("no speech"), no transcript;
  * a non-empty ``<stem>.transcript.txt`` already lies next to it — the file is removed, no new
    transcription;
  * an error in one file does not crash the stage — the file is removed with ``borderline=True``
    and the reason (including a timeout: the per-file limit is ``asr.file_timeout``);
  * a file left without an outcome (the transcription processes failed) is removed after the loop
    with ``borderline=True`` — media never reaches the archive;
  * no faster-whisper or no model in the cache, or ``[media].transcribe = false`` — all media files
    are removed with the reason and one ``ctx.note`` (their speech will not reach the archive).

Counters (no paths): ``media:transcribed``, ``media:no_speech``, ``media:no_audio_stream``,
``media:failed``, ``media:skipped``, ``media:had_transcript``, ``media:audio_seconds``,
``media:utterances``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .. import asr

NAME = "transcribe"

AUDIO_EXT = frozenset({"mp3", "m4a", "ogg", "oga", "opus", "wav", "flac", "aac", "wma", "amr",
                       "aiff", "aif"})
VIDEO_EXT = frozenset({"mp4", "mov", "mkv", "webm", "avi", "m4v", "3gp", "wmv", "flv", "mpg",
                       "mpeg"})
MEDIA_EXT = AUDIO_EXT | VIDEO_EXT
OUT_SUFFIX = ".transcript.txt"
NAME_MAX_BYTES = 255


def is_media(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in MEDIA_EXT


def media_files(ctx) -> list[Path]:
    return [p for p in ctx.files() if is_media(p)]


def fit_name(base: str, limit: int) -> str:
    """Name base of at most ``limit`` UTF-8 bytes: a long one is cut and gets a hash tail."""
    if len(base.encode("utf-8")) <= limit:
        return base
    tail = "~" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    keep = limit - len(tail)
    cut = base
    while len(cut.encode("utf-8")) > keep:
        cut = cut[:-1]
    return cut + tail


def transcript_targets(paths: list[Path]) -> dict[Path, Path]:
    """{media: transcript path}: ``<stem>.transcript.txt`` next to it; two media with one stem in a
    directory (a.mp3 and a.mp4) both get ``<name.ext>.transcript.txt``; a long name is cut."""
    by_stem: dict[tuple, int] = {}
    for p in paths:
        key = (p.parent, p.stem.lower())
        by_stem[key] = by_stem.get(key, 0) + 1
    out = {}
    for p in paths:
        clash = by_stem[(p.parent, p.stem.lower())] > 1
        base = fit_name(p.name if clash else p.stem, NAME_MAX_BYTES - len(OUT_SUFFIX))
        out[p] = p.parent / (base + OUT_SUFFIX)
    return out


def _has_transcript(target: Path) -> bool:
    try:
        return target.is_file() and target.stat().st_size > 0
    except OSError:
        return False


def run(ctx) -> None:
    media = media_files(ctx)
    if not media:
        return
    cfg = ctx.config.media
    transcripts = ctx.data.setdefault("transcripts", {})

    if not cfg.transcribe:
        ctx.note(f"transcription is off ([media].transcribe = false): media files removed without "
                 f"transcription — {len(media)}; their speech will not reach the archive")
        for p in media:
            ctx.remove(p, "media", "not transcribed: transcription is off in the config")
            ctx.count("media:skipped")
        return

    ok, why = asr.availability(cfg.asr_model)
    if not ok:
        ctx.note(f"{why}; media files removed without transcription — {len(media)}; their speech "
                 f"will not reach the archive")
        short = why.split(" — ")[0].split(" (")[0]
        for p in media:
            ctx.remove(p, "media", f"not transcribed: {short}")
            ctx.count("media:skipped")
        return

    targets = transcript_targets(media)
    todo = []
    for p in media:
        target = targets[p]
        if _has_transcript(target):
            transcripts[ctx.rel(p)] = ctx.rel(target)
            ctx.remove(p, "media", "a transcript is already next to it")
            ctx.count("media:had_transcript")
        else:
            todo.append(p)
    if not todo:
        return

    notes: dict[str, None] = {}
    model, slots = asr_plan(ctx, len(todo))
    try:
        for p, res, err in asr.transcribe_many(todo, model=model, device="cpu", slots=slots,
                                               diarize=cfg.diarize,
                                               note=lambda m: notes.setdefault(m)):
            _finish(ctx, p, res, err, targets, transcripts, notes)
    except Exception as exc:  # noqa: BLE001 — a transcription failure does not crash the stage: media
        notes.setdefault(f"transcription interrupted: {asr._format_error(exc)}")  # still leave the output
    # The stage guarantee: media never reaches the archive. Anything from todo still in work
    # (a worker did not answer, a thread crashed, an outcome got lost) is removed with the "check" mark.
    missed = 0
    for p in todo:
        if p.exists():
            ctx.remove(p, "media", "transcript not received (the transcription process failed)",
                       borderline=True)
            ctx.count("media:failed")
            missed += 1
    if missed:
        notes.setdefault(f"media files without a transcription outcome removed with the 'check' mark — "
                         f"{missed}")
    for msg in notes:
        ctx.note(msg)


def asr_plan(ctx, n_files: int) -> tuple[str, list]:
    """Transcription model and slots from the hw plan — the same one ``localanon hw`` shows: device
    (``cpu``/``cuda:N``), precision (``compute_type``) and threads per worker. If the plan's model is not
    in the cache — ``[media].asr_model`` (the stage has already checked it exists). Without a GPU — CPU
    and a note."""
    from .. import hw
    cfg = ctx.config.media
    try:
        sp = hw.stage_plan(ctx, "asr", hw.Workload(n_audio=n_files))
        res = hw.resources(ctx)
    except Exception as exc:  # noqa: BLE001 — the plan does not crash the stage: CPU, as without a GPU
        ctx.note(f"transcribe: hardware plan not built ({type(exc).__name__}) — transcription on CPU")
        return cfg.asr_model, []
    model = sp.model or cfg.asr_model
    if model != cfg.asr_model and not asr.available(model):
        ctx.note(f"transcribe: plan model {model} is not in the cache — transcription with model {cfg.asr_model}")
        model = cfg.asr_model
    if not sp.on_gpu:
        dev = (cfg.asr_device or "auto").strip().lower()
        if dev == "auto":
            dev = (ctx.config.compute.devices or "auto").strip().lower()
        if dev == "cpu":
            why = "set in the config"
        elif not res.gpus:
            why = "no GPU found"
        else:
            why = "the GPU did not fit (the reason is in the hw note: asr)"
        ctx.note(f"transcribe: transcription on CPU — {why}")
    return model, list(sp.slots)


def _finish(ctx, p: Path, res, err, targets: dict, transcripts: dict, notes: dict) -> None:
    """One file outcome → removal with a reason, the transcript next to it, counters."""
    rel = ctx.rel(p)
    if err is not None or res is None:
        ctx.remove(p, "media", f"transcription error: {err or 'no result'}"[:300],
                   borderline=True)
        ctx.count("media:failed")
        return
    if isinstance(res, str):
        res = asr.Result(text=res)
    if res.note:
        notes.setdefault(res.note)
    ctx.count("media:audio_seconds", int(res.duration))
    if res.no_audio:
        ctx.count("media:no_audio_stream")
    if not res.text.strip():
        ctx.remove(p, "media", "no speech")
        ctx.count("media:no_speech")
        return
    target = targets[p]
    target.write_text(res.text, encoding="utf-8")
    transcripts[rel] = ctx.rel(target)
    ctx.remove(p, "media", "transcribed")
    ctx.count("media:transcribed")
    ctx.count("media:utterances", res.utterances)
