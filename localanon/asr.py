"""Audio and video transcription: faster-whisper on CPU, no network.

Contract for the ``stages/transcribe.py`` stage:
  ``available(model)`` — whether faster-whisper is installed and the model is in the project cache
  (``localanon.fetch.models_dir()``; ``python -m localanon.fetch --asr <model>`` downloads it at
  install time);
  ``transcribe(path, model, device, diarize) -> str`` — the transcript as lines
  "[HH:MM:SS-HH:MM:SS] SPEAKER_00: utterance" (empty — no speech);
  ``transcribe_many(paths, …)`` — a batch of files in parallel through worker subprocesses
  (``python -m localanon.asr --worker``, no fork), the model is loaded once per process;
  results — ``(path, Result | None, error | None)`` as they are ready.

How it works:
  * audio is decoded by PyAV (it comes in the faster-whisper wheels, no separate ffmpeg needed) into
    a 16 kHz mono stream; if PyAV cannot open the file — a fallback through ``imageio-ffmpeg`` (if
    installed);
  * long recordings are cut into ~10 minute chunks at the quietest spot in the last two minutes
    of the chunk, so the whole file is never loaded into memory; timestamps are shifted by the
    chunk start;
  * the model never sees chunks without speech (Silero VAD from faster-whisper);
  * the language is detected on the first chunk with speech, preferring Russian (``pick_language``);
  * speaker labeling (``diarize``) — pyannote, only if installed and the model is in the cache;
    otherwise all utterances are SPEAKER_00 and ``Result.note`` gets a note;
  * every file goes through a worker subprocess with a time limit ``file_timeout`` (from the
    recording length: the worker finds it and reports it before decoding; before that — an estimate
    by file size); a hung worker is killed and restarted, the file gets an error, the rest go on;
  * the main run process does not load faster-whisper/ctranslate2 and does not open media with PyAV:
    everything that can crash in native code runs in subprocesses, the fallback too (a one-shot
    subprocess per file).

Line format and utterance merging are as in the main pipeline (pure functions moved from
``tools/asr_worker.py``), so local and main pipeline transcripts look the same.
"""
from __future__ import annotations

import os

# Before huggingface_hub is imported (faster_whisper pulls it in): no network at run time, models
# come only from the cache. The library reads the variable once on import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import itertools  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Iterable, Iterator  # noqa: E402

from . import fetch, hw  # noqa: E402

SAMPLE_RATE = 16000
CHUNK_S = 600.0          # target chunk length, s
CHUNK_MAX_S = 720.0      # upper bound (if no silence was found)
CUT_SEARCH_S = 120.0     # where to look for the quietest spot: the last N seconds of the chunk
CUT_WINDOW_S = 0.1       # loudness estimation window
LANG_HINT = "ru"
LANG_ALLOWED = ("ru", "en")
BEAM_SIZE = 5
DEFAULT_MERGE_GAP = 0.8      # s: the pause up to which utterances of one speaker are still merged
DEFAULT_MAX_UTT_S = 120.0    # s: max length of a merged utterance
DIAR_MODEL = "pyannote/speaker-diarization-3.1"
SPEAKER_DEFAULT = "SPEAKER_00"
_SPEAKER_RX = re.compile(r"SPEAKER_\d{2,}\Z")
# Memory estimate of a process with the model (int8, CPU) for the worker count, GiB
_MODEL_GIB = {"tiny": 0.6, "base": 0.8, "small": 1.5, "medium": 2.5, "large": 5.0,
              "turbo": 3.0, "distil": 3.0}
_SYSTEM_RESERVE_GIB = 2.0
_MAX_WORKERS = 8
_THREADS_PER_WORKER_MAX = 8
# Time limits (s): for worker startup (model load) and for one file — from the recording length,
# but not below the minimum; length unknown — an estimate by file size.
_START_TIMEOUT = 600.0
_FILE_TIMEOUT_MIN = 300.0
_FILE_TIMEOUT_FACTOR = 3.0
_BYTES_PER_SECOND_GUESS = 16000      # ~128 kbit/s
_VAD_CLIP_TOLERANCE_S = 0.5          # margin when clipping a segment by a VAD region (speech_pad)

_PATH_RX = re.compile(r"(?:[A-Za-z]:)?[\\/](?:[^\s\\/'\"]+[\\/])*([^\s\\/'\"]*)")
_QUOTED_PATH_RX = re.compile(r"(['\"])((?:[A-Za-z]:)?[\\/][^'\"]*)\1")


def strip_paths(text: str, *known) -> str:
    """Paths in error text → only the file name (no working copy directories). ``known`` —
    known paths (the media file itself): replaced explicitly, so directories with spaces in the
    name are cut too; the rest — by shape (quoted and unquoted)."""
    text = str(text or "")
    for k in known:
        if not k:
            continue
        k = Path(k)
        for variant in (str(k), k.as_posix()):
            text = text.replace(variant, k.name)
        if k.parent != k and str(k.parent) not in (".", "/"):
            for variant in (str(k.parent), k.parent.as_posix()):
                text = text.replace(variant, "")
    text = _QUOTED_PATH_RX.sub(lambda m: m.group(1) + m.group(2).replace("\\", "/")
                               .rstrip("/").rsplit("/", 1)[-1] + m.group(1), text)
    return _PATH_RX.sub(lambda m: m.group(1), text)


class ModelMissing(RuntimeError):
    """faster-whisper is installed, but the model is not in the cache (or there is no library at all)."""


class NoAudioStream(RuntimeError):
    """The file is readable, but has no audio track (a screen recording without sound)."""


class DecodeError(RuntimeError):
    """Neither PyAV nor ffmpeg could read the file (broken, empty, not media)."""


@dataclass
class Result:
    text: str = ""              # finished transcript; empty — no speech
    duration: float = 0.0       # audio length, s
    language: str | None = None
    utterances: int = 0
    speakers: int = 0
    diarized: bool = False
    no_audio: bool = False      # no audio track
    elapsed: float = 0.0        # transcription time, s
    note: str = ""              # warning for the report (no personal data), empty — all normal


# ── pure format functions (moved from the main pipeline's tools/asr_worker.py) ─────────────

def format_ts(seconds) -> str:
    """Seconds → HH:MM:SS."""
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def clean_text(text) -> str:
    """Utterance text on one line: control and format characters → space, spaces collapsed."""
    out = []
    for ch in str(text or ""):
        cat = unicodedata.category(ch)
        out.append(" " if cat in ("Cc", "Cf", "Zl", "Zp", "Zs") else ch)
    return " ".join("".join(out).split())


def overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def pick_speaker(start, end, diar_segments, prev=None) -> str:
    """Speaker of an utterance: the labeling segment with the largest overlap; no overlap — the nearest
    in time (on a tie — the previous speaker); no labeling — SPEAKER_00."""
    if not diar_segments:
        return SPEAKER_DEFAULT
    best, best_ov = None, 0.0
    for d_start, d_end, speaker in diar_segments:
        ov = overlap(start, end, d_start, d_end)
        if ov > best_ov:
            best, best_ov = speaker, ov
    if best is not None:
        return best
    near, near_gap = None, None
    for d_start, d_end, speaker in diar_segments:
        gap = max(d_start - end, start - d_end, 0.0)
        if near_gap is None or gap < near_gap or (gap == near_gap and speaker == prev):
            near, near_gap = speaker, gap
    return near or prev or SPEAKER_DEFAULT


def normalize_speakers(diar_segments) -> list:
    """Labeling labels → SPEAKER_NN in order of appearance (pyannote already gives SPEAKER_00…)."""
    labels = []
    for _s, _e, sp in diar_segments:
        if sp not in labels:
            labels.append(sp)
    if all(_SPEAKER_RX.match(str(sp)) for sp in labels):
        return [(float(s), float(e), str(sp)) for s, e, sp in diar_segments]
    remap = {sp: f"SPEAKER_{i:02d}" for i, sp in enumerate(labels)}
    return [(float(s), float(e), remap[sp]) for s, e, sp in diar_segments]


def assign_speakers(transcribed, diar_segments, merge_gap=DEFAULT_MERGE_GAP,
                    max_utt_s=DEFAULT_MAX_UTT_S) -> list:
    """[(start, end, text)] + labeling → [[start, end, speaker, text]]. Neighbouring utterances of one
    speaker are merged with a pause ≤ merge_gap and a merged length ≤ max_utt_s; empty text is
    dropped; timestamps are monotonic."""
    out = []
    prev_speaker = None
    last_start = 0.0
    for start, end, text in transcribed:
        text = clean_text(text)
        if not text:
            continue
        start = max(float(start), last_start, 0.0)
        end = max(float(end), start)
        speaker = pick_speaker(start, end, diar_segments, prev_speaker)
        if (out and out[-1][2] == speaker
                and (start - out[-1][1]) <= merge_gap
                and (end - out[-1][0]) <= max_utt_s):
            out[-1][3] = f"{out[-1][3]} {text}"
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end, speaker, text])
        prev_speaker = speaker
        last_start = start
    return out


def render_lines(utterances) -> list:
    """Utterances → lines "[HH:MM:SS-HH:MM:SS] SPEAKER_NN: text"; empty ones dropped, a malformed
    speaker → SPEAKER_00, time is monotonic."""
    lines = []
    last_start = 0.0
    for u in utterances or []:
        try:
            start, end, speaker, text = u[0], u[1], u[2], u[3]
            start, end = float(start), float(end)
        except (TypeError, ValueError, IndexError):
            continue
        text = clean_text(text)
        if not text:
            continue
        speaker = str(speaker)
        if not _SPEAKER_RX.match(speaker):
            speaker = SPEAKER_DEFAULT
        start = max(start, last_start, 0.0)
        end = max(end, start)
        last_start = start
        lines.append(f"[{format_ts(start)}-{format_ts(end)}] {speaker}: {text}")
    return lines


def render_text(utterances) -> str:
    lines = render_lines(utterances)
    return "\n".join(lines) + "\n" if lines else ""


def pick_language(detected: str | None, probability: float, all_probs=None,
                  hint: str = LANG_HINT, allowed=LANG_ALLOWED) -> str:
    """Transcription language by autodetection, preferring Russian: Russian — if it was detected;
    if detection is unsure (< 0.6) or the language is not an expected one (ru, en) — Russian;
    if Russian is second with probability ≥ 0.25 (Russian speech with English terms) — Russian.
    Otherwise — the detected one (English)."""
    detected = (detected or "").strip().lower()
    if not detected or detected == hint:
        return hint
    try:
        prob = float(probability)
    except (TypeError, ValueError):
        prob = 0.0
    if prob < 0.6 or detected not in allowed:
        return hint
    for lang, p in all_probs or []:
        if str(lang).lower() == hint and float(p) >= 0.25:
            return hint
    return detected


def find_cut(samples, lo: int, hi: int, window: int) -> int:
    """Cut index in [lo, hi): the middle of the quietest window of ``window`` samples (by mean
    energy). No window fits — ``hi``."""
    import numpy as np
    lo = max(0, lo)
    hi = min(len(samples), hi)
    if hi <= lo or window <= 0:
        return hi
    seg = samples[lo:hi]
    m = len(seg) // window
    if m == 0:
        return hi
    energy = (seg[: m * window].reshape(m, window).astype(np.float32) ** 2).mean(axis=1)
    i = int(energy.argmin())
    return lo + i * window + window // 2


def iter_chunks(pcm: Iterable, chunk_s: float = CHUNK_S, max_s: float = CHUNK_MAX_S,
                search_s: float = CUT_SEARCH_S, rate: int = SAMPLE_RATE) -> Iterator[tuple]:
    """A stream of float32 arrays → chunks ``(start_s, array)`` of about ``chunk_s``, cut at the
    quietest spot in the last ``search_s`` seconds; a tail is never longer than ``max_s``."""
    import numpy as np
    chunk_n, max_n, search_n = int(chunk_s * rate), int(max_s * rate), int(search_s * rate)
    window = max(1, int(CUT_WINDOW_S * rate))
    buf, n, offset = [], 0, 0.0
    for arr in pcm:
        if arr is None or len(arr) == 0:
            continue
        buf.append(arr)
        n += len(arr)
        while n >= chunk_n:
            data = np.concatenate(buf) if len(buf) > 1 else buf[0]
            cut = find_cut(data, chunk_n - search_n, min(len(data), max_n), window)
            yield offset, data[:cut]
            offset += cut / rate
            rest = data[cut:]
            buf, n = ([rest] if len(rest) else []), len(rest)
    if n:
        yield offset, (np.concatenate(buf) if len(buf) > 1 else buf[0])


def clip_to_speech(start: float, end: float, speech: list, rate: int = SAMPLE_RATE,
                   tolerance: float = _VAD_CLIP_TOLERANCE_S) -> float:
    """Segment end clipped by the border of the VAD region holding its start: whisper glues speech
    regions, and a segment started in one may "cover" the pause up to the next one.
    Start outside the regions — ``end`` unchanged."""
    for region in speech or []:
        r_start = float(region["start"]) / rate - tolerance
        r_end = float(region["end"]) / rate + tolerance
        if r_start <= start <= r_end:
            return max(start, min(end, r_end))
    return end


# ── decoding ──────────────────────────────────────────────────────────────────────────────

def _pcm_av(path: Path, state: dict) -> Iterator:
    """PyAV → float32 16 kHz mono arrays. Opening: NoAudioStream / DecodeError. A failure in the
    middle of the file after some audio was read — stop with the state['partial'] mark."""
    import av
    import numpy as np
    try:
        container = av.open(str(path), mode="r", metadata_errors="ignore")
    except Exception as exc:  # noqa: BLE001 — the PyAV exception hierarchy depends on the version
        raise DecodeError(f"PyAV could not open the file: {exc.__class__.__name__}: "
                          f"{strip_paths(str(exc), path)[:200]}") from exc
    with container:
        if not container.streams.audio:
            raise NoAudioStream("no audio track")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        got = False

        def frames():
            it = container.decode(stream)
            while True:
                try:
                    frame = next(it)
                except StopIteration:
                    return
                except av.error.InvalidDataError:
                    continue
                except Exception as exc:  # noqa: BLE001
                    if not got:
                        raise DecodeError(f"PyAV could not read the audio: {exc.__class__.__name__}: "
                                          f"{strip_paths(str(exc), path)[:200]}") from exc
                    state["partial"] = f"{exc.__class__.__name__}"
                    return
                frame.pts = None
                yield frame

        for frame in itertools.chain(frames(), [None]):
            for out in resampler.resample(frame):
                arr = out.to_ndarray().reshape(-1)
                if arr.dtype != np.int16:
                    arr = arr.astype(np.int16)
                if len(arr):
                    got = True
                    yield arr.astype(np.float32) / 32768.0


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 — no package or the binary is not found
        return None


_NO_AUDIO_MARK = "does not contain any stream"


def _pcm_ffmpeg(path: Path, exe: str, state: dict) -> Iterator:
    """Fallback: ffmpeg from imageio-ffmpeg → s16le 16 kHz mono to stdout, read as a stream."""
    import numpy as np
    block = SAMPLE_RATE * 2 * 30            # 30 s
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            [exe, "-nostdin", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
             "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
        got = False
        tail = b""
        try:
            while True:
                buf = proc.stdout.read(block)
                if not buf:
                    break
                buf = tail + buf
                cut = len(buf) - (len(buf) % 2)
                tail = buf[cut:]
                arr = np.frombuffer(buf[:cut], dtype=np.int16)
                if len(arr):
                    got = True
                    yield arr.astype(np.float32) / 32768.0
        finally:
            proc.stdout.close()
            rc = proc.wait()
        err.seek(0)
        text = err.read().decode("utf-8", "replace")
        if rc != 0 and not got:
            if _NO_AUDIO_MARK in text:
                raise NoAudioStream("no audio track")
            last = strip_paths(" | ".join(text.strip().splitlines()[-3:]), path)[:300]
            raise DecodeError(f"ffmpeg could not read the file (rc={rc}): {last}")
        if rc != 0:
            state["partial"] = f"ffmpeg rc={rc}"


def decode_pcm(path: Path, state: dict | None = None) -> Iterator:
    """A float32 16 kHz mono stream: PyAV; if opening fails — ffmpeg from imageio-ffmpeg.
    Exceptions: NoAudioStream, DecodeError."""
    state = state if state is not None else {}
    try:
        yield from _pcm_av(path, state)
        return
    except NoAudioStream:
        raise
    except (DecodeError, ImportError) as exc:
        first = exc
    exe = _ffmpeg_exe()
    if exe is None:
        raise first
    state["fallback"] = "ffmpeg"
    yield from _pcm_ffmpeg(path, exe, state)


def media_duration(path) -> float | None:
    """Recording length from the container (PyAV, no decoding), s; unknown — None.
    Called only in the worker (the ``probe`` event): the main process does not open media."""
    try:
        import av
        with av.open(str(path), mode="r", metadata_errors="ignore") as container:
            if container.duration:
                return float(container.duration) / 1_000_000       # av.time_base — microseconds
            for stream in container.streams.audio:
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:  # noqa: BLE001 — no PyAV, broken container: estimate by size
        return None
    return None


def file_timeout(path, duration: float | None = None) -> float:
    """Time limit for transcribing one file, s: max(minimum, 3 × length).
    ``duration`` — from the worker probe; without it (the worker has not answered yet or did not
    find it) — an estimate by file size. The file is not opened here: the main process has no PyAV."""
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None
    if duration is not None and not (0.0 <= duration < float("inf")):
        duration = None
    if duration is None:
        try:
            duration = Path(path).stat().st_size / _BYTES_PER_SECOND_GUESS
        except OSError:
            duration = 0.0
    return max(_FILE_TIMEOUT_MIN, _FILE_TIMEOUT_FACTOR * duration)


# ── model ─────────────────────────────────────────────────────────────────────────────────

def availability(model: str = "medium", models_dir=None) -> tuple[bool, str]:
    """(whether transcription is possible, the reason if not). The package is looked up without an
    import: the main run process does not load faster-whisper/ctranslate2 (their native code lives
    only in the workers)."""
    import importlib.util
    try:
        found = importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        return False, ("faster-whisper is not installed (run.sh / run.ps1 installs it; by hand: "
                       ".venv/bin/python -m pip install -e .)")
    if fetch.whisper_path(model, models_dir) is None:
        return False, (f"whisper model '{model}' not found in {fetch.models_dir(models_dir)} — "
                       f"download it at install time: python -m localanon.fetch --asr {model}")
    return True, ""


def available(model: str = "medium", models_dir=None) -> bool:
    """Whether faster-whisper is installed and the model is in the local cache."""
    return availability(model, models_dir)[0]


_CUDA_WHEELS = "nvidia-cublas-cu12 and nvidia-cudnn-cu12==9.*"
_GPU_TYPES = ("float16", "int8_float16")       # hw.plan slot precisions (asr)
_CPU_TYPES = ("int8", "int8_float32", "float32")


class Engine:
    """The whisper model (and, optionally, speaker labeling) — one per process."""

    def __init__(self, model: str = "medium", device: str = "cpu", diarize: bool = False,
                 models_dir=None, cpu_threads: int = 0, compute_type: str = ""):
        """``device`` — ``cpu``, ``cuda`` or ``cuda:N`` (an hw plan slot); ``compute_type`` — the slot
        precision (``float16``/``int8_float16`` on a GPU, ``int8`` on CPU), empty — float16 / int8."""
        self.notes: list[str] = []
        path = fetch.whisper_path(model, models_dir)
        if path is None:
            raise ModelMissing(availability(model, models_dir)[1])
        device = (device or "cpu").lower()
        device, _, index = device.partition(":")
        index = int(index) if index.isdigit() else 0
        if device == "cuda":
            # cuBLAS 12 / cuDNN 9 from the nvidia-*-cu12 wheels — before WhisperModel; if they did not load,
            # the probe below fails the model on cuda and gives one note with a hint
            hw.preload_ct2_cuda_libs()
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ModelMissing(f"faster-whisper is not installed ({exc})") from exc
        if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
            self.notes.append("[media].asr_device = cuda, but CUDA is unavailable — transcribing on CPU")
            device = "cpu"
        if device not in ("cpu", "cuda"):
            device = "cpu"
        self.model_name = model
        self.model = None
        if device == "cuda":
            try:
                self.model = WhisperModel(str(path), device="cuda", device_index=index,
                                          compute_type=compute_type if compute_type in _GPU_TYPES
                                          else "float16",
                                          cpu_threads=int(cpu_threads or 0),
                                          local_files_only=True)
                # ctranslate2 loads cuBLAS/cuDNN on the first encoder run, not on model load: one
                # second of silence here makes the failure happen once per process, not per file
                import numpy as np
                self.model.detect_language(audio=np.zeros(SAMPLE_RATE, dtype=np.float32))
            except Exception as exc:  # noqa: BLE001 — CUDA libraries did not load
                self.model = None
                why = strip_paths(_format_error(exc))[:200]
                hint = (f" — pip packages needed: {_CUDA_WHEELS}"
                        if re.search(r"cublas|cudnn|\.so\.|\.dll|library", why, re.I) else "")
                self.notes.append(f"[media].asr_device = cuda, but ctranslate2 on CUDA did not "
                                  f"work ({why}){hint}; transcribing on CPU")
                device = "cpu"
        self.device = device
        self.device_index = index if device == "cuda" else 0
        if device == "cpu":
            self.compute = compute_type if compute_type in _CPU_TYPES else "int8"
        else:
            self.compute = compute_type if compute_type in _GPU_TYPES else "float16"
        if self.model is None:
            self.model = WhisperModel(str(path), device=device, compute_type=self.compute,
                                      cpu_threads=int(cpu_threads or 0), local_files_only=True)
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        self._vad = get_speech_timestamps
        self._vad_opts = VadOptions()
        self.diar = None
        if diarize:
            self.diar = self._load_diarizer(models_dir)

    def _load_diarizer(self, models_dir):
        try:
            import torch  # noqa: F401
            from pyannote.audio import Pipeline
        except ImportError:
            self.notes.append("speaker labeling ([media].diarize) is unavailable: no pyannote.audio "
                              "and torch — all utterances are SPEAKER_00")
            return None
        try:
            try:
                pipe = Pipeline.from_pretrained(DIAR_MODEL, cache_dir=str(fetch.models_dir(models_dir)))
            except TypeError:
                pipe = Pipeline.from_pretrained(DIAR_MODEL)
            if pipe is None:
                raise RuntimeError("no model returned (not downloaded at install time?)")
            return pipe
        except Exception as exc:  # noqa: BLE001
            self.notes.append("speaker labeling is unavailable: the pyannote model did not load "
                              f"({exc.__class__.__name__}) — all utterances are SPEAKER_00")
            return None

    # ── steps ──

    def speech_chunks(self, audio) -> list:
        return self._vad(audio, self._vad_opts)

    def detect_language(self, audio) -> str:
        """Language by the chunk speech (up to three 30 s windows), preferring Russian."""
        import numpy as np
        chunks = self.speech_chunks(audio)
        if chunks:
            audio = np.concatenate([audio[c["start"]:c["end"]] for c in chunks])
        audio = audio[: 3 * 30 * SAMPLE_RATE]
        try:
            lang, prob, all_probs = self.model.detect_language(
                audio=audio, language_detection_segments=3)
        except Exception:  # noqa: BLE001 — a model without multilingual support etc.
            return LANG_HINT
        return pick_language(lang, prob, all_probs)

    def transcribe_chunk(self, audio, language: str, offset: float = 0.0,
                         speech: list | None = None) -> list:
        """Chunk → [(start, end, text)] with absolute timestamps; ``speech`` — VAD regions of the chunk
        (in samples): a segment end does not go past the region it started in."""
        segments, _info = self.model.transcribe(
            audio, language=language, vad_filter=True, vad_parameters=self._vad_opts,
            beam_size=BEAM_SIZE, condition_on_previous_text=True)
        out = []
        for seg in segments:
            text = clean_text(seg.text)
            if not text:
                continue
            start, end = float(seg.start), float(seg.end)
            if speech:
                end = clip_to_speech(start, end, speech)
            out.append((offset + start, offset + end, text))
        return out

    def diarize(self, audio) -> list:
        import numpy as np
        import torch
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        result = self.diar({"waveform": wav, "sample_rate": SAMPLE_RATE})
        obj = getattr(result, "exclusive_speaker_diarization", None)
        if obj is None:
            obj = getattr(result, "speaker_diarization", result)
        segs = []
        for turn, _track, speaker in obj.itertracks(yield_label=True):
            segs.append((float(turn.start), float(turn.end), str(speaker)))
        segs.sort(key=lambda x: x[0])
        return normalize_speakers(segs)

    def process(self, path) -> Result:
        import numpy as np
        path = Path(path)
        t0 = time.monotonic()
        res = Result()
        notes = list(self.notes)
        state: dict = {}
        language = None
        raw: list = []
        keep: list = []
        try:
            for offset, chunk in iter_chunks(decode_pcm(path, state)):
                res.duration += len(chunk) / SAMPLE_RATE
                if self.diar is not None:
                    keep.append(chunk)
                speech = self.speech_chunks(chunk)
                if not speech:
                    continue
                if language is None:
                    language = self.detect_language(chunk)
                raw.extend(self.transcribe_chunk(chunk, language, offset, speech))
        except NoAudioStream:
            res.no_audio = True
            res.note = "; ".join(dict.fromkeys(notes))
            return res
        if state.get("partial"):
            notes.append("some media files were not read to the end (broken tail) — transcribed "
                         "what could be read")
        res.language = language
        diar_segments: list = []
        if raw and self.diar is not None and keep:
            try:
                diar_segments = self.diarize(np.concatenate(keep))
                res.diarized = bool(diar_segments)
            except Exception as exc:  # noqa: BLE001
                notes.append("speaker labeling failed on some files "
                             f"({exc.__class__.__name__}) — SPEAKER_00")
        keep.clear()
        utts = assign_speakers(raw, diar_segments)
        res.text = render_text(utts)
        res.utterances = len(utts)
        res.speakers = len({u[2] for u in utts})
        res.note = "; ".join(dict.fromkeys(notes))
        res.elapsed = round(time.monotonic() - t0, 1)
        return res


# ── one process: the model is cached ──────────────────────────────────────────────────────

_ENGINE: Engine | None = None
_ENGINE_KEY: tuple | None = None


def _engine(model: str, device: str, diarize: bool, models_dir=None, cpu_threads: int = 0,
            compute_type: str = "") -> Engine:
    global _ENGINE, _ENGINE_KEY
    key = (model, device, bool(diarize), str(models_dir or ""), int(cpu_threads or 0), compute_type or "")
    if _ENGINE is None or _ENGINE_KEY != key:
        _ENGINE = Engine(model, device, diarize, models_dir, cpu_threads, compute_type or "")
        _ENGINE_KEY = key
    return _ENGINE


def transcribe_file(path, model: str = "medium", device: str = "cpu", diarize: bool = False,
                    models_dir=None, cpu_threads: int = 0, compute_type: str = "") -> Result:
    """One file → Result (the model is loaded once per process). Exceptions: ModelMissing,
    DecodeError and other file errors."""
    return _engine(model, device, diarize, models_dir, cpu_threads, compute_type).process(path)


def transcribe(path, model: str = "medium", device: str = "cpu", diarize: bool = False) -> str:
    """→ transcript text: lines "[HH:MM:SS-HH:MM:SS] SPEAKER_00: utterance"; empty — no speech."""
    return transcribe_file(path, model, device, diarize).text


# ── batch: spawn processes ────────────────────────────────────────────────────────────────

def _available_memory() -> int | None:
    """Free memory, bytes; unknown — None."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 — psutil is optional
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
                return int(st.ullAvailPhys)
        except Exception:  # noqa: BLE001
            return None
        return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def model_memory_gib(model: str) -> float:
    m = model.lower()
    for key, gib in _MODEL_GIB.items():
        if key in m:
            return gib
    return 2.5


def plan_workers(n_files: int, model: str = "medium", cpus: int | None = None,
                 mem_bytes: int | None = None) -> int:
    """Number of transcription processes: by files, cores (≥ 4 per process) and memory (model size
    plus a reserve for the system), at most 8."""
    if n_files <= 1:
        return 1
    cpus = cpus or os.cpu_count() or 2
    by_cpu = max(1, cpus // 4)
    mem = mem_bytes if mem_bytes is not None else _available_memory()
    if mem is None:
        mem = 8 * 2 ** 30
    by_mem = max(1, int((mem / 2 ** 30 - _SYSTEM_RESERVE_GIB) // model_memory_gib(model)))
    return max(1, min(n_files, by_cpu, by_mem, _MAX_WORKERS))


def threads_per_worker(workers: int, cpus: int | None = None) -> int:
    cpus = cpus or os.cpu_count() or 2
    return max(1, min(_THREADS_PER_WORKER_MAX, cpus // max(1, workers)))


def _format_error(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {str(exc)[:200]}".strip(": ")


def _safe(path: Path, model, device, diarize, models_dir, cpu_threads, compute_type: str = "") -> tuple:
    try:
        return transcribe_file(path, model, device, diarize, models_dir, cpu_threads, compute_type), None
    except Exception as exc:  # noqa: BLE001 — an error of one file does not fail the batch
        return None, _format_error(exc)


# ── batch: worker subprocesses ────────────────────────────────────────────────────────────
#
# Not multiprocessing: under "python -m localanon" a spawn child runs localanon/__main__.py again
# (without a guard it would start the whole run once more). Our own subprocess
# "python -m localanon.asr --worker" with JSON lines on stdin/stdout does not depend on the entry
# point and is the same on Windows and Linux. The model is loaded once per process.

# A fuse for when the library or the model itself is broken and everything crashes: the pool
# stops if at least _CRASH_FUSE files crashed in a row (no worker answer between the crashes,
# across all threads) and that is more than half of the files the pool took. The rest goes to
# one-shot processes as a trial: the first worker answer means the library is alive, the rest goes
# back to the pool; at most _POOL_RETURNS such returns per run, after that — one at a time to the end.
_CRASH_FUSE = 5
_POOL_RETURNS = 3


def _worker_cmd(model, device, diarize, models_dir, threads, compute_type: str = "") -> list[str]:
    cmd = [sys.executable, "-m", "localanon.asr", "--worker", "--model", str(model),
           "--device", str(device), "--threads", str(int(threads))]
    if compute_type:
        cmd += ["--compute-type", str(compute_type)]
    if diarize:
        cmd.append("--diarize")
    if models_dir:
        cmd += ["--models-dir", str(models_dir)]
    return cmd


def _worker_env() -> dict:
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def worker_main(argv=None) -> int:
    """Worker process: the model once (``{"event": "ready"}``), then one JSON line per file from
    stdin → ``{"event": "probe", "path", "duration"}`` (length or null, before decoding)
    and ``{"path", "result", "error"}`` to stdout. stdout carries the protocol: descriptor 1 is
    redirected to stderr while working, so chatty libraries do not break the JSON."""
    import argparse
    import dataclasses
    import json

    ap = argparse.ArgumentParser(prog="python -m localanon.asr")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--compute-type", default="")
    ns = ap.parse_args(argv)
    if not ns.worker:
        ap.print_help()
        return 2
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    try:
        sys.stdin.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    def emit(obj) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    try:
        _engine(ns.model, ns.device, ns.diarize, ns.models_dir, ns.threads, ns.compute_type)
    except Exception as exc:  # noqa: BLE001 — the reason goes to the parent as text
        emit({"event": "fatal", "error": _format_error(exc)})
        return 3
    emit({"event": "ready"})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            emit({"event": "fatal", "error": "protocol: not JSON"})
            return 2
        if msg.get("op") == "quit":
            break
        path = str(msg.get("path") or "")
        # Length — before decoding: the parent sets the file time limit by it.
        # Container parsing (the most fragile part of libav) happens here, not in the main process.
        emit({"event": "probe", "path": path, "duration": media_duration(path)})
        res, err = _safe(Path(path), ns.model, ns.device, ns.diarize, ns.models_dir, ns.threads,
                         ns.compute_type)
        emit({"path": path, "result": dataclasses.asdict(res) if res is not None else None,
              "error": err})
    return 0


class _Worker:
    """One worker subprocess from the parent side: stdout is read by a separate thread (the answer
    is awaited with a time limit), stderr — into a temporary file (its tail goes to diagnostics)."""

    def __init__(self, cmd: list[str], env: dict):
        import queue
        import threading
        self.err = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.err, env=env, text=True,
                                         encoding="utf-8", errors="replace", bufsize=1)
        except BaseException:
            self.err.close()
            raise
        self.lines: queue.Queue = queue.Queue()
        self.reader = threading.Thread(target=self._pump, daemon=True)
        self.reader.start()

    def _pump(self) -> None:
        try:
            for line in self.proc.stdout:
                self.lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.proc.stdout.close()
            except (OSError, ValueError):
                pass
            self.lines.put(None)

    def recv(self, timeout: float) -> dict | None:
        """The next JSON message; None — the process closed stdout; TimeoutError — silent for longer
        than ``timeout`` s. Non-JSON lines (library chatter) are skipped."""
        import json
        import queue
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no answer in {timeout:.0f} s")
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty:
                raise TimeoutError(f"no answer in {timeout:.0f} s") from None
            if line is None:
                return None
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if isinstance(msg, dict):
                return msg

    def send(self, obj) -> None:
        import json
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def stderr_tail(self, lines: int = 3, limit: int = 300, known=None) -> str:
        """The last lines of the worker stderr without paths (for the report mark); ``known`` — the
        path of the file the worker crashed on."""
        try:
            self.err.seek(0)
            text = self.err.read().decode("utf-8", "replace")
        except (OSError, ValueError):
            return ""
        return strip_paths(" | ".join(text.strip().splitlines()[-lines:]), known)[:limit]

    def close(self, kill: bool = False) -> None:
        try:
            if not kill:
                self.send({"op": "quit"})
                self.proc.stdin.close()
                self.proc.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            kill = True
        if kill and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        self.reader.join(timeout=5)
        try:
            self.err.close()
        except OSError:
            pass


def _parse_result(msg: dict) -> tuple:
    """Worker answer → (Result | None, error | None); an unexpected shape is a protocol error, not
    an exception (another package version in the worker must not fail the thread)."""
    raw = msg.get("result")
    err = msg.get("error")
    if raw is None:
        return None, (str(err) if err else "protocol: the worker returned neither a result nor an error")
    try:
        res = Result(**raw)
        res.text = str(res.text or "")
        res.duration = float(res.duration or 0.0)
        res.utterances = int(res.utterances or 0)
        res.no_audio = bool(res.no_audio)
        res.note = str(res.note or "")
    except Exception as exc:  # noqa: BLE001 — any shape mismatch
        return None, f"protocol: unexpected answer of the transcription process ({_format_error(exc)})"
    return res, (str(err) if err else None)


def _start_worker(cmd: list[str], env: dict) -> tuple:
    """(worker, None) after the ready event; (None, reason) — did not start (killed)."""
    try:
        worker = _Worker(cmd, env)
    except OSError as exc:
        return None, _format_error(exc)
    try:
        ready = worker.recv(_START_TIMEOUT)
    except TimeoutError:
        ready = {"event": "fatal", "error": f"transcription process did not start in "
                                            f"{_START_TIMEOUT:.0f} s"}
    if not ready or ready.get("event") != "ready":
        why = (ready or {}).get("error") or "transcription process did not start"
        tail = worker.stderr_tail()
        worker.close(kill=True)
        return None, f"{why} ({tail})" if tail and not ready else str(why)
    return worker, None


def _teller(note: Callable[[str], None] | None) -> Callable[[str], None]:
    def tell(msg: str) -> None:
        if note is not None:
            try:
                note(msg)
            except Exception:  # noqa: BLE001 — a note must not fail the batch
                pass
    return tell


def _ask(worker: _Worker, p: Path, tell: Callable[[str], None]) -> tuple:
    """One file through a started worker → ``(Result | None, error | None, outcome)``: outcome
    ``"ok"`` — the worker answered and is alive; ``"timeout"`` and ``"crash"`` — the worker is killed and closed.

    Time limit: before the probe — an estimate by file size; after ``probe`` — from the recording
    length, counted again from the probe moment."""
    limit = file_timeout(p)
    deadline = time.monotonic() + limit
    try:
        worker.send({"path": str(p)})
        while True:
            msg = worker.recv(deadline - time.monotonic())
            if msg is None or msg.get("event") != "probe":
                break
            limit = file_timeout(p, msg.get("duration"))
            deadline = time.monotonic() + limit
    except TimeoutError:
        worker.close(kill=True)
        return None, (f"transcription time exceeded ({limit:.0f} s) — process stopped"), "timeout"
    except (OSError, ValueError):
        msg = None
    if msg is None or msg.get("event") == "fatal" or "path" not in msg:
        rc = worker.proc.poll()
        if rc is None and msg is None:          # stdout closed — the process is about to exit
            try:
                rc = worker.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        tail = worker.stderr_tail(known=p)
        worker.close(kill=True)
        why = strip_paths(str(msg.get("error") or ""), p) if msg else ""
        if tail:
            tell(f"transcription process crashed (rc={rc}): {tail}")
        return None, (f"transcription process crashed on this file (rc={rc})"
                      + (f": {why}" if why else "")), "crash"
    res, err = _parse_result(msg)
    return res, err, "ok"


def _slot_cmds(n: int, model, device, diarize, models_dir, slots, threads: int) -> list[list[str]]:
    """Commands of ``n`` workers: by the hw plan slots (device, precision, threads — per worker),
    without slots — all the same on ``device``."""
    if not slots:
        return [_worker_cmd(model, device, diarize, models_dir, threads)] * n
    return [_worker_cmd(model, s.device, diarize, models_dir, s.threads, s.compute_type)
            for s in (list(slots) * n)[:n]]


def _run_workers(order: list, workers: int, model, device, diarize, models_dir,
                 note: Callable[[str], None] | None = None, slots=None) -> Iterator[tuple]:
    """A batch through ``workers`` subprocesses. Yields ``(path, Result | None, error | None)`` —
    each file exactly once; at the end — ``(None, None, [paths not taken], reason)``: files the
    workers did not take because they did not start or the fuse tripped (an empty list —
    everything was handed out).

    A worker crash is isolated to the file: the file gets an error, the worker is restarted, the
    queue goes on. The fuse (``_CRASH_FUSE`` crashes in a row across the pool and more than half of
    the files taken) stops the pool: threads finish their current files and take no more, the rest
    goes to the fallback. The shared queue is not drained by errors."""
    import queue
    import threading

    cmds = _slot_cmds(workers, model, device, diarize, models_dir, slots, threads_per_worker(workers))
    env = _worker_env()
    pending: queue.Queue = queue.Queue()
    for p in order:
        pending.put(p)
    results: queue.Queue = queue.Queue()
    fatal: list[str] = []
    fused: list[str] = []
    lock = threading.Lock()
    stop = threading.Event()    # the fuse tripped: no more files are taken
    streak = [0]                # crashes in a row across the pool, with no worker answer between them
    handled = [0]               # files taken by the pool
    tell = _teller(note)

    def loop(cmd: list[str]) -> None:
        worker = None
        current = None
        try:
            while not stop.is_set():
                try:
                    current = pending.get_nowait()
                except queue.Empty:
                    break
                p = current
                if worker is None:
                    worker, why = _start_worker(cmd, env)
                    if worker is None:
                        fatal.append(str(why))
                        pending.put(p)
                        current = None
                        return
                with lock:
                    handled[0] += 1
                res, err, how = _ask(worker, p, tell)
                if how != "ok":
                    worker = None               # killed and closed in _ask
                results.put((p, res, err))
                current = None
                with lock:
                    if how == "ok":
                        streak[0] = 0           # the worker answered — the crash streak is broken
                    elif how == "crash":
                        streak[0] += 1
                        if (streak[0] >= _CRASH_FUSE and 2 * streak[0] > handled[0]
                                and not stop.is_set()):
                            fused.append(f"transcription process keeps crashing ({streak[0]} times in a row)")
                            stop.set()          # the rest of the queue goes to the fallback
        except Exception as exc:  # noqa: BLE001 — a parent failure: the file is not lost
            why = f"transcription process control failed: {_format_error(exc)}"
            if current is not None:
                results.put((current, None, why))
            tell(why)
            if worker is not None:
                worker.close(kill=True)
                worker = None
        finally:
            if worker is not None:
                worker.close()
            results.put(None)

    threads = [threading.Thread(target=loop, args=(cmd,), daemon=True) for cmd in cmds]
    for t in threads:
        t.start()
    alive = workers
    while alive:
        try:
            item = results.get(timeout=5)
        except queue.Empty:
            if not any(t.is_alive() for t in threads) and results.empty():
                break
            continue
        if item is None:
            alive -= 1
            continue
        yield item
    for t in threads:
        t.join(timeout=5)
    left = []
    while True:
        try:
            left.append(pending.get_nowait())
        except queue.Empty:
            break
    position = {str(p): i for i, p in enumerate(order)}
    left.sort(key=lambda p: position.get(str(p), len(order)))   # large ones first, as in order
    why = [f"transcription processes did not start ({fatal[0]})"] if fatal else []
    yield None, None, left, "; ".join(why + fused[:1]) or "transcription processes stopped"


def _run_one_shot(order: list, model, device, diarize, models_dir,
                  note: Callable[[str], None] | None = None,
                  trial: bool = False, slots=None) -> Iterator[tuple]:
    """Fallback: each file in its own one-shot subprocess (the same worker, protocol and time
    limit; the model is loaded per file). A subprocess crash (signal, abort, OOM) is an error of
    that file, the batch goes on. The main process does not load the model and does not open media.
    The subprocess did not start — an error for all remaining files (startup does not depend on the file).

    ``trial`` — a trial: after the first file the process answered on (a file error counts too),
    yields ``(None, None, [paths not taken])`` and stops — the library is alive, the rest can go
    back to the pool."""
    if not order:
        return
    tell = _teller(note)
    cmd = _slot_cmds(1, model, device, diarize, models_dir, slots, threads_per_worker(1))[0]
    env = _worker_env()
    for i, p in enumerate(order):
        worker = None
        try:
            worker, why = _start_worker(cmd, env)
            if worker is None:
                rest = order[i:]
                tell(f"transcription process did not start one at a time either ({why}) — files with an error: "
                     f"{len(rest)}")
                err = f"transcription process did not start: {strip_paths(why)}"[:300]
                for q in rest:
                    yield q, None, err
                return
            res, err, how = _ask(worker, p, tell)
            if how == "ok":
                worker.close()
        except Exception as exc:  # noqa: BLE001 — a parent failure: the file is not lost
            if worker is not None:
                worker.close(kill=True)
            res, err, how = None, f"transcription process control failed: {_format_error(exc)}", ""
        yield p, res, err
        if trial and how == "ok" and i + 1 < len(order):
            yield None, None, order[i + 1:]
            return


def transcribe_many(paths: Iterable, model: str = "medium", device: str = "cpu",
                    diarize: bool = False, workers: int | None = None, models_dir=None,
                    note: Callable[[str], None] | None = None, slots=None) -> Iterator[tuple]:
    """A batch of files → ``(path, Result | None, error | None)`` as they are ready, each path
    exactly once. Always through worker subprocesses (one file too — so the per-file time limit
    applies); the model is loaded once per process, large files first. Workers did not start or
    the fuse tripped — the remaining files go one at a time, each in its own one-shot subprocess
    (``_run_one_shot``), the reason — through ``note``; the first answer of a one-shot process
    returns the rest to the pool (at most ``_POOL_RETURNS`` times per run).
    The main process touches neither the model nor media: a native code crash does not fail the run.

    ``slots`` — hw plan slots (``hw.stage_plan(ctx, "asr").slots``): a worker per slot, each with its own
    device (``cpu``/``cuda:N``), precision and threads; ``device`` is then unused, ``workers``
    defaults to the number of slots."""
    paths = [Path(p) for p in paths]
    if not paths:
        return

    def size(p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    order = sorted(paths, key=lambda p: (-size(p), str(p)))
    slots = list(slots or [])
    if workers is None:
        workers = len(slots) if slots else plan_workers(len(order), model)
    workers = max(1, int(workers))
    md = str(models_dir) if models_dir else None

    rest, returns = order, 0
    while rest:
        left: list = []
        for item in _run_workers(rest, min(workers, len(rest)), model, device, diarize, md, note,
                                 slots=slots):
            if item[0] is None:
                left, why = item[2], item[3]
                break
            yield item
        if not left:
            return
        trial = returns < _POOL_RETURNS
        if note:
            note(f"{why} — files ({len(left)}) are transcribed one at a time, each in a separate "
                 f"process" + ("" if trial else f"; no more returns to the worker "
                                                 f"processes (returns so far: {returns})"))
        rest = []
        for item in _run_one_shot(left, model, device, diarize, md, note, trial=trial, slots=slots):
            if item[0] is None:
                rest = item[2]
                break
            yield item
        if rest:
            returns += 1
            if note:
                note(f"transcription process answers again — the remaining files ({len(rest)}) go "
                     f"through worker processes again (return {returns} of {_POOL_RETURNS})")


if __name__ == "__main__":
    raise SystemExit(worker_main())
