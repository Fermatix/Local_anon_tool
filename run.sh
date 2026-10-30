#!/usr/bin/env bash
# LocalAnonPipeline in one command (Linux, macOS).
#
#   ./run.sh <export dir or archive> [output dir]      # output: out/<export name> in this folder
#
# The models archive (localanon-models-<date>.zip, sent separately) must lie in this folder before the
# first start: it is unpacked into models/. The first start also installs everything into .venv next to
# this script: the package with all its dependencies, onnxruntime-gpu with CUDA libraries when an NVIDIA
# card is present, and the speech recognition model (whisper) sized for this machine. This is the only
# step that uses the network.
# Later starts reuse .venv (reinstalling only when pyproject.toml changes) and run offline; if the
# hardware changed and the run plan needs another whisper size, it is downloaded first.
#
#   LOCALANON_GPU=no ./run.sh ...        # do not install onnxruntime-gpu (yes: always; default: auto)
#   PYTHON=python3.12 ./run.sh ...       # interpreter for .venv (Python 3.11 or newer)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
MARK="$VENV/.localanon-installed"
CONFIG="$ROOT/localanon.toml"
GPU="${LOCALANON_GPU:-auto}"

say() { echo "== $*"; }
die() { echo "run.sh: $*" >&2; exit 2; }

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "usage: ./run.sh <export dir or archive> [output dir]" >&2
    exit 2
fi
IN="$1"
[ -e "$IN" ] || die "input not found: $IN"
NAME="$(basename "${IN%/}")"
if [ -f "$IN" ]; then
    case "$NAME" in
        *.tar.gz) NAME="${NAME%.tar.gz}" ;;
        *.zip|*.tar|*.tgz|*.7z|*.rar) NAME="${NAME%.*}" ;;
    esac
fi
OUT="${2:-$ROOT/out/$NAME}"

find_python() {    # first interpreter >= 3.11
    local c
    for c in ${PYTHON:-} python3.13 python3.12 python3.11 python3 python; do
        if command -v "$c" >/dev/null 2>&1 &&
           "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

if [ -x "$PY" ]; then
    BASE_PY="$PY"
else
    BASE_PY="$(find_python)" || die "Python 3.11 or newer is required (https://www.python.org/downloads/)"
fi

# The models come as a separate archive: the first start unpacks it into models/ (checked by models.lock.json)
PYTHONPATH="$ROOT" "$BASE_PY" -m localanon.fetch --ensure-models "$ROOT" || exit 1

if [ "$GPU" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then GPU=yes; else GPU=no; fi
fi
STAMP="$("$BASE_PY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
         "$ROOT/pyproject.toml") gpu=$GPU"

pkg_version() {   # installed version of a distribution or nothing
    "$PY" -c "import importlib.metadata as m, sys
try: print(m.version(sys.argv[1]))
except m.PackageNotFoundError: pass" "$1"
}

install() {
    say "installing into $VENV (first start or pyproject.toml changed)"
    [ -x "$PY" ] || "$BASE_PY" -m venv "$VENV"
    "$PY" -m pip install --upgrade pip
    # editable: the package stays in this folder and finds models/ next to it
    "$PY" -m pip install -e "$ROOT"
    # the engine's C accelerator (Aho-Corasick): optional, a fresh Python may have no wheel for it
    "$PY" -m pip install -q pyahocorasick >/dev/null 2>&1 ||
        echo "run.sh: pyahocorasick is not installed — the engine works without it, only slower" >&2
    if [ "$GPU" = "yes" ]; then
        say "NVIDIA card: onnxruntime-gpu instead of the CPU onnxruntime"
        local ver
        ver="$(pkg_version onnxruntime)"
        # GPU build of the same version as the CPU build; if PyPI has none, the latest one
        if [ -z "$ver" ] || ! "$PY" -m pip install "onnxruntime-gpu[cuda,cudnn]==$ver"; then
            "$PY" -m pip install "onnxruntime-gpu[cuda,cudnn]"
        fi
        # Both builds write the same onnxruntime module: removing the CPU build deletes files of the GPU
        # build too, so the GPU build is reinstalled on top without deps (otherwise pip brings the CPU one back).
        "$PY" -m pip uninstall -y onnxruntime
        "$PY" -m pip install --force-reinstall --no-deps "onnxruntime-gpu==$(pkg_version onnxruntime-gpu)"
        # faster-whisper (ctranslate2) is built for CUDA 12 and needs cuBLAS 12 and cuDNN 9 for CUDA 12, while
        # onnxruntime-gpu[cuda,cudnn] brings CUDA 13 libraries. nvidia-cudnn-cu12 and -cu13 write the same
        # files, the one installed last stays: cu12 is always the last pip step (onnxruntime-gpu works with it).
        say "speech recognition on the card: cuBLAS 12 and cuDNN 9 (last)"
        "$PY" -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
        say "check: localanon hw --probe-cuda"
        "$VENV/bin/localanon" hw --probe-cuda ||
            echo "run.sh: the card did not pass the check (reason above) — processing runs on CPU" >&2
    fi
    say "speech recognition model (whisper) for this machine"
    "$PY" -m localanon.fetch --asr auto
    echo "$STAMP" > "$MARK"
}

if [ ! -x "$PY" ] || [ ! -f "$MARK" ] || [ "$(cat "$MARK")" != "$STAMP" ]; then
    install
elif ! HF_HUB_OFFLINE=1 "$PY" -m localanon.fetch --asr auto --check; then
    # the hardware changed after the installation (more memory, another card): the run plan picks another
    # whisper size; without it the run falls back to the size in the cache or removes media with a mark
    say "speech recognition model (whisper) for this machine"
    "$PY" -m localanon.fetch --asr auto ||
        echo "run.sh: whisper download failed — media are transcribed with the model in the cache, if any" >&2
fi

if [ ! -f "$CONFIG" ]; then
    # fresh salt: tokens are derived from it, so the file stays on this machine
    "$VENV/bin/localanon" init "$CONFIG"
fi

say "processing $IN"
rc=0
HF_HUB_OFFLINE=1 "$VENV/bin/localanon" run --in "$IN" --out "$OUT" --config "$CONFIG" || rc=$?

echo
echo "Output folder: $OUT"
echo "  report (stays with you): $OUT/report.html"
if [ "$rc" -eq 0 ]; then
    for f in "$OUT"/*.zip "$OUT/local_manifest.json"; do
        [ -f "$f" ] && echo "  to hand over: $f"
    done
else
    echo "  no archive (exit code $rc): see the report and the message above"
fi
echo "Settings (executor names, chats to drop, files to restore): $CONFIG — edit and run again."
exit "$rc"
