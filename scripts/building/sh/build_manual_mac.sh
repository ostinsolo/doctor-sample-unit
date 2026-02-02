#!/usr/bin/env bash
# =============================================================================
# DSU Manual Build (Mac) – same idea as build_manual.bat on Windows
# =============================================================================
# Uses build_dsu.py (same as GitHub workflow for Win + Mac). Run after
# setup_local_mac.sh. Windows local uses build_manual.bat -> setup.py instead.
# Kills any running dsu-* processes, builds to dist/dsu, smoke-tests.
#
# Usage: ./build_manual_mac.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "DSU Build (Mac)"
echo "============================================================"
echo ""

# Kill running workers that might lock dist/dsu (e.g. lib)
if pkill -f "dsu-demucs" 2>/dev/null; then echo "Stopped dsu-demucs"; fi
if pkill -f "dsu-bsroformer" 2>/dev/null; then echo "Stopped dsu-bsroformer"; fi
if pkill -f "dsu-audio-separator" 2>/dev/null; then echo "Stopped dsu-audio-separator"; fi
sleep 1

# Python: prefer runtime from scripts/building/sh (setup_local_mac), else current python3
if [[ -f "$SCRIPT_DIR/runtime/bin/python" ]]; then
  PYTHON="$SCRIPT_DIR/runtime/bin/python"
elif [[ -f "$PROJECT_ROOT/runtime/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/runtime/bin/python"
else
  PYTHON="python3"
  echo "WARNING: No runtime/ found. Using python3. Run setup_local_mac.sh first."
fi

# Check architecture for platform-specific setup
ARCH=$(uname -m)
if [[ "$ARCH" = "arm64" ]]; then
  # Mac ARM: audio-separator pins samplerate==0.1.0 (x86_64-only dylib). Use 0.2.3+ (universal2).
  echo "Detected: Apple Silicon (arm64)"
  echo "Ensuring samplerate>=0.2.3 for VR separation (Mac ARM)..."
  $PYTHON -m pip install 'samplerate>=0.2.3' --force-reinstall -q
  $PYTHON -c "import samplerate; print(f'  samplerate {samplerate.__version__}')"
  
  echo "Ensuring FFmpeg (required by torchcodec for Demucs save/load)..."
  if ! brew list ffmpeg &>/dev/null; then
    echo "Installing FFmpeg via Homebrew..."
    brew install ffmpeg
  fi
  
  echo "Ensuring torchcodec (Demucs/torchaudio save)..."
  $PYTHON -m pip install 'torchcodec>=0.5' -q
  $PYTHON -c "import torchcodec; print('  torchcodec OK')"
else
  # Intel Mac: torchcodec not available, using soundfile fallback
  echo "Detected: Intel (x86_64)"
  echo "Intel Mac: torchcodec not available, using soundfile for audio I/O"
  echo "Ensuring soundfile is available..."
  $PYTHON -c "import soundfile; print(f'  soundfile {soundfile.__version__} OK')"
fi

echo "Building with build_dsu.py..."
$PYTHON scripts/building/py/build_dsu.py
echo ""

# Smoke test (build_dsu.py outputs to scripts/building/py/dist/dsu)
DSU_DIST="$PROJECT_ROOT/scripts/building/py/dist/dsu"
echo "Smoke test..."
"$DSU_DIST/dsu-demucs" --help >/dev/null 2>&1 || { echo "ERROR: dsu-demucs failed to start."; exit 1; }
"$DSU_DIST/dsu-bsroformer" --help >/dev/null 2>&1 || { echo "ERROR: dsu-bsroformer failed to start."; exit 1; }
"$DSU_DIST/dsu-audio-separator" --help >/dev/null 2>&1 || { echo "ERROR: dsu-audio-separator failed to start."; exit 1; }

echo ""
echo "============================================================"
echo "Build complete! Test with:"
echo "  ./dist/dsu/dsu-demucs --help"
echo "  ./dist/dsu/dsu-demucs --worker"
echo "============================================================"
