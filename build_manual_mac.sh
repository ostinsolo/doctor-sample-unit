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
cd "$(dirname "$0")"

echo "============================================================"
echo "DSU Build (Mac)"
echo "============================================================"
echo ""

# Kill running workers that might lock dist/dsu (e.g. lib)
if pkill -f "dsu-demucs" 2>/dev/null; then echo "Stopped dsu-demucs"; fi
if pkill -f "dsu-bsroformer" 2>/dev/null; then echo "Stopped dsu-bsroformer"; fi
if pkill -f "dsu-audio-separator" 2>/dev/null; then echo "Stopped dsu-audio-separator"; fi
sleep 1

# Python: prefer runtime from setup_local_mac, else current python3
if [[ -f "./runtime/bin/python" ]]; then
  PYTHON="./runtime/bin/python"
else
  PYTHON="python3"
  echo "WARNING: No runtime/ found. Using python3. Run setup_local_mac.sh first."
fi

# Mac ARM: audio-separator pins samplerate==0.1.0 (x86_64-only dylib). Use 0.2.3+ (universal2).
echo "Ensuring samplerate>=0.2.3 for VR separation (Mac ARM)..."
$PYTHON -m pip install 'samplerate>=0.2.3' --force-reinstall -q
$PYTHON -c "import samplerate; print(f'  samplerate {samplerate.__version__}')"

echo "Building with build_dsu.py..."
$PYTHON build_dsu.py
echo ""

# Smoke test
echo "Smoke test..."
./dist/dsu/dsu-demucs --help >/dev/null 2>&1 || { echo "ERROR: dsu-demucs failed to start."; exit 1; }
./dist/dsu/dsu-bsroformer --help >/dev/null 2>&1 || { echo "ERROR: dsu-bsroformer failed to start."; exit 1; }
./dist/dsu/dsu-audio-separator --help >/dev/null 2>&1 || { echo "ERROR: dsu-audio-separator failed to start."; exit 1; }

echo ""
echo "============================================================"
echo "Build complete! Test with:"
echo "  ./dist/dsu/dsu-demucs --help"
echo "  ./dist/dsu/dsu-demucs --worker"
echo "============================================================"
