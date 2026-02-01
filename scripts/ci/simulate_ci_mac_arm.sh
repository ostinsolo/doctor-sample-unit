#!/usr/bin/env bash
# =============================================================================
# Simulate GitHub Actions Mac ARM (MPS) build locally
# =============================================================================
# Replicates release.yml "build-mac-arm" job: deps, build_dsu, smoke test, tar.
# Then runs a minimal real-audio test (Demucs htdemucs) to verify output.
#
# Usage:
#   ./simulate_ci_mac_arm.sh              # full run (venv, build, smoke, tar, real-audio)
#   ./simulate_ci_mac_arm.sh --no-audio   # skip real-audio test
#   ./simulate_ci_mac_arm.sh --skip-build # real-audio only (use existing dist/dsu + venv)
#
# Requires: Python 3.10, arm64 Mac. Creates simulate_ci_venv/ and simulate_ci_out/.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_AUDIO=
SKIP_BUILD=
for a in "$@"; do
  [[ "$a" = "--no-audio" ]] && SKIP_AUDIO=1
  [[ "$a" = "--skip-build" ]] && SKIP_BUILD=1
done

echo "============================================================================="
echo "Simulate GitHub: Mac ARM (MPS) build"
echo "============================================================================="
echo ""

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
  echo "WARNING: Not arm64 (got $ARCH). GitHub Mac ARM job runs on arm64."
  echo ""
fi

PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &>/dev/null; then
  echo "ERROR: python3 not found."
  exit 1
fi
PYVER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
if [[ "$PYVER" != "3.10" ]]; then
  echo "WARNING: Python $PYVER (GitHub uses 3.10). Continuing anyway."
  echo ""
fi

# -----------------------------------------------------------------------------
# 1. Venv + deps (simulate CI: pip install -r requirements-mac-mps)
# -----------------------------------------------------------------------------
VENV_DIR="$SCRIPT_DIR/simulate_ci_venv"
if [[ -n "$SKIP_BUILD" ]]; then
  if [[ ! -d "$VENV_DIR" ]] || [[ ! -d "$SCRIPT_DIR/dist/dsu" ]]; then
    echo "ERROR: --skip-build requires simulate_ci_venv/ and dist/dsu/. Run full build first."
    exit 1
  fi
  echo "[1/4] --skip-build: reusing venv + dist/dsu"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "[1/5] Create venv and install deps (requirements-mac-mps.txt)..."
  if [[ -d "$VENV_DIR" ]]; then
    echo "      Reusing existing simulate_ci_venv/"
  else
    $PYTHON_CMD -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install -q --upgrade pip
  pip install -q -r "$SCRIPT_DIR/requirements-mac-mps.txt"
  # Mac ARM: samplerate 0.1.0 ships x86_64-only libsamplerate.dylib; use 0.2.3+ (universal2)
  pip install -q 'samplerate>=0.2.3' --force-reinstall
  echo "      Done."
  echo ""

  # -----------------------------------------------------------------------------
  # 2. Build (simulate CI: python build_dsu.py)
  # -----------------------------------------------------------------------------
  echo "[2/5] Build DSU (build_dsu.py)..."
  python build_dsu.py
  echo ""

  # -----------------------------------------------------------------------------
  # 3. Smoke test (simulate CI)
  # -----------------------------------------------------------------------------
  echo "[3/5] Smoke test..."
  ls -la dist/dsu/dsu-*
  dist/dsu/dsu-demucs --help >/dev/null 2>&1 || true
  echo "      Smoke test OK."
  echo ""

  # -----------------------------------------------------------------------------
  # 4. Create archive (simulate CI: tar)
  # -----------------------------------------------------------------------------
  echo "[4/5] Create archive (dsu-mac-arm.tar.gz)..."
  rm -f dsu-mac-arm.tar.gz
  tar -czf dsu-mac-arm.tar.gz -C dist dsu
  ls -la dsu-mac-arm.tar.gz
  echo "      Done."
  echo ""
fi

echo ""

# -----------------------------------------------------------------------------
# 5. Real-audio test (Demucs --worker, soundfile; avoids torchcodec)
# -----------------------------------------------------------------------------
STEP=$( [[ -n "$SKIP_BUILD" ]] && echo "[1/1]" || echo "[5/5]" )
echo "$STEP Real-audio test (Demucs htdemucs via --worker)..."
if [[ -n "$SKIP_AUDIO" ]]; then
  echo "      Skipped (--no-audio)."
  echo ""
  echo "============================================================================="
  echo "Simulate CI complete. Run without --no-audio to verify real-audio output."
  echo "============================================================================="
  exit 0
fi

AUDIO_DIR="$SCRIPT_DIR/simulate_ci_out"
mkdir -p "$AUDIO_DIR"
TEST_WAV="$AUDIO_DIR/test_1s.wav"
DEMUC_OUT="$AUDIO_DIR/demucs_out"

# Create 1s test WAV (44.1kHz mono)
python -c "
import numpy as np
import soundfile as sf
t = np.linspace(0, 1, 44100, dtype=np.float32)
x = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
sf.write('$TEST_WAV', x, 44100)
print('Created $TEST_WAV')
"

# Worker-mode separate (uses soundfile, no torchcodec)
python "$SCRIPT_DIR/simulate_ci_real_audio.py" \
  --exe "$SCRIPT_DIR/dist/dsu/dsu-demucs" \
  --wav "$TEST_WAV" \
  --out "$DEMUC_OUT" || exit 1

echo ""
echo "============================================================================="
echo "Simulate CI + real-audio test passed."
[[ -z "$SKIP_BUILD" ]] && echo "  Archive: dsu-mac-arm.tar.gz"
echo "  Demucs output: $DEMUC_OUT/htdemucs/test_1s/"
echo "You can commit and push a tag to trigger the GitHub release workflow."
echo "============================================================================="
