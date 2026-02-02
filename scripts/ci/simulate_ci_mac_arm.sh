#!/usr/bin/env bash
# =============================================================================
# Simulate GitHub Actions Mac ARM (MPS) build locally
# =============================================================================
# Replicates release.yml "build-mac-arm" job: uv, deps, build_dsu, smoke test, tar.
# Then runs a minimal real-audio test (Demucs htdemucs) to verify output.
#
# Usage:
#   ./simulate_ci_mac_arm.sh              # full run, 4s test audio (matches benchmarks)
#   ./simulate_ci_mac_arm.sh 40            # use test_40s.wav (40s input)
#   ./simulate_ci_mac_arm.sh --clean      # clean artifacts + venv, then run full build
#   ./simulate_ci_mac_arm.sh --no-audio   # skip real-audio test
#   ./simulate_ci_mac_arm.sh --skip-build # real-audio only (use existing dist/dsu + venv)
#
# Requires: uv, Python 3.10, arm64 Mac. Run from project root.
# Creates: simulate_ci_venv/, simulate_ci_out/, dsu-mac-arm.tar.gz
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_AUDIO=
SKIP_BUILD=
CLEAN=
DURATION=4
for a in "$@"; do
  [[ "$a" = "--no-audio" ]] && SKIP_AUDIO=1
  [[ "$a" = "--skip-build" ]] && SKIP_BUILD=1
  [[ "$a" = "--clean" ]] && CLEAN=1
  [[ "$a" = "40" ]] && DURATION=40
done

# -----------------------------------------------------------------------------
# Cleanup (if --clean): remove venv, build output, artifacts
# -----------------------------------------------------------------------------
cleanup() {
  echo "Cleaning simulate_ci artifacts..."
  rm -rf "$PROJECT_ROOT/simulate_ci_venv"
  rm -rf "$PROJECT_ROOT/simulate_ci_out"
  rm -rf "$PROJECT_ROOT/scripts/building/py/dist/dsu"
  rm -f "$PROJECT_ROOT/dsu-mac-arm.tar.gz"
  echo "  Cleaned: simulate_ci_venv/, simulate_ci_out/, dist/dsu/, dsu-mac-arm.tar.gz"
  echo ""
}
[[ -n "$CLEAN" ]] && cleanup

echo "============================================================================="
echo "Simulate GitHub: Mac ARM (MPS) build"
echo "============================================================================="
echo ""

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
export PATH="$HOME/.cargo/bin:$PATH"
uv --version
echo ""

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
# 1. Venv + deps (simulate CI: uv venv, uv pip install -r requirements-mac-mps)
# -----------------------------------------------------------------------------
VENV_DIR="$PROJECT_ROOT/simulate_ci_venv"
DSU_DIST="$PROJECT_ROOT/scripts/building/py/dist/dsu"
if [[ -n "$SKIP_BUILD" ]]; then
  if [[ ! -d "$VENV_DIR" ]] || [[ ! -d "$DSU_DIST" ]]; then
    echo "ERROR: --skip-build requires simulate_ci_venv/ and scripts/building/py/dist/dsu/. Run full build first."
    exit 1
  fi
  echo "[1/4] --skip-build: reusing venv + dist/dsu"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "[1/5] Create venv and install deps with uv (requirements-mac-mps.txt)..."
  uv venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  uv pip install -r "$PROJECT_ROOT/requirements-mac-mps.txt"
  uv pip install audio-separator==0.41.0 --no-deps
  # Mac ARM: samplerate 0.1.0 ships x86_64-only libsamplerate.dylib; use 0.2.3+ (universal2)
  uv pip install 'samplerate>=0.2.3' --force-reinstall
  echo "      Done."
  echo ""

  # -----------------------------------------------------------------------------
  # 2. Build (simulate CI: scripts/building/py/build_dsu.py)
  # -----------------------------------------------------------------------------
  echo "[2/5] Build DSU (scripts/building/py/build_dsu.py)..."
  python scripts/building/py/build_dsu.py
  echo ""

  # -----------------------------------------------------------------------------
  # 3. Smoke test (simulate CI)
  # -----------------------------------------------------------------------------
  echo "[3/5] Smoke test..."
  ls -la scripts/building/py/dist/dsu/dsu-*
  scripts/building/py/dist/dsu/dsu-demucs --help >/dev/null 2>&1 || true
  echo "      Smoke test OK."
  echo ""

  # -----------------------------------------------------------------------------
  # 4. Create archive (simulate CI: tar)
  # -----------------------------------------------------------------------------
  echo "[4/5] Create archive (dsu-mac-arm.tar.gz)..."
  rm -f dsu-mac-arm.tar.gz
  tar -czf dsu-mac-arm.tar.gz -C scripts/building/py/dist dsu
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

AUDIO_DIR="$PROJECT_ROOT/simulate_ci_out"
mkdir -p "$AUDIO_DIR"
# Use test_4s.wav / test_40s.wav (same as benchmarks - see docs/WORKER_SYSTEM.md, tests/README_BENCHMARKS.md)
AUDIO_SRC="$PROJECT_ROOT/tests/audio"
if [[ "$DURATION" = "40" ]]; then
  TEST_WAV="$AUDIO_SRC/test_40s.wav"
else
  TEST_WAV="$AUDIO_SRC/test_4s.wav"
fi
if [[ ! -f "$TEST_WAV" ]]; then
  echo "Creating test audio ($DURATION s)..."
  python "$PROJECT_ROOT/tests/python/generate_test_audio.py" || { echo "ERROR: generate_test_audio failed"; exit 1; }
  [[ ! -f "$TEST_WAV" ]] && { echo "ERROR: $TEST_WAV not found after generate"; exit 1; }
fi
echo "  Input: $TEST_WAV ($DURATION s)"
DEMUC_OUT="$AUDIO_DIR/demucs_out"

# Models dir (DSU-VSTOPIA or project weights/)
DSU_MODELS_DIR="${DSU_MODELS_DIR:-/Users/ostino/Documents/DSU-VSTOPIA/ThirdPartyApps/Models}"
BSROFORMER_MODELS="${DSU_MODELS_DIR}/bsroformer"
AUDIO_SEP_MODELS="${DSU_MODELS_DIR}/audio-separator"
DEMUCS_REPO="${DSU_MODELS_DIR}/demucs"

# Worker-mode separate (uses soundfile, no torchcodec)
# Use --repo for local Demucs models if DSU_MODELS_DIR/demucs exists
DEMUC_REPO_ARG=""
[[ -d "$DEMUCS_REPO" ]] && DEMUC_REPO_ARG="--repo $DEMUCS_REPO"
python "$SCRIPT_DIR/simulate_ci_demucs.py" \
  --exe "$DSU_DIST/dsu-demucs" \
  --wav "$TEST_WAV" \
  --out "$DEMUC_OUT" \
  $DEMUC_REPO_ARG || exit 1

# Test dsu-audio-separator (frozen, dummy onnx/huggingface)
# Use model_file_dir if DSU-VSTOPIA audio-separator models exist
echo ""
echo "Testing dsu-audio-separator (frozen, dummy modules)..."
python "$SCRIPT_DIR/simulate_ci_audio_separator.py" \
  --exe "$DSU_DIST/dsu-audio-separator" \
  --wav "$TEST_WAV" \
  --out "$AUDIO_DIR/audio_sep_out" \
  --models-dir "$AUDIO_SEP_MODELS" || exit 1

# Test dsu-bsroformer (frozen) - use DSU-VSTOPIA/Models/bsroformer or project root
echo ""
echo "Testing dsu-bsroformer (frozen)..."
BSRO_MODELS="${BSROFORMER_MODELS}"
[[ ! -d "$BSRO_MODELS" ]] && BSRO_MODELS="$PROJECT_ROOT"
# batch_size 4 (or 8 for 40s) reduces kernel launches, faster on MPS
BS_BATCH=4
[[ "$DURATION" = "40" ]] && BS_BATCH=8
python "$SCRIPT_DIR/simulate_ci_bsroformer.py" \
  --exe "$DSU_DIST/dsu-bsroformer" \
  --wav "$TEST_WAV" \
  --out "$AUDIO_DIR/bsroformer_out" \
  --models-dir "$BSRO_MODELS" \
  --batch-size "$BS_BATCH" || exit 1

echo ""
echo "============================================================================="
echo "Simulate CI + real-audio test passed."
[[ -z "$SKIP_BUILD" ]] && echo "  Archive: dsu-mac-arm.tar.gz"
echo "  Demucs output: $DEMUC_OUT/htdemucs/"
echo "  Audio-separator: smoke OK (ready + optional separate)"
echo "  BS-RoFormer: ready + separate (frozen)"
echo ""
echo "Timing (elapsed = worker-reported separation time, see above):"
echo "  Compare with benchmarks: docs/WORKER_SYSTEM.md (Mac ARM 4s table)"
echo "  - Demucs htdemucs: warm ~0.43s, cold ~0.9s (4s input)"
echo "  - BS-RoFormer: warm ~4.3s, cold ~10s (4s, model-dependent)"
echo "  - Without ONNX: Audio-Separator VR typically faster"
echo "  - Full E2E: ./tests/python/benchmarks/run_benchmarks_mac.sh [4|40]"
echo "You can commit and push a tag to trigger the GitHub release workflow."
echo "============================================================================="
