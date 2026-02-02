#!/usr/bin/env bash
# =============================================================================
# DSU local Mac setup: create environment, smoke-test workers, then freeze
# =============================================================================
# 1. Creates runtime/ venv (MPS on Apple Silicon, Intel on x86_64)
# 2. Smoke-tests workers via ./run_local.sh
# 3. Optionally runs freeze (build_dsu.py) to produce dist/dsu/ executables
#
# Usage:
#   ./setup_local_mac.sh           # interactive
#   ./setup_local_mac.sh -y        # non-interactive (rebuild runtime if exists)
#   ./setup_local_mac.sh -y freeze # same + run freeze after smoke test
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

NONINTERACTIVE=
DO_FREEZE=
SKIP_ENV=
for a in "$@"; do
    [[ "$a" = "-y" || "$a" = "--yes" ]] && NONINTERACTIVE=1
    [[ "$a" = "freeze" ]] && DO_FREEZE=1
    [[ "$a" = "--skip-env" ]] && SKIP_ENV=1
done

RUN_ARGS=()
[[ -n "$NONINTERACTIVE" ]] && RUN_ARGS=(-y)
[[ -n "$SKIP_ENV" ]] && RUN_ARGS+=("--skip-env")

echo "============================================================================="
echo "DSU Local Mac Setup"
echo "============================================================================="
echo ""

ARCH=$(uname -m)
if [[ "$ARCH" = "arm64" ]]; then
    echo "Detected: Apple Silicon (arm64) -> build_runtime_mac_mps.sh"
    RUNTIME_SCRIPT="$PROJECT_ROOT/scripts/building/sh/build_runtime_mac_mps.sh"
else
    echo "Detected: Intel (x86_64) -> build_runtime_mac_intel.sh"
    RUNTIME_SCRIPT="$PROJECT_ROOT/scripts/building/sh/build_runtime_mac_intel.sh"
fi
echo ""

# -----------------------------------------------------------------------------
# 1. Create runtime environment (skip if --skip-env)
# -----------------------------------------------------------------------------
if [[ -n "$SKIP_ENV" ]]; then
    echo "[1/3] Skipping runtime environment (--skip-env flag)"
    echo "      Using existing runtime, proceeding to freeze..."
    echo ""
else
    echo "[1/3] Creating runtime environment..."
    if [[ -n "$NONINTERACTIVE" ]]; then
        $RUNTIME_SCRIPT -y
    else
        $RUNTIME_SCRIPT
    fi
    echo ""
fi

# -----------------------------------------------------------------------------
# 2. Smoke-test workers
# -----------------------------------------------------------------------------
echo "[2/3] Smoke-testing workers (run_local.sh demucs --help)..."
"$PROJECT_ROOT/scripts/run_local.sh" demucs --help
echo "[2/3] Smoke-test: demucs --help OK"
echo ""

# Runtime lives in scripts/building/sh/runtime after running the runtime script
RUNTIME_PY="$PROJECT_ROOT/scripts/building/sh/runtime/bin/python"
BUILD_DSU="$PROJECT_ROOT/scripts/building/py/build_dsu.py"
DSU_DIST="$PROJECT_ROOT/scripts/building/py/dist/dsu"

# -----------------------------------------------------------------------------
# 3. Freeze (optional)
# -----------------------------------------------------------------------------
if [[ -n "$DO_FREEZE" ]]; then
    echo "[3/3] Freezing executables (build_dsu.py)..."
    # Use optimize=1 for faster builds (removes assertions, safe for torch)
    # Set DSU_BUILD_OPTIMIZE=2 for maximum speed (removes docstrings too, may break introspection)
    export DSU_BUILD_OPTIMIZE=${DSU_BUILD_OPTIMIZE:-1}
    echo "  Using bytecode optimization level: $DSU_BUILD_OPTIMIZE"
    "$RUNTIME_PY" "$BUILD_DSU"
    echo ""
    echo "Freeze complete. Output: scripts/building/py/dist/dsu/"
    echo "  $DSU_DIST/dsu-demucs --help"
    echo "  $DSU_DIST/dsu-demucs --worker"
else
    echo "[3/3] Skip freeze (run with 'freeze' to build: ./setup_local_mac.sh -y freeze)"
    echo ""
    echo "To freeze locally later:"
    echo "  $RUNTIME_PY $BUILD_DSU"
    echo "  -> scripts/building/py/dist/dsu/dsu-demucs, dsu-bsroformer, dsu-audio-separator (macOS)"
fi

echo ""
echo "============================================================================="
echo "Setup complete."
echo "============================================================================="
echo ""
echo "Test workers:"
echo "  ./scripts/run_local.sh demucs --help"
echo "  ./scripts/run_local.sh demucs --worker"
echo "  ./scripts/run_local.sh bsroformer --help"
echo "  ./scripts/run_local.sh audio-separator --help"
echo ""
echo "Freeze (create Mac executables):"
echo "  scripts/building/sh/runtime/bin/python scripts/building/py/build_dsu.py"
echo ""
