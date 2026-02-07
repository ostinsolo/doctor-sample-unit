#!/bin/bash
# =============================================================================
# Doctor Sample Unit (DSU) - Shared Runtime Builder
# Platform: macOS Apple Silicon (arm64) - MPS GPU Acceleration
# =============================================================================
# Creates a shared Python runtime with all dependencies for:
#   - BS-RoFormer (28 models)
#   - Audio-Separator (VR/MDX/MDXC models)
#   - Demucs (8 models)
#   - Apollo (audio restoration)
#
# Uses Metal Performance Shaders (MPS) for GPU acceleration on Apple Silicon
# =============================================================================

set -e  # Exit on error

# -y / --yes: non-interactive (skip confirmations, rebuild runtime if exists)
# freeze: also build frozen executables after creating runtime
# --skip-env: skip environment rebuild, only freeze (for Python file changes only)
NONINTERACTIVE=
DO_FREEZE=
SKIP_ENV=
for arg in "$@"; do
    [[ "$arg" = "-y" || "$arg" = "--yes" ]] && NONINTERACTIVE=1
    [[ "$arg" = "freeze" ]] && DO_FREEZE=1
    [[ "$arg" = "--skip-env" ]] && SKIP_ENV=1
done

echo "============================================================================="
echo "Doctor Sample Unit (DSU) - Shared Runtime Builder"
echo "============================================================================="
echo "Platform: macOS Apple Silicon (MPS)"
echo "Python: 3.10"
echo "============================================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/runtime"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "WARNING: You are NOT on Apple Silicon!"
    echo "Detected architecture: $ARCH"
    echo "This script is optimized for Apple Silicon (M1/M2/M3/M4)."
    echo "Use build_runtime_mac_intel.sh for Intel Macs."
    echo ""
    if [ -n "$NONINTERACTIVE" ]; then
        echo "(-y): Continuing anyway."
    else
        read -p "Continue anyway? (y/N): " CONFIRM
        if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
            echo "Aborted."
            exit 1
        fi
    fi
fi

# Check macOS version for MPS support (requires 12.3+)
MACOS_VERSION=$(sw_vers -productVersion)
echo "macOS version: $MACOS_VERSION"

# Check Python
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Install with: brew install python@3.10"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "WARNING: Python 3.10 recommended for best compatibility"
    echo ""
fi

# Check if runtime already exists
if [ -d "$RUNTIME_DIR" ]; then
    if [ -n "$SKIP_ENV" ]; then
        echo ""
        echo "(--skip-env): Skipping environment rebuild, using existing runtime."
        echo "Location: $RUNTIME_DIR"
        echo ""
        if [ -n "$DO_FREEZE" ]; then
            echo "Skipping to freeze step..."
            SKIP_TO_FREEZE=1
        else
            echo "Runtime exists. Use 'freeze' to build executables, or remove --skip-env to rebuild."
            exit 0
        fi
    else
        echo ""
        echo "WARNING: runtime directory already exists!"
        echo "Location: $RUNTIME_DIR"
        echo ""
        if [ -n "$NONINTERACTIVE" ]; then
            echo "(-y): Deleting and rebuilding."
            rm -rf "$RUNTIME_DIR"
        else
            read -p "Delete and rebuild? (y/N): " CONFIRM
            if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
                echo "Aborted."
                exit 1
            fi
            echo "Removing previous runtime..."
            rm -rf "$RUNTIME_DIR"
        fi
    fi
fi

# Skip environment creation if --skip-env
if [ -n "$SKIP_ENV" ] && [ -d "$RUNTIME_DIR" ]; then
    echo ""
    echo "(--skip-env): Skipping environment creation, using existing runtime."
    echo "Proceeding directly to freeze step (if requested)..."
    SKIP_TO_FREEZE=1
fi

# Create virtual environment (unless skipping to freeze)
if [ -z "$SKIP_TO_FREEZE" ]; then
echo ""
echo "[1/5] Creating Python virtual environment..."
$PYTHON_CMD -m venv "$RUNTIME_DIR"

# Activate
echo ""
echo "[2/5] Activating environment..."
source "$RUNTIME_DIR/bin/activate"

# Check if uv is available, install if not (like build_runtime_mac_intel.sh)
echo ""
echo "[3/5] Checking for uv (fast Python package manager)..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "Using uv for package installation (faster, more reliable)"

# FFmpeg (required by torchcodec for Demucs save/load)
if ! brew list ffmpeg &>/dev/null; then
    echo ""
    echo "Installing FFmpeg (required by torchcodec)..."
    brew install ffmpeg
fi

# Install dependencies
echo ""
echo "[4/5] Installing dependencies (MPS optimized)..."
echo "This will take 5-10 minutes..."
echo ""

REQ_FILE="$PROJECT_ROOT/requirements-mac-mps.txt"
if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: Requirements file not found: $REQ_FILE"
    exit 1
fi
# Use uv (like build_runtime_mac_intel.sh) - requirements from project root
uv pip install -r "$REQ_FILE"

# CRITICAL: Remove conflicting OpenMP library (mirror Intel script)
# On Intel: remove ALL libomp.dylib, keep libiomp5.dylib (torch ships libiomp5)
# On ARM: remove only sklearn's libomp (torch ships libomp, no libiomp5 - must keep torch's)
echo ""
echo "Removing conflicting OpenMP library (libomp.dylib from sklearn)..."
echo "  (sklearn's libomp conflicts with torch; sklearn will use torch's OpenMP instead)"
REMOVED_COUNT=0
for omp_file in $(find "$RUNTIME_DIR" -path "*sklearn*" -name "libomp.dylib" 2>/dev/null); do
    echo "  Removing: $omp_file"
    rm -f "$omp_file"
    REMOVED_COUNT=$((REMOVED_COUNT + 1))
done
if [ $REMOVED_COUNT -gt 0 ]; then
    echo "  ✅ Removed $REMOVED_COUNT sklearn libomp.dylib (keeps torch's libomp on ARM)"
else
    echo "  No sklearn libomp.dylib found (or already removed)"
fi

uv pip install audio-separator==0.41.0 --no-deps
# Mac ARM: samplerate 0.1.0 ships x86_64-only libsamplerate.dylib; use 0.2.3+ (universal2)
uv pip install 'samplerate>=0.2.3' --force-reinstall

# Verify installation
echo ""
echo "[5/5] Verifying installation..."
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import torch; print(f'MPS built: {torch.backends.mps.is_built()}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
python -c "import soundfile; print(f'SoundFile: {soundfile.__version__}')"
python -c "import demucs; print(f'Demucs: {demucs.__version__}')"
python -c "import torchcodec; print('TorchCodec: OK (torchaudio save/load)')"
python -c "import audio_separator; v=getattr(audio_separator,'__version__','(no version)'); print(f'Audio-Separator: {v}')"
python -c "import samplerate; print(f'samplerate: {samplerate.__version__}')"

# Test MPS
echo ""
echo "Testing MPS acceleration..."
python -c "
import torch
if torch.backends.mps.is_available():
    x = torch.randn(1000, 1000, device='mps')
    y = x @ x.T
    print('MPS test: SUCCESS - GPU acceleration working!')
else:
    print('MPS test: FAILED - Will use CPU (slower)')
"

# Deactivate
deactivate 2>/dev/null || true

# Mark runtime platform (avoid mixing Intel vs MPS runtimes)
echo "mps" > "$RUNTIME_DIR/RUNTIME_PLATFORM.txt"

echo ""
echo "============================================================================="
echo "BUILD COMPLETE!"
echo "============================================================================="
echo ""
echo "Runtime location: $RUNTIME_DIR"
echo "Platform: Apple Silicon (MPS) - RUNTIME_PLATFORM.txt=mps"
echo "Python executable: $RUNTIME_DIR/bin/python"
fi

# Optional: Freeze executables
if [[ -n "$DO_FREEZE" ]] || [[ -n "$SKIP_TO_FREEZE" ]]; then
    echo ""
    echo "============================================================================="
    echo "Freezing executables (build_dsu.py)..."
    echo "============================================================================="
    echo ""
    source "$RUNTIME_DIR/bin/activate"
    BUILD_DSU_SCRIPT="$(dirname "$SCRIPT_DIR")/py/build_dsu.py"
    if [ ! -f "$BUILD_DSU_SCRIPT" ]; then
        BUILD_DSU_SCRIPT="$PROJECT_ROOT/scripts/building/py/build_dsu.py"
    fi
    "$RUNTIME_DIR/bin/python" "$BUILD_DSU_SCRIPT"
    deactivate 2>/dev/null || true
    echo ""
    BUILD_PY_DIR="$(dirname "$SCRIPT_DIR")/py"
    echo "============================================================================="
    echo "Freeze complete! Executables in: $BUILD_PY_DIR/dist/dsu/"
    echo "============================================================================="
    echo ""
    echo "Test: $BUILD_PY_DIR/dist/dsu/dsu-bsroformer --worker"
    echo ""
else
    echo "Usage (run workers directly on Mac):"
    echo "  $RUNTIME_DIR/bin/python workers/bsroformer_worker.py --worker"
    echo "  $RUNTIME_DIR/bin/python workers/demucs_worker.py --worker"
    echo "  $RUNTIME_DIR/bin/python workers/audio_separator_worker.py --worker"
    echo ""
    echo "To freeze: ./build_runtime_mac_mps.sh -y freeze"
    echo ""
fi

echo "============================================================================="
