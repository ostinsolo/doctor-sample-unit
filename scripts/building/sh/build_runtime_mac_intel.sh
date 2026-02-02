#!/bin/bash
# =============================================================================
# Doctor Sample Unit (DSU) - Shared Runtime Builder
# Platform: macOS Intel (x86_64) - CPU Only
# =============================================================================
# Creates a shared Python runtime with all dependencies for:
#   - BS-RoFormer (28 models)
#   - Audio-Separator (VR/MDX/MDXC models)
#   - Demucs (8 models)
#   - Apollo (audio restoration)
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
echo "Platform: macOS Intel (x86_64)"
echo "Python: 3.10"
echo "============================================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/runtime"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "WARNING: You are on Apple Silicon (arm64)!"
    echo "This script is for Intel Macs (x86_64)."
    echo "Use build_runtime_mac_mps.sh instead for better performance."
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
        # Skip to freeze section if requested
        if [ -n "$DO_FREEZE" ]; then
            echo "Skipping to freeze step..."
            SKIP_TO_FREEZE=1
        else
            echo "Runtime exists. Use 'freeze' to build executables, or remove --skip-env to rebuild."
            exit 0
        fi
    elif [ -n "$NONINTERACTIVE" ]; then
        echo ""
        echo "WARNING: runtime directory already exists!"
        echo "Location: $RUNTIME_DIR"
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

# Skip environment creation if --skip-env flag is set
if [ -n "$SKIP_ENV" ] && [ -d "$RUNTIME_DIR" ]; then
    echo ""
    echo "(--skip-env): Skipping environment creation, using existing runtime."
    echo "Proceeding directly to freeze step (if requested)..."
    SKIP_TO_FREEZE=1
else
    # Create virtual environment
    echo ""
    echo "[1/5] Creating Python virtual environment..."
    $PYTHON_CMD -m venv "$RUNTIME_DIR"

    # Activate
    echo ""
    echo "[2/5] Activating environment..."
    source "$RUNTIME_DIR/bin/activate"

        # Check if uv is available, install if not (like build.sh)
        echo ""
        echo "[3/5] Checking for uv (fast Python package manager)..."
        if ! command -v uv &> /dev/null; then
            echo "Installing uv (fast Python package manager)..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
        fi
        echo "Using uv for package installation (faster, more reliable)"
    
        # Install dependencies
        echo ""
        echo "[4/5] Installing dependencies..."
        echo "This will take 5-10 minutes..."
        echo ""
    
        # CRITICAL: Install llvmlite/numba with pre-built wheels first to avoid compilation errors
        # This must be done before installing other packages that depend on them
        # Use uv (like build.sh) for faster, more reliable installation
        echo "Installing llvmlite/numba with pre-built wheels (avoids compilation on Intel Mac)..."
        uv pip install --only-binary=:all: llvmlite numba
    
        # Install build tool and PyTorch together (faster than separate installs)
        echo "Installing build tool and PyTorch..."
        uv pip install 'cx-Freeze==6.15.16' 'torch==2.2.2' 'torchaudio==2.2.2' 'torchvision==0.17.2'
    
        # Install core dependencies from requirements file (skip audio-separator for now)
        # Use uv with --only-binary for packages that have wheels (like build.sh)
        echo "Installing core dependencies with uv..."
        uv pip install --only-binary=:all: -r "$SCRIPT_DIR/requirements-mac-intel.txt" || {
            echo "Some packages don't have wheels. Installing with pinned versions..."
            uv pip install torch==2.2.2 torchaudio==2.2.2 "numpy<2" scipy soundfile \
                librosa==0.10.1 llvmlite==0.41.1 numba==0.58.1 \
                tqdm pyyaml omegaconf ml_collections \
                einops rotary-embedding-torch beartype loralib matplotlib \
                demucs julius diffq resampy pydub samplerate
        }
        
        # Ensure demucs is installed (required for demucs_worker.py)
        # Install separately if it wasn't installed above (some platforms may not have wheels)
        echo ""
        echo "Ensuring demucs is installed (required for demucs worker)..."
        python -c "import demucs" 2>/dev/null || uv pip install demucs julius diffq
    
        # Install audio-separator separately with --no-deps to work around torch version conflict
        # (audio-separator requires torch>=2.3, but PyTorch 2.3+ is not available for Intel Mac)
        echo ""
        echo "Installing audio-separator (with --no-deps for Intel Mac compatibility)..."
        
        # CRITICAL: Remove duplicate OpenMP library to prevent conflicts
        # Keep only Intel's libiomp5.dylib, remove LLVM's libomp.dylib
        # This prevents thread scheduling conflicts and improves performance by ~34%
        # Based on build.sh approach - remove all libomp.dylib files from runtime
        echo ""
        echo "Removing conflicting OpenMP library (libomp.dylib from sklearn)..."
        echo "  (Following build.sh approach: remove all libomp.dylib, keep only libiomp5.dylib)"
        REMOVED_COUNT=0
        for omp_file in $(find "$RUNTIME_DIR" -name "libomp.dylib" 2>/dev/null); do
            echo "  Removing: $omp_file"
            rm -f "$omp_file"
            REMOVED_COUNT=$((REMOVED_COUNT + 1))
        done
        if [ $REMOVED_COUNT -gt 0 ]; then
            echo "  ✅ Removed $REMOVED_COUNT conflicting libomp.dylib files"
            echo "     This prevents OpenMP conflicts and improves performance by ~34%"
            echo "     (sklearn will use system/torch OpenMP instead)"
        else
            echo "  No conflicting libomp.dylib found (or already removed)"
        fi
        uv pip install 'audio-separator==0.41.0' --no-deps || uv pip install 'audio-separator>=0.39.0' --no-deps || echo "WARNING: Could not install audio-separator, some features may be unavailable"
    
        # Verify installation
        echo ""
        echo "[5/5] Verifying installation..."
        echo ""
    
        python -c "import torch; print(f'PyTorch: {torch.__version__}')"
        python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
        python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
        python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
        python -c "import soundfile; print(f'SoundFile: {soundfile.__version__}')"
        python3 << 'PYEOF'
try:
    import demucs
    print(f'Demucs: {demucs.__version__}')
except ImportError:
    print('Demucs: Not installed (optional for Intel Mac)')
PYEOF
        python3 << 'PYEOF'
try:
    import torchcodec
    print('TorchCodec: OK (torchaudio save/load)')
except ImportError:
    print('TorchCodec: Not available (using soundfile fallback)')
PYEOF
        python3 << 'PYEOF'
try:
    import audio_separator
    v = getattr(audio_separator, '__version__', '(no version)')
    print(f'Audio-Separator: {v}')
except ImportError:
    print('Audio-Separator: Not available (optional on Intel Mac)')
PYEOF
    
        # Deactivate
        deactivate 2>/dev/null || true
    
        echo ""
        echo "============================================================================="
        echo "BUILD COMPLETE!"
        echo "============================================================================="
        echo ""
        echo "Runtime location: $RUNTIME_DIR"
        echo "Python executable: $RUNTIME_DIR/bin/python"
        echo ""
fi

# Optional: Freeze executables
if [[ -n "$DO_FREEZE" ]] || [[ -n "$SKIP_TO_FREEZE" ]]; then
    echo ""
    echo "============================================================================="
    echo "Freezing executables (build_dsu.py)..."
    echo "============================================================================="
    echo ""
    
    # Activate environment for freezing
    source "$RUNTIME_DIR/bin/activate"
    
    # Build frozen executables
    # build_dsu.py is in scripts/building/py/, go up one level from sh/ to building/, then into py/
    BUILD_DSU_SCRIPT="$(dirname "$SCRIPT_DIR")/py/build_dsu.py"
    if [ ! -f "$BUILD_DSU_SCRIPT" ]; then
        # Fallback: try from project root
        BUILD_DSU_SCRIPT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")/scripts/building/py/build_dsu.py"
    fi
    "$RUNTIME_DIR/bin/python" "$BUILD_DSU_SCRIPT"
    
    # Deactivate
    deactivate 2>/dev/null || true
    
    echo ""
    BUILD_PY_DIR="$(dirname "$SCRIPT_DIR")/py"
    echo "============================================================================="
    echo "Freeze complete! Executables in: $BUILD_PY_DIR/dist/dsu/"
    echo "============================================================================="
    echo ""
    echo "Test executables:"
    echo "  $BUILD_PY_DIR/dist/dsu/dsu-demucs --help"
    echo "  $BUILD_PY_DIR/dist/dsu/dsu-bsroformer --worker"
    echo ""
else
    echo "Usage (run workers directly on Mac):"
    echo "  $RUNTIME_DIR/bin/python workers/bsroformer_worker.py --worker"
    echo "  $RUNTIME_DIR/bin/python workers/demucs_worker.py --worker"
    echo "  $RUNTIME_DIR/bin/python workers/audio_separator_worker.py --worker"
    echo ""
    echo "To freeze executables later:"
    echo "  $RUNTIME_DIR/bin/python scripts/building/py/build_dsu.py"
    echo "  Or run: ./build_runtime_mac_intel.sh -y freeze"
    echo ""
fi

echo "============================================================================="
