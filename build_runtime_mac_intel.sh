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
NONINTERACTIVE=
[[ "$1" = "-y" || "$1" = "--yes" ]] && NONINTERACTIVE=1

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

# Create virtual environment
echo ""
echo "[1/5] Creating Python virtual environment..."
$PYTHON_CMD -m venv "$RUNTIME_DIR"

# Activate
echo ""
echo "[2/5] Activating environment..."
source "$RUNTIME_DIR/bin/activate"

# Upgrade pip
echo ""
echo "[3/5] Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "[4/5] Installing dependencies..."
echo "This will take 5-10 minutes..."
echo ""

# CRITICAL: Install llvmlite/numba with pre-built wheels first to avoid compilation errors
# This must be done before installing other packages that depend on them
echo "Installing llvmlite/numba with pre-built wheels (avoids compilation on Intel Mac)..."
pip install --only-binary=:all: llvmlite numba

# Install build tool
echo "Installing build tool (cx-Freeze)..."
pip install 'cx-Freeze==6.15.16'

# Install PyTorch and core dependencies from requirements file
echo "Installing PyTorch and core dependencies..."
pip install -r "$SCRIPT_DIR/requirements-mac-intel.txt"

# Install audio-separator separately with --no-deps to work around torch version conflict
# (audio-separator requires torch>=2.3, but PyTorch 2.3+ is not available for Intel Mac)
# Then install its dependencies manually (matching build-intel.sh approach)
echo ""
echo "Installing audio-separator (with --no-deps for Intel Mac compatibility)..."
pip install 'audio-separator==0.41.0' --no-deps || pip install 'audio-separator>=0.39.0' --no-deps || echo "WARNING: Could not install audio-separator, some features may be unavailable"

# Install audio-separator dependencies manually (from build-intel.sh)
# These are already in requirements-mac-intel.txt, but ensuring they're installed
echo "Ensuring audio-separator dependencies are installed..."
pip install \
    requests \
    librosa \
    samplerate \
    six \
    tqdm \
    pydub \
    onnx \
    onnx2torch \
    onnxruntime \
    julius \
    diffq \
    einops \
    pyyaml \
    ml_collections \
    resampy \
    beartype \
    rotary-embedding-torch \
    scipy \
    soundfile \
    pytorch-lightning \
    huggingface_hub \
    omegaconf \
    torchvision

# Verify installation
echo ""
echo "[5/5] Verifying installation..."
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
python -c "import soundfile; print(f'SoundFile: {soundfile.__version__}')"
python -c "import demucs; print(f'Demucs: {demucs.__version__}')"
python -c "try:
    import torchcodec
    print('TorchCodec: OK (torchaudio save/load)')
except ImportError:
    print('TorchCodec: Not available (using soundfile fallback)')"
python -c "try:
    import audio_separator
    v=getattr(audio_separator,'__version__','(no version)')
    print(f'Audio-Separator: {v}')
except ImportError:
    print('Audio-Separator: Not available (optional on Intel Mac)')"

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
echo "Usage (run workers directly on Mac):"
echo "  $RUNTIME_DIR/bin/python workers/bsroformer_worker.py --worker"
echo "  $RUNTIME_DIR/bin/python workers/demucs_worker.py --worker"
echo "  $RUNTIME_DIR/bin/python workers/audio_separator_worker.py --worker"
echo ""
echo "============================================================================="
