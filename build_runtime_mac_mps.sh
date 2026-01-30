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
NONINTERACTIVE=
[[ "$1" = "-y" || "$1" = "--yes" ]] && NONINTERACTIVE=1

echo "============================================================================="
echo "Doctor Sample Unit (DSU) - Shared Runtime Builder"
echo "============================================================================="
echo "Platform: macOS Apple Silicon (MPS)"
echo "Python: 3.10"
echo "============================================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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

pip install -r "$SCRIPT_DIR/requirements-mac-mps.txt"
# Mac ARM: samplerate 0.1.0 ships x86_64-only libsamplerate.dylib; use 0.2.3+ (universal2)
pip install 'samplerate>=0.2.3' --force-reinstall

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
echo "Workers will auto-detect MPS and use GPU acceleration!"
echo "============================================================================="
