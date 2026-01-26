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
    read -p "Continue anyway? (y/N): " CONFIRM
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "Aborted."
        exit 1
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
    read -p "Delete and rebuild? (y/N): " CONFIRM
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "Aborted."
        exit 1
    fi
    echo "Removing previous runtime..."
    rm -rf "$RUNTIME_DIR"
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

pip install -r "$SCRIPT_DIR/requirements-mac-intel.txt"

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
python -c "import audio_separator; print(f'Audio-Separator: {audio_separator.__version__}')"

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
