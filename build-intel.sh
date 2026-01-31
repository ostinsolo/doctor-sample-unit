#!/bin/bash
# =============================================================================
# Build script for audio-separator-cxfreeze (Intel macOS)
# =============================================================================
# Creates a frozen binary for Intel Macs with audio-separator + Apollo.
#
# Key notes:
# - Uses torch 2.2.2 (Intel compatible, not 2.3+ which needs newer macOS)
# - Uses numpy <2 (required for torch 2.2.2 compatibility)
# - Uses --only-binary=:all: for llvmlite/numba to avoid compilation errors
# - Based on: https://github.com/nomadkaraoke/python-audio-separator
#
# Performance notes (from testing):
# - VR models (.pth) are slightly faster than MDX (.onnx) on Intel
# - julius/diffq only needed for Demucs models
# - beartype only needed for BS-RoFormer models
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_NAME="venv-intel-build"
TARGET_DIR="audio-separator-intel"

echo "========================================"
echo " Audio Separator + Apollo Build (Intel)"
echo "========================================"

# Step 1: Create fresh venv
echo ""
echo "=== Step 1: Creating fresh venv ==="
rm -rf "$VENV_NAME"
python3 -m venv "$VENV_NAME"

# Use venv python directly
VENV_PIP="./$VENV_NAME/bin/pip"
VENV_PYTHON="./$VENV_NAME/bin/python"

# Upgrade pip
$VENV_PIP install --upgrade pip

# Step 2: Install dependencies
echo ""
echo "=== Step 2: Installing dependencies ==="

# Build tool
$VENV_PIP install 'cx-Freeze==6.15.16'

# CRITICAL: Install llvmlite/numba with pre-built wheels to avoid compilation
$VENV_PIP install --only-binary=:all: llvmlite numba

# PyTorch (Intel compatible - torch 2.2.2 works, 2.3+ may not)
$VENV_PIP install 'numpy<2' 'torch==2.2.2' 'torchaudio==2.2.2'

# Audio separator core (no-deps to avoid version conflicts)
$VENV_PIP install --no-deps audio-separator

# Dependencies from python-audio-separator pyproject.toml
$VENV_PIP install \
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
    soundfile

# Additional for cx_Freeze bundling
$VENV_PIP install torchvision pytorch-lightning huggingface_hub omegaconf

# Step 3: Verify imports
echo ""
echo "=== Step 3: Verifying imports ==="
$VENV_PYTHON -c "
import sys
sys.path.insert(0, '.')
from audio_separator.separator import Separator
from apollo.apollo_separator import main
print('Core imports OK!')
"

# Step 4: Build
echo ""
echo "=== Step 4: Building frozen binary ==="
rm -rf "$TARGET_DIR"

$VENV_PYTHON -m cx_Freeze main.py \
    --target-dir="$TARGET_DIR" \
    --target-name=audio-separator \
    --packages=audio_separator,onnxruntime,samplerate,apollo,apollo.look2hear,apollo.look2hear.models,soundfile,omegaconf,scipy,requests,librosa,pydub,einops,julius,diffq,resampy \
    --include-files=apollo

# Make executable
chmod +x "$TARGET_DIR/audio-separator"

# Step 5: Test
echo ""
echo "=== Step 5: Testing binary ==="
./"$TARGET_DIR/audio-separator" --help | head -5

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo " BUILD COMPLETE!"
    echo "========================================"
    echo ""
    echo "Binary: $SCRIPT_DIR/$TARGET_DIR/audio-separator"
    echo "Size: $(du -sh "$TARGET_DIR" | cut -f1)"
    echo ""
    echo "To install to SplitWizard:"
    echo "  rm -rf \"/Users/\$USER/Documents/Max 9/SplitWizard/ThirdPartyApps/audio-separator/audio-separator-cxfreeze\""
    echo "  cp -r $TARGET_DIR \"/Users/\$USER/Documents/Max 9/SplitWizard/ThirdPartyApps/audio-separator/audio-separator-cxfreeze\""
else
    echo "BUILD FAILED - binary test failed"
    exit 1
fi
