# How to Recreate a Clean Virtual Environment

Your current `venv` folder is **3GB** because it contains packages from old experiments (TensorFlow, PyTorch, Jupyter, etc.) that are no longer needed.

## Recommended: Using `uv` (Fast & Modern)

`uv` is a fast Python package installer (written in Rust) that's much faster than pip and handles dependencies better.

### Steps with `uv`:

1. **Remove old venv**:
   ```bash
   rm -rf venv
   ```

2. **Create new venv and install packages** (all in one command):
   ```bash
   uv venv
   source venv/bin/activate  # On macOS/Linux
   uv pip install -r requirements.txt
   ```

   That's it! `uv` automatically uses pre-built wheels and is much faster.

## Alternative: Using `pip` (Traditional)

1. **Deactivate current venv** (if active):
   ```bash
   deactivate
   ```

2. **Remove old venv**:
   ```bash
   rm -rf venv
   ```

3. **Create new venv**:
   ```bash
   python3 -m venv venv
   ```

4. **Activate new venv**:
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

5. **Upgrade pip and install wheel first** (to avoid building from source):
   ```bash
   pip install --upgrade pip wheel
   ```

6. **Install only required packages** (using pre-built wheels):
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** If you get compilation errors with `llvmlite`/`numba`, use pre-built wheels:
   ```bash
   pip install --only-binary :all: -r requirements.txt
   ```

## Expected Size

The new `venv` should be around **~300MB** instead of 3GB (about 90% smaller!).

## What Gets Removed

- TensorFlow (1.1GB) - was for DeepFilterNet/VoiceFixer
- PyTorch (561MB) - was for demucs/voicefixer  
- Jupyter/Notebook (64MB) - not needed
- Many other unused packages

## What Stays

- numpy
- librosa
- soundfile
- scipy
- matplotlib (optional, for analysis)
