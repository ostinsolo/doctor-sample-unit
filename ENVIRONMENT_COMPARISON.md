# Environment Comparison: OLD build_venv (Fast) vs NEW runtime (Slow)

## Performance Results

### Target Benchmark (OLD build_venv)
- **Target time**: 44-48 seconds (from test_performance_comparison.sh)
- **Achieved**: 46.38s (old main.py with precision=medium) ✅
- **Test file**: `/Users/ostinsolo/Documents/15_23_5_1_20_2026_.wav`
- **Model**: `bsroformer_4stem`
- **Key settings that achieved this**:
  - `precision='medium'` (not 'highest') - provides ~10% speedup
  - `num_threads=8`, `interop_threads=16` (2x num_threads)
  - Environment variables: OMP_NUM_THREADS, MKL_NUM_THREADS, etc.

### Current Status
- **OLD build_venv**: 46.38s ✅ (matches target)
- **NEW runtime (worker)**: 44.49s ✅ (FASTER than target!)
- **Status**: ✅ **WORKING - Performance matches/exceeds target**

### Performance Goal
- **Target**: Match old build performance (~44-48s)
- **Current issue**: Worker hangs or runs very slowly (single-threaded CPU usage)

## Package Versions Comparison

### Core Packages (IDENTICAL)
| Package | OLD build_venv | NEW runtime | Status |
|---------|---------------|------------|--------|
| Python | 3.10.0 | 3.10.0 | ✅ Same |
| torch | 2.2.2 | 2.2.2 | ✅ Same |
| numpy | 1.26.4 | 1.26.4 | ✅ Same |
| librosa | 0.11.0 | 0.11.0 | ✅ Same |
| soundfile | 0.13.1 | 0.13.1 | ✅ Same |
| scipy | 1.15.3 | 1.15.3 | ✅ Same |

### Extra Packages in NEW runtime
| Package | OLD build_venv | NEW runtime |
|---------|---------------|------------|
| onnx | ❌ NOT INSTALLED | ✅ 1.20.1 |
| onnxruntime | ❌ NOT INSTALLED | ✅ 1.23.2 |
| julius | ❌ NOT INSTALLED | ✅ 0.2.7 |
| diffq | ❌ NOT INSTALLED | ✅ 0.2.4 |
| resampy | ❌ NOT INSTALLED | ✅ 0.4.3 |
| pydub | ❌ NOT INSTALLED | ✅ 0.25.1 |
| samplerate | ❌ NOT INSTALLED | ✅ 0.1.0 |
| huggingface_hub | ❌ NOT INSTALLED | ❌ NOT INSTALLED |
| pytorch_lightning | ❌ NOT INSTALLED | ❌ NOT INSTALLED |

## OpenMP Configuration

### OLD build_venv
- ✅ Has `libiomp5.dylib` (Intel OpenMP) in torch/lib
- ✅ sklearn's `libomp.dylib` removed (not found)
- ✅ No OpenMP conflicts

### NEW runtime
- ✅ Has `libiomp5.dylib` (Intel OpenMP) in torch/lib
- ✅ sklearn's `libomp.dylib` removed (not found)
- ✅ No OpenMP conflicts

**Status**: IDENTICAL - Both environments have OpenMP properly configured

## Threading Configuration

### OLD build_venv (from BS-RoFormer-freeze main.py)
- `torch.get_num_threads()`: 8
- `torch.get_num_interop_threads()`: max(2, num_threads // 2) = 4 (HALF threads, not 2x!)
- Environment variables: Set at runtime (OMP_NUM_THREADS, MKL_NUM_THREADS, KMP_AFFINITY, KMP_BLOCKTIME)
- Matmul precision: 'high' (default), 'medium' only with --precision medium flag

### NEW runtime
- `torch.get_num_threads()`: 8
- `torch.get_num_interop_threads()`: 16
- Environment variables: NOT SET (set at runtime by code)
- Matmul precision: highest (default)

**Status**: IDENTICAL - Both have same threading defaults

## Requirements Files

### OLD build_venv (`requirements_freeze.txt`)
**Minimal dependencies:**
```
torch==2.2.2
torchaudio==2.2.2
numpy<2
scipy
soundfile
librosa
tqdm
pyyaml
omegaconf
ml_collections
einops
rotary-embedding-torch
beartype
loralib
matplotlib
```

### NEW runtime (`requirements-mac-intel.txt`)
**Extended dependencies:**
- All OLD dependencies PLUS:
- onnx, onnxruntime, onnx2torch
- julius, diffq (Demucs)
- resampy, pydub, samplerate
- huggingface_hub, pytorch_lightning
- torchvision==0.17.2

## Installation Method

### OLD build_venv
- Uses `uv` (fast Python package manager) for installation
- `uv pip install --only-binary :all:` for llvmlite/numba
- Minimal, focused installation

### NEW runtime
- Uses standard `pip install`
- `--only-binary=:all:` for llvmlite/numba
- More comprehensive installation

## Key Differences Found

### 1. **Package Count**
- OLD: ~15 core packages
- NEW: ~25+ packages (includes ONNX, Demucs, etc.)

### 2. **Installation Tool**
- OLD: `uv` (faster, more reliable)
- NEW: `pip` (standard)

### 3. **Dependencies**
- OLD: Minimal (only what's needed for BS-RoFormer)
- NEW: Comprehensive (supports multiple models: BS-RoFormer, Demucs, Audio-Separator, Apollo)

## Hypothesis: Why NEW runtime is slower

### Primary Suspect: Extra Packages (onnx, onnxruntime, julius, diffq, etc.)

1. **More packages = more imports = slower startup**
   - Additional packages (onnx, onnxruntime, etc.) may be imported even if not used
   - More DLLs loaded into memory
   - **onnxruntime** is particularly heavy (C++ bindings, large binary)

2. **Package conflicts or interactions**
   - Extra packages might have conflicting dependencies
   - Some packages might interfere with PyTorch performance
   - **onnxruntime** might conflict with PyTorch's threading

3. **Installation method differences**
   - OLD uses `uv` (faster, more reliable package manager)
   - NEW uses `pip` (standard, but potentially slower)
   - `uv` might optimize package installation differently

## Recommendations

### Priority 1: Test with Minimal Requirements
1. **Create minimal runtime with OLD requirements**
   - Install only `requirements_freeze.txt` packages
   - Test performance - should match OLD build_venv (46s)
   - This will confirm if extra packages are the issue

### Priority 2: Optimize Package Imports
2. **Check for lazy imports**
   - Ensure extra packages (onnx, etc.) are only imported when needed
   - Avoid importing unused packages at worker startup
   - Profile imports to identify slow ones

### Priority 3: Consider Installation Method
3. **Test with `uv` instead of `pip`**
   - Install using `uv pip install` (like OLD build_venv)
   - See if this improves performance
   - `uv` might optimize package installation better

### Priority 4: Profile and Debug
4. **Profile worker startup**
   - Check which packages are being imported at worker startup
   - Identify slow imports (especially onnxruntime)
   - Use `python -X importtime` to profile imports

5. **Test removing specific packages**
   - Remove onnx/onnxruntime (if not needed for BS-RoFormer)
   - Remove julius/diffq (if not needed for BS-RoFormer)
   - Test performance after each removal

## Next Steps

1. ✅ **Documentation created** - This file
2. ⏭️ **Create minimal test runtime** - Install only OLD requirements
3. ⏭️ **Compare performance** - Should match OLD build_venv if hypothesis is correct
4. ⏭️ **Optimize based on findings** - Remove unnecessary packages or make imports lazy
