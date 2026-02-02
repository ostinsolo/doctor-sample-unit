# Doctor Sample Unit (DSU) - Shared Runtime

A properly frozen shared runtime for professional audio source separation. Three small executables (~15KB each) share a common `lib/` folder containing all dependencies - no Python installation required.

## Key Benefits

| Metric | Separate Executables | DSU Shared Runtime |
|--------|---------------------|-------------------|
| **Total Size** | ~6 GB (3 x ~2GB) | **~2 GB** |
| **Startup** | ~8s (torch loads each time) | **~3s** (once per worker) |
| **Cached Jobs** | N/A (restart each time) | **0.2-1s** (models stay loaded) |
| **Python Required** | No | **No** |

## Architecture

```
dsu/
├── dsu-demucs[.exe]         # Demucs worker (~15KB entry point)
├── dsu-bsroformer[.exe]     # BS-RoFormer worker (~15KB entry point)
├── dsu-audio-separator[.exe]# Audio Separator + Apollo (~15KB entry point)
├── lib/                     # Shared frozen dependencies (~2GB)
│   ├── torch/              # PyTorch 2.10.x (CUDA 12.6 on Win, MPS on Mac ARM)
│   ├── numpy/
│   ├── librosa/
│   └── ...
├── models/                  # Model architectures (bs_roformer, scnet, etc.)
├── configs/                 # Model configs (YAML files)
└── VERSION.txt
```

- **Windows**: `dsu-*.exe`. **macOS/Linux**: no extension (e.g. `dsu-demucs`).
- All three executables share the same `lib/` folder - **~4GB disk space saved** compared to bundling each separately!

## Supported Architectures

| Architecture | Models | Best For | GPU Support |
|-------------|--------|----------|-------------|
| **Demucs** | htdemucs, htdemucs_ft, htdemucs_6s, mdx, mdx_extra, mdx_q | 4/6 stem separation | CUDA, MPS |
| **BS-RoFormer** | 28+ models | Vocals, Denoise, Dereverb, Drumsep | CUDA, MPS |
| **Audio-Separator** | VR, MDX, MDXC | Various separation tasks | CUDA, MPS |
| **Apollo** | lew_uni, lew_v2, edm_big, official | Audio restoration | CUDA, MPS |

## Performance (Persistent Workers)

The key advantage: **workers stay running, torch loads once, models stay cached in GPU memory**.

## Benchmarking / Timing (reproducible)

**Models dir**: `~/Documents/DSU/ThirdPartyApps/Models` (or `DSU_MODELS`). Contains `bsroformer/`, `audio-separator/`, `apollo/`, `demucs/`.

**Test audio**: `tests/audio/`. Create with `python tests/generate_test_audio.py` (adds `test_4s.wav`, `test_40s.wav`, etc.).

**Mac (MPS)**: `./tests/run_benchmarks_mac.sh` runs Demucs, BS-RoFormer, Audio-Separator, Apollo. Use `./tests/run_benchmarks_mac.sh 40` for 40s input. See `tests/README.md` (index), `tests/README_BENCHMARKS.md`, `tests/README_NODE_TESTS.md` (Max/MSP).

**Windows (CUDA)** – example:

```batch
runtime\Scripts\python.exe tests\benchmark_worker_e2e.py ^
  --exe dist\dsu\dsu-bsroformer.exe ^
  --worker bsroformer ^
  --models-dir "%USERPROFILE%\Documents\DSU\ThirdPartyApps\Models\bsroformer" ^
  --model bsrofo_sw ^
  --input "tests\audio\test_4s.wav" ^
  --output-dir "tests\benchmark_output\bsroformer" ^
  --device cuda ^
  --timeout 1200
```

This prints a JSON summary:
- **`t_start_to_first_status_s`**: process start → first JSON line (shared runtime / DLL / Python bootstrap)
- **`t_start_to_ready_s`**: process start → worker `ready` (imports + basic init)
- **`t_load_model_s`**: `load_model` command → `model_loaded`
- **`t_separate_*_wall_s`**: `separate` command → `done` (wall-clock)
- **`done.elapsed`**: worker-reported separation time (inside the model pipeline)

### Example numbers (Windows – RTX 3070 Laptop, Torch 2.10.0+cu126, SageAttention)

From `dist\dsu\dsu-bsroformer.exe` using model `bsrofo_sw`:

- **Shared runtime / process bootstrap**: ~0.07s to first JSON
- **Worker ready (imports/init)**: ~2.18s
- **Model load**: ~4.75s
- **Separation (cold)**: ~18.94s
- **Separation (warm, cached)**: ~0.91s

### Example numbers (Mac ARM – PyTorch 2.10, MPS, 4s test)

From `dist/dsu/dsu-demucs` (htdemucs) and `dsu-bsroformer` (bsrofo_sw), `tests/run_benchmarks_mac.sh`:

- **Demucs**: ready ~0.64s, model load ~0.19s, separate cold ~0.9s, warm ~0.43s
- **BS-RoFormer**: ready ~0.73s, model load ~18.5s, separate cold ~10.2s, warm ~4.3s

### First Batch (cold start, model loading)
| Worker | Processing | Total |
|--------|-----------|-------|
| Demucs | 2.5s | 8.3s |
| BS-RoFormer | 3.3s | 8.4s |
| Apollo | 12.0s | 12.0s |

### Second Batch (warm, models cached)
| Worker | Processing | Total | Speedup |
|--------|-----------|-------|---------|
| Demucs | 0.24s | **0.24s** | **35x** |
| BS-RoFormer | 0.42s | **0.43s** | **20x** |
| Apollo | 0.99s | **1.01s** | **12x** |

## Quick Start

### Windows

```batch
REM Extract the release
7z x dsu-win-cuda.7z

REM Start a persistent worker
dsu\dsu-demucs.exe --worker

REM In another terminal, or via Node.js, send JSON commands to stdin
```

### macOS Apple Silicon

```bash
tar -xzf dsu-mac-arm.tar.gz
cd dsu

# Start worker
./dsu-demucs --worker
```

## Worker JSON Protocol

All workers use stdin/stdout JSON communication - perfect for Node.js/Max integration.

### Starting a Worker

```batch
dsu-demucs.exe --worker
```

The worker outputs a ready message, then waits for JSON commands on stdin:
```json
{"status": "ready", "device": "cuda", "threads": 8}
```

### Commands

**Demucs:**
```json
{"cmd": "separate", "input": "C:/audio/song.wav", "output": "C:/output/", "model": "htdemucs"}
```

**BS-RoFormer (with direct model path):**
```json
{"cmd": "separate", "input": "C:/audio/song.wav", "output_dir": "C:/output/", "model_path": "C:/Models/bsroformer/weights/resurrection_vocals.ckpt", "config_path": "C:/dsu/configs/config_resurrection_vocals.yaml", "extract_instrumental": true}
```

**Apollo (audio restoration):**
```json
{"cmd": "apollo", "input": "C:/audio/compressed.wav", "output": "C:/output/restored.wav", "model_path": "C:/Models/apollo/apollo_lew_v2.ckpt", "config_path": "C:/Models/apollo/apollo_lew_v2.yaml"}
```

**Other commands:**
```json
{"cmd": "ping"}
{"cmd": "exit"}
```

### Responses

```json
{"status": "ready", "device": "cuda", "threads": 8}
{"status": "loading_model", "model": "htdemucs"}
{"status": "separating", "input": "song.wav"}
{"status": "done", "elapsed": 2.47, "output_dir": "C:/output/htdemucs/song", "files": ["drums.wav", "bass.wav", "other.wav", "vocals.wav"], "stems": ["drums", "bass", "other", "vocals"]}
{"status": "error", "message": "..."}
{"status": "exiting"}
```

## Ensemble (BS-RoFormer)

Ensembling means **running multiple models on the same input** and then **merging the resulting stems**. This is useful when:

- You want **higher perceived quality** (less artifacts / less bleed) than any single model.
- Different models are strong on different sources (e.g., one is better drums, another better vocals).
- You want a more “stable” output on tricky mixes.

### Two kinds of “ensemble” supported

### 1) Model ensemble (recommended)

This runs multiple models and merges each stem with an algorithm (default `avg_wave`).

- **What it does**: model A separates → model B separates → merge `vocals.wav` with `vocals.wav`, etc.
- **Output layout**: `--store_dir\<input_basename>\{stem}.wav`
- **Performance**: runs models sequentially, so it’s slower than a single model (but often higher quality).

#### CLI example (Windows, using an external model pack)

```batch
REM Run an ensemble of two 4-stem models
runtime\Scripts\python.exe workers\bsroformer_worker.py ^
  --models-dir "C:\Users\soloo\Documents\DSU-VSTOPIA\ThirdPartyApps\Models\bsroformer" ^
  --ensemble bsroformer_4stem,scnet_xl_ihf ^
  --ensemble-type avg_wave ^
  --input_folder "C:\Users\soloo\Documents\0_20_56_1_27_2026_.wav" ^
  --store_dir "C:\Users\soloo\Desktop\shared_runtime\output_ensemble_test" ^
  --overlap 2 --batch-size 1 --fast
```

#### Flags

- **`--ensemble M1,M2,...`**: comma-separated model names (must exist in `models.json`)
- **`--ensemble-type`**: how stems are merged:
  - `avg_wave` (default): weighted average in waveform domain
  - `median_wave`, `min_wave`, `max_wave`
  - `avg_fft`, `median_fft`, `min_fft`, `max_fft` (merge in STFT domain)
- **`--ensemble-weights W1,W2,...`**: optional weights (same count as models). Example: `--ensemble-weights 0.7,0.3`
- **`--models-dir PATH`**: directory that contains `models.json` and `weights\...`

### 2) File ensemble utility (merge already-rendered audio files)

This is a lower-level helper that **does not run models**. It just merges audio files you already produced.

Typical use case:

- You ran two different separations (or two workers) and want to combine *only* a specific stem (e.g., two `vocals.wav` files).

#### Example

```batch
REM Merge two already-separated vocal files into one output
runtime\Scripts\python.exe workers\bsroformer_worker.py --worker --ensemble ^
  --files "C:\outA\vocals.wav" "C:\outB\vocals.wav" ^
  --type avg_wave ^
  --output "C:\out\vocals_ensemble.wav"
```

### Using ensemble “from the worker” (JSON mode)

The persistent JSON worker (`--worker` + stdin/stdout commands) currently supports:

- `load_model`
- `separate`
- `list_models`
- `get_status`

It does **not** currently implement a single JSON command like `{"cmd":"ensemble", ...}`.

If you need ensembling in a Node/Max workflow today, use one of these patterns:

- **Pattern A (simplest)**: run the **CLI model ensemble** as a subprocess (one-shot job).
- **Pattern B (persistent)**: run separation twice (two models) via worker(s), then combine stems with the **file ensemble utility** above.

## Node.js Integration

```javascript
const { spawn } = require('child_process');
const path = require('path');

// Path to DSU folder
const DSU_DIR = 'C:/path/to/dsu';

// Spawn persistent worker
const worker = spawn(
    path.join(DSU_DIR, 'dsu-demucs.exe'),
    ['--worker'],
    { cwd: DSU_DIR, stdio: ['pipe', 'pipe', 'pipe'] }
);

// Send JSON command
function sendCommand(cmd) {
    worker.stdin.write(JSON.stringify(cmd) + '\n');
}

// Handle responses
worker.stdout.on('data', (data) => {
    const lines = data.toString().trim().split('\n');
    for (const line of lines) {
        try {
            const response = JSON.parse(line);
            console.log('Status:', response.status);
            
            if (response.status === 'done') {
                console.log(`Separated in ${response.elapsed}s`);
                console.log('Output files:', response.files);
            }
        } catch (e) {
            // Non-JSON output (warnings, progress)
            console.log('[Worker]', line);
        }
    }
});

// Wait for ready, then separate
worker.stdout.once('data', () => {
    sendCommand({
        cmd: 'separate',
        input: 'C:/audio/song.wav',
        output: 'C:/output/',
        model: 'htdemucs'
    });
});

// Graceful shutdown
process.on('SIGINT', () => {
    sendCommand({ cmd: 'exit' });
    setTimeout(() => worker.kill(), 1000);
});
```

## Building from Source

### Prerequisites
- Python 3.10
- cx_Freeze 6.15.16

### Windows CUDA Build

```batch
REM Create build environment
python -m venv build_env
build_env\Scripts\activate

REM Install dependencies
pip install -r requirements-cuda.txt

REM Build
python setup.py build_exe
REM or (from project root)
python scripts\building\py\build_dsu.py

REM Output in scripts\building\py\dist\dsu\
```

### macOS Apple Silicon (MPS)

```bash
# Setup runtime + deps (PyTorch 2.10, MPS)
./scripts/setup/setup_local_mac.sh -y

# Optional: freeze executables
./scripts/building/sh/build_manual_mac.sh
# Output: scripts/building/py/dist/dsu/dsu-demucs, dsu-bsroformer, dsu-audio-separator (no .exe)

# Benchmarks (4s / 40s test audio)
./tests/run_benchmarks_mac.sh
./tests/run_benchmarks_mac.sh 40
```

See `scripts/building/sh/build_runtime_mac_mps.sh`, `scripts/building/sh/build_manual_mac.sh`, and `tests/README.md`.  
**Mac ARM + VR separation**: If you see `libsamplerate.dylib` "incompatible architecture (have 'x86_64', need 'arm64')", run `pip install 'samplerate>=0.2.3' --force-reinstall` after installing deps. See **`docs/LIBSAMPLERATE_MAC_ARM.md`**.

### Manual Build Script (Windows)

```batch
REM build_manual.bat
call build_env\Scripts\activate
rd /s /q dist\dsu 2>nul
python setup.py build_exe
echo Build complete!
```

## Technical Details

### Critical Fix: PyTorch 2.x + cx_Freeze

PyTorch 2.x uses `inspect.getsourcelines()` during import, which fails in frozen executables. The workers include a monkey-patch that must run **before** any torch imports:

```python
if getattr(sys, 'frozen', False):
    import inspect
    _original_getsourcelines = inspect.getsourcelines
    
    def _safe_getsourcelines(obj):
        try:
            return _original_getsourcelines(obj)
        except OSError:
            return ([''], 0)
    
    inspect.getsourcelines = _safe_getsourcelines
    # Also patch getsource and findsource
```

This fix is already included in all worker scripts.

### Build Options

Critical cx_Freeze settings for PyTorch compatibility:

```python
build_options = {
    "zip_include_packages": [],      # Don't compress .pyc files
    "zip_exclude_packages": "*",     # Don't zip anything
    "optimize": 0,                   # Don't optimize bytecode
    "replace_paths": [("*", "")],    # Remove absolute paths
}
```

## Platform Support

| Platform | Status | PyTorch | GPU |
|----------|--------|---------|-----|
| Windows CUDA | ✅ Ready | 2.10.x+cu126 | NVIDIA CUDA 12.6 |
| Windows CPU | ✅ Ready | 2.10.x+cpu | None |
| macOS ARM (M1/M2/M3) | ✅ Ready | **2.10.x** | MPS (Metal) |

Mac ARM uses **PyTorch 2.10** (not 2.5). See `requirements-mac-mps.txt`. Validated with `runtime` + frozen `dist/dsu/` builds.

## Troubleshooting

### "could not get source code" error
This is fixed in the worker scripts. If you see this, ensure the inspect monkey-patch runs before torch imports.

### "CUDA not available"
```batch
nvidia-smi
```
Requires NVIDIA driver supporting CUDA 12.6.

### Mac ARM: libsamplerate "incompatible architecture (x86_64 / arm64)"
VR separation uses the **samplerate** package; **0.1.0** ships an x86_64-only `.dylib`, so it fails on Apple Silicon. After `pip install -r requirements-mac-mps.txt`, run:
```bash
pip install 'samplerate>=0.2.3' --force-reinstall
```
Then rebuild. Full explanation: **`docs/LIBSAMPLERATE_MAC_ARM.md`**.

### Worker hangs / no output
When using subprocess, combine stderr with stdout:
```python
proc = subprocess.Popen(
    [exe_path, "--worker"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,  # Important!
    text=True,
    cwd=exe_dir  # Must run from DSU directory
)
```

### "bin" directory not found
Ensure you run the worker from the DSU directory (set `cwd` in subprocess).

## Releases

Download pre-built releases from GitHub Releases.

| Platform | File | GPU | Size |
|----------|------|-----|------|
| Windows CUDA | `dsu-win-cuda.7z` | NVIDIA CUDA 12.6 | ~2 GB |
| Windows CPU | `dsu-win-cpu.zip` | None | ~800 MB |
| macOS ARM | `dsu-mac-arm.tar.gz` | MPS (Metal) | ~1.5 GB |

## License

- **DSU Runtime**: MIT
- **PyTorch**: BSD-3
- **Demucs**: MIT
- **audio-separator**: MIT
- **Individual models**: Check each model's license

## Credits

Created by Ostin Solo
- Website: ostinsolo.co.uk
