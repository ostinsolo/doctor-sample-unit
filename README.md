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
├── dsu-demucs.exe           # Demucs worker (~15KB entry point)
├── dsu-bsroformer.exe       # BS-RoFormer worker (~15KB entry point)
├── dsu-audio-separator.exe  # Audio Separator + Apollo (~15KB entry point)
├── lib/                     # Shared frozen dependencies (~2GB)
│   ├── torch/              # PyTorch 2.10.0 + CUDA 12.6
│   ├── numpy/
│   ├── librosa/
│   └── ...
├── models/                  # Model architectures (bs_roformer, scnet, etc.)
├── configs/                 # Model configs (YAML files)
└── VERSION.txt
```

All three executables share the same `lib/` folder - **~4GB disk space saved** compared to bundling each separately!

## Supported Architectures

| Architecture | Models | Best For | GPU Support |
|-------------|--------|----------|-------------|
| **Demucs** | htdemucs, htdemucs_ft, htdemucs_6s, mdx, mdx_extra, mdx_q | 4/6 stem separation | CUDA, MPS |
| **BS-RoFormer** | 28+ models | Vocals, Denoise, Dereverb, Drumsep | CUDA, MPS |
| **Audio-Separator** | VR, MDX, MDXC | Various separation tasks | CUDA, MPS |
| **Apollo** | lew_uni, lew_v2, edm_big, official | Audio restoration | CUDA, MPS |

## Performance (Persistent Workers)

The key advantage: **workers stay running, torch loads once, models stay cached in GPU memory**.

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
REM or
python build_dsu.py

REM Output in dist/dsu/
```

### Manual Build Script

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
| Windows CUDA | ✅ Ready | 2.10.0+cu126 | NVIDIA CUDA 12.6 |
| Windows CPU | ✅ Ready | 2.10.0+cpu | None |
| macOS ARM | ✅ Ready | 2.5.0+ | MPS (Metal) |

## Troubleshooting

### "could not get source code" error
This is fixed in the worker scripts. If you see this, ensure the inspect monkey-patch runs before torch imports.

### "CUDA not available"
```batch
nvidia-smi
```
Requires NVIDIA driver supporting CUDA 12.6.

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
