# Doctor Sample Unit (DSU) - Shared Runtime

A lightweight, shared Python runtime for professional audio source separation. Supports multiple architectures with a single runtime installation, saving ~85% disk space compared to individual executables.

## Features

- **~85% disk space savings** - Single runtime for all architectures
- **Instant model switching** - No restart needed between models
- **No Python installation required** - Self-contained distribution
- **Cross-platform** - Windows, Mac Intel, Mac Apple Silicon
- **GPU accelerated** - CUDA (Windows/Linux), MPS (Apple Silicon)
- **Persistent workers** - Eliminates startup overhead

## Supported Architectures

| Architecture | Models | Stems | GPU Support |
|-------------|--------|-------|-------------|
| **BS-RoFormer** | 28 models | vocals, instrumental, denoise, etc. | CUDA, MPS |
| **Audio-Separator** | VR, MDX, MDXC | vocals, instrumental, drums, bass | CUDA, MPS |
| **Demucs** | 8 models | drums, bass, vocals, other, guitar, piano | CUDA, MPS |
| **Apollo** | Restoration | audio quality enhancement | CUDA, MPS |

## Directory Structure

```
shared_runtime/
├── bin/                              # Windows executables (~12MB)
│   ├── dsu-bsroformer.exe
│   ├── dsu-audio-separator.exe
│   └── dsu-demucs.exe
│
├── runtime/                          # Shared Python environment
│   ├── Scripts/python.exe           # Windows (~3.5GB CUDA, ~2.5GB CPU)
│   └── bin/python                   # Mac/Linux (~2.5GB)
│
├── workers/                          # Worker scripts
│   ├── bsroformer_worker.py
│   ├── audio_separator_worker.py
│   └── demucs_worker.py
│
├── launchers/                        # Launcher scripts (Windows)
│
├── requirements-cuda.txt             # Windows NVIDIA GPU
├── requirements-cpu.txt              # Windows/Linux CPU
├── requirements-mac-intel.txt        # Mac Intel (x86_64)
├── requirements-mac-mps.txt          # Mac Apple Silicon (arm64)
│
├── build_runtime.bat                 # Build Windows runtime
├── build_runtime_mac_intel.sh        # Build Mac Intel runtime
├── build_runtime_mac_mps.sh          # Build Mac Apple Silicon runtime
└── build_dsu_launchers.py            # Build Windows executables
```

## Build Instructions

### Windows NVIDIA GPU (CUDA)

```batch
REM 1. Build the shared runtime (~15 minutes)
build_runtime.bat cuda

REM 2. Build DSU executables
runtime\Scripts\python.exe build_dsu_launchers.py

REM 3. Test
bin\dsu-bsroformer.exe --help
```

### Windows CPU Only

```batch
REM 1. Build CPU-only runtime (~10 minutes)
build_runtime.bat cpu

REM 2. Build executables
runtime\Scripts\python.exe build_dsu_launchers.py

REM 3. Test
bin\dsu-bsroformer.exe --help
```

### Mac Intel (x86_64)

```bash
# 1. Make script executable
chmod +x build_runtime_mac_intel.sh

# 2. Build runtime (~10 minutes)
./build_runtime_mac_intel.sh

# 3. Run workers directly (no executables needed on Mac)
runtime/bin/python workers/bsroformer_worker.py --worker
```

### Mac Apple Silicon (MPS)

```bash
# 1. Make script executable
chmod +x build_runtime_mac_mps.sh

# 2. Build runtime with MPS support (~10 minutes)
./build_runtime_mac_mps.sh

# 3. Run workers directly
runtime/bin/python workers/bsroformer_worker.py --worker
```

## Usage

### Worker Mode (Persistent Process)

Workers stay alive between jobs, eliminating 30-40 second startup overhead:

```bash
# Windows (via executable)
bin\dsu-bsroformer.exe --worker --models-dir "C:\Models\bsroformer"

# Windows (via runtime)
runtime\Scripts\python.exe workers\bsroformer_worker.py --worker

# Mac/Linux
runtime/bin/python workers/bsroformer_worker.py --worker
```

### JSON Protocol

Communicate via stdin/stdout JSON:

**Commands:**
```json
{"cmd": "ping"}
{"cmd": "list_models"}
{"cmd": "load_model", "model": "denoise"}
{"cmd": "separate", "input": "/audio.wav", "output": "/out/", "model": "denoise"}
{"cmd": "exit"}
```

**Responses:**
```json
{"status": "ready", "device": "cuda"}
{"status": "pong", "model_loaded": "denoise"}
{"status": "models", "models": ["denoise", "vocals", ...]}
{"status": "done", "elapsed": 2.5, "files": ["dry.wav"]}
{"status": "error", "message": "..."}
```

### Node.js Integration

```javascript
const { spawn } = require('child_process');

const worker = spawn('bin/dsu-bsroformer.exe', [
    '--worker',
    '--models-dir', 'C:/Models/bsroformer'
]);

worker.stdin.write(JSON.stringify({
    cmd: 'separate',
    input: 'C:/audio/song.wav',
    output: 'C:/output/',
    model: 'denoise'
}) + '\n');

worker.stdout.on('data', (data) => {
    const response = JSON.parse(data.toString().trim());
    if (response.status === 'done') {
        console.log('Separated in', response.elapsed, 'seconds');
    }
});
```

## Platform Specifications

| Platform | PyTorch | GPU | Runtime Size |
|----------|---------|-----|--------------|
| Windows CUDA | 2.10.0+cu126 | NVIDIA (CUDA 12.6) | ~3.5 GB |
| Windows CPU | 2.10.0+cpu | None | ~2.5 GB |
| Mac Intel | 2.2.2 | None | ~2.5 GB |
| Mac Apple Silicon | 2.5.0+ | MPS (Metal) | ~2.5 GB |

### Dependencies

All platforms include:
- **PyTorch** - Neural network framework
- **audio-separator** - VR/MDX/MDXC model support
- **demucs** - Facebook's Demucs models
- **librosa** - Audio processing
- **soundfile** - Audio I/O
- **numba/llvmlite** - JIT compilation (Windows)
- **diffq** - Quantized model support
- **beartype** - Runtime type checking
- **loralib** - LoRA model support

## Distribution

To distribute to end users:

**Windows:**
```
YourApp/
├── bin/                    # DSU executables
├── runtime/                # Shared Python environment
├── workers/                # Worker scripts
└── launchers/              # Launcher scripts
```

**Mac:**
```
YourApp/
├── runtime/                # Shared Python environment
└── workers/                # Worker scripts
```

**Size Comparison:**
| Approach | Disk Space |
|----------|-----------|
| Fat executables (4 workers) | 15-20 GB |
| Shared runtime | **~3.5 GB** |

## Troubleshooting

### Windows: "llvmlite not found"
The requirements include numba and llvmlite. If still failing:
```batch
runtime\Scripts\pip.exe install numba llvmlite --force-reinstall
```

### "CUDA not available"
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA version matches (12.6 required)
3. Use CPU build if no NVIDIA GPU

### Mac: "MPS not available"
1. Requires macOS 12.3+ and Apple Silicon
2. Check with: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. Falls back to CPU automatically

### Model loading errors
Ensure models directory contains:
- `models.json` - Model registry
- `configs/` - Configuration files
- `weights/` - Checkpoint files

## License

- **Runtime**: MIT
- **PyTorch**: BSD-3
- **audio-separator**: MIT
- **Demucs**: MIT
- **Individual models**: Check each model's license
