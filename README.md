# Doctor Sample Unit (DSU) - Shared Runtime

A lightweight, shared Python runtime for professional audio source separation. Supports multiple architectures with a single runtime installation, dramatically reducing disk space and eliminating startup overhead.

## Key Benefits

| Metric | Old (Fat Executables) | New (Shared Runtime) |
|--------|----------------------|----------------------|
| **Download size** | ~6-9 GB | **1.7 GB** |
| **Disk space** | ~15-20 GB | **~5 GB** |
| **Startup time** | 30-40 sec | **~3 sec** |
| **Model switching** | Restart required | **Instant** |
| **Python required** | No | **No** |

## Supported Architectures

| Architecture | Models | Best For | GPU Support |
|-------------|--------|----------|-------------|
| **BS-RoFormer** | 28+ models | Vocals, Denoise, Dereverb | CUDA, MPS |
| **Demucs** | 8 models | 4/6 stem separation | CUDA, MPS |
| **Audio-Separator VR** | PTH models | Vocals/Instrumental | CUDA, MPS |
| **Apollo** | 4 models | Audio restoration | CUDA, MPS |

## Tested Performance

Sequential test with same input file on Windows CUDA:

| Step | Architecture | Model | Separation Time | Total Time |
|------|-------------|-------|-----------------|------------|
| 1 | Demucs | htdemucs | 2.4s | 9.0s |
| 2 | BS-RoFormer | dereverb | 2.2s | 12.5s |
| 3 | Audio-Separator VR | 2_HP-UVR.pth | 6.4s | 11.5s |
| 4 | Apollo | apollo_lew_v2 | 5.9s | 10.0s |
| | **Total** | | | **43s** |

**Model switching within same worker:**
- First model: ~3-6s (cold load)
- Second model: ~0.3-1s (GPU warm)

## Quick Start

### Windows CUDA

```batch
REM Extract the release
7z x dsu-win-cuda.7z

REM Test workers
runtime\Lib\site-packages\python.exe workers\bsroformer_worker.py --help
runtime\Lib\site-packages\python.exe workers\demucs_worker.py --help
runtime\Lib\site-packages\python.exe workers\audio_separator_worker.py --help
```

### macOS Apple Silicon

```bash
tar -xzf dsu-mac-arm.tar.gz
cd dsu-mac-arm
runtime/bin/python workers/bsroformer_worker.py --help
```

## Worker JSON Protocol

All workers use the same JSON protocol via stdin/stdout:

### Commands

```json
{"cmd": "ping"}
{"cmd": "list_models"}
{"cmd": "load_model", "model": "dereverb"}
{"cmd": "separate", "input": "/path/audio.wav", "output": "/path/out/"}
{"cmd": "exit"}
```

### Audio-Separator VR (PTH models)

```json
{"cmd": "separate", "input": "/audio.wav", "output_dir": "/out/", "model": "2_HP-UVR.pth", "model_file_dir": "/Models/audio-separator"}
```

### Apollo Restoration

```json
{"cmd": "apollo", "input": "/audio.wav", "output": "/out/restored.wav", "model_path": "/Models/apollo/apollo_lew_v2.ckpt", "config_path": "/Models/apollo/apollo_lew_v2.yaml"}
```

### Responses

```json
{"status": "ready", "device": "cuda", "threads": 8}
{"status": "loading_model", "model": "dereverb"}
{"status": "model_loaded", "model": "dereverb", "stems": ["noreverb"]}
{"status": "separating", "input": "song.wav"}
{"status": "done", "elapsed": 2.5, "files": ["noreverb.wav"]}
{"status": "error", "message": "..."}
{"status": "exiting"}
```

## Node.js Integration

```javascript
const { spawn } = require('child_process');

// Spawn persistent worker
const worker = spawn('runtime/Lib/site-packages/python.exe', [
    'workers/bsroformer_worker.py',
    '--worker',
    '--models-dir', 'C:/Models/bsroformer'
]);

// Send commands
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
                console.log('Separated in', response.elapsed, 'seconds');
                console.log('Output files:', response.files);
            }
        } catch (e) {
            // Not JSON, might be warning/log
        }
    }
});

// Load model and separate
sendCommand({ cmd: 'load_model', model: 'dereverb' });
sendCommand({ cmd: 'separate', input: 'C:/audio/song.wav', output: 'C:/output/' });
```

## Directory Structure

```
dsu-win-cuda/
├── runtime/                    # Shared Python environment (~5GB)
│   └── Lib/site-packages/
│       └── python.exe
├── workers/                    # Worker scripts
│   ├── bsroformer_worker.py
│   ├── demucs_worker.py
│   └── audio_separator_worker.py
├── models/                     # BS-RoFormer model architectures
├── configs/                    # Model configuration files
├── utils/                      # Utility modules
├── apollo/                     # Apollo restoration module
└── README.md
```

## Platform Support

| Platform | Status | PyTorch | GPU |
|----------|--------|---------|-----|
| Windows CUDA | ✅ Ready | 2.10.0+cu126 | NVIDIA CUDA 12.6 |
| Windows CPU | ✅ Ready | 2.10.0+cpu | None |
| macOS ARM | ✅ Ready | 2.5.0+ | MPS (Metal) |
| macOS Intel | 🔧 Manual | 2.2.2 | None |

## Building from Source

### Windows

```batch
REM Clone repo
git clone https://github.com/ostinsolo/doctor-sample-unit.git
cd doctor-sample-unit

REM Build runtime (CUDA)
build_runtime.bat cuda

REM Or CPU only
build_runtime.bat cpu
```

### macOS Apple Silicon

```bash
chmod +x build_runtime_mac_mps.sh
./build_runtime_mac_mps.sh
```

## Model Requirements

### BS-RoFormer
- `models.json` - Model registry with paths
- `configs/` - YAML configuration files
- `weights/` - Checkpoint files (.ckpt)

### Audio-Separator VR
- `.pth` model files
- `vr_model_data.json` - Model parameters (in model directory)

### Apollo
- `.ckpt` checkpoint file
- `.yaml` config file

## Troubleshooting

### "CUDA not available"
```batch
nvidia-smi   REM Check driver
```
Requires NVIDIA driver with CUDA 12.6 support.

### "Model hash not found"
Add the model's MD5 hash to `vr_model_data.json` in your models directory.

### "No module named 'models'"
Ensure `models/`, `utils/`, `configs/`, `apollo/` folders are in the same directory as `workers/`.

## Releases

Download pre-built releases from:
https://github.com/ostinsolo/doctor-sample-unit/releases

| Platform | File | Size |
|----------|------|------|
| Windows CUDA | dsu-win-cuda.7z | ~1.7 GB |
| Windows CPU | dsu-win-cpu.zip | ~500 MB |
| macOS ARM | dsu-mac-arm.tar.gz | ~1 GB |

## License

- **Runtime**: MIT
- **PyTorch**: BSD-3
- **audio-separator**: MIT
- **Demucs**: MIT
- **Individual models**: Check each model's license

## Credits

Created by Ostin Solo
- Website: ostinsolo.co.uk
- Contact: contact@ostinsolo.co.uk
