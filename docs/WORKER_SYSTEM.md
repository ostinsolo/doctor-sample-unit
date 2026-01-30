# DSU Worker System Documentation

## Overview

The DSU (Deep Stem Unmixer) Worker System provides three persistent worker processes for audio separation and restoration:

1. **dsu-demucs** (Windows: `dsu-demucs.exe`) - Demucs hybrid transformer separation
2. **dsu-bsroformer** (Windows: `dsu-bsroformer.exe`) - BS-RoFormer, MelBand-RoFormer, SCNet, MDX23C separation
3. **dsu-audio-separator** (Windows: `dsu-audio-separator.exe`) - VR architecture separation + Apollo restoration

All workers use a JSON-over-stdin/stdout protocol for Node.js integration.

### Platforms & PyTorch

| Platform | PyTorch | GPU | Executables |
|----------|---------|-----|-------------|
| Windows CUDA | 2.10.x+cu126 | NVIDIA | `dsu-*.exe` |
| Windows CPU | 2.10.x+cpu | None | `dsu-*.exe` |
| **macOS ARM (M1/M2/M3)** | **2.10.x** | **MPS (Metal)** | `dsu-*` (no extension) |

Mac ARM uses **PyTorch 2.10** (not 2.5). See `requirements-mac-mps.txt`. Validated with runtime and frozen `dist/dsu/` builds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Node.js App                               │
│  (Sends JSON commands, receives JSON responses)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ dsu-demucs    │    │ dsu-bsroformer│    │ dsu-audio-sep │
│   Worker      │    │    Worker     │    │    Worker     │
├───────────────┤    ├───────────────┤    ├───────────────┤
│ htdemucs      │    │ BSRoformer    │    │ VR Models     │
│ htdemucs_ft   │    │ MelBandRofo   │    │ Apollo        │
│ htdemucs_6s   │    │ SCNet         │    │               │
│ mdx variants  │    │ MDX23C        │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## JSON Protocol

### Common Commands (All Workers)

```json
// Check if worker is alive
{"cmd": "ping"}
// Response: {"status": "pong", "model_loaded": "model_name"}

// Get current status
{"cmd": "get_status"}
// Response: {"status": "status", "model_loaded": "...", "device": "cuda", "ready": true}

// Shutdown worker
{"cmd": "exit"}
// Response: {"status": "exiting"}
```

### Demucs Worker Commands

```json
// List available models
{"cmd": "list_models"}
// Response: {"status": "models", "models": ["htdemucs", "htdemucs_ft", ...]}

// Pre-load a model
{"cmd": "load_model", "model": "htdemucs_ft"}
// Response: {"status": "model_loaded", "model": "htdemucs_ft", "stems": [...]}

// Run separation
{
  "cmd": "separate",
  "input": "/path/to/audio.wav",
  "output": "/output/dir",
  "model": "htdemucs_ft",
  "two_stems": "vocals",      // Optional: output vocals + no_vocals
  "shifts": 1,                // Optional: random shifts for quality
  "overlap": 0.25             // Optional: overlap between chunks
}
// Response: {"status": "done", "elapsed": 3.13, "files": ["drums.wav", ...]}
```

### BSRoformer Worker Commands

```json
// List models from registry
{"cmd": "list_models"}
// Response: {"status": "models", "models": ["bsroformer_4stem", "scnet_xl_ihf", ...]}

// Run separation (by model name from registry)
{
  "cmd": "separate",
  "input": "/path/to/audio.wav",
  "output": "/output/dir",
  "model": "bsroformer_4stem"
}

// Run separation (by direct path)
{
  "cmd": "separate",
  "input": "/path/to/audio.wav",
  "output": "/output/dir",
  "model_path": "/path/to/checkpoint.ckpt",
  "config_path": "/path/to/config.yaml",
  "model_type": "bs_roformer"  // or "mel_band_roformer", "scnet", "mdx23c", "apollo"
}
// Response: {"status": "done", "elapsed": 0.78, "files": [...], "stems": [...]}
```

### Audio Separator Worker Commands

```json
// VR Model separation
{
  "cmd": "separate",
  "input": "/path/to/audio.wav",
  "output_dir": "/output/dir",
  "model": "UVR-MDX-NET-Inst_HQ_3.onnx",
  "model_file_dir": "/path/to/models",
  "vr_aggression": 5,           // -100..100, typically 5 (optional). Maps to VR aggressiveness; we pass through to audio-separator.
  "vr_enable_tta": false,
  "vr_high_end_process": false,
  "vr_post_process": false,
  "single_stem": "Instrumental" // or "Vocals" (optional)
}

// Apollo restoration
{
  "cmd": "apollo",
  "input": "/path/to/audio.wav",
  "output": "/path/to/restored.wav",
  "model_path": "/path/to/apollo.ckpt",
  "config_path": "/path/to/apollo.yaml",
  "feature_dim": 384,         // Model architecture (256/384/192)
  "layer": 6,                 // Number of layers (6 or 8)
  "chunk_seconds": 7.0,       // Chunk size for processing
  "chunk_overlap": 0.5        // Overlap between chunks
}
// Response: {"status": "done", "elapsed": 1.47, "files": ["/path/to/restored.wav"]}

// Batch: run multiple commands in one request (same worker, same order)
{
  "cmd": "batch",
  "jobs": [
    {"cmd": "separate", "input": "a.wav", "output_dir": "/out", "model": "3_HP-Vocal-UVR.pth", "model_file_dir": "/models"},
    {"cmd": "apollo", "input": "/out/a_(Instrumental).wav", "output": "/out/restored.wav", "model_path": "/models/apollo.ckpt", "config_path": "/models/apollo.yaml"}
  ]
}
// Response: {"status": "done"|"error", "results": [{...}, {...}], "message": "..."}
```

---

## Performance Optimization

### Timing Breakdown (3.5s test audio)

| Phase | Cold Start | Warm (Cached) |
|-------|-----------|---------------|
| Worker startup | 2.5-4.4s | 0s (already running) |
| Model loading | 6-13s | 0s (already loaded) |
| **Processing** | - | **0.7-1.5s** |

### Mac ARM (PyTorch 2.10, MPS) – 4s test audio

From `tests/run_benchmarks_mac.sh` / `benchmark_worker_e2e.py`:

| Worker | Ready | Model load | Separate (cold) | Separate (warm) |
|--------|-------|------------|-----------------|-----------------|
| Demucs htdemucs | ~0.64s | ~0.19s | ~0.9s | ~0.43s |
| BS-RoFormer bsrofo_sw | ~0.73s | ~18.5s | ~10.2s | ~4.3s |

Mac uses **PyTorch 2.10**; MPS backend. Models dir: `~/Documents/DSU/ThirdPartyApps/Models` (or `DSU_MODELS`).

**Node optimized workflow** (`run_optimized_workflow.js`, BS-RoFormer model1=bsroformer_4stem, model2=scnet_xl_ihf, 4s, MPS):

| Step | Total | Processing | Overhead |
|------|-------|------------|----------|
| model1 COLD | ~5.0s | ~3.7s | ~1.3s |
| model1 CACHED | ~3.0s | ~3.0s | ~0 |
| model2 SWITCH | ~10.3s | ~9.9s | ~0.4s |
| model1 SWITCH BACK | ~6.1s | ~6.1s | ~0 |

### MPS and native PyTorch ops

We use **MPS** for all Mac ARM architectures (Demucs, BS-RoFormer, Audio-Separator, Apollo). PyTorch’s built‑in layers run on MPS without extra dependencies:

- **Convs, matmul, etc.** — `nn.Conv1d` / `Conv2d`, `torch.matmul`, and similar ops use MPS when the model and tensors are on `mps`. We already place models and inputs on MPS via `model.to(device)` and `input.to(device)` in the workers.
- **Attention** — `nn.MultiheadAttention` and the transformer layers in Demucs / BS-RoFormer use MPS where the tensors are on `mps`. They’re slower than CUDA but work. Some dependency code (e.g. audio_separator’s `attend`) uses CUDA‑specific SDP/flash attention when on CUDA and falls back to a CPU‑friendly path on MPS; the underlying attention still runs on MPS when inputs are on MPS.
- **STFT / ISTFT** — In **PyTorch 2.10**, `torch.stft` and `torch.istft` **run on MPS** (verified). Some upstream code (e.g. audio_separator’s `uvr_lib` STFT/ISTFT) still uses a **CPU fallback** when the device is MPS (a conservative workaround from when MPS didn’t support these ops). The main model forward (convs, attention, etc.) stays on MPS; only the STFT/ISTFT steps are moved to CPU and back. Upstream could optionally try MPS for STFT/ISTFT to reduce CPU round‑trips.

**Implementation:** We explicitly select MPS when available (`get_device()` in Apollo, device selection in Demucs / Audio-Separator workers: CUDA → MPS → CPU) and use `torch.autocast(device_type="mps")` where it helps (e.g. Audio-Separator VR).

**Summary:** We already use native PyTorch MPS ops for our architectures. No extra dependencies are required. Remaining CPU use is mainly the optional STFT/ISTFT fallback in dependencies; that could be revisited as PyTorch MPS support evolves.

### Optimization Strategies

1. **Keep workers running** - Start on app launch, don't kill between jobs
2. **Pre-load models** - Load common models in background during startup
3. **Batch by model** - Run all jobs for model-A, then model-B (avoid reloads)
4. **Model caching** - Workers automatically cache loaded models

### Sequential Workflow Performance

```
Cold Start (3 workers, 3 models):  ~38s
Optimized (cached):                ~3.2s
Speedup:                           12x faster!
```

---

## Benchmarks

All benchmarks performed with mmap optimization enabled (Jan 2026).

### Model Loading Benchmarks

#### torch.load() with mmap=True Optimization

| Method | Time (500MB model) | Speedup |
|--------|-------------------|---------|
| Standard torch.load | 0.312s | baseline |
| **mmap=True** | **0.111s** | **2.8x faster** |

#### First Load vs Cached (3.5s test audio)

| Model | First Load (cold) | Cached | Model Load Overhead |
|-------|------------------|--------|---------------------|
| BSRoformer 4-stem (503MB) | 6.90s | **0.63s** | 4.23s |
| SCNet XL (204MB) | 5.83s | 0.65s | 1.00s |
| Apollo lew_uni (140MB) | 3.90s | 0.82s | 0.59s |
| Demucs htdemucs_ft | 15.92s | 0.87s | 10.45s |

### Model Switching Benchmarks

| Operation | Total Time | Processing | Overhead | Type |
|-----------|-----------|------------|----------|------|
| BSRoformer (COLD) | 6.90s | 2.67s | **4.23s** | First load |
| BSRoformer (CACHED) | **0.63s** | 0.63s | 0s | Same model |
| SCNet (SWITCH) | 5.83s | 4.83s | **1.00s** | Different model |
| BSRoformer (BACK) | 3.07s | 0.73s | **2.34s** | Switch back |

**Key insight**: Same model = instant (~0.6s), Model switch = 1-4s overhead

### Long Audio Processing (4 minutes / 240 seconds)

| Model | Total Time | Processing | Speed |
|-------|-----------|------------|-------|
| BSRoformer 4-stem | 37.9s | 30.6s | **7.8x realtime** |

### Worker Configuration Benchmarks

#### Parallel vs Single Worker Performance

| Configuration | Sequence Time | Notes |
|--------------|---------------|-------|
| **Single worker only** | **15.21s** | **WINNER** |
| Parallel (3 workers) | 17.83s | 17% slower |
| Sequential (start/stop) | 21.18s | 39% slower |

**Conclusion**: Idle workers consume resources. Use single worker for best performance.

#### Per-Operation with Different Configurations

| Operation | 3 Workers Running | 1 Worker Only | Improvement |
|-----------|------------------|---------------|-------------|
| BSRoformer #1 | 7.94s | 7.71s | 3% |
| BSRoformer #2 | 0.79s | 0.75s | 5% |
| SCNet | 6.33s | **4.42s** | **30%** |
| Apollo | 2.76s | 2.32s | 16% |

### Sequence Test Results (Short Audio, 3.5s)

| Sequence | Before mmap | After mmap | Improvement |
|----------|-------------|------------|-------------|
| Same model 3x | 41.0s | **8.0s** | **5x faster** |
| Model switching | 11.6s | 8.7s | 1.3x faster |
| Full pipeline | 22.6s | 18.6s | 1.2x faster |
| All architectures | 7.5s | 6.5s | 1.2x faster |

### Summary Performance Table (Final Validated Results)

| Scenario | Time | Notes |
|----------|------|-------|
| Worker startup | 2.5-3s | One-time cost |
| **Same model (cached)** | **0.6s** | **INSTANT!** |
| Model switch (new model) | 0.6-2.2s overhead | Load new weights |
| Model switch (back to prev) | 2.2s overhead | Reload weights |
| Cold start (first model) | 6.6s overhead | One-time per model |
| 4-min audio processing | 30-38s | 7-8x realtime |

### Final Model Switching Benchmark

| Operation | Total | Processing | Overhead |
|-----------|-------|------------|----------|
| BSRoformer (cold) | 10.09s | 3.47s | 6.62s |
| BSRoformer (cached) | **0.60s** | 0.59s | 0s |
| SCNet (switch) | 4.01s | 3.43s | 0.58s |
| BSRoformer (back) | 2.87s | 0.71s | 2.16s |

### Recommended Usage Pattern

Based on benchmarks:

1. **Start 1 worker** (not all 3) - saves resources
2. **Keep worker alive** - model stays cached
3. **Same model = instant** (~0.6s)
4. **Model switch = 1-4s** - acceptable for different operations
5. **Group by model** when processing multiple files

### Sequence pipeline (real use case)

Often the project runs **chains**: same file through several archs in order, each step’s **output** = next step’s **input**:

```
input.wav → [VR] → instrumental.wav → [Apollo] → restored.wav
```

- Use **one worker** (or minimal workers) and **keep them alive** between steps.
- **`run_sequence_pipeline.js`** exercises this: `--sequence vr,apollo` (single audio-separator) or `bsroformer,apollo` (bsroformer + audio-separator). See **`tests/README_NODE_TESTS.md`**.

---

## Preloading Models

### Why Preload?

Model loading is the slowest part of processing:
- **BSRoformer models**: 6-13 seconds to load
- **Demucs models**: 3-5 seconds to load
- **Apollo models**: 4-8 seconds to load
- **Actual processing**: Only 0.7-1.5 seconds!

By preloading models when your app starts, users experience instant processing.

### How to Preload Each Worker

#### Demucs Worker

Demucs has a dedicated `load_model` command:

```json
// Preload htdemucs_ft model
{"cmd": "load_model", "model": "htdemucs_ft"}

// Response confirms model is ready
{"status": "model_loaded", "model": "htdemucs_ft", "stems": ["drums", "bass", "other", "vocals"]}
```

#### BSRoformer Worker

BSRoformer preloads via the first `separate` command. The model stays cached:

```json
// First call loads the model (slow ~7s)
{"cmd": "separate", "input": "file1.wav", "output": "out1", "model": "bsroformer_4stem"}

// Subsequent calls use cached model (fast ~0.8s)
{"cmd": "separate", "input": "file2.wav", "output": "out2", "model": "bsroformer_4stem"}
```

To preload without processing, you can use a very short dummy audio file.

#### Audio Separator Worker

Same pattern - first call loads, subsequent calls use cache:

```json
// Apollo preloads on first use
{"cmd": "apollo", "input": "file1.wav", "output": "out1.wav", "model_path": "apollo.ckpt", ...}
```

---

## Use Cases & Best Practices

### Case 1: Single File Processing

**Scenario**: User drops one file, processes with one model.

**Strategy**: Just run it. Preloading doesn't help for single operations.

```
User drops file → Start worker → Load model → Process → Done
Total time: 8-15s (acceptable for single file)
```

### Case 2: Batch Processing (Same Model)

**Scenario**: User queues 10 files with the same model.

**Strategy**: Process all files sequentially on same worker. Model loads once.

```
File 1: 8s (includes model load)
File 2: 1s (cached)
File 3: 1s (cached)
...
File 10: 1s (cached)
Total: ~17s instead of ~80s
```

**Node.js Pseudocode**:
```javascript
async function batchProcess(files, model) {
  const worker = await startWorker('dsu-bsroformer.exe');
  
  for (const file of files) {
    // Model loads on first file, cached for rest
    await worker.send({
      cmd: 'separate',
      input: file,
      output: getOutputDir(file),
      model: model
    });
  }
  
  // Keep worker alive for future batches
}
```

### Case 3: Sequential Workflow (Different Models)

**Scenario**: Process one file through BSRoformer → Demucs → Apollo.

**Strategy**: Start all workers on app launch, preload default models in background.

```
App startup (background):
  - Start dsu-bsroformer.exe, preload bsroformer_4stem
  - Start dsu-demucs.exe, preload htdemucs_ft
  - Start dsu-audio-separator.exe (Apollo loads on demand)

User runs workflow:
  - BSRoformer: 0.8s (preloaded!)
  - Demucs: 0.9s (preloaded!)
  - Apollo: 1.5s (loads on first use, or preload if common)
  Total: ~3.2s instead of ~38s
```

**Node.js Pseudocode**:
```javascript
class WorkerPool {
  constructor() {
    this.workers = {};
  }

  async initialize() {
    // Start all workers in parallel
    const [bsroformer, demucs, audioSep] = await Promise.all([
      this.startWorker('dsu-bsroformer.exe'),
      this.startWorker('dsu-demucs.exe'),
      this.startWorker('dsu-audio-separator.exe')
    ]);
    
    this.workers = { bsroformer, demucs, audioSep };
    
    // Preload default models in background (don't await)
    this.preloadDefaults();
  }

  async preloadDefaults() {
    // These run in background while app is starting
    await Promise.all([
      this.workers.demucs.send({ cmd: 'load_model', model: 'htdemucs_ft' }),
      // BSRoformer/Apollo will preload on first use
    ]);
  }

  async runWorkflow(inputFile) {
    // All models already loaded - instant processing!
    const bsResult = await this.workers.bsroformer.send({...});
    const demucsResult = await this.workers.demucs.send({...});
    const apolloResult = await this.workers.audioSep.send({...});
    return { bsResult, demucsResult, apolloResult };
  }
}
```

### Case 4: Mixed Batch (Multiple Models, Multiple Files)

**Scenario**: Queue has files for different models - some BSRoformer, some SCNet, some Demucs.

**Strategy**: Group by model type to minimize reloads.

```
BAD (interleaved - many reloads):
  File1 + BSRoformer (load 7s) → process 1s
  File2 + SCNet (load 6s) → process 1s
  File3 + BSRoformer (load 7s) → process 1s  ← Reloaded!
  File4 + SCNet (load 6s) → process 1s       ← Reloaded!
  Total: ~30s

GOOD (grouped - minimal reloads):
  File1 + BSRoformer (load 7s) → process 1s
  File3 + BSRoformer (cached) → process 1s
  File2 + SCNet (load 6s) → process 1s
  File4 + SCNet (cached) → process 1s
  Total: ~17s
```

**Node.js Pseudocode**:
```javascript
async function processBatchOptimized(jobs) {
  // Group jobs by model
  const grouped = {};
  for (const job of jobs) {
    const key = job.model;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(job);
  }
  
  // Process each group (minimizes model switches)
  const results = [];
  for (const [model, modelJobs] of Object.entries(grouped)) {
    for (const job of modelJobs) {
      const result = await worker.send({
        cmd: 'separate',
        model: model,
        input: job.input,
        output: job.output
      });
      results.push({ job, result });
    }
  }
  
  return results;
}
```

### Case 5: Model Switching Within Worker

**Scenario**: Need to switch from BSRoformer to SCNet on same worker.

**What happens**:
- Worker unloads current model from GPU
- Worker loads new model (6-13s)
- Subsequent calls with new model are cached

```json
// Using bsroformer_4stem (loaded)
{"cmd": "separate", "model": "bsroformer_4stem", ...}  // 0.8s

// Switch to scnet_xl_ihf (requires reload)
{"cmd": "separate", "model": "scnet_xl_ihf", ...}  // 6-7s (reload)

// scnet_xl_ihf now cached
{"cmd": "separate", "model": "scnet_xl_ihf", ...}  // 0.8s
```

**Note**: We don't support keeping multiple models loaded simultaneously due to GPU memory constraints. Each worker holds ONE model at a time.

---

## Recommended App Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your App Startup                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Start worker processes (parallel)         ~3s               │
│  2. Preload default models (background)       ~10-15s           │
│  3. Show UI (immediately)                     0s                │
│  4. Workers ready when user needs them        ✓                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        User Processing                           │
├─────────────────────────────────────────────────────────────────┤
│  • Single file: Use preloaded model           ~1-2s             │
│  • Batch same model: First cached             ~1s per file      │
│  • Model switch: One-time reload              ~6-13s            │
│  • Sequential workflow: All preloaded         ~3s total         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        App Shutdown                              │
├─────────────────────────────────────────────────────────────────┤
│  Send {"cmd": "exit"} to all workers                            │
│  Workers clean up GPU memory and exit                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Graceful Shutdown (Preventing Zombie Processes)

Workers automatically handle shutdown in these scenarios:

1. **Normal exit**: `{"cmd": "exit"}` - Worker responds with `{"status": "exiting"}` and cleans up
2. **Signal handling**: SIGTERM, SIGINT, SIGBREAK (Windows) - Worker exits gracefully
3. **Parent process died**: stdin closed - Worker detects and exits

### Node.js Best Practices

```javascript
const { spawn } = require('child_process');

class DSUWorker {
  constructor(exePath) {
    this.proc = spawn(exePath, ['--worker'], {
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    // Track if we've sent exit command
    this.exiting = false;
    
    // Handle process events
    this.proc.on('exit', (code) => {
      console.log(`Worker exited with code ${code}`);
    });
  }

  // Always send exit command before killing
  async shutdown() {
    if (this.exiting) return;
    this.exiting = true;
    
    try {
      // Send graceful exit command
      this.send({ cmd: 'exit' });
      
      // Wait for clean exit (with timeout)
      await this.waitForExit(5000);
    } catch (e) {
      // Force kill if graceful shutdown fails
      this.proc.kill('SIGTERM');
    }
  }

  // Helper to wait for process exit
  waitForExit(timeout) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Timeout')), timeout);
      this.proc.on('exit', () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }
}

// App-level cleanup
class WorkerPool {
  constructor() {
    this.workers = [];
    
    // CRITICAL: Handle app exit to prevent zombies
    process.on('exit', () => this.shutdownAll());
    process.on('SIGINT', () => { this.shutdownAll(); process.exit(0); });
    process.on('SIGTERM', () => { this.shutdownAll(); process.exit(0); });
    process.on('uncaughtException', (e) => {
      console.error(e);
      this.shutdownAll();
      process.exit(1);
    });
  }

  async shutdownAll() {
    await Promise.all(this.workers.map(w => w.shutdown()));
  }
}
```

### Shutdown Response

When a worker exits due to a signal or parent process death (rather than `{"cmd": "exit"}`), it sends:

```json
{"status": "shutdown", "reason": "signal_or_parent_died"}
```

This allows your Node.js code to distinguish between:
- `{"status": "exiting"}` - Normal exit via command
- `{"status": "shutdown", ...}` - Unexpected exit (crash recovery scenario)

### Preventing Zombie Processes

Common causes and fixes:

| Cause | Fix |
|-------|-----|
| Node.js crashes without sending exit | Workers detect stdin closure and exit |
| SIGTERM sent to Node.js | Handle SIGTERM, send exit to workers first |
| Worker started but never used | Workers have signal handlers, will exit on SIGTERM |
| Multiple instances of same worker | Check `tasklist` before spawning, or use PID tracking |

---

## Audio I/O

All workers use **soundfile** for audio I/O to avoid torchcodec dependency issues (Windows and **Mac**):

```python
# Loading audio
import soundfile as sf
audio, sr = sf.read(input_path)

# Saving audio  
sf.write(output_path, audio, sample_rate, subtype='PCM_16')
```

- **Demucs CLI**: `demucs_worker.py` patches `demucs.audio.save_audio` to use soundfile so frozen `dsu-demucs` (CLI mode) avoids `torchaudio.save` → torchcodec. Worker mode already used soundfile.
- No torchcodec/ffmpeg dependency issues, consistent WAV output, cross-platform compatibility.

---

## Apollo Model Configurations

Apollo models require specific `feature_dim` and `layer` parameters:

| Model | feature_dim | layer | Use Case |
|-------|-------------|-------|----------|
| apollo_official | 256 | 6 | General restoration |
| apollo_lew_uni | 384 | 6 | **RECOMMENDED** - Best quality |
| apollo_lew_v2 | 192 | 6 | Lightweight, vocals |
| apollo_edm_big | 256 | 6 | EDM/Electronic |
| apollo_vocal_msst | 384 | 8 | Vocal enhancement |
| apollo_vocal_msst_full | 384 | 8 | Full vocal model |

Config files store these parameters. Wrong values cause model loading to fail.

---

## Achievements & Changelog

### ✅ Completed

1. **Soundfile-based audio I/O** (Jan 2026)
   - Replaced torchaudio save with soundfile
   - Eliminated torchcodec dependency issues (Windows and **Mac**)
   - Demucs CLI patch in `demucs_worker.py` so frozen `dsu-demucs` uses soundfile; worker mode already did
   - Cleaner, simpler code

2. **Model caching** (Jan 2026)
   - All workers cache loaded models
   - 12x speedup for sequential operations
   - Automatic cache invalidation on model change

3. **Apollo chunking optimization** (Jan 2026)
   - Default: 7s chunks, 0.5s overlap
   - Safe for 8GB GPUs
   - Prevents OOM on long files

4. **Apollo config verification** (Jan 2026)
   - All Apollo models have verified configs
   - Created missing config files
   - Verified feature_dim/layer parameters

5. **Worker protocol standardization** (Jan 2026)
   - Consistent JSON protocol across all workers
   - Status messages: loading, ready, separating, done, error
   - Model information in responses

6. **Memory-mapped model loading** (Jan 2026)
   - Added `mmap=True` to all `torch.load()` calls
   - **4.6x faster** model loading (19.7s -> 4.2s for BSRoformer cold start)
   - **2.8x faster** raw checkpoint loading (0.31s -> 0.11s)
   - Same model 3x sequence: 41s -> 8s (**5x faster**)

7. **Single worker optimization** (Jan 2026)
   - Discovered idle workers consume resources
   - Single worker is **17% faster** than 3 parallel workers
   - Recommendation: Start workers on-demand, not all at once

8. **Mac ARM + PyTorch 2.10** (Jan 2026)
   - **PyTorch 2.10** on macOS Apple Silicon (MPS). Not 2.5; see `requirements-mac-mps.txt`.
   - Frozen builds produce `dsu-demucs`, `dsu-bsroformer`, `dsu-audio-separator` (no `.exe`).
   - `tests/run_benchmarks_mac.sh` for Demucs, BS-RoFormer, Audio-Separator, Apollo (4s/40s).
   - Default models dir: `~/Documents/DSU/ThirdPartyApps/Models`. Override with `DSU_MODELS`.

### 🔄 In Progress

- Progress reporting for long operations

### 📋 Roadmap

1. **GPU memory management** - Automatic model unloading when VRAM low
2. **Batch processing** - Process multiple files in single command
3. **Progress callbacks** - Real-time progress updates during separation
4. **Model download integration** - Download models on demand

---

## Testing

### Test Files & Paths

- **Models dir**: `~/Documents/DSU/ThirdPartyApps/Models` (or `DSU_MODELS`). Contains `bsroformer/`, `audio-separator/`, `apollo/`, `demucs/`.
- **Test audio**: `tests/audio/`. Create with `python tests/generate_test_audio.py` (adds `test_4s.wav`, `test_40s.wav`, etc.).
- **Index**: `tests/README.md`. **Benchmarks**: `tests/README_BENCHMARKS.md`. **Node tests (Max/MSP)**: `tests/README_NODE_TESTS.md`.

### Test Layout

```
tests/
├── README.md                       # Index
├── README_BENCHMARKS.md            # Python benchmarks, 4s/40s
├── README_NODE_TESTS.md            # Node tests, optimized + sequence pipeline
├── node_test_config.js             # Shared config (Mac + Windows)
├── run_sequence_pipeline.js        # Node: input→op1→op2→… (vr,apollo / bsroformer,apollo)
├── run_optimized_workflow.js       # Node: COLD→CACHED→SWITCH, real files
├── test_workers_optimized.js       # Node: all workers, preload+cache
├── test_all_workers_final.js       # Node: cold vs cached, performance_report
├── benchmark_worker_e2e.py         # E2E timings
├── run_benchmarks_mac.sh           # Mac benchmark runner (4s/40s)
├── generate_test_audio.py          # test_4s.wav, test_40s.wav, etc.
├── create_long_audio.py            # 4-min test audio
├── run_bsroformer_models.js /.py   # BS-RoFormer via Python CLI
├── debug/                          # One-off checks (SDPA, SageAttention, etc.)
└── audio/
```

### Running Tests

```bash
# Test audio
python tests/generate_test_audio.py

# Mac benchmarks (Python)
./tests/run_benchmarks_mac.sh
./tests/run_benchmarks_mac.sh 40

# Node optimized workflow (Max/MSP)
node tests/run_sequence_pipeline.js --input tests/audio/test_4s.wav --sequence vr,apollo
node tests/run_optimized_workflow.js --worker bsroformer --input tests/audio/test_4s.wav
node tests/test_workers_optimized.js
node tests/test_all_workers_final.js

# Manual E2E (Python)
python tests/benchmark_worker_e2e.py --exe dist/dsu/dsu-demucs --worker demucs \
  --model htdemucs --input tests/audio/test_4s.wav --output-dir tests/benchmark_output/demucs --device mps
```

### Expected Results

| Test | Expected Time |
|------|---------------|
| Demucs (first run) | 3-5s |
| Demucs (cached) | 0.8-1s |
| BSRoformer (first run) | 7-13s |
| BSRoformer (cached) | 0.7-1s |
| Apollo (first run) | 5-8s |
| Apollo (cached) | 1.5-2s |

---

## Troubleshooting

### Common Issues

1. **Model not found**
   - Use **absolute paths** for `input`, `output` / `output_dir`, `model_path`, `config_path`. Workers often run with `cwd` = executable directory.
   - Verify checkpoint file exists
   - Check config_path for BSRoformer models

2. **CUDA / MPS out of memory**
   - Reduce chunk_seconds for Apollo
   - Process shorter audio segments
   - Close other GPU applications

3. **Worker not responding**
   - Check stderr for Python errors (use `stderr=subprocess.STDOUT` when spawning)
   - Verify executable path is correct
   - Windows: ensure CUDA drivers installed. Mac: MPS uses built-in Metal.

4. **Mac: torchcodec / torchaudio.save errors**
   - Use soundfile for I/O. Demucs CLI is patched in `demucs_worker.py`; worker mode uses soundfile. Ensure you're on a build that includes this patch.

5. **Wrong output quality**
   - Verify config file matches checkpoint
   - Check feature_dim and layer parameters
   - Ensure correct model_type is specified

6. **Mac / Max (DSU_VSTOPIA etc.): `No module named 'loralib'`**
   - **loralib is optional** for inference. BS-RoFormer separation does not use it; only LoRA training/adapter features do.
   - The repo’s `utils/model_utils.py` uses a **try/except** around `import loralib` and sets `lora = None` if missing. That allows inference without loralib.
   - If you see this error, the **deployed** `model_utils` (e.g. `dsu/utils/model_utils.py` in DSU_VSTOPIA / Max) is an older version with an unconditional `import loralib`, or a **different** `utils` package is on `sys.path` first (e.g. another project folder) and that one still imports loralib.
   - **Debug:** Run the worker with `DSU_DEBUG_LORALIB=1`. If the optional-import version is loaded, stderr will show `[model_utils] loaded from ...` and `loralib available: False`. If you never see that and still get `No module named 'loralib'`, the running code is still the old one.
   - **Fix:** Update the `model_utils` that is **actually** used at runtime (check `sys.path` and worker cwd) to match the repo’s `utils/model_utils.py` (at least the optional import block). **Do not** add loralib as a required dependency for inference; keep it optional.

---

## File Locations

```
Executables:     dist/dsu/dsu-*       (Mac/Linux: no extension; Windows: dsu-*.exe)
Workers:         workers/*.py
Configs:         configs/*.yaml
Models:          ~/Documents/DSU/ThirdPartyApps/Models (or DSU_MODELS)

**macOS FFmpeg (torchcodec):** Demucs uses torchcodec for WAV save. FFmpeg libs must be findable:
- **Build/CI:** `brew install ffmpeg` (Homebrew)
- **Deployment:** Config at `DSU/dsu_config.json` (e.g. `/Users/ostino/DSU/dsu_config.json`). Example:
  ```json
  {"ffmpeg": {"lib_path": "/path/to/DSU/ThirdPartyApps/ffmpeg/lib"}}
  ```
  Or set env `DSU_FFMPEG_LIB_PATH`. See `configs/dsu_config.example.json`.
Test audio:      tests/audio/         (test_4s.wav, test_40s.wav, test_mix.wav, ...)
Benchmark out:   tests/benchmark_output/
```

---

## Contact & Support

For issues with the worker system, check:
1. This documentation
2. Test files for usage examples
3. Worker source code in `workers/` directory
