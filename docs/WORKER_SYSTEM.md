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
| Audio-Separator VR (5_HP-Karaoke-UVR) | - | - | ~5.5s | - |

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

### Checking GPU utilization (BS-RoFormer)

On **CUDA**, BS-RoFormer is designed for high GPU use and should show high GPU utilization during separation:

- **Model and data on GPU:** The worker moves the model and input chunks to `cuda`; `demix_fast` keeps `result` and `counter` on device and does a single `.cpu()` only at the end. Chunks are processed in batches on GPU.
- **STFT / ISTFT on GPU:** In the BS-RoFormer model code, when the device is CUDA (`x_is_mps` is false), `torch.stft` and `torch.istft` run on GPU (no CPU fallback).
- **Attention on GPU:** Attention uses PyTorch SDP (flash or mem-efficient when available) on CUDA. The heavy work (band_split, transformer blocks, mask estimation, STFT/ISTFT) all runs on GPU.

**How to verify (Windows / Linux with NVIDIA GPU):**

1. Start the worker with CUDA:  
   `dsu-bsroformer.exe --worker --models-dir <path> --device cuda`  
   Confirm the ready message includes `"device": "cuda"`.
2. In a second terminal, poll GPU usage while a separation runs:  
   `nvidia-smi -l 1`  
   (or `nvidia-smi` repeatedly).
3. During separation you should see:
   - **GPU-Util** high (often 80–100%) while the model is running.
   - **Memory-Usage** high and stable (model + activations on GPU).

If GPU-Util stays low on CUDA, check: (1) the worker really reports `"device": "cuda"`, (2) you are not bottlenecked by very short audio or batch size 1 with tiny chunks, (3) no other process is starving the GPU.

**MPS (Mac ARM):** In the BS-RoFormer model code, when the device is MPS, STFT/ISTFT and some scatter/mask ops use CPU fallbacks (`x_is_mps` path) for compatibility. The transformer and attention still run on MPS, so GPU (Metal) utilization can be high for that part, but overall utilization may be lower than on CUDA because of the CPU STFT/ISTFT work. This is a known trade-off; moving more of that path to MPS would require testing on macOS 14+ and PyTorch 2.10.

### CPU multithreading (BS-RoFormer on CPU)

When running with `--device cpu`, the worker sets PyTorch and OpenMP/BLAS threading to use **physical core count** (`get_physical_cores()`): `torch.set_num_threads`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, plus `KMP_AFFINITY` / `KMP_BLOCKTIME` for Intel-style pinning. So during separation you should see **high multi-core CPU usage** in Activity Monitor (macOS) or Task Manager (Windows)—e.g. **600–800% CPU** (6–8 cores) on an 8-core machine. That indicates the worker is using CPU power well with multithreading.

### Verifying MPS usage (Mac ARM)

To confirm you are using MPS well on Mac:

1. **Worker reports MPS:** Start the worker with `--device mps` (or let it auto-detect on ARM). The ready message should include `"device": "mps"` and `"threads": <N>` (physical cores, used when CPU fallbacks run).
2. **Activity Monitor:** Open **Activity Monitor** → **Window** → **GPU History**. Run a separation (e.g. BS-RoFormer on a 4s file). You should see **GPU** activity during the run; the main model (transformer, attention) runs on Metal/MPS.
3. **Validation reports:** Reports under `tests/validation_output/` (e.g. `mps_timing_after.json`) can show `"mps_ops": true` and timings per model—confirming MPS paths are active and giving you good throughput (e.g. ~0.46–0.77s per 4s clip for BSRoformer / MelBand models in the repo’s validation runs).

So you can both see **high CPU usage (600–800%)** when on CPU and **good MPS usage** on Mac when on `--device mps`, with GPU History and report fields as proof.

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

#### Per-process restart vs persistent worker (CPU / MPS)

The benchmarks above (single worker vs start/stop) were measured in a **persistent-worker** setup (CUDA/MPS, model cached). On **CPU**, user testing showed that **starting and restarting the worker per process** can be faster than keeping a persistent worker in some setups.

**How CPU “per-process restart” works:**

1. Spawn worker process (e.g. `dsu-bsroformer --worker --device cpu`).
2. Send one job (e.g. `{"cmd": "separate", ...}`).
3. Wait for `{"status": "done"}`.
4. Send `{"cmd": "exit"}`, wait for process exit.
5. For the next job, go to step 1 (spawn a fresh worker).

So each job runs in a **new process**: no model kept in memory between jobs. This can be faster on CPU when the cost of keeping the process alive (threading, MKL/BLAS state, memory) or warm-up effects outweigh the cost of a fresh startup + load + one separation.

**When to consider per-process restart (CPU):**

- Single or sparse jobs (one separation, then long idle).
- Memory or stability issues with a long-lived process.
- You observe that “one job then exit” is faster than “keep worker, same model cached” in your own benchmarks.

**MPS (Mac ARM):** The same comparison (per-process restart vs persistent worker) has not yet been fully benchmarked on MPS. If you run it, use the same pattern: spawn worker with `--device mps`, one `separate`, `exit`, respawn for the next job; compare total time to one persistent worker doing the same sequence. Results can be added here (e.g. “MPS: per-process restart vs persistent, 4s input, bsrofo_sw: …”).

**Recommendation:** On **GPU (CUDA/MPS)**, keep one worker alive and reuse it (model cached). On **CPU**, measure both strategies for your workload; prefer per-process restart if it is faster in your tests.

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
2. **Keep worker alive** - model stays cached (best on GPU/CUDA/MPS)
3. **Same model = instant** (~0.6s)
4. **Model switch = 1-4s** - acceptable for different operations
5. **Group by model** when processing multiple files
6. **On CPU:** In some setups, **per-process restart** (spawn → one job → exit → respawn) can be faster than a persistent worker. See [Per-process restart vs persistent worker (CPU / MPS)](#per-process-restart-vs-persistent-worker-cpu--mps). Benchmark both for your workload.

### Reducing timing further (even 10+ seconds)

To shave more time off separation and model load:

1. **Larger `batch_size` in the JSON job (BS-RoFormer)**  
   Pass `"batch_size": 4` or `"batch_size": 8` in your `separate` command when you have enough GPU/RAM. Configs often use 1–2; a higher value processes more chunks per forward pass and can cut **several seconds** on long files. Example: `{"cmd": "separate", "input": "...", "output_dir": "...", "model": "bsrofo_sw", "batch_size": 4}`. If you hit OOM, lower it.

2. **Keep `use_fast: true` (default)**  
   The worker already defaults to the fast demix path (result/counter on device, single `.cpu()` at end). Do not set `"use_fast": false` unless you need the legacy path.

3. **Model loading: mmap + SSD**  
   Ensure storage is reported as SSD so the worker uses mmap for checkpoint load (faster first load). Use the `set_storage_type` / storage config if your app exposes it.

4. **Overlap**  
   The worker uses `num_overlap: 1` by default for speed. Passing a higher overlap (e.g. 2) can improve quality but increases time; leave at 1 when optimizing for speed.

5. **Warm runs**  
   First separation after load is “cold” (kernels, caches). Running a short dummy separation after load, or reusing the same model for many files, keeps runs warm and closer to the best timings in the doc.

6. **Optional: `torch.compile` (advanced)**  
   For PyTorch 2+, compiling the model once after load can reduce warm separation time at the cost of a longer first run. Not enabled by default; enable only if you profile and see benefit.

### CPU optimizations (and MPS relevance)

BS-RoFormer and Audio-Separator workers apply these CPU-specific optimizations (inspired by the original fast build):

| Optimization | Effect | MPS relevance |
|--------------|--------|---------------|
| **OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS** | Set to physical cores *before* importing torch. Maximizes parallelization for OpenMP/BLAS ops. | MPS uses CPU fallbacks for STFT/ISTFT on older macOS; these env vars apply to those fallbacks (set at process start). |
| **KMP_AFFINITY, KMP_BLOCKTIME** | Thread pinning and immediate release. ~34% speedup on Intel. | Less impact on Apple Silicon; env vars are harmless. |
| **torch.set_num_threads(physical_cores)** | PyTorch intra-op parallelism. | Skipped for MPS/CUDA (GPU handles parallelism). |
| **torch.set_num_interop_threads(cores//2)** | Inter-op parallelism. | Skipped for MPS/CUDA. |
| **Precision 'medium'** | ~10% speedup on CPU vs 'high'. | MPS uses 'high'; mixed precision via autocast. |
| **MAX_CACHED_MODELS=0 on CPU** | No model cache; load/unload per job. ~27% faster on CPU (avoids memory pressure/throttling). | MPS keeps cache (2 models); GPU benefits from caching. |

**Libraries:** No extra libraries needed for these gains. `torch.compile` (PyTorch 2+) can optionally reduce warm time; profile before enabling.

**Run all runnable tests:** `tests/scripts/run_all_tests.sh` runs Demucs E2E and BS-RoFormer E2E (when models dir exists), using frozen exe or Python worker wrappers. Compare printed timings to the expected ranges; aim to match or beat them. Every 10 seconds saved on cold or warm path is a concrete win.

**Profile Demucs when cold is slower than ~0.9s:** Use `scripts/utils/profile_demucs.py` to break down separation into load_track, apply_model, and save. With `--cprofile` it prints top functions and saves `/tmp/demucs_profile.prof` for snakeviz. Typical findings (Mac ARM MPS, 4s input): apply_model ~65–75%, save ~18%, load_track ~7–16%. Hotspots: MPS→CPU transfers for stem saves (~0.2s), group_norm/var_mean, conv1d. Try `--shifts 0` for faster (lower quality) runs.

**Profile BS-RoFormer when cold is slow (~10s):** Use `scripts/utils/profile_bsroformer.py` to break down separation into load_model, audio_load, model_to_device, demix, and save. With `--cprofile` it prints top functions and saves `/tmp/bsroformer_profile.prof` for snakeviz. Typical findings (Mac ARM MPS, 4s input): demix ~90%, audio_load ~5%, model_to_device ~5%. Hotspots in demix: GLU (~2s), scaled_dot_product_attention (~1.2s), linear layers (~0.25s). Use `--batch-size 4` or 8 for faster inference.

### Faster results (verified)

These settings are **verified faster** in project tests (frozen exe or Python worker, MPS on Mac ARM, 4s test input):

| Setting | Effect | Verified numbers |
|--------|--------|------------------|
| **Warm run (same model)** | Second separation in same process is much faster than first (cold). | Demucs: cold ~1.35s, **warm ~0.55s** (≈2.5× faster). Keep worker and model loaded for repeated files. |
| **BS-RoFormer `batch_size` 4 or 8** | More chunks per forward pass; fewer kernel launches. | Pass `"batch_size": 4` (or 8 if VRAM allows) in `separate` JSON. Configs often use 1–2; 4/8 can save **several seconds** on longer files. |
| **`use_fast: true` (default)** | Uses demix_fast (result on device, one `.cpu()` at end). | Worker defaults to this. Do not set `"use_fast": false` when optimizing for speed. |
| **E2E benchmark with `--batch-size`** | BS-RoFormer benchmark can pass batch_size to the worker. | `python tests/python/benchmarks/benchmark_worker_e2e.py ... --batch-size 4` for BS-RoFormer. |
| **Run-all script** | Single command runs Demucs + BS-RoFormer with faster defaults. | `./tests/scripts/run_all_tests.sh` uses **`--batch-size 4`** for BS-RoFormer when models dir exists. |

**Demucs (4s input, MPS):** Start→ready ~0.6s, load model ~0.2s, separate cold ~1.3s, **separate warm ~0.55s**. Total benchmark (ready + load + cold + warm) ~3s. Warm is the faster result; reuse the same worker and model for batch jobs.

**BS-RoFormer:** When you have a models dir, run the benchmark with `--batch-size 4` (or 8) to get faster separation than config default. The run-all script does this automatically.

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

9. **BS-RoFormer worker: use_fast default True** (Feb 2026)
   - Worker JSON `separate` now defaults to `use_fast: true` (demix_fast path) for best performance: result/counter stay on device, single `.cpu()` at end. Clients can pass `"use_fast": false` if needed.
   - **Final performance verification (release checklist)** added under Testing: device check, E2E benchmarks, mmap, CPU/MPS/CUDA verification steps so each release keeps high performance.

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

**Faster results:** Warm separation and larger `batch_size` (BS-RoFormer) are verified faster. See [Faster results (verified)](#faster-results-verified).

### Final performance verification (release checklist)

Before tagging a release or shipping a final update, run these checks to ensure high performance is preserved:

1. **Worker device and ready**
   - Start each worker with the target device (`--device cuda`, `--device mps`, or `--device cpu`). Confirm the ready message shows the correct `"device"` and (for CPU) `"threads"` (physical cores).

2. **E2E benchmarks**
   - Run `tests/python/benchmarks/benchmark_worker_e2e.py` for at least BS-RoFormer and Demucs (and Apollo if you use it), with your frozen exe and test audio (e.g. 4s WAV). Save the JSON output and compare to the [Expected Results](#expected-results) and [Mac ARM timing table](#mac-arm-pytorch-210-mps--4s-test-audio) (or your platform’s baseline). Cold/warm separation times should be in the same ballpark as documented.

3. **BS-RoFormer: use_fast default**
   - The worker defaults to `use_fast: true` (demix_fast path) for best throughput. If you override it, ensure you intend to; otherwise leave default for release.

4. **Model loading: mmap**
   - Workers use `safe_torch_load(..., mmap=True)` when supported and storage is SSD. Confirm no regression: first load and “same model cached” timings should match prior baselines (see [Model Loading Benchmarks](#model-loading-benchmarks)).

5. **CPU (when testing on CPU)**
   - During a separation, check Activity Monitor / Task Manager: you should see high multi-core usage (e.g. 600–800% on an 8-core machine). See [CPU multithreading (BS-RoFormer on CPU)](#cpu-multithreading-bsroformer-on-cpu).

6. **MPS (Mac ARM)**
   - Confirm ready shows `"device": "mps"`. Run a separation and check Activity Monitor → Window → GPU History for Metal activity. Optional: compare `tests/validation_output/mps_timing_after.json` (or similar) to previous runs. See [Verifying MPS usage (Mac ARM)](#verifying-mps-usage-mac-arm).

7. **CUDA (Windows/Linux)**
   - Run a separation and run `nvidia-smi -l 1` in another terminal. GPU-Util and Memory-Usage should be high during the run. See [Checking GPU utilization (BS-RoFormer)](#checking-gpu-utilization-bsroformer).

**Quick command (Mac, frozen exe in `dist/dsu/`, 4s test audio):**
```bash
python tests/python/benchmarks/benchmark_worker_e2e.py --exe dist/dsu/dsu-bsroformer --worker bsroformer --device mps --input tests/audio/test_4s.wav --output-dir tests/benchmark_output/bsroformer
```
Compare printed timings to the expected ranges above; repeat for Demucs/Apollo as needed.

**Verification script:** `tests/scripts/verify_performance.sh [bsroformer|demucs]` runs the E2E benchmark for the given worker (default BS-RoFormer), infers device (MPS on Mac ARM, else CPU), and prints timings. Prereqs: frozen exe in `dist/dsu/`, test audio, and for BS-RoFormer a models dir (or `DSU_MODELS`). Optional env: `DSU_EXE_DIR`, `DSU_DEVICE`, `DSU_TEST_INPUT`, `DSU_MODELS`.

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
