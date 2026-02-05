# Resource-Aware System Spec

## Overview

The system detects computer resources, saves settings, and adapts behavior to optimize timing. It learns from pipeline results (model + duration + file size) to build a scheme of how to work best on each machine.

---

## How "Power Up" Works

**When can the system run faster / more aggressively?**

1. **System report** — On first run (or every 24h), the orchestrator detects:
   - CPU cores, RAM, GPU (CUDA/MPS), VRAM
   - Writes `system_report` into `~/.dsu/runtime_policy.json`

2. **Resource tier** — Derived from the report:
   - **weak**: CPU only, &lt;8GB RAM → conservative (small chunks, no cache, sequential)
   - **medium**: 8–16GB RAM or MPS → default (moderate chunks, cache 1)
   - **strong**: 16GB+ RAM or 8GB+ VRAM → power up (larger chunks, batch_size 4, parallel light models)

3. **Power-up behavior** (when tier is **strong**):
   - `batch_size` 4 for bsroformer (vs 1–2 on weak)
   - `chunk_seconds` 10 for Apollo (vs 5 on weak)
   - `parallel_light_models: true` → multi-model sequences with light models (&lt;100MB) run 2 workers in parallel instead of sequential

4. **Learning** — Each run appends to `~/.dsu/pipeline_results.jsonl`. After 2+ runs per model, `get_suggested_params_for_model()` picks the chunk/batch/shifts config that had the best sec-per-minute. Those params are applied automatically when building jobs.

**Flow:** `run_real_test` → `update_runtime_policy_with_report()` → `derive_resource_tier()` → `update_with_resource_tier()` → policy gets `resource_tier` + tier defaults. Jobs use these when not overridden.

**Code:** `orchestrator/system_report.py` — `derive_resource_tier()`, `get_chunk_defaults_for_tier()`, `update_with_resource_tier()`

---

## 1. System Report (Resource Detection)

Detect and persist a **system report** that updates periodically and informs behavior:

| Resource | How to detect | Use |
|----------|---------------|-----|
| **CPU** | `os.cpu_count()`, physical cores (sysctl / wmic / /proc) | Thread count, parallel vs sequential |
| **RAM** | `psutil.virtual_memory()` or platform-specific | Chunk size, model cache size |
| **GPU** | `torch.cuda.is_available()`, `torch.backends.mps.is_available()` | Device choice, batch size |
| **VRAM** | `torch.cuda.get_device_properties(0).total_memory` | Max cached models, chunk size |
| **Disk** | `shutil.disk_usage()` | SSD vs HDD (mmap preference) |
| **Storage type** | Heuristic (SSD if fast random read) | mmap on/off |

**Output:** `~/.dsu/system_report.json` (or in runtime_policy)

```json
{
  "cpu_cores": 8,
  "cpu_physical": 8,
  "ram_gb": 16,
  "gpu": "mps",
  "vram_gb": 8,
  "disk_free_gb": 120,
  "storage_type": "ssd",
  "updated_at": "2026-02-02T12:00:00Z"
}
```

---

## 2. Model Size Registry

Models have different memory footprints. Small models can run in parallel or with larger chunks on weak machines.

| Size tier | Examples | Behavior |
|-----------|----------|----------|
| **Light** | mdx_q, fast-2, audiosep_base | Can run 2 in parallel on weak CPU |
| **Medium** | htdemucs_ft, logic_roformer | Single, moderate chunks |
| **Heavy** | bsrofo_sw, aname_4stem_large | Smaller chunks on weak machines |

**Source:** Add `params_m` (millions) or `size_mb` to registry; or scan checkpoint file sizes at startup.

---

## 3. Runtime Policy Extension

Extend `~/.dsu/runtime_policy.json` to store resource-aware settings:

```json
{
  "system_report": {
    "cpu_cores": 8,
    "ram_gb": 16,
    "gpu": "mps",
    "vram_gb": 8,
    "storage_type": "ssd",
    "updated_at": "2026-02-02T12:00:00Z"
  },
  "bsroformer": {
    "max_cached_models": 1,
    "chunk_seconds": 7.0,
    "chunk_overlap": 0.5,
    "batch_size": 4
  },
  "demucs": {
    "max_cached_models": 2,
    "shifts": 1
  },
  "audiosep": {
    "chunk_seconds": 30,
    "use_chunk": "auto"
  },
  "resource_tier": "strong",
  "parallel_light_models": true
}
```

**Resource tier** (derived from system_report in `derive_resource_tier()`):

| Condition | Tier |
|-----------|------|
| No GPU and RAM &lt; 8GB | weak |
| GPU (CUDA/MPS) and (RAM ≥ 16GB or VRAM ≥ 8GB) | strong |
| Otherwise | medium |

- **weak**: smaller chunks, no cache, sequential
- **medium**: default chunks, cache 1
- **strong**: larger chunks, batch_size 4, `parallel_light_models: true`

---

## 4. Chunk Settings by Resource Tier

| Tier | Apollo chunk_seconds | Apollo chunk_overlap | BS-RoFormer batch_size | Audiosep use_chunk |
|------|----------------------|----------------------|-------------------------|--------------------|
| weak | 5.0 | 0.5 | 1 | true (always) |
| medium | 7.0 | 0.5 | 2 | auto |
| strong | 10.0 | 0.5 | 4–8 | auto |

Workers read these from runtime_policy when not overridden by the job.

---

## 5. Pipeline Result Collection

Collect timing data from each run to build a **performance scheme**:

| Field | Source |
|-------|--------|
| model | Job |
| worker | Job |
| duration_sec | Audio file length |
| elapsed_sec | Worker response |
| device | Worker ready message |
| chunk_seconds | Job or policy |
| batch_size | Job or policy |

**Storage:** `~/.dsu/pipeline_results.jsonl` (append-only)

```jsonl
{"model":"bsroformer_4stem","worker":"bsroformer","duration_sec":33.2,"elapsed_sec":4.1,"device":"mps","chunk_seconds":7,"batch_size":4,"timestamp":"2026-02-02T12:05:00Z"}
{"model":"scnet_xl_ihf","worker":"bsroformer","duration_sec":33.2,"elapsed_sec":3.8,"device":"mps","chunk_seconds":7,"batch_size":4,"timestamp":"2026-02-02T12:06:00Z"}
```

**Use:** Aggregate to compute `sec_per_minute_audio` per model. When planning, pick chunk/batch that historically worked best for this model + device.

---

## 6. Parallel Execution for Light Models

When `resource_tier` is `strong` and `parallel_light_models` is true:

- For **multi-model sequence** (e.g. "bass different models"): if all models are **light**, run 2 workers in parallel instead of sequential.
- For **ensemble**: already single worker; no change.
- Heuristic: model is "light" if `params_m < 50` or checkpoint < 100MB.

---

## 7. Implementation Phases

| Phase | Scope | Status |
|-------|-------|--------|
| **1** | System report detection + save to runtime_policy | Done (`orchestrator/system_report.py`) |
| **2** | Resource tier derivation + chunk defaults per tier | Done |
| **3** | Pipeline result logging (orchestrator/run_real_test) | Done (`~/.dsu/pipeline_results.jsonl`) |
| **4** | Model size in registry (params_m or file size scan) | Done (`scripts/scan_model_sizes.py` → `model_sizes.json`) |
| **5** | Parallel light models (optional) | Done (when strong + all light) |
| **6** | Use collected results to suggest chunk/batch | Done (`get_suggested_params_for_model`) |

---

## 8. File Locations

| File | Purpose |
|------|---------|
| `~/.dsu/runtime_policy.json` | Policy + system_report + resource_tier |
| `~/.dsu/pipeline_results.jsonl` | Append-only timing log |
| `orchestrator/models_registry/registry.json` | Add `params_m` or `size_mb` per model |
| `orchestrator/system_report.py` | System report detection (orchestrator-only, not utils/workers) |

---

## 9. Backward Compatibility

- If `system_report` is missing, workers use current defaults.
- If `resource_tier` is missing, infer from first run or assume `medium`.
- Existing `runtime_policy.json` keys (max_cached_models, etc.) remain; new keys are additive.
