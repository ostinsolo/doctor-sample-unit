# Release Notes

## v1.4.2 (2026-02-05)

### Changes
- **CI**: Removed test-orchestrator workflow (orchestrator moved to separate repo Orchestrator_intent)
- **Gitignore**: orchestrator/, test_audiosep/, test_for_node.amxd

---

## v1.4.1 (2026-02-02)

### Changes
- **Audio Separator worker**: Apollo CPU threading controls and thread reporting in `ready/status`
- **Build scripts**: Manual mac build verifies `--max-cached-models` is supported
- **Docs**: Rebuild steps and Apollo threading parameters

---

## v1.4.0 (2026-02-02)

### Changes
- **Demucs worker**: Soundfile fallback for WAV saving on Windows (no TorchCodec wheels)
- **Model caching**: Configurable `--max-cached-models` and `set_cache_size` for Demucs & BS-RoFormer
- **BS-RoFormer**: `cache_enabled`, `max_cached_models` in ready status; `model_from_cache` in done
- **Audio Separator**: `cache_enabled`, `max_cached_models` in ready status; Apollo CPU unload after use
- **BS-RoFormer model**: Explicit `center=True` in STFT kwargs for consistency

### Builds (all architectures)
| Platform | File | GPU |
|----------|------|-----|
| Windows CUDA | `dsu-win-cuda.7z` | NVIDIA CUDA 12.6 |
| Windows CPU | `dsu-win-cpu.zip` | None |
| macOS ARM | `dsu-mac-arm.tar.gz` | MPS (Metal) |
| macOS Intel | `dsu-mac-intel.tar.gz` | CPU (manual build) |

---

## Frozen Executable Performance Benchmarks

### BS-RoFormer Worker

**Target Benchmark:** 40-45s  
**Actual Performance:** 36.05s (worker time) / 42.27s (total time)

**Status:** ✅ **Exceeds benchmark by 10%**

**Performance Comparison:**
- Our build: **36.05s** ✅ (fastest)
- Target: 40-45s
- ThirdPartyApps: 49.19s
- Earlier test: 45.68s

**Test Configuration:**
- Model: `bsrofo_sw` (6-stem separation)
- Platform: Intel Mac (x86_64)
- Build: Frozen executable with bundled Python runtime
- Location: `scripts/building/py/dist/dsu/dsu-bsroformer`

### All Workers

| Worker | Performance | Status |
|--------|------------|--------|
| BS-RoFormer | 36.05s | ✅ Exceeds target |
| Demucs | 21.05s | ✅ Matches target |
| Audio Separator | 18.41s | ✅ Matches target |

## Build Information

**Build Location:** `scripts/building/py/dist/dsu/`

**Executables:**
- `dsu-demucs` (53 KB)
- `dsu-bsroformer` (53 KB)
- `dsu-audio-separator` (53 KB)
- `lib/` folder (1.42 GB shared dependencies)

**Total Size:** 1.43 GB

## Release Instructions

For Intel Mac builds, upload manually to GitHub releases:
1. Navigate to `scripts/building/py/dist/dsu/`
2. Create archive: `tar -czf dsu-mac-intel.tar.gz dsu-* lib/`
3. Upload to GitHub release
