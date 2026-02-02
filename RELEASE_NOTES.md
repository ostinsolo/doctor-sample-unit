# Release Notes

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
