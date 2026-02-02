# libsamplerate on macOS ARM (Apple Silicon)

## The issue

When running the **audio-separator** worker (VR separation) on an **ARM Mac**, you may see:

```text
cannot load library '.../lib/samplerate/_samplerate_data/libsamplerate.dylib':
mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64')
```

VR separation then returns no output files (worker sends `"done"` with `files: []`; we now emit an error in that case).

## Root cause

1. **audio-separator** (and **librosa**) use the Python package **samplerate** for high-quality resampling when `res_type` is `sinc_best`, `sinc_medium`, etc.

2. **samplerate 0.1.0** (pinned by audio-separator) ships a **single** `libsamplerate.dylib` that is **x86_64 only**. There is no universal2/arm64 build in that version.

3. On an ARM Mac:
   - If the venv was created with **native ARM Python**, pip still installs **samplerate 0.1.0** (to satisfy audio-separator). That package’s bundled `.dylib` is x86_64, so loading it fails on ARM.
   - If the venv was created with **x86_64 Python** (e.g. under Rosetta), the same x86_64 dylib is used; when you run the **frozen** app (built on that machine or elsewhere), the frozen bundle can contain that x86_64 dylib, and when run on ARM it fails.

4. **cx_Freeze** copies whatever is in `site-packages/samplerate` (including `_samplerate_data/libsamplerate.dylib`) into `dist/dsu/lib/`. So the frozen app ends up with the wrong architecture if the build env had 0.1.0.

## Correct fix: use samplerate ≥ 0.2.3 on Mac ARM

- **samplerate 0.2.3** on PyPI provides **universal2** wheels for macOS (e.g. `macosx_10_9_universal2.whl`), i.e. **ARM64 + x86_64** in one wheel.
- 0.2.3 also **statically links** libsamplerate (pybind11); there is **no separate `.dylib`** to bundle, so the frozen app gets the right architecture from the wheel.
- The **API** is compatible: `samplerate.resample(input_data, ratio, converter_type='sinc_best')` works the same way, so **librosa** and **audio-separator** work without code changes.

audio-separator’s metadata pins `samplerate==0.1.0`, so pip will install 0.1.0 by default. To get the correct version on Mac ARM you must **override** that pin after installing dependencies.

## What to do

### 1. Local venv (Mac ARM)

After creating the venv and installing from `requirements-mac-mps.txt`, force the correct samplerate:

```bash
pip install -r requirements-mac-mps.txt
pip install 'samplerate>=0.2.3' --force-reinstall
```

**Expected pip message:** Pip may print a dependency conflict: `audio-separator 0.41.0 requires samplerate==0.1.0, but you have samplerate 0.2.3 which is incompatible.` That is **expected and safe to ignore**. We intentionally override the pin so the correct (universal2) library is used on Mac ARM. The API is compatible (`samplerate.resample(..., converter_type=...)` and `librosa.resample(..., res_type='sinc_best')` work with 0.2.3); VR separation has been verified with 0.2.3.

Then build as usual (`python scripts/building/py/build_dsu.py` from project root). The frozen app will use the universal2 samplerate (no x86_64 dylib).

### 2. CI (GitHub Actions Mac ARM)

The release workflow already runs a step that forces `samplerate>=0.2.3` after installing `requirements-mac-mps.txt`, so the Mac ARM artifact is built with the correct architecture.

### 3. Verify architecture (optional)

Check that the **venv** no longer has an x86_64-only dylib:

```bash
# If you still have samplerate 0.1.0 and a dylib:
file venv/lib/python3.10/site-packages/samplerate/_samplerate_data/libsamplerate.dylib

# After upgrading to 0.2.3 there is no _samplerate_data dylib; the native code is in a .so:
file venv/lib/python3.10/site-packages/samplerate/*.so
# Should show: Mach-O 64-bit bundle ... arm64 (or universal2)
```

### 4. If you cannot upgrade samplerate

- **resampy** is already a dependency; librosa can use it for `res_type` like `kaiser_best` / `kaiser_fast`. It does not use libsamplerate.  
- audio-separator’s VR models specify `res_type` in their JSON (e.g. `sinc_best`, `polyphase`, `kaiser_fast`). For models that use `sinc_*`, librosa loads the **samplerate** package; if that fails, the whole VR step can fail. So fixing the samplerate package is the right solution.

## Summary

| Item | Detail |
|------|--------|
| **Wrong** | samplerate **0.1.0** on Mac ARM → x86_64-only `libsamplerate.dylib` → load error on ARM. |
| **Correct** | samplerate **≥ 0.2.3** on Mac ARM → universal2 wheel, no separate dylib, same API. |
| **Action** | After `pip install -r requirements-mac-mps.txt`, run `pip install 'samplerate>=0.2.3' --force-reinstall` on Mac ARM (and in CI for the Mac ARM build). |
