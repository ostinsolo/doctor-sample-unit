# audio-separator: Install from source, libsamplerate, MPS

## Why use source instead of pip?

- **Inspect and patch** the VR/MDX code paths (e.g. STFT/ISTFT, device handling).
- **Apply MPS optimizations** (e.g. try MPS-first for PyTorch STFT in MDX) that upstream may not yet ship.
- **Align with libsamplerate** fix on Mac ARM (see [LIBSAMPLERATE_MAC_ARM.md](LIBSAMPLERATE_MAC_ARM.md)).

We normally install `audio-separator` from PyPI. For development or MPS tuning, you can install from the GitHub source.

## Clone and install from source

```bash
git clone https://github.com/karaokenerds/python-audio-separator.git
cd python-audio-separator
# or: git clone https://github.com/nomadkaraoke/python-audio-separator.git

pip install -e ".[cpu]"
# or [gpu] if you use CUDA
```

**Mac ARM (MPS):** After installing deps, force the correct samplerate (see below). Use `[cpu]`; we run models on MPS via PyTorch, not CUDA.

## libsamplerate on Mac ARM

audio-separator pins `samplerate==0.1.0`. That version ships an **x86_64-only** `libsamplerate.dylib`, which fails on Apple Silicon. Use **samplerate ≥ 0.2.3** (universal2, same API).

```bash
pip install 'samplerate>=0.2.3' --force-reinstall
```

Pip may report: `audio-separator 0.41.0 requires samplerate==0.1.0, but you have samplerate 0.2.3 which is incompatible.` **That is expected and safe to ignore.** We intentionally override the pin. VR separation (and any `sinc_*` resampling via librosa) works with 0.2.3. See [LIBSAMPLERATE_MAC_ARM.md](LIBSAMPLERATE_MAC_ARM.md).

## VR vs MDX: where STFT/ISTFT run

| Architecture | STFT/ISTFT | Device handling |
|-------------|------------|------------------|
| **VR** | `librosa.stft` / `librosa.istft` in `spec_utils` | NumPy/CPU. Model forward runs on MPS. |
| **MDX** | PyTorch `torch.stft` / `torch.istft` in `uvr_lib_v5/stft.py` | Currently: MPS → CPU for STFT/ISTFT, then back to device. |

VR uses librosa (CPU) for spectrogram conversion; the heavy model forward is on MPS. MDX uses PyTorch STFT/ISTFT; the packaged code explicitly moves tensors to CPU for those ops when device is MPS.

## Optional: MPS-first STFT patch for MDX

In **PyTorch 2.10+**, `torch.stft` and `torch.istft` run on MPS. The MDX `stft.py` class still forces a CPU round-trip for non-CUDA/non-CPU devices. You can patch it to **try MPS first**, then fall back to CPU only on error (similar to our `models/bs_roformer/mps_ops.py`).

**Location when installed from source (editable):**  
`python-audio-separator/audio_separator/separator/uvr_lib_v5/stft.py`

**Change:** In `STFT.__call__` and `STFT.inverse`, replace the “non-standard device → CPU” logic with:

1. Run `torch.stft` / `torch.istft` on the tensor’s device (including MPS).
2. On exception, move to CPU, run STFT/ISTFT, then move result back to the original device.

Example pattern (same idea as `mps_ops.stft_mps_fallback` / `istft_mps_fallback`):

```python
def __call__(self, input_tensor):
    dev = input_tensor.device
    window = self.hann_window.to(dev)
    # ... reshape etc. ...
    try:
        stft_output = torch.stft(..., window=window, ...)
    except Exception:
        stft_output = torch.stft(..., window=window.to("cpu"), ...)
        input_tensor = input_tensor.cpu()
        # ... run stft on cpu, then .to(dev) ...
    return ...
```

Apply the analogous logic in `inverse`. This reduces CPU round-trips for MDX on MPS when PyTorch supports STFT/ISTFT on MPS.

**Note:** VR is unaffected; it uses `spec_utils` (librosa), not `stft.py`.

## Checking that you use the correct setup

- **Samplerate:**  
  `python -c "import samplerate; print(samplerate.__version__)"`  
  should be `0.2.3` or higher on Mac ARM.

- **audio-separator source:**  
  `pip show -f audio-separator`  
  should point at your local clone if you used `pip install -e .`.

- **MPS:**  
  `python -c "import torch; print(torch.backends.mps.is_available())"`  
  should be `True` on supported Macs.

## References

- [LIBSAMPLERATE_MAC_ARM.md](LIBSAMPLERATE_MAC_ARM.md) — libsamplerate fix on Mac ARM.
- [WORKER_SYSTEM.md](WORKER_SYSTEM.md) — workers, device selection, MPS usage.
- [python-audio-separator](https://github.com/karaokenerds/python-audio-separator) — upstream repo.
