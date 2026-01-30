# Model Changes from Original (bs_roformer & scnet)

This document summarizes modifications made to the `models/bs_roformer/` and `models/scnet/` files in the DSU project compared to:

- **Local original:** commit `86830b2` ("Add model architectures, configs, and utilities")
- **Upstream GitHub:** [ostinsolo/BS-RoFormer-freeze](https://github.com/ostinsolo/BS-RoFormer-freeze) — [models/](https://github.com/ostinsolo/BS-RoFormer-freeze/tree/main/models)

**View originals:**
```bash
git show 86830b2:models/bs_roformer/<filename>
# or raw from upstream:
# https://raw.githubusercontent.com/ostinsolo/BS-RoFormer-freeze/main/models/bs_roformer/<filename>
# https://raw.githubusercontent.com/ostinsolo/BS-RoFormer-freeze/main/models/scnet/<filename>
```

---

## bs_roformer/ — Summary

| File | Status |
|------|--------|
| `__init__.py` | **Unchanged** |
| `attend.py` | **Modified** |
| `attend_sage.py` | **Modified** |
| `bs_conformer.py` | **Unchanged** |
| `bs_roformer.py` | **Unchanged** |
| `bs_roformer_experimental.py` | **Unchanged** |
| `mel_band_conformer.py` | **Unchanged** |
| `mel_band_roformer.py` | **Unchanged** |
| `mel_band_roformer_experimental.py` | **Unchanged** |

Only **`attend.py`** and **`attend_sage.py`** have been modified. All other files match [BS-RoFormer-freeze](https://github.com/ostinsolo/BS-RoFormer-freeze/tree/main/models/bs_roformer).

---

## scnet/ — Summary

| File | Status |
|------|--------|
| `__init__.py` | **Unchanged** |
| `scnet.py` | **Modified** |
| `scnet_masked.py` | **Unchanged** |
| `scnet_tran.py` | **Unchanged** |
| `separation.py` | **Unchanged** |

Only **`scnet.py`** has been modified for MPS and cx_Freeze compatibility.

---

## 1. attend.py (bs_roformer)

**Purpose of changes:** Improve cross-platform compatibility (especially Windows frozen builds) and add optional SageAttention support for faster inference.

### 1.1 Optional SageAttention

- **Added:** Optional import of `sageattention.sageattn`. If installed, it is used as the primary attention backend on CUDA when `flash=True`.
- **Added:** `DSU_DISABLE_SAGEATTN=1` env var to force-disable SageAttention.
- **Added:** `DSU_ATTENTION_DEBUG=1` env var to print which attention backend is used (once).

### 1.2 SDPA robustness (Windows / frozen builds)

- **Added:** `_sdpa_is_available_probe()` – Safely checks if Flash/MemEfficient/Math SDPA backends are available. On some Windows builds, `torch.backends.cuda.is_*_sdp_available` can be `None` or non-callable; the probe avoids `TypeError`.
- **Added:** `_sdpa_call()` – Wraps `F.scaled_dot_product_attention()` and suppresses noisy SDPA-related `UserWarning`s (e.g. "Flash attention kernel not used", "cuDNN attention has been runtime disabled").
- **Replaced:** Deprecated `torch.backends.cuda.sdp_kernel()` usage with `torch.nn.attention.sdpa_kernel()` where available.
- **Added:** Fallback chain: SageAttention → PyTorch SDPA (with probes) → PyTorch SDPA (direct) → einsum.

### 1.3 Einsum fallback

- **Extracted:** `_einsum_attn()` – Encapsulates the original einsum-based attention.
- **Added:** Explicit fallback to einsum when SDPA reports "No available kernel" (e.g. on builds without flash/cudnn kernels).

### 1.4 Debug attributes

- **Added:** `self.use_sage` – Whether SageAttention is used.
- **Added:** `self.last_backend` – Last backend used (`"sageattention"`, `"pytorch_sdpa"`, or `"einsum"`).

---

## 2. attend_sage.py (bs_roformer)

**Purpose of changes:** Same SDPA robustness and warning suppression as `attend.py`, for the Sage-first attention path. Also fixes a bug in [BS-RoFormer-freeze](https://raw.githubusercontent.com/ostinsolo/BS-RoFormer-freeze/main/models/bs_roformer/attend_sage.py): the original had `return out` before the `try/except` block, making the fallback path unreachable.

### 2.1 SDPA helpers

- **Added:** `_sdpa_is_available_probe()` – Same as in `attend.py`.
- **Added:** `_sdpa_call()` – Same as in `attend.py`, but with `attn_mask` and `is_causal` parameters for the Sage variant.

### 2.2 SageAttention path cleanup

- **Removed:** Duplicate/dead code (`out = sageattn(...); return out;` followed by another `out = sageattn(...)` in a try block).
- **Simplified:** Single `return sageattn(...)` in the try block.

### 2.3 PyTorch SDPA fallback

- **Replaced:** Direct use of `torch.backends.cuda.sdp_kernel()` with the same robust logic as in `attend.py`:
  - Probe availability before calling.
  - Use `torch.nn.attention.sdpa_kernel()` when available.
  - Fall back to direct SDPA call if probes are missing.
  - Fall back to einsum on "No available kernel".
- **Added:** `warnings` import and suppression of SDPA-related warnings in `_sdpa_call`.

---

## 3. scnet.py (scnet)

**Purpose of changes:** MPS (Apple Silicon) and cx_Freeze compatibility for SCNet inference.

### 3.1 STFT window buffer

- **Added:** `self.register_buffer("stft_window", torch.ones(win_size), persistent=False)` — Rectangular window (all ones), matching original BS-RoFormer-freeze / PyTorch default. Registered as buffer for MPS/cx_Freeze device portability.
- **Modified:** `torch.stft` / `torch.istft` now receive `stft_config` with `window=self.stft_window.to(device)` instead of relying on default window creation. This ensures correct behavior when tensors are on MPS and the window must match device.

### 3.2 complex32 → complex64

- **Added:** `if x.dtype == torch.complex32: x = x.to(torch.complex64)` — On some MPS/ISTFT paths, PyTorch can produce `complex32`. Converting to `complex64` avoids downstream errors.

### 3.3 Reference

Upstream: [BS-RoFormer-freeze scnet.py](https://raw.githubusercontent.com/ostinsolo/BS-RoFormer-freeze/main/models/scnet/scnet.py) does not include these changes.

---

## Rationale

1. **Windows frozen builds:** Many Windows DSU builds ship without Flash/cuDNN attention kernels. The original code could raise `TypeError` when probing backend availability or emit noisy warnings. The changes add safe probing and fallbacks so the model still runs.

2. **SageAttention (optional):** When installed, SageAttention provides faster attention on CUDA. The changes integrate it as an optional, highest-priority backend with clean fallback.

3. **PyTorch 2.x API changes:** `torch.backends.cuda.sdp_kernel()` is deprecated. The new code prefers `torch.nn.attention.sdpa_kernel()` when present.

4. **Consistency:** Both `attend.py` and `attend_sage.py` now share the same SDPA probing and fallback behavior.

5. **MPS / cx_Freeze (scnet):** SCNet needs explicit STFT window handling and `complex32` conversion to run correctly on Apple Silicon and in frozen executables.

---

## Viewing the full diff

```bash
git diff 86830b2..HEAD -- models/bs_roformer/
git diff 86830b2..HEAD -- models/scnet/
```

Or compare with [BS-RoFormer-freeze](https://github.com/ostinsolo/BS-RoFormer-freeze/tree/main/models) on GitHub.
