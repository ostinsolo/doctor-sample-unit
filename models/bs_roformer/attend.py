from functools import wraps
from packaging import version
from collections import namedtuple

import os
import warnings
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

# -----------------------------------------------------------------------------
# Optional SageAttention
# -----------------------------------------------------------------------------

try:
    from sageattention import sageattn  # type: ignore
    _has_sage_attention = True
except Exception:  # pragma: no cover
    sageattn = None  # type: ignore[assignment]
    _has_sage_attention = False

# constants

FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# -----------------------------------------------------------------------------
# SDPA helpers
# -----------------------------------------------------------------------------

def _sdpa_is_available_probe(cuda_backend, name: str) -> bool:
    """
    Robustly query SDPA backend availability across PyTorch builds.

    Some Windows builds expose these attributes as `None` (or otherwise non-callable),
    so treat unknown / non-callable values as "not provably available" rather than
    raising TypeError.
    """
    attr = getattr(cuda_backend, name, None)
    if attr is None:
        return False
    if callable(attr):
        try:
            return bool(attr())
        except Exception:
            return False
    return bool(attr)

def _sdpa_call(q, k, v, *, dropout_p: float):
    """
    Call PyTorch SDPA while:
    - Avoiding deprecated torch.backends.cuda.sdp_kernel()
    - Suppressing extremely noisy SDPA backend warnings (Windows builds often
      don't ship flash/cudnn kernels, so these warnings are expected).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*Flash attention kernel not used.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*Torch was not compiled with flash attention.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*Memory efficient kernel not used because.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*Memory Efficient attention has been runtime disabled.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*cuDNN attention kernel not used because.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*cuDNN attention has been runtime disabled.*", category=UserWarning)
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p
        )

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        # Debug / control knobs (useful for frozen runtime verification + benchmarking):
        # - DSU_DISABLE_SAGEATTN=1  -> force-disable SageAttention even if installed
        # - DSU_ATTENTION_DEBUG=1   -> print one-line backend selection once
        disable_sage = str(os.environ.get("DSU_DISABLE_SAGEATTN", "")).strip().lower() in ("1", "true", "yes", "on")
        self.debug_attention = str(os.environ.get("DSU_ATTENTION_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on")

        # Prefer SageAttention when flash=True and installed (inference-focused).
        self.use_sage = bool(flash and _has_sage_attention and not disable_sage)
        self.last_backend = "init"
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')

        if device_version >= version.parse('8.0'):
            if os.name == 'nt':
                print_once('Windows OS detected, using math or mem efficient attention if input tensor is on cuda')
                self.cuda_config = FlashAttentionConfig(False, True, True)
            else:
                print_once('GPU Compute Capability equal or above 8.0, using flash attention if input tensor is on cuda')
                self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('GPU Compute Capability below 8.0, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def _einsum_attn(self, q, k, v, scale):
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        return einsum(f"b h i j, b h j d -> b h i d", attn, v)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # ---------------------------------------------------------------------
        # Priority 1: SageAttention (fast kernel on Windows builds)
        # ---------------------------------------------------------------------
        # SageAttention does not implement dropout in the same way; for safety,
        # only use it when dropout is effectively off (common for inference).
        if self.use_sage and q.is_cuda and (not self.training or self.dropout == 0.0):
            try:
                # BS-RoFormer uses q/k/v in (B, H, N, D) layout -> SageAttention HND.
                self.last_backend = "sageattention"
                if self.debug_attention:
                    print_once("Attention backend: SageAttention")
                return sageattn(q, k, v, tensor_layout="HND", is_causal=False)  # type: ignore[misc]
            except Exception as e:
                print_once(f"SageAttention failed ({type(e).__name__}: {e}); falling back to PyTorch SDPA / einsum.")
                self.use_sage = False

        # Check if there is a compatible device for flash attention
        # (kept for compatibility with original upstream logic)
        _ = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        # NOTE: On some Windows builds, torch.backends.cuda.is_*_sdp_available are None.
        # We therefore prefer "try SDPA, then fallback" rather than relying on probes.
        has_sdpa_backend = False
        if q.is_cuda and hasattr(torch.backends, "cuda"):
            cuda_b = torch.backends.cuda
            flash_available = _sdpa_is_available_probe(cuda_b, "is_flash_sdp_available")
            mem_available = _sdpa_is_available_probe(cuda_b, "is_mem_efficient_sdp_available")
            math_available = _sdpa_is_available_probe(cuda_b, "is_math_sdp_available")
            has_sdpa_backend = flash_available or mem_available or math_available

        if q.is_cuda and hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "SDPBackend"):
            backends_enum = torch.nn.attention.SDPBackend
            enabled_backends = []
            cuda_b = torch.backends.cuda
            # Prefer enumerating supported backends; availability probes may be unavailable on Windows.
            if _sdpa_is_available_probe(cuda_b, "is_flash_sdp_available") and hasattr(backends_enum, "FLASH_ATTENTION"):
                enabled_backends.append(backends_enum.FLASH_ATTENTION)
            if _sdpa_is_available_probe(cuda_b, "is_mem_efficient_sdp_available") and hasattr(backends_enum, "MEM_EFFICIENT"):
                enabled_backends.append(backends_enum.MEM_EFFICIENT)
            if _sdpa_is_available_probe(cuda_b, "is_math_sdp_available") and hasattr(backends_enum, "MATH"):
                enabled_backends.append(backends_enum.MATH)
            # If probes are missing, still try enabling everything present in the enum.
            if not enabled_backends:
                for attr in ("FLASH_ATTENTION", "MEM_EFFICIENT", "MATH"):
                    if hasattr(backends_enum, attr):
                        enabled_backends.append(getattr(backends_enum, attr))
            if enabled_backends:
                try:
                    with torch.nn.attention.sdpa_kernel(*enabled_backends):
                        self.last_backend = "pytorch_sdpa"
                        if self.debug_attention:
                            print_once("Attention backend: PyTorch SDPA (sdpa_kernel context)")
                        return _sdpa_call(q, k, v, dropout_p=self.dropout if self.training else 0.)
                except (RuntimeError, TypeError) as e:
                    if "No available kernel" not in str(e):
                        raise
                    print_once("SDPA reported no available kernel; using fallback.")

        # Finally, try plain SDPA regardless of probe results.
        # This covers builds where probes are unavailable but SDPA math kernel still works.
        if q.is_cuda or has_sdpa_backend:
            try:
                self.last_backend = "pytorch_sdpa"
                if self.debug_attention:
                    print_once("Attention backend: PyTorch SDPA")
                return _sdpa_call(q, k, v, dropout_p=self.dropout if self.training else 0.)
            except (RuntimeError, TypeError) as e:
                if "No available kernel" not in str(e):
                    raise
                print_once("SDPA reported no available kernel; using einsum attention.")

        scale = default(self.scale, q.shape[-1] ** -0.5)
        self.last_backend = "einsum"
        return self._einsum_attn(q, k, v, scale)

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        # similarity

        self.last_backend = "einsum"
        return self._einsum_attn(q, k, v, scale)