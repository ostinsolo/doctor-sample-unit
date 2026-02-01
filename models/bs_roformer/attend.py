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

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

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