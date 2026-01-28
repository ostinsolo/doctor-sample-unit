import os
import sys
import time

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.bs_roformer.attend import Attend


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_once(*, disable_sage: bool, shape=(1, 8, 512, 64), iters: int = 200, warmup: int = 30) -> dict:
    os.environ["DSU_DISABLE_SAGEATTN"] = "1" if disable_sage else "0"
    os.environ["DSU_ATTENTION_DEBUG"] = "1"

    if not torch.cuda.is_available():
        return {"ok": False, "reason": "CUDA not available"}

    device = torch.device("cuda")
    dtype = torch.float16

    b, h, n, d = shape
    q = torch.randn(b, h, n, d, device=device, dtype=dtype)
    k = torch.randn(b, h, n, d, device=device, dtype=dtype)
    v = torch.randn(b, h, n, d, device=device, dtype=dtype)

    attn = Attend(flash=True, dropout=0.0).to(device)
    attn.eval()

    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = attn(q, k, v)
        _sync()

        t0 = time.perf_counter()
        for _ in range(iters):
            out = attn(q, k, v)
        _sync()
        t1 = time.perf_counter()

    # quick correctness sanity
    ok = out.isfinite().all().item()
    ms_per = (t1 - t0) * 1000.0 / float(iters)

    return {
        "ok": bool(ok),
        "disable_sage": disable_sage,
        "backend": getattr(attn, "last_backend", "unknown"),
        "shape": tuple(shape),
        "dtype": str(dtype),
        "device": str(device),
        "ms_per_iter": ms_per,
    }


def main() -> int:
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    r1 = bench_once(disable_sage=False)
    print("\n=== Sage enabled ===")
    print(r1)

    r2 = bench_once(disable_sage=True)
    print("\n=== Sage disabled (SDPA baseline) ===")
    print(r2)

    if not r1.get("ok", False) or not r2.get("ok", False):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

