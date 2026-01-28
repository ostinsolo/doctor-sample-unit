import sys


def main() -> int:
    print("=== Python ===")
    print(sys.version)

    print("\n=== Torch ===")
    try:
        import torch
    except Exception as e:
        print("torch import FAILED:", type(e).__name__, e)
        return 2

    print("torch.__version__:", getattr(torch, "__version__", "unknown"))
    print("torch.version.cuda:", getattr(getattr(torch, "version", None), "cuda", None))
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("has F.scaled_dot_product_attention:", hasattr(__import__("torch.nn.functional", fromlist=["scaled_dot_product_attention"]), "scaled_dot_product_attention"))
    print("has torch.nn.attention.sdpa_kernel:", hasattr(getattr(torch.nn, "attention", object()), "sdpa_kernel"))

    # SDPA availability (PyTorch 2.x)
    print("\n=== SDPA backends ===")
    try:
        cuda_b = getattr(torch.backends, "cuda", None)
        if cuda_b is None:
            print("torch.backends.cuda: <missing>")
        else:
            for name in ("is_flash_sdp_available", "is_mem_efficient_sdp_available", "is_math_sdp_available"):
                attr = getattr(cuda_b, name, None)
                if callable(attr):
                    try:
                        val = attr()
                        print(f"{name}():", val, f"(type={type(val).__name__})")
                    except Exception as e:
                        print(f"{name}(): EXCEPTION", type(e).__name__, e)
                else:
                    print(f"{name}:", attr, f"(callable={callable(attr)}, type={type(attr).__name__})")
    except Exception as e:
        print("SDPA backend probe FAILED:", type(e).__name__, e)

    print("\n=== Optional deps ===")
    for mod in ("sageattention", "triton", "triton_windows", "triton-windows"):
        try:
            __import__(mod)
            print(f"{mod}: OK")
        except Exception as e:
            print(f"{mod}: FAIL ({type(e).__name__}: {e})")

    # SageAttention version, if present
    try:
        import sageattention  # type: ignore

        print("sageattention.__version__:", getattr(sageattention, "__version__", "unknown"))
        try:
            from sageattention import sageattn  # noqa: F401
            print("sageattention.sageattn: OK")
        except Exception as e:
            print("sageattention.sageattn: FAIL", type(e).__name__, e)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

