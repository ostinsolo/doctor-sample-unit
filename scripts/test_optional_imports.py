#!/usr/bin/env python3
"""Test optional package imports (same logic as build_dsu.py)."""
for pkg in ("sageattention", "triton", "torchcodec"):
    try:
        __import__(pkg)
        kind = "CUDA" if pkg in ("sageattention", "triton") else "torchcodec"
        print(f"  Including optional package ({kind}): {pkg}")
    except (ImportError, RuntimeError, OSError) as e:
        print(f"  Skipping optional package (not available): {pkg} - {type(e).__name__}")
