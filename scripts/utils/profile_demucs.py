#!/usr/bin/env python3
"""
Profile Demucs cold separation to find bottlenecks.

Usage:
  python scripts/utils/profile_demucs.py [--wav PATH] [--cprofile] [--model htdemucs]

Output:
  - Phase timings: load_track, apply_model, save
  - Optional: cProfile top functions (cumulative + tottime)
  - Optional: save .prof for snakeviz: python -m snakeviz /tmp/demucs_profile.prof

Compare with docs: cold ~0.9s target (4s input). If slower, use this to find where.

Typical profile findings (Mac ARM MPS, 4s input):
  - apply_model: ~65-75% of separation time (htdemucs.forward, hdemucs, group_norm)
  - save: ~18% (torchaudio + .cpu() transfers)
  - load_track: ~7-16% (ffmpeg/torchaudio)

cProfile hotspots:
  - method 'cpu' (MPS->CPU): ~0.2s for stem saves
  - torch.var_mean / group_norm: ~0.1s (74 calls)
  - torch.conv1d, pad: ~0.09s each
  - shifts=1 doubles inference; shifts=0 faster but lower quality
"""
import argparse
import cProfile
import os
import pstats
import sys
import time

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def ensure_test_audio(wav_path: str) -> str:
    """Ensure test WAV exists; generate if missing."""
    if os.path.isfile(wav_path):
        return wav_path
    audio_dir = os.path.dirname(wav_path)
    os.makedirs(audio_dir, exist_ok=True)
    gen = os.path.join(PROJECT_ROOT, "tests", "python", "generate_test_audio.py")
    if os.path.isfile(gen):
        import subprocess
        subprocess.run([sys.executable, gen], cwd=PROJECT_ROOT, check=True)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Test WAV not found: {wav_path}")
    return wav_path


def profile_demucs(wav_path: str, model_name: str, use_cprofile: bool, out_dir: str, shifts: int | None = None):
    """Run Demucs cold separation with phase timings and optional cProfile."""
    import torch
    from pathlib import Path

    # Import worker helpers (same flow as demucs_worker.py)
    from workers.demucs_worker import (
        _configure_demucs_cache,
        _load_track_safe,
        _save_audio_demucs_or_fallback,
    )
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    _configure_demucs_cache()

    # Device (same as worker)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Input: {wav_path}")
    if shifts is not None:
        print(f"Shifts: {shifts} (override)")
    print()

    # --- Phase 1: Load model (cold) ---
    t0 = time.perf_counter()
    model = get_model(model_name)
    model.to(device)
    model.eval()
    load_model_time = time.perf_counter() - t0
    print(f"  [1] Load model:     {load_model_time:.3f}s")

    # --- Phase 2: Load track ---
    t0 = time.perf_counter()
    wav, load_err = _load_track_safe(wav_path, model.audio_channels, model.samplerate)
    if load_err:
        raise RuntimeError(load_err)
    load_track_time = time.perf_counter() - t0
    print(f"  [2] Load track:     {load_track_time:.3f}s")

    # Normalize (same as worker)
    ref = wav.mean(0)
    wav = wav - ref.mean()
    s = ref.std()
    if s > 1e-8:
        wav = wav / s

    # --- Phase 3: apply_model (inference) ---
    if shifts is None:
        shifts = 0 if model_name.lower() in ("mdx_q", "mdx_extra_q") else 1
    apply_kw = dict(shifts=shifts, overlap=0.25, progress=False, device=device, num_workers=0)

    def run_apply():
        with torch.no_grad():
            return apply_model(model, wav[None].to(device), **apply_kw)[0]

    if use_cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
    t0 = time.perf_counter()
    sources = run_apply()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    apply_time = time.perf_counter() - t0
    if use_cprofile:
        profiler.disable()

    print(f"  [3] apply_model:   {apply_time:.3f}s")

    # Denormalize
    if s > 1e-8:
        sources = sources * s
    sources = sources + ref.mean()

    # --- Phase 4: Save stems ---
    os.makedirs(out_dir, exist_ok=True)
    stems = list(model.sources)
    save_kw = {"samplerate": model.samplerate, "clip": "rescale", "bits_per_sample": 16}

    t0 = time.perf_counter()
    for i, stem in enumerate(stems):
        out_path = os.path.join(out_dir, f"{stem}.wav")
        _save_audio_demucs_or_fallback(sources[i], out_path, **save_kw)
    save_time = time.perf_counter() - t0
    print(f"  [4] Save stems:     {save_time:.3f}s")

    total = load_track_time + apply_time + save_time
    print()
    print(f"  Total (excl. load_model): {total:.3f}s")
    print(f"  Total (incl. load_model): {total + load_model_time:.3f}s")
    print()
    print("  Phase breakdown:")
    print(f"    load_track:   {100 * load_track_time / total:.1f}%")
    print(f"    apply_model:  {100 * apply_time / total:.1f}%")
    print(f"    save:         {100 * save_time / total:.1f}%")

    if use_cprofile:
        prof_path = "/tmp/demucs_profile.prof"
        profiler.dump_stats(prof_path)
        print()
        print(f"  cProfile stats saved to: {prof_path}")
        print("  View with: python -m snakeviz /tmp/demucs_profile.prof")
        print()
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        print("=" * 70)
        print("TOP 25 BY CUMULATIVE TIME:")
        print("=" * 70)
        stats.print_stats(25)
        print()
        stats.sort_stats("tottime")
        print("=" * 70)
        print("TOP 25 BY SELF TIME:")
        print("=" * 70)
        stats.print_stats(25)

    return total


def main():
    ap = argparse.ArgumentParser(description="Profile Demucs cold separation")
    ap.add_argument("--wav", default=None, help="Input WAV (default: tests/audio/test_4s.wav)")
    ap.add_argument("--model", default="htdemucs", help="Model name (default: htdemucs)")
    ap.add_argument("--shifts", type=int, default=None, help="Overlap shifts (0=faster/less quality, 1=default)")
    ap.add_argument("--cprofile", action="store_true", help="Run cProfile and print top functions")
    ap.add_argument("--out", default=None, help="Output dir (default: /tmp/demucs_profile_out)")
    args = ap.parse_args()

    wav = args.wav or os.path.join(PROJECT_ROOT, "tests", "audio", "test_4s.wav")
    out = args.out or "/tmp/demucs_profile_out"

    wav = ensure_test_audio(wav)
    os.makedirs(out, exist_ok=True)

    print("Demucs cold separation profile")
    print("-" * 50)
    profile_demucs(wav, args.model, args.cprofile, out, shifts=args.shifts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
