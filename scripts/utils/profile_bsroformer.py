#!/usr/bin/env python3
"""
Profile BS-RoFormer cold separation to find bottlenecks.

Usage:
  python scripts/utils/profile_bsroformer.py [--wav PATH] [--models-dir PATH] [--cprofile]

Output:
  - Phase timings: load_model, audio_load, model_to_device, demix, save
  - Optional: cProfile top functions (cumulative + tottime)
  - Optional: save .prof for snakeviz: python -m snakeviz /tmp/bsroformer_profile.prof

Compare with docs: cold ~10s (4s input). Use this to find where time is spent.

Typical profile findings (Mac ARM MPS, 4s input):
  - demix: ~90% of separation time (BSRoformer.forward)
  - audio_load: ~5% (librosa; first run can be slower due to cold imports)
  - model_to_device: ~5%
  - save: <1%

cProfile hotspots in demix:
  - torch._C._nn.glu: ~2s (248 calls)
  - scaled_dot_product_attention: ~1.2s
  - torch._C._nn.linear: ~0.25s
  - torch.istft: ~0.05s
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


def get_device():
    """Same device selection as worker."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def profile_bsroformer(wav_path: str, models_dir: str, model_name: str,
                       batch_size: int, use_fast: bool, use_cprofile: bool, out_dir: str):
    """Run BS-RoFormer cold separation with phase timings and optional cProfile."""
    import torch
    import numpy as np
    import soundfile as sf
    import librosa

    from workers.bsroformer_worker import load_model_impl, separate_impl

    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Models dir: {models_dir}")
    print(f"Input: {wav_path}")
    print(f"Batch size: {batch_size}, use_fast: {use_fast}")
    print()

    # --- Phase 1: Load model (cold) ---
    t0 = time.perf_counter()
    model, config, model_info = load_model_impl(model_name, models_dir)
    load_model_time = time.perf_counter() - t0
    print(f"  [1] Load model:       {load_model_time:.3f}s")

    # --- Phase 2-5: separate_impl with manual phase timing ---
    # We need to time phases; separate_impl doesn't return them, so we replicate the flow
    # for timing, then optionally run under cProfile.
    from utils.model_utils import demix, demix_fast

    sample_rate = getattr(config.audio, "samplerate", None) or getattr(config.audio, "sample_rate", 44100)
    config.inference.num_overlap = 1
    config.inference.batch_size = batch_size
    torch_device = torch.device(device)

    # Phase 2: Load audio
    t0 = time.perf_counter()
    audio, sr = librosa.load(wav_path, sr=sample_rate, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    audio_load_time = time.perf_counter() - t0
    print(f"  [2] Load audio:       {audio_load_time:.3f}s")

    # Phase 3: Model to device
    t0 = time.perf_counter()
    model = model.to(torch_device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    model_to_device_time = time.perf_counter() - t0
    print(f"  [3] Model to device:  {model_to_device_time:.3f}s")

    # Phase 4: Demix (inference)
    def run_demix():
        if use_fast:
            return demix_fast(
                config, model, audio, torch_device,
                model_type=model_info.get("type", "bs_roformer"), pbar=False
            )
        return demix(
            config, model, audio, torch_device,
            model_type=model_info.get("type", "bs_roformer"), pbar=False
        )

    if use_cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
    t0 = time.perf_counter()
    waveforms = run_demix()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    demix_time = time.perf_counter() - t0
    if use_cprofile:
        profiler.disable()

    print(f"  [4] Demix (inference): {demix_time:.3f}s")

    # Normalize to dict
    if not isinstance(waveforms, dict):
        stems = model_info.get("stems", ["vocals", "accompaniment"])
        waveforms = {stems[i]: waveforms[i] for i in range(min(len(stems), waveforms.shape[0]))}

    # Phase 5: Save stems
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    out_subdir = os.path.join(out_dir, basename)
    os.makedirs(out_subdir, exist_ok=True)
    subtype = "FLOAT"

    t0 = time.perf_counter()
    for stem, wav in waveforms.items():
        out_path = os.path.join(out_subdir, f"{stem.lower()}.wav")
        sf.write(out_path, wav.T, sample_rate, subtype=subtype)
    save_time = time.perf_counter() - t0
    print(f"  [5] Save stems:       {save_time:.3f}s")

    total = audio_load_time + model_to_device_time + demix_time + save_time
    print()
    print(f"  Total (excl. load_model): {total:.3f}s")
    print(f"  Total (incl. load_model): {total + load_model_time:.3f}s")
    print()
    print("  Phase breakdown:")
    print(f"    audio_load:   {100 * audio_load_time / total:.1f}%")
    print(f"    model_to_dev: {100 * model_to_device_time / total:.1f}%")
    print(f"    demix:        {100 * demix_time / total:.1f}%")
    print(f"    save:         {100 * save_time / total:.1f}%")

    if use_cprofile:
        prof_path = "/tmp/bsroformer_profile.prof"
        profiler.dump_stats(prof_path)
        print()
        print(f"  cProfile stats saved to: {prof_path}")
        print("  View with: python -m snakeviz /tmp/bsroformer_profile.prof")
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
    ap = argparse.ArgumentParser(description="Profile BS-RoFormer cold separation")
    ap.add_argument("--wav", default=None, help="Input WAV (default: tests/audio/test_4s.wav)")
    ap.add_argument("--models-dir", default=None, help="Models dir (contains models.json; default: DSU_MODELS or project root)")
    ap.add_argument("--model", default="bsroformer_4stem", help="Model name from models.json")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    ap.add_argument("--fast", action="store_true", help="Use demix_fast (default)")
    ap.add_argument("--no-fast", action="store_true", help="Use demix instead of demix_fast")
    ap.add_argument("--cprofile", action="store_true", help="Run cProfile and print top functions")
    ap.add_argument("--out", default=None, help="Output dir (default: /tmp/bsroformer_profile_out)")
    args = ap.parse_args()

    wav = args.wav or os.path.join(PROJECT_ROOT, "tests", "audio", "test_4s.wav")
    out = args.out or "/tmp/bsroformer_profile_out"
    use_fast = not args.no_fast

    # Try common locations: --models-dir, DSU_MODELS/bsroformer, DSU_MODELS, project root
    dsu = os.environ.get("DSU_MODELS") or os.environ.get("DSU_MODELS_DIR")
    candidates = [
        args.models_dir,
        os.path.join(dsu, "bsroformer") if dsu else None,
        dsu,
        PROJECT_ROOT,
    ]
    models_dir = None
    for c in candidates:
        if c and os.path.isdir(c):
            models_dir = c
            break
    if not models_dir:
        models_dir = args.models_dir or dsu or PROJECT_ROOT
    if not os.path.isdir(models_dir):
        print(f"ERROR: Models dir not found: {models_dir}", file=sys.stderr)
        return 1

    wav = ensure_test_audio(wav)
    os.makedirs(out, exist_ok=True)

    print("BS-RoFormer cold separation profile")
    print("-" * 50)
    profile_bsroformer(
        wav, models_dir, args.model,
        batch_size=args.batch_size,
        use_fast=use_fast,
        use_cprofile=args.cprofile,
        out_dir=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
