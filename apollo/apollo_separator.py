#!/usr/bin/env python3
"""
Apollo Audio Restoration wrapper for frozen binary.
Converts lossy compressed audio to higher quality.

Models supported:
- Official JusperLee (feature_dim=256)
- Lew Universal (feature_dim=384) - RECOMMENDED
- Lew V2 (feature_dim=192)
- Big/EDM by essid (feature_dim=256)
"""
import sys
import os
import torch
import numpy as np
import soundfile as sf
from scipy import signal
import argparse

# Add apollo directory to path
apollo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, apollo_dir)

from look2hear.models.apollo import Apollo


def get_device():
    """Get best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS has issues with complex numbers in STFT, force CPU for now
    # elif torch.backends.mps.is_available():
    #     return torch.device('mps')
    return torch.device('cpu')


def load_audio(file_path, sr=44100):
    """Load audio file and resample if needed using soundfile/scipy"""
    audio, samplerate = sf.read(file_path, dtype='float32')
    
    # Convert to [C, T] format (soundfile returns [T, C] or [T] for mono)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # Mono: [1, T]
    else:
        audio = audio.T  # Stereo: [C, T]
    
    # Resample if needed using scipy
    if samplerate != sr:
        num_samples = int(len(audio[0]) * sr / samplerate)
        audio = np.array([signal.resample(ch, num_samples) for ch in audio])
    
    # Convert to torch tensor [1, C, T]
    return torch.from_numpy(audio).unsqueeze(0).float()


def save_audio(file_path, audio, samplerate=44100):
    """Save audio to file using soundfile"""
    audio = audio.squeeze(0).cpu().numpy()  # [C, T]
    
    # Convert to [T, C] for soundfile (or [T] for mono)
    if audio.shape[0] == 1:
        audio = audio[0]  # Mono
    else:
        audio = audio.T  # Stereo
    
    sf.write(file_path, audio, samplerate, subtype='PCM_16')


def _chunk_restore(model, audio, sr, chunk_seconds, overlap_seconds, device):
    """Restore audio in overlapping chunks to limit memory usage."""
    total_samples = audio.shape[-1]
    chunk_samples = max(1, int(chunk_seconds * sr))
    overlap_samples = max(0, int(overlap_seconds * sr))
    step = chunk_samples - overlap_samples
    if step <= 0:
        raise ValueError("chunk_seconds must be greater than overlap_seconds")

    output = torch.zeros_like(audio, device="cpu")
    weight_sum = torch.zeros_like(audio, device="cpu")

    for start in range(0, total_samples, step):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[:, :, start:end]
        chunk_len = chunk.shape[-1]
        if chunk_len < chunk_samples:
            pad = torch.zeros((chunk.shape[0], chunk.shape[1], chunk_samples - chunk_len), device=chunk.device, dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=-1)

        with torch.inference_mode():
            restored = model(chunk)

        restored = restored[:, :, :chunk_len].to("cpu")

        weight = torch.ones((1, 1, chunk_len), device="cpu", dtype=restored.dtype)
        if overlap_samples > 0:
            if start > 0:
                fade_in_len = min(overlap_samples, chunk_len)
                weight[:, :, :fade_in_len] *= torch.linspace(0.0, 1.0, fade_in_len, device="cpu", dtype=restored.dtype)
            if end < total_samples:
                fade_out_len = min(overlap_samples, chunk_len)
                weight[:, :, -fade_out_len:] *= torch.linspace(1.0, 0.0, fade_out_len, device="cpu", dtype=restored.dtype)

        output[:, :, start:end] += restored * weight
        weight_sum[:, :, start:end] += weight

        if device.type == "cuda":
            del restored
            torch.cuda.empty_cache()

    weight_sum = torch.clamp(weight_sum, min=1e-8)
    return output / weight_sum


def _configure_cpu_threads(auto_cpu_threads, cpu_threads):
    """Configure torch CPU thread usage."""
    if cpu_threads is not None:
        threads = max(1, int(cpu_threads))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        print(f"[Apollo] CPU threads set to {threads}")
        return

    if not auto_cpu_threads:
        return

    cores = os.cpu_count() or 4
    threads = max(1, int(cores * 0.75))
    try:
        import psutil  # optional
        load = psutil.cpu_percent(interval=0.3)
        if load >= 70:
            threads = max(1, int(cores * 0.25))
        elif load >= 40:
            threads = max(1, int(cores * 0.50))
        print(f"[Apollo] CPU load {load:.1f}%, auto threads {threads}/{cores}")
    except Exception:
        print(f"[Apollo] CPU auto threads {threads}/{cores}")

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint, handling different formats"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False, mmap=True)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Strip 'audio_model.' prefix if present (from Lightning checkpoints)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('audio_model.'):
            new_state_dict[k.replace('audio_model.', '')] = v
        else:
            new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Note: Using strict=False for checkpoint loading")
        model.load_state_dict(new_state_dict, strict=False)
        
    return model


def get_model_config(model_name):
    """Get model configuration based on model name"""
    configs = {
        'apollo_official': {'feature_dim': 256, 'layer': 6},
        'apollo_lew_uni': {'feature_dim': 384, 'layer': 6},
        'apollo_lew_v2': {'feature_dim': 192, 'layer': 6},
        'apollo_edm_big': {'feature_dim': 256, 'layer': 6},
    }
    
    # Try to match model name
    model_lower = model_name.lower()
    if 'uni' in model_lower or 'universal' in model_lower:
        return configs['apollo_lew_uni']
    elif 'v2' in model_lower:
        return configs['apollo_lew_v2']
    elif 'big' in model_lower or 'edm' in model_lower:
        return configs['apollo_edm_big']
    else:
        return configs['apollo_official']


def restore_audio(
    input_path,
    output_path,
    model_path,
    config_path=None,
    feature_dim=None,
    layer=None,
    chunk_seconds=20.0,
    overlap_seconds=2.0,
    auto_chunk_seconds=30.0,
    cpu_threads=None,
    auto_cpu_threads=False,
):
    """
    Main restoration function
    
    Args:
        input_path: Path to input audio
        output_path: Path for output audio
        model_path: Path to model checkpoint
        config_path: Optional path to config YAML
        feature_dim: Model feature dimension (overrides config)
        layer: Number of layers (overrides config)
    """
    device = get_device()
    print(f"[Apollo] Using device: {device}")

    # Use smaller default chunks on CUDA to avoid OOM on 8GB GPUs
    if device.type == "cuda" and chunk_seconds == 20.0 and overlap_seconds == 2.0:
        chunk_seconds = 5.0
        overlap_seconds = 0.5

    if device.type == "cpu":
        _configure_cpu_threads(auto_cpu_threads, cpu_threads)
    
    # Get model config
    if config_path and os.path.exists(config_path):
        try:
            from omegaconf import OmegaConf
            conf = OmegaConf.load(config_path)
            if 'model' in conf:
                _layer = conf.model.get('layer', 6)
                _feature_dim = conf.model.get('feature_dim', 256)
                print(f"[Apollo] Config: layer={_layer}, feature_dim={_feature_dim}")
                if layer is None:
                    layer = _layer
                if feature_dim is None:
                    feature_dim = _feature_dim
        except ImportError:
            print("[Apollo] omegaconf not available, using defaults")
    
    # Auto-detect from model name if not specified
    if feature_dim is None or layer is None:
        model_name = os.path.basename(model_path)
        auto_config = get_model_config(model_name)
        if feature_dim is None:
            feature_dim = auto_config['feature_dim']
        if layer is None:
            layer = auto_config['layer']
    
    print(f"[Apollo] Model params: feature_dim={feature_dim}, layer={layer}")
    
    # Create model
    model = Apollo(sr=44100, win=20, feature_dim=feature_dim, layer=layer)
    
    # Load checkpoint
    print(f"[Apollo] Loading checkpoint: {model_path}")
    model = load_checkpoint(model, model_path, device)
    model.to(device)
    model.eval()
    
    # Check input exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Process
    print(f"[Apollo] Processing: {input_path}")
    audio = load_audio(input_path).to(device)

    duration_seconds = audio.shape[-1] / 44100.0
    use_chunking = chunk_seconds > 0 and (auto_chunk_seconds <= 0 or duration_seconds > auto_chunk_seconds)
    if use_chunking:
        print(
            f"[Apollo] Chunked inference: chunk={chunk_seconds:.2f}s overlap={overlap_seconds:.2f}s (duration={duration_seconds:.2f}s)"
        )
        restored = _chunk_restore(model, audio, 44100, chunk_seconds, overlap_seconds, device)
    else:
        with torch.inference_mode():
            restored = model(audio)
    
    # Save output
    save_audio(output_path, restored)
    print(f"[Apollo] Saved: {output_path}")
    
    return output_path


def main():
    """CLI entry point for Apollo"""
    parser = argparse.ArgumentParser(
        description="Apollo Audio Restoration - Convert lossy audio to higher quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  apollo-restore input.mp3 -o restored.wav -m apollo_lew_uni.ckpt
  apollo-restore input.wav --model_path /path/to/model.ckpt --feature_dim 384

Models:
  - apollo_official (feature_dim=256) - General restoration
  - apollo_lew_uni (feature_dim=384) - RECOMMENDED, best quality
  - apollo_lew_v2 (feature_dim=192) - Lightweight, vocal enhancement
  - apollo_edm_big (feature_dim=256) - EDM/Electronic music
        """
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", help="Output audio file (default: input_restored.wav)")
    parser.add_argument("-m", "--model_path", required=True, help="Path to model checkpoint (.ckpt or .bin)")
    parser.add_argument("-c", "--config_path", help="Path to config YAML (optional)")
    parser.add_argument("--feature_dim", type=int, help="Model feature dimension (auto-detected if not set)")
    parser.add_argument("--layer", type=int, help="Number of layers (auto-detected if not set)")
    parser.add_argument("--chunk_seconds", type=float, default=20.0, help="Chunk length in seconds (0 to disable)")
    parser.add_argument("--chunk_overlap", type=float, default=2.0, help="Chunk overlap in seconds")
    parser.add_argument(
        "--auto_chunk_seconds",
        type=float,
        default=30.0,
        help="Only chunk when audio duration exceeds this (0 to always chunk)",
    )
    parser.add_argument("--cpu_threads", type=int, help="Set torch CPU thread count")
    parser.add_argument("--auto_cpu_threads", action="store_true", help="Auto-tune CPU threads based on load")
    parser.add_argument("--output_dir", help="Output directory (default: current directory)")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = args.output_dir or os.path.dirname(args.input) or "."
        output_path = os.path.join(output_dir, f"{base}_restored.wav")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    try:
        restore_audio(
            input_path=args.input,
            output_path=output_path,
            model_path=args.model_path,
            config_path=args.config_path,
            feature_dim=args.feature_dim,
            layer=args.layer,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.chunk_overlap,
            auto_chunk_seconds=args.auto_chunk_seconds,
            cpu_threads=args.cpu_threads,
            auto_cpu_threads=args.auto_cpu_threads,
        )
        print(f"[Apollo] Restoration complete!")
    except Exception as e:
        print(f"[Apollo] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

