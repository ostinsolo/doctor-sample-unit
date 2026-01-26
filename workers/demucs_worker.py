#!/usr/bin/env python3
"""
Demucs Thin Worker Script - Shared Runtime Version

This is a thin worker that uses the shared Python runtime.
Implements the worker mode JSON protocol for Node.js integration.

Usage:
  runtime/python.exe workers/demucs_worker.py --worker

Available models:
  - htdemucs: Hybrid Transformer Demucs (default)
  - htdemucs_ft: Fine-tuned version (better quality)
  - htdemucs_6s: 6-stem version (drums, bass, other, vocals, guitar, piano)
  - hdemucs_mmi: Multi-instrument trained
  - mdx, mdx_extra, mdx_q, mdx_extra_q: MDX variants
"""

import os
import sys
import json
import time
import argparse
import traceback


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def send_json(data):
    """Send JSON response to stdout"""
    print(json.dumps(data), flush=True)


def get_physical_cores():
    """Get number of physical CPU cores"""
    try:
        import multiprocessing
        return max(1, multiprocessing.cpu_count() // 2)
    except:
        return 4


# ============================================================================
# AVAILABLE MODELS
# ============================================================================

AVAILABLE_MODELS = [
    'htdemucs',       # Hybrid Transformer Demucs (default, 4 stems)
    'htdemucs_ft',    # Fine-tuned version (better quality)
    'htdemucs_6s',    # 6-stem version
    'hdemucs_mmi',    # Multi-instrument trained
    'mdx',            # MDX architecture
    'mdx_extra',      # MDX with extra training
    'mdx_q',          # MDX quantized
    'mdx_extra_q'     # MDX extra quantized
]


# ============================================================================
# WORKER MODE
# ============================================================================

def worker_mode():
    """
    Persistent worker mode for Node.js integration.
    
    Protocol:
        Input (one JSON per line):
            {"cmd": "separate", "input": "/path/audio.wav", "output": "/output/", "model": "htdemucs_ft", ...}
            {"cmd": "load_model", "model": "htdemucs_ft"}
            {"cmd": "list_models"}
            {"cmd": "ping"}
            {"cmd": "exit"}
        
        Output (one JSON per line):
            {"status": "ready", "device": "cuda"}
            {"status": "done", "elapsed": 12.5, "files": [...]}
            {"status": "error", "message": "..."}
    """
    
    send_json({"status": "loading", "message": "Initializing Demucs worker..."})
    
    # Import heavy modules
    try:
        import torch
        import torchaudio
        from pathlib import Path
        
        # Import demucs modules
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import soundfile as sf
        import numpy as np
        
        # Custom save function (avoid torchcodec dependency)
        def save_audio_sf(tensor, path, samplerate):
            """Save audio tensor using soundfile"""
            # Convert from torch tensor to numpy [channels, samples] -> [samples, channels]
            audio = tensor.cpu().numpy().T
            sf.write(path, audio, samplerate)
        
        # Determine device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        # Threading optimization
        num_threads = get_physical_cores()
        torch.set_num_threads(num_threads)
        
        send_json({"status": "ready", "device": str(device), "threads": num_threads})
        
    except Exception as e:
        send_json({"status": "error", "message": f"Failed to initialize: {str(e)}"})
        return 1
    
    # State
    model = None
    current_model_name = None
    
    # Main job loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            job = json.loads(line)
            cmd = job.get("cmd", "")
            
            if cmd == "exit":
                send_json({"status": "exiting"})
                break
            
            elif cmd == "ping":
                send_json({"status": "pong", "model_loaded": current_model_name})
            
            elif cmd == "list_models":
                send_json({"status": "models", "models": AVAILABLE_MODELS})
            
            elif cmd == "load_model":
                model_name = job.get("model", "htdemucs_ft")
                
                if model_name not in AVAILABLE_MODELS:
                    send_json({"status": "error", "message": f"Unknown model: {model_name}. Available: {AVAILABLE_MODELS}"})
                    continue
                
                try:
                    send_json({"status": "loading_model", "model": model_name})
                    
                    # Unload previous model
                    if model is not None:
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                    
                    model = get_model(model_name)
                    model.to(device)
                    model.eval()
                    current_model_name = model_name
                    
                    # Get stems for this model
                    stems = list(model.sources)
                    
                    send_json({
                        "status": "model_loaded",
                        "model": model_name,
                        "stems": stems
                    })
                    
                except Exception as e:
                    send_json({"status": "error", "message": f"Failed to load model: {str(e)}"})
            
            elif cmd == "separate":
                input_path = job.get("input")
                output_dir = job.get("output", "output")
                model_name = job.get("model")
                two_stems = job.get("two_stems")  # e.g., "vocals" for vocals + no_vocals
                shifts = job.get("shifts", 1)
                overlap = job.get("overlap", 0.25)
                
                if not input_path:
                    send_json({"status": "error", "message": "No input file specified"})
                    continue
                
                if not os.path.exists(input_path):
                    send_json({"status": "error", "message": f"Input file not found: {input_path}"})
                    continue
                
                # Load model if different or not loaded
                if model is None or (model_name and model_name != current_model_name):
                    target_model = model_name or "htdemucs_ft"
                    
                    if target_model not in AVAILABLE_MODELS:
                        send_json({"status": "error", "message": f"Unknown model: {target_model}"})
                        continue
                    
                    try:
                        send_json({"status": "loading_model", "model": target_model})
                        
                        if model is not None:
                            del model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        model = get_model(target_model)
                        model.to(device)
                        model.eval()
                        current_model_name = target_model
                        
                    except Exception as e:
                        send_json({"status": "error", "message": f"Failed to load model: {str(e)}"})
                        continue
                
                try:
                    send_json({"status": "separating", "input": os.path.basename(input_path)})
                    start_time = time.time()
                    
                    # Load audio using soundfile (more compatible than torchaudio)
                    audio_data, sr = sf.read(input_path, dtype='float32')
                    
                    # Convert to torch tensor [channels, samples]
                    if len(audio_data.shape) == 1:
                        # Mono - duplicate to stereo
                        wav = torch.from_numpy(np.stack([audio_data, audio_data]))
                    else:
                        # Stereo or multi-channel - transpose to [channels, samples]
                        wav = torch.from_numpy(audio_data.T)
                    
                    # Resample if needed
                    if sr != model.samplerate:
                        wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
                    
                    # Add batch dimension
                    wav = wav.unsqueeze(0).to(device)
                    
                    # Apply model
                    with torch.no_grad():
                        sources = apply_model(
                            model, wav,
                            shifts=shifts,
                            overlap=overlap,
                            progress=False
                        )
                    
                    # Create output directory
                    input_basename = os.path.splitext(os.path.basename(input_path))[0]
                    out_dir = Path(output_dir) / current_model_name / input_basename
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save stems
                    output_files = []
                    stems = list(model.sources)
                    
                    if two_stems:
                        # Two-stems mode: output target + no_target
                        target_idx = None
                        for i, stem in enumerate(stems):
                            if stem.lower() == two_stems.lower():
                                target_idx = i
                                break
                        
                        if target_idx is not None:
                            # Save target stem
                            target_path = out_dir / f"{two_stems}.wav"
                            save_audio_sf(sources[0, target_idx], str(target_path), model.samplerate)
                            output_files.append(str(target_path))
                            
                            # Combine and save "no_target"
                            no_target = sources[0].sum(dim=0) - sources[0, target_idx]
                            no_target_path = out_dir / f"no_{two_stems}.wav"
                            save_audio_sf(no_target, str(no_target_path), model.samplerate)
                            output_files.append(str(no_target_path))
                        else:
                            # Fallback to all stems
                            for i, stem in enumerate(stems):
                                stem_path = out_dir / f"{stem}.wav"
                                save_audio_sf(sources[0, i], str(stem_path), model.samplerate)
                                output_files.append(str(stem_path))
                    else:
                        # All stems
                        for i, stem in enumerate(stems):
                            stem_path = out_dir / f"{stem}.wav"
                            save_audio_sf(sources[0, i], str(stem_path), model.samplerate)
                            output_files.append(str(stem_path))
                    
                    elapsed = time.time() - start_time
                    
                    send_json({
                        "status": "done",
                        "elapsed": round(elapsed, 2),
                        "output_dir": str(out_dir),
                        "files": output_files,
                        "stems": stems
                    })
                    
                except Exception as e:
                    send_json({
                        "status": "error",
                        "message": f"Separation failed: {str(e)}",
                        "traceback": traceback.format_exc()
                    })
            
            elif cmd == "get_status":
                send_json({
                    "status": "status",
                    "model_loaded": current_model_name,
                    "device": str(device),
                    "ready": True
                })
            
            else:
                send_json({"status": "error", "message": f"Unknown command: {cmd}"})
        
        except json.JSONDecodeError as e:
            send_json({"status": "error", "message": f"Invalid JSON: {str(e)}"})
        except Exception as e:
            send_json({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
    
    # Cleanup
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Demucs Worker (Shared Runtime)')
    parser.add_argument('--worker', action='store_true', help='Run in worker mode')
    
    args = parser.parse_args()
    
    if args.worker:
        sys.exit(worker_mode())
    else:
        print("Usage: python demucs_worker.py --worker")
        print("\nAvailable models:")
        for m in AVAILABLE_MODELS:
            print(f"  - {m}")
        sys.exit(1)


if __name__ == '__main__':
    main()
