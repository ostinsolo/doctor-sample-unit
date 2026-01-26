#!/usr/bin/env python3
"""
BS-RoFormer Thin Worker Script - Shared Runtime Version

This is a thin worker that uses the shared Python runtime.
It imports from the BS-RoFormer models directory and implements
the worker mode JSON protocol for Node.js integration.

Usage:
  runtime/python.exe workers/bsroformer_worker.py --worker --models-dir /path/to/models

This script:
1. Uses the shared runtime (torch, numpy, etc.)
2. Imports model definitions from the bs-roformer-freeze-repo
3. Implements the same JSON protocol as the fat executable
"""

import argparse
import os
import sys
import time
import json
import traceback

# ============================================================================
# PATH SETUP - Find the bs-roformer models directory
# ============================================================================

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Add bs-roformer-freeze-repo to path for model imports
BSROFORMER_DIR = os.path.join(PROJECT_ROOT, 'dev', 'bs-roformer-freeze-repo')
BSROFORMER_FROZEN_DIR = os.path.join(BSROFORMER_DIR, 'frozen')

if os.path.isdir(BSROFORMER_DIR):
    sys.path.insert(0, BSROFORMER_DIR)
if os.path.isdir(BSROFORMER_FROZEN_DIR):
    sys.path.insert(0, BSROFORMER_FROZEN_DIR)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_physical_cores():
    """Get number of physical CPU cores"""
    import platform
    import subprocess
    
    system = platform.system()
    try:
        if system == 'Windows':
            result = subprocess.run(['wmic', 'cpu', 'get', 'NumberOfCores'], 
                                    capture_output=True, text=True)
            lines = [l.strip() for l in result.stdout.split('\n') if l.strip().isdigit()]
            return sum(int(l) for l in lines)
    except:
        pass
    return max(1, os.cpu_count() // 2)


def send_json(data):
    """Send JSON response to stdout"""
    print(json.dumps(data), flush=True)


# ============================================================================
# MODEL LOADING (imported from bs-roformer-freeze-repo)
# ============================================================================

# These will be imported after we verify paths
load_model = None
separate = None
load_models_registry = None
DEFAULT_MODELS = None


def init_model_functions(models_dir=None):
    """Initialize model functions by importing from bs-roformer-freeze-repo"""
    global load_model, separate, load_models_registry, DEFAULT_MODELS
    
    try:
        # Try importing from the frozen directory's main module
        # We need to extract key functions from main.py
        
        # Check for models.json
        if models_dir:
            models_json = os.path.join(models_dir, 'models.json')
        else:
            models_json = os.path.join(BSROFORMER_FROZEN_DIR, 'models.json')
        
        if os.path.exists(models_json):
            with open(models_json, 'r') as f:
                DEFAULT_MODELS = json.load(f)
        else:
            DEFAULT_MODELS = {}
        
        return True
        
    except Exception as e:
        send_json({"status": "error", "message": f"Failed to init model functions: {str(e)}"})
        return False


def load_models_registry(models_dir=None):
    """Load models from registry"""
    if models_dir:
        models_json = os.path.join(models_dir, 'models.json')
    else:
        models_json = os.path.join(BSROFORMER_FROZEN_DIR, 'models.json')
    
    if os.path.exists(models_json):
        with open(models_json, 'r') as f:
            return json.load(f)
    return DEFAULT_MODELS or {}


def load_model_impl(model_name, models_dir=None):
    """Load a model by name"""
    import torch
    import yaml
    from ml_collections import ConfigDict
    
    models = load_models_registry(models_dir)
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model_info = models[model_name]
    model_type = model_info.get("type", "bs_roformer")
    
    # Resolve paths - try models_dir first, then BSROFORMER dirs
    if models_dir:
        config_path = os.path.join(models_dir, model_info["config"])
        checkpoint_path = os.path.join(models_dir, model_info["checkpoint"])
    else:
        config_path = os.path.join(BSROFORMER_DIR, model_info["config"])
        checkpoint_path = os.path.join(BSROFORMER_DIR, model_info["checkpoint"])
    
    # Try alternative config path if not found
    if not os.path.exists(config_path):
        alt_config = os.path.join(BSROFORMER_DIR, model_info["config"])
        if os.path.exists(alt_config):
            config_path = alt_config
    
    # Load config using ConfigDict (same as original code)
    # This handles list/tuple types correctly for beartype
    with open(config_path, 'r') as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    
    # Import appropriate model class
    if model_type == "bs_roformer":
        from models.bs_roformer import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == "mel_band_roformer":
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == "scnet":
        from models.scnet import SCNet
        model = SCNet(**dict(config.model))
    elif model_type == "mdx23c":
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**dict(config.model))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights (weights_only=False needed for PyTorch 2.6+ checkpoints)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'state' in state_dict:
            state_dict = state_dict['state']
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    return model, config, model_info


def separate_impl(model, config, input_path, output_dir, model_info, device='cpu',
                  overlap=None, batch_size=None, use_tta=False, output_format='wav',
                  pcm_type='FLOAT', extract_instrumental=False, selected_stems=None,
                  two_stems=None):
    """Run separation with the loaded model"""
    import torch
    import numpy as np
    import soundfile as sf
    import librosa
    
    start_time = time.time()
    
    # Load audio - use getattr for ConfigDict compatibility
    sample_rate = getattr(config.audio, 'samplerate', None) or getattr(config.audio, 'sample_rate', 44100)
    audio, sr = librosa.load(input_path, sr=sample_rate, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    
    # Move to device
    torch_device = torch.device(device)
    audio_tensor = audio_tensor.to(torch_device)
    model = model.to(torch_device)
    
    # Run separation
    with torch.no_grad():
        if use_tta:
            # Time-domain TTA
            sources = model(audio_tensor)
            sources_flipped = model(audio_tensor.flip(-1)).flip(-1)
            sources = (sources + sources_flipped) / 2
        else:
            sources = model(audio_tensor)
    
    # Get stem names
    stems = model_info.get("stems", ["vocals", "accompaniment"])
    
    # Create output directory
    basename = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(output_dir, basename)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save stems
    sources_np = sources.squeeze(0).cpu().numpy()
    
    for i, stem_name in enumerate(stems):
        if selected_stems and stem_name not in selected_stems:
            continue
        
        if i < sources_np.shape[0]:
            stem_audio = sources_np[i]
            out_path = os.path.join(out_dir, f"{stem_name}.{output_format}")
            
            # Handle mono/stereo
            if stem_audio.ndim == 1:
                stem_audio = np.stack([stem_audio, stem_audio])
            
            sf.write(out_path, stem_audio.T, sr)
    
    # Extract instrumental if requested
    if extract_instrumental and "vocals" in stems:
        vocal_idx = stems.index("vocals")
        instrumental = audio - sources_np[vocal_idx]
        out_path = os.path.join(out_dir, f"instrumental.{output_format}")
        sf.write(out_path, instrumental.T, sr)
    
    return time.time() - start_time


# ============================================================================
# WORKER MODE
# ============================================================================

def worker_mode(models_dir=None, device="cpu"):
    """
    Persistent worker mode for Node.js integration.
    
    Accepts JSON commands via stdin, returns JSON responses via stdout.
    """
    
    # Signal loading
    send_json({"status": "loading", "message": "Initializing worker..."})
    
    # Import heavy modules
    try:
        import torch
        import numpy as np
        import soundfile as sf
        import librosa
        
        # Set up device
        if device == "cuda" and torch.cuda.is_available():
            torch_device = torch.device("cuda")
        elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
        
        # Threading optimization
        num_threads = get_physical_cores()
        torch.set_num_threads(num_threads)
        
        send_json({"status": "ready", "device": str(torch_device), "threads": num_threads})
        
    except Exception as e:
        send_json({"status": "error", "message": f"Failed to initialize: {str(e)}"})
        return 1
    
    # State
    model = None
    config = None
    model_info = None
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
                models = load_models_registry(models_dir)
                send_json({"status": "models", "models": list(models.keys())})
            
            elif cmd == "load_model":
                model_name = job.get("model")
                if not model_name:
                    send_json({"status": "error", "message": "No model specified"})
                    continue
                
                try:
                    send_json({"status": "loading_model", "model": model_name})
                    
                    # Unload previous model
                    if model is not None:
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    model, config, model_info = load_model_impl(model_name, models_dir)
                    model = model.to(torch_device)
                    current_model_name = model_name
                    
                    send_json({
                        "status": "model_loaded",
                        "model": model_name,
                        "stems": model_info.get("stems", [])
                    })
                    
                except Exception as e:
                    send_json({"status": "error", "message": f"Failed to load model: {str(e)}"})
            
            elif cmd == "separate":
                input_path = job.get("input")
                output_dir = job.get("output", "output")
                model_name = job.get("model")
                
                if not input_path:
                    send_json({"status": "error", "message": "No input file specified"})
                    continue
                
                if not os.path.exists(input_path):
                    send_json({"status": "error", "message": f"Input file not found: {input_path}"})
                    continue
                
                # Load model if needed
                if model is None or (model_name and model_name != current_model_name):
                    target_model = model_name or "bsrofo_sw"
                    try:
                        send_json({"status": "loading_model", "model": target_model})
                        
                        if model is not None:
                            del model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        model, config, model_info = load_model_impl(target_model, models_dir)
                        model = model.to(torch_device)
                        current_model_name = target_model
                        
                    except Exception as e:
                        send_json({"status": "error", "message": f"Failed to load model: {str(e)}"})
                        continue
                
                try:
                    send_json({"status": "separating", "input": os.path.basename(input_path)})
                    
                    elapsed = separate_impl(
                        model, config, input_path, output_dir, model_info,
                        device=str(torch_device),
                        use_tta=job.get("use_tta", False),
                        output_format=job.get("format", "wav"),
                        extract_instrumental=job.get("extract_instrumental", False),
                        selected_stems=job.get("stems")
                    )
                    
                    # Find output files
                    basename = os.path.splitext(os.path.basename(input_path))[0]
                    out_subdir = os.path.join(output_dir, basename)
                    output_files = []
                    if os.path.isdir(out_subdir):
                        output_files = [f for f in os.listdir(out_subdir) 
                                       if f.endswith(('.wav', '.flac'))]
                    
                    send_json({
                        "status": "done",
                        "elapsed": round(elapsed, 2),
                        "output_dir": out_subdir,
                        "files": output_files,
                        "stems": model_info.get("stems", [])
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
                    "device": str(torch_device),
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
    parser = argparse.ArgumentParser(description='BS-RoFormer Worker (Shared Runtime)')
    parser.add_argument('--worker', action='store_true', help='Run in worker mode')
    parser.add_argument('--models-dir', type=str, default=None, help='Models directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    if args.worker:
        sys.exit(worker_mode(args.models_dir, args.device))
    else:
        print("Usage: python bsroformer_worker.py --worker [--models-dir /path] [--device cuda|cpu]")
        sys.exit(1)


if __name__ == '__main__':
    main()
