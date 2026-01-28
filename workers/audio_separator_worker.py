#!/usr/bin/env python3
"""
Audio Separator Thin Worker Script - Shared Runtime Version

This is a thin worker that uses the shared Python runtime.
It uses the audio-separator library and Apollo for audio restoration.
Implements the worker mode JSON protocol for Node.js integration.

Usage:
  runtime/python.exe workers/audio_separator_worker.py --worker

This script:
1. Uses the shared runtime (torch, numpy, audio-separator, etc.)
2. Implements the same JSON protocol as the fat executable
3. Supports both VR separation and Apollo restoration
"""

import os
import sys

# =============================================================================
# CRITICAL FIX for cx_Freeze + PyTorch 2.x compatibility
# =============================================================================
# PyTorch 2.x's torch.compiler.config module uses inspect.getsourcelines()
# during import, which fails in frozen executables because source code
# isn't available. This monkey-patch must run BEFORE any torch imports.
# =============================================================================
if getattr(sys, 'frozen', False):
    sys._MEIPASS = os.path.dirname(sys.executable)

    # Fix for PyTorch DLL loading on Windows frozen builds.
    # Torch may call os.add_dll_directory with relative paths (e.g. "bin"), which
    # raises WinError 87. Normalize to absolute and add common DLL dirs.
    if sys.platform == "win32":
        _exe_dir = os.path.dirname(sys.executable)
        _dll_dir_handles = []
        try:
            _orig_add_dll_directory = os.add_dll_directory

            class _NoopDLLDir:
                def close(self):
                    return None

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            def _safe_add_dll_directory(p):
                if p and not os.path.isabs(p):
                    p = os.path.join(_exe_dir, p)
                if not p or not os.path.isdir(p):
                    return _NoopDLLDir()
                return _orig_add_dll_directory(p)

            os.add_dll_directory = _safe_add_dll_directory  # type: ignore[attr-defined]
        except Exception:
            pass

        for rel in ("", "lib", os.path.join("lib", "torch", "lib"), os.path.join("lib", "torchaudio", "lib")):
            p = os.path.join(_exe_dir, rel)
            if os.path.isdir(p):
                try:
                    _dll_dir_handles.append(os.add_dll_directory(p))
                except Exception:
                    pass
    
    # Monkey-patch inspect to handle frozen modules gracefully
    import inspect
    _original_getsourcelines = inspect.getsourcelines
    _original_getsource = inspect.getsource
    _original_findsource = inspect.findsource
    
    def _safe_getsourcelines(obj):
        try:
            return _original_getsourcelines(obj)
        except OSError:
            return ([''], 0)
    
    def _safe_getsource(obj):
        try:
            return _original_getsource(obj)
        except OSError:
            return ''
    
    def _safe_findsource(obj):
        try:
            return _original_findsource(obj)
        except OSError:
            return ([''], 0)
    
    inspect.getsourcelines = _safe_getsourcelines
    inspect.getsource = _safe_getsource
    inspect.findsource = _safe_findsource

import argparse
import time
import json
import traceback

# ============================================================================
# PATH SETUP
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Parent of workers folder (dsu-win-cuda/) contains apollo/, models/, etc.
DIST_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Add distribution root to path (contains apollo/ folder)
if os.path.isdir(os.path.join(DIST_ROOT, 'apollo')):
    sys.path.insert(0, DIST_ROOT)

# Also check for audio-separator-cxfreeze in dev (for development)
APOLLO_DIR = os.path.join(PROJECT_ROOT, 'audio-separator-cxfreeze', 'apollo')
if os.path.isdir(APOLLO_DIR):
    sys.path.insert(0, os.path.dirname(APOLLO_DIR))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def send_json(data):
    """Send JSON response to stdout"""
    print(json.dumps(data), flush=True)


# ============================================================================
# WORKER MODE - SHUTDOWN HANDLING
# ============================================================================

import signal
import threading
import select

# Global shutdown flag
_shutdown_requested = threading.Event()

def _signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    _shutdown_requested.set()

def _stdin_monitor():
    """
    Monitor stdin for closure (parent process died).
    On Windows, stdin.read() blocks forever even after parent dies,
    so we use a thread that polls stdin.
    """
    try:
        if sys.platform == "win32":
            # Windows: poll stdin in a loop
            while not _shutdown_requested.is_set():
                try:
                    if sys.stdin.closed:
                        _shutdown_requested.set()
                        break
                    _shutdown_requested.wait(0.5)
                except Exception:
                    _shutdown_requested.set()
                    break
        else:
            # Unix: use select
            while not _shutdown_requested.is_set():
                try:
                    readable, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if readable and sys.stdin.closed:
                        _shutdown_requested.set()
                        break
                except Exception:
                    _shutdown_requested.set()
                    break
    except Exception:
        _shutdown_requested.set()


def _setup_shutdown_handlers():
    """Install signal handlers for graceful shutdown"""
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, _signal_handler)
    
    monitor = threading.Thread(target=_stdin_monitor, daemon=True)
    monitor.start()


# ============================================================================
# WORKER MODE
# ============================================================================

def worker_mode():
    """
    Persistent worker mode for Node.js integration.
    
    Handles both VR separation and Apollo restoration based on commands.
    
    Handles graceful shutdown on:
    - SIGTERM/SIGINT signals
    - stdin closure (parent process died)
    - {"cmd": "exit"} command
    
    Protocol:
        Input (one JSON per line):
            {"cmd": "separate", "input": "/path/to/audio.wav", "output_dir": "/output/", "model": "17_HP-Wind_Inst-UVR.pth", ...}
            {"cmd": "apollo", "input": "/path/to/audio.wav", "output": "/path/to/restored.wav", "model_path": "/models/apollo.ckpt", ...}
            {"cmd": "load_model", "model": "17_HP-Wind_Inst-UVR.pth"}
            {"cmd": "exit"}
        
        Output (one JSON per line):
            {"status": "ready"}
            {"status": "progress", "percent": 45}
            {"status": "done", "files": [...], "elapsed": 12.5}
            {"status": "error", "message": "..."}
    """
    
    # Install shutdown handlers
    _setup_shutdown_handlers()
    
    # Signal loading
    send_json({"status": "loading", "message": "Initializing audio-separator worker..."})
    
    # State
    separator = None
    current_model = None
    apollo_model_path = None
    apollo_model = None  # Cached Apollo model
    apollo_device = None  # Device Apollo model is on
    
    try:
        # Import heavy modules
        from audio_separator.separator import Separator
        
        # Initialize separator (but don't load model yet)
        separator = Separator(
            log_level=30,  # WARNING level
            output_format="WAV"
        )
        
        send_json({"status": "ready", "message": "Worker ready"})
        
    except Exception as e:
        send_json({"status": "error", "message": f"Failed to initialize: {str(e)}"})
        return 1
    
    # Main job loop - checks shutdown flag
    while not _shutdown_requested.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                # EOF - stdin closed (parent died)
                break
            
            line = line.strip()
            if not line:
                continue
            
            job = json.loads(line)
            cmd = job.get("cmd", "")
            
            if cmd == "exit":
                send_json({"status": "exiting"})
                break
            
            elif cmd == "ping":
                send_json({
                    "status": "pong",
                    "separator_model": current_model,
                    "apollo_model": apollo_model_path
                })
            
            elif cmd == "load_model":
                model_name = job.get("model")
                model_dir = job.get("model_file_dir")
                
                if not model_name:
                    send_json({"status": "error", "message": "No model specified"})
                    continue
                
                try:
                    send_json({"status": "loading_model", "model": model_name})
                    
                    if model_dir:
                        separator.model_file_dir = model_dir
                    
                    separator.load_model(model_name)
                    current_model = model_name
                    
                    send_json({"status": "model_loaded", "model": model_name})
                    
                except Exception as e:
                    send_json({"status": "error", "message": f"Failed to load model: {str(e)}"})
            
            elif cmd == "load_apollo":
                # Preload Apollo model for restoration
                model_path = job.get("model_path")
                config_path = job.get("config_path")
                feature_dim = job.get("feature_dim")
                layer = job.get("layer")
                
                if not model_path:
                    send_json({"status": "error", "message": "No model_path specified"})
                    continue
                
                try:
                    send_json({"status": "loading_model", "model": os.path.basename(model_path)})
                    
                    import torch
                    from apollo.apollo_separator import get_device, get_model_config, load_checkpoint
                    from apollo.look2hear.models.apollo import Apollo
                    
                    # Get device
                    device = get_device()
                    
                    # Get config from YAML if provided
                    if config_path and os.path.exists(config_path):
                        try:
                            from omegaconf import OmegaConf
                            conf = OmegaConf.load(config_path)
                            if 'model' in conf:
                                if feature_dim is None:
                                    feature_dim = conf.model.get('feature_dim', 384)
                                if layer is None:
                                    layer = conf.model.get('layer', 6)
                        except ImportError:
                            pass
                    
                    # Auto-detect from model name if not specified
                    if feature_dim is None or layer is None:
                        model_name = os.path.basename(model_path)
                        auto_config = get_model_config(model_name)
                        if feature_dim is None:
                            feature_dim = auto_config['feature_dim']
                        if layer is None:
                            layer = auto_config['layer']
                    
                    # Unload previous Apollo model
                    if apollo_model is not None:
                        del apollo_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Create and load Apollo model
                    apollo_model = Apollo(sr=44100, win=20, feature_dim=feature_dim, layer=layer)
                    apollo_model = load_checkpoint(apollo_model, model_path, device)
                    apollo_model.to(device)
                    apollo_model.eval()
                    apollo_device = device
                    apollo_model_path = model_path
                    
                    send_json({
                        "status": "model_loaded",
                        "model": os.path.basename(model_path),
                        "type": "apollo",
                        "feature_dim": feature_dim,
                        "layer": layer,
                        "device": str(device)
                    })
                    
                except Exception as e:
                    send_json({"status": "error", "message": f"Failed to load Apollo model: {str(e)}"})
            
            elif cmd == "separate":
                input_path = job.get("input")
                output_dir = job.get("output_dir", os.getcwd())
                model_name = job.get("model")
                model_dir = job.get("model_file_dir")
                
                # VR-specific settings
                vr_enable_tta = job.get("vr_enable_tta", False)
                vr_high_end_process = job.get("vr_high_end_process", False)
                vr_post_process = job.get("vr_post_process", False)
                single_stem = job.get("single_stem")
                
                if not input_path:
                    send_json({"status": "error", "message": "No input file specified"})
                    continue
                
                if not os.path.exists(input_path):
                    send_json({"status": "error", "message": f"Input file not found: {input_path}"})
                    continue
                
                # Load model if different or not loaded
                if model_name and model_name != current_model:
                    try:
                        send_json({"status": "loading_model", "model": model_name})
                        if model_dir:
                            separator.model_file_dir = model_dir
                        separator.load_model(model_name)
                        current_model = model_name
                    except Exception as e:
                        send_json({"status": "error", "message": f"Failed to load model: {str(e)}"})
                        continue
                
                if current_model is None:
                    send_json({"status": "error", "message": "No model loaded. Use load_model first."})
                    continue
                
                try:
                    send_json({"status": "separating", "input": os.path.basename(input_path)})
                    start_time = time.time()
                    
                    # Update output dir
                    separator.output_dir = output_dir
                    if separator.model_instance:
                        separator.model_instance.output_dir = output_dir
                    
                    # Apply VR-specific settings to model instance
                    if separator.model_instance:
                        if hasattr(separator.model_instance, 'enable_tta'):
                            separator.model_instance.enable_tta = vr_enable_tta
                        if hasattr(separator.model_instance, 'high_end_process'):
                            separator.model_instance.high_end_process = vr_high_end_process
                        if hasattr(separator.model_instance, 'post_process_threshold'):
                            separator.model_instance.post_process_threshold = 0.2 if vr_post_process else None
                    
                    # Run separation (with optional single stem)
                    if single_stem:
                        output_files = separator.separate(input_path, primary_stem=single_stem)
                    else:
                        output_files = separator.separate(input_path)
                    elapsed = time.time() - start_time
                    
                    send_json({
                        "status": "done",
                        "elapsed": round(elapsed, 2),
                        "files": output_files
                    })
                    
                except Exception as e:
                    send_json({
                        "status": "error",
                        "message": f"Separation failed: {str(e)}",
                        "traceback": traceback.format_exc()
                    })
            
            elif cmd == "apollo":
                # Apollo restoration
                input_path = job.get("input")
                output_path = job.get("output")
                model_path = job.get("model_path")
                config_path = job.get("config_path")
                feature_dim = job.get("feature_dim")
                layer = job.get("layer")
                chunk_seconds = job.get("chunk_seconds", 7.0)  # 7s chunks = good balance of speed and VRAM
                chunk_overlap = job.get("chunk_overlap", 0.5)  # 0.5s overlap for smooth transitions
                
                if not input_path:
                    send_json({"status": "error", "message": "No input file specified"})
                    continue
                
                if not output_path:
                    base = os.path.splitext(os.path.basename(input_path))[0]
                    output_path = f"{base}_restored.wav"
                
                if not model_path:
                    send_json({"status": "error", "message": "No model_path specified for Apollo"})
                    continue
                
                if not os.path.exists(input_path):
                    send_json({"status": "error", "message": f"Input file not found: {input_path}"})
                    continue
                
                if not os.path.exists(model_path):
                    send_json({"status": "error", "message": f"Model not found: {model_path}"})
                    continue
                
                try:
                    send_json({"status": "restoring", "input": os.path.basename(input_path)})
                    start_time = time.time()
                    
                    # Check if we can use cached Apollo model
                    use_cached = (apollo_model is not None and 
                                  apollo_model_path == model_path)
                    
                    if use_cached:
                        # Use cached model - much faster!
                        import torch
                        from apollo.apollo_separator import load_audio, save_audio, _chunk_restore
                        
                        # Load input audio
                        audio = load_audio(input_path)
                        audio = audio.to(apollo_device)
                        
                        # Process with chunking
                        total_samples = audio.shape[-1]
                        sr = 44100
                        duration_seconds = total_samples / sr
                        
                        if chunk_seconds > 0 and duration_seconds > chunk_seconds:
                            restored = _chunk_restore(apollo_model, audio, sr, chunk_seconds, chunk_overlap, apollo_device)
                        else:
                            with torch.inference_mode():
                                restored = apollo_model(audio)
                        
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                        
                        # Save output
                        save_audio(output_path, restored)
                    else:
                        # Load fresh (first time or different model)
                        from apollo.apollo_separator import restore_audio
                        
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                        
                        restore_audio(
                            input_path=input_path,
                            output_path=output_path,
                            model_path=model_path,
                            config_path=config_path,
                            feature_dim=feature_dim,
                            layer=layer,
                            chunk_seconds=chunk_seconds,
                            overlap_seconds=chunk_overlap
                        )
                        apollo_model_path = model_path
                    
                    elapsed = time.time() - start_time
                    
                    send_json({
                        "status": "done",
                        "elapsed": round(elapsed, 2),
                        "files": [output_path],
                        "cached": use_cached
                    })
                    
                except Exception as e:
                    send_json({
                        "status": "error",
                        "message": f"Apollo restoration failed: {str(e)}",
                        "traceback": traceback.format_exc()
                    })
            
            elif cmd == "get_status":
                send_json({
                    "status": "status",
                    "separator_model": current_model,
                    "apollo_model": apollo_model_path,
                    "ready": True
                })
            
            else:
                send_json({"status": "error", "message": f"Unknown command: {cmd}"})
        
        except json.JSONDecodeError as e:
            send_json({"status": "error", "message": f"Invalid JSON: {str(e)}"})
        except Exception as e:
            send_json({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
    
    # Determine shutdown reason
    if _shutdown_requested.is_set():
        send_json({"status": "shutdown", "reason": "signal_or_parent_died"})
    
    return 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Audio Separator Worker (Shared Runtime)')
    parser.add_argument('--worker', action='store_true', help='Run in worker mode')
    parser.add_argument('--apollo', action='store_true',
                        help='Run Apollo restoration CLI')

    args, remaining = parser.parse_known_args()

    if args.worker:
        sys.exit(worker_mode())
    elif args.apollo:
        sys.argv = [sys.argv[0]] + remaining
        from apollo.apollo_separator import main as apollo_main
        apollo_main()
        sys.exit(0)
    else:
        sys.argv = [sys.argv[0]] + remaining
        from audio_separator.utils.cli import main as separator_main
        separator_main()
        sys.exit(0)


if __name__ == '__main__':
    main()
