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

import argparse
import os
import sys
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
# WORKER MODE
# ============================================================================

def worker_mode():
    """
    Persistent worker mode for Node.js integration.
    
    Handles both VR separation and Apollo restoration based on commands.
    
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
    
    # Signal loading
    send_json({"status": "loading", "message": "Initializing audio-separator worker..."})
    
    # State
    separator = None
    current_model = None
    apollo_model_path = None
    
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
            
            elif cmd == "separate":
                input_path = job.get("input")
                output_dir = job.get("output_dir", os.getcwd())
                model_name = job.get("model")
                model_dir = job.get("model_file_dir")
                
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
                    
                    # Run separation
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
                chunk_seconds = job.get("chunk_seconds", 20.0)
                chunk_overlap = job.get("chunk_overlap", 2.0)
                
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
                    
                    # Import and run Apollo
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
                    
                    elapsed = time.time() - start_time
                    apollo_model_path = model_path
                    
                    send_json({
                        "status": "done",
                        "elapsed": round(elapsed, 2),
                        "files": [output_path]
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
    
    return 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Audio Separator Worker (Shared Runtime)')
    parser.add_argument('--worker', action='store_true', help='Run in worker mode')
    
    args = parser.parse_args()
    
    if args.worker:
        sys.exit(worker_mode())
    else:
        print("Usage: python audio_separator_worker.py --worker")
        sys.exit(1)


if __name__ == '__main__':
    main()
