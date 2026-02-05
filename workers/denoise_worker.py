#!/usr/bin/env python3
"""
Denoise DSU Worker - Envelope-Matched Noise Reduction

Lightweight worker using noise_reduction/denoise.py.
No heavy models; envelope-matched spectral subtraction only.
Uses JSON-over-stdin/stdout protocol for Node.js integration.

Usage:
  dsu-denoise --worker

Spawn on demand; exits when no longer needed.
"""

import os
import sys
import json
import time
import traceback
import argparse
import signal
import threading
import select
from contextlib import redirect_stdout
from io import StringIO

# =============================================================================
# PATH SETUP - Ensure noise_reduction is importable
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Parent of workers/ = project root (main/)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if getattr(sys, "frozen", False):
    # Frozen exe: exe dir is output root (dist/dsu/)
    EXE_DIR = os.path.dirname(sys.executable)
    if EXE_DIR not in sys.path:
        sys.path.insert(0, EXE_DIR)
else:
    # Development: project root contains noise_reduction/
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# UTILITY
# =============================================================================

def send_json(data):
    """Send JSON response to stdout"""
    print(json.dumps(data), flush=True)


# Global shutdown flag
_shutdown_requested = threading.Event()


def _signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    _shutdown_requested.set()


def _stdin_monitor():
    """
    Monitor stdin for closure (parent process died).
    On Windows, stdin.read() blocks forever after parent dies.
    """
    try:
        if sys.platform == "win32":
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


# =============================================================================
# WORKER MODE
# =============================================================================

def worker_mode():
    """
    Worker mode: JSON-over-stdin/stdout protocol.
    
    Commands:
        ping          -> {"status": "pong", "model_loaded": null}
        get_status    -> {"status": "status", "ready": true, "model_loaded": null}
        load_model    -> no-op (no model); {"status": "model_loaded", "model": "envelope_matched"}
        denoise       -> run denoise_audio(); {"status": "done", "elapsed": ..., "files": [...]}
        exit          -> {"status": "exiting"}
    """
    _setup_shutdown_handlers()
    
    send_json({"status": "loading", "message": "Initializing denoise worker..."})
    
    try:
        from noise_reduction.denoise import denoise_audio
    except ImportError as e:
        send_json({"status": "error", "message": f"Failed to import denoiser: {str(e)}"})
        return 1
    
    send_json({
        "status": "ready",
        "message": "Denoise worker ready (envelope-matched spectral subtraction)",
        "model_loaded": None,
        "device": "cpu",
    })
    
    while not _shutdown_requested.is_set():
        try:
            if sys.stdin.closed:
                break
            
            if sys.platform != "win32":
                readable, _, _ = select.select([sys.stdin], [], [], 0.5)
                if not readable:
                    continue
            
            line = sys.stdin.readline()
            if not line:
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
                send_json({"status": "pong", "model_loaded": None})
            
            elif cmd == "get_status":
                send_json({
                    "status": "status",
                    "model_loaded": None,
                    "device": "cpu",
                    "ready": True,
                })
            
            elif cmd == "load_model":
                # No-op: denoiser has no model to load
                send_json({
                    "status": "model_loaded",
                    "model": "envelope_matched",
                    "message": "Envelope-matched subtraction (no model)",
                })
            
            elif cmd == "denoise":
                input_path = job.get("input")
                noise_path = job.get("noise_profile") or job.get("noise_path")
                output_path = job.get("output")
                subtraction_factor = job.get("subtraction_factor", 0.9)
                release_time = job.get("release_time")
                attack_time = job.get("attack_time", 0.01)
                mode = (job.get("mode") or "default").lower()
                
                if not input_path:
                    send_json({"status": "error", "message": "No input file specified"})
                    continue
                if not noise_path:
                    send_json({"status": "error", "message": "No noise_profile specified"})
                    continue
                if not os.path.exists(input_path):
                    send_json({"status": "error", "message": f"Input file not found: {input_path}"})
                    continue
                if not os.path.exists(noise_path):
                    send_json({"status": "error", "message": f"Noise profile not found: {noise_path}"})
                    continue
                
                # Mode presets (match denoise.py CLI)
                if release_time is None:
                    if mode == "drums":
                        release_time = 0.3
                    elif mode == "slow":
                        release_time = 0.5
                    else:
                        release_time = 0.1
                
                try:
                    send_json({"status": "denoising", "input": os.path.basename(input_path)})
                    start_time = time.time()
                    
                    # Suppress denoise_audio's print() so we don't corrupt JSON stdout
                    with redirect_stdout(StringIO()):
                        out_path = denoise_audio(
                            input_path,
                            noise_path,
                            output_path=output_path,
                            subtraction_factor=float(subtraction_factor),
                            release_time=float(release_time),
                            attack_time=float(attack_time),
                        )
                    
                    elapsed = time.time() - start_time
                    send_json({
                        "status": "done",
                        "elapsed": round(elapsed, 2),
                        "files": [out_path],
                    })
                    
                except Exception as e:
                    send_json({
                        "status": "error",
                        "message": f"Denoise failed: {str(e)}",
                        "traceback": traceback.format_exc(),
                    })
            
            else:
                send_json({"status": "error", "message": f"Unknown command: {cmd}"})
        
        except json.JSONDecodeError as e:
            send_json({"status": "error", "message": f"Invalid JSON: {str(e)}"})
        except Exception as e:
            send_json({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
    
    if _shutdown_requested.is_set():
        send_json({"status": "shutdown", "reason": "signal_or_parent_died"})
    
    return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Denoise DSU Worker - Envelope-matched noise reduction for amplitude-correlated noise",
        epilog="""
Worker mode (JSON over stdin/stdout):
  dsu-denoise --worker

CLI mode:
  dsu-denoise <audio> <noise_profile> [output] [--drums|--slow]

Modes:
  default  (100ms release) - vocals, melodic
  --drums  (300ms release) - kicks, drums, percussive; prevents pumping
  --slow   (500ms release) - pads, sustained sounds

See docs/WORKER_SYSTEM.md and noise_reduction/README.md for details.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--worker", action="store_true", help="Run in worker mode (JSON protocol)")
    parser.add_argument("--drums", action="store_true", help="Drums mode: 300ms release for kicks/percussive")
    parser.add_argument("--slow", action="store_true", help="Slow mode: 500ms release for pads/sustained")
    args, remaining = parser.parse_known_args()
    
    if args.worker:
        sys.exit(worker_mode())
    
    # CLI fallback: dsu-denoise <audio> <noise_profile> [output] [--drums|--slow]
    from noise_reduction.denoise import denoise_audio
    args_list = [a for a in remaining if not a.startswith("--")]
    flags = [a for a in remaining if a.startswith("--")]
    if len(args_list) < 2:
        parser.print_help()
        sys.exit(1)
    audio_file = args_list[0]
    noise_file = args_list[1]
    output_file = args_list[2] if len(args_list) > 2 else None
    release_time = 0.5 if (args.slow or "--slow" in flags) else (0.3 if (args.drums or "--drums" in flags) else 0.1)
    denoise_audio(audio_file, noise_file, output_file, release_time=release_time)


if __name__ == "__main__":
    main()
