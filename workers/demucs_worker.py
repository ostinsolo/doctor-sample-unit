#!/usr/bin/env python3
"""
Demucs Audio Source Separation - DSU Worker
With Worker Mode for persistent process (Node.js integration)

Worker mode keeps the model loaded in memory between jobs,
eliminating startup overhead for batch processing.

Usage:
  Standard CLI: dsu-demucs -n htdemucs_ft song.wav
  Worker mode:  dsu-demucs --worker
"""

import os
import sys
import json
import time
import traceback

# =============================================================================
# CRITICAL FIX for PyTorch 2.x with cx_Freeze
# Must be BEFORE any torch imports
# =============================================================================
if getattr(sys, "frozen", False) and sys.platform == "win32":
    # In some frozen layouts, torch attempts `os.add_dll_directory('bin')` (relative),
    # which fails on Windows (WinError 87). Make add_dll_directory robust and add
    # the key DLL directories explicitly.
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
            # If the directory doesn't exist (e.g. torch requests "bin"),
            # treat it as a no-op rather than crashing the app.
            if not p or not os.path.isdir(p):
                return _NoopDLLDir()
            return _orig_add_dll_directory(p)

        os.add_dll_directory = _safe_add_dll_directory  # type: ignore[attr-defined]
    except Exception:
        # os.add_dll_directory may not exist on older Pythons; ignore.
        pass

    for rel in ("", "lib", os.path.join("lib", "torch", "lib"), os.path.join("lib", "torchaudio", "lib")):
        p = os.path.join(_exe_dir, rel)
        if os.path.isdir(p):
            try:
                _dll_dir_handles.append(os.add_dll_directory(p))  # keep handle alive
            except Exception:
                pass

import inspect

_original_getsourcelines = inspect.getsourcelines
_original_getsource = inspect.getsource
_original_findsource = inspect.findsource


def _configure_demucs_cache():
    """
    Configure Demucs for frozen Windows builds:
    - Prefer an existing torch hub cache (avoid downloads)
    - Force torch.hub to load Demucs checkpoints with weights_only=False
    
    Note: Audio I/O is handled directly via soundfile, not torchaudio.
    """
    # Prefer user's existing DSU torch cache if present.
    # Node can override by setting DSU_TORCH_HOME or TORCH_HOME in the environment.
    dsu_torch_home = os.environ.get("DSU_TORCH_HOME") or os.environ.get("TORCH_HOME")
    if not dsu_torch_home:
        # Default location (derived, not hardcoded to a specific username):
        #   %USERPROFILE%\Documents\DSU-VSTOPIA\ThirdPartyApps\torch_cache
        user_home = os.path.expanduser("~")
        dsu_torch_home = os.path.join(user_home, "Documents", "DSU-VSTOPIA", "ThirdPartyApps", "torch_cache")
    if dsu_torch_home and os.path.isdir(dsu_torch_home):
        os.environ.setdefault("TORCH_HOME", dsu_torch_home)

    # PyTorch builds that default to weights_only=True can break Demucs, because
    # Demucs checkpoints are not always "weights-only" safe-loadable.
    # Make Demucs robust by defaulting to weights_only=False when supported.
    try:
        import torch
        import inspect as _inspect

        _orig_lsdfu = getattr(torch.hub, "load_state_dict_from_url", None)
        if callable(_orig_lsdfu):
            sig = None
            try:
                sig = _inspect.signature(_orig_lsdfu)
            except Exception:
                sig = None

            if sig is not None and "weights_only" in sig.parameters:
                def _lsdfu_wrapped(url, *args, **kwargs):
                    kwargs.setdefault("weights_only", False)
                    return _orig_lsdfu(url, *args, **kwargs)

                torch.hub.load_state_dict_from_url = _lsdfu_wrapped  # type: ignore[assignment]
    except Exception:
        pass


def save_audio_soundfile(wav, path, sample_rate):
    """
    Save audio tensor to WAV file using soundfile (avoids torchcodec issues).
    
    Args:
        wav: Tensor of shape (channels, samples) or (samples,)
        path: Output file path
        sample_rate: Sample rate in Hz
    """
    import soundfile as sf
    
    # Move to CPU and convert to numpy
    data = wav.detach().cpu().numpy()
    
    # soundfile expects (samples, channels), we have (channels, samples)
    if data.ndim == 2:
        data = data.T  # (C, T) -> (T, C)
    
    sf.write(path, data, sample_rate)

def _patched_getsourcelines(obj):
    try:
        return _original_getsourcelines(obj)
    except (OSError, TypeError):
        return (["# Source not available in frozen executable\n"], 0)

def _patched_getsource(obj):
    try:
        return _original_getsource(obj)
    except (OSError, TypeError):
        return "# Source not available in frozen executable\n"

def _patched_findsource(obj):
    try:
        return _original_findsource(obj)
    except (OSError, TypeError):
        return (["# Source not available in frozen executable\n"], 0)

inspect.getsourcelines = _patched_getsourcelines
inspect.getsource = _patched_getsource
inspect.findsource = _patched_findsource

# Check for worker mode early (before heavy imports)
WORKER_MODE = '--worker' in sys.argv


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


def worker_mode():
    """
    Persistent worker mode for Node.js integration.
    
    Accepts JSON commands via stdin, returns JSON responses via stdout.
    Model stays loaded in memory between jobs.
    
    Handles graceful shutdown on:
    - SIGTERM/SIGINT signals
    - stdin closure (parent process died)
    - {"cmd": "exit"} command
    
    Protocol:
        Input (one JSON per line):
            {"cmd": "separate", "input": "/path/to/audio.wav", "output": "/output/", "model": "htdemucs_ft", ...}
            {"cmd": "load_model", "model": "htdemucs_ft"}
            {"cmd": "list_models"}
            {"cmd": "exit"}
        
        Output (one JSON per line):
            {"status": "ready"}
            {"status": "progress", "percent": 45}
            {"status": "done", "files": [...], "elapsed": 12.5}
            {"status": "error", "message": "..."}
    """
    
    # Install shutdown handlers
    _setup_shutdown_handlers()
    
    def send_json(data):
        """Send JSON response to stdout"""
        print(json.dumps(data), flush=True)
    
    send_json({"status": "loading", "message": "Initializing Demucs worker..."})
    
    # Import heavy modules now
    try:
        import torch
        import torchaudio
        import soundfile as sf
        import numpy as np
        from pathlib import Path

        _configure_demucs_cache()
        
        # Import demucs modules
        from demucs.pretrained import get_model, SOURCES
        from demucs.apply import apply_model
        
        # Determine device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        send_json({"status": "ready", "device": str(device)})
        
    except Exception as e:
        send_json({"status": "error", "message": f"Failed to initialize: {str(e)}"})
        return 1
    
    # State
    model = None
    current_model_name = None
    model_ref = [None]  # Mutable reference for cleanup
    
    # Available models
    AVAILABLE_MODELS = [
        'htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi',
        'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q'
    ]
    
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
                send_json({"status": "pong", "model_loaded": current_model_name})
            
            elif cmd == "list_models":
                send_json({"status": "models", "models": AVAILABLE_MODELS})
            
            elif cmd == "load_model":
                model_name = job.get("model", "htdemucs_ft")
                
                try:
                    send_json({"status": "loading_model", "model": model_name})
                    
                    # Unload previous model
                    if model is not None:
                        del model
                        model_ref[0] = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                    
                    model = get_model(model_name)
                    model.to(device)
                    model.eval()
                    model_ref[0] = model  # Keep reference for cleanup
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
                    try:
                        send_json({"status": "loading_model", "model": target_model})
                        
                        if model is not None:
                            del model
                            model_ref[0] = None
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        model = get_model(target_model)
                        model.to(device)
                        model.eval()
                        model_ref[0] = model  # Keep reference for cleanup
                        current_model_name = target_model
                        
                    except Exception as e:
                        send_json({"status": "error", "message": f"Failed to load model: {str(e)}"})
                        continue
                
                try:
                    send_json({"status": "separating", "input": os.path.basename(input_path)})
                    start_time = time.time()
                    
                    # Load audio using soundfile
                    audio_np, sr = sf.read(input_path)
                    # Convert to torch tensor in expected format (channels, samples)
                    if audio_np.ndim == 1:
                        audio_np = np.stack([audio_np, audio_np])  # Mono to stereo
                    else:
                        audio_np = audio_np.T  # (samples, channels) -> (channels, samples)
                    wav = torch.from_numpy(audio_np.astype(np.float32))
                    
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
                    
                    # Save stems using soundfile (avoids torchcodec issues)
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
                            save_audio_soundfile(sources[0, target_idx], str(target_path), model.samplerate)
                            output_files.append(f"{two_stems}.wav")
                            
                            # Combine and save "no_target"
                            no_target = sources[0].sum(dim=0) - sources[0, target_idx]
                            no_target_path = out_dir / f"no_{two_stems}.wav"
                            save_audio_soundfile(no_target, str(no_target_path), model.samplerate)
                            output_files.append(f"no_{two_stems}.wav")
                        else:
                            # All stems (target not found)
                            for i, stem in enumerate(stems):
                                stem_path = out_dir / f"{stem}.wav"
                                save_audio_soundfile(sources[0, i], str(stem_path), model.samplerate)
                                output_files.append(f"{stem}.wav")
                    else:
                        # All stems
                        for i, stem in enumerate(stems):
                            stem_path = out_dir / f"{stem}.wav"
                            save_audio_soundfile(sources[0, i], str(stem_path), model.samplerate)
                            output_files.append(f"{stem}.wav")
                    
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
    
    # Determine shutdown reason
    if _shutdown_requested.is_set():
        send_json({"status": "shutdown", "reason": "signal_or_parent_died"})
    
    # Cleanup GPU memory
    if model is not None:
        del model
        model_ref[0] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return 0


def main():
    """Main entry point - routes to worker mode or standard CLI"""
    if WORKER_MODE:
        # Remove --worker from args before processing
        sys.argv.remove('--worker')
        sys.exit(worker_mode())
    else:
        # Configure torch cache for demucs checkpoints
        _configure_demucs_cache()
        # Standard Demucs CLI
        from demucs.separate import main as demucs_main
        demucs_main()


if __name__ == '__main__':
    main()
