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

        # Patch torch.hub.load_state_dict_from_url
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
        
        # Patch torch.load to default to weights_only=False (PyTorch 2.6+ changed default to True)
        # This is critical for loading custom Demucs models like inaki
        _orig_torch_load = torch.load
        try:
            sig = _inspect.signature(_orig_torch_load)
            if "weights_only" in sig.parameters:
                def _torch_load_wrapped(f, *args, **kwargs):
                    kwargs.setdefault("weights_only", False)
                    return _orig_torch_load(f, *args, **kwargs)
                
                torch.load = _torch_load_wrapped  # type: ignore[assignment]
        except Exception:
            pass
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


def _patch_demucs_save_audio_to_use_soundfile():
    """
    Replace demucs.audio.save_audio with a soundfile-based implementation.
    Avoids torchaudio.save -> torchcodec on Windows/Mac when using CLI (e.g.
    dsu-demucs -n htdemucs file.wav -o out). Worker mode already uses
    save_audio_soundfile; this patch fixes the CLI path.
    """
    import demucs.audio as _da
    from pathlib import Path
    import soundfile as sf

    _prevent_clip = _da.prevent_clip
    _encode_mp3 = _da.encode_mp3

    def _save_audio(
        wav,
        path,
        samplerate,
        bitrate=320,
        clip="rescale",
        bits_per_sample=16,
        as_float=False,
        preset=2,
    ):
        wav = _prevent_clip(wav, mode=clip)
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".mp3":
            _encode_mp3(wav, path, samplerate, bitrate, preset, verbose=True)
            return
        if suffix == ".wav":
            if as_float:
                subtype = "FLOAT"
            else:
                subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(
                    bits_per_sample, "PCM_16"
                )
        elif suffix == ".flac":
            subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(
                bits_per_sample, "PCM_16"
            )
        else:
            raise ValueError(f"Invalid suffix for path: {suffix}")

        data = wav.detach().cpu().numpy()
        if data.ndim == 2:
            data = data.T
        sf.write(str(path), data, samplerate, subtype=subtype)

    _da.save_audio = _save_audio
    # demucs.separate does "from .audio import save_audio" and uses that ref
    import demucs.separate as _ds
    _ds.save_audio = _save_audio

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


# ============================================================================
# SAFE TORCH LOAD WITH MMAP SUPPORT
# ============================================================================

# Global flag for mmap support (checked once at startup)
_MMAP_SUPPORTED = None
_MMAP_PREFERRED = True  # Can be set to False for HDD storage

def check_mmap_support():
    """Check if torch.load supports mmap parameter (PyTorch 2.1+)"""
    global _MMAP_SUPPORTED
    if _MMAP_SUPPORTED is not None:
        return _MMAP_SUPPORTED
    
    try:
        import torch
        import inspect as _inspect
        sig = _inspect.signature(torch.load)
        _MMAP_SUPPORTED = 'mmap' in sig.parameters
    except Exception:
        _MMAP_SUPPORTED = False
    
    return _MMAP_SUPPORTED


def safe_torch_load(checkpoint_path, map_location='cpu', weights_only=False, use_mmap=None):
    """
    Safely load a PyTorch checkpoint with mmap support when available.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        map_location: Device to map tensors to (default: 'cpu')
        weights_only: If True, only load weights (safer but may fail on some checkpoints)
        use_mmap: Override mmap behavior:
                  - None: Auto-detect (use mmap if supported and preferred)
                  - True: Force mmap (will fallback if not supported)
                  - False: Disable mmap (use for HDD storage)
    
    Returns:
        Loaded checkpoint dictionary
    """
    import torch
    
    mmap_available = check_mmap_support()
    
    # Determine if we should use mmap
    if use_mmap is False:
        should_mmap = False
    elif use_mmap is True:
        should_mmap = mmap_available
    else:
        should_mmap = mmap_available and _MMAP_PREFERRED
    
    try:
        if should_mmap:
            return torch.load(checkpoint_path, map_location=map_location, 
                            weights_only=weights_only, mmap=True)
        else:
            return torch.load(checkpoint_path, map_location=map_location, 
                            weights_only=weights_only)
    except TypeError as e:
        if 'mmap' in str(e) and should_mmap:
            global _MMAP_SUPPORTED
            _MMAP_SUPPORTED = False
            return torch.load(checkpoint_path, map_location=map_location, 
                            weights_only=weights_only)
        raise
    except Exception as e:
        if should_mmap:
            try:
                return torch.load(checkpoint_path, map_location=map_location, 
                                weights_only=weights_only)
            except:
                pass
        raise


def set_mmap_preferred(preferred):
    """Set whether mmap should be preferred (False for HDD storage)"""
    global _MMAP_PREFERRED
    _MMAP_PREFERRED = preferred


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

        Custom models (inaki, filosax) require "repo": "/path/to/demucs/models" (folder with
        .th/.yaml). Same idea as Demucs CLI --repo. Use with model "inaki" or "filosax".
        
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
    current_repo = None  # Track current repo for custom models
    model_ref = [None]  # Mutable reference for cleanup
    
    # Model cache for keeping multiple models in GPU memory
    # This enables fast switching between models (e.g., htdemucs_ft + inaki)
    MAX_CACHED_MODELS = 2  # Limit to prevent OOM (reduced from 3 for better performance)
    model_cache = {}  # {model_identifier: (model, repo_path, stems)}
    
    def get_cache_key(model_name, repo_path=None):
        """Create unique cache key for model+repo combination"""
        return f"{model_name}|{repo_path or 'default'}"
    
    def get_cached_model(model_name, repo_path=None):
        """Get a model from cache if available"""
        key = get_cache_key(model_name, repo_path)
        return model_cache.get(key)
    
    def add_to_cache(model_name, repo_path, model_obj, stems):
        """Add model to cache, evict oldest if full"""
        nonlocal model_cache
        key = get_cache_key(model_name, repo_path)
        if len(model_cache) >= MAX_CACHED_MODELS:
            # Evict oldest (first inserted)
            oldest_key = next(iter(model_cache))
            model_cache.pop(oldest_key)  # Remove from cache, GC handles cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            send_json({"status": "cache_evicted", "model": oldest_key.split('|')[0]})
        model_cache[key] = (model_obj, repo_path, stems)
    
    def clear_cache():
        """Clear all cached models"""
        nonlocal model_cache
        # Simply clear the dict - Python GC will handle model cleanup
        model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Available pretrained models
    AVAILABLE_MODELS = [
        'htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi',
        'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q'
    ]
    
    # Custom community models (require --repo)
    CUSTOM_MODELS = {
        'filosax': 'filosax_demucs_v3_14.22_SDR',
        'inaki': 'inaki'
    }
    
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
                repo_path = job.get("repo")  # Optional: path to custom model repo
                
                try:
                    # Check cache first
                    cached = get_cached_model(model_name, repo_path)
                    if cached:
                        model, current_repo, stems = cached
                        model_ref[0] = model
                        current_model_name = model_name
                        send_json({
                            "status": "model_loaded",
                            "model": model_name,
                            "stems": stems,
                            "repo": repo_path,
                            "cached": True
                        })
                        continue
                    
                    send_json({"status": "loading_model", "model": model_name})
                    
                    # Determine actual model name for custom models
                    actual_model_name = CUSTOM_MODELS.get(model_name, model_name)
                    
                    # Load model - with or without custom repo
                    if repo_path:
                        # Custom model from local repo (e.g., filosax, inaki)
                        from pathlib import Path
                        repo = Path(repo_path)
                        if not repo.is_dir():
                            raise ValueError(f"Repo path does not exist: {repo_path}")
                        model = get_model(actual_model_name, repo=repo)
                        current_repo = repo_path
                    else:
                        # Standard pretrained model
                        model = get_model(actual_model_name)
                        current_repo = None
                    
                    model.to(device)
                    model.eval()
                    model_ref[0] = model  # Keep reference for cleanup
                    current_model_name = model_name
                    
                    # Get stems for this model
                    stems = list(model.sources)
                    
                    # Add to cache
                    add_to_cache(model_name, repo_path, model, stems)
                    
                    send_json({
                        "status": "model_loaded",
                        "model": model_name,
                        "stems": stems,
                        "repo": repo_path,
                        "cached": False
                    })
                    
                except Exception as e:
                    send_json({"status": "error", "message": f"Failed to load model: {str(e)}"})
            
            elif cmd == "separate":
                input_path = job.get("input")
                output_dir = job.get("output", "output")
                model_name = job.get("model")
                repo_path = job.get("repo")  # Optional: path to custom model repo
                two_stems = job.get("two_stems")  # e.g., "vocals" for vocals + no_vocals
                shifts = job.get("shifts", 1)
                overlap = job.get("overlap", 0.25)
                
                if not input_path:
                    send_json({"status": "error", "message": "No input file specified"})
                    continue
                
                if not os.path.exists(input_path):
                    send_json({"status": "error", "message": f"Input file not found: {input_path}"})
                    continue
                
                # Load model if different or not loaded (including repo change)
                needs_reload = (
                    model is None or 
                    (model_name and model_name != current_model_name) or
                    (repo_path and repo_path != current_repo)
                )
                
                if needs_reload:
                    target_model = model_name or "htdemucs_ft"
                    
                    try:
                        # Check cache first
                        cached = get_cached_model(target_model, repo_path)
                        if cached:
                            model, current_repo, _ = cached
                            model_ref[0] = model
                            current_model_name = target_model
                        else:
                            send_json({"status": "loading_model", "model": target_model})
                            
                            # Determine actual model name for custom models
                            actual_model_name = CUSTOM_MODELS.get(target_model, target_model)
                            
                            # Load model - with or without custom repo
                            if repo_path:
                                from pathlib import Path
                                repo = Path(repo_path)
                                if not repo.is_dir():
                                    raise ValueError(f"Repo path does not exist: {repo_path}")
                                model = get_model(actual_model_name, repo=repo)
                                current_repo = repo_path
                            else:
                                model = get_model(actual_model_name)
                                current_repo = None
                            
                            model.to(device)
                            model.eval()
                            model_ref[0] = model  # Keep reference for cleanup
                            current_model_name = target_model
                            
                            # Add to cache
                            stems = list(model.sources)
                            add_to_cache(target_model, repo_path, model, stems)
                        
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
                    "ready": True,
                    "cached_models": [k.split('|')[0] for k in model_cache.keys()],
                    "cache_size": len(model_cache),
                    "max_cache_size": MAX_CACHED_MODELS
                })
            
            elif cmd == "list_cached":
                # List all models currently in cache
                cached_info = []
                for key, (m, repo, stems) in model_cache.items():
                    model_name = key.split('|')[0]
                    cached_info.append({
                        "model": model_name,
                        "repo": repo,
                        "stems": stems,
                        "active": model_name == current_model_name
                    })
                send_json({
                    "status": "cached_models",
                    "models": cached_info,
                    "count": len(model_cache),
                    "max": MAX_CACHED_MODELS
                })
            
            elif cmd == "preload_model":
                # Preload a model into cache without making it active
                model_name = job.get("model", "htdemucs_ft")
                repo_path = job.get("repo")
                
                try:
                    # Check if already cached
                    if get_cached_model(model_name, repo_path):
                        send_json({
                            "status": "model_preloaded",
                            "model": model_name,
                            "already_cached": True
                        })
                        continue
                    
                    send_json({"status": "preloading_model", "model": model_name})
                    
                    # Determine actual model name for custom models
                    actual_model_name = CUSTOM_MODELS.get(model_name, model_name)
                    
                    # Load model - with or without custom repo
                    if repo_path:
                        from pathlib import Path
                        repo = Path(repo_path)
                        if not repo.is_dir():
                            raise ValueError(f"Repo path does not exist: {repo_path}")
                        new_model = get_model(actual_model_name, repo=repo)
                    else:
                        new_model = get_model(actual_model_name)
                    
                    new_model.to(device)
                    new_model.eval()
                    
                    # Get stems and add to cache
                    stems = list(new_model.sources)
                    add_to_cache(model_name, repo_path, new_model, stems)
                    
                    send_json({
                        "status": "model_preloaded",
                        "model": model_name,
                        "stems": stems,
                        "cached_count": len(model_cache)
                    })
                    
                except Exception as e:
                    send_json({"status": "error", "message": f"Failed to preload model: {str(e)}"})
            
            elif cmd == "clear_cache":
                # Clear all cached models
                count = len(model_cache)
                clear_cache()
                model = None
                current_model_name = None
                current_repo = None
                model_ref[0] = None
                send_json({"status": "cache_cleared", "models_cleared": count})
            
            elif cmd == "set_storage_type":
                # Configure mmap preference based on storage type
                # SSD: use mmap (faster initial load, reads from disk on-demand)
                # HDD: disable mmap (loading entire model is faster than random seeks)
                storage_type = job.get("type", "ssd").lower()
                use_mmap = storage_type == "ssd"
                set_mmap_preferred(use_mmap)
                send_json({
                    "status": "storage_configured",
                    "type": storage_type,
                    "mmap_enabled": use_mmap,
                    "mmap_supported": check_mmap_support()
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
    
    # Cleanup GPU memory - clear cache and current model
    clear_cache()
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
        # Use soundfile for saving (avoids torchaudio.save -> torchcodec on Win/Mac)
        _patch_demucs_save_audio_to_use_soundfile()
        # Standard Demucs CLI
        from demucs.separate import main as demucs_main
        demucs_main()


if __name__ == '__main__':
    main()
