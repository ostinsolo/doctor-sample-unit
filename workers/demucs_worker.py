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
    
    Worker mode uses Demucs load_track (ffmpeg / torchaudio + convert_audio) and
    save_audio (torchaudio) as in the original CLI.
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


def _load_track_safe(path, audio_channels, samplerate):
    """
    Demucs-style load: ffmpeg (AudioFile) or torchaudio + convert_audio.
    Same channel/resample behavior as demucs.separate.load_track; no custom
    mono→stereo. Returns (wav, None) on success or (None, error_msg) on failure.
    """
    import subprocess
    from pathlib import Path
    from demucs.audio import AudioFile, convert_audio

    path = Path(path)
    errors = {}
    wav = None
    try:
        wav = AudioFile(path).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels,
        )
    except FileNotFoundError:
        errors["ffmpeg"] = "FFmpeg is not installed."
    except subprocess.CalledProcessError:
        errors["ffmpeg"] = "FFmpeg could not read the file."

    if wav is None:
        try:
            import torchaudio as ta
            wav, sr = ta.load(str(path))
        except RuntimeError as err:
            errors["torchaudio"] = str(err.args[0]) if err.args else str(err)
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        msg = "Could not load file. " + " ".join(f"{k}: {v}" for k, v in errors.items())
        return None, msg
    return wav, None


def _save_audio_demucs_or_fallback(wav, path, samplerate, clip="rescale", bits_per_sample=16, **kwargs):
    """
    Use Demucs save_audio (torchaudio) when possible. If torchcodec is missing
    and torchaudio.save fails, fall back to soundfile with prevent_clip + PCM_16
    so behavior matches Demucs and avoids crackling.
    """
    from demucs.audio import save_audio, prevent_clip

    try:
        save_audio(wav, path, samplerate=samplerate, clip=clip, bits_per_sample=bits_per_sample, **kwargs)
        return
    except Exception as e:
        err = str(e).lower()
        if "torchcodec" not in err and "module" not in err:
            raise
    import soundfile as sf
    wav = prevent_clip(wav, mode=clip)
    data = wav.detach().cpu().numpy()
    if data.ndim == 2:
        data = data.T
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(bits_per_sample, "PCM_16")
    sf.write(str(path), data, samplerate, subtype=subtype)


def _patch_cli_save_audio_to_soundfile():
    """
    When running standard CLI (not worker), replace Demucs save_audio with a
    soundfile-based implementation. Avoids torchaudio.save -> torchcodec when
    torchcodec is not installed. Uses prevent_clip + PCM_16 like Demucs.
    """
    import demucs.audio as _da
    from pathlib import Path
    import soundfile as sf

    _prevent_clip = _da.prevent_clip
    _encode_mp3 = _da.encode_mp3

    def _save(wav, path, samplerate, bitrate=320, clip="rescale", bits_per_sample=16, as_float=False, preset=2):
        wav = _prevent_clip(wav, mode=clip)
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".mp3":
            _encode_mp3(wav, path, samplerate, bitrate, preset, verbose=True)
            return
        if suffix == ".wav":
            subtype = "FLOAT" if as_float else {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(bits_per_sample, "PCM_16")
        elif suffix == ".flac":
            subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(bits_per_sample, "PCM_16")
        else:
            raise ValueError(f"Invalid suffix for path: {suffix}")
        data = wav.detach().cpu().numpy()
        if data.ndim == 2:
            data = data.T
        sf.write(str(path), data, samplerate, subtype=subtype)

    _da.save_audio = _save
    import demucs.separate as _ds
    _ds.save_audio = _save


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
        from pathlib import Path

        _configure_demucs_cache()
        
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
                overlap = job.get("overlap", 0.25)
                jobs = max(0, int(job.get("jobs", 0)))  # -j / num_workers: segment-level parallelism (CPU-only)
                segment = job.get("segments") or job.get("segment")  # optional override (seconds)
                if segment is not None:
                    try:
                        segment = float(segment)
                    except (TypeError, ValueError):
                        segment = None
                if "shifts" in job:
                    shifts = job["shifts"]
                else:
                    m = (model_name or current_model_name or "").lower()
                    shifts = 0 if m in ("mdx_q", "mdx_extra_q") else 1

                if not input_path:
                    send_json({"status": "error", "message": "No input file specified"})
                    continue
                
                if not os.path.exists(input_path):
                    send_json({"status": "error", "message": f"Input file not found: {input_path}"})
                    continue
                
                # Normalize for comparison (avoid spurious reload from trim/case mismatch)
                def _n(s):
                    return (s or "").strip().lower()
                # Load model if different or not loaded (including repo change)
                needs_reload = (
                    model is None or
                    (model_name and _n(model_name) != _n(current_model_name or "")) or
                    (repo_path is not None and (current_repo is None or (repo_path or "").strip() != (current_repo or "").strip()))
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

                    wav, load_err = _load_track_safe(
                        input_path, model.audio_channels, model.samplerate
                    )
                    if load_err:
                        send_json({"status": "error", "message": load_err})
                        continue

                    # Demucs CLI: normalize input, denormalize output (same as separate.py)
                    ref = wav.mean(0)
                    wav = wav - ref.mean()
                    s = ref.std()
                    if s > 1e-8:
                        wav = wav / s

                    apply_kw = dict(
                        shifts=shifts,
                        overlap=overlap,
                        progress=False,
                        device=device,
                        num_workers=jobs,
                    )
                    if segment is not None and segment > 0:
                        apply_kw["segment"] = segment
                    with torch.no_grad():
                        sources = apply_model(model, wav[None].to(device), **apply_kw)[0]
                    if s > 1e-8:
                        sources = sources * s
                    sources = sources + ref.mean()

                    input_basename = os.path.splitext(os.path.basename(input_path))[0]
                    out_dir = Path(output_dir) / current_model_name / input_basename
                    out_dir.mkdir(parents=True, exist_ok=True)

                    save_kw = {
                        "samplerate": model.samplerate,
                        "clip": "rescale",
                        "bits_per_sample": 16,
                    }
                    output_files = []
                    stems = list(model.sources)

                    if two_stems:
                        target_idx = None
                        for i, stem in enumerate(stems):
                            if stem.lower() == two_stems.lower():
                                target_idx = i
                                break
                        if target_idx is not None:
                            _save_audio_demucs_or_fallback(
                                sources[target_idx], str(out_dir / f"{two_stems}.wav"), **save_kw
                            )
                            output_files.append(f"{two_stems}.wav")
                            no_target = sources.sum(dim=0) - sources[target_idx]
                            _save_audio_demucs_or_fallback(
                                no_target, str(out_dir / f"no_{two_stems}.wav"), **save_kw
                            )
                            output_files.append(f"no_{two_stems}.wav")
                        else:
                            for i, stem in enumerate(stems):
                                _save_audio_demucs_or_fallback(sources[i], str(out_dir / f"{stem}.wav"), **save_kw)
                                output_files.append(f"{stem}.wav")
                    else:
                        for i, stem in enumerate(stems):
                            _save_audio_demucs_or_fallback(sources[i], str(out_dir / f"{stem}.wav"), **save_kw)
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
        _configure_demucs_cache()
        _patch_cli_save_audio_to_soundfile()
        from demucs.separate import main as demucs_main
        demucs_main()


if __name__ == '__main__':
    main()
