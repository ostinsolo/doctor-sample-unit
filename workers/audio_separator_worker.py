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
# CRITICAL: Set thread environment variables BEFORE any heavy imports
# This must be done early to ensure OpenMP/BLAS use correct thread counts
# =============================================================================
def _set_thread_env_early():
    """Set thread environment variables before any heavy imports"""
    import subprocess
    import platform
    
    def get_physical_cores():
        """Get number of physical CPU cores (not logical/hyperthreaded)"""
        system = platform.system()
        try:
            if system == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'hw.physicalcpu'], 
                                        capture_output=True, text=True)
                return int(result.stdout.strip())
            elif system == 'Linux':
                with open('/proc/cpuinfo') as f:
                    content = f.read()
                cores = set()
                physical_id = core_id = None
                for line in content.split('\n'):
                    if 'physical id' in line:
                        physical_id = line.split(':')[1].strip()
                    elif 'core id' in line:
                        core_id = line.split(':')[1].strip()
                        if physical_id is not None:
                            cores.add((physical_id, core_id))
                return len(cores) if cores else os.cpu_count() // 2
            elif system == 'Windows':
                result = subprocess.run(['wmic', 'cpu', 'get', 'NumberOfCores'], 
                                        capture_output=True, text=True)
                lines = [l.strip() for l in result.stdout.split('\n') if l.strip().isdigit()]
                return sum(int(l) for l in lines)
        except:
            pass
        return max(1, os.cpu_count() // 2)
    
    num_threads = get_physical_cores()
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    os.environ['KMP_BLOCKTIME'] = '0'

_set_thread_env_early()

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
# HELPER: Setup CPU threading optimizations (for CPU device only)
# ============================================================================
def get_physical_cores():
    """Get number of physical CPU cores (not logical/hyperthreaded)"""
    import subprocess
    import platform
    
    system = platform.system()
    try:
        if system == 'Darwin':  # macOS
            result = subprocess.run(['sysctl', '-n', 'hw.physicalcpu'], 
                                    capture_output=True, text=True)
            return int(result.stdout.strip())
        elif system == 'Linux':
            with open('/proc/cpuinfo') as f:
                content = f.read()
            cores = set()
            physical_id = core_id = None
            for line in content.split('\n'):
                if 'physical id' in line:
                    physical_id = line.split(':')[1].strip()
                elif 'core id' in line:
                    core_id = line.split(':')[1].strip()
                    if physical_id is not None:
                        cores.add((physical_id, core_id))
            return len(cores) if cores else os.cpu_count() // 2
        elif system == 'Windows':
            result = subprocess.run(['wmic', 'cpu', 'get', 'NumberOfCores'], 
                                    capture_output=True, text=True)
            lines = [l.strip() for l in result.stdout.split('\n') if l.strip().isdigit()]
            return sum(int(l) for l in lines)
    except:
        pass
    return max(1, os.cpu_count() // 2)

def _setup_cpu_threading(device, num_threads=None):
    """
    Configure CPU-specific threading optimizations.
    
    IMPORTANT: Only applies to CPU device. MPS/CUDA handle their own threading.
    
    Args:
        device: Device string ("cpu", "cuda", "mps")
        num_threads: Number of threads (None = auto-detect physical cores)
    
    Returns:
        num_threads: Number of threads used
    """
    import torch
    
    if device != "cpu":
        # For MPS/CUDA, don't override threading - GPU handles parallelism internally
        return get_physical_cores() if num_threads is None else num_threads
    
    if num_threads is None:
        num_threads = get_physical_cores()
    
    torch.set_num_threads(num_threads)
    # Only set interop threads if not already set (PyTorch doesn't allow changing after parallel work starts)
    try:
        current_interop = torch.get_num_interop_threads()
        if current_interop == 1:  # Default value, safe to change
            torch.set_num_interop_threads(max(2, num_threads // 2))  # Half threads for inter-op
    except RuntimeError:
        # Already set or parallel work started, skip
        pass
    
    # Intel CPU/OpenMP/BLAS optimizations for maximum parallelization (from old build)
    # These are CPU-specific and should NOT be set for MPS/CUDA
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'  # Thread pinning
    os.environ['KMP_BLOCKTIME'] = '0'  # Immediate thread release
    
    return num_threads

def _setup_cpu_precision(precision='medium'):
    """
    Configure CPU matmul precision for speed optimization.
    
    IMPORTANT: Only applies to CPU. MPS/CUDA have their own precision handling.
    
    Args:
        precision: 'medium' (faster, ~10% speedup) or 'high' (best quality)
    """
    import torch
    if hasattr(torch, 'set_float32_matmul_precision') and precision == 'medium':
        torch.set_float32_matmul_precision('medium')

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
            {"cmd": "batch", "jobs": [{"cmd": "separate", ...}, {"cmd": "apollo", ...}]}
            {"cmd": "exit"}
        
        Output (one JSON per line):
            {"status": "ready"}
            {"status": "done", "files": [...], "elapsed": 12.5}
            {"status": "error", "message": "..."}
            For batch: {"status": "done"|"error", "results": [...], "message": "..."}
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
        import torch
        
        # =============================================================================
        # CRITICAL FIX: PyTorch 2.2.2 compatibility for audio-separator
        # =============================================================================
        # audio-separator library tries to use torch.amp.autocast_mode.is_autocast_available
        # which doesn't exist in PyTorch 2.2.2. We need to add this attribute.
        # =============================================================================
        if not hasattr(torch.amp.autocast_mode, 'is_autocast_available'):
            def _is_autocast_available(device_type=None):
                """Check if autocast is available (CUDA or MPS)
                
                Args:
                    device_type: Optional device type (ignored, kept for compatibility)
                """
                return torch.cuda.is_available() or (
                    getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                )
            torch.amp.autocast_mode.is_autocast_available = _is_autocast_available
        
        # =============================================================================
        # CRITICAL FIX: Make ONNX optional for .pth models
        # =============================================================================
        # audio-separator imports onnxruntime at module level, but we only use .pth models
        # Create a dummy onnxruntime module if it's not available
        # =============================================================================
        import sys
        if 'onnxruntime' not in sys.modules:
            try:
                import onnxruntime
            except ImportError:
                # Create a dummy onnxruntime module for .pth models (ONNX not needed)
                # audio-separator uses: import onnxruntime as ort
                # Then: ort.InferenceSession(...) and ort.get_available_providers()
                class DummyInferenceSession:
                    """Dummy InferenceSession that raises error if used (ONNX models not supported)"""
                    def __init__(self, *args, **kwargs):
                        raise ImportError(
                            "ONNX runtime not available. Only .pth models are supported. "
                            "Install onnxruntime if you need ONNX model support."
                        )
                
                def dummy_get_available_providers():
                    """Return empty list - no ONNX providers available"""
                    return []
                
                import types
                dummy_ort = types.ModuleType('onnxruntime')
                dummy_ort.InferenceSession = DummyInferenceSession
                dummy_ort.get_available_providers = dummy_get_available_providers
                sys.modules['onnxruntime'] = dummy_ort
                print("[AudioSep] ONNX not available - using dummy module (OK for .pth models)", file=sys.stderr)
        
        # =============================================================================
        # CRITICAL FIX: Make huggingface_hub optional for Apollo models
        # =============================================================================
        # Apollo's BaseModel uses PyTorchModelHubMixin from huggingface_hub, but we load
        # models from local files, so huggingface_hub is not needed.
        # Create a dummy PyTorchModelHubMixin if it's not available.
        # =============================================================================
        if 'huggingface_hub' not in sys.modules:
            try:
                from huggingface_hub import PyTorchModelHubMixin
            except ImportError:
                # Create a dummy PyTorchModelHubMixin for local model loading
                # Apollo models are loaded from local .ckpt files, not from HuggingFace Hub
                # The mixin is used in class definition: class BaseModel(nn.Module, PyTorchModelHubMixin, repo_url="...", pipeline_tag="...")
                # We need to handle the __init_subclass__ method to accept keyword arguments
                class DummyPyTorchModelHubMixin:
                    """Dummy mixin that does nothing - models loaded from local files"""
                    def __init_subclass__(cls, **kwargs):
                        # Accept any keyword arguments (repo_url, pipeline_tag, etc.) but do nothing
                        super().__init_subclass__()
                
                import types
                dummy_hf = types.ModuleType('huggingface_hub')
                dummy_hf.PyTorchModelHubMixin = DummyPyTorchModelHubMixin
                sys.modules['huggingface_hub'] = dummy_hf
                print("[AudioSep] HuggingFace Hub not available - using dummy module (OK for local Apollo models)", file=sys.stderr)
        
        # =============================================================================
        # CRITICAL FIX: Make pytorch_lightning optional for inference
        # =============================================================================
        # Bandit/MDX23c models and apollo/look2hear import pytorch_lightning at module level.
        # We only need inference - create a minimal dummy so those imports succeed.
        # =============================================================================
        if 'pytorch_lightning' not in sys.modules:
            try:
                import pytorch_lightning
            except ImportError:
                import types
                from typing import Optional, Any
                # Minimal LightningModule for inference (just nn.Module)
                class DummyLightningModule(torch.nn.Module):
                    """Dummy LightningModule - inference only, no training"""
                    pass
                def _rank_zero_only(f):
                    return f
                dummy_pl = types.ModuleType('pytorch_lightning')
                dummy_pl.__version__ = '0.0.0-dummy'
                dummy_pl.LightningModule = DummyLightningModule
                dummy_utils = types.ModuleType('pytorch_lightning.utilities')
                dummy_utils.rank_zero_only = _rank_zero_only
                dummy_types = types.ModuleType('pytorch_lightning.utilities.types')
                dummy_types.STEP_OUTPUT = Optional[Any]
                dummy_utils.types = dummy_types
                dummy_pl.utilities = dummy_utils
                dummy_callbacks = types.ModuleType('pytorch_lightning.callbacks')
                dummy_progress = types.ModuleType('pytorch_lightning.callbacks.progress')
                dummy_rich = types.ModuleType('pytorch_lightning.callbacks.progress.rich_progress')
                dummy_progress.rich_progress = dummy_rich
                dummy_callbacks.progress = dummy_progress
                dummy_pl.callbacks = dummy_callbacks
                sys.modules['pytorch_lightning'] = dummy_pl
                sys.modules['pytorch_lightning.utilities'] = dummy_utils
                sys.modules['pytorch_lightning.utilities.types'] = dummy_types
                sys.modules['pytorch_lightning.callbacks'] = dummy_callbacks
                sys.modules['pytorch_lightning.callbacks.progress'] = dummy_progress
                sys.modules['pytorch_lightning.callbacks.progress.rich_progress'] = dummy_rich
                print("[AudioSep] PyTorch Lightning not available - using dummy module (OK for inference)", file=sys.stderr)
        
        from audio_separator.separator import Separator
        
        # Device for use_autocast and VR speed opts (CUDA/MPS benefit from both)
        # CRITICAL: Proper MPS detection (only on Apple Silicon, not Intel Mac)
        def _should_use_mps():
            """Check if MPS should actually be used (only on Apple Silicon)"""
            import platform
            import subprocess
            if platform.machine() != 'arm64':
                return False
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=1)
                if result.returncode == 0:
                    cpu_brand = result.stdout.strip()
                    if not any(x in cpu_brand for x in ['Apple', 'M1', 'M2', 'M3', 'M4']):
                        return False
            except:
                return False
            try:
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    return False
                x = torch.randn(10, 10).to('mps')
                _ = x * 2
                torch.mps.synchronize()
                return True
            except:
                return False
        
        _device = "cuda" if torch.cuda.is_available() else (
            "mps" if _should_use_mps() else "cpu"
        )
        
        # CRITICAL: Setup CPU threading optimizations BEFORE any computation
        # This matches the optimizations in BS-RoFormer worker for maximum CPU performance
        device_str = str(_device)
        if device_str == "cpu":
            _setup_cpu_threading(device_str)
            _setup_cpu_precision('medium')  # ~10% speedup on CPU
        
        _use_autocast = _device in ("cuda", "mps")
        _vr_params = {
            "batch_size": 1,
            "window_size": 512,
            "aggression": 5,
            "enable_tta": False,
            "enable_post_process": False,
            "post_process_threshold": 0.2,
            "high_end_process": False,
        }
        if _use_autocast:
            # Faster VR on GPU/MPS: larger batches (no quality impact). Keep window_size=512 for quality.
            _vr_params["batch_size"] = 8
        
        # Initialize separator (but don't load model yet)
        separator = Separator(
            log_level=30,  # WARNING level
            output_format="WAV",
            use_autocast=_use_autocast,
            vr_params=_vr_params,
        )
        
        # CRITICAL: Re-apply CPU threading optimizations AFTER Separator init
        # The Separator class may reset thread settings during initialization
        if device_str == "cpu":
            _setup_cpu_threading(device_str)
            _setup_cpu_precision('medium')
            # Verify threads are set correctly
            if torch.get_num_threads() < get_physical_cores():
                torch.set_num_threads(get_physical_cores())
                # Don't try to set interop threads again - may fail if parallel work started
                try:
                    if torch.get_num_interop_threads() == 1:
                        torch.set_num_interop_threads(max(2, get_physical_cores() // 2))
                except RuntimeError:
                    pass
        
        send_json({"status": "ready", "message": "Worker ready"})
        
    except Exception as e:
        send_json({"status": "error", "message": f"Failed to initialize: {str(e)}"})
        return 1
    
    def _handle_one(job, emit):
        """Run a single job (load_model, load_apollo, separate, apollo, get_status, set_storage_type). Use emit instead of send_json."""
        nonlocal current_model, apollo_model_path, apollo_model, apollo_device, _device
        cmd = job.get("cmd", "")
        if cmd == "load_model":
            model_name = job.get("model")
            model_dir = job.get("model_file_dir")
            if not model_name:
                emit({"status": "error", "message": "No model specified"})
                return
            try:
                emit({"status": "loading_model", "model": model_name})
                if model_dir:
                    separator.model_file_dir = model_dir
                separator.load_model(model_name)
                current_model = model_name
                emit({"status": "model_loaded", "model": model_name})
            except Exception as e:
                emit({"status": "error", "message": f"Failed to load model: {str(e)}"})
            return
        if cmd == "load_apollo":
            model_path = job.get("model_path")
            config_path = job.get("config_path")
            feature_dim = job.get("feature_dim")
            layer = job.get("layer")
            if not model_path:
                emit({"status": "error", "message": "No model_path specified"})
                return
            try:
                emit({"status": "loading_model", "model": os.path.basename(model_path)})
                import torch
                import platform
                from apollo.apollo_separator import get_device, get_model_config, load_checkpoint
                from apollo.look2hear.models.apollo import Apollo
                # CRITICAL: Apollo doesn't work well with MPS on Intel Mac (FFT operations not supported)
                # Force CPU on Intel Mac, use get_device() only on Apple Silicon
                if platform.machine() == 'arm64':
                    device = get_device()
                else:
                    # Intel Mac: force CPU (MPS FFT operations not supported)
                    device = torch.device('cpu')
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
                if feature_dim is None or layer is None:
                    model_name = os.path.basename(model_path)
                    auto_config = get_model_config(model_name)
                    if feature_dim is None:
                        feature_dim = auto_config['feature_dim']
                    if layer is None:
                        layer = auto_config['layer']
                if apollo_model is not None:
                    del apollo_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                apollo_model = Apollo(sr=44100, win=20, feature_dim=feature_dim, layer=layer)
                apollo_model = load_checkpoint(apollo_model, model_path, device)
                apollo_model.to(device)
                apollo_model.eval()
                apollo_device = device
                apollo_model_path = model_path
                emit({"status": "model_loaded", "model": os.path.basename(model_path), "type": "apollo",
                      "feature_dim": feature_dim, "layer": layer, "device": str(device)})
            except Exception as e:
                emit({"status": "error", "message": f"Failed to load Apollo model: {str(e)}"})
            return
        if cmd == "separate":
            input_path = job.get("input")
            output_dir = job.get("output_dir", os.getcwd())
            model_name = job.get("model")
            model_dir = job.get("model_file_dir")
            vr_enable_tta = job.get("vr_enable_tta", False)
            vr_high_end_process = job.get("vr_high_end_process", False)
            vr_post_process = job.get("vr_post_process", False)
            vr_aggression = job.get("vr_aggression")
            single_stem = job.get("single_stem")
            if not input_path:
                emit({"status": "error", "message": "No input file specified"})
                return
            if not os.path.exists(input_path):
                emit({"status": "error", "message": f"Input file not found: {input_path}"})
                return
            if model_name and model_name != current_model:
                try:
                    emit({"status": "loading_model", "model": model_name})
                    if model_dir:
                        separator.model_file_dir = model_dir
                    separator.load_model(model_name)
                    current_model = model_name
                except Exception as e:
                    emit({"status": "error", "message": f"Failed to load model: {str(e)}"})
                    return
            if current_model is None:
                emit({"status": "error", "message": "No model loaded. Use load_model first."})
                return
            try:
                emit({"status": "separating", "input": os.path.basename(input_path)})
                start_time = time.time()
                separator.output_dir = output_dir
                if separator.model_instance:
                    separator.model_instance.output_dir = output_dir
                if separator.model_instance:
                    if hasattr(separator.model_instance, 'enable_tta'):
                        separator.model_instance.enable_tta = vr_enable_tta
                    if hasattr(separator.model_instance, 'high_end_process'):
                        separator.model_instance.high_end_process = vr_high_end_process
                    if hasattr(separator.model_instance, 'post_process_threshold'):
                        separator.model_instance.post_process_threshold = 0.2 if vr_post_process else 0.2
                    if vr_aggression is not None and hasattr(separator.model_instance, 'aggression'):
                        agg = float(int(vr_aggression)) / 100.0
                        separator.model_instance.aggression = agg
                        mp = separator.model_instance.model_params.param
                        separator.model_instance.aggressiveness = {
                            "value": agg,
                            "split_bin": mp["band"][1]["crop_stop"],
                            "aggr_correction": mp.get("aggr_correction"),
                        }
                # CRITICAL: Re-apply CPU threading optimizations right before separation
                # This ensures maximum CPU utilization during the actual computation
                if _device == "cpu":
                    import torch  # Ensure torch is imported in this scope
                    _setup_cpu_threading("cpu")
                    _setup_cpu_precision('medium')
                    # Force thread count to physical cores for maximum performance
                    num_threads = get_physical_cores()
                    torch.set_num_threads(num_threads)
                    # Don't try to set interop threads - may fail if parallel work started
                    try:
                        if torch.get_num_interop_threads() == 1:
                            torch.set_num_interop_threads(max(2, num_threads // 2))
                    except RuntimeError:
                        pass
                
                if single_stem:
                    output_files = separator.separate(input_path, primary_stem=single_stem)
                else:
                    output_files = separator.separate(input_path)
                elapsed = time.time() - start_time
                if not (output_files and len(output_files) > 0):
                    emit({"status": "error", "message": "Separation produced no output files", "elapsed": round(elapsed, 2), "files": []})
                else:
                    emit({"status": "done", "elapsed": round(elapsed, 2), "files": output_files})
            except Exception as e:
                emit({"status": "error", "message": f"Separation failed: {str(e)}", "traceback": traceback.format_exc()})
            return
        if cmd == "apollo":
            input_path = job.get("input")
            output_path = job.get("output")
            model_path = job.get("model_path")
            config_path = job.get("config_path")
            feature_dim = job.get("feature_dim")
            layer = job.get("layer")
            chunk_seconds = job.get("chunk_seconds", 7.0)
            chunk_overlap = job.get("chunk_overlap", 0.5)
            if not input_path:
                emit({"status": "error", "message": "No input file specified"})
                return
            if not output_path:
                base = os.path.splitext(os.path.basename(input_path))[0]
                output_path = f"{base}_restored.wav"
            if not model_path:
                emit({"status": "error", "message": "No model_path specified for Apollo"})
                return
            if not os.path.exists(input_path):
                emit({"status": "error", "message": f"Input file not found: {input_path}"})
                return
            if not os.path.exists(model_path):
                emit({"status": "error", "message": f"Model not found: {model_path}"})
                return
            try:
                emit({"status": "restoring", "input": os.path.basename(input_path)})
                start_time = time.time()
                use_cached = (apollo_model is not None and apollo_model_path == model_path)
                if use_cached:
                    import torch
                    from apollo.apollo_separator import load_audio, save_audio, _chunk_restore
                    audio = load_audio(input_path)
                    audio = audio.to(apollo_device)
                    total_samples = audio.shape[-1]
                    sr = 44100
                    duration_seconds = total_samples / sr
                    if chunk_seconds > 0 and duration_seconds > chunk_seconds:
                        restored = _chunk_restore(apollo_model, audio, sr, chunk_seconds, chunk_overlap, apollo_device)
                    else:
                        with torch.inference_mode():
                            restored = apollo_model(audio)
                    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                    save_audio(output_path, restored)
                else:
                    from apollo.apollo_separator import restore_audio
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
                emit({"status": "done", "elapsed": round(elapsed, 2), "files": [output_path], "cached": use_cached})
            except Exception as e:
                emit({"status": "error", "message": f"Apollo restoration failed: {str(e)}", "traceback": traceback.format_exc()})
            return
        if cmd == "get_status":
            emit({"status": "status", "separator_model": current_model, "apollo_model": apollo_model_path, "ready": True})
            return
        if cmd == "set_storage_type":
            storage_type = job.get("type", "ssd").lower()
            use_mmap = storage_type == "ssd"
            set_mmap_preferred(use_mmap)
            emit({"status": "storage_configured", "type": storage_type, "mmap_enabled": use_mmap, "mmap_supported": check_mmap_support()})
            return
        emit({"status": "error", "message": f"Unknown command: {cmd}"})

    # Main job loop
    while not _shutdown_requested.is_set():
        try:
            # Check if stdin is closed before reading (prevents busy loop)
            if sys.stdin.closed:
                break
            
            # Use select with timeout to avoid blocking indefinitely
            # This allows the shutdown flag to be checked periodically
            if sys.platform != "win32":
                readable, _, _ = select.select([sys.stdin], [], [], 0.5)
                if not readable:
                    # No input available, check shutdown flag and continue
                    continue
            
            line = sys.stdin.readline()
            if not line:
                # EOF or stdin closed
                break
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            cmd = job.get("cmd", "")
            if cmd == "exit":
                send_json({"status": "exiting"})
                break
            if cmd == "ping":
                send_json({"status": "pong", "separator_model": current_model, "apollo_model": apollo_model_path})
                continue
            if cmd == "batch":
                jobs = job.get("jobs", [])
                results = []
                for sub in jobs:
                    cur = []
                    _handle_one(sub, cur.append)
                    results.append(cur[-1] if cur else {"status": "error", "message": "no response"})
                    if results[-1].get("status") == "error":
                        break
                err = results and results[-1].get("status") == "error"
                send_json({"status": "error" if err else "done", "results": results, "message": results[-1].get("message") if err and results else None})
                continue
            _handle_one(job, send_json)
        
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
