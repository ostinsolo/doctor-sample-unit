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

# Import torch AFTER the frozen exe setup code above
import torch

# ============================================================================
# PATH SETUP - Find the bs-roformer models directory
# ============================================================================

# Global configs directory (can be set via --configs-dir)
CONFIGS_DIR = None

# Get script directory
if getattr(sys, 'frozen', False):
    # Frozen exe: exe is in dsu/ folder, configs are also in dsu/
    SCRIPT_DIR = os.path.dirname(sys.executable)
    DIST_ROOT = SCRIPT_DIR  # configs/ is in same folder as exe
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    # Development: script is in workers/ folder
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DIST_ROOT = os.path.dirname(SCRIPT_DIR)  # parent of workers/
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Add distribution root to path (contains models/, utils/, configs/, apollo/)
if os.path.isdir(os.path.join(DIST_ROOT, 'models')):
    sys.path.insert(0, DIST_ROOT)

# Also check for bs-roformer-freeze-repo in dev (for development)
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


# Global flag for mmap support (checked once at startup)
_MMAP_SUPPORTED = None
_MMAP_PREFERRED = True  # Can be set to False for HDD storage

def check_mmap_support():
    """Check if torch.load supports mmap parameter (PyTorch 2.1+)"""
    global _MMAP_SUPPORTED
    if _MMAP_SUPPORTED is not None:
        return _MMAP_SUPPORTED
    
    try:
        import inspect
        sig = inspect.signature(torch.load)
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
    
    Notes:
        - mmap=True works best with SSD storage
        - mmap=True reads from disk on-demand (faster initial load, less RAM)
        - Older PyTorch versions (< 2.1) don't support mmap
        - Intel Mac builds may use older PyTorch without mmap support
    """
    mmap_available = check_mmap_support()
    
    # Determine if we should use mmap
    if use_mmap is False:
        # Explicitly disabled (e.g., HDD storage)
        should_mmap = False
    elif use_mmap is True:
        # Explicitly requested - use if available
        should_mmap = mmap_available
    else:
        # Auto: use if available AND preferred
        should_mmap = mmap_available and _MMAP_PREFERRED
    
    try:
        if should_mmap:
            return torch.load(checkpoint_path, map_location=map_location, 
                            weights_only=weights_only, mmap=True)
        else:
            return torch.load(checkpoint_path, map_location=map_location, 
                            weights_only=weights_only)
    except TypeError as e:
        # mmap parameter not supported - fallback
        if 'mmap' in str(e) and should_mmap:
            # Mark mmap as not supported for future calls
            global _MMAP_SUPPORTED
            _MMAP_SUPPORTED = False
            return torch.load(checkpoint_path, map_location=map_location, 
                            weights_only=weights_only)
        raise
    except Exception as e:
        # If mmap fails for other reasons (corrupted file on mmap, etc.), try without
        if should_mmap:
            try:
                return torch.load(checkpoint_path, map_location=map_location, 
                                weights_only=weights_only)
            except:
                pass
        raise


def set_mmap_preferred(preferred):
    """
    Set whether mmap should be preferred for model loading.
    Set to False if using HDD storage for better performance.
    
    Args:
        preferred: True for SSD storage, False for HDD storage
    """
    global _MMAP_PREFERRED
    _MMAP_PREFERRED = preferred


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
        
        DEFAULT_MODELS = load_models_registry(models_dir)
        
        return True
        
    except Exception as e:
        send_json({"status": "error", "message": f"Failed to init model functions: {str(e)}"})
        return False


def _get_models_json_path(models_dir=None):
    if models_dir:
        return os.path.join(models_dir, 'models.json')
    candidates = [
        os.path.join(PROJECT_ROOT, 'models.json'),
        os.path.join(DIST_ROOT, 'models.json'),
        os.path.join(BSROFORMER_FROZEN_DIR, 'models.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_models_registry(models_dir=None):
    """Load models from registry"""
    models_json = _get_models_json_path(models_dir)
    if os.path.exists(models_json):
        with open(models_json, 'r') as f:
            return json.load(f)
    return DEFAULT_MODELS or {}


def save_models_registry(models, models_dir=None):
    """Save models registry to models.json"""
    models_json = _get_models_json_path(models_dir)
    os.makedirs(os.path.dirname(models_json), exist_ok=True)
    with open(models_json, 'w', encoding='utf-8') as f:
        json.dump(models, f, indent=2, ensure_ascii=False)
    return models_json


def init_models_registry(models_dir=None):
    """Initialize models.json in models_dir from bundled registry"""
    target_path = _get_models_json_path(models_dir)
    if os.path.exists(target_path):
        return target_path, False
    bundled = load_models_registry(None)
    saved_path = save_models_registry(bundled, models_dir)
    return saved_path, True


def detect_model_type_from_checkpoint(checkpoint_path):
    """Detect model type from checkpoint filename patterns"""
    filename = os.path.basename(checkpoint_path).lower()
    
    # Apollo models (must check before roformer since some apollo models have 'roformer' in path)
    if 'apollo' in filename:
        return 'apollo'
    elif 'mel_band' in filename or 'melband' in filename or 'melbandroformer' in filename:
        return 'mel_band_roformer'
    elif 'scnet' in filename:
        return 'scnet'
    elif 'mdx23c' in filename or 'drumsep' in filename:
        return 'mdx23c'
    elif 'bs_roformer' in filename or 'bsroformer' in filename or 'roformer' in filename:
        return 'bs_roformer'
    else:
        # Default to bs_roformer for resurrection/revive style models
        return 'bs_roformer'


def get_apollo_model_config(model_path, config_path=None):
    """Get Apollo model configuration from config file.
    
    Config file is REQUIRED - if missing, raises an error to surface download bugs.
    """
    if not config_path:
        raise ValueError(f"Apollo model requires a config file. No config_path provided for: {model_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Apollo config file not found: {config_path}\n"
            f"This config should have been downloaded with the model.\n"
            f"Try re-downloading the model to get the config file."
        )
    
    # Load config from file
    import yaml
    try:
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse Apollo config file {config_path}: {e}")
    
    if not conf or 'model' not in conf:
        raise ValueError(f"Invalid Apollo config file (missing 'model' section): {config_path}")
    
    feature_dim = conf['model'].get('feature_dim')
    layer = conf['model'].get('layer')
    
    if feature_dim is None:
        raise ValueError(f"Apollo config missing 'feature_dim' in model section: {config_path}")
    if layer is None:
        raise ValueError(f"Apollo config missing 'layer' in model section: {config_path}")
    
    return {
        'feature_dim': feature_dim,
        'layer': layer
    }


def load_model_from_path(checkpoint_path, config_path=None, model_type=None):
    """Load a model directly from checkpoint and config paths"""
    import torch
    import yaml
    from ml_collections import ConfigDict
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Auto-detect model type if not provided
    if not model_type:
        model_type = detect_model_type_from_checkpoint(checkpoint_path)
    
    # Special handling for Apollo models
    if model_type == "apollo":
        return load_apollo_model(checkpoint_path, config_path)
    
    # Find config if not provided
    if not config_path:
        # Try to find config next to checkpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        
        # Try various config naming patterns
        for pattern in [
            f"{checkpoint_name}.yaml",
            f"config_{checkpoint_name}.yaml",
            f"{checkpoint_name}_config.yaml"
        ]:
            candidate = os.path.join(checkpoint_dir, pattern)
            if os.path.exists(candidate):
                config_path = candidate
                break
        
        # Try bundled configs directory
        if not config_path:
            configs_dir = os.path.join(DIST_ROOT, 'configs')
            if os.path.isdir(configs_dir):
                # Look for model-specific bundled config
                for pattern in [
                    f"config_{checkpoint_name}.yaml",
                    f"config_resurrection_vocals.yaml",  # Common vocal model
                    "config_bs_roformer_vocals.yaml"
                ]:
                    candidate = os.path.join(configs_dir, pattern)
                    if os.path.exists(candidate):
                        config_path = candidate
                        break
    
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found for checkpoint: {checkpoint_path}")
    
    # Load config
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
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == "apollo":
        from models.look2hear.models.apollo import Apollo
        model = Apollo(**dict(config.model))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights (with safe mmap support)
    state_dict = safe_torch_load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'state' in state_dict:
        state_dict = state_dict['state']
    
    # Handle Apollo checkpoint format (may have 'audio_model.' prefix)
    if model_type == "apollo":
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('audio_model.'):
                new_state_dict[k.replace('audio_model.', '')] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Try with strict=False for partial matches
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    # Build model_info from config
    stems = getattr(config, 'training', {}).get('instruments', ['vocals', 'other'])
    if hasattr(config, 'inference') and hasattr(config.inference, 'instruments'):
        stems = list(config.inference.instruments)
    
    model_info = {
        "type": model_type,
        "stems": stems,
        "checkpoint": checkpoint_path,
        "config": config_path
    }
    
    return model, config, model_info


def load_apollo_model(checkpoint_path, config_path=None):
    """Load an Apollo model for audio restoration"""
    import torch
    
    # Import Apollo from the models directory structure
    # models/look2hear/models/apollo.py contains the Apollo class
    try:
        from models.look2hear.models.apollo import Apollo
    except ImportError:
        # Try alternative import paths
        try:
            from look2hear.models.apollo import Apollo
        except ImportError:
            try:
                # Fallback for standalone apollo directory
                apollo_dir = os.path.join(DIST_ROOT, 'apollo')
                if os.path.isdir(apollo_dir) and apollo_dir not in sys.path:
                    sys.path.insert(0, apollo_dir)
                from apollo.look2hear.models.apollo import Apollo
            except ImportError:
                raise ImportError("Apollo model not available. Ensure models/look2hear/models/apollo.py exists.")
    
    # Get model configuration
    apollo_config = get_apollo_model_config(checkpoint_path, config_path)
    feature_dim = apollo_config['feature_dim']
    layer = apollo_config['layer']
    
    # Create model
    model = Apollo(sr=44100, win=20, feature_dim=feature_dim, layer=layer)
    
    # Load checkpoint (with safe mmap support)
    checkpoint = safe_torch_load(checkpoint_path, map_location='cpu', weights_only=False)
    
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
    except RuntimeError:
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    
    # Build model_info for Apollo
    model_info = {
        "type": "apollo",
        "stems": ["restored", "addition"],  # Apollo outputs restored audio
        "checkpoint": checkpoint_path,
        "config": config_path,
        "feature_dim": feature_dim,
        "layer": layer
    }
    
    # Create a minimal config object for compatibility
    class ApolloConfig:
        class audio:
            samplerate = 44100
            sample_rate = 44100
    
    return model, ApolloConfig(), model_info


def load_model_impl(model_name, models_dir=None):
    """Load a model by name from registry"""
    import torch
    import yaml
    from ml_collections import ConfigDict
    
    models = load_models_registry(models_dir)
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model_info = models[model_name]
    model_type = model_info.get("type", "bs_roformer")
    
    # Resolve paths - try models_dir first, then project/dist roots
    if models_dir:
        config_path = os.path.join(models_dir, model_info["config"])
        checkpoint_path = os.path.join(models_dir, model_info["checkpoint"])
    else:
        config_path = os.path.join(PROJECT_ROOT, model_info["config"])
        checkpoint_path = os.path.join(PROJECT_ROOT, model_info["checkpoint"])
    
    # Try alternative config path if not found
    if not os.path.exists(config_path):
        config_filename = os.path.basename(model_info["config"])
        search_paths = [
            # Try with full relative path
            os.path.join(DIST_ROOT, model_info["config"]),
            # Try just filename in bundled configs
            os.path.join(DIST_ROOT, "configs", config_filename),
            # Try in project root
            os.path.join(PROJECT_ROOT, model_info["config"]),
            os.path.join(PROJECT_ROOT, "configs", config_filename),
        ]
        # Also try user-specified configs directory
        if 'CONFIGS_DIR' in globals() and CONFIGS_DIR:
            search_paths.insert(0, os.path.join(CONFIGS_DIR, config_filename))
            search_paths.insert(0, os.path.join(CONFIGS_DIR, model_info["config"]))
        
        for alt_config in search_paths:
            if os.path.exists(alt_config):
                config_path = alt_config
                break
    
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
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == "apollo":
        from models.look2hear.models.apollo import Apollo
        model = Apollo(**dict(config.model))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights (with safe mmap support, weights_only=False for PyTorch 2.6+ checkpoints)
    if os.path.exists(checkpoint_path):
        state_dict = safe_torch_load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'state' in state_dict:
            state_dict = state_dict['state']
        
        # Handle Apollo checkpoint format (may have 'audio_model.' prefix)
        if model_type == "apollo":
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('audio_model.'):
                    new_state_dict[k.replace('audio_model.', '')] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    return model, config, model_info


def separate_impl(model, config, input_path, output_dir, model_info, device='cpu',
                  overlap=None, batch_size=None, use_tta=False, output_format='wav',
                  pcm_type='FLOAT', extract_instrumental=False, selected_stems=None,
                  two_stems=None, chunk_seconds=None, chunk_overlap=None, use_fast=False):
    """Run separation with the loaded model"""
    import torch
    import numpy as np
    import soundfile as sf
    import librosa
    
    start_time = time.time()
    model_type = model_info.get("type", "bs_roformer")
    
    # Handle Apollo models differently (restoration, not separation)
    if model_type == "apollo":
        return apollo_restore_impl(
            model, input_path, output_dir, model_info, device,
            output_format, chunk_seconds, chunk_overlap
        )
    
    from utils.model_utils import demix, demix_fast, apply_tta

    # Load audio - use getattr for ConfigDict compatibility
    sample_rate = getattr(config.audio, 'samplerate', None) or getattr(config.audio, 'sample_rate', 44100)
    audio, sr = librosa.load(input_path, sr=sample_rate, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)

    mix_orig = audio.copy()

    # Override config inference parameters if specified
    if overlap is not None:
        config.inference.num_overlap = overlap
    if batch_size is not None:
        config.inference.batch_size = batch_size

    torch_device = torch.device(device)
    model = model.to(torch_device)

    # Run separation
    if use_fast:
        waveforms = demix_fast(config, model, audio, torch_device, model_type=model_info.get("type", "bs_roformer"), pbar=False)
    else:
        waveforms = demix(config, model, audio, torch_device, model_type=model_info.get("type", "bs_roformer"), pbar=False)

    # Apply TTA if requested
    if use_tta and isinstance(waveforms, dict):
        waveforms = apply_tta(config, model, audio, waveforms, torch_device, model_info.get("type", "bs_roformer"), use_fast=use_fast)

    # Normalize waveforms to dict if needed
    if not isinstance(waveforms, dict):
        stems = model_info.get("stems", ["vocals", "accompaniment"])
        waveforms = {stems[i]: waveforms[i] for i in range(min(len(stems), waveforms.shape[0]))}

    # Extract instrumental if requested
    if extract_instrumental:
        vocals_key = None
        for key in waveforms.keys():
            if key.lower() == 'vocals':
                vocals_key = key
                break
        if vocals_key:
            waveforms['instrumental'] = mix_orig - waveforms[vocals_key]

    # Two-stems mode: target + no_target
    if two_stems:
        target_key = None
        for key in waveforms.keys():
            if key.lower() == two_stems.lower():
                target_key = key
                break
        if target_key:
            other_stems = [waveforms[k] for k in waveforms.keys() if k != target_key]
            if other_stems:
                no_stem = sum(other_stems)
            else:
                no_stem = mix_orig - waveforms[target_key]
            waveforms = {
                target_key: waveforms[target_key],
                f'no_{target_key.lower()}': no_stem
            }

    # Validate PCM type for codec
    codec = output_format
    subtype = pcm_type
    if subtype not in sf.available_subtypes(codec):
        subtype = sf.default_subtype(codec)

    # Filter stems if specified (case-insensitive)
    stems_to_save = list(waveforms.keys())
    if selected_stems is not None:
        stem_map = {s.lower(): s for s in waveforms.keys()}
        stems_to_save = [stem_map[s.lower()] for s in selected_stems if s.lower() in stem_map]

    # Create output directory
    basename = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(output_dir, basename)
    os.makedirs(out_dir, exist_ok=True)

    # Save stems
    for stem in stems_to_save:
        wav = waveforms[stem]
        stem_filename = stem.lower()
        out_path = os.path.join(out_dir, f"{stem_filename}.{output_format}")
        sf.write(out_path, wav.T, sample_rate, subtype=subtype)

    return time.time() - start_time


def apollo_restore_impl(model, input_path, output_dir, model_info, device='cpu',
                        output_format='wav', chunk_seconds=None, chunk_overlap=None):
    """Run Apollo audio restoration"""
    import torch
    import numpy as np
    import soundfile as sf
    from scipy import signal
    
    start_time = time.time()
    sr = 44100  # Apollo uses fixed 44.1kHz
    
    # Default chunking parameters
    if chunk_seconds is None:
        chunk_seconds = 5.0 if device == 'cuda' else 7.0
    if chunk_overlap is None:
        chunk_overlap = 0.5 if device == 'cuda' else 0.5
    
    # Load audio
    audio_np, file_sr = sf.read(input_path, dtype='float32')
    
    # Convert to [C, T] format
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]  # Mono: [1, T]
    else:
        audio_np = audio_np.T  # Stereo: [C, T]
    
    # Resample if needed
    if file_sr != sr:
        num_samples = int(audio_np.shape[1] * sr / file_sr)
        audio_np = np.array([signal.resample(ch, num_samples) for ch in audio_np])
    
    # Convert to torch tensor [1, C, T]
    audio = torch.from_numpy(audio_np).unsqueeze(0).float()
    
    # Move to device
    torch_device = torch.device(device)
    audio = audio.to(torch_device)
    model = model.to(torch_device)
    
    # Process with chunking for memory efficiency
    total_samples = audio.shape[-1]
    duration_seconds = total_samples / sr
    
    if chunk_seconds > 0 and duration_seconds > chunk_seconds:
        # Chunked processing
        restored = _apollo_chunk_restore(model, audio, sr, chunk_seconds, chunk_overlap, torch_device)
    else:
        # Full audio processing
        with torch.inference_mode():
            restored = model(audio)
    
    # Create output directory
    basename = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(output_dir, basename)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save restored audio
    restored_np = restored.squeeze(0).cpu().numpy()  # [C, T]
    
    # Convert to [T, C] for soundfile
    if restored_np.shape[0] == 1:
        restored_np = restored_np[0]  # Mono
    else:
        restored_np = restored_np.T  # Stereo
    
    out_path = os.path.join(out_dir, f"restored.{output_format}")
    sf.write(out_path, restored_np, sr, subtype='PCM_16')
    
    # Also save the "addition" (difference between restored and original)
    original_np = audio.squeeze(0).cpu().numpy()
    if original_np.shape[0] == 1:
        original_np = original_np[0]
    else:
        original_np = original_np.T
    
    # Match lengths if needed
    min_len = min(len(restored_np), len(original_np))
    if restored_np.ndim == 1:
        addition_np = restored_np[:min_len] - original_np[:min_len]
    else:
        addition_np = restored_np[:min_len] - original_np[:min_len]
    
    addition_path = os.path.join(out_dir, f"addition.{output_format}")
    sf.write(addition_path, addition_np, sr, subtype='PCM_16')
    
    return time.time() - start_time


def _apollo_chunk_restore(model, audio, sr, chunk_seconds, overlap_seconds, device):
    """Restore audio in overlapping chunks to limit memory usage.
    Accumulates on device and transfers once at end to reduce MPS/CUDA sync bottleneck.
    """
    import torch
    
    total_samples = audio.shape[-1]
    chunk_samples = max(1, int(chunk_seconds * sr))
    overlap_samples = max(0, int(overlap_seconds * sr))
    step = chunk_samples - overlap_samples
    if step <= 0:
        step = chunk_samples // 2
    
    # Accumulate on device to avoid per-chunk GPU->CPU sync
    output = torch.zeros_like(audio, device=device)
    weight_sum = torch.zeros_like(audio, device=device)
    
    for start in range(0, total_samples, step):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[:, :, start:end]
        chunk_len = chunk.shape[-1]
        
        # Pad if needed
        if chunk_len < chunk_samples:
            pad = torch.zeros((chunk.shape[0], chunk.shape[1], chunk_samples - chunk_len), 
                            device=chunk.device, dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=-1)
        
        with torch.inference_mode():
            restored = model(chunk)
        
        restored = restored[:, :, :chunk_len]  # Keep on device
        
        # Apply crossfade weights (on device)
        weight = torch.ones((1, 1, chunk_len), device=device, dtype=restored.dtype)
        if overlap_samples > 0:
            if start > 0:
                fade_in_len = min(overlap_samples, chunk_len)
                weight[:, :, :fade_in_len] *= torch.linspace(0.0, 1.0, fade_in_len, device=device, dtype=restored.dtype)
            if end < total_samples:
                fade_out_len = min(overlap_samples, chunk_len)
                weight[:, :, -fade_out_len:] *= torch.linspace(1.0, 0.0, fade_out_len, device=device, dtype=restored.dtype)
        
        output[:, :, start:end] += restored * weight
        weight_sum[:, :, start:end] += weight
        
        if device.type == "cuda":
            del restored
            torch.cuda.empty_cache()
    
    weight_sum = torch.clamp(weight_sum, min=1e-8)
    # Single transfer at end
    return (output / weight_sum).cpu()


# ============================================================================
# INFERENCE CLI (non-worker)
# ============================================================================

def collect_input_files(input_path):
    """Collect supported audio files from a file or folder path."""
    if not input_path:
        return []
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        return []
    supported_ext = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')
    files = []
    for name in os.listdir(input_path):
        if name.lower().endswith(supported_ext):
            files.append(os.path.join(input_path, name))
    return sorted(files)


def inference_cli():
    """Run separation from CLI using utils.settings.parse_args_inference()."""
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--threads", type=int, default=0,
                              help="CPU threads (0 = auto-detect physical cores)")
    extra_parser.add_argument("--precision", type=str, choices=['high', 'medium'], default='high',
                              help="CPU matmul precision: high (default) or medium (faster)")
    extra_parser.add_argument("--stems", type=str, default=None,
                              help="Comma-separated list of stems to output (e.g., 'vocals,drums')")
    extra_parser.add_argument("--model", "-m", default=None,
                              help="Model name from models.json registry")
    extra_parser.add_argument("--models-dir", default=None,
                              help="External models directory (contains models.json)")
    extra_parser.add_argument("--list-models", action="store_true",
                              help="List available models in registry and exit")
    extra_parser.add_argument("--init-registry", action="store_true",
                              help="Initialize models.json in models-dir and exit")
    extra_parser.add_argument("--overlap", type=int, default=None,
                              help="Overlap factor (1=none, 2=50%, 4=75%)")
    extra_parser.add_argument("--batch-size", type=int, default=None,
                              help="Batch size for inference")
    extra_parser.add_argument("--fast", action="store_true",
                              help="Use vectorized chunking (faster, experimental)")
    extra_parser.add_argument("--two-stems", type=str, default=None,
                              help="Two-stems mode: output stem + no_stem")
    extra_parser.add_argument("--ensemble", type=str, default=None,
                              help="Comma-separated model list to ensemble")
    extra_parser.add_argument("--ensemble-type", default="avg_wave",
                              choices=["avg_wave", "median_wave", "min_wave", "max_wave",
                                       "avg_fft", "median_fft", "min_fft", "max_fft"],
                              help="Ensemble algorithm")
    extra_parser.add_argument("--ensemble-weights", type=str, default=None,
                              help="Comma-separated ensemble weights")

    if any(flag in sys.argv for flag in ("-h", "--help")):
        try:
            from utils.settings import parse_args_inference
            parse_args_inference(None)
        except Exception:
            pass
        print("\nAdditional BS-RoFormer flags:")
        print("  --threads N        CPU threads (0 = auto-detect physical cores)")
        print("  --precision X      Matmul precision: high (default) or medium")
        print("  --stems A,B        Output only specific stems")
        print("  --model NAME       Model name from models.json registry")
        print("  --models-dir PATH  External models directory")
        print("  --list-models      List available models and exit")
        print("  --init-registry    Initialize models.json in models-dir and exit")
        print("  --overlap N        Overlap factor (1/2/4)")
        print("  --batch-size N     Inference batch size")
        print("  --fast             Vectorized chunking (experimental)")
        print("  --two-stems NAME   Output stem + no_stem")
        print("  --ensemble M1,M2   Ensemble models")
        print("  --ensemble-type X  Ensemble algorithm")
        print("  --ensemble-weights W1,W2  Ensemble weights")
        return 0

    extra_args, remaining_args = extra_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args
    if extra_args.init_registry:
        saved_path, created = init_models_registry(extra_args.models_dir)
        if created:
            print(f"Created models.json at: {saved_path}")
        else:
            print(f"models.json already exists at: {saved_path}")
        return 0

    if extra_args.list_models:
        models = load_models_registry(extra_args.models_dir)
        base = extra_args.models_dir if extra_args.models_dir else PROJECT_ROOT
        if models:
            print("Available models:")
            for name, info in models.items():
                checkpoint_path = os.path.join(base, info.get("checkpoint", ""))
                installed = "[OK]" if os.path.exists(checkpoint_path) else "[--]"
                stems = ", ".join(info.get("stems", []))
                desc = info.get("description", "")
                print(f"  {installed} {name}: {desc}")
                if stems:
                    print(f"      Stems: {stems}")
        else:
            print("No models found in registry.")
        return 0

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    for dev_path in [BSROFORMER_DIR, BSROFORMER_FROZEN_DIR]:
        if dev_path in sys.path:
            sys.path.remove(dev_path)
            sys.path.append(dev_path)
    # Ensure we use the shared_runtime utils package, not dev repo
    if "utils.settings" in sys.modules:
        del sys.modules["utils.settings"]
    if "utils" in sys.modules:
        utils_mod = sys.modules["utils"]
        utils_path = getattr(utils_mod, "__file__", "") or ""
        if "shared_runtime" not in utils_path:
            del sys.modules["utils"]
    from utils.settings import parse_args_inference
    args = parse_args_inference(None)
    args.threads = extra_args.threads
    args.precision = extra_args.precision
    args.stems = extra_args.stems
    args.model = extra_args.model
    args.models_dir = extra_args.models_dir
    args.overlap = extra_args.overlap
    args.batch_size = extra_args.batch_size
    args.fast = extra_args.fast
    args.two_stems = extra_args.two_stems
    args.ensemble = extra_args.ensemble
    args.ensemble_type = extra_args.ensemble_type
    args.ensemble_weights = extra_args.ensemble_weights
    input_files = collect_input_files(args.input_folder)
    if not input_files:
        print("Error: --input_folder must be a file or folder with audio files", file=sys.stderr)
        return 2
    if not args.model and not args.start_check_point and not args.ensemble:
        print("Error: provide --model or --start_check_point for inference mode", file=sys.stderr)
        return 2

    output_dir = args.store_dir or "output"
    os.makedirs(output_dir, exist_ok=True)

    # Map settings CLI to worker model loader
    model_type = args.model_type
    config_path = args.config_path or None
    checkpoint_path = args.start_check_point

    import torch

    # Threading and precision settings (CPU path)
    if args.threads == 0:
        num_threads = get_physical_cores()
    else:
        num_threads = max(1, int(args.threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(2, num_threads // 2))
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

    if hasattr(torch, 'set_float32_matmul_precision') and args.precision == 'medium':
        torch.set_float32_matmul_precision('medium')

    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    output_format = "flac" if args.flac_file else "wav"

    # Parse stems selection
    selected_stems = None
    if args.stems:
        selected_stems = [s.strip() for s in args.stems.split(",") if s.strip()]

    # Ensemble mode
    if args.ensemble:
        from utils.ensemble import average_waveforms
        import tempfile
        import shutil
        import librosa
        import soundfile as sf
        import numpy as np

        models = [m.strip() for m in args.ensemble.split(",") if m.strip()]
        weights = [float(w) for w in args.ensemble_weights.split(",")] if args.ensemble_weights else [1.0] * len(models)
        if len(weights) != len(models):
            print("Error: Number of weights must match number of models", file=sys.stderr)
            return 2

        base_temp_dir = tempfile.mkdtemp(prefix="mss_ensemble_")
        try:
            for model_name in models:
                model_out_dir = os.path.join(base_temp_dir, model_name)
                model, config, model_info = load_model_impl(model_name, args.models_dir)
                for input_path in input_files:
                    separate_impl(
                        model, config, input_path, model_out_dir, model_info,
                        device=device,
                        overlap=args.overlap,
                        batch_size=args.batch_size,
                        use_tta=args.use_tta,
                        output_format="wav",
                        pcm_type="FLOAT",
                        extract_instrumental=args.extract_instrumental,
                        selected_stems=selected_stems,
                        two_stems=args.two_stems,
                        use_fast=args.fast
                    )

            for input_path in input_files:
                basename = os.path.splitext(os.path.basename(input_path))[0]
                final_out_subdir = os.path.join(output_dir, basename)
                os.makedirs(final_out_subdir, exist_ok=True)

                first_model_dir = os.path.join(base_temp_dir, models[0], basename)
                if not os.path.exists(first_model_dir):
                    continue

                stems = [f.replace('.wav', '') for f in os.listdir(first_model_dir) if f.endswith('.wav')]
                for stem in stems:
                    waveforms = []
                    valid_weights = []
                    for i, model_name in enumerate(models):
                        stem_path = os.path.join(base_temp_dir, model_name, basename, f"{stem}.wav")
                        if os.path.exists(stem_path):
                            wav, sr = librosa.load(stem_path, sr=None, mono=False)
                            if len(wav.shape) == 1:
                                wav = np.stack([wav, wav], axis=0)
                            waveforms.append(wav)
                            valid_weights.append(weights[i])
                    if waveforms:
                        min_len = min(w.shape[1] for w in waveforms)
                        waveforms = [w[:, :min_len] for w in waveforms]
                        merged_wav = average_waveforms(np.array(waveforms), valid_weights, args.ensemble_type)
                        out_path = os.path.join(final_out_subdir, f"{stem}.{output_format}")
                        sf.write(out_path, merged_wav.T, sr, subtype=args.pcm_type if output_format == "wav" else None)

        finally:
            shutil.rmtree(base_temp_dir, ignore_errors=True)

        return 0

    try:
        if args.model:
            model, config, model_info = load_model_impl(args.model, args.models_dir)
        else:
            model, config, model_info = load_model_from_path(
                checkpoint_path, config_path, model_type
            )
    except Exception as e:
        print(f"Error: failed to load model: {e}", file=sys.stderr)
        return 2

    for input_path in input_files:
        try:
            separate_impl(
                model, config, input_path, output_dir, model_info,
                device=device,
                overlap=args.overlap,
                batch_size=args.batch_size,
                use_tta=args.use_tta,
                output_format=output_format,
                pcm_type=args.pcm_type,
                extract_instrumental=args.extract_instrumental,
                selected_stems=selected_stems,
                two_stems=args.two_stems,
                use_fast=args.fast
            )
            print(f"Done: {input_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}", file=sys.stderr)
            return 1

    return 0


# ============================================================================
# WORKER MODE - SHUTDOWN HANDLING
# ============================================================================

import signal
import threading
import select

# Global shutdown flag
_shutdown_requested = threading.Event()

def _cleanup_and_exit(model_ref, torch_module):
    """Cleanup GPU memory before exit"""
    if model_ref[0] is not None:
        del model_ref[0]
        model_ref[0] = None
        if torch_module and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

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
            import msvcrt
            while not _shutdown_requested.is_set():
                # Check if stdin has data or is closed
                try:
                    # Try to peek - if parent died, this may raise or return empty
                    if sys.stdin.closed:
                        _shutdown_requested.set()
                        break
                    # Small sleep to avoid busy-waiting
                    _shutdown_requested.wait(0.5)
                except Exception:
                    _shutdown_requested.set()
                    break
        else:
            # Unix: use select
            while not _shutdown_requested.is_set():
                try:
                    readable, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if readable:
                        # Check if EOF
                        if sys.stdin.closed:
                            _shutdown_requested.set()
                            break
                except Exception:
                    _shutdown_requested.set()
                    break
    except Exception:
        _shutdown_requested.set()


def _setup_shutdown_handlers():
    """Install signal handlers for graceful shutdown"""
    # Handle SIGTERM (kill command) and SIGINT (Ctrl+C)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    
    # Windows-specific: handle SIGBREAK (Ctrl+Break)
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, _signal_handler)
    
    # Start stdin monitor thread (detects parent process death)
    monitor = threading.Thread(target=_stdin_monitor, daemon=True)
    monitor.start()


# ============================================================================
# WORKER MODE
# ============================================================================

def worker_mode(models_dir=None, device="cpu", configs_dir=None):
    """
    Persistent worker mode for Node.js integration.
    
    Accepts JSON commands via stdin, returns JSON responses via stdout.
    Handles graceful shutdown on:
    - SIGTERM/SIGINT signals
    - stdin closure (parent process died)
    - {"cmd": "exit"} command
    
    Args:
        models_dir: Directory containing models.json and model weights
        device: cuda, cpu, or mps
        configs_dir: Directory containing config files (fallback for missing configs)
    """
    global CONFIGS_DIR
    CONFIGS_DIR = configs_dir
    
    # Install shutdown handlers
    _setup_shutdown_handlers()
    
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
    
    # State - use list for mutable reference in cleanup
    model = None
    config = None
    model_info = None
    current_model_name = None
    model_ref = [None]  # Mutable reference for cleanup handler
    
    # Model cache for keeping multiple models in GPU memory
    # This enables fast switching between models (e.g., bsroformer_4stem + drumsep)
    MAX_CACHED_MODELS = 2  # Limit to prevent OOM (reduced from 3 for better performance)
    model_cache = {}  # {model_identifier: (model, config, model_info)}
    
    def get_cached_model(identifier):
        """Get a model from cache if available"""
        return model_cache.get(identifier)
    
    def add_to_cache(identifier, model_tuple):
        """Add model to cache, evict oldest if full"""
        nonlocal model_cache
        if len(model_cache) >= MAX_CACHED_MODELS:
            # Evict oldest (first inserted)
            oldest_key = next(iter(model_cache))
            model_cache.pop(oldest_key)  # Remove from cache, GC handles cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            send_json({"status": "cache_evicted", "model": os.path.basename(oldest_key)})
        model_cache[identifier] = model_tuple
    
    def clear_cache():
        """Clear all cached models"""
        nonlocal model_cache
        # Simply clear the dict - Python GC will handle model cleanup
        model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Main job loop - checks shutdown flag on each iteration
    while not _shutdown_requested.is_set():
        try:
            # Use readline with timeout check
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
                models = load_models_registry(models_dir)
                send_json({"status": "models", "models": list(models.keys())})
            
            elif cmd == "load_model":
                model_name = job.get("model")
                model_path = job.get("model_path")  # Direct checkpoint path
                config_path = job.get("config_path")  # Direct config path
                model_type = job.get("model_type")  # bs_roformer, mel_band_roformer, etc.
                
                if not model_name and not model_path:
                    send_json({"status": "error", "message": "No model or model_path specified"})
                    continue
                
                try:
                    load_identifier = model_path or model_name
                    
                    # Check cache first
                    cached = get_cached_model(load_identifier)
                    if cached:
                        model, config, model_info = cached
                        model_ref[0] = model
                        current_model_name = load_identifier
                        send_json({
                            "status": "model_loaded",
                            "model": os.path.basename(load_identifier),
                            "stems": model_info.get("stems", []),
                            "cached": True
                        })
                        continue
                    
                    send_json({"status": "loading_model", "model": os.path.basename(load_identifier)})
                    
                    # Load by direct path or by registry name
                    if model_path:
                        model, config, model_info = load_model_from_path(
                            model_path, config_path, model_type
                        )
                    else:
                        model, config, model_info = load_model_impl(model_name, models_dir)
                    
                    model = model.to(torch_device)
                    model_ref[0] = model  # Keep reference for cleanup
                    current_model_name = load_identifier
                    
                    # Add to cache
                    add_to_cache(load_identifier, (model, config, model_info))
                    
                    send_json({
                        "status": "model_loaded",
                        "model": os.path.basename(load_identifier),
                        "stems": model_info.get("stems", []),
                        "cached": False
                    })
                    
                except Exception as e:
                    send_json({"status": "error", "message": f"Failed to load model: {str(e)}"})
            
            elif cmd == "separate":
                input_path = job.get("input")
                output_dir = job.get("output_dir") or job.get("output", "output")
                model_name = job.get("model")
                model_path = job.get("model_path")  # Direct checkpoint path
                config_path = job.get("config_path")  # Direct config path
                model_type = job.get("model_type")  # Optional: bs_roformer, mel_band_roformer, etc.
                overlap = job.get("overlap")
                batch_size = job.get("batch_size")
                pcm_type = job.get("pcm_type", "FLOAT")
                two_stems = job.get("two_stems")
                use_fast = job.get("use_fast", False)
                
                if not input_path:
                    send_json({"status": "error", "message": "No input file specified"})
                    continue
                
                if not os.path.exists(input_path):
                    send_json({"status": "error", "message": f"Input file not found: {input_path}"})
                    continue
                
                # Load model if needed (with cache support)
                # Priority: model_path (direct) > model (registry name)
                if model_path:
                    # Direct path loading - check cache first
                    if model is None or current_model_name != model_path:
                        try:
                            cached = get_cached_model(model_path)
                            if cached:
                                model, config, model_info = cached
                                model_ref[0] = model
                                current_model_name = model_path
                            else:
                                send_json({"status": "loading_model", "model": os.path.basename(model_path)})
                                
                                model, config, model_info = load_model_from_path(
                                    model_path, config_path, model_type
                                )
                                model = model.to(torch_device)
                                model_ref[0] = model  # Keep reference for cleanup
                                current_model_name = model_path
                                
                                # Add to cache
                                add_to_cache(model_path, (model, config, model_info))
                            
                        except Exception as e:
                            send_json({"status": "error", "message": f"Failed to load model: {str(e)}", "traceback": traceback.format_exc()})
                            continue
                
                elif model is None or (model_name and model_name != current_model_name):
                    # Registry-based loading - check cache first
                    target_model = model_name or "bsrofo_sw"
                    try:
                        cached = get_cached_model(target_model)
                        if cached:
                            model, config, model_info = cached
                            model_ref[0] = model
                            current_model_name = target_model
                        else:
                            send_json({"status": "loading_model", "model": target_model})
                            
                            model, config, model_info = load_model_impl(target_model, models_dir)
                            model = model.to(torch_device)
                            model_ref[0] = model  # Keep reference for cleanup
                            current_model_name = target_model
                            
                            # Add to cache
                            add_to_cache(target_model, (model, config, model_info))
                        
                    except Exception as e:
                        send_json({"status": "error", "message": f"Failed to load model: {str(e)}"})
                        continue
                
                try:
                    # Check if this is an Apollo model (restoration)
                    is_apollo = model_info.get("type") == "apollo"
                    status_msg = "restoring" if is_apollo else "separating"
                    send_json({"status": status_msg, "input": os.path.basename(input_path)})
                    
                    elapsed = separate_impl(
                        model, config, input_path, output_dir, model_info,
                        device=str(torch_device),
                        overlap=overlap,
                        batch_size=batch_size,
                        use_tta=job.get("use_tta", False),
                        output_format=job.get("format", "wav"),
                        pcm_type=pcm_type,
                        extract_instrumental=job.get("extract_instrumental", False),
                        selected_stems=job.get("stems"),
                        two_stems=two_stems,
                        chunk_seconds=job.get("chunk_seconds"),
                        chunk_overlap=job.get("chunk_overlap"),
                        use_fast=use_fast
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
                    "ready": True,
                    "cached_models": list(model_cache.keys()),
                    "cache_size": len(model_cache),
                    "max_cache_size": MAX_CACHED_MODELS
                })
            
            elif cmd == "list_cached":
                # List all models currently in cache
                cached_info = []
                for key, (m, c, info) in model_cache.items():
                    cached_info.append({
                        "identifier": key,
                        "name": os.path.basename(key),
                        "stems": info.get("stems", []),
                        "active": key == current_model_name
                    })
                send_json({
                    "status": "cached_models",
                    "models": cached_info,
                    "count": len(model_cache),
                    "max": MAX_CACHED_MODELS
                })
            
            elif cmd == "preload_model":
                # Preload a model into cache without making it active
                model_name = job.get("model")
                model_path = job.get("model_path")
                config_path = job.get("config_path")
                model_type = job.get("model_type")
                
                if not model_name and not model_path:
                    send_json({"status": "error", "message": "No model or model_path specified"})
                    continue
                
                try:
                    load_identifier = model_path or model_name
                    
                    # Check if already cached
                    if get_cached_model(load_identifier):
                        send_json({
                            "status": "model_preloaded",
                            "model": os.path.basename(load_identifier),
                            "already_cached": True
                        })
                        continue
                    
                    send_json({"status": "preloading_model", "model": os.path.basename(load_identifier)})
                    
                    # Load the model
                    if model_path:
                        new_model, new_config, new_info = load_model_from_path(
                            model_path, config_path, model_type
                        )
                    else:
                        new_model, new_config, new_info = load_model_impl(model_name, models_dir)
                    
                    new_model = new_model.to(torch_device)
                    
                    # Add to cache (but don't make it the active model)
                    add_to_cache(load_identifier, (new_model, new_config, new_info))
                    
                    send_json({
                        "status": "model_preloaded",
                        "model": os.path.basename(load_identifier),
                        "stems": new_info.get("stems", []),
                        "cached_count": len(model_cache)
                    })
                    
                except Exception as e:
                    send_json({"status": "error", "message": f"Failed to preload model: {str(e)}"})
            
            elif cmd == "clear_cache":
                # Clear all cached models
                count = len(model_cache)
                clear_cache()
                model = None
                config = None
                model_info = None
                current_model_name = None
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    if "--worker" in sys.argv:
        parser = argparse.ArgumentParser(description='BS-RoFormer Worker (Shared Runtime)')
        parser.add_argument('--worker', action='store_true', help='Run in worker mode')
        parser.add_argument('--models-dir', type=str, default=None, help='Models directory (contains models.json)')
        parser.add_argument('--configs-dir', type=str, default=None, help='Configs directory (fallback for config files)')
        parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'],
                            help='Device to use')
        parser.add_argument('--ensemble', action='store_true',
                            help='Run ensemble utility and exit')
        parser.add_argument('--files', type=str, nargs='+',
                            help='Input audio files to ensemble')
        parser.add_argument('--type', type=str, default='avg_wave',
                            help='Ensemble type: avg_wave, median_wave, min_wave, max_wave, '
                                 'avg_fft, median_fft, min_fft, max_fft')
        parser.add_argument('--weights', type=float, nargs='+',
                            help='Weights for ensemble (must match number of files)')
        parser.add_argument('--output', type=str, default='res.wav',
                            help='Output wav path for ensemble')
        
        args = parser.parse_args()
        
        if args.ensemble:
            if not args.files:
                print("Error: --files is required when using --ensemble", file=sys.stderr)
                sys.exit(2)
            from utils.ensemble import ensemble_files
            ensemble_args = ["--files", *args.files, "--type", args.type, "--output", args.output]
            if args.weights:
                ensemble_args += ["--weights", *[str(w) for w in args.weights]]
            ensemble_files(ensemble_args)
            sys.exit(0)
        elif args.worker:
            sys.exit(worker_mode(args.models_dir, args.device, args.configs_dir))
        else:
            print("Usage: python bsroformer_worker.py --worker [--models-dir /path] [--device cuda|cpu]")
            print("   or: python bsroformer_worker.py --ensemble --files a.wav b.wav --output res.wav")
            sys.exit(1)
    else:
        sys.exit(inference_cli())


if __name__ == '__main__':
    main()
