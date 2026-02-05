"""
cx_Freeze setup for Doctor Sample Unit (DSU) - Shared Runtime
Based on working builds from audio-separator-cxfreeze and bs-roformer-freeze-repo

Usage:
    python setup.py build_exe

Or with cxfreeze CLI:
    cxfreeze main.py --target-dir=dist/dsu --target-name=dsu ...
"""

import sys
import os
import site
from cx_Freeze import setup, Executable

# Increase recursion limit for torch
sys.setrecursionlimit(5000)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# =============================================================================
# Find llvmlite.libs folder (contains MSVC runtime DLLs needed on Windows)
# =============================================================================
LLVMLITE_LIBS = None
if sys.platform == "win32":
    for site_pkg in site.getsitepackages():
        libs_path = os.path.join(site_pkg, "llvmlite.libs")
        if os.path.isdir(libs_path):
            LLVMLITE_LIBS = libs_path
            break
    # Also check in the venv
    if not LLVMLITE_LIBS:
        venv_libs = os.path.join(SCRIPT_DIR, "build_env", "Lib", "site-packages", "llvmlite.libs")
        if os.path.isdir(venv_libs):
            LLVMLITE_LIBS = venv_libs

# =============================================================================
# TORCH PACKAGES - All torch submodules that need to be explicitly included
# This prevents partial imports that cause "bad magic number" errors
# =============================================================================
TORCH_PACKAGES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.modules",
    "torch.nn.utils",
    "torch.utils",
    "torch.utils.data",
    "torch._C",
    "torch._jit_internal",
    "torch.package",
    "torch.package._mangling",
    "torch.package.analyze",
    "torch.package.package_exporter",
    "torch.functional",
    "torch.autograd",
    "torch.cuda",
    "torch.cuda.amp",
    "torch.amp",
    "torch.backends",
    "torch.backends.mkl",
    "torch.backends.mkldnn",
    "torch.backends.openmp",
    "torch.backends.cudnn",
    "torch.backends.cuda",
    "torch.fft",
    "torch.linalg",
    "torch.special",
    "torch.sparse",
    "torch.distributions",
    "torch.optim",
    "torch.serialization",
    "torch.onnx",
    "torch.profiler",
    "torch.ao",
    "torch.jit",
    "torchaudio",
]

# =============================================================================
# OTHER PACKAGES needed by all workers
# =============================================================================
OTHER_PACKAGES = [
    # Core audio processing
    "audio_separator",
    "demucs",
    "demucs.htdemucs",
    "soundfile",
    "librosa",
    "scipy",
    "scipy.signal",
    "scipy.fft",
    "samplerate",
    "pydub",
    "resampy",
    
    # ML and inference
    "numpy",
    "onnx",
    "onnxruntime",
    "einops",
    "julius",
    "diffq",
    
    # Configuration
    "yaml",
    "omegaconf",
    "ml_collections",
    
    # Model dependencies
    "beartype",
    "rotary_embedding_torch",
    "numba",
    "llvmlite",
    "pytorch_lightning",
    "huggingface_hub",
    
    # Network (required by huggingface_hub)
    "requests",
    "tqdm",
    
    # Standard library modules needed by dependencies
    "urllib",
    "urllib.request",
    "urllib.parse", 
    "urllib.error",
    "http",
    "http.client",
    "email",
    "email.mime",
    "email.mime.text",
    "importlib",
    "importlib.metadata",
    "unittest",  # Required by torch._dispatch.python
    
    # Project packages
    "models",
    "models.bs_roformer",
    "models.bandit",
    "models.bandit.core",
    "models.scnet",
    "models.scnet_unofficial",
    "utils",
    "apollo",
    "apollo.look2hear",
    "apollo.look2hear.models",
]

ALL_PACKAGES = TORCH_PACKAGES + OTHER_PACKAGES

# =============================================================================
# CUDA-only packages (only available on Windows/Linux with NVIDIA GPU)
# These are optional and will be skipped if not installed
# =============================================================================
CUDA_OPTIONAL_PACKAGES = ["sageattention", "triton"]

for pkg in CUDA_OPTIONAL_PACKAGES:
    try:
        __import__(pkg)
        ALL_PACKAGES.append(pkg)
    except ImportError:
        pass  # Skip if not installed (e.g., on macOS)

# =============================================================================
# EXCLUDES - Modules not needed at runtime
# NOTE: unittest is required by torch, don't exclude it
# =============================================================================
EXCLUDES = [
    "tkinter",
    # "unittest",  # Required by torch
    "test",
    "tests",
    "distutils",
    "setuptools",
    "pip",
    "wheel",
    "pkg_resources",
    "pydoc_data",
    "curses",
    "IPython",
    "jupyter",
    "notebook",
    "matplotlib.backends.backend_qt5agg",
    "PyQt5",
    "PySide2",
    "cx_Freeze",
]

# =============================================================================
# Build include_files list
# =============================================================================
include_files = []

# Include llvmlite.libs folder for Windows (contains msvcp140-*.dll)
if sys.platform == "win32" and LLVMLITE_LIBS:
    include_files.append((LLVMLITE_LIBS, "lib/llvmlite.libs"))

# Only include configs folder (YAML files needed at runtime)
# Note: models, utils, apollo are Python packages bundled via 'packages' list
configs_path = os.path.join(SCRIPT_DIR, "configs")
if os.path.exists(configs_path):
    include_files.append((configs_path, "configs"))

# =============================================================================
# BUILD OPTIONS - Critical settings from working builds
# =============================================================================
build_options = {
    "packages": ALL_PACKAGES,
    "excludes": EXCLUDES,
    "include_files": include_files,
    
    # CRITICAL: Don't compress .pyc files - torch doesn't like it
    "zip_include_packages": [],
    "zip_exclude_packages": "*",  # Don't zip anything - prevents path issues
    
    # Bytecode optimization: optimize=1 removes assertions (safe for torch)
    # optimize=2 also removes docstrings (may break some torch introspection)
    # Set via environment variable: DSU_BUILD_OPTIMIZE=1 or 2 (default: 0 for safety)
    "optimize": int(os.environ.get("DSU_BUILD_OPTIMIZE", "0")),
    
    # Build output directory (allow override to avoid Windows file locks)
    "build_exe": os.environ.get(
        "DSU_BUILD_EXE_DIR",
        os.path.join(SCRIPT_DIR, "dist", "dsu")
    ),
    
    # CRITICAL: Replace paths to make bundle portable
    "replace_paths": [("*", "")],  # Remove all absolute paths
}

# =============================================================================
# EXECUTABLES
# =============================================================================
base = "Console" if sys.platform == "win32" else None

executables = [
    Executable(
        os.path.join(SCRIPT_DIR, "workers", "demucs_worker.py"),
        base=base,
        target_name="dsu-demucs",
    ),
    Executable(
        os.path.join(SCRIPT_DIR, "workers", "bsroformer_worker.py"),
        base=base,
        target_name="dsu-bsroformer",
    ),
    Executable(
        os.path.join(SCRIPT_DIR, "workers", "audio_separator_worker.py"),
        base=base,
        target_name="dsu-audio-separator",
    ),
]

# =============================================================================
# SETUP
# =============================================================================
setup(
    name="dsu",
    version="1.4.2",
    description="Doctor Sample Unit - Shared Audio Processing Runtime",
    options={"build_exe": build_options},
    executables=executables,
)
