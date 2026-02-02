#!/usr/bin/env python3
"""
Build Doctor Sample Unit (DSU) - Properly Frozen Shared Runtime

Creates frozen executables that SHARE dependencies:
- dsu-demucs.exe      - Demucs worker
- dsu-bsroformer.exe  - BS-RoFormer worker  
- dsu-audio-separator.exe - Audio Separator worker
- lib/                (~2-4GB) - SHARED frozen dependencies (torch, numpy, etc.)

All executables share the same lib/ folder - no duplication.

Based on working builds from:
- audio-separator-cxfreeze
- bs-roformer-freeze-repo  
- demucs-cxfreeze
"""

import os
import sys
import site
import shutil

# Increase recursion limit for cx_Freeze with large packages like torch
sys.setrecursionlimit(5000)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Windows: Add FFmpeg bin to DLL search path before any torchcodec import.
# Torchcodec finds ffmpeg via shutil.which() and adds its dir - but we add it
# early so PATH from the caller (CI, simulate_windows_build.bat) is respected.
# gyan.dev full-shared has all DLLs in one folder; no conda/MinGW needed.
# =============================================================================
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    ff = shutil.which("ffmpeg")
    if ff:
        try:
            os.add_dll_directory(os.path.dirname(ff))
        except OSError:
            pass

# Application branding
APP_NAME = "Doctor Sample Unit"
APP_VERSION = "1.3.9"
APP_COMPANY = "Doctor Sample Unit"
APP_COPYRIGHT = "Copyright (c) 2026 Doctor Sample Unit"

# =============================================================================
# CRITICAL: All torch submodules that need to be explicitly included
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
# All other packages needed by all workers
# =============================================================================
AUDIO_PACKAGES = [
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
    # onnx and onnxruntime are optional (not installed to improve performance)
    # They will be added to OPTIONAL_PACKAGES below if available
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
    # pytorch_lightning and huggingface_hub are optional (used in training code, not inference)
    # They will be added to OPTIONAL_PACKAGES below if available
    # "pytorch_lightning",
    # "huggingface_hub",
    
    # Network (required by huggingface_hub and other packages)
    "requests",
    "tqdm",
    
    # Build/package utilities (required by librosa and other packages)
    "pkg_resources",  # CRITICAL: Required by librosa.core.intervals
]

# =============================================================================
# Standard library modules that MUST be included (dependencies need them)
# NOTE: Don't exclude these - torch, requests, etc. need them
# =============================================================================
STDLIB_PACKAGES = [
    "codecs",  # Required for filesystem encoding
    "zlib",  # CRITICAL: Required for zipimport (library.zip decompression)
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
]

# =============================================================================
# Project-specific packages (our models and utils)
# =============================================================================
PROJECT_PACKAGES = [
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

# Combine all packages
ALL_PACKAGES = TORCH_PACKAGES + AUDIO_PACKAGES + STDLIB_PACKAGES + PROJECT_PACKAGES

# =============================================================================
# Optional packages (skipped if not installed or fails to load)
# - CUDA: sageattention, triton (Windows/Linux NVIDIA)
# - torchcodec: torchaudio save/load (Demucs). If missing/fails, Demucs uses
#   soundfile fallback. torchcodec can raise RuntimeError on Windows when
#   FFmpeg shared DLLs are not findable (even if the package is installed).
# =============================================================================
# Optional packages (only included if available)
# onnx/onnxruntime: Not installed by default (removed for performance)
# They're optional dependencies of audio-separator but not required
# huggingface_hub: Used by Apollo but not required for core functionality
# =============================================================================
OPTIONAL_PACKAGES = ["sageattention", "triton", "torchcodec", "onnx", "onnxruntime", "huggingface_hub", "pytorch_lightning"]

for pkg in OPTIONAL_PACKAGES:
    try:
        __import__(pkg)
        ALL_PACKAGES.append(pkg)
        if pkg in ("sageattention", "triton"):
            kind = "CUDA"
        elif pkg in ("onnx", "onnxruntime"):
            kind = "ONNX (optional)"
        elif pkg == "huggingface_hub":
            kind = "Apollo (optional)"
        elif pkg == "pytorch_lightning":
            kind = "Training (optional)"
        else:
            kind = "torchcodec"
        print(f"  Including optional package ({kind}): {pkg}")
    except (ImportError, RuntimeError, OSError) as e:
        print(f"  Skipping optional package (not available): {pkg} - {type(e).__name__}")

# =============================================================================
# Modules to exclude (reduce size, not needed at runtime)
# NOTE: Be conservative - only exclude what's truly unnecessary
# =============================================================================
EXCLUDES = [
    "tkinter",
    "test",
    "tests",
    # NOTE: sklearn is NOT excluded - it's required by librosa (scikit-learn>=0.19.1)
    # We handle the missing libomp.dylib by creating a dummy file before build
    "distutils",
    "setuptools",
    "pip",
    "wheel",
    # NOTE: pkg_resources is NOT excluded - it's required by librosa
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


def find_llvmlite_libs():
    """Find llvmlite.libs folder (contains MSVC runtime DLLs needed on Windows)"""
    if sys.platform != "win32":
        return None
    
    # Check site-packages
    for site_pkg in site.getsitepackages():
        libs_path = os.path.join(site_pkg, "llvmlite.libs")
        if os.path.isdir(libs_path):
            return libs_path
    
    # Check in the venv
    venv_libs = os.path.join(SCRIPT_DIR, "build_env", "Lib", "site-packages", "llvmlite.libs")
    if os.path.isdir(venv_libs):
        return venv_libs
    
    return None


def copy_llvmlite_libs(output_dir):
    """Copy llvmlite.libs to output directory (required for Windows)"""
    llvmlite_libs = find_llvmlite_libs()
    if not llvmlite_libs:
        print("  WARNING: llvmlite.libs not found")
        return False
    
    dest = os.path.join(output_dir, "lib", "llvmlite.libs")
    print(f"  Copying llvmlite.libs from: {llvmlite_libs}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(llvmlite_libs, dest, dirs_exist_ok=True)
    print(f"  llvmlite.libs copied to: {dest}")
    return True


def build_dsu():
    print("=" * 70)
    print("Building Doctor Sample Unit (DSU) - Frozen Shared Runtime")
    print("=" * 70)
    
    try:
        from cx_Freeze import setup, Executable
    except ImportError:
        print("ERROR: cx_Freeze not installed")
        print("Run: pip install cx-Freeze==6.15.16")
        return 1
    
    # Output directory
    output_dir = os.path.join(SCRIPT_DIR, 'dist', 'dsu')
    if os.path.exists(output_dir):
        print(f"Removing previous build: {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except (OSError, PermissionError) as e:
            print(f"  WARNING: Could not fully remove {output_dir}: {e}")
            print(f"  Attempting to remove lib directory manually...")
            lib_dir = os.path.join(output_dir, 'lib')
            if os.path.exists(lib_dir):
                try:
                    shutil.rmtree(lib_dir)
                except:
                    pass
    
    # Ensure output directory exists (cx_Freeze needs it)
    os.makedirs(output_dir, exist_ok=True)
    
    # Workers are in project root, not in build script directory
    # Go up from scripts/building/py/ to project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    workers_dir = os.path.join(PROJECT_ROOT, 'workers')
    
    # Workers to freeze
    workers = [
        {
            'script': os.path.join(workers_dir, 'demucs_worker.py'),
            'exe_name': 'dsu-demucs',
            'description': f'{APP_NAME} - Demucs Worker',
        },
        {
            'script': os.path.join(workers_dir, 'bsroformer_worker.py'),
            'exe_name': 'dsu-bsroformer',
            'description': f'{APP_NAME} - BS-RoFormer Worker',
        },
        {
            'script': os.path.join(workers_dir, 'audio_separator_worker.py'),
            'exe_name': 'dsu-audio-separator',
            'description': f'{APP_NAME} - Audio Separator Worker',
        },
    ]
    
    # Build include_files list
    include_files = []
    
    # Only include configs folder (YAML files needed at runtime)
    # Note: models, utils, apollo are Python packages and are bundled via 'packages' list
    configs_path = os.path.join(PROJECT_ROOT, "configs")
    if os.path.exists(configs_path):
        include_files.append((configs_path, "configs"))
    # models.json for BS-RoFormer worker (registry; weights/ downloaded separately)
    models_json = os.path.join(PROJECT_ROOT, "models.json")
    if os.path.exists(models_json):
        include_files.append((models_json, "models.json"))
    
    # =============================================================================
    # CRITICAL BUILD OPTIONS - from working builds
    # =============================================================================
    build_exe_options = {
        "packages": ALL_PACKAGES,
        "excludes": EXCLUDES,
        "include_files": include_files,
        
        # CRITICAL: Don't compress .pyc files - torch doesn't like it
        "zip_include_packages": [],
        "zip_exclude_packages": "*",  # Don't zip anything - prevents path issues
        
        # Bytecode optimization: optimize=1 removes assertions (safe for torch)
        # optimize=2 also removes docstrings (may break some torch introspection)
        # For Intel Mac, we can use optimize=1 to speed up build slightly
        # Set via environment variable: DSU_BUILD_OPTIMIZE=1 or 2 (default: 0 for safety)
        "optimize": int(os.environ.get("DSU_BUILD_OPTIMIZE", "0")),
        
        # Build in output directory
        "build_exe": output_dir,
        
        # CRITICAL: Replace paths to make bundle portable
        "replace_paths": [("*", "")],  # Remove all absolute paths
    }
    if sys.platform == "win32":
        build_exe_options["include_msvcr"] = True
    
    # Platform: .exe on Windows, no extension on macOS/Linux
    exe_suffix = ".exe" if sys.platform == "win32" else ""
    base = "Console" if sys.platform == "win32" else None
    
    # Create executables
    executables = []
    for worker in workers:
        if os.path.exists(worker['script']):
            target = f"{worker['exe_name']}{exe_suffix}"
            exe = Executable(
                script=worker['script'],
                target_name=target,
                base=base,
                copyright=APP_COPYRIGHT,
            )
            executables.append(exe)
            print(f"  Adding: {os.path.basename(worker['script'])} -> {target}")
        else:
            print(f"  WARNING: {worker['script']} not found!")
    
    if not executables:
        print("ERROR: No worker scripts found")
        return 1
    
    print()
    print("Building with cx_Freeze (this takes 5-15 minutes)...")
    print(f"  Output: {output_dir}")
    print()
    
    # macOS fix: Create lib directory before build (cx_Freeze needs it for Python library)
    if sys.platform == "darwin":
        lib_dir = os.path.join(output_dir, "lib")
        os.makedirs(lib_dir, exist_ok=True)
        print(f"  Created lib directory: {lib_dir}")
        
        # CRITICAL: Create dummy libomp.dylib if sklearn's version is missing
        # This prevents cx_Freeze from failing when trying to copy sklearn's libomp.dylib
        # (We removed it earlier to prevent OpenMP conflicts)
        # NOTE: We can't import sklearn because it fails without libomp.dylib,
        # so we construct the path directly using sys.prefix (runtime environment)
        try:
            # Construct sklearn path directly (can't import sklearn - it needs libomp.dylib)
            sklearn_base = os.path.join(sys.prefix, "lib", "python3.10", "site-packages", "sklearn")
            sklearn_omp = os.path.join(sklearn_base, ".dylibs", "libomp.dylib")
            
            if not os.path.exists(sklearn_omp):
                # Create .dylibs directory if it doesn't exist
                dylibs_dir = os.path.dirname(sklearn_omp)
                os.makedirs(dylibs_dir, exist_ok=True)
                # Create an empty dummy file (cx_Freeze will copy it but it won't be used)
                with open(sklearn_omp, 'wb') as f:
                    f.write(b'')  # Empty file
                print(f"  Created dummy libomp.dylib for sklearn at: {sklearn_omp}")
                print(f"    (This prevents cx_Freeze copy error - file is empty and won't cause OpenMP conflicts)")
            else:
                print(f"  libomp.dylib already exists at: {sklearn_omp}")
        except Exception as e:
            # If we can't create it, warn but continue (might not be needed)
            print(f"  WARNING: Could not create dummy libomp.dylib: {e}")
    
    # Add project root to Python path so apollo module can be found
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    # Run cx_Freeze build
    orig_argv = sys.argv
    sys.argv = [sys.argv[0], 'build_exe']
    
    try:
        setup(
            name=APP_NAME,
            version=APP_VERSION,
            description=f"{APP_NAME} - Audio Source Separation",
            author=APP_COMPANY,
            options={"build_exe": build_exe_options},
            executables=executables,
        )
    except Exception as e:
        print(f"ERROR during build: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        sys.argv = orig_argv
    
    # =============================================================================
    # Post-build: Fix macOS Python library path issue
    # =============================================================================
    if sys.platform == "darwin":
        lib_dir = os.path.join(output_dir, "lib")
        if not os.path.exists(lib_dir):
            print()
            print("Post-build: Creating lib directory (macOS fix)...")
            os.makedirs(lib_dir, exist_ok=True)
        
        # CRITICAL: Ensure Python library is copied (contains zlib and other built-in modules)
        # Try multiple possible locations for the Python library
        python_lib_sources = [
            os.path.join(sys.prefix, "lib", "python3.10", "site-packages", "cx_Freeze", "bases", "lib", "Python"),
            os.path.join(sys.prefix, "lib", "libpython3.10.dylib"),
            os.path.join(sys.prefix, "Python"),
        ]
        
        python_lib_dest = os.path.join(lib_dir, "Python")
        python_lib_copied = False
        
        for python_lib_source in python_lib_sources:
            if os.path.exists(python_lib_source):
                try:
                    if os.path.isdir(python_lib_source):
                        shutil.copytree(python_lib_source, python_lib_dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(python_lib_source, python_lib_dest)
                    print(f"  Copied Python library to: {python_lib_dest}")
                    python_lib_copied = True
                    break
                except Exception as e:
                    continue
        
        if not python_lib_copied:
            print(f"  WARNING: Could not find/copy Python library (zlib may not work)")
            print(f"  Searched in: {python_lib_sources}")
        
        # CRITICAL: Copy zlib shared library if it exists (needed for library.zip decompression)
        # On macOS venv, zlib often lives in base_prefix (framework Python), not in the venv
        zlib_name = "zlib.cpython-310-darwin.so"
        zlib_candidates = [
            os.path.join(sys.prefix, "lib", "python3.10", "lib-dynload", zlib_name),
            os.path.join(getattr(sys, "base_prefix", sys.prefix), "lib", "python3.10", "lib-dynload", zlib_name),
        ]
        zlib_copied = False
        for zlib_source in zlib_candidates:
            if os.path.exists(zlib_source):
                zlib_dest = os.path.join(lib_dir, zlib_name)
                try:
                    shutil.copy2(zlib_source, zlib_dest)
                    print(f"  Copied zlib shared library to: {zlib_dest}")
                    zlib_copied = True
                    break
                except Exception as e:
                    print(f"  WARNING: Could not copy zlib shared library: {e}")
        if not zlib_copied:
            print(f"  WARNING: zlib shared library not found (tried: {zlib_candidates}); zipimport may fail at runtime")
    
    # =============================================================================
    # Post-build: Copy llvmlite.libs (Windows only)
    # =============================================================================
    if sys.platform == "win32":
        print()
        print("Post-build: Copying llvmlite.libs...")
        copy_llvmlite_libs(output_dir)
    
    # Report results
    print()
    print("=" * 70)
    print("BUILD RESULTS")
    print("=" * 70)
    
    # Check executables
    exe_total = 0
    for worker in workers:
        target = f"{worker['exe_name']}{exe_suffix}"
        exe_path = os.path.join(output_dir, target)
        if os.path.exists(exe_path):
            size_kb = os.path.getsize(exe_path) / 1024
            print(f"  {target}: {size_kb:.0f} KB")
            exe_total += 1
        else:
            print(f"  {target}: NOT FOUND")
    
    # Check lib folder
    lib_dir = os.path.join(output_dir, 'lib')
    if os.path.exists(lib_dir):
        lib_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(lib_dir)
            for f in filenames
        )
        lib_size_gb = lib_size / (1024 * 1024 * 1024)
        print(f"  lib/ folder: {lib_size_gb:.2f} GB (shared dependencies)")
    
    # Total size
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, filenames in os.walk(output_dir)
        for f in filenames
    )
    total_gb = total_size / (1024 * 1024 * 1024)
    print()
    print(f"Total size: {total_gb:.2f} GB")
    print(f"Built: {exe_total}/{len(workers)} executables")
    print(f"Output: {output_dir}")
    
    # Create VERSION.txt
    version_file = os.path.join(output_dir, 'VERSION.txt')
    with open(version_file, 'w') as f:
        f.write(f"{APP_NAME} v{APP_VERSION}\n")
        f.write(f"\nExecutables:\n")
        for worker in workers:
            f.write(f"  - {worker['exe_name']}{exe_suffix}\n")
        f.write(f"\nAll executables share lib/ folder (no duplication)\n")
    
    print()
    print("Next: Test with:")
    d1 = f"{workers[0]['exe_name']}{exe_suffix}"
    d2 = f"{workers[1]['exe_name']}{exe_suffix}"
    sep = "\\" if sys.platform == "win32" else "/"
    print(f"  {output_dir}{sep}{d1} --help")
    print(f"  {output_dir}{sep}{d2} --worker")
    
    return 0 if exe_total == len(workers) else 1


if __name__ == '__main__':
    sys.exit(build_dsu())
