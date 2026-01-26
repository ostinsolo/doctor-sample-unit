#!/usr/bin/env python3
"""
Build Doctor Sample Unit (DSU) launcher executables

Creates branded .exe files that show "Doctor Sample Unit" in Task Manager.
"""

import os
import sys
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Application branding
APP_NAME = "Doctor Sample Unit"
APP_SHORT_NAME = "DSU"
APP_VERSION = "1.0.0"
APP_COMPANY = "Doctor Sample Unit"
APP_COPYRIGHT = "Copyright (c) 2026 Doctor Sample Unit"

def build_all():
    print("Building Doctor Sample Unit (DSU) Launchers...")
    print("=" * 60)
    
    # Check cx_Freeze
    try:
        from cx_Freeze import setup, Executable
    except ImportError:
        print("ERROR: cx_Freeze not installed")
        return 1
    
    # Output directory
    output_dir = os.path.join(SCRIPT_DIR, 'bin')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    launchers_dir = os.path.join(SCRIPT_DIR, 'launchers')
    
    # Launchers to build with DSU branding
    launchers = [
        {
            'script': 'bsroformer_launcher.py',
            'exe_name': 'dsu-bsroformer',
            'description': f'{APP_NAME} - BS-RoFormer Worker',
            'internal_name': 'DSU-BSRoFormer',
        },
        {
            'script': 'audio_separator_launcher.py',
            'exe_name': 'dsu-audio-separator',
            'description': f'{APP_NAME} - Audio Separator Worker',
            'internal_name': 'DSU-AudioSeparator',
        },
        {
            'script': 'demucs_launcher.py',
            'exe_name': 'dsu-demucs',
            'description': f'{APP_NAME} - Demucs Worker',
            'internal_name': 'DSU-Demucs',
        },
    ]
    
    # Build options - exclude everything heavy
    build_exe_options = {
        "excludes": [
            "torch", "torchaudio", "torchvision", "numpy", "scipy",
            "librosa", "soundfile", "audio_separator", "demucs",
            "onnx", "onnxruntime", "PIL", "matplotlib", "pandas",
            "tkinter", "unittest", "test", "email", "html", "http",
            "xml", "multiprocessing", "concurrent", "asyncio",
            "sqlite3", "ssl", "cx_Freeze", "setuptools",
        ],
        "build_exe": output_dir,
    }
    
    executables = []
    for launcher in launchers:
        script_path = os.path.join(launchers_dir, launcher['script'])
        if os.path.exists(script_path):
            # Create executable with Windows metadata
            exe = Executable(
                script_path,
                target_name=f"{launcher['exe_name']}.exe",
                base=None,  # Console application
                # Windows version info - this shows in Task Manager
                copyright=APP_COPYRIGHT,
                # These require cx_Freeze 6.x+
            )
            executables.append(exe)
            print(f"  Adding: {launcher['script']} -> {launcher['exe_name']}.exe")
        else:
            print(f"  WARNING: {script_path} not found")
    
    if not executables:
        print("ERROR: No launcher scripts found")
        return 1
    
    # Run setup
    print("\nBuilding with cx_Freeze...")
    print(f"  App Name: {APP_NAME}")
    print(f"  Version: {APP_VERSION}")
    
    # Save original argv
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
    finally:
        sys.argv = orig_argv
    
    # Check results
    print("\n" + "=" * 60)
    print("BUILD RESULTS")
    print("=" * 60)
    
    success_count = 0
    for launcher in launchers:
        exe_path = os.path.join(output_dir, f"{launcher['exe_name']}.exe")
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"  {launcher['exe_name']}.exe: {size_mb:.1f} MB")
            success_count += 1
        else:
            print(f"  {launcher['exe_name']}.exe: NOT FOUND")
    
    print(f"\nBuilt: {success_count}/{len(launchers)}")
    print(f"Output: {output_dir}")
    
    # Create version info file for reference
    version_file = os.path.join(output_dir, 'VERSION.txt')
    with open(version_file, 'w') as f:
        f.write(f"{APP_NAME}\n")
        f.write(f"Version: {APP_VERSION}\n")
        f.write(f"\nExecutables:\n")
        for launcher in launchers:
            f.write(f"  - {launcher['exe_name']}.exe: {launcher['description']}\n")
    
    return 0 if success_count == len(launchers) else 1


if __name__ == '__main__':
    sys.exit(build_all())
