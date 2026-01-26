#!/usr/bin/env python3
"""
Demucs Thin Launcher - Finds shared runtime and launches worker

This is a minimal launcher that gets compiled with cx_Freeze.
It contains NO heavy imports - just finds python.exe and runs the worker script.

When frozen, this becomes a ~15MB executable that:
1. Finds the shared runtime directory
2. Finds the worker script
3. Executes: runtime/python.exe workers/demucs_worker.py [args]
"""

import os
import sys
import subprocess


def find_runtime_dir():
    """Find the shared runtime directory"""
    if getattr(sys, 'frozen', False):
        # Running as frozen exe
        exe_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search paths relative to exe location
    search_paths = [
        os.path.join(exe_dir, '..', 'runtime'),           # ../runtime
        os.path.join(exe_dir, '..', '..', 'runtime'),     # ../../runtime
        os.path.join(exe_dir, 'runtime'),                  # ./runtime
    ]
    
    for path in search_paths:
        python_exe = os.path.join(path, 'Scripts', 'python.exe')
        if os.path.exists(python_exe):
            return os.path.abspath(path)
    
    return None


def find_worker_script():
    """Find the demucs worker script"""
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_paths = [
        os.path.join(exe_dir, '..', 'workers', 'demucs_worker.py'),
        os.path.join(exe_dir, '..', '..', 'workers', 'demucs_worker.py'),
        os.path.join(exe_dir, 'workers', 'demucs_worker.py'),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None


def main():
    runtime_dir = find_runtime_dir()
    if not runtime_dir:
        print("ERROR: Could not find shared runtime directory", file=sys.stderr)
        print("Expected: runtime/Scripts/python.exe", file=sys.stderr)
        sys.exit(1)
    
    worker_script = find_worker_script()
    if not worker_script:
        print("ERROR: Could not find demucs_worker.py", file=sys.stderr)
        sys.exit(1)
    
    python_exe = os.path.join(runtime_dir, 'Scripts', 'python.exe')
    
    # Build command: python.exe worker_script.py [original args]
    cmd = [python_exe, worker_script] + sys.argv[1:]
    
    # Execute and pass through exit code
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
