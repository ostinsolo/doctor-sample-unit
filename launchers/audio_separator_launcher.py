#!/usr/bin/env python3
"""
Audio-Separator Worker Launcher - Thin executable launcher

This is a minimal launcher that:
1. Finds the shared runtime
2. Sets up PYTHONPATH
3. Executes the worker script

NO heavy imports (torch, numpy, etc.) - those come from shared runtime.
"""

import os
import sys
import subprocess

def find_shared_runtime():
    """Find the shared runtime directory"""
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    
    candidates = [
        os.path.join(exe_dir, '..', 'runtime'),
        os.path.join(exe_dir, '..', '..', 'runtime'),
        os.path.join(exe_dir, '..', '..', 'shared_runtime', 'runtime'),
    ]
    
    for candidate in candidates:
        runtime_dir = os.path.normpath(candidate)
        python_exe = os.path.join(runtime_dir, 'Scripts', 'python.exe')
        if os.path.exists(python_exe):
            return runtime_dir
    
    return None

def find_worker_script():
    """Find the worker script"""
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    
    candidates = [
        os.path.join(exe_dir, '..', 'workers', 'audio_separator_worker.py'),
        os.path.join(exe_dir, '..', '..', 'workers', 'audio_separator_worker.py'),
        os.path.join(exe_dir, 'workers', 'audio_separator_worker.py'),
    ]
    
    for candidate in candidates:
        path = os.path.normpath(candidate)
        if os.path.exists(path):
            return path
    
    return None

def main():
    runtime_dir = find_shared_runtime()
    if not runtime_dir:
        print('{"status": "error", "message": "Shared runtime not found"}')
        sys.exit(1)
    
    worker_script = find_worker_script()
    if not worker_script:
        print('{"status": "error", "message": "Worker script not found"}')
        sys.exit(1)
    
    python_exe = os.path.join(runtime_dir, 'Scripts', 'python.exe')
    cmd = [python_exe, worker_script] + sys.argv[1:]
    
    if os.name == 'nt':
        result = subprocess.run(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        sys.exit(result.returncode)
    else:
        os.execv(python_exe, cmd)

if __name__ == '__main__':
    main()
