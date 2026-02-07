#!/usr/bin/env python3
"""
CLI entry point for envelope-matched noise reduction.
Denoise logic lives in utils.denoise (bundled with DSU, no separate venv).

Usage:
    python noise_reduction/denoise.py <audio_file> <noise_profile> [output_file]
    python -m utils.denoise <audio_file> <noise_profile> [output_file]
"""
import sys
import os

# Ensure project root is on path when run as script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.denoise import main

if __name__ == "__main__":
    main()
