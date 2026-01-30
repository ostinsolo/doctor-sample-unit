#!/usr/bin/env python3
"""
Ensure FFmpeg full-shared is available on Windows (torchcodec needs DLLs).
Prints the bin directory path to stdout for the caller to add to PATH.
Exits 0 if OK, 1 on error. Used by simulate_windows_build.bat.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request


def find_ffmpeg_with_dlls():
    """Return path to ffmpeg dir that has avcodec DLLs, or None."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    d = os.path.dirname(ff)
    for _ in os.listdir(d):
        if _.startswith("avcodec") and _.endswith(".dll"):
            return d
    return None


def main():
    if sys.platform != "win32":
        return 1  # Only for Windows

    path = find_ffmpeg_with_dlls()
    if path:
        print(path, end="")
        return 0

    # Download ffmpeg-release-full-shared.7z from gyan.dev
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z"
    tmp = tempfile.gettempdir()
    out_7z = os.path.join(tmp, "ffmpeg-shared-cache.7z")
    extract_dir = os.path.join(tmp, "ffmpeg-extract")

    cache_ok = os.path.isfile(out_7z) and os.path.getsize(out_7z) > 50 * 1024 * 1024
    if not cache_ok:
        print("Downloading FFmpeg full-shared (~100MB, may take 2-5 min)...", file=sys.stderr)
        urllib.request.urlretrieve(url, out_7z)
    print("Extracting FFmpeg...", file=sys.stderr)

    os.makedirs(extract_dir, exist_ok=True)
    for exe in [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]:
        if os.path.isfile(exe):
            subprocess.run(
                [exe, "x", out_7z, f"-o{extract_dir}", "-y"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            break
    else:
        print("ERROR: 7-Zip not found. Install from https://www.7-zip.org/", file=sys.stderr)
        return 1

    # Find bin dir (structure: extract_dir/ffmpeg-x.x.x-full-shared/bin/)
    subs = [f for f in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, f))]
    root = os.path.join(extract_dir, subs[0]) if subs else extract_dir
    bin_path = os.path.join(root, "bin")
    if os.path.isfile(os.path.join(bin_path, "ffmpeg.exe")):
        path = os.path.abspath(bin_path)
    else:
        path = os.path.abspath(root)

    print(path, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
