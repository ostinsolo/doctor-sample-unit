#!/usr/bin/env python3
"""
Smoke test for frozen dsu-demucs: start --worker, wait for ready, run separate,
verify output stem WAVs. Uses soundfile (no torchcodec).
"""
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="Path to dsu-demucs executable")
    ap.add_argument("--wav", required=True, help="Path to input test WAV")
    ap.add_argument("--out", required=True, help="Output directory for stems")
    ap.add_argument("--repo", help="Path to Demucs models dir (local .th + yaml)")
    args = ap.parse_args()

    if not os.path.isfile(args.exe):
        print(f"ERROR: Exe not found: {args.exe}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.wav):
        print(f"ERROR: WAV not found: {args.wav}", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out)
    wav_path = os.path.abspath(args.wav)
    exe_dir = os.path.dirname(os.path.abspath(args.exe))
    os.makedirs(out_dir, exist_ok=True)

    proc = subprocess.Popen(
        [os.path.abspath(args.exe), "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=exe_dir,
    )

    try:
        # Wait for ready
        deadline = time.time() + 120
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("status") == "ready":
                    break
                if obj.get("status") == "error":
                    print(f"Worker error before separate: {obj}", file=sys.stderr)
                    return 1
            except json.JSONDecodeError:
                pass

        # Send separate
        cmd = {
            "cmd": "separate",
            "input": wav_path,
            "output": out_dir,
            "model": "htdemucs",
        }
        if args.repo and os.path.isdir(args.repo):
            cmd["repo"] = os.path.abspath(args.repo)
        proc.stdin.write(json.dumps(cmd) + "\n")
        proc.stdin.flush()

        # Read until done or error
        result = None
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                s = obj.get("status")
                if s == "done":
                    result = obj
                    break
                if s == "error":
                    print(f"Worker error: {obj}", file=sys.stderr)
                    return 1
            except json.JSONDecodeError:
                pass

        if not result:
            print("ERROR: No 'done' response from worker", file=sys.stderr)
            return 1

        # Verify stem files
        output_dir = result.get("output_dir", "")
        if not output_dir or not os.path.isdir(output_dir):
            print(f"ERROR: Output dir missing: {output_dir}", file=sys.stderr)
            return 1
        stems = result.get("files") or result.get("stems") or []
        elapsed = result.get("elapsed")
        if elapsed is not None:
            print(f"  Demucs separation: {elapsed:.2f}s (worker-reported)")
        for name in stems:
            if not name.endswith(".wav"):
                name = name + ".wav"
            path = os.path.join(output_dir, name)
            if not os.path.isfile(path):
                print(f"ERROR: Stem not found: {path}", file=sys.stderr)
                return 1
            print(f"  OK {path} ({os.path.getsize(path)} bytes)")

        # Graceful exit
        try:
            proc.stdin.write('{"cmd":"exit"}\n')
            proc.stdin.flush()
        except Exception:
            pass
        proc.wait(timeout=5)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        proc.kill()
        return 1
    finally:
        try:
            proc.kill()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
