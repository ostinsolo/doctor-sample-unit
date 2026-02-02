#!/usr/bin/env python3
"""
Test frozen dsu-bsroformer: start --worker, wait for ready, run separate.
Requires --models-dir pointing to project root (models.json + weights/).
"""
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="Path to dsu-bsroformer executable")
    ap.add_argument("--wav", required=True, help="Path to input test WAV")
    ap.add_argument("--out", required=True, help="Output directory for stems")
    ap.add_argument("--models-dir", required=True, help="Project root (models.json, weights/)")
    ap.add_argument("--model", default="bsroformer_4stem", help="Model name from models.json")
    ap.add_argument("--batch-size", type=int, default=4, help="Chunk batch size (4-8 faster, config default often 2)")
    args = ap.parse_args()

    if not os.path.isfile(args.exe):
        print(f"ERROR: Exe not found: {args.exe}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.wav):
        print(f"ERROR: WAV not found: {args.wav}", file=sys.stderr)
        return 1
    if not os.path.isdir(args.models_dir):
        print(f"ERROR: Models dir not found: {args.models_dir}", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out)
    wav_path = os.path.abspath(args.wav)
    models_dir = os.path.abspath(args.models_dir)
    exe_dir = os.path.dirname(os.path.abspath(args.exe))
    os.makedirs(out_dir, exist_ok=True)

    proc = subprocess.Popen(
        [os.path.abspath(args.exe), "--worker", "--models-dir", models_dir],
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
        ready = False
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
                    ready = True
                    print("  dsu-bsroformer: ready OK")
                    break
                if obj.get("status") == "error":
                    print(f"Worker error: {obj}", file=sys.stderr)
                    return 1
            except json.JSONDecodeError:
                pass

        if not ready:
            print("ERROR: No 'ready' from dsu-bsroformer (timeout)", file=sys.stderr)
            return 1

        # Send separate (batch_size 4+ reduces kernel launches, faster on MPS/CUDA)
        cmd = {
            "cmd": "separate",
            "input": wav_path,
            "output_dir": out_dir,
            "model": args.model,
            "batch_size": args.batch_size,
        }
        proc.stdin.write(json.dumps(cmd) + "\n")
        proc.stdin.flush()

        # Read until done or error
        result = None
        deadline = time.time() + 180
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
                    stems = obj.get("files") or obj.get("stems") or []
                    elapsed = obj.get("elapsed")
                    if elapsed is not None:
                        print(f"  dsu-bsroformer separate: OK ({len(stems)} stems, {elapsed:.2f}s)")
                    else:
                        print(f"  dsu-bsroformer separate: OK ({len(stems)} stems)")
                    output_dir = obj.get("output_dir", "")
                    if output_dir and os.path.isdir(output_dir):
                        for f in (obj.get("files") or [])[:4]:
                            path = os.path.join(output_dir, f if f.endswith(".wav") else f + ".wav")
                            if os.path.isfile(path):
                                print(f"    OK {path} ({os.path.getsize(path)} bytes)")
                    break
                if s == "error":
                    msg = obj.get("message", "error")
                    print(f"  dsu-bsroformer separate: {msg}")
                    print("    (weights may be missing - run from project with weights/ downloaded)")
                    break
            except json.JSONDecodeError:
                pass

        # Exit
        proc.stdin.write('{"cmd":"exit"}\n')
        proc.stdin.flush()
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
