#!/usr/bin/env python3
"""
Smoke test for frozen dsu-audio-separator: start --worker, wait for ready,
optionally run a VR separate if model is available.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="Path to dsu-audio-separator executable")
    ap.add_argument("--wav", help="Path to input WAV (optional, for separate test)")
    ap.add_argument("--out", help="Output dir for separate (required if --wav)")
    ap.add_argument("--model", default="5_HP-Karaoke-UVR.pth", help="VR model name (skip separate if not found)")
    ap.add_argument("--models-dir", help="Path to audio-separator models (model_file_dir)")
    args = ap.parse_args()

    if not os.path.isfile(args.exe):
        print(f"ERROR: Exe not found: {args.exe}", file=sys.stderr)
        return 1

    exe_dir = os.path.dirname(os.path.abspath(args.exe))
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
        deadline = time.time() + 90
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
                    print("  dsu-audio-separator: ready OK")
                    break
                if obj.get("status") == "error":
                    print(f"Worker error: {obj}", file=sys.stderr)
                    return 1
            except json.JSONDecodeError:
                pass

        if not ready:
            print("ERROR: No 'ready' from dsu-audio-separator (timeout)", file=sys.stderr)
            return 1

        # Optional: get_status
        proc.stdin.write('{"cmd":"get_status"}\n')
        proc.stdin.flush()
        time.sleep(0.5)

        # Optional: separate if wav provided
        if args.wav and args.out and os.path.isfile(args.wav):
            out_dir = os.path.abspath(args.out)
            os.makedirs(out_dir, exist_ok=True)
            cmd = {
                "cmd": "separate",
                "input": os.path.abspath(args.wav),
                "output_dir": out_dir,
                "model": args.model,
            }
            if args.models_dir and os.path.isdir(args.models_dir):
                cmd["model_file_dir"] = os.path.abspath(args.models_dir)
            proc.stdin.write(json.dumps(cmd) + "\n")
            proc.stdin.flush()
            deadline = time.time() + 120
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
                        elapsed = obj.get("elapsed")
                        stems_count = len(obj.get("files", []))
                        if elapsed is not None:
                            print(f"  dsu-audio-separator separate: OK ({stems_count} stems, {elapsed:.2f}s)")
                        else:
                            print(f"  dsu-audio-separator separate: OK ({stems_count} stems)")
                        break
                    if s == "error":
                        print(f"  dsu-audio-separator separate: {obj.get('message', 'error')} (model may be missing)")
                        break
                except json.JSONDecodeError:
                    pass
        else:
            print("  dsu-audio-separator: smoke OK (no separate - use --wav --out for full test)")

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
