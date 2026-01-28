import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Optional, Tuple


def _now() -> float:
    return time.perf_counter()


def _read_json_line(proc: subprocess.Popen, timeout_s: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Read one stdout line and parse JSON.
    Returns (obj, raw_line). If timeout, returns (None, None).
    """
    deadline = _now() + timeout_s
    while _now() < deadline:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            # process may have exited or no output yet; small sleep to avoid spinning
            if proc.poll() is not None:
                return None, None
            time.sleep(0.01)
            continue
        raw = line.strip()
        if not raw:
            continue
        try:
            return json.loads(raw), raw
        except Exception:
            # not JSON, return raw for debugging
            return None, raw
    return None, None


def _wait_for_status(
    proc: subprocess.Popen,
    *,
    want: Iterable[str],
    timeout_s: float,
) -> Tuple[Dict[str, Any], float]:
    want_set = set(want)
    t0 = _now()
    while True:
        obj, raw = _read_json_line(proc, timeout_s=timeout_s)
        if obj is None:
            # Ignore non-JSON chatter (e.g. torch/attention one-time prints) but keep it if needed later.
            if raw is not None:
                continue
            raise TimeoutError(f"Timed out waiting for {want_set}")
        status = obj.get("status")
        if status in want_set:
            return obj, _now() - t0


def _send(proc: subprocess.Popen, payload: Dict[str, Any]) -> None:
    line = json.dumps(payload)
    proc.stdin.write(line + "\n")  # type: ignore[union-attr]
    proc.stdin.flush()  # type: ignore[union-attr]


def bench_bsroformer(
    exe_path: str,
    *,
    models_dir: str,
    model_name: str,
    input_wav: str,
    output_dir: str,
    device: str = "cuda",
    timeout_s: float = 600.0,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"worker": "bsroformer", "exe": exe_path}

    t_start = _now()
    proc = subprocess.Popen(
        [exe_path, "--worker", "--models-dir", models_dir, "--device", device],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(exe_path) or None,
    )
    try:
        # time to first output (shared runtime / library load)
        obj_first, dt_first = _wait_for_status(proc, want=["loading", "ready", "error"], timeout_s=timeout_s)
        results["t_start_to_first_status_s"] = round(dt_first, 3)
        results["first_status"] = obj_first.get("status")

        if obj_first.get("status") == "error":
            results["error"] = obj_first
            return results

        # time to ready (imports + basic init)
        if obj_first.get("status") != "ready":
            obj_ready, dt_ready = _wait_for_status(proc, want=["ready", "error"], timeout_s=timeout_s)
        else:
            obj_ready, dt_ready = obj_first, 0.0
        results["t_start_to_ready_s"] = round((_now() - t_start), 3)
        results["ready"] = obj_ready

        # load model
        _send(proc, {"cmd": "load_model", "model": model_name})
        obj_loaded, dt_loaded = _wait_for_status(proc, want=["model_loaded", "error"], timeout_s=timeout_s)
        results["t_load_model_s"] = round(dt_loaded, 3)
        results["model_loaded"] = obj_loaded
        if obj_loaded.get("status") == "error":
            results["error"] = obj_loaded
            return results

        # cold separate (model already loaded, but first run includes kernels/caches)
        _send(proc, {"cmd": "separate", "input": input_wav, "output_dir": output_dir, "model": model_name})
        obj_done1, dt_done1 = _wait_for_status(proc, want=["done", "error"], timeout_s=timeout_s)
        results["t_separate_cold_wall_s"] = round(dt_done1, 3)
        results["separate_cold"] = obj_done1
        if obj_done1.get("status") == "error":
            results["error"] = obj_done1
            return results

        # warm separate (immediately again)
        _send(proc, {"cmd": "separate", "input": input_wav, "output_dir": output_dir, "model": model_name})
        obj_done2, dt_done2 = _wait_for_status(proc, want=["done", "error"], timeout_s=timeout_s)
        results["t_separate_warm_wall_s"] = round(dt_done2, 3)
        results["separate_warm"] = obj_done2

        return results
    finally:
        try:
            _send(proc, {"cmd": "exit"})
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass


def bench_demucs(
    exe_path: str,
    *,
    model_name: str,
    input_wav: str,
    output_dir: str,
    timeout_s: float = 600.0,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"worker": "demucs", "exe": exe_path}

    t_start = _now()
    proc = subprocess.Popen(
        [exe_path, "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(exe_path) or None,
    )
    try:
        obj_first, dt_first = _wait_for_status(proc, want=["loading", "ready", "error"], timeout_s=timeout_s)
        results["t_start_to_first_status_s"] = round(dt_first, 3)
        results["first_status"] = obj_first.get("status")
        if obj_first.get("status") == "error":
            results["error"] = obj_first
            return results

        if obj_first.get("status") != "ready":
            obj_ready, _ = _wait_for_status(proc, want=["ready", "error"], timeout_s=timeout_s)
        else:
            obj_ready = obj_first
        results["t_start_to_ready_s"] = round((_now() - t_start), 3)
        results["ready"] = obj_ready
        if obj_ready.get("status") == "error":
            results["error"] = obj_ready
            return results

        _send(proc, {"cmd": "load_model", "model": model_name})
        obj_loaded, dt_loaded = _wait_for_status(proc, want=["model_loaded", "error"], timeout_s=timeout_s)
        results["t_load_model_s"] = round(dt_loaded, 3)
        results["model_loaded"] = obj_loaded
        if obj_loaded.get("status") == "error":
            results["error"] = obj_loaded
            return results

        _send(proc, {"cmd": "separate", "input": input_wav, "output": output_dir, "model": model_name})
        obj_done1, dt_done1 = _wait_for_status(proc, want=["done", "error"], timeout_s=timeout_s)
        results["t_separate_cold_wall_s"] = round(dt_done1, 3)
        results["separate_cold"] = obj_done1
        if obj_done1.get("status") == "error":
            results["error"] = obj_done1
            return results

        _send(proc, {"cmd": "separate", "input": input_wav, "output": output_dir, "model": model_name})
        obj_done2, dt_done2 = _wait_for_status(proc, want=["done", "error"], timeout_s=timeout_s)
        results["t_separate_warm_wall_s"] = round(dt_done2, 3)
        results["separate_warm"] = obj_done2
        if obj_done2.get("status") == "error":
            results["error"] = obj_done2
        return results
    finally:
        try:
            _send(proc, {"cmd": "exit"})
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass


def bench_audio_separator(
    exe_path: str,
    *,
    model_file_dir: str,
    model_name: str,
    input_wav: str,
    output_dir: str,
    timeout_s: float = 600.0,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"worker": "audio_separator", "exe": exe_path}

    t_start = _now()
    proc = subprocess.Popen(
        [exe_path, "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(exe_path) or None,
    )
    try:
        obj_first, dt_first = _wait_for_status(proc, want=["loading", "ready", "error"], timeout_s=timeout_s)
        results["t_start_to_first_status_s"] = round(dt_first, 3)
        results["first_status"] = obj_first.get("status")
        if obj_first.get("status") == "error":
            results["error"] = obj_first
            return results

        if obj_first.get("status") != "ready":
            obj_ready, _ = _wait_for_status(proc, want=["ready", "error"], timeout_s=timeout_s)
        else:
            obj_ready = obj_first
        results["t_start_to_ready_s"] = round((_now() - t_start), 3)
        results["ready"] = obj_ready
        if obj_ready.get("status") == "error":
            results["error"] = obj_ready
            return results

        _send(proc, {"cmd": "load_model", "model": model_name, "model_file_dir": model_file_dir})
        obj_loaded, dt_loaded = _wait_for_status(proc, want=["model_loaded", "error"], timeout_s=timeout_s)
        results["t_load_model_s"] = round(dt_loaded, 3)
        results["model_loaded"] = obj_loaded
        if obj_loaded.get("status") == "error":
            results["error"] = obj_loaded
            return results

        _send(proc, {"cmd": "separate", "input": input_wav, "output_dir": output_dir, "model": model_name, "model_file_dir": model_file_dir})
        obj_done1, dt_done1 = _wait_for_status(proc, want=["done", "error"], timeout_s=timeout_s)
        results["t_separate_cold_wall_s"] = round(dt_done1, 3)
        results["separate_cold"] = obj_done1
        if obj_done1.get("status") == "error":
            results["error"] = obj_done1
            return results

        _send(proc, {"cmd": "separate", "input": input_wav, "output_dir": output_dir, "model": model_name, "model_file_dir": model_file_dir})
        obj_done2, dt_done2 = _wait_for_status(proc, want=["done", "error"], timeout_s=timeout_s)
        results["t_separate_warm_wall_s"] = round(dt_done2, 3)
        results["separate_warm"] = obj_done2
        if obj_done2.get("status") == "error":
            results["error"] = obj_done2
        return results
    finally:
        try:
            _send(proc, {"cmd": "exit"})
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark DSU worker E2E timings (startup/model load/separate).")
    ap.add_argument("--exe", required=True, help="Path to frozen worker exe (e.g. dist\\dsu\\dsu-bsroformer.exe)")
    ap.add_argument("--worker", choices=["bsroformer", "demucs", "audio-separator"], default="bsroformer")
    ap.add_argument("--models-dir", default=None, help="BSRoFormer models dir containing models.json + weights/")
    ap.add_argument("--model", required=True, help="Model name (bsroformer registry model, demucs model name, or audio-separator .pth)")
    ap.add_argument("--model-file-dir", default=None, help="audio-separator model folder containing .pth files")
    ap.add_argument("--input", required=True, help="Input wav path")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    exe_path = os.path.abspath(args.exe)
    if not os.path.exists(exe_path):
        print(f"ERROR: exe not found: {exe_path}", file=sys.stderr)
        return 2
    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)

    if args.worker == "bsroformer":
        if not args.models_dir:
            print("ERROR: --models-dir is required for bsroformer", file=sys.stderr)
            return 2
        res = bench_bsroformer(
            exe_path,
            models_dir=args.models_dir,
            model_name=args.model,
            input_wav=args.input,
            output_dir=args.output_dir,
            device=args.device,
            timeout_s=args.timeout,
        )
    elif args.worker == "demucs":
        res = bench_demucs(
            exe_path,
            model_name=args.model,
            input_wav=args.input,
            output_dir=args.output_dir,
            timeout_s=args.timeout,
        )
    elif args.worker == "audio-separator":
        if not args.model_file_dir:
            print("ERROR: --model-file-dir is required for audio-separator", file=sys.stderr)
            return 2
        res = bench_audio_separator(
            exe_path,
            model_file_dir=args.model_file_dir,
            model_name=args.model,
            input_wav=args.input,
            output_dir=args.output_dir,
            timeout_s=args.timeout,
        )
    else:
        print("Unsupported worker", file=sys.stderr)
        return 2

    print(json.dumps(res, indent=2))
    return 0 if "error" not in res else 3


if __name__ == "__main__":
    raise SystemExit(main())

