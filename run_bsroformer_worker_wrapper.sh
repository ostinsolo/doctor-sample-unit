#!/usr/bin/env bash
# Wrapper to run BS-RoFormer worker via Python (for benchmarking without frozen exe).
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec "$ROOT/simulate_ci_venv/bin/python" workers/bsroformer_worker.py "$@"
