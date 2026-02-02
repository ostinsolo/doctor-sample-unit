#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec "$ROOT/simulate_ci_venv/bin/python" workers/demucs_worker.py "$@"
