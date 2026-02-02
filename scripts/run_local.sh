#!/usr/bin/env bash
# Run DSU workers locally (workers method). Use runtime Python when present.
# Usage: ./run_local.sh <demucs|bsroformer|audio-separator> [args...]
# Example: ./run_local.sh demucs --help
#          ./run_local.sh demucs --worker

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME=
WORKER=

case "$1" in
  demucs)        WORKER=demucs_worker.py ;;
  bsroformer)    WORKER=bsroformer_worker.py ;;
  audio-separator) WORKER=audio_separator_worker.py ;;
  *)
    echo "Usage: $0 <demucs|bsroformer|audio-separator> [args...]"
    echo "  ./run_local.sh demucs --help"
    echo "  ./run_local.sh demucs --worker"
    exit 1
    ;;
esac
shift

if [[ -f "$PROJECT_ROOT/runtime/bin/python" ]]; then
  RUNTIME="$PROJECT_ROOT/runtime/bin/python"
elif [[ -f "$PROJECT_ROOT/scripts/building/sh/runtime/bin/python" ]]; then
  RUNTIME="$PROJECT_ROOT/scripts/building/sh/runtime/bin/python"
elif [[ -f "$PROJECT_ROOT/runtime/Scripts/python.exe" ]]; then
  RUNTIME="$PROJECT_ROOT/runtime/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
  RUNTIME=python3
  echo "WARNING: No runtime/ found. Using system python3. Run scripts/building/sh/build_runtime_mac_mps.sh (Mac) first." >&2
else
  echo "ERROR: No Python found. Create runtime with scripts/building/sh/build_runtime_mac_mps.sh (Mac) or use a venv with deps." >&2
  exit 1
fi

WORKER_PATH="$PROJECT_ROOT/workers/$WORKER"
if [[ ! -f "$WORKER_PATH" ]]; then
  echo "ERROR: $WORKER_PATH not found." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
exec "$RUNTIME" "$WORKER_PATH" "$@"
