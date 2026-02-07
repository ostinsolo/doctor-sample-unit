#!/usr/bin/env bash
# Run DSU workers locally (workers method). Use runtime Python when present.
# Usage: ./run_local.sh <demucs|bsroformer|audio-separator|denoise> [args...]
#
# RUNTIME LOCATIONS (do not mix):
#   - scripts/building/sh/runtime/  <- Canonical (Intel or MPS, from build_runtime_mac_*.sh)
#   - runtime/                      <- Legacy at project root (deprecated)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME=
WORKER=

case "$1" in
  demucs)        WORKER=demucs_worker.py ;;
  bsroformer)    WORKER=bsroformer_worker.py ;;
  audio-separator) WORKER=audio_separator_worker.py ;;
  denoise)       WORKER=denoise_worker.py ;;
  *)
    echo "Usage: $0 <demucs|bsroformer|audio-separator|denoise> [args...]"
    echo "  ./run_local.sh demucs --help"
    echo "  ./run_local.sh demucs --worker"
    exit 1
    ;;
esac
shift

# Prefer canonical runtime (scripts/building/sh/runtime) - created by build_runtime_mac_*.sh
CANONICAL_RUNTIME="$PROJECT_ROOT/scripts/building/sh/runtime/bin/python"
LEGACY_RUNTIME="$PROJECT_ROOT/runtime/bin/python"
WINDOWS_RUNTIME="$PROJECT_ROOT/runtime/Scripts/python.exe"

if [[ -f "$CANONICAL_RUNTIME" ]]; then
  RUNTIME="$CANONICAL_RUNTIME"
  # Validate platform match (Intel vs MPS)
  if [[ -f "$PROJECT_ROOT/scripts/building/sh/runtime/RUNTIME_PLATFORM.txt" ]]; then
    RUNTIME_PLATFORM=$(cat "$PROJECT_ROOT/scripts/building/sh/runtime/RUNTIME_PLATFORM.txt" 2>/dev/null || true)
    ARCH=$(uname -m)
    if [[ "$RUNTIME_PLATFORM" = "intel" && "$ARCH" = "arm64" ]]; then
      echo "WARNING: Runtime is Intel but you're on Apple Silicon. Run build_runtime_mac_mps.sh." >&2
    elif [[ "$RUNTIME_PLATFORM" = "mps" && "$ARCH" = "x86_64" ]]; then
      echo "WARNING: Runtime is MPS (Apple Silicon) but you're on Intel. Run build_runtime_mac_intel.sh." >&2
    fi
  fi
elif [[ -f "$LEGACY_RUNTIME" ]]; then
  RUNTIME="$LEGACY_RUNTIME"
  echo "WARNING: Using legacy runtime/ (project root). Prefer scripts/building/sh/build_runtime_mac_*.sh for canonical runtime." >&2
elif [[ -f "$WINDOWS_RUNTIME" ]]; then
  RUNTIME="$WINDOWS_RUNTIME"
elif command -v python3 &>/dev/null; then
  RUNTIME=python3
  echo "WARNING: No runtime/ found. Using system python3. Run scripts/building/sh/build_runtime_mac_mps.sh (Mac) or build_runtime_mac_intel.sh (Intel) first." >&2
else
  echo "ERROR: No Python found. Create runtime with scripts/building/sh/build_runtime_mac_mps.sh (Mac ARM) or build_runtime_mac_intel.sh (Mac Intel)." >&2
  exit 1
fi

WORKER_PATH="$PROJECT_ROOT/workers/$WORKER"
if [[ ! -f "$WORKER_PATH" ]]; then
  echo "ERROR: $WORKER_PATH not found." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
exec "$RUNTIME" "$WORKER_PATH" "$@"
