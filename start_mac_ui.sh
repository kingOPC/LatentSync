#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Python environment is missing. Run ./setup_mac.sh first." >&2
  exit 1
fi

if [[ ! -x "$ROOT_DIR/run_mac.sh" ]]; then
  echo "run_mac.sh is missing or not executable." >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.75}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.65}"

cd "$ROOT_DIR"
exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/gradio_mac.py" "$@"
