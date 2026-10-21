#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This installer is intended for Apple Silicon macOS." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it from https://brew.sh and rerun this script." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  brew install ffmpeg
fi

if ! command -v uv >/dev/null 2>&1; then
  brew install uv
fi

uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt

HF_CLI="$ROOT_DIR/.venv/bin/huggingface-cli"
mkdir -p checkpoints
export HF_HUB_DISABLE_XET=1
"$HF_CLI" download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
"$HF_CLI" download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints
"$HF_CLI" download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.safetensors \
  --local-dir checkpoints/sd-vae-ft-mse

EXPECTED_UNET_SHA256="0a478e89eb660f82da4c35dbdde8a5adfb27f99d1b4e50edd03729e1e98316d3"
ACTUAL_UNET_SHA256="$(shasum -a 256 checkpoints/latentsync_unet.pt | awk '{print $1}')"
if [[ "$ACTUAL_UNET_SHA256" != "$EXPECTED_UNET_SHA256" ]]; then
  echo "LatentSync UNet checksum verification failed." >&2
  exit 1
fi

# Cache InsightFace's CPU detection/landmark models during setup so the first
# inference does not unexpectedly pause for another download.
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python - <<'PY'
from latentsync.utils.face_detector import FaceDetector

FaceDetector("cpu")
print("InsightFace models are ready.")
PY

echo "LatentSync Mac environment and checkpoints are ready."
