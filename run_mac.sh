#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: ./run_mac.sh base.mp4 audio.wav output.mp4" >&2
  exit 2
fi

CALL_DIR="$(pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$1" in /*) INPUT_VIDEO="$1" ;; *) INPUT_VIDEO="$CALL_DIR/$1" ;; esac
case "$2" in /*) INPUT_AUDIO="$2" ;; *) INPUT_AUDIO="$CALL_DIR/$2" ;; esac
case "$3" in /*) OUTPUT_VIDEO="$3" ;; *) OUTPUT_VIDEO="$CALL_DIR/$3" ;; esac

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO" >&2
  exit 1
fi
if [[ ! -f "$INPUT_AUDIO" ]]; then
  echo "Input audio not found: $INPUT_AUDIO" >&2
  exit 1
fi
if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Python environment is missing. Run ./setup_mac.sh first." >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/checkpoints/latentsync_unet.pt" ||
      ! -f "$ROOT_DIR/checkpoints/whisper/tiny.pt" ||
      ! -f "$ROOT_DIR/checkpoints/sd-vae-ft-mse/diffusion_pytorch_model.safetensors" ||
      ! -f "$ROOT_DIR/checkpoints/auxiliary/models/buffalo_l/det_10g.onnx" ]]; then
  echo "Model checkpoints are missing. Run ./setup_mac.sh first." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_VIDEO")"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/latentsync-mac.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT INT TERM

STANDARD_VIDEO="$WORK_DIR/base_25fps_720p.mp4"
STANDARD_AUDIO="$WORK_DIR/audio_16k_mono.wav"
MAX_DIM="${LATENTSYNC_MAX_DIM:-1280}"
NUM_FRAMES="${LATENTSYNC_NUM_FRAMES:-8}"
INFERENCE_STEPS="${LATENTSYNC_INFERENCE_STEPS:-20}"
SEED="${LATENTSYNC_SEED:-1247}"
# The 1280px default is the 720p memory-safe profile. Set
# LATENTSYNC_MAX_DIM=1920 before running if a 1080p source is required.

echo "[1/2] Normalizing video to 25 fps with a ${MAX_DIM}px long edge..."
ffmpeg -hide_banner -loglevel error -y -nostdin \
  -i "$INPUT_VIDEO" -map 0:v:0 -an \
  -vf "fps=25,scale='if(gt(iw,ih),${MAX_DIM},-2)':'if(gt(iw,ih),-2,${MAX_DIM})':flags=lanczos,format=yuv420p" \
  -c:v libx264 -preset medium -crf 18 -movflags +faststart "$STANDARD_VIDEO"

ffmpeg -hide_banner -loglevel error -y -nostdin \
  -i "$INPUT_AUDIO" -vn -ac 1 -ar 16000 -c:a pcm_s16le "$STANDARD_AUDIO"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.75}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.65}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "[2/2] Running LatentSync on MPS (${NUM_FRAMES} frames, ${INFERENCE_STEPS} steps, fp16)..."
cd "$ROOT_DIR"
"$ROOT_DIR/.venv/bin/python" -m scripts.inference \
  --unet_config_path "configs/unet/stage2_512.yaml" \
  --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
  --vae_path "checkpoints/sd-vae-ft-mse" \
  --video_path "$STANDARD_VIDEO" \
  --audio_path "$STANDARD_AUDIO" \
  --video_out_path "$OUTPUT_VIDEO" \
  --device mps \
  --num_frames "$NUM_FRAMES" \
  --inference_steps "$INFERENCE_STEPS" \
  --guidance_scale 1.0 \
  --temp_dir "$WORK_DIR/output" \
  --seed "$SEED"

echo "Done: $OUTPUT_VIDEO"
