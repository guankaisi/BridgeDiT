#!/usr/bin/env bash
# CRR inference: concise user prompt -> (video_caption, audio_caption)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${T2SV_MODEL_ROOT:-${ROOT}/models}"
PYTHON="${CAPTION_PYTHON:-/home/kaisi/miniconda/envs/caption/bin/python}"

cd "${ROOT}/caption_pipeline"

"${PYTHON}" crr.py \
  --mode infer \
  --input_json "${1:-examples/user_prompts.json}" \
  --llm_path "${MODEL_ROOT}/Qwen2.5-7B-Instruct" \
  --output_file "${2:-recaption/crr_infer_output.json}" \
  --batch_size 1

echo "Saved to ${2:-recaption/crr_infer_output.json}"
