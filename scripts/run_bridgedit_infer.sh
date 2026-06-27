#!/usr/bin/env bash
# BridgeEdit single-sample inference
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${T2SV_MODEL_ROOT:-${ROOT}/models}"
PYTHON="${BRIDGEDIT_PYTHON:-/home/kaisi/miniconda/envs/bridgedit/bin/python}"
CKPT="${CKPT_PATH:-${MODEL_ROOT}/bridgedit/vgg-ss/bicross-1.3B/epoch=20-step=3318.ckpt}"

cd "${ROOT}/bridgedit"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON}" infer.py \
  --ckpt_path "${CKPT}" \
  --save_file "${SAVE_NAME:-demo}" \
  "$@"

echo "Output: ${ROOT}/bridgedit/save_videos/cross/${SAVE_NAME:-demo}.mp4"
