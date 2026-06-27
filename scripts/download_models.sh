#!/usr/bin/env bash
# Download pretrained backbones and BridgeEdit checkpoints.
# Usage:
#   export HF_TOKEN=your_token   # optional but recommended
#   export T2SV_MODEL_ROOT=/path/to/models
#   bash scripts/download_models.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${T2SV_MODEL_ROOT:-${ROOT}/models}"
mkdir -p "${MODEL_ROOT}"

if ! command -v huggingface-cli &>/dev/null; then
  echo "Install huggingface_hub first: pip install huggingface_hub"
  exit 1
fi

download_model() {
  local repo="$1"
  local dir="$2"
  echo ">>> Downloading ${repo} -> ${dir}"
  mkdir -p "${dir}"
  huggingface-cli download "${repo}" --local-dir "${dir}" --local-dir-use-symlinks False ${HF_TOKEN:+--token "$HF_TOKEN"}
}

# --- Backbones for BridgeEdit ---
download_model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" "${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers"
download_model "stabilityai/stable-audio-open-1.0" "${MODEL_ROOT}/stable-audio-open-1.0"

# --- CRR caption models (7B for dev / 72B for paper reproduction) ---
download_model "Qwen/Qwen2.5-VL-7B-Instruct" "${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct"
download_model "Qwen/Qwen2-Audio-7B-Instruct" "${MODEL_ROOT}/Qwen2-Audio-7B-Instruct"
download_model "Qwen/Qwen2.5-7B-Instruct" "${MODEL_ROOT}/Qwen2.5-7B-Instruct"

# Optional 72B models (large GPU memory required):
# download_model "Qwen/Qwen2.5-VL-72B-Instruct" "${MODEL_ROOT}/Qwen2.5-VL-72B-Instruct"
# download_model "Qwen/Qwen2.5-72B-Instruct" "${MODEL_ROOT}/Qwen2.5-72B-Instruct"

# --- BridgeEdit fine-tuned checkpoints (replace with your release URL) ---
mkdir -p "${MODEL_ROOT}/bridgedit/vgg-ss/bicross-1.3B"
mkdir -p "${MODEL_ROOT}/bridgedit/avsync/bicross-1.3B"
echo "Place BridgeEdit .ckpt files under ${MODEL_ROOT}/bridgedit/ (see README)."

echo "Done. Set T2SV_MODEL_ROOT=${MODEL_ROOT} in your shell or config paths."
