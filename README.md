# BridgeDiT: Taming Text-to-Sounding Video Generation

Official implementation of **Taming Text-to-Sounding Video Generation via Advanced Modality Condition and Interaction** (ECCV 2026).

## Repository Structure

```
t2sv-release/
├── caption_pipeline/          # CRR caption framework
│   ├── crr.py                 # Cross-Referential Rewriter (main, paper method)
│   ├── count.py               # CRR + per-stage timing benchmark
│   ├── caption_pipeline.py    # Early baseline (simple rewrite)
│   ├── vllm_utils.py          # vLLM tensor-parallel helper
│   ├── examples/              # Sample inputs
│   └── recaption/             # Caption JSON outputs
├── bridgedit/                 # BridgeEdit model (train + infer + eval)
│   ├── train_joint.py         # JointDiT_T2AV Lightning module
│   ├── infer.py               # Single-sample inference CLI
│   ├── main.py                # Training & batch inference entry points
│   ├── config/                # YAML configs (train / sample / dataset)
│   ├── dataset/               # VGGSound, AVSync, Landscape loaders
│   ├── models/                # Wan video + Stable Audio + dual-stream DiT
│   └── evaluation/            # FVD, FAD, CLAP, ImageBind, AVAlign, etc.
└── scripts/
    ├── download_models.sh
    ├── run_crr_infer.sh
    └── run_bridgedit_infer.sh
```

---

## Environment Setup

Two separate conda environments are recommended:

### 1. `caption` — Caption pipeline (vLLM + Qwen)

```bash
conda create -n caption python=3.10 -y
conda activate caption
pip install -r caption_pipeline/requirements.txt
# Install vLLM for your CUDA version: https://docs.vllm.ai
pip install qwen-vl-utils
```

Key packages: `torch==2.6.0`, `vllm==0.8.4`, `transformers==4.52.0`

### 2. `bridgedit` — Model training & inference

```bash
conda create -n bridgedit python=3.10 -y
conda activate bridgedit
pip install -r bridgedit/requirements.txt
```

Key packages: `torch==2.6.0`, `pytorch-lightning==2.5.5`, `deepspeed==0.18.1`, `diffusers>=0.36.0`

---

## Model Weights

Download weights into `./models` (or set a custom root):

```bash
export T2SV_MODEL_ROOT=./models   # default when unset: <repo>/models
bash scripts/download_models.sh
```

`bridgedit/config/sample.yaml` already points to `../models/...` relative to the `bridgedit/` working directory. Caption scripts resolve model paths via `caption_pipeline/paths.py` using the same `T2SV_MODEL_ROOT`.

### Required backbones (BridgeEdit)


| Model             | HuggingFace ID                      | Local path                                   |
| ----------------- | ----------------------------------- | -------------------------------------------- |
| Wan2.1 T2V 1.3B   | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`  | `$T2SV_MODEL_ROOT/Wan2.1-T2V-1.3B-Diffusers` |
| Stable Audio Open | `stabilityai/stable-audio-open-1.0` | `$T2SV_MODEL_ROOT/stable-audio-open-1.0`     |


### CRR caption models


| Role                        | Paper (72B)               | Dev / smoke test (7B)    |
| --------------------------- | ------------------------- | ------------------------ |
| Video raw caption           | `Qwen2.5-VL-72B-Instruct` | `Qwen2.5-VL-7B-Instruct` |
| Audio raw caption           | `Qwen2-Audio-7B-Instruct` | same                     |
| Semantic Checker + Rewriter | `Qwen2.5-72B-Instruct`    | `Qwen2.5-7B-Instruct`    |


Download with:

```bash
bash scripts/download_models.sh
```

### BridgeEdit fine-tuned checkpoints

Update paths in `bridgedit/config/sample.yaml` only if you use a non-default `T2SV_MODEL_ROOT`.

---

## Quick Start

Both pipelines were run successfully on **2025-06-26** with A100 80GB GPUs.

### A. CRR inference — user prompt → aligned captions

```bash
conda activate caption
cd caption_pipeline

# Single GPU (7B LLM)
export VLLM_TENSOR_PARALLEL_SIZE=1
export CUDA_VISIBLE_DEVICES=0

python crr.py \
  --mode infer \
  --input_json examples/user_prompts.json \
  --output_file recaption/crr_infer_output.json
```

Or use the helper script:

```bash
bash scripts/run_crr_infer.sh
```

**Output fields per sample:** `semantic_anchors`, `crr_video_caption`, `crr_audio_caption`

### B. CRR training — sounding video → aligned captions

```bash
python crr.py \
  --mode train \
  --video_path /path/to/videos \
  --output_file recaption/my_captions.json
```

For paper reproduction with 72B models, set `VLLM_TENSOR_PARALLEL_SIZE=4` (VL/Audio) or `8` (72B LLM) across multiple GPUs.

### C. BridgeEdit inference — captions → sounding video

```bash
conda activate bridgedit
cd bridgedit

export CUDA_VISIBLE_DEVICES=0
python infer.py \
  --ckpt_path ${T2SV_MODEL_ROOT}/bridgedit/vgg-ss/bicross-1.3B/epoch=20-step=3318.ckpt \
  --video_prompt "A German Shepherd barks alertly in a sunny backyard beside a swimming pool." \
  --audio_prompt "Alert dog barks echo in a quiet backyard with gentle pool water sounds." \
  --save_file demo
```

Output: `bridgedit/save_videos/cross/demo.mp4` (~5.4 s, 480×834, 15 fps)

Or:

```bash
bash scripts/run_bridgedit_infer.sh
```

### D. End-to-end pipeline

```bash
# Step 1: CRR expands a short user prompt
conda activate caption
python caption_pipeline/crr.py --mode infer \
  --input_json caption_pipeline/examples/user_prompts.json \
  --output_file /tmp/crr_out.json

# Step 2: BridgeEdit generates video+audio from CRR captions
conda activate bridgedit
python bridgedit/infer.py \
  --ckpt_path ${T2SV_MODEL_ROOT}/bridgedit/vgg-ss/bicross-1.3B/epoch=20-step=3318.ckpt \
  --video_prompt "<crr_video_caption from JSON>" \
  --audio_prompt "<crr_audio_caption from JSON>" \
  --save_file e2e_demo
```

---

## Training (BridgeEdit)

Edit dataset paths in `bridgedit/config/dataset.yaml`, then:

```bash
conda activate bridgedit
cd bridgedit

# AVSync15 fine-tuning (example — see main.py for all recipes)
python -c "from main import train_avsync; train_avsync()"

# VGG-Sound SS
python -c "from main import train_vgg_ss; train_vgg_ss()"
```

Training configs:


| Config                    | Description                 |
| ------------------------- | --------------------------- |
| `config/train.yaml`       | 1.3B BridgeEdit fine-tuning |
| `config/train_large.yaml` | 14B / multi-node DeepSpeed  |
| `config/sample.yaml`      | Inference sampling settings |


---

## Evaluation

Metrics live under `bridgedit/evaluation/`:


| Metric           | Script                              |
| ---------------- | ----------------------------------- |
| FVD              | `eval_fvd.py`                       |
| FAD              | `eval_fad.py`                       |
| CLAP score       | `eval_clap.py`                      |
| ImageBind (IB)   | `eval_ib_all.py`                    |
| AVAlign / desync | `eval_avalign.py`, `eval_desync.py` |


Batch evaluation on a benchmark:

```bash
cd bridgedit
python main.py   # default: predict_avsync_parallel() — edit __main__ as needed
```

---

## Multi-GPU Notes


| Setting              | Env var                                    | Typical value                                |
| -------------------- | ------------------------------------------ | -------------------------------------------- |
| vLLM tensor parallel | `VLLM_TENSOR_PARALLEL_SIZE`                | `1` (7B, 1 GPU), `4` (72B VL), `8` (72B LLM) |
| BridgeEdit inference | `CUDA_VISIBLE_DEVICES`                     | `0` (single sample)                          |
| BridgeEdit training  | PyTorch Lightning `devices=8` in `main.py` | 8× GPU DDP                                   |


---

## Citation

```bibtex
@misc{guan2025tamingtexttosoundingvideogeneration,
      title={Taming Text-to-Sounding Video Generation via Advanced Modality Condition and Interaction}, 
      author={Kaisi Guan and Xihua Wang and Zhengfeng Lai and Xin Cheng and Peng Zhang and XiaoJiang Liu and Ruihua Song and Meng Cao},
      year={2025},
      eprint={2510.03117},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.03117}, 
}
```

---

## License

Research code only. Pretrained backbones (Wan2.1, Stable Audio, Qwen) follow their respective licenses.
