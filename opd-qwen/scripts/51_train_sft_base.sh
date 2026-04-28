#!/usr/bin/env bash
# SFT on Qwen3-1.7B-Base：等 effective FLOPs 对照
# 实测 SFT 1.45 s/step；OPD 1000 step × 1.76× FLOPs ≈ SFT 1760 step ≈ 42 min
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

python src/sft_baseline.py \
    --student models/student-base \
    --lr 2e-5 \
    --max_steps "${MAX_STEPS:-1760}" \
    --per_device_batch 2 --grad_accum 4 \
    --max_length 1024 \
    --attn sdpa \
    --output_dir runs/sft-qwen3-1.7b-base \
    --run_name sft-qwen3-1.7b-base-gsm8k \
    "$@"
