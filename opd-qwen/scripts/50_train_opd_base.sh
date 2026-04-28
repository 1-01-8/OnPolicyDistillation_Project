#!/usr/bin/env bash
# OPD on Qwen3-1.7B-Base：1000 step ≈ 9.3 h（35 s/step × 1000）
# 头空间 ~50 pt（baseline 20% → 期望 70%+）
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,2
export TOKENIZERS_PARALLELISM=false

python src/opd_train.py \
    --student models/student-base \
    --lmbda 0.5 --beta 0.5 --temperature 0.9 \
    --lr 2e-5 \
    --max_steps "${MAX_STEPS:-1000}" \
    --max_new_tokens 256 \
    --per_device_batch 2 --grad_accum 4 \
    --attn sdpa \
    --output_dir runs/opd-qwen3-1.7b-base \
    --run_name opd-qwen3-1.7b-base-gsm8k \
    "$@"
