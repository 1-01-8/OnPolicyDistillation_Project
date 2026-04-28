#!/usr/bin/env bash
# OPD 训练（CUDA_VISIBLE_DEVICES=0,2）
# 实测：trainer 强制单卡（GPU 0），bs=2/ga=4/mnt=256 = ~35 s/step；300 步 ≈ 3 h
# 想跑 1 epoch（GSM8K 7473 / eff_batch 8 ≈ 935 steps）改 --max_steps 935
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,2
export TOKENIZERS_PARALLELISM=false

python src/opd_train.py \
    --lmbda 0.5 --beta 0.5 --temperature 0.9 \
    --max_steps "${MAX_STEPS:-300}" \
    --max_new_tokens 256 \
    --per_device_batch 2 --grad_accum 4 \
    --attn sdpa \
    --output_dir runs/opd-qwen3-1.7b \
    --run_name opd-qwen3-1.7b-gsm8k \
    "$@"
