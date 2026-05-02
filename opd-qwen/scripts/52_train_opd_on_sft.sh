#!/usr/bin/env bash
# OPD on top of (1.7B-Base + SFT)：与 50_train_opd_base.sh 完全同条件，仅 student 起点不同
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,2
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

python src/opd_train.py \
    --student models/student-base-sft-merged \
    --lmbda 0.5 --beta 0.5 --temperature 0.9 \
    --lr 2e-5 \
    --max_steps "${MAX_STEPS:-1000}" \
    --max_new_tokens 256 \
    --per_device_batch 2 --grad_accum 4 \
    --attn sdpa \
    --output_dir runs/opd-on-sft-1.7b-base \
    --run_name opd-on-sft-1.7b-base-gsm8k \
    "$@"
