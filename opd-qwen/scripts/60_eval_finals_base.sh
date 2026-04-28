#!/usr/bin/env bash
# Final eval (base)：OPD-base / SFT-base 全量 1319 题
set -euo pipefail
cd "$(dirname "$0")/.."

# OPD-base on GPU 0
CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py \
    --model models/student-base \
    --lora runs/opd-qwen3-1.7b-base/final \
    --n -1 --batch_size 16 \
    | tee runs/eval/opd_base_final.log &
PID_O=$!

# SFT-base on GPU 2
CUDA_VISIBLE_DEVICES=2 python src/eval_gsm8k.py \
    --model models/student-base \
    --lora runs/sft-qwen3-1.7b-base/final \
    --n -1 --batch_size 16 \
    | tee runs/eval/sft_base_final.log &
PID_S=$!

wait "$PID_O" "$PID_S"
echo "[eval-base] done."
