#!/usr/bin/env bash
# Final eval：OPD ckpt → GPU 0；SFT ckpt → GPU 2；并行
set -euo pipefail
cd "$(dirname "$0")/.."
N=${N:-1319}

CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py \
    --model models/student --lora runs/opd-qwen3-1.7b/final \
    --n "$N" --batch_size 16 \
    | tee runs/eval/opd_final.log &
PID_O=$!

CUDA_VISIBLE_DEVICES=2 python src/eval_gsm8k.py \
    --model models/student --lora runs/sft-qwen3-1.7b/final \
    --n "$N" --batch_size 16 \
    | tee runs/eval/sft_final.log &
PID_S=$!

wait "$PID_O" "$PID_S"
echo "[eval] done."
