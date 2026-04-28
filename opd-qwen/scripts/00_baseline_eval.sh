#!/usr/bin/env bash
# Baseline eval：GPU 0=teacher, GPU 2=student；GPU 3 被外部用户占用，不依赖
set -euo pipefail
cd "$(dirname "$0")/.."
N=${N:-500}

CUDA_VISIBLE_DEVICES=2 python src/eval_gsm8k.py \
    --model models/student --n "$N" --batch_size 16 \
    | tee runs/eval/baseline_student.log &
PID_S=$!

CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py \
    --model models/teacher --n "$N" --batch_size 4 \
    | tee runs/eval/baseline_teacher.log &
PID_T=$!

wait "$PID_S" "$PID_T"
echo "[baseline] done."
