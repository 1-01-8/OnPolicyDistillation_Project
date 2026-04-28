#!/usr/bin/env bash
# Base student baseline eval：Qwen3-1.7B-Base on GSM8K
# 期望 acc：~10-25%（远低于 instruct 75%，给 OPD/SFT 留头空间）
set -euo pipefail
cd "$(dirname "$0")/.."
N=${N:-500}

CUDA_VISIBLE_DEVICES=2 python src/eval_gsm8k.py \
    --model models/student-base --n "$N" --batch_size 16 \
    | tee runs/eval/baseline_student_base.log
echo "[baseline-base] done."
