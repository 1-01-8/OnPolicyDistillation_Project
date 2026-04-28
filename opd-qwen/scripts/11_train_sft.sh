#!/usr/bin/env bash
# SFT 单卡，独占物理 GPU 2（OPD 的 trainer 实际不用 GPU 2，所以可与 OPD 并行）
# 实测：1.45 s/step；compute-matched 对照 = OPD 300 step × 1.76× FLOPs ≈ 540 step
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

python src/sft_baseline.py \
    --max_steps "${MAX_STEPS:-540}" \
    --per_device_batch 2 --grad_accum 4 \
    --max_length 1024 \
    --attn sdpa \
    --output_dir runs/sft-qwen3-1.7b \
    --run_name sft-qwen3-1.7b-gsm8k \
    "$@"
