#!/usr/bin/env bash
# 提前停训后的收尾流水：ckpt-400 full eval + SFT-base 训练 + SFT-base full eval
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/eval

echo "[finalize] $(date) — start"

# 阶段 A：并行 — OPD ckpt-400 full eval (GPU 0) ‖ SFT-base 训练 (GPU 2)
CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py \
    --model models/student-base \
    --lora runs/opd-qwen3-1.7b-base/checkpoint-400 \
    --n -1 --batch_size 8 --device cuda:0 \
    > runs/eval/opd_base_final.log 2>&1 &
PID_E=$!
echo "[finalize] OPD ckpt-400 full eval PID=$PID_E (GPU 0, ~30min)"

sleep 30
CUDA_VISIBLE_DEVICES=2 python src/sft_baseline.py \
    --student models/student-base \
    --lr 2e-5 \
    --max_steps 1760 \
    --per_device_batch 2 --grad_accum 4 \
    --max_length 1024 --attn sdpa \
    --output_dir runs/sft-qwen3-1.7b-base \
    --run_name sft-qwen3-1.7b-base-gsm8k \
    > runs/sft_base_train.log 2>&1 &
PID_S=$!
echo "[finalize] SFT-base train PID=$PID_S (GPU 2, ~42min)"

wait "$PID_E"
echo "[finalize] OPD eval done — $(tail -1 runs/eval/opd_base_final.log)"
wait "$PID_S"
echo "[finalize] SFT-base train done"

# 阶段 B：SFT-base full eval (GPU 0)
CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py \
    --model models/student-base \
    --lora runs/sft-qwen3-1.7b-base/final \
    --n -1 --batch_size 8 --device cuda:0 \
    > runs/eval/sft_base_final.log 2>&1
echo "[finalize] SFT-base eval done — $(tail -1 runs/eval/sft_base_final.log)"

echo "[finalize] $(date) — ALL DONE"
