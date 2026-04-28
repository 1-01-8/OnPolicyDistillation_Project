#!/usr/bin/env bash
# lmbda 扫描：每个 150 step / ga=2 (eff_batch=4 加快)，单步 ~15 s → 每个 ~38 min
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,2
export TOKENIZERS_PARALLELISM=false

for L in 0.0 0.5 1.0; do
    OUT="runs/abl/lmbda_${L}"
    [ -d "$OUT/final" ] && { echo "[abl] skip $L"; continue; }
    echo "[abl] training lmbda=$L → $OUT"
    python src/opd_train.py \
        --lmbda "$L" --beta 0.5 --temperature 0.9 \
        --max_steps 150 --max_new_tokens 256 \
        --per_device_batch 2 --grad_accum 2 \
        --attn sdpa \
        --output_dir "$OUT" --run_name "opd-abl-lmbda${L}"
done

for L in 0.0 0.5 1.0; do
    CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py \
        --model models/student --lora "runs/abl/lmbda_${L}/final" \
        --n 500 --batch_size 16 \
        | tee "runs/eval/abl_lmbda_${L}.log"
done
