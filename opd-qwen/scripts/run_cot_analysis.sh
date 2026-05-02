#!/usr/bin/env bash
# CoT 分析全流程
# 输出: runs/cot/*.jsonl + runs/cot/metrics_summary.csv + figs/cot_*.png
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/cot

# ============ Phase 1: dump CoT ============
# 注：每个模型在 GSM8K test 子集 (n=500) 上 greedy 生成一次 → 主指标
#     OPD/SFT 各再做 5-sample temperature=0.7 → diversity 用

DEV=${DEV:-cuda:0}
N=${N:-500}

# 1. baseline (Qwen3-1.7B-Base, no LoRA) — greedy
CUDA_VISIBLE_DEVICES=0 python src/dump_cot.py \
    --model models/student-base \
    --tag baseline_base \
    --out runs/cot/baseline_base.jsonl \
    --n "$N" --batch_size 16 --device "$DEV"

# 2. SFT-base — greedy
CUDA_VISIBLE_DEVICES=0 python src/dump_cot.py \
    --model models/student-base \
    --lora runs/sft-qwen3-1.7b-base/final \
    --tag sft_base \
    --out runs/cot/sft_base.jsonl \
    --n "$N" --batch_size 16 --device "$DEV"

# 3. OPD-base ckpt-400 — greedy
CUDA_VISIBLE_DEVICES=0 python src/dump_cot.py \
    --model models/student-base \
    --lora runs/opd-qwen3-1.7b-base/checkpoint-400 \
    --tag opd_base \
    --out runs/cot/opd_base.jsonl \
    --n "$N" --batch_size 16 --device "$DEV"

# 4. teacher (Qwen3-8B bf16 via auto device) — greedy (作为风格参考)
CUDA_VISIBLE_DEVICES=0 python src/dump_cot.py \
    --model models/teacher \
    --tag teacher \
    --out runs/cot/teacher.jsonl \
    --n "$N" --batch_size 8 --device "$DEV"

# ============ Phase 1b: diversity ============
# 同一题 5 次 sample，对 OPD/SFT 比较探索性
for cfg in \
    "sft_base|runs/sft-qwen3-1.7b-base/final" \
    "opd_base|runs/opd-qwen3-1.7b-base/checkpoint-400"; do
    tag="${cfg%%|*}"; lora="${cfg##*|}"
    CUDA_VISIBLE_DEVICES=0 python src/dump_cot.py \
        --model models/student-base --lora "$lora" \
        --tag "${tag}_div" \
        --out "runs/cot/${tag}_div.jsonl" \
        --n 100 --n_samples 5 --temperature 0.7 --top_p 0.95 \
        --batch_size 8 --device "$DEV"
done

# ============ Phase 2: 指标 + 图 ============
python src/cot_metrics.py \
    --files runs/cot/baseline_base.jsonl runs/cot/sft_base.jsonl \
            runs/cot/opd_base.jsonl runs/cot/teacher.jsonl \
    --tags  baseline sft opd teacher \
    --teacher_tag teacher

# ============ Phase 3 (可选): LLM-as-judge 错误分类 ============
if [[ "${RUN_JUDGE:-0}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=0 python src/cot_judge.py \
        --files runs/cot/sft_base.jsonl runs/cot/opd_base.jsonl \
        --tags  sft opd \
        --max_per_tag 100
fi

echo "[cot] all done — see runs/cot/ + figs/cot_*.png"
