#!/usr/bin/env bash
# 从 phase 4 续跑（phase 1-3 已完成）
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
STATE="$ROOT/runs/.pipeline_state"
export PATH="$ROOT/.venv/bin:$PATH"
exec >>"$ROOT/logs/auto_pipeline.log" 2>&1
echo "[$(date '+%F %T')] === RESUME from phase 4 ==="

#---- Phase 4 ----
echo "PHASE_4_TRAIN" > "$STATE"
WANDB_MODE=disabled MAX_STEPS=1000 \
    nohup bash scripts/52_train_opd_on_sft.sh > logs/opd_on_sft_train.log 2>&1 &
TRAIN_PID=$!
echo "[$(date '+%F %T')] training started PID=$TRAIN_PID"
sleep 90  # 让训练完成模型加载，避开早期 OOM 假阳性
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
    if [ -f runs/opd-on-sft-1.7b-base/checkpoint-400/trainer_state.json ]; then
        echo "[$(date '+%F %T')] ckpt-400 ready, SIGINT $TRAIN_PID"
        kill -INT $TRAIN_PID || true
        sleep 30
        pkill -P $TRAIN_PID 2>/dev/null || true
        kill -TERM $TRAIN_PID 2>/dev/null || true
        break
    fi
done
wait $TRAIN_PID 2>/dev/null || true
ls runs/opd-on-sft-1.7b-base/checkpoint-400 >/dev/null
echo "[$(date '+%F %T')] phase 4 done"

#---- Phase 5 ----
echo "PHASE_5_EVAL" > "$STATE"
CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py \
    --model models/student-base-sft-merged \
    --lora  runs/opd-on-sft-1.7b-base/checkpoint-400 \
    --n -1 --batch_size 16 \
    | tee runs/eval/opd_on_sft_final.log

echo "PHASE_5b_DUMP" > "$STATE"
CUDA_VISIBLE_DEVICES=0 python src/dump_cot.py \
    --model models/student-base-sft-merged \
    --lora  runs/opd-on-sft-1.7b-base/checkpoint-400 \
    --tag sft_then_opd \
    --out runs/cot/sft_then_opd.jsonl \
    --n -1 --batch_size 32
echo "[$(date '+%F %T')] phase 5 done"

#---- Phase 6 ----
echo "PHASE_6_PLOT_7WAY" > "$STATE"
COT_ORDER=auto python src/cot_compare_3way.py
COT_ORDER=base python src/cot_compare_3way.py
echo "[$(date '+%F %T')] phase 6 done"

#---- Phase 7 ----
echo "PHASE_7_DOCS" > "$STATE"
python src/auto_update_docs.py || echo "[warn] doc update skipped"
echo "[$(date '+%F %T')] phase 7 done"

#---- Phase 8 ----
echo "PHASE_8_COMMIT" > "$STATE"
cd "$ROOT/.."
git add -A
git -c user.email=auto@local -c user.name=auto-pipeline commit \
    -m "auto: SFT→OPD two-stage experiment + 7-way CoT analysis

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" \
    || echo "[info] nothing to commit"
echo "PHASE_8_DONE" > "$STATE"
echo "[$(date '+%F %T')] === pipeline DONE ==="
touch "$ROOT/runs/.pipeline_done"
