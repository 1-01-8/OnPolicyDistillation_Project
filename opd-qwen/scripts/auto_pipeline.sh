#!/usr/bin/env bash
# 自动化 pipeline：
#   1) 等当前 instruct_sft/instruct_opd CoT dump 完成
#   2) 重画 base+instruct 6-way CoT 图（不含 sft_then_opd）
#   3) merge 1.7B-Base + SFT LoRA
#   4) 在合并后的 student 上跑 OPD 训练 (max_steps=400, 与主线同条件)
#   5) full eval (n=1319) + CoT dump
#   6) 重画 7-way CoT 图
#   7) 写 RESULTS.md / README.md / COT_PLAN.md 增补
#   8) git commit
#
# 任意一步失败 → 写 PIPE_STATE=fail 并退出，不影响已完成步骤
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LOG="$ROOT/logs/auto_pipeline.log"
STATE="$ROOT/runs/.pipeline_state"
PY="$ROOT/.venv/bin/python"
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs runs runs/eval
exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] === pipeline start ==="
echo "PHASE_1_INSTRUCT_DUMP" > "$STATE"

#---- Phase 1: wait for instruct variants dump ----
WATCH_LOG="$ROOT/logs/cot_instruct_variants.log"
while pgrep -f "dump_cot.*instruct" >/dev/null 2>&1; do
    sleep 30
done
# 验证
for tag in instruct_sft instruct_opd; do
    f="$ROOT/runs/cot/$tag.jsonl"
    [ -s "$f" ] && [ "$(wc -l < "$f")" -ge 1300 ] || {
        echo "[FAIL] $f missing or short"; echo "PHASE_1_FAIL" > "$STATE"; exit 1; }
done
echo "[$(date '+%F %T')] phase 1 done"

#---- Phase 2: 6-way CoT plots (auto picks all available) ----
echo "PHASE_2_PLOT_6WAY" > "$STATE"
COT_ORDER=auto python src/cot_compare_3way.py
echo "[$(date '+%F %T')] phase 2 done"

#---- Phase 3: merge SFT LoRA ----
echo "PHASE_3_MERGE_LORA" > "$STATE"
CUDA_VISIBLE_DEVICES=0 python src/merge_lora.py \
    --base models/student-base \
    --lora runs/sft-qwen3-1.7b-base/final \
    --out  models/student-base-sft-merged
echo "[$(date '+%F %T')] phase 3 done"

#---- Phase 4: SFT→OPD training (max_steps=1000 启动 lr scheduler，watcher 到 ckpt-400 停训以严格对齐主线) ----
echo "PHASE_4_TRAIN" > "$STATE"
WANDB_MODE=disabled MAX_STEPS=1000 \
    nohup bash scripts/52_train_opd_on_sft.sh > logs/opd_on_sft_train.log 2>&1 &
TRAIN_PID=$!
echo "[$(date '+%F %T')] training started PID=$TRAIN_PID, waiting for checkpoint-400 ..."
# 轮询：ckpt-400 出现且其内 trainer_state.json 完整 → kill -INT
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
    if [ -f runs/opd-on-sft-1.7b-base/checkpoint-400/trainer_state.json ]; then
        echo "[$(date '+%F %T')] checkpoint-400 ready, sending SIGINT to PID=$TRAIN_PID"
        kill -INT $TRAIN_PID || true
        sleep 30
        kill -TERM $TRAIN_PID 2>/dev/null || true
        # 杀子进程
        pkill -P $TRAIN_PID 2>/dev/null || true
        break
    fi
done
wait $TRAIN_PID 2>/dev/null || true
ls runs/opd-on-sft-1.7b-base/checkpoint-400 >/dev/null
echo "[$(date '+%F %T')] phase 4 done"

#---- Phase 5: eval + CoT dump ----
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

#---- Phase 6: final 7-way plots ----
echo "PHASE_6_PLOT_7WAY" > "$STATE"
COT_ORDER=auto python src/cot_compare_3way.py
COT_ORDER=base python src/cot_compare_3way.py   # base 系 5-way 单独图
echo "[$(date '+%F %T')] phase 6 done"

#---- Phase 7: docs ----
echo "PHASE_7_DOCS" > "$STATE"
python src/auto_update_docs.py || echo "[warn] doc update skipped"
echo "[$(date '+%F %T')] phase 7 done"

#---- Phase 8: commit ----
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
