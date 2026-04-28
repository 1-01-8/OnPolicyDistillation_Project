#!/usr/bin/env bash
# 一键流水线（GPU 0+2，跳过 GPU 3）
# 总耗时估算（如 GPU 3 闲置不再变化）：
#   baseline 25 min → 并行(OPD 3h ‖ SFT 13min) ≈ 3h →
#   final eval 30 min → ablation ~2.5 h → 出图
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/00_baseline_eval.sh

bash scripts/10_train_opd.sh &
PID_OPD=$!
sleep 60
bash scripts/11_train_sft.sh &
PID_SFT=$!
wait "$PID_OPD" "$PID_SFT"

bash scripts/20_eval_finals.sh
bash scripts/30_ablation_lmbda.sh
python src/plot_results.py
echo "[run_all] done — 看 figs/ 与 runs/eval/*.log"
