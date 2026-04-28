#!/usr/bin/env bash
# Base student 一键流水线（GPU 0+2）
# 总耗时估算：baseline 8 min → OPD 9.3h ‖ SFT 42 min 并行 ≈ 9.3h
#         → final eval 30 min ≈ 总 ~10 h
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/40_baseline_eval_base.sh

bash scripts/50_train_opd_base.sh &
PID_OPD=$!
sleep 60
bash scripts/51_train_sft_base.sh &
PID_SFT=$!
wait "$PID_OPD" "$PID_SFT"

bash scripts/60_eval_finals_base.sh
python src/plot_results.py
echo "[run_all_base] done — see runs/eval/*_base_*.log"
