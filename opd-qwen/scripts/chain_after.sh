#!/usr/bin/env bash
# 等待 PID 结束后接着跑 run_all_base.sh
# 用法：bash scripts/chain_after.sh <PID>
#       默认 PID 取当前 run_all.sh 的进程
set -uo pipefail
cd "$(dirname "$0")/.."

WAIT_PID="${1:-}"
if [[ -z "$WAIT_PID" ]]; then
  WAIT_PID=$(pgrep -f 'bash scripts/run_all.sh' | head -1 || true)
fi
if [[ -z "$WAIT_PID" ]]; then
  echo "[chain] 没找到要等的 PID，立刻启动 run_all_base.sh"
else
  echo "[chain] 等 PID=$WAIT_PID 结束 ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[chain] PID=$WAIT_PID 已结束 $(date '+%F %T')"
fi

LOG="runs/chain_run_all_base_$(date +%Y%m%d_%H%M%S).log"
echo "[chain] 启动 run_all_base.sh，日志 → $LOG"
mkdir -p runs
exec bash scripts/run_all_base.sh > "$LOG" 2>&1
