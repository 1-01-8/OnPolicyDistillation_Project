#!/usr/bin/env bash
# Wait for target PID, then open a new tmux window in the given session
# and start run_all_base.sh inside it.
# Usage: bash scripts/chain_after_tmux.sh <PID> <tmux_session>
set -uo pipefail
cd "$(dirname "$0")/.."
PROJ_DIR="$(pwd)"

WAIT_PID="${1:?need PID}"
SESS="${2:?need tmux session}"

echo "[chain-tmux] $(date '+%F %T') waiting PID=$WAIT_PID, target tmux session=$SESS"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "[chain-tmux] $(date '+%F %T') PID=$WAIT_PID exited."

# Use the same tmux server the user is on
TMUX_BIN=$(command -v tmux)
"$TMUX_BIN" has-session -t "$SESS" 2>/dev/null || {
    echo "[chain-tmux] tmux session '$SESS' missing, falling back to nohup"
    LOG="runs/chain_run_all_base_$(date +%Y%m%d_%H%M%S).log"
    exec bash scripts/run_all_base.sh > "$LOG" 2>&1
}

WIN_NAME="opd-base-$(date +%H%M)"
CMD="cd '$PROJ_DIR' && source .venv/bin/activate && bash scripts/run_all_base.sh; echo '[chain-tmux] run_all_base.sh finished'; exec bash"
"$TMUX_BIN" new-window -t "$SESS" -n "$WIN_NAME" "$CMD"
echo "[chain-tmux] launched tmux window '$SESS:$WIN_NAME'"
