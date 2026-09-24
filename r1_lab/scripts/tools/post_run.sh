#!/usr/bin/env bash
# Wait for a training run to finish (optional pid), then evaluate the last checkpoint, record videos, write a summary.
#   usage: post_run.sh <run_dir> <label> [pid] [task-play-id]
set -uo pipefail
RUN="$1"; LABEL="$2"; PID="${3:-}"; TASK="${4:-Isaac-Velocity-Flat-R1-Play-v0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES
if [ -n "$PID" ]; then while kill -0 "$PID" 2>/dev/null; do sleep 30; done; sleep 10; fi
LAST=$(ls "$RUN" | grep -aoE "model_[0-9]+\.pt" | sort -t_ -k2 -n | tail -1)
MID1=$(ls "$RUN" | grep -aoE "model_[0-9]+\.pt" | sort -t_ -k2 -n | awk 'NR==int(0.33*NR_TOTAL)+1' NR_TOTAL=$(ls "$RUN" | grep -acE "model_[0-9]+\.pt"))
MID2=$(ls "$RUN" | grep -aoE "model_[0-9]+\.pt" | sort -t_ -k2 -n | awk 'NR==int(0.66*NR_TOTAL)+1' NR_TOTAL=$(ls "$RUN" | grep -acE "model_[0-9]+\.pt"))
{
echo "=== $LABEL finished $(date)  run dir: $RUN  last checkpoint: $LAST ==="
echo; echo "=== evaluation of final checkpoint (200 robots, 10 s, no pushes) ==="
timeout 900 python scripts/rsl_rl/eval_policy.py --task "$TASK" --num_envs 200 --checkpoint "$RUN/$LAST" --eval_steps 500 > "logs/eval_${LABEL}.log" 2>&1
sed -n '/=== policy evaluation ===/,$p' "logs/eval_${LABEL}.log" | grep -avE "Warning|Shutting|^\s*$|SimulationContext"
echo; echo "=== videos ==="
bash scripts/tools/record_videos.sh "$RUN" "$TASK" "$MID1" "$MID2" "$LAST" 2>&1 | grep -E "^==|->|!!"
} > "logs/${LABEL}_summary.txt" 2>&1
