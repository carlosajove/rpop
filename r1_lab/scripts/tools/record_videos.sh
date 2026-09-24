#!/usr/bin/env bash
# Record play videos (headless) for selected checkpoints of a training run.
#   usage: record_videos.sh <run_dir> <task-play-id> <checkpoint name>...
#   e.g.   record_videos.sh logs/rsl_rl/r1_flat/2026-09-10_17-29-08 Isaac-Velocity-Flat-R1-Play-v0 model_500.pt model_1499.pt
# Videos land in <run_dir>/videos/play/<checkpoint>.mp4
set -uo pipefail
RUN="$1"; TASK="$2"; shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$ROOT"
for CK in "$@"; do
  echo "== recording $CK"
  timeout 900 python scripts/rsl_rl/play.py --task "$TASK" --num_envs 16 --checkpoint "$RUN/$CK" --video --video_length 400 > "$RUN/videos_play_${CK%.pt}.log" 2>&1
  f=$(ls -t "$RUN"/videos/play/rl-video-step-*.mp4 2>/dev/null | head -1)
  [ -n "$f" ] && mv "$f" "$RUN/videos/play/${CK%.pt}.mp4" && echo "   -> $RUN/videos/play/${CK%.pt}.mp4" || echo "   !! no video produced (see $RUN/videos_play_${CK%.pt}.log)"
done
