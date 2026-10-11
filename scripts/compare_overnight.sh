#!/bin/zsh
# Runs the Phase 6 comparison for a fixed wall-clock time, as consecutive chunks of at most 55 minutes.
# Resumable: finished runs are skipped, so rerunning continues. A run in progress at the hard stop is lost
# (results are written atomically, so nothing half-written is left). Needs no network.
# Usage: scripts/compare_overnight.sh [hours, default 7]
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore
HOURS=${1:-7}
START=$(date +%s)
END=$(( START + HOURS * 3600 ))
echo "start $(date), hard stop $(date -r $END)"
( while [ $(date +%s) -lt $END ]; do sleep 30; done; pkill -f "src.pipeline" ) &
WATCHER=$!
while true; do
  LEFT=$(( (END - $(date +%s)) / 60 ))
  [ $LEFT -lt 3 ] && break
  CHUNK=$(( LEFT < 55 ? LEFT : 55 ))
  echo "=== chunk of $CHUNK min, $(date) ==="
  caffeinate -i .venv/bin/python -m src.pipeline --minutes $CHUNK compare --seeds 1-20 2>&1 | grep -v -E "INFO|Warning|warn"
  [ "$(ls results/v2/runs/*/seed*.csv 2>/dev/null | wc -l)" -ge 120 ] && echo "all 120 runs done" && break
done
kill $WATCHER 2>/dev/null
find results/v2/runs -name '*.tmp' -delete 2>/dev/null
echo "finished $(date): $(ls results/v2/runs/*/seed*.csv 2>/dev/null | wc -l) of 120 runs on disk"
