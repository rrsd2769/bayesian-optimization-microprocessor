#!/bin/zsh
# Runs the Phase 6 comparison as consecutive 55-minute chunks. Resumable: finished runs are skipped, so rerunning
# the script (or stopping it) loses at most the run in progress. Needs no network.
# Usage: scripts/compare_overnight.sh [number of chunks, default 12]
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore
for i in $(seq 1 ${1:-12}); do
  echo "=== chunk $i $(date) ==="
  caffeinate -i .venv/bin/python -m src.pipeline --minutes 55 compare --seeds 1-20 2>&1 | grep -v -E "INFO|Warning|warn"
  [ "$(ls results/v2/runs/*/seed*.csv 2>/dev/null | wc -l)" -ge 120 ] && echo "all 120 runs done" && break
done
