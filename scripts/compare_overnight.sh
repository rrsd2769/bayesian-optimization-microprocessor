#!/bin/zsh
# Runs the Phase 6 comparison for a fixed amount of AWAKE time, as consecutive chunks of at most 55 minutes.
# Time is counted in 30 s ticks of `sleep`, which does not advance while the Mac is asleep, so a sleep pauses the
# budget instead of exhausting it. Resumable: finished runs are skipped. A run in progress at the hard stop is lost
# (results are written atomically). Needs no network. Keep the Mac on power: caffeinate -s only works on AC.
# Usage: scripts/compare_overnight.sh [hours, default 7]
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore
HOURS=${1:-7}
TOTAL_TICKS=$(( HOURS * 120 ))
TICKFILE=$(mktemp)
echo 0 > $TICKFILE
echo "start $(date): $HOURS h of awake time"
( while true; do sleep 30; echo $(( $(cat $TICKFILE) + 1 )) > $TICKFILE
    [ $(cat $TICKFILE) -ge $TOTAL_TICKS ] && pkill -f "src.pipeline" && break; done ) &
WATCHER=$!
while true; do
  LEFT=$(( (TOTAL_TICKS - $(cat $TICKFILE)) / 2 ))
  [ $LEFT -lt 3 ] && break
  CHUNK=$(( LEFT < 55 ? LEFT : 55 ))
  echo "=== chunk of $CHUNK min, $(date) ==="
  caffeinate -ims .venv/bin/python -m src.pipeline --minutes $CHUNK compare --seeds 1-20 2>&1 | grep -v -E "INFO|Warning|warn"
  [ "$(ls results/v2/runs/*/seed*.csv 2>/dev/null | wc -l)" -ge 120 ] && echo "all 120 runs done" && break
done
kill $WATCHER 2>/dev/null
rm -f $TICKFILE
find results/v2/runs -name '*.tmp' -delete 2>/dev/null
echo "finished $(date): $(ls results/v2/runs/*/seed*.csv 2>/dev/null | wc -l) of 120 runs on disk"
