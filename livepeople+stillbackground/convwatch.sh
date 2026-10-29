#!/bin/bash
N=${1:-walmart10s3}; TOTAL=${2:-300}; LOG=/workspace/${N}_convert.log; D=/workspace/Chronosplat/data/$N
while true; do
  clear
  DONE=$(find $D -maxdepth 1 -name '*.sog' 2>/dev/null | wc -l)
  echo "=== converting $N ==="
  echo "frames : $DONE / $TOTAL"
  if [ "$DONE" -gt 1 ]; then
    FIRST=$(find $D -maxdepth 1 -name '*.sog' -printf '%T@\n' | sort -n | head -1 | cut -d. -f1)
    EL=$(( $(date +%s) - FIRST ))
    LEFT=$(( EL * (TOTAL - DONE) / (DONE - 1) ))
    echo "time   : $((EL/60)) min $((EL%60)) s so far"
    echo "speed  : $(awk "BEGIN{printf \"%.2f\", $EL/($DONE-1)}") s per frame"
    echo "left   : about $((LEFT/60)) min $((LEFT%60)) s"
  fi
  echo "size   : $(du -sh $D 2>/dev/null | cut -f1)"
  echo "--- last log lines ---"; tail -3 "$LOG"
  echo; tail -2 /workspace/${N}_status.txt
  sleep 5
done
