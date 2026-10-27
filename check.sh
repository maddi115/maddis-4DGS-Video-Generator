#!/bin/bash
DIR=/workspace/maddis-4DGS-Video-Generator
OUT=$DIR/output_3dgs/meowvideo
TOTAL=2204
RED='\e[1;31m'; GRN='\e[1;32m'; NC='\e[0m'
cd $DIR

FILES=$(ls $OUT/*_standard.ply 2>/dev/null)
done=$(echo "$FILES" | grep -c . )
echo "Frames:  $done / $TOTAL ($((done*100/TOTAL))%)"

if pgrep -f run_headless_pipeline.py > /dev/null; then
  echo -e "Status:  ${GRN}RUNNING${NC}"
  rate=$(find $OUT -name '*_standard.ply' -mmin -1 | wc -l)
  if [ "$rate" -gt 0 ]; then
    left=$(( (TOTAL - done) / rate ))
    echo "Speed:   $rate frames/min | about $((left/60))h $((left%60))m left | done around $(TZ=America/Los_Angeles date -d "+$left min" '+%I:%M %p')"
  else
    echo "Speed:   measuring... (no frames finished in the last minute)"
  fi
else
  echo -e "Status:  ${RED}STOPPED${NC}"
fi

fmt() { printf "%dh %02dm" $(($1/3600)) $(($1%3600/60)); }
PID=$(pgrep -f run_headless_pipeline.py | head -1)
STAMPS=$(find $OUT -name '*_standard.ply' -printf '%T@\n' 2>/dev/null | cut -d. -f1 | sort -n)
first=$(echo "$STAMPS" | head -1)
last=$(echo "$STAMPS" | tail -1)
if [ -n "$first" ]; then
  if [ -n "$PID" ]; then
    total=$(( $(date +%s) - first ))
    run=$(ps -o etimes= -p $PID | tr -d ' ')
    echo "Time:    $(fmt $total) total | $(fmt $run) this run"
  else
    echo "Time:    $(fmt $((last - first))) total"
  fi
fi

echo "Size:    $(du -sh $OUT 2>/dev/null | cut -f1) used | $(df -h / | awk 'NR==2{print $4}') free"

zero=$(find $OUT -name '*_standard.ply' -size 0 2>/dev/null)
if [ -n "$zero" ]; then
  echo -e "${RED}0-BYTE FILES FOUND:${NC}"
  echo "$zero" | xargs -n1 basename
else
  echo -e "0-byte:  ${GRN}none${NC}"
fi

sizes=$(echo "$FILES" | xargs -r stat -c %s | sort | uniq -c)
if [ $(echo "$sizes" | grep -c .) -gt 1 ]; then
  echo -e "${RED}SIZE MISMATCH (a frame may be cut off):${NC}"
  echo "$sizes"
else
  echo -e "Sizes:   ${GRN}all match${NC} ($(echo "$sizes" | awk '{print $2}') bytes each)"
fi

bad=$(grep -c '\[BAD\]' run.log 2>/dev/null)
if [ "$bad" -gt 0 ]; then echo -e "Bad:     ${RED}$bad frames caught and deleted${NC}"; else echo -e "Bad:     ${GRN}0${NC}"; fi

echo "Last:    $(grep "^\[" run.log | tail -n 1)"

echo
echo "Commands:"
echo "  startjob   Start or resume (never duplicates)"
echo "  stopjob    Stop the job (asks first)"
echo "  progress   Check once"
echo "  live       Keep checking every 5 seconds"
