#!/bin/bash
OUT=/workspace/Chronosplat/data/meowvideo
LOG=/workspace/maddis-4DGS-Video-Generator/convert.log
TOTAL=2204
RED='\e[1;31m'; GRN='\e[1;32m'; NC='\e[0m'
fmt() { printf "%dh %02dm" $(($1/3600)) $(($1%3600/60)); }

echo "=== CONVERTER (PLY -> SOG) ==="
done=$(find $OUT -name '*.sog' 2>/dev/null | wc -l)
echo "Frames:  $done / $TOTAL ($((done*100/TOTAL))%)"

if pgrep -f "tools/convert.py" > /dev/null; then
  echo -e "Status:  ${GRN}RUNNING${NC}"
  rate=$(find $OUT -name '*.sog' -mmin -1 2>/dev/null | wc -l)
  if [ "$rate" -gt 0 ]; then
    left=$(( (TOTAL - done) / rate ))
    echo "Speed:   $rate frames/min | about $(fmt $((left*60))) left | done around $(TZ=America/Los_Angeles date -d "+$left min" '+%I:%M %p')"
  else
    echo "Speed:   measuring... (still checking frames, or no frame finished in the last minute)"
  fi
else
  echo -e "Status:  ${RED}STOPPED${NC}"
fi

STAMPS=$(find $OUT -name '*.sog' -printf '%T@\n' 2>/dev/null | cut -d. -f1 | sort -n)
first=$(echo "$STAMPS" | head -1); last=$(echo "$STAMPS" | tail -1)
if [ -n "$first" ]; then
  if pgrep -f "tools/convert.py" > /dev/null; then
    echo "Time:    $(fmt $(( $(date +%s) - first ))) so far"
  else
    echo "Time:    $(fmt $((last - first))) total"
  fi
fi

echo "Size:    $(du -sh $OUT 2>/dev/null | cut -f1) converted | $(df -h / | awk 'NR==2{print $4}') free"

zero=$(find $OUT -name '*.sog' -size 0 2>/dev/null | wc -l)
if [ "$zero" -gt 0 ]; then echo -e "0-byte:  ${RED}$zero empty .sog files${NC}"; else echo -e "0-byte:  ${GRN}none${NC}"; fi

errs=$(grep -ci "error\|traceback\|failed" $LOG 2>/dev/null)
if [ "$errs" -gt 0 ]; then echo -e "Errors:  ${RED}$errs lines in convert.log${NC}"; else echo -e "Errors:  ${GRN}none${NC}"; fi

echo "Last:    $(grep -v '^\s*$' $LOG 2>/dev/null | tail -n 1)"
echo
echo "Commands:"
echo "  convertprogress   Check converter once"
echo "  convertlive       Keep checking every 5 seconds"
