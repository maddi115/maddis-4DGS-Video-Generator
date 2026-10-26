#!/bin/bash
DIR=/workspace/maddis-4DGS-Video-Generator
OUT=$DIR/output_3dgs/meowvideo
RED='\e[1;31m'; GRN='\e[1;32m'; NC='\e[0m'
cd $DIR

FILES=$(ls $OUT/*_standard.ply 2>/dev/null)
done=$(echo "$FILES" | grep -c . )
echo "Frames:  $done / 2204 ($((done*100/2204))%)"

if pgrep -f run_headless_pipeline.py > /dev/null; then
  echo -e "Status:  ${GRN}RUNNING${NC}"
else
  echo -e "Status:  ${RED}STOPPED${NC}"
fi

echo "Size:    $(du -sh $OUT 2>/dev/null | cut -f1) used | $(df -h / | awk 'NR==2{print $4}') free"

zero=$(find $OUT -name '*_standard.ply' -size 0 2>/dev/null)
if [ -n "$zero" ]; then
  echo -e "${RED}0-BYTE FILES FOUND:${NC}"
  echo "$zero" | xargs -n1 basename
else
  echo -e "0-byte:  ${GRN}none${NC}"
fi

sizes=$(echo "$FILES" | head -n -1 | xargs -r stat -c %s | sort | uniq -c)
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
