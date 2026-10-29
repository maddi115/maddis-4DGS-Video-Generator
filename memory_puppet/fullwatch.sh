#!/bin/bash
N=${1:-walmartfull}; ST=/workspace/${N}_status.txt; LOG=/workspace/${N}_convert.log
D=/workspace/Chronosplat/data/$N; RAW=/workspace/Chronosplat/raw_data/$N
while true; do
  clear
  echo "################  $N  -  $(date +%H:%M)  ################"
  if [ ! -f "$ST" ]; then
    echo "STEP 0: SETTING UP THE POD"; cat /workspace/setup_status.txt 2>/dev/null
    grep -q FAILED /workspace/setup_status.txt 2>/dev/null && { echo; echo "--- last error lines ---"; tail -15 /workspace/setup.log; }
  elif ! grep -q "converting" "$ST"; then
    echo "STEP 1 of 2: BUILDING"; head -8 "$ST" | tail -7
    B=$(grep -oP 'about \K[0-9]+' "$ST" | head -1); T=$(grep -oP 'frames : [0-9]+ / \K[0-9]+' "$ST")
    [ -n "$B" ] && [ -n "$T" ] && echo "whole job : about $(( B + T * 162 / 6000 )) min left (build + convert)"
  elif ! grep -q "ALL DONE\|FAILED" "$ST"; then
    TOTAL=$(ls $RAW/*.ply 2>/dev/null | wc -l)
    if grep -q "converting stores" "$ST"; then echo "STEP 2 of 2: CONVERTING - stores (almost done)"
    else
      DONE=$(find $D -maxdepth 1 -name '*.sog' 2>/dev/null | wc -l)
      echo "STEP 2 of 2: CONVERTING people frames"; echo "frames    : $DONE / $TOTAL"
      if [ "$DONE" -gt 1 ]; then
        FIRST=$(find $D -maxdepth 1 -name '*.sog' -printf '%T@\n' | sort -n | head -1 | cut -d. -f1)
        EL=$(( $(date +%s) - FIRST )); LEFT=$(( EL * (TOTAL - DONE) / (DONE - 1) ))
        echo "speed     : $(awk "BEGIN{printf \"%.2f\", $EL/($DONE-1)}") s per frame"; echo "left      : about $((LEFT/60)) min"
      fi
      echo "size      : $(du -sh $D 2>/dev/null | cut -f1)"
    fi
  else
    grep -q FAILED "$ST" && echo "!!! SOMETHING FAILED - paste this screen to Claude !!!" || echo "*** ALL DONE ***"; tail -6 "$ST"
  fi
  echo "-------------------------------------------------"
  [ -f "$ST" ] && echo "checks    : $(grep -oP 'checks : \K.*' "$ST")"
  echo "disk      : $(df -h /workspace | awk 'NR==2{print $4" free"}')"
  sleep 10
done
