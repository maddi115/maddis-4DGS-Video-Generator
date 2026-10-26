#!/bin/bash
while true; do
  clear
  /workspace/maddis-4DGS-Video-Generator/check.sh
  echo
  echo "Updates every 5s."
  echo "Ctrl + C exits live but never stops the job."
  sleep 5
done
