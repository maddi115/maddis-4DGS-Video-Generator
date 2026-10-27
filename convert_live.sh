#!/bin/bash
while true; do
  clear
  /workspace/maddis-4DGS-Video-Generator/convert_check.sh
  echo
  echo "Updates every 5s."
  echo "Ctrl + C exits convertlive but never stops the converter."
  sleep 5
done
