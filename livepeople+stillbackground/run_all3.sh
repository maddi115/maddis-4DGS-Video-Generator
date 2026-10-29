#!/bin/bash
NAME="$3"; ST="/workspace/${NAME}_status.txt"
cd /workspace/maddis-4DGS-Video-Generator || exit 1
python /workspace/segment_run3.py "$1" "$2" "$NAME" > "/workspace/${NAME}.log" 2>&1 || { echo "BUILD FAILED - see /workspace/${NAME}.log" >> "$ST"; exit 1; }
cd /workspace/Chronosplat || exit 1
echo "converting people frames..." >> "$ST"
bash tools/seq.sh "$NAME" --source-fps 30 --min-alpha 0.001 > "/workspace/${NAME}_convert.log" 2>&1 || { echo "CONVERT FAILED - see /workspace/${NAME}_convert.log" >> "$ST"; exit 1; }
echo "converting stores..." >> "$ST"
bash tools/seq.sh "${NAME}_bg" --source-fps 30 --min-alpha 0.001 >> "/workspace/${NAME}_convert.log" 2>&1 || { echo "CONVERT FAILED (stores) - see /workspace/${NAME}_convert.log" >> "$ST"; exit 1; }
python3 tools/add_background.py "$NAME" >> "/workspace/${NAME}_convert.log" 2>&1 || { echo "ATTACH FAILED - see /workspace/${NAME}_convert.log" >> "$ST"; exit 1; }
tail -4 "/workspace/${NAME}_convert.log" | sed 's/^/   /' >> "$ST"
echo "ALL DONE - download /workspace/Chronosplat/data/${NAME} to your PC" >> "$ST"
