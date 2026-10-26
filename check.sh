#!/bin/bash
cd /workspace/maddis-4DGS-Video-Generator
done=$(ls output_3dgs/meowvideo 2>/dev/null | wc -l)
echo "$done / 2204 frames ($((done*100/2204))%)"
if pgrep -f run_headless_pipeline.py > /dev/null; then echo "Status: RUNNING"; else echo "Status: STOPPED"; fi
tail -n 1 run.log
