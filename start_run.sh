#!/bin/bash
cd /workspace/maddis-4DGS-Video-Generator
if pgrep -f run_headless_pipeline.py > /dev/null; then
  echo "Already running, not starting a second copy:"
  pgrep -af run_headless_pipeline.py
  exit 0
fi
setsid nohup python3 -u run_headless_pipeline.py >> run.log 2>&1 < /dev/null &
sleep 2
echo "Started in background (PID $(pgrep -f run_headless_pipeline.py))"
echo "Safe to log off. Check with: progress  or  live"
