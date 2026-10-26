#!/bin/bash
if ! pgrep -f run_headless_pipeline.py > /dev/null; then
  echo "Job is not running."
  exit 0
fi
read -p "Stop the job? Finished frames are kept. (y/n) " ans
if [ "$ans" = "y" ]; then
  pkill -f run_headless_pipeline.py
  sleep 2
  echo "Job stopped. Resume anytime with: startjob"
else
  echo "Cancelled. Job still running."
fi
