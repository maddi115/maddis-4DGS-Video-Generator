#!/bin/bash
cd "$(dirname "$0")"
nohup python3 -u run_headless_pipeline.py > run.log 2>&1 &
echo "Started. Watch with: tail -f run.log"
