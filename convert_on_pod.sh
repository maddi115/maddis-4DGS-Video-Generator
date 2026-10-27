#!/bin/bash
SRC=/workspace/maddis-4DGS-Video-Generator/output_3dgs/meowvideo
CHRONO=/workspace/Chronosplat
NAME=meowvideo

if pgrep -f run_headless_pipeline.py > /dev/null && [[ " $* " != *" --dry-run "* ]]; then
  echo "Frame job is still running. Wait until it's done (or add --dry-run to test)."
  exit 1
fi

if pgrep -f "tools/convert.py" > /dev/null; then
  echo "Converter is already running. Not starting a second copy."
  exit 1
fi

mkdir -p $CHRONO/raw_data/$NAME
for f in $SRC/frame_*_standard.ply; do
  n=$(basename "$f" | grep -o '^frame_[0-9]\{6\}')
  ln -sf "$f" "$CHRONO/raw_data/$NAME/$n.ply"
done
echo "Linked $(ls $CHRONO/raw_data/$NAME | wc -l) frames"

cd $CHRONO && bash tools/seq.sh $NAME --source-fps 60 --workers 12 "$@"
find /workspace/Chronosplat/data/$NAME -name ".*.tmp" -delete 2>/dev/null; echo "Cleaned up temp files."
