#!/bin/bash
ST=/workspace/setup_status.txt; LOG=/workspace/setup.log
step(){ echo "$(date +%H:%M) $1" >> $ST; echo "=== $1" >> $LOG; }
fail(){ echo "$(date +%H:%M) FAILED: $1 - paste this screen to Claude" >> $ST; exit 1; }
: > $ST; : > $LOG; cd /workspace
step "1/7 getting our code from GitHub"
[ -d maddis-4DGS-Video-Generator ] || git clone -b livepeople+stillbackground https://github.com/maddi115/maddis-4DGS-Video-Generator >> $LOG 2>&1 || fail "code download"
step "2/7 SHARP + Node + Chronosplat (about 5 min)"
bash maddis-4DGS-Video-Generator/setup_pod.sh >> $LOG 2>&1 || fail "main setup"
step "3/7 person finder (about 10 min)"
cp maddis-4DGS-Video-Generator/livepeople+stillbackground/setup_body_pod.sh /workspace/
bash /workspace/setup_body_pod.sh >> $LOG 2>&1
python -c "import detectron2" >> $LOG 2>&1 || fail "person finder install"
step "4/7 putting our scripts in place"
M=/workspace/maddis-4DGS-Video-Generator/livepeople+stillbackground; cp $M/fullwatch.sh /workspace/; ln -sf /workspace/fullwatch.sh /usr/local/bin/fullwatch; C=/workspace/Chronosplat
cp $M/segment_run2.py $M/segment_run3.py $M/convwatch.sh /workspace/ && cp $M/add_background.py $C/tools/ || fail "copy scripts"
sed -i 's/    N = max(counts\[k\] for k in keys)/    N = 1 if os.environ.get("NOPAD") else max(counts[k] for k in keys)/' /workspace/segment_run3.py
grep -q 'N = 1 if os.environ.get("NOPAD")' /workspace/segment_run3.py || fail "no-padding patch"
sed -e 's/--source-fps 30/--source-fps 60 --workers 12/g' $M/run_all3.sh > /workspace/run_all60.sh
sed -i 's/f"opacity,gt,{alpha_to_logit(min_alpha):.6f}"/f"opacity,gt,{min_alpha}"/' $C/tools/convert.py
sed -i 's/        if fr.schema.splat_count != ref_count:/        if fr.schema.splat_count != ref_count and not __import__("os").environ.get("NOPAD"):/' $C/tools/convert.py
grep -q 'environ.get("NOPAD")' $C/tools/convert.py || fail "converter patch"
step "5/7 testing SHARP (downloads its model)"
cd /workspace/maddis-4DGS-Video-Generator
python -c "import torch, run_headless_pipeline as rp; rp.load_model(torch.device('cuda')); print('SHARP OK')" >> $LOG 2>&1 || fail "SHARP test"
step "6/7 testing person finder (downloads its model)"
python -c "import sys; sys.path.insert(0,'/workspace/sam-3d-body'); from tools.build_detector import load_detectron2_vitdet; load_detectron2_vitdet(); print('FINDER OK')" >> $LOG 2>&1 || fail "person finder test"
step "7/7 all tests passed - starting the full video"
rm -f /workspace/walmartfull_status.txt /workspace/walmartfull_convert.log
NOPAD=1 STEP=1 SEG=60 nohup bash /workspace/run_all60.sh 0 40 walmartfull > /dev/null 2>&1 &
echo "$(date +%H:%M) STARTED walmartfull" >> $ST
