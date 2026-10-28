p = "run_headless_pipeline.py"
s = open(p, encoding="utf-8-sig").read()
old = '    cap = cv2.VideoCapture(str(VIDEO))\n    for i in range(total):'
new = ('    cap = cv2.VideoCapture(str(VIDEO))\n'
       '    START = int(os.environ.get("START_FRAME", "0"))\n'
       '    cap.set(cv2.CAP_PROP_POS_FRAMES, START)\n'
       '    for i in range(START, min(START + total, info["frame_count"])):')
if "START_FRAME" in s:
    print("already patched")
elif old in s:
    open(p, "w", encoding="utf-8").write(s.replace(old, new, 1)); print("patched")
else:
    print("PATCH FAILED - text not found")