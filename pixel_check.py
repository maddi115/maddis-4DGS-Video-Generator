import cv2, numpy as np
cap = cv2.VideoCapture("sample-video/walmartvideo.mp4")
def frame(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
for i in [0, 100, 1000]:
    a, b = frame(i), frame(i + 1)
    bright = b.mean() - a.mean()
    b_fixed = b - bright
    print(f"Frames {i}/{i+1}: brightness jump {bright:+.2f} | changed before fix {(np.abs(a-b)>10).mean()*100:.1f}% | after brightness fix {(np.abs(a-b_fixed)>10).mean()*100:.1f}%")
    if i == 100:
        heat = cv2.applyColorMap(np.clip(np.abs(a - b_fixed) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite("diff_100.png", heat)
print("Saved diff_100.png  (blue = no change, red = changed)")