import cv2, numpy as np, sys
from plyfile import PlyData
from scipy.spatial import cKDTree
cap = cv2.VideoCapture("sample-video/walmartvideo.mp4")
ok, f0 = cap.read(); ok, f1 = cap.read()
g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY).astype(np.float32)
g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32)
shift, _ = cv2.phaseCorrelate(g0, g1)
print("=== VIDEO (frame 0 vs 1) ===")
print(f"Camera shift:            x {shift[0]:.2f} px, y {shift[1]:.2f} px")
print(f"Average pixel change:    {np.abs(g0 - g1).mean():.2f} out of 255")
print(f"Pixels that changed:     {(np.abs(g0 - g1) > 10).mean()*100:.1f}%")
def pos(p):
    v = PlyData.read(p)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
a = pos(sys.argv[1]); b = pos(sys.argv[2])
size = np.median(np.linalg.norm(a, axis=1))
idx = np.random.default_rng(0).choice(len(b), 200000, replace=False)
d, _ = cKDTree(a).query(b[idx])
print("=== SPLATS (matched by position, not by number) ===")
for t in [0.005, 0.01, 0.02]:
    print(f"Has a match within {t*100:g}% of scene size: {(d/size < t).mean()*100:.1f}%")