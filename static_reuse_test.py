import cv2, numpy as np, glob, os
from plyfile import PlyData
VIDEO = "sample-video/walmartvideo.mp4"; D = "output_3dgs/proto60"; F = 1303.9
THRESH = 12; DILATE = 7
files = sorted(glob.glob(os.path.join(D, "frame_*_standard.ply")))
cap = cv2.VideoCapture(VIDEO)
frames = []
for i in range(len(files)):
    ok, f = cap.read(); frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
H, W = frames[0].shape
orb = cv2.ORB_create(4000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
k0, d0 = orb.detectAndCompute(frames[0], None)
def pos(p):
    v = PlyData.read(p)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
def project(P, sx, sy, sz):
    z = P[:, 2] * sz
    u = F * P[:, 0] * sx / z + W / 2; v = F * P[:, 1] * sy / z + H / 2
    return u, v, (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
P0 = pos(files[0]); best = None
for sx in (1, -1):
    for sy in (1, -1):
        for sz in (1, -1):
            r = project(P0, sx, sy, sz)[2].mean()
            if best is None or r > best[0]: best = (r, sx, sy, sz)
r, sx, sy, sz = best
print(f"Self-check: {r*100:.1f}% of splats land inside the picture (want 90%+)")
kernel = np.ones((DILATE, DILATE), np.uint8)
moving_total = 0; n = len(files)
for t in range(1, n):
    kt, dt = orb.detectAndCompute(frames[t], None)
    m = sorted(bf.match(d0, dt), key=lambda x: x.distance)[:1000]
    src = np.float32([k0[x.queryIdx].pt for x in m]); dst = np.float32([kt[x.trainIdx].pt for x in m])
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    f0w = cv2.warpAffine(frames[0], M, (W, H))
    valid = cv2.warpAffine(np.ones_like(frames[0]), M, (W, H))
    changed = ((cv2.absdiff(frames[t], f0w) > THRESH) | (valid == 0)).astype(np.uint8)
    changed = cv2.dilate(changed, kernel)
    P = pos(files[t]); u, v, ok = project(P, sx, sy, sz)
    mov = np.ones(len(P), bool)
    mov[ok] = changed[v[ok].astype(int), u[ok].astype(int)] > 0
    moving_total += mov.sum()
    if t == 1 or t % 10 == 0:
        print(f"Frame {t}: changed pixels {changed.mean()*100:.1f}% | moving splats {mov.mean()*100:.1f}%")
N = len(P0); full = N * n; new = N + moving_total
print(f"\nAll {n} frames stored fully:   {full:,} splats")
print(f"Background once + moving only: {new:,} splats")
print(f"SAVINGS: {(1 - new/full)*100:.1f}%  (about {full/new:.1f}x smaller)")