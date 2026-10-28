import cv2, numpy as np, glob, os, sys
from plyfile import PlyData
D = sys.argv[1]; START = int(sys.argv[2]); KEY_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 10
VIDEO = "sample-video/walmartvideo.mp4"; F = 1303.9; THRESH = 12
files = sorted(glob.glob(os.path.join(D, "frame_*_standard.ply")))
cap = cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES, START)
frames = [cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY) for _ in files]
H, W = frames[0].shape
orb = cv2.ORB_create(4000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
k3 = np.ones((3, 3), np.uint8)
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
_, sx, sy, sz = best
N = len(P0); stored = 0; ref = None
for t in range(len(files)):
    if t % KEY_EVERY == 0:
        ref = frames[t]; kr, dr = orb.detectAndCompute(ref, None)
        stored += N; continue
    kt, dt = orb.detectAndCompute(frames[t], None)
    m = sorted(bf.match(dr, dt), key=lambda x: x.distance)[:1500]
    src = np.float32([kr[x.queryIdx].pt for x in m]); dst = np.float32([kt[x.trainIdx].pt for x in m])
    Hm, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    rw = cv2.warpPerspective(ref, Hm, (W, H))
    valid = cv2.warpPerspective(np.ones_like(ref), Hm, (W, H))
    hi = cv2.dilate(rw, k3); lo = cv2.erode(rw, k3)
    f = frames[t].astype(np.int16)
    diff = np.maximum(f - hi.astype(np.int16), lo.astype(np.int16) - f)
    changed = ((diff > THRESH) | (valid == 0)).astype(np.uint8)
    P = pos(files[t]); u, v, ok = project(P, sx, sy, sz)
    mov = np.ones(len(P), bool)
    mov[ok] = changed[v[ok].astype(int), u[ok].astype(int)] > 0
    stored += mov.sum()
    if t % 10 == 5:
        print(f"Frame {START+t}: changed pixels {changed.mean()*100:.1f}% | moving splats {mov.mean()*100:.1f}%")
full = N * len(files)
print(f"\nFrames: {len(files)} | background refreshed every {KEY_EVERY} frames")
print(f"SAVINGS: {(1 - stored/full)*100:.1f}%  (about {full/stored:.1f}x smaller)")

