import cv2, numpy as np, glob, os, sys
from plyfile import PlyData, PlyElement
D = sys.argv[1]; START = int(sys.argv[2]); OUT = sys.argv[3]
KEY_EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 30
VIDEO = "sample-video/walmartvideo.mp4"; F = 1303.9; THRESH = 12
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(os.path.join(D, "frame_*_standard.ply")))[:int(sys.argv[5]) if len(sys.argv) > 5 else None]
cap = cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES, START)
frames = [cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY) for _ in files]
H, W = frames[0].shape
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]]); Ki = np.linalg.inv(K)
orb = cv2.ORB_create(4000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
k3 = np.ones((3, 3), np.uint8)
feats = [orb.detectAndCompute(f, None) for f in frames]
def homog(a, b):
    ka, da = feats[a]; kb, db = feats[b]
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:1500]
    src = np.float32([ka[x.queryIdx].pt for x in m]); dst = np.float32([kb[x.trainIdx].pt for x in m])
    return cv2.findHomography(src, dst, cv2.RANSAC, 3.0)[0]
def rot_from_h(Hm):
    M = Ki @ Hm @ K; M /= np.cbrt(np.linalg.det(M))
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    return -R if np.linalg.det(R) < 0 else R
def load(p):
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    with open(p, "rb") as f:
        n = 0; props = []; in_vertex = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element"):
                in_vertex = line.startswith("element vertex")
                if in_vertex: n = int(line.split()[-1])
            elif line.startswith("property") and in_vertex:
                _, t, name = line.split()
                props.append((name, types[t]))
            elif line == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(props), count=n)
def xyz(a): return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
P0 = xyz(load(files[0])); best = None
for sx in (1, -1):
    for sy in (1, -1):
        for sz in (1, -1):
            z = P0[:, 2] * sz; u = F * P0[:, 0] * sx / z + W / 2; v = F * P0[:, 1] * sy / z + H / 2
            r = ((z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).mean()
            if best is None or r > best[0]: best = (r, sx, sy, sz)
S = np.array(best[1:], dtype=np.float64)
def project(Pcv):
    z = Pcv[:, 2]; u = F * Pcv[:, 0] / z + W / 2; v = F * Pcv[:, 1] / z + H / 2
    return u, v, (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
def to_global(arr, R):
    out = arr.copy(); g = ((xyz(arr) * S) @ R) * S
    out["x"], out["y"], out["z"] = g[:, 0], g[:, 1], g[:, 2]
    return out
print("Measuring shake for every frame...")
Rs = [np.eye(3)] + [rot_from_h(homog(0, t)) for t in range(1, len(files))]
names = []; stored = 0; counts = []
for t in range(len(files)):
    cur = load(files[t]); name = os.path.join(OUT, f"frame_{START + t:06d}.ply")
    if t % KEY_EVERY == 0:
        key = t; key_arr = cur; ku, kv, kok = project(xyz(cur) * S)
        comp = to_global(cur, Rs[t]); stored += len(cur)
        print(f"Frame {START+t}: KEYFRAME (full background)")
    else:
        Hm = homog(key, t)
        rw = cv2.warpPerspective(frames[key], Hm, (W, H))
        valid = cv2.warpPerspective(np.ones_like(frames[key]), Hm, (W, H))
        hi = cv2.dilate(rw, k3); lo = cv2.erode(rw, k3); f = frames[t].astype(np.int16)
        diff = np.maximum(f - hi.astype(np.int16), lo.astype(np.int16) - f)
        changed = cv2.dilate(((diff > THRESH) | (valid == 0)).astype(np.uint8), np.ones((5, 5), np.uint8))
        u, v, ok = project(xyz(cur) * S)
        mov = np.ones(len(cur), bool); mov[ok] = changed[v[ok].astype(int), u[ok].astype(int)] > 0
        pts = np.stack([ku, kv, np.ones_like(ku)], 1) @ Hm.T
        ut = pts[:, 0] / pts[:, 2]; vt = pts[:, 1] / pts[:, 2]
        inb = kok & (ut >= 0) & (ut < W) & (vt >= 0) & (vt < H)
        keep = np.ones(len(key_arr), bool)
        keep[inb] = changed[vt[inb].astype(int), ut[inb].astype(int)] == 0
        comp = np.concatenate([to_global(key_arr[keep], Rs[key]), to_global(cur[mov], Rs[t])])
        stored += mov.sum()
        if t % 10 == 5: print(f"Frame {START+t}: moving splats {mov.mean()*100:.1f}%")
    PlyData([PlyElement.describe(comp, "vertex")]).write(name)
    names.append(name); counts.append(len(comp))
print("Padding all frames to the same splat count...")
N = max(counts)
for name, c in zip(names, counts):
    if c == N: continue
    arr = load(name); pad = np.repeat(arr[:1], N - c)
    pad["opacity"] = -20.0
    for s in ("scale_0", "scale_1", "scale_2"): pad["scale_" + s[-1]] = -20.0
    PlyData([PlyElement.describe(np.concatenate([arr, pad]), "vertex")]).write(name)
full = len(P0) * len(files)
print(f"\nDone: {len(names)} frames written to {OUT}")
print(f"Unique splats needed: {stored:,} of {full:,}  ->  SAVINGS {(1-stored/full)*100:.1f}%")

