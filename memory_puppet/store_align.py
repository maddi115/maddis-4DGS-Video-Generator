"""store_align.py - can we line up the STORE across frames? (step 1 of the store sculpture)
For every frame vs the middle frame: hide her, match store points, lift them to 3D with SHARP depth,
solve turn + slide + depth-scale fix, then measure how far the same store surface is apart.
Usage: python store_align.py <sharp_dir> <body_track.npz> <clip.mp4> <video.mp4> [n=15]
"""
import glob, os, re, sys
import numpy as np, cv2

if len(sys.argv) < 5:
    print(__doc__); sys.exit(1)
SHARP, TRACK, CLIP, VIDEO = sys.argv[1:5]
NF = int(sys.argv[5]) if len(sys.argv) > 5 else 15
F = float(os.environ.get("FOCAL", "1303.9"))
VDA = os.environ.get("VDA")                     # steady depth file -> un-bend SHARP with it
SIG = float(os.environ.get("SIGMA", "25"))      # how smooth the un-bend is (half-size pixels)
VD = np.load(VDA)["depths"] if VDA else None
UNIT = 1.0 if VDA else 1.057

TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
def load(p):
    with open(p, "rb") as f:
        count = 0; props = []; inv = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element"):
                inv = line.startswith("element vertex")
                if inv: count = int(line.split()[-1])
            elif line.startswith("property") and inv:
                _, t, name = line.split(); props.append((name, TYPES[t]))
            elif line == "end_header": break
        return np.fromfile(f, dtype=np.dtype(props), count=count)
xyz = lambda a: np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)

files = sorted(glob.glob(os.path.join(SHARP, "frame_*_standard.ply")))
fnum = {int(re.search(r"frame_(\d+)", os.path.basename(p)).group(1)): p for p in files}
T = np.load(TRACK)
V_all = T["pred_vertices"].astype(np.float64) + T["pred_cam_t"][:, None].astype(np.float64)
faces = T["faces"].astype(np.int64)
def grab(path, idx, small=True):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = c.read(); c.release()
    if not ok: return None
    return cv2.resize(fr, (160, 284)).astype(np.float32) if small else fr
c0 = grab(CLIP, 0); first = min(fnum)
off = min(range(first - 3, first + 4), key=lambda i: np.inf if grab(VIDEO, i) is None else np.mean((grab(VIDEO, i) - c0) ** 2))
cap = cv2.VideoCapture(VIDEO); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
frames = [f for f in range(off, off + NF) if f in fnum and (f - off) < len(V_all)]
KEY = frames[len(frames) // 2]
print(f"clip frame 0 = video frame {off} | {len(frames)} frames | lining everything up to frame {KEY}")

def best_signs(P, fn):
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                S = np.array([sx, sy, sz], float); Q = P * S
                with np.errstate(divide="ignore", invalid="ignore"):
                    u = F * Q[:, 0] / Q[:, 2] + W / 2; v = F * Q[:, 1] / Q[:, 2] + H / 2
                r = fn(Q, u, v)
                if best is None or r > best[0]: best = (r, S)
    return best
_, S_SH = best_signs(xyz(load(fnum[KEY]))[::50], lambda Q, u, v: ((Q[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).mean())
k2 = T["pred_keypoints_2d"][0][:21]; k3 = T["pred_keypoints_3d"][0][:21] + T["pred_cam_t"][0]
_, S_MB = best_signs(k3.astype(np.float64), lambda Q, u, v: -np.median(np.hypot(u - k2[:, 0], v - k2[:, 1])))
V_all = V_all * S_MB

DS = 2
def her_mask(V):
    with np.errstate(divide="ignore", invalid="ignore"):
        u = F * V[:, 0] / V[:, 2] + W / 2; v = F * V[:, 1] / V[:, 2] + H / 2
    m = np.zeros((H, W), np.uint8); pts = np.stack([u, v], 1)
    for t in faces: cv2.fillConvexPoly(m, np.round(pts[t]).astype(np.int32), 1)
    return cv2.dilate(m, np.ones((61, 61), np.uint8))
def depth_map(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        u = F * P[:, 0] / P[:, 2] + W / 2; v = F * P[:, 1] / P[:, 2] + H / 2
    ok = (P[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    D = np.full((H // DS + 1, W // DS + 1), np.inf, np.float32)
    np.minimum.at(D, ((v[ok] / DS).astype(int), (u[ok] / DS).astype(int)), P[ok, 2].astype(np.float32))
    return D
def lift(px, D):
    z = D[(px[:, 1] / DS).astype(int), (px[:, 0] / DS).astype(int)]
    return np.stack([(px[:, 0] - W / 2) / F * z, (px[:, 1] - H / 2) / F * z, z], 1)
def umeyama(A, B):
    ma, mb = A.mean(0), B.mean(0); a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a / len(A))
    d = np.sign(np.linalg.det(U @ Vt)); Dm = np.diag([1, 1, d])
    R = U @ Dm @ Vt; s = (S * np.diag(Dm)).sum() / ((a ** 2).sum() / len(A))
    return s, R, mb - s * R @ ma
def ransac(A, B, iters=2000, rel=0.02):
    rng = np.random.default_rng(0); best = None
    for _ in range(iters):
        i = rng.choice(len(A), 3, replace=False)
        try: s, R, t = umeyama(A[i], B[i])
        except np.linalg.LinAlgError: continue
        err = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
        inl = err < rel * B[:, 2]
        if best is None or inl.sum() > best.sum(): best = inl
    s, R, t = umeyama(A[best], B[best])
    return s, R, t, best

def unbend(P, Dv, mask):
    """push every SHARP splat along its camera ray so the frame follows the steady depth"""
    with np.errstate(divide="ignore", invalid="ignore"):
        u = F * P[:, 0] / P[:, 2] + W / 2; v = F * P[:, 1] / P[:, 2] + H / 2
    ok = (P[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    Ds = depth_map(P); hh, ww = Ds.shape
    Dv2 = cv2.resize(Dv, (ww, hh), interpolation=cv2.INTER_LINEAR)
    mk2 = cv2.resize(mask, (ww, hh), interpolation=cv2.INTER_NEAREST) > 0
    good = np.isfinite(Ds) & (Dv2 > 0) & ~mk2
    lr = np.zeros(Ds.shape, np.float32); lr[good] = np.log(Dv2[good] / Ds[good])
    wgt = good.astype(np.float32); k = int(SIG * 3) | 1
    num = cv2.GaussianBlur(lr * wgt, (k, k), SIG); den = cv2.GaussianBlur(wgt, (k, k), SIG)
    field = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    r = np.ones(len(P)); r[ok] = np.exp(field[(v[ok] / DS).astype(int), (u[ok] / DS).astype(int)])
    return P * r[:, None]
orb = cv2.ORB_create(5000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
data = {}
for f in frames:
    P = xyz(load(fnum[f])) * S_SH
    img = cv2.cvtColor(grab(VIDEO, f, small=False), cv2.COLOR_BGR2GRAY)
    m = her_mask(V_all[f - off])
    if VD is not None: P = unbend(P, VD[f - off], m)
    kp, des = orb.detectAndCompute(img, (1 - m) * 255)
    Dsrc = cv2.resize(VD[f - off], (W // DS + 1, H // DS + 1)) if os.environ.get("SOURCE") == "vda" else depth_map(P)
    col = cv2.resize(grab(VIDEO, f, small=False), (W // DS + 1, H // DS + 1))
    data[f] = dict(D=Dsrc, kp=kp, des=des, mask=m, col=col)

K = data[KEY]; Dk = K["D"]
with np.errstate(invalid="ignore", divide="ignore"):
    lz = np.log(np.where(np.isfinite(Dk), Dk, np.nan)); gy_, gx_ = np.gradient(lz)
EDGE = cv2.dilate((~(np.hypot(gx_, gy_) < 0.02)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # skip depth edges
mk = cv2.resize(K["mask"], (Dk.shape[1], Dk.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
print(f"\n{'frame':>6} {'matches':>8} {'turn':>7} {'slide':>8} {'depth fix':>9} | {'store off (typical)':>19} {'off > 2%':>9}")
offs_all = []
for f in frames:
    if f == KEY: continue
    d = data[f]
    m = sorted(bf.match(d["des"], K["des"]), key=lambda x: x.distance)[:2000]
    pa = np.float32([d["kp"][x.queryIdx].pt for x in m]); pb = np.float32([K["kp"][x.trainIdx].pt for x in m])
    A = lift(pa, d["D"]); B = lift(pb, Dk)
    good = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    A, B = A[good], B[good]
    if len(A) < 20:
        print(f"{f:>6}  too few store points ({len(A)})"); continue
    s, R, t, inl = ransac(A, B)
    turn = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    hh, ww = d["D"].shape
    vv, uu = np.mgrid[0:hh, 0:ww]
    z = d["D"]; ok = np.isfinite(z)
    md = cv2.resize(d["mask"], (ww, hh), interpolation=cv2.INTER_NEAREST) > 0
    ok &= ~md
    pts = np.stack([(uu[ok] * DS - W / 2) / F * z[ok], (vv[ok] * DS - H / 2) / F * z[ok], z[ok]], 1)
    q = (s * (R @ pts.T)).T + t
    with np.errstate(divide="ignore", invalid="ignore"):
        qu = (F * q[:, 0] / q[:, 2] + W / 2) / DS; qv = (F * q[:, 1] / q[:, 2] + H / 2) / DS
    inb = (q[:, 2] > 0) & (qu >= 0) & (qu < ww) & (qv >= 0) & (qv < hh)
    qi, qj = qv[inb].astype(int), qu[inb].astype(int)
    ref = Dk[qi, qj]; both = np.isfinite(ref) & ~mk[qi, qj] & ~EDGE[qi, qj]
    src_col = d["col"][vv[ok][inb], uu[ok][inb]].astype(int); key_col = K["col"][qi, qj].astype(int)
    both &= np.abs(src_col - key_col).max(1) < 30   # skip things that moved (other shoppers) or got uncovered
    rel = np.abs(q[inb][both, 2] - ref[both]) / ref[both]
    offs_all.append(rel)
    typ = np.median(np.abs(q[inb][both, 2] - ref[both])) * 100 / UNIT
    print(f"{f:>6} {inl.sum():>5}/{len(A):<4} {turn:>5.2f}deg {np.linalg.norm(t) * 100 / UNIT:>6.1f}cm {s:>9.3f} | "
          f"{typ:>16.1f} cm {np.mean(rel > 0.02) * 100:>8.1f}%")
allr = np.concatenate(offs_all) if offs_all else np.array([np.nan])
print(f"\nOVERALL: store surfaces {np.median(allr) * 100:.2f}% of their distance apart after lining up "
      f"({np.mean(allr > 0.02) * 100:.1f}% of the store is off by more than 2%)")
print("  under ~1%  -> a store sculpture will line up well")
print("  several %  -> SHARP bends depth differently per frame; we need a bend-fix first")
