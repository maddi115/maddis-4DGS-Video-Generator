"""store_build.py - the store, done right: ONE shared store + live her + live movers.
Usage: python store_build.py <sharp_dir> <body_track.npz> <clip.mp4> <video.mp4> <raw_dir> <name> [n=15]
env: MOVE_THR=35  FOCAL=1303.9
"""
import glob, json, os, re, sys
import numpy as np, cv2
from scipy.spatial.transform import Rotation

if len(sys.argv) < 7:
    print(__doc__); sys.exit(1)
SHARP, TRACK, CLIP, VIDEO, RAW, NAME = sys.argv[1:7]
NF = int(sys.argv[7]) if len(sys.argv) > 7 else 15
F = float(os.environ.get("FOCAL", "1303.9"))
MOVE_THR = int(os.environ.get("MOVE_THR", "35"))

TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
RTYPES = {v: k for k, v in TYPES.items()}
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
def save(p, arr):
    h = ["ply", "format binary_little_endian 1.0", f"element vertex {len(arr)}"]
    h += [f"property {RTYPES[arr.dtype[k].str.replace('|', '')]} {k}" for k in arr.dtype.names]
    h.append("end_header")
    with open(p, "wb") as f:
        f.write(("\n".join(h) + "\n").encode()); np.ascontiguousarray(arr).tofile(f)
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
print(f"clip frame 0 = video frame {off} | {len(frames)} frames | shared store from frame {KEY}")

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
D3 = np.diag(S_SH)

def pix(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        u = F * P[:, 0] / P[:, 2] + W / 2; v = F * P[:, 1] / P[:, 2] + H / 2
    ok = (P[:, 2] > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    return u, v, ok
def in_mask(mask, u, v, ok):
    out = np.zeros(len(u), bool); out[ok] = mask[v[ok].astype(int), u[ok].astype(int)] > 0
    return out
def her_mask(V):
    u, v, _ = pix(V)
    m = np.zeros((H, W), np.uint8); pts = np.stack([u, v], 1)
    for t in faces: cv2.fillConvexPoly(m, np.round(pts[t]).astype(np.int32), 1)
    return cv2.dilate(m, np.ones((61, 61), np.uint8))
def depth_map(P, DS=2):
    u, v, ok = pix(P)
    D = np.full((H // DS + 1, W // DS + 1), np.inf, np.float32)
    np.minimum.at(D, ((v[ok] / DS).astype(int), (u[ok] / DS).astype(int)), P[ok, 2].astype(np.float32))
    return D
def lift(px, D, DS=2):
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
        inl = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1) < rel * B[:, 2]
        if best is None or inl.sum() > best.sum(): best = inl
    s, R, t = umeyama(A[best], B[best])
    return s, R, t, best
def quat_of(a):
    q = np.stack([a["rot_1"], a["rot_2"], a["rot_3"], a["rot_0"]], 1).astype(np.float64)
    return Rotation.from_quat(q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12))
def moved(a, s, R, t):
    out = a.copy(); P = xyz(a) * S_SH
    Q = ((s * (R @ P.T)).T + t) * S_SH
    out["x"], out["y"], out["z"] = Q[:, 0], Q[:, 1], Q[:, 2]
    q = (Rotation.from_matrix(D3 @ R @ D3) * quat_of(a)).as_quat()
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    for c in ("scale_0", "scale_1", "scale_2"): out[c] = a[c] + np.log(s)
    return out

masks = {f: her_mask(V_all[f - off]) for f in frames}
zone = np.zeros((H, W), np.uint8)
for m in masks.values(): zone |= m
PEOPLE = os.environ.get("PEOPLE")
if PEOPLE:   # every person in every frame stays live; only the background is shared
    pm = np.load(PEOPLE); shp = tuple(pm["shape"])
    for f in frames:
        if f"f{f}" in pm:
            m_ = np.unpackbits(pm[f"f{f}"])[: shp[0] * shp[1]].reshape(shp).astype(np.uint8)
            zone |= cv2.dilate(m_, np.ones((25, 25), np.uint8))
    MOVE_THR = 999   # no colour guessing needed any more
print(f"people zone: {zone.mean() * 100:.1f}% of the picture (kept live, orig quality)")

orb = cv2.ORB_create(5000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
img = {f: grab(VIDEO, f, small=False) for f in frames}
plys = {f: load(fnum[f]) for f in frames}
RULE = os.environ.get("RULE", "birth")      # birth = bucket by the pixel each splat was MADE from; land = old rule
SEAM = os.environ.get("SEAM", "0") == "1"   # seam view: store red, live green
def birth_pixels(a):
    """SHARP makes each splat from one picture pixel, in a fixed grid. Work out that pixel."""
    idx = np.arange(len(a)); col = idx % 768
    u, v, ok = pix(xyz(a) * S_SH); best = None
    for Lh in (768, 384, 1536):
        row = (idx // 768) % Lh
        A = np.stack([col[ok], np.ones(ok.sum())], 1); cu = np.linalg.lstsq(A, u[ok], rcond=None)[0]
        B = np.stack([row[ok], np.ones(ok.sum())], 1); cv_ = np.linalg.lstsq(B, v[ok], rcond=None)[0]
        res = np.median(np.hypot(A @ cu - u[ok], B @ cv_ - v[ok]))
        if best is None or res < best[0]: best = (res, Lh, cu, cv_)
    res, Lh, cu, cv_ = best; row = (idx // 768) % Lh
    ub = np.clip(cu[0] * col + cu[1], 0, W - 1).astype(int); vb = np.clip(cv_[0] * row + cv_[1], 0, H - 1).astype(int)
    print(f"birth-pixel map: grid layers of {Lh} rows, fits within {res:.1f} px (want a few px)")
    return ub, vb
UB, VB = birth_pixels(plys[KEY])
D = {f: depth_map(xyz(plys[f]) * S_SH) for f in frames}
feat = {f: orb.detectAndCompute(cv2.cvtColor(img[f], cv2.COLOR_BGR2GRAY), (1 - zone) * 255) for f in frames}
move = {KEY: (1.0, np.eye(3), np.zeros(3))}
for f in frames:
    if f == KEY: continue
    (ka, da), (kb, db) = feat[f], feat[KEY]
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:2000]
    A = lift(np.float32([ka[x.queryIdx].pt for x in m]), D[f]); B = lift(np.float32([kb[x.trainIdx].pt for x in m]), D[KEY])
    g = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    s, R, t, inl = ransac(A[g], B[g])
    move[f] = (s, R, t)

key = plys[KEY]; Pk = xyz(key) * S_SH
uk, vk, okk = pix(Pk)
land_live = in_mask(zone, uk, vk, okk); birth_live = zone[VB, UB] > 0
mis = birth_live & ~land_live
print(f"PROOF: {mis.sum():,} splats made from people pixels were put in the STORE by the old rule "
      f"({(mis & ~okk).sum():,} of them land past the picture edge)")
store = key[~(birth_live if RULE == "birth" else land_live)]
Ps = xyz(store) * S_SH
us, vs, oks = pix(Ps)
blur = lambda im: cv2.GaussianBlur(im, (5, 5), 0).astype(np.int16)
kimg = blur(img[KEY])
scol = np.zeros((len(store), 3), np.int16); scol[oks] = kimg[vs[oks].astype(int), us[oks].astype(int)]
print(f"shared store: {len(store):,} splats (stored once)")

for d in (NAME, NAME + "_lay", NAME + "_lay_bg"):
    os.makedirs(os.path.join(RAW, d), exist_ok=True)
    for old in glob.glob(os.path.join(RAW, d, "*.ply")): os.remove(old)
full, lay, rep = {}, {}, []
for f in frames:
    s, R, t = move[f]
    Ri, ti, si = R.T, -R.T @ t / s, 1 / s
    st = moved(store, si, Ri, ti)
    Pt = xyz(st) * S_SH; ut, vt, okt = pix(Pt)
    fimg = blur(img[f]); diff = np.zeros(len(st), np.int16)
    both = okt & oks
    diff[both] = np.abs(scol[both] - fimg[vt[both].astype(int), ut[both].astype(int)]).max(1)
    mv = np.zeros((H // 2 + 1, W // 2 + 1), np.uint8)
    hit = both & (diff > MOVE_THR)
    mv[(vt[hit] / 2).astype(int), (ut[hit] / 2).astype(int)] = 1
    mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mv)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 300: mv[lab == i] = 0
    mv = cv2.resize(cv2.dilate(mv, np.ones((15, 15), np.uint8)), (W, H), interpolation=cv2.INTER_NEAREST)
    live_zone = np.maximum(zone, mv)
    cur = plys[f]; Pc = xyz(cur) * S_SH; uc, vc, okc = pix(Pc)
    live = cur[(live_zone[VB, UB] > 0) if RULE == "birth" else in_mask(live_zone, uc, vc, okc)]
    st_keep = st[~in_mask(live_zone, ut, vt, okt)]
    if SEAM:
        def tint(a, rgb):
            a = a.copy()
            for ch, val in zip(("f_dc_0", "f_dc_1", "f_dc_2"), rgb): a[ch] = 0.5 * a[ch] + 0.5 * ((val - 0.5) / 0.28209479)
            return a
        full[f] = np.concatenate([tint(st_keep, (1, 0, 0)), tint(live, (0, 1, 0))])
    else:
        full[f] = np.concatenate([st_keep, live])
    lay[f] = moved(live, s, R, t)
    rep.append(f"  frame {f}: camera turn {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.2f} deg, "
               f"depth fix {s:.3f} | moved things {mv.mean() * 100:.1f}% of picture | live splats {len(live):,}")
print("\n".join(rep))

def pad_save(group, folder, names):
    N = max(len(a) for a in group.values())
    for k, a in group.items():
        if len(a) < N:
            p = np.repeat(a[:1], N - len(a)); p["opacity"] = -20
            for c in ("scale_0", "scale_1", "scale_2"): p[c] = -20
            a = np.concatenate([a, p])
        save(os.path.join(RAW, folder, names(k)), a)
    return N
pad_save(full, NAME, lambda k: f"frame_{k:06d}.ply")
Nl = pad_save(lay, NAME + "_lay", lambda k: f"frame_{k:06d}.ply")
save(os.path.join(RAW, NAME + "_lay_bg", "frame_000001.ply"), store)
with open(os.path.join(RAW, NAME + "_lay_layers.json"), "w") as fh:
    json.dump({"start": frames[0], "keyEvery": len(frames), "segments": [0] * len(frames)}, fh)
orig_total = sum(len(plys[f]) for f in frames)
lay_total = len(store) + Nl * len(frames)
print(f"\nsplats stored: orig {orig_total:,} | store-once version {lay_total:,} -> {(1 - lay_total / orig_total) * 100:.0f}% fewer")
print(f"wrote {RAW}/{NAME} (compare with orig) and {NAME}_lay + {NAME}_lay_bg (size test)")
