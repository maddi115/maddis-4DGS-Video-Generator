"""puppet_test.py v2 - memory puppet: glue her keyframe splats onto the body, pose them,
then SNAP them onto where she really is each frame, with smoothing over time.
Usage: python puppet_test.py <sharp_dir> <body_track.npz> <clip.mp4> <video.mp4> <out_dir> [key=0] [n=15]
env: SMOOTH=7  SNAP=2 (0 = off)  SNAP_MAX=0.08  FOCAL=1303.9
"""
import glob, os, re, sys
import numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

if len(sys.argv) < 6:
    print(__doc__); sys.exit(1)
SHARP, TRACK, CLIP, VIDEO, OUT = sys.argv[1:6]
KEY = int(sys.argv[6]) if len(sys.argv) > 6 else 0
NF = int(sys.argv[7]) if len(sys.argv) > 7 else 15
SMOOTH = int(os.environ.get("SMOOTH", "7"))
SNAP = int(os.environ.get("SNAP", "2"))
SNAP_MAX = float(os.environ.get("SNAP_MAX", "0.08"))
F = float(os.environ.get("FOCAL", "1303.9"))

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
if not fnum: sys.exit(f"no SHARP frames in {SHARP}")
T = np.load(TRACK)
V_all = T["pred_vertices"].astype(np.float64) + T["pred_cam_t"][:, None].astype(np.float64)
faces = T["faces"].astype(np.int64)
lost = set(T["lost"].tolist())

def grab(path, idx):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = c.read(); c.release()
    return cv2.resize(fr, (160, 284)).astype(np.float32) if ok else None
c0 = grab(CLIP, 0); first = min(fnum)
off = min(range(first - 3, first + 4), key=lambda i: np.inf if grab(VIDEO, i) is None else np.mean((grab(VIDEO, i) - c0) ** 2))
print(f"clip frame 0 = video frame {off}")

cap = cv2.VideoCapture(VIDEO); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
frames = [off + KEY + j for j in range(NF)]
frames = [f for f in frames if f in fnum and (f - off) < len(V_all)]
if (off + KEY) not in fnum: sys.exit(f"keyframe {off + KEY} has no SHARP file")

def best_signs(P, u_fn):
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                S = np.array([sx, sy, sz], float); Q = P * S
                with np.errstate(divide="ignore", invalid="ignore"):
                    u = F * Q[:, 0] / Q[:, 2] + W / 2; v = F * Q[:, 1] / Q[:, 2] + H / 2
                r = u_fn(Q, u, v)
                if best is None or r > best[0]: best = (r, S)
    return best
key_ply = load(fnum[off + KEY])
inside = lambda Q, u, v: ((Q[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).mean()
r, S_SH = best_signs(xyz(key_ply)[::50], inside)
print(f"SHARP self-check: {r * 100:.1f}% of splats land in the picture")
k2 = T["pred_keypoints_2d"][KEY][:21]; k3 = T["pred_keypoints_3d"][KEY][:21] + T["pred_cam_t"][KEY]
match = lambda Q, u, v: -np.median(np.hypot(u - k2[:, 0], v - k2[:, 1]))
r, S_MB = best_signs(k3.astype(np.float64), match)
print(f"body self-check: joints land {-r:.1f} px from where the tracker drew them (want < 10)")
V_all = V_all * S_MB
if SMOOTH >= 3 and len(V_all) >= SMOOTH:
    V_all = savgol_filter(V_all, SMOOTH | 1, 2, axis=0)

def project(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        return F * P[:, 0] / P[:, 2] + W / 2, F * P[:, 1] / P[:, 2] + H / 2

DS = 4
def body_depth(V):
    u, v = project(V); z = V[:, 2]
    tri = faces[np.argsort(-z[faces].mean(1))]
    D = np.zeros((H // DS + 1, W // DS + 1), np.float32)
    pts = np.stack([u, v], 1) / DS
    for t in tri:
        cv2.fillConvexPoly(D, np.round(pts[t]).astype(np.int32), float(z[t].mean()))
    return D

def her_splats(P, V, D, band=0.25):
    u, v = project(P)
    ok = (P[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    grow = cv2.dilate((D > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
    Dg = cv2.dilate(D, np.ones((9, 9), np.uint8))
    ui = np.clip((u / DS).astype(int), 0, D.shape[1] - 1); vi = np.clip((v / DS).astype(int), 0, D.shape[0] - 1)
    inmask = ok & (grow[vi, ui] > 0)
    dz = np.where(D[vi, ui] > 0, D[vi, ui], Dg[vi, ui])
    core = inmask & (D[vi, ui] > 0)
    front = np.full(D.shape, np.inf, np.float32)
    np.minimum.at(front, (vi[core], ui[core]), P[core, 2].astype(np.float32))
    cells = np.isfinite(front) & (D > 0)
    s = float(np.median(front[cells] / D[cells]))
    near = inmask & (np.abs(P[:, 2] / s - dz) < band * dz + 0.15)
    return near, s

nb = np.zeros(len(V_all[0]), np.int64); nb[faces[:, 0]] = faces[:, 1]; nb[faces[:, 1]] = faces[:, 2]; nb[faces[:, 2]] = faces[:, 0]
def frames_at(V):
    n = np.zeros_like(V)
    fn = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
    for i in range(3): np.add.at(n, faces[:, i], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    t = V[nb] - V; t -= (t * n).sum(1, keepdims=True) * n
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    b = np.cross(n, t)
    return np.stack([t, b, n], 2)

Vk = V_all[KEY]; Dk = body_depth(Vk)
Pk = xyz(key_ply) * S_SH
mine, s_k = her_splats(Pk, Vk, Dk)
her = key_ply[mine]; Ph = Pk[mine] / s_k
dist, idx = cKDTree(Vk).query(Ph, k=4)
w = 1 / (dist + 1e-3); w /= w.sum(1, keepdims=True)
Fk = frames_at(Vk)
offs = np.einsum("nkji,nkj->nki", Fk[idx], Ph[:, None] - Vk[idx])
print(f"\nglued {len(her):,} of her splats (keyframe {off + KEY}) | SHARP units per metre {s_k:.3f}")
print(f"  distance to body surface: typical {np.median(dist[:, 0]) * 100:.1f} cm, far ones (hair/phone) {np.percentile(dist[:, 0], 95) * 100:.1f} cm")

D3 = np.diag(S_SH)
q_her = np.stack([her["rot_1"], her["rot_2"], her["rot_3"], her["rot_0"]], 1).astype(np.float64)
R_her = Rotation.from_quat(q_her / (np.linalg.norm(q_her, axis=1, keepdims=True) + 1e-12))

name = os.path.basename(os.path.normpath(OUT)); os.makedirs(OUT, exist_ok=True)
for old in glob.glob(os.path.join(OUT, "*.ply")): os.remove(old)
nv = len(Vk)
edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
def spread(C, have, iters=12):
    C = C.copy(); wgt = have.astype(np.float64)
    for _ in range(iters):
        acc = np.zeros_like(C); wa = np.zeros(nv)
        np.add.at(acc, edges[:, 0], C[edges[:, 1]] * wgt[edges[:, 1], None]); np.add.at(wa, edges[:, 0], wgt[edges[:, 1]])
        np.add.at(acc, edges[:, 1], C[edges[:, 0]] * wgt[edges[:, 0], None]); np.add.at(wa, edges[:, 1], wgt[edges[:, 0]])
        nbr = acc / np.maximum(wa, 1e-9)[:, None]
        C = np.where(have[:, None], 0.5 * C + 0.5 * np.where(wa[:, None] > 0, nbr, C), np.where(wa[:, None] > 0, nbr, C))
        wgt = np.maximum(wgt, (wa > 0) * 0.5)
    return C

rgb = lambda a: np.stack([a["f_dc_0"], a["f_dc_1"], a["f_dc_2"]], 1).astype(np.float64) * 0.2821
COLW = 0.04

posed = {}
for f in frames:
    j = f - off
    cur = load(fnum[f]); Pc = xyz(cur) * S_SH
    Vt = V_all[j]; Dt = body_depth(Vt)
    hers_t, s_t = her_splats(Pc, Vt, Dt)
    Ft = frames_at(Vt)
    Pt = (w[..., None] * (Vt[idx] + np.einsum("nkij,nkj->nki", Ft[idx], offs))).sum(1)
    posed[f] = dict(cur=cur, hers=hers_t, s=s_t, Ft=Ft, Pt=Pt, Q=Pc[hers_t])
ss = np.array([posed[f]["s"] for f in frames])
if len(ss) >= 5: ss = savgol_filter(ss, 5, 1)
for f, sv in zip(frames, ss): posed[f]["s"] = float(sv)

her_col = rgb(her)
err_before, err_after = {}, {}
for p in range(SNAP):
    Cs = {}
    for f in frames:
        d = posed[f]; Q = d["Q"] / d["s"]
        tree = cKDTree(np.hstack([Q, rgb(d["cur"][d["hers"]]) * COLW]))
        dd, nn = tree.query(np.hstack([d["Pt"], her_col * COLW]), distance_upper_bound=SNAP_MAX * 1.5)
        good = np.isfinite(dd)
        res = np.zeros_like(d["Pt"]); res[good] = Q[nn[good]] - d["Pt"][good]
        if p == 0: err_before[f] = np.median(np.linalg.norm(res[good], axis=1)) * 100 if good.any() else np.nan
        C = np.zeros((nv, 3)); cnt = np.zeros(nv)
        np.add.at(C, idx[good, 0], res[good]); np.add.at(cnt, idx[good, 0], 1)
        have = cnt > 0; C[have] /= cnt[have, None]
        Cs[f] = spread(C, have)
    stack = np.stack([Cs[f] for f in frames])
    if len(frames) >= 5: stack = savgol_filter(stack, 5, 2, axis=0)
    for f, C in zip(frames, stack):
        posed[f]["Pt"] = posed[f]["Pt"] + (w[..., None] * C[idx]).sum(1)
for f in frames:
    d = posed[f]; Q = d["Q"] / d["s"]
    dd, _ = cKDTree(Q).query(d["Pt"], distance_upper_bound=SNAP_MAX * 1.5)
    err_after[f] = np.median(dd[np.isfinite(dd)]) * 100

outs, report = {}, []
for f in frames:
    j = f - off; d = posed[f]; s_t = d["s"]
    bg = d["cur"][~d["hers"]]
    Rm = np.einsum("nij,nkj->nik", d["Ft"][idx[:, 0]], Fk[idx[:, 0]])
    Rp = Rotation.from_matrix(D3 @ Rm @ D3)
    q = (Rp * R_her).as_quat()
    pup = her.copy()
    P_out = (d["Pt"] * s_t) * S_SH
    pup["x"], pup["y"], pup["z"] = P_out[:, 0], P_out[:, 1], P_out[:, 2]
    pup["rot_0"], pup["rot_1"], pup["rot_2"], pup["rot_3"] = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    for c in ("scale_0", "scale_1", "scale_2"): pup[c] = her[c] + np.log(s_t / s_k)
    outs[f] = np.concatenate([bg, pup])
    report.append(f"  frame {f}{' (tracker lost her)' if j in lost else ''}: puppet off by {err_before.get(f, np.nan):.1f} cm -> {err_after[f]:.1f} cm after snap | depth scale {s_t / s_k:.3f}")

N = max(len(a) for a in outs.values())
for f, a in outs.items():
    if len(a) < N:
        pad = np.repeat(a[:1], N - len(a)); pad["opacity"] = -20
        for c in ("scale_0", "scale_1", "scale_2"): pad[c] = -20
        a = np.concatenate([a, pad])
    save(os.path.join(OUT, f"frame_{f:06d}.ply"), a)
print("\n".join(report))
print(f"\nwrote {len(outs)} frames to {OUT}")
