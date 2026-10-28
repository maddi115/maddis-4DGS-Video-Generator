"""sculpture_test.py - YOUR way: look at every frame first, build ONE sculpture of her, put it back.
Pass 2: her body splats from EVERY frame -> one shared pose -> merged into one sculpture.
Pass 3: pose the sculpture into each frame + snap. Hands and hair are LIVE (copied from each frame).
Usage: python sculpture_test.py <sharp_dir> <body_track.npz> <clip.mp4> <video.mp4> <out_dir> [n=15]
env: VOXEL=0.002  MINSEEN=2  SNAP=2  SNAP_MAX=0.08  SMOOTH=7  FOCAL=1303.9
"""
import glob, os, re, sys
import numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

if len(sys.argv) < 6:
    print(__doc__); sys.exit(1)
SHARP, TRACK, CLIP, VIDEO, OUT = sys.argv[1:6]
NF = int(sys.argv[6]) if len(sys.argv) > 6 else 15
SMOOTH = int(os.environ.get("SMOOTH", "7"))
SNAP = int(os.environ.get("SNAP", "2"))
SNAP_MAX = float(os.environ.get("SNAP_MAX", "0.08"))
VOXEL = float(os.environ.get("VOXEL", "0.002"))
MINSEEN = int(os.environ.get("MINSEEN", "2"))
F = float(os.environ.get("FOCAL", "1303.9"))
HANDS = list(range(21, 63))
HEAD = [0, 1, 2, 3, 4]

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
KP_all = T["pred_keypoints_3d"].astype(np.float64) + T["pred_cam_t"][:, None].astype(np.float64)
faces = T["faces"].astype(np.int64)
lost = set(T["lost"].tolist())
def grab(path, idx):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = c.read(); c.release()
    return cv2.resize(fr, (160, 284)).astype(np.float32) if ok else None
c0 = grab(CLIP, 0); first = min(fnum)
off = min(range(first - 3, first + 4), key=lambda i: np.inf if grab(VIDEO, i) is None else np.mean((grab(VIDEO, i) - c0) ** 2))
cap = cv2.VideoCapture(VIDEO); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
frames = [f for f in range(off, off + NF) if f in fnum and (f - off) < len(V_all)]
REF = frames[len(frames) // 2] - off
print(f"clip frame 0 = video frame {off} | {len(frames)} frames | sculpture pose = frame {REF + off}")

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
probe = load(fnum[frames[0]])
r, S_SH = best_signs(xyz(probe)[::50], lambda Q, u, v: ((Q[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).mean())
k2 = T["pred_keypoints_2d"][0][:21]
r2, S_MB = best_signs(KP_all[0][:21], lambda Q, u, v: -np.median(np.hypot(u - k2[:, 0], v - k2[:, 1])))
print(f"self-checks: SHARP {r * 100:.1f}% in picture | body joints {-r2:.1f} px off (want < 10)")
V_all = V_all * S_MB; KP_all = KP_all * S_MB
if SMOOTH >= 3 and len(V_all) >= SMOOTH:
    V_all = savgol_filter(V_all, SMOOTH | 1, 2, axis=0); KP_all = savgol_filter(KP_all, SMOOTH | 1, 2, axis=0)
D3 = np.diag(S_SH)

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
def her_splats(P, D, band=0.25):
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
    return inmask & (np.abs(P[:, 2] / s - dz) < band * dz + 0.15), s

nb = np.zeros(len(V_all[0]), np.int64); nb[faces[:, 0]] = faces[:, 1]; nb[faces[:, 1]] = faces[:, 2]; nb[faces[:, 2]] = faces[:, 0]
def frames_at(V):
    n = np.zeros_like(V)
    fn = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
    for i in range(3): np.add.at(n, faces[:, i], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    t = V[nb] - V; t -= (t * n).sum(1, keepdims=True) * n
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return np.stack([t, np.cross(n, t), n], 2)
def glue(P, V, Fr):
    d, i = cKDTree(V).query(P, k=4)
    w = 1 / (d + 1e-3); w /= w.sum(1, keepdims=True)
    return d, i, w, np.einsum("nkji,nkj->nki", Fr[i], P[:, None] - V[i])
def place(V, Fr, i, w, o):
    return (w[..., None] * (V[i] + np.einsum("nkij,nkj->nki", Fr[i], o))).sum(1)
def parts(P, dsurf, KP):
    hand = cKDTree(KP[HANDS]).query(P)[0] < 0.08
    head_c = KP[HEAD].mean(0)
    hair = (~hand) & (dsurf > 0.015) & (np.linalg.norm(P - head_c, axis=1) < 0.40)
    return hand, hair
def quat_of(a):
    q = np.stack([a["rot_1"], a["rot_2"], a["rot_3"], a["rot_0"]], 1).astype(np.float64)
    return Rotation.from_quat(q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12))
def set_quat(a, R):
    q = R.as_quat(); a["rot_0"], a["rot_1"], a["rot_2"], a["rot_3"] = q[:, 3], q[:, 0], q[:, 1], q[:, 2]

# ---- pass 2: every frame -> one sculpture ----
Vr = V_all[REF]; Fr_ref = frames_at(Vr)
info, pieces = {}, []
for f in frames:
    j = f - off
    cur = load(fnum[f]); Pc = xyz(cur) * S_SH
    Vt = V_all[j]; Ft = frames_at(Vt)
    hers, s = her_splats(Pc, body_depth(Vt))
    info[f] = dict(cur=cur, hers=hers, s=s, Ft=Ft)
ss = savgol_filter(np.array([info[f]["s"] for f in frames]), 5, 1) if len(frames) >= 5 else [info[f]["s"] for f in frames]
for f, sv in zip(frames, ss): info[f]["s"] = float(sv)
for f in frames:
    j = f - off; d = info[f]; Vt = V_all[j]
    Ph = xyz(d["cur"][d["hers"]]) * S_SH / d["s"]
    dist, i, w, o = glue(Ph, Vt, d["Ft"])
    hand, hair = parts(Ph, dist[:, 0], KP_all[j])
    d["live"] = np.zeros(len(d["cur"]), bool); d["live"][np.nonzero(d["hers"])[0][hand | hair]] = True
    body = ~(hand | hair)
    a = d["cur"][d["hers"]][body].copy()
    Pr = place(Vr, Fr_ref, i[body], w[body], o[body])
    Rm = np.einsum("nij,nkj->nik", Fr_ref[i[body, 0]], d["Ft"][i[body, 0]])
    set_quat(a, Rotation.from_matrix(D3 @ Rm @ D3) * quat_of(a))
    for c in ("scale_0", "scale_1", "scale_2"): a[c] -= np.log(d["s"])
    pieces.append((a, Pr, np.full(len(a), f), i[body, 0]))
    d["n_live"] = int(d["live"].sum())

A = np.concatenate([p[0] for p in pieces]); P = np.concatenate([p[1] for p in pieces])
src = np.concatenate([p[2] for p in pieces]); vid = np.concatenate([p[3] for p in pieces])
fi = np.searchsorted(np.array(frames), src); nv0 = len(Vr)
# each body spot comes from ONE frame: the one that saw it best (no averaging = no blur, no ghosts)
score = np.zeros((nv0, len(frames))); np.add.at(score, (vid, fi), 1.0)
ed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
for _ in range(3):   # decide per neighbourhood, so the sculpture isn't a patchwork
    sm = score.copy(); np.add.at(sm, ed[:, 0], score[ed[:, 1]]); np.add.at(sm, ed[:, 1], score[ed[:, 0]]); score = sm
ref_i = frames.index(REF + off); score[:, ref_i] *= 1.3
pick = score.argmax(1)
keep = pick[vid] == fi
sculpture, Ps = A[keep], P[keep]
print(f"\nsculpture: {len(sculpture):,} splats, each spot from its best frame "
      f"({(fi[keep] == ref_i).mean() * 100:.0f}% from frame {REF + off}, rest filled from {len(set(fi[keep])) - 1} other frames)")

# ---- pass 3: pose the sculpture into every frame ----
dist0, i0, w0, o0 = glue(Ps, Vr, Fr_ref)
R0 = quat_of(sculpture)
col = lambda a: np.stack([a["f_dc_0"], a["f_dc_1"], a["f_dc_2"]], 1).astype(np.float64) * 0.2821
nv = len(Vr); edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
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
for f in frames:
    d = info[f]; Vt = V_all[f - off]
    d["Pt"] = place(Vt, d["Ft"], i0, w0, o0)
    body_real = d["hers"] & ~d["live"]
    d["Q"] = xyz(d["cur"][body_real]) * S_SH / d["s"]; d["Qc"] = col(d["cur"][body_real])
sc = col(sculpture) * 0.04; before = {}
for p in range(SNAP):
    Cs = []
    for f in frames:
        d = info[f]
        dd, nn = cKDTree(np.hstack([d["Q"], d["Qc"] * 0.04])).query(np.hstack([d["Pt"], sc]), distance_upper_bound=SNAP_MAX * 1.5)
        g = np.isfinite(dd); res = np.zeros_like(d["Pt"]); res[g] = d["Q"][nn[g]] - d["Pt"][g]
        if p == 0: before[f] = np.median(np.linalg.norm(res[g], axis=1)) * 100
        C = np.zeros((nv, 3)); n_ = np.zeros(nv)
        np.add.at(C, i0[g, 0], res[g]); np.add.at(n_, i0[g, 0], 1)
        hv = n_ > 0; C[hv] /= n_[hv, None]; Cs.append(spread(C, hv))
    Cs = np.stack(Cs)
    if len(frames) >= 5: Cs = savgol_filter(Cs, 5, 2, axis=0)
    for f, C in zip(frames, Cs): info[f]["Pt"] = info[f]["Pt"] + (w0[..., None] * C[i0]).sum(1)

name = os.path.basename(os.path.normpath(OUT)); os.makedirs(OUT, exist_ok=True)
for old in glob.glob(os.path.join(OUT, "*.ply")): os.remove(old)
outs, report = {}, []
for f in frames:
    d = info[f]
    dd = cKDTree(d["Q"]).query(d["Pt"], distance_upper_bound=SNAP_MAX * 1.5)[0]
    after = np.median(dd[np.isfinite(dd)]) * 100
    pup = sculpture.copy()
    Pout = (d["Pt"] * d["s"]) * S_SH
    pup["x"], pup["y"], pup["z"] = Pout[:, 0], Pout[:, 1], Pout[:, 2]
    Rm = np.einsum("nij,nkj->nik", d["Ft"][i0[:, 0]], Fr_ref[i0[:, 0]])
    set_quat(pup, Rotation.from_matrix(D3 @ Rm @ D3) * R0)
    for c in ("scale_0", "scale_1", "scale_2"): pup[c] = sculpture[c] + np.log(d["s"])
    bg = d["cur"][~d["hers"]]; live = d["cur"][d["live"]]
    Pl = xyz(live) * S_SH / d["s"]
    if len(Pl):
        pup = pup[~(cKDTree(Pl).query(d["Pt"], distance_upper_bound=0.008)[0] < 0.008)]
    outs[f] = np.concatenate([bg, pup, live])
    report.append(f"  frame {f}{' (tracker lost her)' if f - off in lost else ''}: sculpture off by {before[f]:.1f} -> {after:.1f} cm | live hands+hair {d['n_live']:,} splats")
N = max(len(a) for a in outs.values())
for f, a in outs.items():
    if len(a) < N:
        pad = np.repeat(a[:1], N - len(a)); pad["opacity"] = -20
        for c in ("scale_0", "scale_1", "scale_2"): pad[c] = -20
        a = np.concatenate([a, pad])
    save(os.path.join(OUT, f"frame_{f:06d}.ply"), a)
print("\n".join(report))
her_total = np.mean([info[f]["hers"].sum() for f in frames])
live_avg = np.mean([info[f]["n_live"] for f in frames])
print(f"\nstored for her: sculpture {len(sculpture):,} once + live {live_avg:,.0f} per frame (vs {her_total:,.0f} per frame in orig)")
print(f"wrote {len(outs)} frames to {OUT}")
