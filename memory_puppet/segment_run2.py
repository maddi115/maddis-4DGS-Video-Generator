"""segment_run2.py - long video: shared store + live people, optimized and safety-checked.
Usage (from the repo folder): python /workspace/segment_run2.py <start_frame> <seconds> <name>
env: STEP=2 (30 fps)  SEG=30  MARGIN=15  EXTRA=30  RAW=/workspace/Chronosplat/raw_data
Live stats: /workspace/<name>_status.txt
"""
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, cv2, torch
from scipy.spatial.transform import Rotation

REPO = Path("/workspace/maddis-4DGS-Video-Generator"); sys.path.insert(0, str(REPO)); sys.path.insert(0, "/workspace/sam-3d-body")
START, SECS, NAME = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
STEP = int(os.environ.get("STEP", "2")); SEG = int(os.environ.get("SEG", "30"))
MARGIN = int(os.environ.get("MARGIN", "15")); EXTRA = int(os.environ.get("EXTRA", "30"))
RAW = Path(os.environ.get("RAW", "/workspace/Chronosplat/raw_data"))
VIDEO = str(REPO / "sample-video" / "walmartvideo.mp4")
SHM = Path("/dev/shm") / NAME; STATUS = Path(f"/workspace/{NAME}_status.txt")
OUT_P, OUT_B = RAW / NAME, RAW / (NAME + "_bg")
for d in (OUT_P, OUT_B, SHM):
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.ply"): old.unlink()
NSPLAT = 1179648

cap = cv2.VideoCapture(VIDEO); FPS = cap.get(cv2.CAP_PROP_FPS); NTOT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W, H = int(cap.get(3)), int(cap.get(4))
frames_all = list(range(START, min(NTOT, START + int(SECS * FPS)), STEP))
segs = [frames_all[i:i + SEG] for i in range(0, len(frames_all), SEG)]

stats = dict(stage="loading models", frames_done=0, frames_total=len(frames_all), pieces_done=0,
             pieces_total=len(segs), zone=[], live=[], warnings=[], started=time.time())
lock = threading.Lock()
def write_status():
    with lock:
        s = stats; el = time.time() - s["started"]; done = s["frames_done"] / max(s["frames_total"], 1)
        eta = el / done * (1 - done) / 60 if done > 0 else float("nan")
        lines = [f"=== {NAME}: {s['stage']} ===",
                 f"frames : {s['frames_done']} / {s['frames_total']}  ({done * 100:.0f}%)",
                 f"pieces : {s['pieces_done']} / {s['pieces_total']} built",
                 f"time   : {el / 60:.1f} min so far, about {eta:.0f} min left",
                 f"speed  : {el / max(s['frames_done'], 1):.2f} s per frame"]
        if s["zone"]:
            lines.append(f"people : {s['zone'][-1]:.1f}% of picture live (this piece) | avg {np.mean(s['zone']):.1f}%")
            lines.append(f"live   : {s['live'][-1]:,} splats/frame (this piece) vs {NSPLAT:,} in orig")
        lines.append("checks : " + ("ALL OK" if not s["warnings"] else f"{len(s['warnings'])} warning(s):"))
        lines += ["   ! " + w for w in s["warnings"][-6:]]
        STATUS.write_text("\n".join(lines) + "\n")
def warn(msg):
    with lock: stats["warnings"].append(msg)
    print("WARNING:", msg, flush=True)
write_status()

import run_headless_pipeline as rp
dev = torch.device("cuda")
sharp = rp.enable_fp16(rp.load_model(dev))
from tools.build_detector import load_detectron2_vitdet
import detectron2.data.transforms as T
det = load_detectron2_vitdet().to("cuda")
with rp.VideoProcessor(Path(VIDEO)) as vp:
    F = vp.estimate_focal_length(W, H)

TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
         "char": "i1", "int8": "i1", "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
         "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}
def load(p):
    with open(p, "rb") as f:
        elems = []
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element"):
                _, nm, cnt = line.split(); elems.append([nm, int(cnt), []])
            elif line.startswith("property"):
                parts = line.split()
                if parts[1] == "list": raise ValueError(f"{p}: list property, cannot fast-read")
                elems[-1][2].append((parts[2], TYPES[parts[1]]))
            elif line == "end_header": break
        off = f.tell()
    for nm, cnt, props in elems:
        dt = np.dtype(props)
        if nm == "vertex": return np.fromfile(p, dtype=dt, count=cnt, offset=off)
        off += dt.itemsize * cnt
    raise ValueError(f"{p}: no vertex element")
RT = {"<f4": "float", "<f8": "double", "|u1": "uchar", "<i4": "int", "<u4": "uint"}
def save(p, arr):
    xs = np.stack([arr["x"], arr["y"], arr["z"]], 1)
    if not np.isfinite(xs).all(): warn(f"{p.name}: non-finite positions fixed")
    h = ["ply", "format binary_little_endian 1.0", f"element vertex {len(arr)}"]
    h += [f"property {RT[arr.dtype[k].str]} {k}" for k in arr.dtype.names]; h.append("end_header")
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(("\n".join(h) + "\n").encode()); np.ascontiguousarray(arr).tofile(f)
    os.replace(tmp, p)
xyz = lambda a: np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)

def make_sharp_file(i, g):
    p = SHM / f"f{i}.ply"
    rp.save_ply(g, F, (H, W), p)
    a = load(p); p.unlink()
    if len(a) != NSPLAT: raise SystemExit(f"frame {i}: SHARP gave {len(a)} splats, expected {NSPLAT}")
    bad = (~np.isfinite(xyz(a))).any(1).mean()
    if bad > 0.001: warn(f"frame {i}: {bad * 100:.2f}% splats have invalid numbers")
    return np.ascontiguousarray(a)
def people_mask(bgr):
    im = T.ResizeShortestEdge(1024, 1024)(T.AugInput(bgr)).apply_image(bgr)
    inp = {"image": torch.as_tensor(im.astype("float32").transpose(2, 0, 1)), "height": H, "width": W}
    with torch.no_grad(): o = det([inp])[0]["instances"]
    keep = (o.pred_classes == 0) & (o.scores > 0.5)
    return (o.pred_masks[keep].any(0).cpu().numpy() if keep.any() else np.zeros((H, W), bool)).astype(np.uint8)

S_SH = None; UB = VB = None
def pix(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        u = F * P[:, 0] / P[:, 2] + W / 2; v = F * P[:, 1] / P[:, 2] + H / 2
    return u, v, (P[:, 2] > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
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
def ransac(A, B, iters=1500, rel=0.02):
    rng = np.random.default_rng(0); best = None
    for _ in range(iters):
        i = rng.choice(len(A), 3, replace=False)
        try: s, R, t = umeyama(A[i], B[i])
        except np.linalg.LinAlgError: continue
        inl = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1) < rel * B[:, 2]
        if best is None or inl.sum() > best.sum(): best = inl
    return umeyama(A[best], B[best]) + (best,)
orb = cv2.ORB_create(5000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
ID = (1.0, np.eye(3), np.zeros(3))
def solve(fa, Da, fb, Db, tag):
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None: warn(f"{tag}: no store points, camera kept still"); return ID
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:2000]
    A = lift(np.float32([ka[x.queryIdx].pt for x in m]), Da); B = lift(np.float32([kb[x.trainIdx].pt for x in m]), Db)
    g = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    if g.sum() < 30: warn(f"{tag}: only {g.sum()} store points, camera kept still"); return ID
    s, R, t, inl = ransac(A[g], B[g])
    if inl.sum() < 60: warn(f"{tag}: weak camera match ({inl.sum()} points)")
    if not (0.8 < s < 1.25): warn(f"{tag}: odd depth scale {s:.2f}, camera kept still"); return ID
    return s, R, t
def compose(a, b):
    (s1, R1, t1), (s2, R2, t2) = a, b
    return s1 * s2, R1 @ R2, s1 * R1 @ t2 + t1
def quat_of(a):
    q = np.stack([a["rot_1"], a["rot_2"], a["rot_3"], a["rot_0"]], 1).astype(np.float64)
    return Rotation.from_quat(q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12))
def moved(a, tr):
    s, R, t = tr; out = a.copy(); P = xyz(a) * S_SH
    Q = ((s * (R @ P.T)).T + t) * S_SH
    out["x"], out["y"], out["z"] = Q[:, 0], Q[:, 1], Q[:, 2]
    D3 = np.diag(S_SH)
    q = (Rotation.from_matrix(D3 @ R @ D3) * quat_of(a)).as_quat()
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    for c in ("scale_0", "scale_1", "scale_2"): out[c] = a[c] + np.log(s)
    return out
def setup_axes(a):
    global S_SH, UB, VB
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                S = np.array([sx, sy, sz], float); Q = xyz(a)[::50] * S
                with np.errstate(divide="ignore", invalid="ignore"):
                    u = F * Q[:, 0] / Q[:, 2] + W / 2; v = F * Q[:, 1] / Q[:, 2] + H / 2
                r = ((Q[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).mean()
                if best is None or r > best[0]: best = (r, S)
    S_SH = best[1]
    idx = np.arange(len(a)); col = idx % 768
    u, v, ok = pix(xyz(a) * S_SH); fit = None
    for Lh in (768, 384, 1536):
        row = (idx // 768) % Lh
        A = np.stack([col[ok], np.ones(ok.sum())], 1); cu = np.linalg.lstsq(A, u[ok], rcond=None)[0]
        B = np.stack([row[ok], np.ones(ok.sum())], 1); cv_ = np.linalg.lstsq(B, v[ok], rcond=None)[0]
        res = np.median(np.hypot(A @ cu - u[ok], B @ cv_ - v[ok]))
        if fit is None or res < fit[0]: fit = (res, Lh, cu, cv_)
    res, Lh, cu, cv_ = fit
    if res > 3: raise SystemExit(f"birth-pixel map does not fit ({res:.1f} px) - stopping so nothing gets mixed up")
    row = (idx // 768) % Lh
    UB = np.clip(cu[0] * col + cu[1], 0, W - 1).astype(int); VB = np.clip(cv_[0] * row + cv_[1], 0, H - 1).astype(int)
    print(f"checks: {best[0] * 100:.0f}% splats in picture | birth-pixel map fits within {res:.1f} px", flush=True)

state = dict(world=None, prev=None)
counts, seg_of = {}, {}
grow = lambda m, r: cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
def build(si, seg, imgs, plys, masks):
    key = seg[len(seg) // 2]
    key_out = grow(masks[key], MARGIN)
    key_cover = grow(masks[key], MARGIN + EXTRA)
    D = {i: depth_map(xyz(plys[i]) * S_SH) for i in seg}
    gray = {i: cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY) for i in seg}
    feat = {i: orb.detectAndCompute(gray[i], (1 - np.maximum(grow(masks[i], MARGIN), key_out)) * 255) for i in seg}
    if state["world"] is None:
        world = ID
    else:
        pkey, pfeat_src, pD, pmask = state["prev"]
        fk = orb.detectAndCompute(gray[key], (1 - np.maximum(key_out, pmask)) * 255)
        world = compose(state["world"], solve(fk, D[key], pfeat_src, pD, f"piece {si + 1} -> piece {si}"))
    store_sel = ~(key_out[VB, UB] > 0)
    save(OUT_B / f"frame_{si + 1:06d}.ply", moved(plys[key][store_sel], world))
    counts[("bg", si)] = int(store_sel.sum())
    zl, ll = [], []
    for i in seg:
        live_m = np.maximum(grow(masks[i], MARGIN), key_cover)
        live_sel = live_m[VB, UB] > 0
        if i == key and not (store_sel | live_sel).all():
            warn(f"piece {si + 1}: store + live do not cover every splat")
        tr = ID if i == key else solve(feat[i], D[i], feat[key], D[key], f"frame {i}")
        save(OUT_P / f"frame_{i:06d}.ply", moved(plys[i][live_sel], compose(world, tr)))
        counts[("p", i)] = int(live_sel.sum()); seg_of[i] = si
        zl.append(live_m.mean() * 100); ll.append(int(live_sel.sum()))
    gk = orb.detectAndCompute(gray[key], (1 - key_out) * 255)
    state["prev"] = (key, gk, D[key], key_out); state["world"] = world
    with lock:
        stats["pieces_done"] += 1; stats["zone"].append(float(np.mean(zl))); stats["live"].append(int(np.mean(ll)))
    write_status()

savers = ThreadPoolExecutor(max_workers=4); builder = ThreadPoolExecutor(max_workers=1); pending_build = None
with lock: stats["stage"] = "running"
for si, seg in enumerate(segs):
    imgs, futs, masks = {}, {}, {}
    for i in seg:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
        if not ok: raise SystemExit(f"could not read frame {i}")
        imgs[i] = fr
        g = rp.predict_image(sharp, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), F, dev)
        futs[i] = savers.submit(make_sharp_file, i, g)
        masks[i] = people_mask(fr)
        waiting = [f for f in futs.values() if not f.done()]
        if len(waiting) > 6: waiting[0].result()
        with lock: stats["frames_done"] += 1
        write_status()
    plys = {i: futs[i].result() for i in seg}
    if S_SH is None: setup_axes(plys[seg[0]])
    if pending_build is not None: pending_build.result()
    pending_build = builder.submit(build, si, seg, imgs, plys, masks)
pending_build.result()

with lock: stats["stage"] = "padding"
write_status()
def pad_folder(keys):
    N = max(counts[k] for k in keys)
    for k, p in keys.items():
        a = load(p)
        if len(a) < N:
            pd = np.repeat(a[:1], N - len(a)); pd["opacity"] = -20
            for c in ("scale_0", "scale_1", "scale_2"): pd[c] = -20
            save(p, np.concatenate([a, pd]))
    return N
Np = pad_folder({("p", i): OUT_P / f"frame_{i:06d}.ply" for i in frames_all})
Nb = pad_folder({("bg", s): OUT_B / f"frame_{s + 1:06d}.ply" for s in range(len(segs))})
with open(RAW / f"{NAME}_layers.json", "w") as fh:
    json.dump({"start": frames_all[0], "keyEvery": SEG, "segments": [seg_of[i] for i in frames_all]}, fh)
live_all = [counts[("p", i)] for i in frames_all]
orig = NSPLAT * len(frames_all); mine = Np * len(frames_all) + Nb * len(segs)
with lock:
    stats["stage"] = (f"DONE - {(1 - mine / orig) * 100:.0f}% fewer splats than orig "
                      f"(live median {int(np.median(live_all)):,}, max {max(live_all):,} per frame)")
write_status()
print(open(STATUS).read(), flush=True)
