"""build_layers.py - split splat frames into a shared BACKGROUND layer + small MOVING layers.

For each segment of KEY_EVERY frames:
  - background = the keyframe's splats in areas that stay still for the WHOLE segment
                 (saved once per segment)
  - moving     = each frame's own splats in areas that change anywhere in the segment
                 (saved every frame, small)
Everything is lined up in one fixed 3D space (phone shake removed).
Works one segment at a time, so memory stays small even for long videos.

Usage:
  python build_layers.py <frames_dir> <start_frame> <chronosplat_raw_data_dir> <name> [key_every=10] [limit]
  (set VIDEO=... and FOCAL=... to override the video path and focal length)

Writes:
  <raw_data>/<name>/frame_XXXXXX.ply      moving layer, one per frame
  <raw_data>/<name>_bg/frame_XXXXXX.ply   background layer, one per segment
  <raw_data>/<name>_layers.json           which background each frame uses
"""
import cv2, numpy as np, glob, os, sys, json

if len(sys.argv) < 5:
    print(__doc__); sys.exit(1)
D, START, RAW, NAME = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
KEY_EVERY = int(sys.argv[5]) if len(sys.argv) > 5 else 10
LIMIT = int(sys.argv[6]) if len(sys.argv) > 6 else None
VIDEO = os.environ.get("VIDEO", "sample-video/walmartvideo.mp4")
F = float(os.environ.get("FOCAL", "1303.9"))
THRESH = int(os.environ.get("THRESH", "12"))
SMOOTH = os.environ.get("SMOOTH", "1") == "1"
STAB = os.environ.get("STAB", "1") == "1"   # STAB=0: leave moving splats exactly as SHARP made them
FILL = os.environ.get("FILL", "1") == "1"   # fill the inside of moving things (no frozen patches)
DEPTH = os.environ.get("DEPTH", "1") == "1" # match each frame's depth scale to its background
SEAM = os.environ.get("SEAM", "0") == "1"   # test view: background tinted RED, moving tinted GREEN

def tint(arr, rgb):
    """Seam view: mix each splat's colour halfway toward rgb (shape stays visible)."""
    if not SEAM: return arr
    out = arr.copy()
    for c, val in zip(("f_dc_0", "f_dc_1", "f_dc_2"), rgb):
        out[c] = 0.5 * out[c] + 0.5 * ((val - 0.5) / 0.28209479)
    return out

OUT_MOV = os.path.join(RAW, NAME)
OUT_BG = os.path.join(RAW, NAME + "_bg")
for d in (OUT_MOV, OUT_BG):
    os.makedirs(d, exist_ok=True)
    for old in glob.glob(os.path.join(d, "*.ply")):
        os.remove(old)

files = sorted(glob.glob(os.path.join(D, "frame_*_standard.ply")))[:LIMIT]
if not files:
    sys.exit(f"No frames found in {D}")
n = len(files)

# ---------- fast PLY read / write (whole block at once) ----------
TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
RTYPES = {v: k for k, v in TYPES.items()}

def load(p):
    with open(p, "rb") as f:
        count = 0; props = []; in_vertex = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element"):
                in_vertex = line.startswith("element vertex")
                if in_vertex: count = int(line.split()[-1])
            elif line.startswith("property") and in_vertex:
                _, t, name = line.split()
                props.append((name, TYPES[t]))
            elif line == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(props), count=count)

def save(p, arr):
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(arr)}"]
    for name in arr.dtype.names:
        header.append(f"property {RTYPES[arr.dtype[name].str.replace('|', '')]} {name}")
    header.append("end_header")
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        np.ascontiguousarray(arr).tofile(f)
    os.replace(tmp, p)

def xyz(a): return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)

def pad_to(arr, N, template):
    if len(arr) >= N: return arr
    pad = np.repeat(template[:1], N - len(arr))
    pad["opacity"] = -20.0
    for s in ("scale_0", "scale_1", "scale_2"): pad[s] = -20.0
    return np.concatenate([arr, pad])

# ---------- alignment helpers ----------
orb = cv2.ORB_create(4000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
k3 = np.ones((3, 3), np.uint8); k5 = np.ones((5, 5), np.uint8)
bad_align = []

def sane(Hm):
    """Shake is small: reject warps that tilt, flip or zoom a lot."""
    if Hm is None or not np.all(np.isfinite(Hm)): return False
    Hm = Hm / Hm[2, 2]
    det = np.linalg.det(Hm[:2, :2])
    return abs(Hm[2, 0]) < 1e-3 and abs(Hm[2, 1]) < 1e-3 and 0.8 < det < 1.25

def homog(fa, fb, tag):
    """Pixel map from features fa to features fb (perspective, with safe fallbacks)."""
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        bad_align.append(tag); return np.eye(3)
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:1500]
    if len(m) < 8:
        bad_align.append(tag); return np.eye(3)
    src = np.float32([ka[x.queryIdx].pt for x in m]); dst = np.float32([kb[x.trainIdx].pt for x in m])
    Hm = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)[0]
    if sane(Hm): return Hm / Hm[2, 2]
    A = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)[0]
    if A is not None:
        Hm = np.vstack([A, [0, 0, 1]])
        if sane(Hm): return Hm
    bad_align.append(tag); return np.eye(3)

cap = cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES, START)
def read_frame():
    ok, fr = cap.read()
    if not ok: sys.exit("Video ended early - check START and the frame count")
    return fr

first = read_frame(); H, W = first.shape[:2]
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]]); Ki = np.linalg.inv(K)

def rot_from_h(Hm):
    M = Ki @ Hm @ K; M /= np.cbrt(np.linalg.det(M))
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    return -R if np.linalg.det(R) < 0 else R

# splat axis convention (pick the sign flips that put splats inside the picture)
template = load(files[0]); P0 = xyz(template); best = None
for sx in (1, -1):
    for sy in (1, -1):
        for sz in (1, -1):
            z = P0[:, 2] * sz; u = F * P0[:, 0] * sx / z + W / 2; v = F * P0[:, 1] * sy / z + H / 2
            r = ((z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).mean()
            if best is None or r > best[0]: best = (r, sx, sy, sz)
print(f"Self-check: {best[0]*100:.1f}% of splats land inside the picture (want 90%+)")
S = np.array(best[1:], dtype=np.float64)

def project(P):
    Pc = P * S; z = Pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = F * Pc[:, 0] / z + W / 2; v = F * Pc[:, 1] / z + H / 2
    return u, v, (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

def warp_pts(u, v, Hm):
    p = np.stack([u, v, np.ones_like(u)], 1) @ Hm.T
    return p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]

def lookup(mask, u, v, ok):
    out = np.zeros(len(u), bool)
    inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out[inb] = mask[v[inb].astype(int), u[inb].astype(int)] > 0
    return out

def to_global(arr, R):
    out = arr.copy(); g = ((xyz(arr) * S) @ R) * S
    out["x"], out["y"], out["z"] = g[:, 0], g[:, 1], g[:, 2]
    return out

def changed_mask(key_bgr, cur_bgr, Hm):
    """Pixels of cur that really changed vs the (aligned) keyframe. Color-aware,
    and tolerant of 1-pixel edge wobble."""
    rw = cv2.warpPerspective(key_bgr, Hm, (W, H))
    valid = cv2.warpPerspective(np.full((H, W), 255, np.uint8), Hm, (W, H)) > 0
    hi = cv2.dilate(rw, k3).astype(np.int16); lo = cv2.erode(rw, k3).astype(np.int16)
    f = cur_bgr.astype(np.int16)
    diff = np.maximum(f - hi, lo - f).max(axis=2)
    return ((diff > THRESH) & valid).astype(np.uint8)

# ---------- main loop: one segment at a time ----------
segments = [t // KEY_EVERY for t in range(n)]
H0 = np.eye(3)            # pixel map: frame 0 -> current keyframe (chained, so long videos are fine)
prev_key_feat = None
bg_counts, mov_counts = {}, {}
next_frame = first
print(f"Building {n} frames in segments of {KEY_EVERY}...")
for s in range(max(segments) + 1):
    members = [t for t in range(n) if segments[t] == s]
    imgs = []
    for _ in members:
        imgs.append(next_frame if next_frame is not None else read_frame()); next_frame = None
    feats = [orb.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None) for im in imgs]
    kf = members[0]
    if prev_key_feat is not None:
        H0 = homog(prev_key_feat, feats[0], (kf - KEY_EVERY, kf)) @ H0
    prev_key_feat = feats[0]

    # 1. union of everything that changes during this segment (keyframe picture)
    Hs = [np.eye(3)] + [homog(feats[0], feats[j], (kf, kf + j)) for j in range(1, len(members))]
    union = np.zeros((H, W), np.uint8)
    for j in range(1, len(members)):
        ch = changed_mask(imgs[0], imgs[j], Hs[j])
        union |= cv2.warpPerspective(ch, np.linalg.inv(Hs[j]), (W, H), flags=cv2.INTER_NEAREST)
    union = cv2.dilate(union, k5)
    if FILL:
        # A plain shirt that moves barely changes colour inside, so only its edges
        # were marked. That left parts of the person frozen in the background ->
        # cracks + jitter. Close the gaps, then fill each moving shape solid.
        kc = max(5, (H // 60) | 1)
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((kc, kc), np.uint8))
        cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(union, cnts, -1, 1, cv2.FILLED)
    moving_pct = union.mean() * 100

    # 1b. shake per frame, smoothed: the raw estimate is a little noisy frame to
    #      frame, and on a still background that noise shows up as the moving
    #      person jittering. Real hand shake is smooth at 60 fps, so fit a gentle
    #      curve through the segment instead of trusting each frame alone.
    rv = np.array([cv2.Rodrigues(rot_from_h(Hs[j] @ H0))[0].ravel() for j in range(len(members))])
    if SMOOTH and len(members) >= 4:
        tt = np.arange(len(members))
        fit = np.stack([np.polyval(np.polyfit(tt, rv[:, c], 2), tt) for c in range(3)], 1)
        jitter = np.degrees(np.abs(rv - fit).max(1)).mean()
        rv = fit
    else:
        jitter = 0.0
    Rseg = [cv2.Rodrigues(r.reshape(3, 1))[0] for r in rv]
    if not STAB:
        Rseg = [np.eye(3)] * len(members); jitter = 0.0

    # 2. background = keyframe splats outside the union
    key = load(files[kf]); ku, kv, kok = project(xyz(key))
    R_key = Rseg[0]
    bg = to_global(key[~lookup(union, ku, kv, kok)], R_key)
    save(os.path.join(OUT_BG, f"frame_{s + 1:06d}.ply"), tint(bg, (1, 0, 0)))
    bg_counts[s] = len(bg)
    kz = np.full((H // 8 + 1, W // 8 + 1), np.inf)
    sel = kok & ~lookup(union, ku, kv, kok)
    np.minimum.at(kz, ((kv[sel] // 8).astype(int), (ku[sel] // 8).astype(int)), (xyz(key) * S)[sel, 2])
    ratios = []

    # 3. moving = each frame's own splats inside the union
    for j, t in enumerate(members):
        cur = key if j == 0 else load(files[t])
        u, v, ok = project(xyz(cur))
        uk, vk = warp_pts(u, v, np.linalg.inv(Hs[j]))
        mov = lookup(union, uk, vk, ok)
        sub = cur[mov]
        if DEPTH and j > 0:
            # SHARP guesses depth fresh every frame, so the whole frame can come out
            # a few % nearer or further. Next to a fixed background that reads as
            # the person pumping/jittering. Measure it on the still areas, undo it.
            st = ok & ~mov & (uk >= 0) & (uk < W) & (vk >= 0) & (vk < H)
            cz = np.full_like(kz, np.inf)
            np.minimum.at(cz, ((vk[st] // 8).astype(int), (uk[st] // 8).astype(int)), (xyz(cur) * S)[st, 2])
            both = np.isfinite(kz) & np.isfinite(cz)
            r = float(np.median(kz[both] / cz[both])) if both.sum() > 200 else 1.0
            r = float(np.clip(r, 0.8, 1.25)); ratios.append(r)
            sub = sub.copy()
            for c in ("x", "y", "z"): sub[c] *= r
            for c in ("scale_0", "scale_1", "scale_2"): sub[c] += np.log(r)
        m = to_global(sub, Rseg[j])
        save(os.path.join(OUT_MOV, f"frame_{START + t:06d}.ply"), tint(m, (0, 1, 0)))
        mov_counts[t] = len(m)
    print(f"  segment {s + 1}: background {len(bg):,} splats | moving avg "
          f"{np.mean([mov_counts[t] for t in members]):,.0f} per frame | moving area {moving_pct:.0f}%")
    if ratios:
        print(f"      depth fix: {min(ratios):.3f} .. {max(ratios):.3f}  (1.000 = no change)")

# ---------- pad each layer to a constant splat count (the converter requires it) ----------
for folder, counts, names in (
    (OUT_BG, bg_counts, {s: f"frame_{s + 1:06d}.ply" for s in bg_counts}),
    (OUT_MOV, mov_counts, {t: f"frame_{START + t:06d}.ply" for t in mov_counts}),
):
    N = max(max(counts.values()), 1)
    for k, c in counts.items():
        if c == N: continue
        p = os.path.join(folder, names[k])
        arr = load(p) if c > 0 else template[:0]
        save(p, pad_to(arr, N, template))

with open(os.path.join(RAW, NAME + "_layers.json"), "w") as f:
    json.dump({"start": START, "keyEvery": KEY_EVERY, "segments": segments}, f)

full = len(template) * n
uniq = sum(bg_counts.values()) + sum(mov_counts.values())
if bad_align:
    print(f"\n! Warning: {len(set(bad_align))} frame pair(s) could not be lined up "
          f"(treated as 'no shake'). Check those frames for ghosts.")
print(f"\nDone: {n} moving frames + {len(bg_counts)} backgrounds")
print(f"Splats stored: {uniq:,} of {full:,}  ->  SAVINGS {(1 - uniq / full) * 100:.1f}%")
print(f"Next: convert both in Chronosplat, then run tools\\add_background.py {NAME}")
