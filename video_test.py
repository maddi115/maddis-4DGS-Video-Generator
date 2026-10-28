"""video_test.py - OFFLINE test: squeeze SHARP frames through a video codec and back.

SHARP stores its splats as a picture grid, so each splat property (depth, colour,
size, ...) is really an image. We turn every property into its own grey video,
let H.265 compress it, decode it again and write normal PLY files you can view
next to the original in Chronosplat.

Usage:
  python video_test.py <frames_dir> <start_frame> <chronosplat_raw_data_dir> <name> [limit=15]
  env: FPS=15, quality per part (lower = better, bigger): CRF_GEO=8 CRF_COL=18 CRF_SC=22 CRF_ROT=26

Prints the REAL compressed size (the video files) and how far splats moved.
"""
import glob, os, shutil, subprocess, sys, time
import numpy as np

if len(sys.argv) < 5:
    print(__doc__); sys.exit(1)
D, START, RAW, NAME = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else 15
FPS = os.environ.get("FPS", "15")
CRF_GEO = os.environ.get("CRF_GEO", "8")
CRF_COL = os.environ.get("CRF_COL", "18")    # colour
CRF_SC = os.environ.get("CRF_SC", "22")      # opacity + size
CRF_ROT = os.environ.get("CRF_ROT", "26")    # rotation (matters little for round splats)
CLIP = float(os.environ.get("CLIP", "0.02")) # % of splats per property kept exact instead of stretching the range
WIDTH = 768  # SHARP grid width (check_order.py said 768)

files = sorted(glob.glob(os.path.join(D, "frame_*_standard.ply")))[:LIMIT]
if not files: sys.exit(f"No frames found in {D}")
n = len(files)

# ---------- ffmpeg ----------
ff = shutil.which("ffmpeg")
if not ff:
    try:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("No ffmpeg found. Run:  pip install imageio-ffmpeg   then try again.")
enc = subprocess.run([ff, "-hide_banner", "-h", "encoder=libx265"], capture_output=True, text=True).stdout
if "gray12le" in enc:
    CODEC, PIX, BITS = "libx265", "gray12le", 12
else:
    enc = subprocess.run([ff, "-hide_banner", "-h", "encoder=libx264"], capture_output=True, text=True).stdout
    if "gray10le" not in enc: sys.exit("This ffmpeg has no 10/12-bit grey encoder (need libx265 or libx264).")
    CODEC, PIX, BITS = "libx264", "gray10le", 10
QMAX = (1 << BITS) - 1

# ---------- fast PLY read / write ----------
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
                _, t, name = line.split(); props.append((name, TYPES[t]))
            elif line == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(props), count=count)

def save(p, arr):
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(arr)}"]
    header += [f"property {RTYPES[arr.dtype[k].str.replace('|', '')]} {k}" for k in arr.dtype.names]
    header.append("end_header")
    with open(p, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii")); np.ascontiguousarray(arr).tofile(f)

# ---------- splat <-> smooth images ----------
# Positions are stored as  where-it-points (x/z, y/z)  +  log depth.
# Those are smooth pictures, which is what video codecs are good at.
GEO = ["u", "v", "logz"]
REST = ["f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"]

def to_channels(a):
    z = a["z"].astype(np.float64); zs = np.sign(np.median(z)); az = np.abs(z)
    ch = {"u": a["x"] / z, "v": a["y"] / z, "logz": np.log(np.maximum(az, 1e-6))}
    q = np.stack([a[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    q *= np.where(q[:, :1] < 0, -1, 1)  # q and -q are the same turn; pick one so the image is smooth
    for k in REST: ch[k] = a[k]
    for i in range(4): ch[f"rot_{i}"] = q[:, i]
    return {k: np.asarray(v, np.float64) for k, v in ch.items()}, zs

def from_channels(ch, zs, template):
    out = np.zeros(len(template), template.dtype)
    z = np.exp(ch["logz"]) * zs
    out["x"], out["y"], out["z"] = ch["u"] * z, ch["v"] * z, z
    for k in REST: out[k] = ch[k]
    return out

# ---------- pass 1: value ranges ----------
t0 = time.time()
template = load(files[0]); N = len(template); HROWS = N // WIDTH
if N % WIDTH: sys.exit(f"{N} splats is not a multiple of {WIDTH} - not a SHARP grid?")
names = GEO + REST
# Range per property from the bulk of the splats. A handful of strange splats
# (e.g. sitting right at the camera, so x/z is huge) would otherwise stretch the
# range and make every normal splat coarse. Those few are stored exactly instead.
lo = {k: np.inf for k in names}; hi = {k: -np.inf for k in names}; signs = []
for p in files:
    ch, zs = to_channels(load(p)); signs.append(zs)
    for k in names:
        v = ch[k][np.isfinite(ch[k])]
        a_, b_ = np.percentile(v, [CLIP, 100 - CLIP])
        lo[k] = min(lo[k], a_); hi[k] = max(hi[k], b_)
def outliers(ch):
    bad = np.zeros(N, bool)
    for k in names:
        v = ch[k]; bad |= ~np.isfinite(v) | (v < lo[k]) | (v > hi[k])
    return bad
print(f"{n} frames, {N:,} splats each, grid {WIDTH} x {HROWS}, codec {CODEC} {BITS}-bit")

# ---------- pass 2: encode (one grey video per property) ----------
work = os.path.join(RAW, NAME + "_video"); os.makedirs(work, exist_ok=True)
procs = {}
for k in names:
    crf = (CRF_GEO if k in GEO else CRF_COL if k.startswith("f_dc") else
           CRF_ROT if k.startswith("rot") else CRF_SC)
    procs[k] = subprocess.Popen(
        [ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", PIX,
         "-s", f"{WIDTH}x{HROWS}", "-r", FPS, "-i", "-", "-c:v", CODEC, "-crf", crf,
         "-preset", "medium", "-pix_fmt", PIX,
         *(["-x265-params", "log-level=error"] if CODEC == "libx265" else []), os.path.join(work, f"{k}.mp4")],
        stdin=subprocess.PIPE)

def quant(k, v):
    span = hi[k] - lo[k] or 1.0
    return np.clip(np.round((np.nan_to_num(v, nan=lo[k]) - lo[k]) / span * QMAX), 0, QMAX).astype("<u2")

def dequant(k, q):
    return lo[k] + q.astype(np.float64) / QMAX * (hi[k] - lo[k])

side = {}   # frame -> (indices, exact original splats) for the few out-of-range ones
for t, p in enumerate(files):
    a = load(p); ch, _ = to_channels(a)
    bad = outliers(ch); idx = np.nonzero(bad)[0].astype("<u4"); side[t] = (idx, a[idx])
    for k in names: procs[k].stdin.write(quant(k, ch[k]).tobytes())
np.savez_compressed(os.path.join(work, "exact.npz"),
                    **{f"i{t}": side[t][0] for t in side}, **{f"s{t}": side[t][1] for t in side})
for k in names:
    procs[k].stdin.close(); procs[k].wait()
    if procs[k].returncode: sys.exit(f"ffmpeg failed on {k}")
vid_bytes = {k: os.path.getsize(os.path.join(work, f"{k}.mp4")) for k in names}
vid_bytes["exact"] = os.path.getsize(os.path.join(work, "exact.npz"))
n_exact = sum(len(side[t][0]) for t in side) / n
print(f"encoded in {time.time() - t0:.0f}s")

# ---------- pass 3: decode, rebuild PLYs, measure error ----------
out_dir = os.path.join(RAW, NAME); os.makedirs(out_dir, exist_ok=True)
for old in glob.glob(os.path.join(out_dir, "*.ply")): os.remove(old)
dec = {k: subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-i", os.path.join(work, f"{k}.mp4"),
                            "-f", "rawvideo", "-pix_fmt", PIX, "-"], stdout=subprocess.PIPE) for k in names}
frame_bytes = N * 2
err_all, col_all = [], []
for t, p in enumerate(files):
    orig = load(p)
    ch = {}
    for k in names:
        buf = dec[k].stdout.read(frame_bytes)
        if len(buf) != frame_bytes: sys.exit(f"decode of {k} ended early at frame {t}")
        ch[k] = dequant(k, np.frombuffer(buf, "<u2"))
    rec = from_channels(ch, signs[t], orig)
    idx, exact = side[t]; rec[idx] = exact
    save(os.path.join(out_dir, f"frame_{START + t:06d}.ply"), rec)
    P = np.stack([orig[c] for c in "xyz"], 1).astype(np.float64)
    Q = np.stack([rec[c] for c in "xyz"], 1).astype(np.float64)
    err_all.append(np.linalg.norm(P - Q, axis=1) / np.maximum(np.abs(P[:, 2]), 1e-6))
    col_all.append(np.abs(np.stack([orig[f"f_dc_{i}"] - rec[f"f_dc_{i}"] for i in range(3)], 1)).max(1) * 0.2821 * 255)
for k in names: dec[k].wait()

err = np.concatenate(err_all) * 100; col = np.concatenate(col_all)
total = sum(vid_bytes.values())
mib = lambda b: f"{b / 1024 / 1024:.1f} MB"
geo = sum(vid_bytes[k] for k in GEO)
print("\n=== SIZE (the real compressed size) ===")
print(f"  positions : {mib(geo)}")
print(f"  the rest  : {mib(total - geo)}")
print("  each part : " + ", ".join(f"{k} {v / 1024 / 1024:.1f}" for k, v in vid_bytes.items()))
print(f"  kept exact: {n_exact:,.0f} odd splats per frame ({n_exact / N * 100:.2f}%)")
print(f"  TOTAL     : {mib(total)}  for {n} frames  ({mib(total / n)} per frame)")
print(f"  orig SOG was about 10.7 MB per frame  ->  {10.7 * 1024 * 1024 / (total / n):.1f}x smaller")
print(f"  full video (2,204 frames) would be about {total / n * 2204 / 1024**3:.2f} GB")
print("\n=== HOW FAR SPLATS MOVED (as % of their distance from camera) ===")
print(f"  typical splat : {np.median(err):.3f}%")
print(f"  worst 1%      : {np.percentile(err, 99):.3f}%")
print(f"  moved > 1%    : {(err > 1).mean() * 100:.2f}% of splats   (these could look like floating dots)")
print(f"  colour change : typical {np.median(col):.1f} / worst 1% {np.percentile(col, 99):.1f}  (out of 255)")
sc = np.stack([template[f"scale_{i}"] for i in range(3)], 1).astype(np.float64)
ratio = np.exp(sc.max(1) - sc.min(1))
print(f"\n  splat shape: {(ratio < 1.5).mean() * 100:.0f}% are nearly round (rotation hardly matters for those)")
print(f"\nNext: tools\\seq.cmd {NAME} --source-fps {FPS}   then compare {NAME} with calm15_orig")
