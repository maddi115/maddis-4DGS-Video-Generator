"""check_order.py - are SHARP's splats stored in pixel order? (needed for video compression)

Usage:  python check_order.py <one_frame.ply>

Prints:
  1. YES/NO: splats are a picture grid (and which layout)
  2. how many splats are nearly invisible (free savings)
"""
import sys, numpy as np

TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}

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

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
a = load(sys.argv[1]); N = len(a)
x, y, z = (a[c].astype(np.float64) for c in "xyz")
with np.errstate(divide="ignore", invalid="ignore"):
    px, py = x / z, y / z          # where each splat points, as seen from the camera
print(f"splats: {N:,}")

# ---- 1. grid test: neighbours in the file should be neighbours in the picture ----
def jump(gx, gy, axis):
    """typical distance (in view direction) between neighbouring cells along an axis"""
    d = np.hypot(np.diff(gx, axis=axis), np.diff(gy, axis=axis))
    return np.nanmedian(d[np.isfinite(d)])

rng = np.random.default_rng(0); perm = rng.permutation(N)
shuffled = jump(px[perm], py[perm], 0)
results = []
for L in (1, 2, 3, 4):
    if N % L: continue
    per = N // L
    for w in range(256, 4097):
        if per % w: continue
        h = per // w
        if h < 256 or h > 4096: continue
        for name, shape, move in (
            ("layer, row, col", (L, h, w), None),
            ("row, col, layer", (h, w, L), (2, 0)),
        ):
            gx = px.reshape(shape); gy = py.reshape(shape)
            if move: gx = np.moveaxis(gx, *move); gy = np.moveaxis(gy, *move)
            worst = max(jump(gx, gy, -1), jump(gx, gy, -2))   # across AND down must both be smooth
            results.append((worst / shuffled, name, L, h, w))

results.sort()
print("\n1) PIXEL ORDER TEST")
if results and results[0][0] < 0.05:
    score, name, L, h, w = results[0]
    print(f"   YES - splats are a {w} x {h} picture, {L} layer(s), stored as [{name}]")
    print(f"   (neighbour jump is {score*100:.2f}% of random -> clearly a grid)")
    print("   => the video-compression idea CAN work.")
else:
    best = f"{results[0][0]*100:.0f}% of random" if results else "no grid size fits"
    print(f"   NO - no clean picture layout found (best: {best})")
    print("   => video compression would need an extra sorting step first.")

# ---- 2. free savings: splats you can't see ----
op = 1 / (1 + np.exp(-a["opacity"].astype(np.float64)))
size = np.exp(np.max(np.stack([a[f"scale_{i}"] for i in range(3)]), axis=0).astype(np.float64))
print("\n2) INVISIBLE SPLATS (free savings)")
for t in (0.01, 0.02, 0.05):
    print(f"   opacity under {t:.0%}: {(op < t).mean()*100:5.1f}% of splats")
print(f"   median opacity: {np.median(op):.2f}   median size: {np.median(size):.4f}")
