import numpy as np, sys
from plyfile import PlyData
def load(p):
    v = PlyData.read(p)["vertex"]
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    col = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
    return pos, col, op
a, ca, oa = load(sys.argv[1]); b, cb, ob = load(sys.argv[2])
size = np.median(np.linalg.norm(a, axis=1))
moved = np.linalg.norm(a - b, axis=1) / size
print(f"Splats: {len(a):,}")
print(f"Position: median diff {np.median(moved)*100:.4f}% of scene size, 99% under {np.percentile(moved, 99)*100:.3f}%")
print(f"Color:    median diff {np.median(np.abs(ca - cb)):.5f}")
print(f"Opacity:  median diff {np.median(np.abs(oa - ob)):.5f}")