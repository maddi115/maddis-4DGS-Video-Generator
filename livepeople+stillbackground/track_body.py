"""track_body.py - fit a 3D body (SAM 3D Body / MHR rig) to ONE person in every frame,
then measure how steady it is.

Run from the sam-3d-body folder on the pod:
  python track_body.py <video.mp4> <out_dir> [focal=1303.9] [max_frames=300]

Writes:
  <out_dir>/body_track.npz   pose, joints, body shape, mesh for every frame
  <out_dir>/overlay.mp4      the video with her skeleton drawn on top
and prints a STEADINESS report (the thing we are testing).
"""
import os, sys, time
import numpy as np, cv2, torch

if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)
VIDEO, OUT = sys.argv[1], sys.argv[2]
FOCAL = float(sys.argv[3]) if len(sys.argv) > 3 else 1303.9
MAXF = int(sys.argv[4]) if len(sys.argv) > 4 else 300
CKPT = "checkpoints/sam-3d-body-dinov3/model.ckpt"
MHR = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
os.makedirs(OUT, exist_ok=True)

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.build_detector import HumanDetector
from sam_3d_body.metadata.mhr70 import pose_info

dev = torch.device("cuda")
model, cfg = load_sam_3d_body(CKPT, device=dev, mhr_path=MHR)
est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg,
                         human_detector=HumanDetector(name="vitdet", device=dev),
                         human_segmentor=None, fov_estimator=None)

# skeleton lines for the overlay
names = {v["name"]: k for k, v in pose_info["keypoint_info"].items()}
links = [(names[a], names[b]) for a, b in (v["link"] for v in pose_info["skeleton_info"].values())
         if a in names and b in names]

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
K = torch.tensor([[[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1]]], dtype=torch.float32)
writer = cv2.VideoWriter(os.path.join(OUT, "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

KEEP = ["pred_keypoints_3d", "pred_keypoints_2d", "pred_cam_t", "global_rot", "body_pose_params",
        "hand_pose_params", "shape_params", "scale_params", "expr_params", "pred_vertices"]
rec = {k: [] for k in KEEP}; lost = []; prev_box = None; t0 = time.time(); n = 0
while n < MAXF:
    ok, frame = cap.read()
    if not ok: break
    people = est.process_one_image(frame[:, :, ::-1].copy(), cam_int=K, bbox_thr=0.5)
    pick = None
    if people:
        if prev_box is not None:                       # follow the same person as last frame
            scores = [iou(p["bbox"], prev_box) for p in people]
            if max(scores) > 0.1: pick = people[int(np.argmax(scores))]
        if pick is None:                                # first frame / lost her: take the biggest person
            if prev_box is not None: lost.append(n)
            pick = max(people, key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]))
        prev_box = pick["bbox"]
        for k in KEEP: rec[k].append(np.asarray(pick[k], np.float32))
        for i, j in links:
            a, b = pick["pred_keypoints_2d"][i], pick["pred_keypoints_2d"][j]
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), 3)
        for x, y in pick["pred_keypoints_2d"][:21]:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    else:
        lost.append(n)
        for k in KEEP: rec[k].append(rec[k][-1] if rec[k] else None)
    writer.write(frame); n += 1
    if n % 25 == 0: print(f"  {n} frames, {(time.time() - t0) / n:.2f} s/frame")
writer.release()
if not rec["pred_cam_t"] or rec["pred_cam_t"][0] is None:
    sys.exit("Never found a person - check the video.")

arr = {k: np.stack(v) for k, v in rec.items()}
np.savez_compressed(os.path.join(OUT, "body_track.npz"), faces=est.faces, fps=fps, focal=FOCAL, lost=np.array(lost), **arr)

# ---------------- STEADINESS REPORT ----------------
J = arr["pred_keypoints_3d"][:, :21] + arr["pred_cam_t"][:, None]   # main body joints, camera space (metres)
k2 = arr["pred_keypoints_2d"][:, :21]
def noise(x):
    """how much each frame sticks out from the line through its neighbours"""
    return np.linalg.norm(x[1:-1] - 0.5 * (x[:-2] + x[2:]), axis=-1)
speed = np.linalg.norm(np.diff(J, axis=0), axis=-1)
sh = arr["shape_params"]
print(f"\n=== DONE: {n} frames in {time.time() - t0:.0f}s ({(time.time() - t0) / n:.2f} s/frame) ===")
print(f"  lost her on {len(lost)} frame(s)" + (f": {lost[:10]}" if lost else ""))
print("\n=== STEADINESS (lower = steadier) ===")
print(f"  on screen jitter : {np.median(noise(k2)):.1f} px typical, {np.percentile(noise(k2), 95):.1f} px bad frames")
print(f"  3D jitter        : {np.median(noise(J)) * 1000:.0f} mm typical, {np.percentile(noise(J), 95) * 1000:.0f} mm bad frames")
print(f"  real movement    : {np.median(speed) * 1000:.0f} mm per frame (for comparison)")
print(f"  distance wobble  : {arr['pred_cam_t'][:, 2].std() / arr['pred_cam_t'][:, 2].mean() * 100:.1f}% of her distance")
print(f"  body shape wobble: {np.abs(np.diff(sh, axis=0)).mean() / (np.abs(sh).mean() + 1e-9) * 100:.1f}% per frame")
print(f"\nWatch: {os.path.join(OUT, 'overlay.mp4')}")
