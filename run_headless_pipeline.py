import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import torch

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "v02"))
sys.path.append(str(ROOT / "ml-sharp" / "src"))

from video_processor import VideoProcessor
from convert_sharp_ply import convert_sharp_ply_to_standard
from sharp.models import PredictorParams, create_predictor
from sharp.cli.predict import predict_image, DEFAULT_MODEL_URL
from sharp.utils.gaussians import save_ply

# SPEED FIX 3: keep SHARP's covariance decomposition on the GPU (remove its .cpu() call).
import inspect as _inspect
import sharp.utils.gaussians as _sg3
_old_line = ".detach().cpu().to(torch.float64)"
_new_line = ".detach().to(torch.float64)"
for _name, _fn in list(vars(_sg3).items()):
    if not (_inspect.isfunction(_fn) and _fn.__module__ == _sg3.__name__):
        continue
    try:
        _src = _inspect.getsource(_fn)
    except (OSError, TypeError):
        continue
    if _old_line in _src:
        exec(_src.replace(_old_line, _new_line), _sg3.__dict__)
        _new = _sg3.__dict__[_name]
        for _m in list(sys.modules.values()):
            if _m and getattr(_m, "__name__", "").startswith("sharp") and getattr(_m, _name, None) is _fn:
                setattr(_m, _name, _new)
        print(f"[+] {_name} now runs on GPU")

# SPEED FIX 2: SHARP's save_ply builds 1.2M Python tuples per frame. Copy the block in one go instead.
import inspect
import numpy as np
import sharp.utils.gaussians as _sg
_slow = "elements[:] = list(map(tuple, attributes.detach().cpu().numpy()))"
_fast = "elements[:] = np.ascontiguousarray(attributes.detach().cpu().numpy(), dtype=np.float32).view(dtype_full).reshape(-1)"
_src = inspect.getsource(_sg.save_ply)
if _slow in _src:
    _dt = [(f"a{i}", "f4") for i in range(14)]
    _test = torch.randn(1000, 14, dtype=torch.float64)
    _a = np.empty(1000, dtype=_dt); _a[:] = list(map(tuple, _test.numpy()))
    _b = np.empty(1000, dtype=_dt); _b[:] = np.ascontiguousarray(_test.numpy(), dtype=np.float32).view(_dt).reshape(-1)
    assert _a.tobytes() == _b.tobytes(), "fast save does not match, refusing to use it"
    exec(_src.replace(_slow, _fast), _sg.__dict__)
    save_ply = _sg.save_ply
    print("[+] Fast save enabled (verified identical)")
else:
    print("[!] Fast save NOT applied (code changed), using original")

# SPEED FIX: SHARP does a big SVD on the CPU (one core). Run it on the GPU instead.
_orig_svd = torch.linalg.svd
def _gpu_svd(A, *args, **kwargs):
    if A.device.type == "cpu" and A.ndim == 3 and A.shape[0] > 10000:
        U, S, Vh = _orig_svd(A.to("cuda"), *args, **kwargs)
        return U.cpu(), S.cpu(), Vh.cpu()
    return _orig_svd(A, *args, **kwargs)
torch.linalg.svd = _gpu_svd

VIDEO = ROOT / "sample-video" / "walmartvideo.mp4"
OUT_NAME = "meowvideo"
CHECKPOINT = ROOT / "sharp_2572gikvuh.pt"
TEST_FRAMES = None        # None = all frames, or a number like 5 for a test
MAX_BAD_IN_A_ROW = 5      # stop if this many frames in a row come out empty
SAVE_WORKERS = 4          # helper threads that save + convert files
MAX_IN_FLIGHT = 6         # frames allowed to wait for saving at once
TAG = "_Shramko_4DGS_apple-Sharp_Generator_"

def load_model(device):
    if CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, weights_only=True, map_location="cpu")
    else:
        state = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True, map_location="cpu")
    model = create_predictor(PredictorParams())
    model.load_state_dict(state)
    return model.eval().to(device)

def is_good(path):
    return path.exists() and path.stat().st_size > 0

def cleanup(out_dir):
    for p in out_dir.glob("*.tmp.ply"):
        p.unlink()
        print(f"[CLEAN] removed temp file {p.name}")
    for p in out_dir.glob(f"frame_*{TAG}.ply"):
        p.unlink()
        print(f"[CLEAN] removed leftover raw file {p.name}")
    std = list(out_dir.glob(f"frame_*{TAG}_standard.ply"))
    if std:
        normal_size = Counter(p.stat().st_size for p in std).most_common(1)[0][0]
        for p in std:
            if p.stat().st_size != normal_size:
                p.unlink()
                print(f"[CLEAN] removed cut-off frame {p.name}")

def save_and_convert(i, gaussians, f_px, h, w, out_dir):
    raw_ply = out_dir / f"frame_{i:06d}{TAG}.ply"
    tmp_ply = out_dir / f"frame_{i:06d}{TAG}_standard.tmp.ply"
    std_ply = out_dir / f"frame_{i:06d}{TAG}_standard.ply"
    save_ply(gaussians, f_px, (h, w), raw_ply)
    result = convert_sharp_ply_to_standard(raw_ply, tmp_ply)
    raw_ply.unlink(missing_ok=True)
    if result is not None and is_good(tmp_ply):
        os.replace(tmp_ply, std_ply)
        return i, True
    tmp_ply.unlink(missing_ok=True)
    return i, False

def main():
    out_dir = ROOT / "output_3dgs" / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup(out_dir)

    device = torch.device("cuda")
    model = load_model(device)
    print("[+] Model loaded on GPU")

    with VideoProcessor(VIDEO) as vp:
        info = vp.get_info()
        f_px = vp.estimate_focal_length(info["width"], info["height"])
    total = info["frame_count"] if TEST_FRAMES is None else TEST_FRAMES
    print(f"[*] {total} frames | focal {f_px:.1f}px | {SAVE_WORKERS} save workers | out: {out_dir}")

    stats = {"ok": 0, "bad": 0, "skipped": 0, "streak": 0}
    sem = threading.Semaphore(MAX_IN_FLIGHT)
    pool = ThreadPoolExecutor(max_workers=SAVE_WORKERS)
    pending = []

    def collect(wait_all=False):
        nonlocal pending
        still = []
        for fut in pending:
            if not (wait_all or fut.done()):
                still.append(fut)
                continue
            i, good = fut.result()   # a real error stops everything loudly
            if good:
                stats["ok"] += 1
                stats["streak"] = 0
                print(f"[OK] {i}/{total-1}")
            else:
                stats["bad"] += 1
                stats["streak"] += 1
                print(f"[BAD] Frame {i}: empty output, deleted")
                if stats["streak"] >= MAX_BAD_IN_A_ROW:
                    print(f"[STOP] {stats['streak']} bad frames in a row. Something is wrong.")
                    pool.shutdown(wait=True, cancel_futures=True)
                    sys.exit(1)
        pending = still

    cap = cv2.VideoCapture(str(VIDEO))
    for i in range(total):
        got, frame = cap.read()
        if not got:
            print(f"[*] Video ended at frame {i}")
            break
        std_ply = out_dir / f"frame_{i:06d}{TAG}_standard.ply"
        if is_good(std_ply):
            stats["skipped"] += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gaussians = predict_image(model, rgb, f_px, device)

        sem.acquire()
        fut = pool.submit(save_and_convert, i, gaussians, f_px, h, w, out_dir)
        fut.add_done_callback(lambda _f: sem.release())
        pending.append(fut)
        collect()

    cap.release()
    collect(wait_all=True)
    pool.shutdown(wait=True)
    print(f"[+] Done. ok={stats['ok']} bad={stats['bad']} skipped={stats['skipped']}")

if __name__ == "__main__":
    main()
