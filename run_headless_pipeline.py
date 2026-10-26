import sys
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

VIDEO = ROOT / "sample-video" / "walmartvideo.mp4"
OUT_NAME = "meowvideo"
CHECKPOINT = ROOT / "sharp_2572gikvuh.pt"
TEST_FRAMES = 5        # None = all frames, or a number like 5 for a test
MAX_BAD_IN_A_ROW = 5      # stop if this many frames in a row come out empty
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

def main():
    out_dir = ROOT / "output_3dgs" / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = load_model(device)
    print("[+] Model loaded on GPU")

    with VideoProcessor(VIDEO) as vp:
        info = vp.get_info()
        f_px = vp.estimate_focal_length(info["width"], info["height"])
    total = info["frame_count"] if TEST_FRAMES is None else TEST_FRAMES
    print(f"[*] {total} frames | focal {f_px:.1f}px | out: {out_dir}")

    cap = cv2.VideoCapture(str(VIDEO))
    ok = bad = skipped = streak = 0
    for i in range(total):
        got, frame = cap.read()
        if not got:
            print(f"[*] Video ended at frame {i}")
            break

        std_ply = out_dir / f"frame_{i:06d}{TAG}_standard.ply"
        if is_good(std_ply):
            skipped += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        raw_ply = out_dir / f"frame_{i:06d}{TAG}.ply"

        gaussians = predict_image(model, rgb, f_px, device)
        save_ply(gaussians, f_px, (h, w), raw_ply)
        result = convert_sharp_ply_to_standard(raw_ply, std_ply)
        raw_ply.unlink(missing_ok=True)

        if result is not None and is_good(std_ply):
            ok += 1
            streak = 0
            print(f"[OK] {i}/{total-1}")
        else:
            bad += 1
            streak += 1
            std_ply.unlink(missing_ok=True)
            print(f"[BAD] Frame {i}: empty output, deleted")
            if streak >= MAX_BAD_IN_A_ROW:
                print(f"[STOP] {streak} bad frames in a row. Something is wrong.")
                sys.exit(1)

    cap.release()
    print(f"[+] Done. ok={ok} bad={bad} skipped={skipped}")

if __name__ == "__main__":
    main()
