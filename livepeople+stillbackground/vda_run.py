import sys, numpy as np, torch
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames
vid, out = sys.argv[1], sys.argv[2]
m = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=True)
m.load_state_dict(torch.load("checkpoints/metric_video_depth_anything_vits.pth", map_location="cpu"), strict=True)
m = m.cuda().eval()
frames, fps = read_video_frames(vid, -1, -1, 1280)
depths, _ = m.infer_video_depth(frames, fps, input_size=518, device="cuda", fp32=False)
depths = np.asarray(depths, np.float32)
np.savez_compressed(out, depths=depths)
print("saved", out, depths.shape, "median depth", float(np.median(depths)), "m")
