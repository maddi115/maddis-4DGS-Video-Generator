import sys, numpy as np, cv2, torch
import detectron2.data.transforms as T
from tools.build_detector import load_detectron2_vitdet
video, start, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
det = load_detectron2_vitdet().to("cuda")
cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, start)
res = {}
for i in range(n):
    ok, img = cap.read()
    if not ok: break
    h, w = img.shape[:2]
    im = T.ResizeShortestEdge(1024, 1024)(T.AugInput(img)).apply_image(img)
    inp = {"image": torch.as_tensor(im.astype("float32").transpose(2, 0, 1)), "height": h, "width": w}
    with torch.no_grad(): o = det([inp])[0]["instances"]
    keep = (o.pred_classes == 0) & (o.scores > 0.5)
    m = o.pred_masks[keep].any(0).cpu().numpy() if keep.any() else np.zeros((h, w), bool)
    res[f"f{start + i}"] = np.packbits(m)
    print(f"frame {start + i}: {int(keep.sum())} people, {m.mean() * 100:.1f}% of picture")
np.savez_compressed(out, shape=np.array([h, w]), **res)
print("saved", out)
