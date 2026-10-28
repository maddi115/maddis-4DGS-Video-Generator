import cv2, numpy as np
cap = cv2.VideoCapture("sample-video/walmartvideo.mp4")
def frame(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(4000)
for i in [0, 100, 1000]:
    a, b = frame(i), frame(i + 1)
    ka, da = orb.detectAndCompute(a, None); kb, db = orb.detectAndCompute(b, None)
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da), key=lambda x: x.distance)[:1000]
    src = np.float32([kb[x.queryIdx].pt for x in m]); dst = np.float32([ka[x.trainIdx].pt for x in m])
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    b_aligned = cv2.warpAffine(b, M, (a.shape[1], a.shape[0]))
    rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    before = (np.abs(a.astype(float) - b) > 10).mean() * 100
    after = (np.abs(a.astype(float) - b_aligned) > 10)[20:-20, 20:-20].mean() * 100
    print(f"Frames {i}/{i+1}: shake = shift {M[0,2]:+.2f},{M[1,2]:+.2f} px, rotation {rot:+.3f} deg | changed: before {before:.1f}% -> after stabilizing {after:.1f}%")