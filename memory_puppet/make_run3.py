src = open("/workspace/segment_run2.py").read()
def rep(old, new, count=1):
    global src
    assert src.count(old) == count, f"patch spot not found ({src.count(old)}x): {old[:60]}"
    src = src.replace(old, new)
rep('camera kept still"); return ID', 'camera kept still"); return None', 3)
rep('''    if state["world"] is None:
        world = ID
    else:
        pkey, pfeat_src, pD, pmask = state["prev"]
        fk = orb.detectAndCompute(gray[key], (1 - np.maximum(key_out, pmask)) * 255)
        world = compose(state["world"], solve(fk, D[key], pfeat_src, pD, f"piece {si + 1} -> piece {si}"))
    store_sel = ~(key_out[VB, UB] > 0)
    save(OUT_B / f"frame_{si + 1:06d}.ply", moved(plys[key][store_sel], world))
''', '''    store_sel = ~(key_out[VB, UB] > 0)
    save(OUT_B / f"frame_{si + 1:06d}.ply", plys[key][store_sel])
    raw = [ID if i == key else solve(feat[i], D[i], feat[key], D[key], f"frame {i}") for i in seg]
    ok = np.array([r is not None for r in raw]); n = len(seg); ix = np.arange(n); par = np.zeros((n, 7))
    for j, r in enumerate(raw):
        if r is not None: par[j] = [np.log(r[0]), *Rotation.from_matrix(r[1]).as_rotvec(), *r[2]]
    for c in range(7): par[:, c] = np.interp(ix, ix[ok], par[ok, c])
    par = uniform_filter1d(median_filter(par, size=(5, 1), mode="nearest"), 3, axis=0, mode="nearest")
    D3 = np.diag(S_SH)
    for j, i in enumerate(seg):
        s, R, t = ID if i == key else (np.exp(par[j, 0]), Rotation.from_rotvec(par[j, 1:4]).as_matrix(), par[j, 4:7])
        Rf = D3 @ R.T @ D3; c = -(1 / s) * (D3 @ R.T @ t)
        poses[i] = [float(1 / s), *map(float, Rotation.from_matrix(Rf).as_quat()), *map(float, c)]
''')
rep('''        tr = ID if i == key else solve(feat[i], D[i], feat[key], D[key], f"frame {i}")
        save(OUT_P / f"frame_{i:06d}.ply", moved(plys[i][live_sel], compose(world, tr)))
''', '''        save(OUT_P / f"frame_{i:06d}.ply", plys[i][live_sel])
''')
rep('''    gk = orb.detectAndCompute(gray[key], (1 - key_out) * 255)
    state["prev"] = (key, gk, D[key], key_out); state["world"] = world
''', '')
rep('counts, seg_of = {}, {}', 'counts, seg_of, poses = {}, {}, {}\nfrom scipy.ndimage import median_filter, uniform_filter1d')
rep('''    N = max(counts[k] for k in keys)
    for k, p in keys.items():
        a = load(p)
        if len(a) < N:
            pd = np.repeat(a[:1], N - len(a)); pd["opacity"] = -20
            for c in ("scale_0", "scale_1", "scale_2"): pd[c] = -20
''', '''    N = max(counts[k] for k in keys)
    tmpl = next(load(p)[:1].copy() for k, p in keys.items() if counts[k] > 0)
    for k, p in keys.items():
        a = load(p)
        if len(a) < N:
            pd = np.repeat(tmpl, N - len(a)); pd["opacity"] = -20
            for c in ("scale_0", "scale_1", "scale_2"): pd[c] = -20
            pd["opacity"][0] = 0
''')
rep('"segments": [seg_of[i] for i in frames_all]}', '"segments": [seg_of[i] for i in frames_all], "poses": [poses[i] for i in frames_all]}')
src = src.replace("segment_run2.py", "segment_run3.py")
open("/workspace/segment_run3.py", "w").write(src)
compile(src, "segment_run3.py", "exec"); print("segment_run3.py ready")
ab = "/workspace/Chronosplat/tools/add_background.py"; t = open(ab).read()
old = '"keyEvery": layers.get("keyEvery")}'
if '"poses"' not in t:
    assert t.count(old) == 1, "add_background.py patch spot not found"
    open(ab, "w").write(t.replace(old, '"keyEvery": layers.get("keyEvery"), "poses": layers.get("poses")}'))
print("add_background.py ready")
