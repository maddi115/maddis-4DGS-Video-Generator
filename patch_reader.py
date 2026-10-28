p = "build_reuse.py"
s = open(p, encoding="utf-8-sig").read()
old = 'def load(p): return np.array(PlyData.read(p, mmap=False)["vertex"].data)'
new = '''def load(p):
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    with open(p, "rb") as f:
        n = 0; props = []; in_vertex = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element"):
                in_vertex = line.startswith("element vertex")
                if in_vertex: n = int(line.split()[-1])
            elif line.startswith("property") and in_vertex:
                _, t, name = line.split()
                props.append((name, types[t]))
            elif line == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(props), count=n)'''
if "np.fromfile" in s:
    print("already patched")
elif old in s:
    open(p, "w", encoding="utf-8").write(s.replace(old, new, 1)); print("patched")
else:
    print("PATCH FAILED - text not found")