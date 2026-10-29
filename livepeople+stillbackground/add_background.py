"""add_background.py - attach a converted background layer to a converted sequence.

Run from the Chronosplat folder AFTER converting both:
    tools\\seq.cmd <name> --source-fps <fps>
    tools\\seq.cmd <name>_bg --source-fps <fps>
    python tools\\add_background.py <name>

It moves data/<name>_bg/*.sog into data/<name>/bg/, removes data/<name>_bg,
adds a "background" block to data/<name>/manifest.json, and re-indexes.
"""
import json, shutil, subprocess, sys
from pathlib import Path

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
name = sys.argv[1]
repo = Path(__file__).resolve().parent.parent
data, raw = repo / "data", repo / "raw_data"

layers_path = raw / f"{name}_layers.json"
seq_dir, bg_dir = data / name, data / f"{name}_bg"
for p in (layers_path, seq_dir / "manifest.json", bg_dir / "manifest.json"):
    if not p.exists():
        sys.exit(f"Missing {p} - did both conversions finish?")

layers = json.loads(layers_path.read_text(encoding="utf-8"))
manifest = json.loads((seq_dir / "manifest.json").read_text(encoding="utf-8"))
bg_manifest = json.loads((bg_dir / "manifest.json").read_text(encoding="utf-8"))

segments = layers["segments"]
if len(manifest["frames"]) != len(segments):
    sys.exit(f"Frame count mismatch: manifest has {len(manifest['frames'])}, "
             f"layers file has {len(segments)}. Convert without --frame-step.")
n_bg = len(bg_manifest["frames"])
if max(segments) + 1 != n_bg:
    sys.exit(f"Background count mismatch: need {max(segments) + 1}, found {n_bg}.")

out = seq_dir / "bg"
if out.exists(): shutil.rmtree(out)
out.mkdir()
files, bg_bytes = [], 0
for fr in bg_manifest["frames"]:
    src = bg_dir / fr["file"]
    shutil.move(str(src), str(out / src.name))
    files.append(f"bg/{src.name}")
    bg_bytes += (out / src.name).stat().st_size
shutil.rmtree(bg_dir)

manifest["background"] = {"files": files, "segments": segments,
                          "bytes": bg_bytes, "keyEvery": layers.get("keyEvery"), "poses": layers.get("poses")}
(seq_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

mov_bytes = sum(int(f.get("bytes") or 0) for f in manifest["frames"])
mib = lambda b: f"{b / 1024 / 1024:.1f} MiB"
print(f"Attached {len(files)} background(s) to {name}")
print(f"  moving layer : {mib(mov_bytes)}  ({len(segments)} frames)")
print(f"  background   : {mib(bg_bytes)}  ({len(files)} files)")
print(f"  TOTAL        : {mib(mov_bytes + bg_bytes)}")

subprocess.run([sys.executable, str(repo / "tools" / "index_sequences.py"),
                "--data", str(data)], check=False)
