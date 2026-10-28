"""patch_viewer.py - teach Chronosplat's player to show a background layer.

Run from the Chronosplat folder (after putting BackgroundLayer.js in src\\):
    python tools\\patch_viewer.py

Makes 6 small edits to src/main.js. It checks every edit point first and
changes NOTHING if any of them is missing. Safe to run twice.
"""
import sys
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
main = repo / "src" / "main.js"
if not (repo / "src" / "BackgroundLayer.js").exists():
    sys.exit("Put BackgroundLayer.js in the src folder first.")

s = main.read_text(encoding="utf-8")
if "BackgroundLayer" in s:
    print("already patched"); sys.exit(0)

edits = [
    ("import { FrameCache } from './FrameCache.js';",
     "import { FrameCache } from './FrameCache.js';\n"
     "import { BackgroundLayer } from './BackgroundLayer.js';"),
    ("    sequencer.canAdvance = (i) => cache.isReady(i);",
     "    const bgLayer = new BackgroundLayer({\n"
     "      scene, base, background: nextManifest.background, flipX: cache.flipX,\n"
     "    });\n"
     "    sequencer.canAdvance = (i) => cache.isReady(i) && bgLayer.isReadyFor(i);"),
    ("        const flipped = cache.setFlipX(!cache.flipX);",
     "        const flipped = cache.setFlipX(!cache.flipX);\n"
     "        bgLayer.setFlipX(flipped);"),
    ("    session = { sequencer, cache, audio, timeline,",
     "    session = { sequencer, cache, bgLayer, audio, timeline,"),
    ("      timeline.setLoadingFrame(cache.activePending || sequencer.stalled);",
     "      session.bgLayer?.update(sequencer.index, cache._visibleIndex);\n"
     "      timeline.setLoadingFrame(cache.activePending || sequencer.stalled);"),
    ("    session.cache.disposeAll(); // frees every resident SplatMesh",
     "    session.cache.disposeAll(); // frees every resident SplatMesh\n"
     "    session.bgLayer?.dispose();"),
]

missing = [old for old, _ in edits if s.count(old) != 1]
if missing:
    print("PATCH FAILED - these lines were not found exactly once:")
    for m in missing: print("   ", m.strip())
    sys.exit(1)

backup = main.with_name("main.js.bak")
backup.write_text(main.read_text(encoding="utf-8"), encoding="utf-8")
for old, new in edits:
    s = s.replace(old, new, 1)
main.write_text(s, encoding="utf-8")
print(f"backup saved: {backup}")
print("patched: main.js now shows background layers")
