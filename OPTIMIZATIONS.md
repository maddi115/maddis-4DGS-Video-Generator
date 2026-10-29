# Optimizations (livepeople+stillbackground branch)

Goal: much smaller files than orig (SHARP's per-frame splats), with people looking exactly like orig.

## Result: full walmart video, 60 fps, 2,204 frames
| | Orig | Ours |
|---|---|---|
| Size | 22.8 GB | 4.8 GB (about 79% smaller) |
| Splats per frame | 1,179,648 | about 185,000 people + 1 store per second |
| People | SHARP | exactly SHARP (untouched) |

## What we do, in order
1. **Person masks.** Detectron2 ViTDet finds every person in each frame.
2. **Live people.** Splats that SHARP made from person pixels (plus a small margin) are kept every frame, exactly as SHARP made them.
3. **Still store.** Everything else is saved ONCE per second (from the middle frame of that second) and reused for all 60 frames.
4. **Birth-pixel rule.** Each splat is sorted by the pixel that made it, not where it lands. This fixed legs getting cut off at the picture edge.
5. **Camera-view playback.** People are never moved. The viewer moves only the store, using a per-frame camera pose. This fixed "she moves all over the place".
6. **Smoothed camera.** Camera moves are matched on store points only (ORB + depth + RANSAC). Failed frames are filled from neighbours, then smoothed, so a bad guess can't make a jump.
7. **No padding (NOPAD=1).** The converter's equal-count check is switched off. Saves about 50 GB of temporary disk; result is the same.
8. **Min-alpha 0.001.** Invisible splats are dropped while converting. One invisible keep-alive splat stops any frame from being empty.

## Speed (RTX 4090)
- Build: 2.77 s per frame (SHARP + person finder on GPU, file saving in 4 background threads, building in a separate thread).
- Convert: 12 workers, about 15 min for the full video.
- Full video: about 2 hours total. Disk while working: about 40 GB.

## Things that did NOT work (so we don't try them again)
- Moving people into one world with the camera guess: every bad guess moved her whole body.
- Puppet/skeleton (SAM 3D Body) and "statue" sculptures: shaky hands, split hair, cracks and ghosting. Orig was better.
- Averaging frames: blur and ghosts.
- Colour-based "is it moving" detector: cracked the aisles and split her body.
- Video-codec tricks on the splat data: smoother but visibly worse than orig.

## Known limits
- Shaky footage gives camera warnings (379 on the full video). People are not affected; the shelves behind them can slide a little.
- When a person leaves the frame they disappear (same as the source video).

## Next ideas
- Faster downloads: several transfers at once, or stream from cloud storage (Cloudflare R2 + tools/set_host.py), like Polycam.
- Compress the live people further (they are about 90% of the size now).
- "Memory" fill for parts of people the camera saw earlier.
