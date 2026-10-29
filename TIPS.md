# Tips & Commands

## Fresh pod? Run this first
1. `cd /workspace && git clone --recurse-submodules https://github.com/maddi115/maddis-4DGS-Video-Generator.git`
2. `cd maddis-4DGS-Video-Generator && bash setup_pod.sh`
3. Log out and back in (or run `source ~/.bashrc`)

## Generator commands (video -> splat frames)
| Command | What it does |
|---|---|
| `startjob` | Start or resume (never runs twice) |
| `stopjob` | Stop the job (asks first, keeps finished frames) |
| `progress` | Check once: frames, speed, time left, 0-byte and size checks |
| `live` | Same as progress, updates every 5 seconds |

## Converter commands (splat frames -> Chronosplat SOG)
| Command | What it does |
|---|---|
| `convertjob` | Convert all frames (refuses to run while the generator runs) |
| `convertjob --dry-run` | Check frames without converting anything |
| `convertprogress` | Check converter once |
| `convertlive` | Same, updates every 5 seconds |

Run the converter in the background so logging out doesn't stop it:
`nohup convertjob > convert.log 2>&1 &`

## Other
| Command | What it does |
|---|---|
| `tips` | Show this page |
| `claer` | Typo-proof `clear` |

## Safety
- **Ctrl + C** exits `live`, `convertlive` and log views. It never stops the job.
- Anything red in `progress` means something needs a look.
- The job keeps running when you log off or your PC is off. Only stopping the pod stops it.
- **This pod has no permanent storage.** Download results before you stop the pod.
- Never paste GitHub tokens into a chat. Use them only in the terminal.

## Download results to your PC
1. On the pod: `cd /workspace/Chronosplat/data && tar -cf /workspace/meowvideo.tar meowvideo && runpodctl send /workspace/meowvideo.tar`
2. On the PC (PowerShell, in the Chronosplat folder): `.\runpodctl.exe receive THE-CODE`
3. Unpack: `tar -xf meowvideo.tar -C data`
4. Show it: `python tools\index_sequences.py` then `npm run dev`
- If the download shows 0 MB/s: press Enter in that window (clicking pauses it). If still stuck, Ctrl + C both sides and send again.
- Get runpodctl for Windows: `Invoke-WebRequest -Uri https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe -OutFile runpodctl.exe`

## Changing to a new video
These are set for the current video. Change them for a new one:
- `run_headless_pipeline.py`: `VIDEO` and `OUT_NAME`
- `check.sh`: `TOTAL` (frame count) and the output folder
- `convert_on_pod.sh`: `NAME` and `--source-fps` (your video's fps)

## Numbers from the first full run
- 2,204 frames, 1.2M splats each, 63 MB per raw frame (about 139 GB total)
- Generator: about 14 frames/min on an RTX 3090 (after speed fixes)
- Converter: about 90+ frames/min with 12 workers, about 10 MB per SOG frame (about 23 GB total)
- Need about 200 GB pod disk for a 2,200-frame video

## Smaller files: shared store + live people (livepeople+stillbackground branch)
People stay exactly like orig. The still store is saved once per second and follows the camera.
About 78% smaller than orig (full video at 60 fps: about 5 GB instead of 22.8 GB).

### Fresh pod? One command does everything
`cd /workspace && git clone -b livepeople+stillbackground https://github.com/maddi115/maddis-4DGS-Video-Generator && bash maddis-4DGS-Video-Generator/livepeople+stillbackground/rebuild.sh`
- Installs everything (about 20 min), tests SHARP + person finder, then starts the full video by itself.
- No Hugging Face login needed.

| Command | What it does |
|---|---|
| `fullwatch` | Live screen: setup, building, converting, done (Ctrl + C only closes the screen) |
| `tail -f /workspace/setup.log` | Install messages, live |
| `cat /workspace/walmartfull_status.txt` | Check the build once |

### Run a different clip or name
`NOPAD=1 STEP=1 SEG=60 nohup bash /workspace/run_all60.sh <start_frame> <seconds> <name> > /dev/null 2>&1 &`
- `STEP=1` = 60 fps (`STEP=2` = 30 fps, half the size). `SEG=60` = one store per second.
- `NOPAD=1` = no padding (saves about 50 GB of disk, same result).

### Numbers (RTX 4090)
- Build: about 3 s per frame. Convert: under 1 s per frame with 12 workers.
- Disk while working: about 30 GB for the full video (not 200 GB).
- Download: `scp -r -P <PORT> root@<IP>:/workspace/Chronosplat/data/<name> /mnt/c/Users/adamp/Chronosplat/data/` (from WSL)

### On your PC (once)
`src\BackgroundLayer.js` must be the camera-pose version, or the store won't follow the camera.

### Disk warning
Changing the pod's disk size gives you a NEW empty pod. Download results and push code first.

### Real numbers from the full walmart run (RTX 4090)
- 2,204 frames at 60 fps: build 101 min (2.77 s per frame), convert about 15 min.
- Result 4.8 GB (orig 22.8 GB). Disk used while working: about 40 GB.
- 379 camera warnings, all "odd depth scale" or "weak camera match" (shaky footage, handled).
- Download to PC with scp: 1-3 MB/s, about 45 min for 4.8 GB.
- Your PC needs the new viewer file: copy `chronosplat_addons/BackgroundLayer.js` into Chronosplat's `src` folder.
- Full method and history: see OPTIMIZATIONS.md
