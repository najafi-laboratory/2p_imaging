# Motion energy

## Introduction

This is the pipeline to extract motion energy from the behaviour videos.

## Materials

- The raw session videos on PACE
- [clideo.com](https://clideo.com/) (or an equivalent tool) to crop them
- `step13_motion_energy.py`

## Procedure

### 1. Crop the videos

Keep the mouse's body parts and exclude the environment. If a body part and an
environment part are connected in the frame — the joystick, the reward spouts,
the head bars — the call is a judgment call based on what the analysis needs,
but default to keeping the body part unless instructed otherwise.

You can use [clideo.com](https://clideo.com/) to crop. **Do not change the frame
rate or quality of the video** — motion energy is computed frame-to-frame, so
either would corrupt the trace.

The source videos generally live on PACE at:

```
/storage/project/r-fnajafi3-0/shared/2P_Imaging/video_data
```

Find the project and subject you want to analyze under that path.

### 2. Set the paths

Once cropping is done, make sure the cropped videos and their matching
`.camlog` files sit in a folder, and set these paths at the top of
`step13_motion_energy.py`:

```python
VIDEO_DIR  = r'C:\Users\saminnaji3\Downloads\video_data\SA09_LG_Cropped'
CAMLOG_DIR = r'C:\Users\saminnaji3\Downloads\video_data\SA09_LG'
OUT_DIR    = None      # defaults to <VIDEO_DIR>/motion_energy
```

### 3. Run it and check the output

Once the script runs, the output is saved to `OUT_DIR` — one `.npz` per video,
named after the video's own filename: `<video stem>_me.npz`
(e.g. `sa17-20251226-...-run002-...avi` → `sa17-20251226-...-run002-..._me.npz`).

Each `.npz` holds:

**Motion-energy traces** (one array per ROI, length `n_frames`, indexed
frame-for-frame)
- `<roi>` — e.g. `eye`, `body`. `NaN` at frame 0, since ME is a frame-to-frame
  difference and frame 0 has no predecessor.

**Time axis** (also length `n_frames`)
- `frame_time_ms` — ms since the first camera frame. That instant *is* the 2P
  trigger, so this axis shares its origin with the imaging clock with no
  fitting needed. `NaN` where the camlog can't time a frame — never
  interpolated.
- `me_time_ms` — the same axis at the ME midpoints, since `ME[i]` spans frames
  `i−1 → i`. `NaN` at frame 0.

See [MOTION_ENERGY_IO.md](MOTION_ENERGY_IO.md) for the full input/output
reference for step13 through step15, including the reliability flag, the
camlog diagnostics, and the filename-parsing rules.
