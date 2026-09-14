# -*- coding: utf-8 -*-
"""
mains/reward_error_sb/step13_motion_energy.py
──────────────────────────────────────────────
PHASE B of the arousal question: motion energy from the behaviour video.

Motion energy is the mean absolute frame-to-frame difference inside a region:

    ME(t) = mean over ROI pixels | frame(t) - frame(t-1) |

Two ROIs are computed in ONE streaming pass over the file -- one decode, no
duplicated storage, ROIs stay adjustable without re-exporting video, and the
two traces are frame-synchronised by construction. (Pre-cropping into two
separate video files would cost a second decode and lock the ROIs in.)

  eye / pupil region -> blinks, saccades, pupil-driven luminance change
  body region        -> whisking, grooming, postural shifts, fidgeting

These are complementary, not redundant: pupil AREA is the classic arousal
proxy but `camera_pupil` is empty in neural_data for these sessions, so the
video is the only source. Eye-region ME is not pupil area -- it responds to
movement rather than dilation -- but it is informative about the same axis.


USAGE
─────
  1. Look at the frames first; the videos are already cropped and the ROI
     coordinates depend on what the crop contains:

         python step13_motion_energy.py --inspect

     writes a grid-overlaid montage per video so ROI rectangles can be read
     off directly.

  2. Put the coordinates in ROIS below (or pass --rois), then:

         python step13_motion_energy.py

     writes one .npz per video with the per-ROI ME traces, frame timestamps
     from the .camlog, and the ROI definitions used.


OUTPUT -- one .npz per video
────────────────────────────
    <roi>                 n_frames   motion energy, NaN at frame 0
    frame_time_ms         n_frames   ms since the FIRST CAMERA FRAME, which is
                                     the 2P trigger, so this axis shares its
                                     origin with the imaging clock. NaN where
                                     the camlog cannot time a frame.
    me_time_ms            n_frames   the same axis at the ME midpoints (ME[i]
                                     spans frames i-1 -> i)
    time_source                      'camlog' or 'none'
    camlog_fps, camlog_t0_s          measured rate; the camlog's own origin
    time_reliable                    False if the camlog contains anything odd
    time_unreliable_reason           what, in words ('' when reliable)
    camlog_stall_frames/_s           camera stalls, if any
    camlog_backward_frames           frames after which time runs backwards
    frame_timestamps, frame_ids      the raw camlog, verbatim
    rois, downsample                 what was computed

Alignment to the imaging clock is deliberately NOT done here -- it belongs with
the neural analysis. This step's job is to turn video into a per-frame trace
AND to record when each of those frames happened, which is knowable only while
the video and its camlog are in hand.

Two clocks can time these frames. The DAQ voltage trace is the better one: the
rise and fall of the camera exposure line bracket each exposure, so the frame
CENTRE is measurable. But that connection was sometimes loose and the trace is
then missing entirely, whereas labcams always writes a camlog -- and because the
2P start triggers the camera, one timestamp per frame plus that shared origin is
enough to time the whole video on its own. So the camlog axis is computed here
unconditionally, as the fallback, and the voltage route can override it later.
"""

import os, sys, re, glob, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

VIDEO_DIR = r'C:\Users\saminnaji3\Downloads\video_data\SA09_LG_Cropped'
CAMLOG_DIR = r'C:\Users\saminnaji3\Downloads\video_data\SA09_LG'
OUT_DIR   = None      # defaults to <VIDEO_DIR>/motion_energy

# ROI = (x, y, width, height) in pixels of the CROPPED frame.
# None means "whole frame" -- which is what --inspect starts from.
ROIS = {
    'eye':  None,
    'body': None,
}

DOWNSAMPLE   = 2      # spatial stride; ME is robust to this and it is ~4x faster
N_INSPECT    = 6      # sample frames per video in --inspect mode
GRID_STEP    = 50     # px between grid lines in the montage


def _import_reader():
    """cv2 if present, else imageio-ffmpeg. Both stream frame by frame; a
    whole session is ~77k frames and must never be loaded at once."""
    try:
        import cv2
        return 'cv2', cv2
    except ImportError:
        pass
    try:
        import imageio.v2 as imageio
        return 'imageio', imageio
    except ImportError:
        raise SystemExit('Need either cv2 or imageio to read video.\n'
                         '  pip install opencv-python-headless   (or)   pip install imageio[ffmpeg]')


def read_camlog(path):
    """frame_id, timestamp (s) from a labcams .camlog. Comment lines start
    with '#'; the third column is an undocumented queue flag and is ignored."""
    if not os.path.isfile(path):
        return None
    ids, ts = [], []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                ids.append(int(parts[0])); ts.append(float(parts[1]))
            except ValueError:
                continue
    if not ids:
        return None
    return np.array(ids), np.array(ts)


def find_camlog(video_path, camlog_dir=None):
    """Originals keep the camlog beside the video under an identical stem, so
    try that first. Cropped copies are renamed and use '-' where the originals
    use '_', so fall back to the first 8-digit date token -- that one is the
    session date; a second date later in the name is the run start and can
    differ from it.

    The date ALONE is not unique. Three dates in this cohort carry two runs
    each, and taking the first match by sort order pairs the SA16/20251226
    run001 video (62270 frames) with the run000 camlog (4577 frames). That was
    survivable while the camlog was only a frame-count witness; now that it
    supplies the time axis, a mispairing would time the whole session from the
    wrong recording. So the runNNN token is matched too whenever both names
    carry one -- all 119 videos and every camlog here do -- and the date-only
    match remains as the fallback for names that do not."""
    stem = os.path.splitext(video_path)[0]
    if os.path.isfile(stem + '.camlog'):
        return stem + '.camlog'
    base = os.path.basename(video_path)
    m = re.search(r'(?<!\d)(\d{8})(?!\d)', base)
    if m is None:
        return None
    date = m.group(1)
    rm = re.search(r'run(\d+)', base, re.I)
    run = rm.group(1) if rm else None

    dated = []
    for d in [os.path.dirname(video_path), camlog_dir or CAMLOG_DIR]:
        if not d or not os.path.isdir(d):
            continue
        for cand in sorted(glob.glob(os.path.join(d, '*.camlog'))):
            if re.search(r'(?<!\d)' + date + r'(?!\d)', os.path.basename(cand)):
                dated.append(cand)
    if not dated:
        return None
    if run is not None:
        # Compared as an integer, not a string: every name in this cohort pads
        # to three digits (run007), but a rig that writes run7 should still pair
        # with run007 rather than silently falling through to the date-only match.
        exact = [c for c in dated
                 if (lambda r: r and int(r.group(1)) == int(run))(
                     re.search(r'run(\d+)', os.path.basename(c), re.I))]
        if exact:
            return exact[0]
        if len(dated) > 1:
            # Guessing between runs is what this function exists to avoid.
            print('    WARNING: %d camlogs for %s and none matches run%s; '
                  'using %s' % (len(dated), date, run,
                                os.path.basename(dated[0])))
    return dated[0]


# ---------------------------------------------------------------------------
# Per-frame time axis from the camlog
# ---------------------------------------------------------------------------
# The 2P start TRIGGERS the camera, so video frame 0 and imaging frame 0 share
# an origin BY HARDWARE. A camera-clock axis zeroed at its own first frame is
# therefore directly comparable to the imaging `time` axis with no fitting.
#
# The DAQ voltage trace is the better clock -- the rise and fall of the camera
# exposure line bracket each exposure, so the frame CENTRE is measurable there.
# But that connection was sometimes loose and the trace is then missing, while
# labcams always writes the camlog. Given the shared origin, one timestamp per
# frame is enough to time the whole video, so the camlog axis is written here
# unconditionally: it is the fallback clock, and computing it while the video is
# open beats reconstructing it later from a filename match.
#
# Do NOT substitute index/fps for it. Across this cohort's 60 camlogs the frame
# interval is stable (~29.61 fps), but two sessions stall MID SESSION with no
# break in frame_id -- the camera kept counting while real time jumped.
# SA17/20260111 loses 128 s at frame 25213 and SA11/20250812 loses 0.9 s at
# frame 102158. A constant-fps axis is wrong by the whole stall for every frame
# after one, and only the recorded timestamps show that it happened.
#
# WHEN THE CAMLOG IS WEIRD, SAY SO -- DO NOT REPAIR IT
# Seven of the 60 camlogs contain something that should not be there: a stall,
# or a first row whose timestamp is LATER than the second (SA19/20251229, by
# 27.4 s -- time running backwards, so that value is simply garbage). Each of
# those could be read either as a corrupt row to be patched or as a real pause
# to be kept, and the two readings shift the whole session against the imaging
# clock by seconds in opposite directions. Nothing in the camlog distinguishes
# them, so this step does not guess: the axis is written from the timestamps
# exactly as recorded, and the session is FLAGGED as temporally unreliable.
#
# `time_reliable` / `time_unreliable_reason` go into the .npz so a downstream
# step can drop or annotate the session without reopening the camlog, and the
# reason is printed for every affected video and again as a summary at the end
# of the run. The voltage trace, when present, is what actually resolves these
# sessions -- it times each exposure directly instead of trusting the camera's
# own log.

STALL_FACTOR = 5.0        # an interval above this x the median is a stall

# Frame-count mismatch tolerated as ordinary re-encode loss, as a fraction of
# the camlog length. Matches MAX_MISMATCH_FRAC in step14, which applies the same
# judgement to the imaging clock. Beyond it, which video frame is which
# acquisition is no longer knowable, so the session is flagged.
MAX_COUNT_MISMATCH_FRAC = 0.005


def _infer_unit_scale(ts):
    """(scale to ms, median interval in the file's own unit).

    The unit is not declared in the camlog. A behaviour camera runs at tens of
    Hz, so its frame interval is ~0.034 in seconds or ~34 in milliseconds --
    three orders apart, so the two cannot be confused. Every camlog in this
    cohort is seconds; the check exists so a rig that writes milliseconds is
    caught rather than silently yielding an axis 1000x too short."""
    d = np.diff(np.asarray(ts, float))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 1000.0, np.nan
    step = float(np.median(d))
    return (1000.0, step) if step < 1.0 else (1.0, step)


def camlog_time_axis(ids, ts, n_video):
    """Per-VIDEO-frame time in ms, zeroed at the first camera frame.

    Milliseconds because that is the unit of every clock in neural_data.h5
    (`time`, `flir_time`, `js_time`), so this axis drops straight in beside
    them; zeroed at the first frame because that instant is the 2P trigger and
    is therefore the one point the camera and the imaging clock are known to
    share.

    Returns (t_ms, info). t_ms has length n_video, so it indexes element for
    element with the ME traces. A frame with no camlog row is NaN, never
    interpolated -- an invented timestamp is exactly the error this axis exists
    to prevent, and so is an invented correction: the timestamps are used as
    recorded, and info['reasons'] lists anything that makes them untrustworthy.

    Video frame i is mapped to camlog frame_id ids[0] + i, not to camlog ROW i.
    The two coincide while the log is contiguous (all 60 in this cohort are),
    but keying on the id means a log that DOES skip an acquisition leaves a NaN
    at the gap instead of silently shifting every frame after it."""
    ids = np.asarray(ids, int)
    scale, _ = _infer_unit_scale(ts)
    raw = np.asarray(ts, float) * scale          # ms, still on the camera clock
    t_log = raw - raw[0]                         # as recorded; nothing repaired

    contiguous = bool(np.all(np.diff(ids) == 1)) if len(ids) > 1 else True
    t = np.full(int(n_video), np.nan)
    if contiguous:
        k = min(len(t_log), int(n_video))
        t[:k] = t_log[:k]
    else:
        at = {int(v): j for j, v in enumerate(ids)}
        for i in range(int(n_video)):
            j = at.get(int(ids[0]) + i)
            if j is not None:
                t[i] = t_log[j]

    d = np.diff(t_log)
    pos = d[d > 0]
    med = float(np.median(pos)) if pos.size else np.nan
    stall = (np.where(d > STALL_FACTOR * med)[0]
             if np.isfinite(med) and med > 0 else np.zeros(0, int))
    back = np.where(d <= 0)[0]
    n_miss = len(ids) - int(n_video)

    # Each entry is a complete sentence naming what is wrong and where, because
    # it is what the run log and the .npz both carry as the explanation.
    reasons = []
    if len(back):
        reasons.append(
            'time runs backwards after frame(s) %s (worst %.2f s)'
            % (', '.join(str(int(x)) for x in back[:6]), d[back].min() / 1000.0))
    if len(stall):
        reasons.append(
            '%d camera stall(s) totalling %.2f s, after frame(s) %s'
            % (len(stall), float((d[stall] - med).sum()) / 1000.0,
               ', '.join(str(int(x)) for x in ids[stall][:6])))
    if not contiguous:
        reasons.append('frame_id is not contiguous (%d frame(s) missing from '
                       'the log)' % int(np.diff(ids).sum() + 1 - len(ids)))
    if abs(n_miss) > MAX_COUNT_MISMATCH_FRAC * max(len(ids), 1):
        reasons.append('camlog %d frames vs video %d (%+d, %.2f%%) -- too many '
                       'to be re-encode loss'
                       % (len(ids), int(n_video), n_miss,
                          100.0 * abs(n_miss) / max(len(ids), 1)))
    if not (np.isfinite(med) and med > 0):
        reasons.append('no usable frame interval in the camlog')

    return t, dict(
        unit=('s' if scale == 1000.0 else 'ms'),
        fps=(1000.0 / med if np.isfinite(med) and med > 0 else np.nan),
        contiguous=contiguous,
        n_camlog=len(ids),
        n_timed=int(np.isfinite(t).sum()),
        # dd[j] is the interval AFTER frame j, so these name the last frame
        # before each gap
        stall_frames=(ids[stall] if len(stall) else np.zeros(0, int)),
        stall_s=((d[stall] - med) / 1000.0 if len(stall) else np.zeros(0)),
        backward_frames=(ids[back] if len(back) else np.zeros(0, int)),
        reasons=reasons,
    )


def _frames(video_path, backend, mod):
    """Yield grayscale, optionally downsampled frames one at a time."""
    if backend == 'cv2':
        cap = mod.VideoCapture(video_path)
        if not cap.isOpened():
            raise SystemExit(f'could not open {video_path}')
        try:
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if fr.ndim == 3:
                    fr = mod.cvtColor(fr, mod.COLOR_BGR2GRAY)
                yield fr[::DOWNSAMPLE, ::DOWNSAMPLE]
        finally:
            cap.release()
    else:
        rd = mod.get_reader(video_path)
        try:
            for fr in rd:
                if fr.ndim == 3:
                    fr = fr.mean(axis=2)
                yield np.asarray(fr)[::DOWNSAMPLE, ::DOWNSAMPLE]
        finally:
            rd.close()


def _roi_slice(roi):
    if roi is None:
        return (slice(None), slice(None))
    x, y, w, h = [int(v / DOWNSAMPLE) for v in roi]
    return (slice(y, y + h), slice(x, x + w))


def motion_energy(video_path, rois, backend, mod, report_every=20000):
    """Single streaming pass; ME for every ROI at once."""
    names = list(rois)
    slices = {n: _roi_slice(rois[n]) for n in names}
    out = {n: [] for n in names}
    prev = None
    n = 0
    for fr in _frames(video_path, backend, mod):
        f = fr.astype(np.float32)
        if prev is not None:
            d = np.abs(f - prev)
            for nm in names:
                sy, sx = slices[nm]
                out[nm].append(float(d[sy, sx].mean()))
        prev = f
        n += 1
        if report_every and n % report_every == 0:
            print(f'      {n} frames ...', flush=True)
    # ME is a difference, so it has one fewer sample than there are frames;
    # pad the front with NaN to keep it frame-indexed.
    return {nm: np.concatenate([[np.nan], np.array(v)]) for nm, v in out.items()}, n


def inspect(video_path, backend, mod, out_dir):
    """Save a montage of sample frames with a coordinate grid, so ROI
    rectangles can be read straight off the image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frames, keep = [], None
    gen = _frames(video_path, backend, mod)
    first = next(gen, None)
    if first is None:
        print(f'    no frames in {os.path.basename(video_path)}'); return
    frames.append(first)
    # sample across the file without decoding all of it twice
    for i, fr in enumerate(gen):
        if i % 3000 == 0 and len(frames) < N_INSPECT:
            frames.append(fr)
        if len(frames) >= N_INSPECT:
            break

    h, w = frames[0].shape
    ncol = min(3, len(frames))
    nrow = int(np.ceil(len(frames) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow * h / max(w, 1)),
                             squeeze=False)
    for i, fr in enumerate(frames):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(fr, cmap='gray')
        step = max(GRID_STEP // DOWNSAMPLE, 5)
        for gx in range(0, w, step):
            ax.axvline(gx, color='#00e5ff', lw=0.3, alpha=0.6)
            ax.text(gx, 2, str(gx * DOWNSAMPLE), fontsize=4, color='#00e5ff', va='top')
        for gy in range(0, h, step):
            ax.axhline(gy, color='#00e5ff', lw=0.3, alpha=0.6)
            ax.text(2, gy, str(gy * DOWNSAMPLE), fontsize=4, color='#00e5ff', ha='left')
        ax.set_title(f'frame ~{i * 3000}', fontsize=7)
        ax.axis('off')
    for j in range(len(frames), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(f'{os.path.basename(video_path)}\n'
                 f'full frame {w * DOWNSAMPLE} x {h * DOWNSAMPLE} px  --  '
                 'grid labels are FULL-RESOLUTION pixel coords; read ROIs as (x, y, w, h)',
                 fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0] + '_inspect.png')
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'    -> {p}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inspect', action='store_true',
                    help='dump grid-overlaid sample frames to pick ROIs, then exit')
    ap.add_argument('--dir', default=VIDEO_DIR)
    ap.add_argument('--camlog-dir', default=None,
                    help='where to look for .camlog files if not beside the video')
    ap.add_argument('--video', default=None,
                    help='process just this one file (for a Slurm array)')
    ap.add_argument('--rois', default=None,
                    help='e.g. "eye=100,40,120,90;body=0,0,640,480"')
    args = ap.parse_args()

    backend, mod = _import_reader()
    print(f'video backend: {backend}')

    vids = sorted(glob.glob(os.path.join(args.dir, '*.avi')))
    if args.video:
        vids = [args.video if os.path.isabs(args.video)
                else os.path.join(args.dir, args.video)]
    if not vids:
        raise SystemExit(f'no .avi found in {args.dir}')
    out_dir = OUT_DIR or os.path.join(args.dir, 'motion_energy')
    os.makedirs(out_dir, exist_ok=True)
    print(f'{len(vids)} videos, output -> {out_dir}\n')

    if args.inspect:
        for v in vids:
            print(f'  {os.path.basename(v)}')
            inspect(v, backend, mod, out_dir)
        print('\nRead ROI rectangles off the montages, then set ROIS in this file '
              '(or pass --rois) and rerun without --inspect.')
        return

    rois = dict(ROIS)
    if args.rois:
        rois = {}
        for part in args.rois.split(';'):
            name, vals = part.split('=')
            rois[name.strip()] = tuple(int(v) for v in vals.split(','))
    if all(v is None for v in rois.values()):
        print('WARNING: every ROI is None (whole frame). Run --inspect and set '
              'ROIS first, or the two traces will be identical.\n')

    unreliable = []
    for v in vids:
        print(f'  {os.path.basename(v)}')
        me, n_frames = motion_energy(v, rois, backend, mod)
        cam = find_camlog(v, args.camlog_dir)
        log = read_camlog(cam) if cam else None
        ids, ts = log if log is not None else (None, None)
        if ts is not None:
            t_ms, ti = camlog_time_axis(ids, ts, n_frames)
            reasons = ti['reasons']
            print(f'    {n_frames} video frames, camlog {ti["n_camlog"]} frames, '
                  f'{ti["fps"]:.2f} fps (camlog unit: {ti["unit"]})')
            if ti['n_camlog'] != n_frames and not reasons:
                print(f'    NOTE: camlog/video frame count differ by '
                      f'{ti["n_camlog"] - n_frames} -- ordinary re-encode loss; '
                      f'{ti["n_timed"]}/{n_frames} video frames timed.')
        else:
            t_ms, ti = np.full(n_frames, np.nan), {}
            reasons = ['no camlog found']
            print(f'    {n_frames} frames, no camlog found.')

        if reasons:
            # One unmissable banner per affected video, and the same sessions
            # are listed again at the end of the run -- a warning buried in the
            # middle of a 60-video Slurm log is a warning nobody reads.
            print('    *** TEMPORAL DATA NOT RELIABLE for this session ***')
            for r in reasons:
                print(f'        - {r}')
            print('        frame_time_ms is written from the timestamps AS '
                  'RECORDED and is not corrected here.')
            print('        Use the voltage recording to time this session, or '
                  'exclude it.')
            unreliable.append((os.path.basename(v), reasons))

        # ME[i] is the difference between frames i-1 and i, so the interval it
        # measures is centred BETWEEN their timestamps. That midpoint, not the
        # frame time, is when the movement happened; both are stored so the
        # choice stays explicit rather than implied by a half-frame convention.
        me_t = np.full(n_frames, np.nan)
        if n_frames > 1:
            me_t[1:] = 0.5 * (t_ms[:-1] + t_ms[1:])
        for nm, tr in me.items():
            good = np.isfinite(tr)
            print(f'    ME[{nm}]: mean={np.nanmean(tr):.3f}  sd={np.nanstd(tr):.3f}  '
                  f'({int(good.sum())} valid)')
        p = os.path.join(out_dir, os.path.splitext(os.path.basename(v))[0] + '_me.npz')
        np.savez_compressed(
            p,
            # the raw camlog, kept verbatim so nothing derived here is lossy
            frame_timestamps=(ts if ts is not None else np.array([])),
            frame_ids=(ids if ids is not None else np.array([])),
            # the time row: ms since the first camera frame -- which IS the 2P
            # trigger -- one entry per video frame, NaN where the camlog cannot
            # say. me_time_ms is the same axis at the ME midpoints.
            frame_time_ms=t_ms.astype(np.float64),
            me_time_ms=me_t.astype(np.float64),
            time_source=np.array('camlog' if ts is not None else 'none'),
            camlog_fps=float(ti.get('fps', np.nan)),
            camlog_t0_s=(float(ts[0]) if ts is not None else np.nan),
            camlog_stall_frames=ti.get('stall_frames', np.zeros(0, int)),
            camlog_stall_s=ti.get('stall_s', np.zeros(0)),
            camlog_backward_frames=ti.get('backward_frames', np.zeros(0, int)),
            # the verdict, so a downstream step can drop or annotate this
            # session without reopening the camlog
            time_reliable=(not reasons),
            time_unreliable_reason=np.array('; '.join(reasons)),
            rois=np.array(str(rois)), downsample=DOWNSAMPLE, **me)
        print(f'    -> {os.path.basename(p)}\n')

    if unreliable:
        print('=' * 72)
        print('TEMPORAL DATA NOT RELIABLE -- %d of %d video(s):'
              % (len(unreliable), len(vids)))
        for name, reasons in unreliable:
            print('  %s' % name)
            for r in reasons:
                print('      - %s' % r)
        print('These sessions need the voltage recording to be timed, or should '
              'be excluded.')
        print('=' * 72)
    else:
        print('All %d video(s) have a clean camlog time axis.' % len(vids))


if __name__ == '__main__':
    main()
