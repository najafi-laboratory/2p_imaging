# Manual ROI workflow — visual-stimulus artifact measurement and removal

Written 2026-09-15. Covers steps 1–6b in this directory.

## What this is for

The visual stimulus leaks light onto the detector. It adds roughly the same
number of ADU to **every pixel of the frame**, so it appears in every ROI as a
stimulus-locked deflection that is not calcium. Manual ROIs are drawn over
regions with no fluorescent structure: whatever they record is therefore pure
artifact, which gives a direct measurement of it and a way to remove it from the
real ROIs.

Steps 1–5 create those ROIs. **Steps 6 and 6b are the verification and removal**
and are not part of the lab's published protocol — the shipped workflow stops at
step 5 and names no verification step at all.

## The model everything rests on

For ROI *i*, with `S_i` the calcium signal and `A(t)` the artifact:

```
F_i(t)     = F0_i     + S_i(t) + A(t)          A is identical for every ROI
Fneu_i(t)  = F0neu_i  + N_i(t) + A(t)
```

A blank ROI has `S_m = 0`, so `F_m − F0_m = A(t)` — the artifact, measured
directly, in ADU, with no division.

Two consequences that drive the whole design:

**In raw units the correct subtraction weight is exactly 1.** No fitting is
needed. It is only dF/F that breaks the symmetry, because dividing by `F0_i`
turns the common `A(t)` into `A(t)/F0_i`, which differs per ROI. A blank patch
is dim, so the same artifact reads as a much larger *fraction* there — 7.3× on
SA18_20260327. This is not a bug and not a scaling error; it is why any weighted
dF/F removal needs `b ≈ F0_manual / F0_auto`.

**Neuropil subtraction already removes 70% of it.** `F − 0.7·Fneu` retains
`0.3·A`, because the artifact sits in both terms. So the pipeline's traces are
contaminated by `0.3·A`, not `A`, and the subtraction must be done between
matching quantities.

## Steps

### 1–5 — creating the ROIs (existing protocol)

| step | file | notes |
|---|---|---|
| 1 | `step1_patch_ops.py` | only if `ops.npy` predates this workflow |
| 2 | `step2_build_data_bin.sh` | Slurm, ~1 h |
| 3 | `step3_create_workspace.sh` | delete any existing `manual_roi_workspace` first |
| 4 | `step4_open_gui.sh` | PACE OOD Desktop, not SSH |
| 5 | `step5_export_workspace.sh` | run *after* step 6 confirms the ROIs are good |

### 6 — the check and the removal

```bash
python step6_manual_vs_auto_responses.py /path/to/SESSION_DIR
```

Reads `manual_roi_workspace/{F,Fneu,stat,stat_orig,ops}.npy` and
`neural_trials.h5` (only `time` and `stim_labels` — timing). Computes dF/F
itself; it never reads a stored `dff` file, because none of them contain the
manual ROIs. Writes `manual_vs_auto_roi_responses.pdf` into the session
directory, and a numeric summary to stdout.

Run it **between steps 4 and 5**, while the ROIs exist only in the workspace: a
bad ROI then costs a redraw rather than a restore-from-backup.

### 6b — the naive-subtraction diagnostic

```bash
python step6b_raw_f_subtraction.py /path/to/SESSION_DIR
```

Demonstrates what happens if you subtract the manual ROI's raw `F` from the auto
ROI's raw `F` literally. Keep it next to step6; it imports step6's loaders.

## Reading the step6 figure

Columns, left to right:

| panel | contents |
|---|---|
| **Raw F** | trial-averaged raw `F`, no neuropil correction, each trace shifted to zero pre-stimulus. Navy auto, red manual, green difference. Baselines given in the title. |
| **Field of view** | green = auto ROIs, red = manual |
| **Auto ROIs** | the population dF/F before any correction, mean ± SEM |
| **THE SUBTRACTION** | both operands in ADU. Navy `F_auto − 0.7·Fneu_auto − F0` (subtracted *from*), red the manual equivalent (subtracted), green dashed the difference. Everything else on the page is downstream of this. |
| **Manual ROI k artifact in ADU** | solid red `F − 0.7·Fneu − F0`, faint grey `F − F0`. Their ratio should be ≈ 0.3. |
| **Manual ROI k** | the same ROI as dF/F |
| *one column per method* | the corrected auto population, uncorrected trace behind in faint blue |

Grey band = the stimulus. Pink band in `nan` = the frames blanked. The auto panel
and every method column share one y-range; manual panels share a second; ADU
panels a third. `--free-y` restores per-panel autoscaling.

## Methods

Every method column plots `auto − b·manual_k`; they differ only in how `b` is
obtained.

| method | b from | bias |
|---|---|---|
| `raw` | **nothing** — subtracts `A` in ADU, then recomputes F0 and dF/F from scratch | none; **the physically correct one** |
| `phys` | closed form, `b = F0_k · mean(1/F0_auto)` | none, but assumes uniformity |
| `split` | errors-in-variables over disjoint trial folds | unbiased; noisy when `rel r` is low |
| `proj` | projection of the auto average onto the artifact waveform | **high** — fit to the trace being tested |
| `ols` / `trial` | per-ROI least squares, whole session / peri-stimulus frames | **low** — regression dilution, a blank ROI is mostly noise |
| `nan` | — deletes the contaminated frames | none; assumes nothing |
| `interp` | — fills them using the AR(1) calcium decay constant | interpolation, not measurement |
| `sub` | b = 1 | **wrong by construction**; kept as the naive baseline |

Default: `raw,phys,split,proj,nan,sub`.

### Reading order

1. **ADU column** — is there an artifact, and is red ≈ 0.3 × grey?
2. **THE SUBTRACTION** — do navy and red overlap? If so, most of the signal is artifact.
3. **`raw`** — the answer.
4. **`nan`** — does it agree without assuming anything? Two disagreeing assumptions reaching the same residual is the strongest evidence available.
5. **`phys` vs `split`** — a closed form and a regression agreeing is the validation.
6. **`rel r`** — only then trust `split`.

## Built-in consistency checks

Printed to stdout on every run. All three should hold; if one fails, the model
does not apply to that session and `raw` should not be trusted there.

| check | expected | SA18_20260422 |
|---|---|---|
| neuropil subtraction retains 0.3 of a uniform artifact | 0.30 | **0.31** |
| auto annuli see the same artifact as blank patches | 1.00 | **1.11** |
| `phys` (F0 ratio) vs `split` (regression) | equal | agree to 0.01 on 0327 |

## Traps

**Suite2p PREPENDS manual ROIs.** It does not append them. On SA18_20260422 the
hand-drawn ROIs are rows 0–7 of a 702-row `stat.npy`; rows 698–701 are ordinary
detections. Any "last k rows" rule silently analyses arbitrary dendrites.
Detection order is `stat['manual']`, then a footprint diff against
`stat_orig.npy`, then `--manual-rows`. Never positional.

**Duplicate manual ROIs.** SA18_20260422 holds each of its four manual ROIs
twice, byte-identical in footprint and in `F.npy`. Step6 drops the duplicates
from the plotted set and keeps them out of the auto population, and prints a
NOTE. **Running step 5 on such a workspace copies the duplicates into the QC
files** — clean it first.

**Negative F0.** If `0.7·Fneu > F` for a patch — plausible over a vessel lumen —
the neuropil-corrected baseline goes negative and every dF/F and every weight
fitted against it flips sign. SA18_20260422's M4 sits at −58 ADU and returns
b = −0.44. Such ROIs are titled `UNSTABLE F0` in red and warned about on stdout.
Read the ADU column, which is immune.

**`ops['fs']` is nominal and wrong.** 30.0 against a real 33.75 ms frame period
(29.63 Hz). Timing comes from `neural_trials/time`.

**`vol_stim_vis` is all zeros** on these sessions. Onsets come from
`stim_labels[:,0]`, durations from column 1.

**Baseline-unstable ROIs destroy a raw dF/F mean.** One ROI whose rolling
baseline crosses zero reaches dF/F = 1.7e4 while the median across ROIs is
−0.002. Excluded from raw dF/F averages only, via `--min-baseline-frac`.

**The window can overlap the next stimulus.** On SA18_20260422 a second stimulus
falls at ~1.75 s, clearly visible in the Raw F panel. Use `--window -0.5 1.5`.

## Results so far

**SA18_20260327** (passive, 257 auto + 5 manual, 2000 onsets) — the artifact is
large and highly reproducible (`rel r` 0.82–0.96). `phys`, `split` and `proj`
agree at **b ≈ 0.15**, and five independent blank patches span only 0.117–0.162.
F0 ratio 7.35× effective.

- uncorrected peak **0.073** at ~0.12 s → corrected **~0.028** at ~0.20 s
- so **~60–70% of the apparent response is artifact**, ~30–40% is neural
- the residual peaks *after* the artifact ends and decays over ~230 ms against a
  fitted calcium τ of 276 ms — different timing and kinetics from the thing
  subtracted
- `nan`, which subtracts nothing and only deletes frames, finds the same
  excursion in frames the artifact never touched. That is what rules out a
  removal error.

**SA18_20260422** — `rel r` 0.17–0.55, so the artifact is far less reproducible
here and the fitted weights are unreliable. F0 ratio 2.53× effective.

## Limits

**The response during the stimulus is unrecoverable** when the artifact is large.
On 0327 it is ~0.5 dF/F against a neural signal of ~0.03 — 15:1. `raw` gives an
estimate there but it rests entirely on the additivity assumption; `nan` honestly
refuses. Onset amplitude and latency cannot be reported from such a session.
Fixing that needs acquisition changes: interleaved stimulus-off trials, or a
shorter stimulus so the calcium peak clears the artifact window.

**Nothing here separates a real response that shares the artifact's shape and
timing.** That confound is in the data, not the estimator.

**`rel r` is computed over the whole window**, so a brief but perfectly
reproducible artifact in a long window still scores low. Low `rel r` means "most
of the window is noise", not "no artifact". Restricting it to the contaminated
frames would be fairer and is not yet done.

**Where to draw future manual ROIs.** A blank patch whose *annulus* contains real
tissue biases the artifact estimate, because `F − 0.7·Fneu` then subtracts real
neuropil calcium along with the artifact. Measured on 0422 the bias is −9%, in
the safe direction. The diagnostic is `manual Fneu − manual F`: near zero means
the surroundings are as empty as the patch.
