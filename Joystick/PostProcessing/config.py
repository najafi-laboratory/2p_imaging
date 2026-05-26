#!/usr/bin/env python3
"""
Central configuration for the post-processing pipeline.
Edit the values here — every step reads from this file.
"""

# ── Data root on PACE ─────────────────────────────────────────────────────────
BASE_PATH = '/storage/project/r-fnajafi3-0/shared/2P_Imaging'

# ── Automated morphological QC thresholds ────────────────────────────────────
RANGE_SKEW      = [0.4, 2.5]   # raised upper: active dendrites can have skew up to ~2.5
MAX_CONNECT     = 30
MAX_ASPECT      = 55
RANGE_FOOTPRINT = [1, 2]
RANGE_COMPACT   = [1.0, 5.0]   # raised upper: keeps elongated dendrites; lower kept at 1.0 so no real dendrites are cut
DIAMETER        = 6           # cellpose diameter for anatomical channel (step 3)

# ── ΔF/F computation ─────────────────────────────────────────────────────────
SIG_BASELINE = 600            # Gaussian baseline sigma in frames (~20 s at 30 Hz)

# ── Event detection  (GCaMP8s, Purkinje dendrites, L7-Cre, ~30 Hz) ───────────
FS                    = 30    # imaging frame rate (Hz)
SMOOTH_SIGMA          = 1.0   # Gaussian smoothing σ in frames (~33 ms) — light smoothing preserves peak amplitude
THRESHOLD_FACTOR      = 0.9   # threshold  = factor × per-neuron MAD noise floor (loosened for dendrites)
MIN_ISI_S             = 0.3   # minimum inter-event interval  (s) — CS refractory
MIN_PROMINENCE_FACTOR = 1.0   # minimum peak prominence in noise units (loosened)
