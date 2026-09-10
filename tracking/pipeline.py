"""
pipeline.py
===========
Helpers for the ROICaT tracking pipeline.

The main entry point for a notebook run is :func:`filter_sessions_by_overlap`,
which handles the load → screen → filter → reload cycle so the notebook never
needs to re-run a cell.
"""

from __future__ import annotations

import numpy as np
import roicat
from results_table import session_dir
from scipy.sparse.csgraph import connected_components

# Per-method parameters for ``Aligner.fit_geometric``.  Only the entry named by
# ``method`` is read, but roicat wants the whole dict, so all options stay here
# documented.  This is the default for a standalone call; the notebook defines
# its own copy and passes the same one to both the screen and the real fit.
KWARGS_ALIGN_METHOD = {
    "RoMa": {  ## Accuracy: Best, Speed: Very slow (can be fast with a GPU).
        "model_type": "outdoor",
        "n_points": 10000,  ## Higher values mean more points are used for the registration. Useful for larger FOV_images. Larger means slower.
        "batch_size": 1000,
    },
    "DISK_LightGlue": {  ## Accuracy: Good, Speed: Fast.
        "num_features": 3000,  ## Number of features to extract and match. I've seen best results around 2048 despite higher values typically being better.
        "threshold_confidence": 0.0,  ## Higher values means fewer but better matches.
        "window_nms": 7,  ## Non-maximum suppression window size. Larger values mean fewer non-suppressed points.
    },
    "LoFTR": {  ## Accuracy: Okay. Speed: Medium.
        "model_type": "indoor_new",
        "threshold_confidence": 0.2,  ## Higher values means fewer but better matches.
    },
    "ECC_cv2": {  ## Accuracy: Okay. Speed: Medium.
        "mode_transform": "euclidean",  ## Must be one of {'translation', 'affine', 'euclidean', 'homography'}. See cv2 documentation on findTransformECC for more details.
        "n_iter": 200,
        "termination_eps": 1e-09,  ## Termination criteria for the registration algorithm. See documentation for more details.
        "gaussFiltSize": 1,  ## Size of the gaussian filter used to smooth the FOV_image before registration. Larger values mean more smoothing.
        "auto_fix_gaussFilt_step": 10,  ## If the registration fails, then the gaussian filter size is reduced by this amount and the registration is tried again.
    },
    "PhaseCorrelation": {  ## Accuracy: Poor. Speed: Very fast. Notes: Only applicable for translations, not rotations or scaling.
        "bandpass_freqs": [1, 30],
        "order": 5,
    },
    "NullRegistration": {},  ## No registration, no warping.
}


def filter_sessions_by_overlap(
    paths_stat: list[str],
    paths_ops: list[str],
    um_per_pixel: float | list[float] = 1.0,
    device: str = "cpu",
    z_threshold: float = 4.0,
    radius_in: float = 4.0,
    radius_out: float = 20.0,
    method: str = "RoMa",
    kwargs_method: dict | None = None,
) -> tuple[roicat.data_importing.Data_suite2p, list[int]]:
    """Load all sessions, run a silent geometric-alignment screening pass, and
    return a Data_suite2p containing only the largest co-registerable group.

    This replaces the manual ``keep = [...]`` pattern and the subsequent
    data-rebuild cell.  Call it once; the returned ``data`` object is ready to
    hand straight to ``aligner.augment_FOV_images``.

    Parameters
    ----------
    paths_stat, paths_ops:
        Parallel lists of stat.npy / ops.npy paths, one per session.
    um_per_pixel:
        Scalar applied to all sessions, or a per-session list.
    device:
        Torch device string (``'cpu'``, ``'cuda'``, ``'mps'``, …).  This is not
        a free choice under the default ``method='RoMa'``: RoMa is the slow,
        GPU-sensitive option, and the screen runs one registration per
        consecutive session pair (more if the match search falls back to
        all-pairs).  Measured on an M-series Mac at 512x512, ``'cpu'`` costs
        roughly 70 s per pair against roughly 29 s on ``'mps'``.  Pass
        ``roicat.helpers.set_device(use_GPU=True)``, which prefers cuda, then
        mps, then xpu.
    z_threshold:
        Alignment z-score threshold.  Session pairs whose score falls below
        this are treated as unaligned.
    radius_in, radius_out:
        Aligner inner / outer radii in micrometers.
    method:
        Geometric registration method, passed to ``fit_geometric``.  **Keep this
        equal to the method the notebook's real alignment run uses.** The screen
        exists to predict whether that run can co-register a set of sessions, so
        a screen using a stronger method green-lights sessions the run then
        aligns badly, and a weaker one drops sessions the run could have
        handled.  See ``KWARGS_ALIGN_METHOD`` for the options.
    kwargs_method:
        Per-method parameters.  Defaults to the module-level
        ``KWARGS_ALIGN_METHOD``; pass the notebook's own dict to guarantee the
        screen and the real fit are tuned identically.  Must contain ``method``
        as a key.

    Returns
    -------
    data : Data_suite2p
        Loaded with only the sessions in the largest co-registerable group.
    keep : list[int]
        Indices into the original path lists of the retained sessions.
    """
    n = len(paths_stat)
    umpp = (
        [um_per_pixel] * n if not isinstance(um_per_pixel, list) else list(um_per_pixel)
    )

    # Validated up front: loading every session takes minutes, and roicat
    # indexes kwargs_method[method] directly, so a typo would otherwise surface
    # as a bare KeyError from inside fit_geometric after all that work.
    kwargs_method = (
        KWARGS_ALIGN_METHOD if kwargs_method is None else dict(kwargs_method)
    )
    if method not in kwargs_method:
        raise ValueError(
            f"kwargs_method has no entry for method={method!r}; "
            f"got keys {sorted(kwargs_method)}"
        )

    # ── 1. Load all sessions ──────────────────────────────────────────────────
    data_all = roicat.data_importing.Data_suite2p(
        paths_statFiles=paths_stat,
        paths_opsFiles=paths_ops,
        um_per_pixel=umpp,
        new_or_old_suite2p="new",
        type_meanImg="meanImgE",
        verbose=False,
    )

    # ── 2. Silent geometric screening pass ───────────────────────────────────
    _aligner = roicat.tracking.alignment.Aligner(
        use_match_search=True,
        all_to_all=False,
        radius_in=radius_in,
        radius_out=radius_out,
        z_threshold=z_threshold,
        um_per_pixel=data_all.um_per_pixel[0],
        device=device,
        verbose=False,
    )
    fovs = _aligner.augment_FOV_images(
        FOV_images=data_all.FOV_images,
        spatialFootprints=data_all.spatialFootprints,
        normalize_FOV_intensities=True,
        roi_FOV_mixing_factor=0.5,
        use_CLAHE=True,
        CLAHE_grid_block_size=10,
        CLAHE_clipLimit=1.0,
        CLAHE_normalize=True,
    )
    _aligner.fit_geometric(
        template=0.5,
        ims_moving=fovs,
        template_method="sequential",
        mask_borders=(0, 0, 0, 0),
        method=method,
        kwargs_method=kwargs_method,
        constraint="affine",
        kwargs_RANSAC={"inl_thresh": 3.0, "max_iter": 100, "confidence": 0.99},
        verbose=False,
    )

    # ── 3. Find the largest co-registerable group ─────────────────────────────
    aligned = _aligner.results_geometric["final"]["alignment_all_to_all"].copy()
    np.fill_diagonal(aligned, True)
    aligned = (aligned | aligned.T).astype(bool)  # symmetrise

    _, group_ids = connected_components(aligned, directed=False)
    keep = sorted(np.where(group_ids == np.argmax(np.bincount(group_ids)))[0].tolist())
    dropped = [i for i in range(n) if i not in keep]

    # session_dir, not parts[-3]: under the suite2p/plane0 layout that index is
    # the literal string "suite2p", which tells you nothing about which session
    # was dropped — the only thing these lines exist to say.
    print(f"Session filter: {len(keep)}/{n} sessions kept")
    for i in keep:
        print(f"  keep  [{i}]  {session_dir(paths_stat[i]).name}")
    for i in dropped:
        print(f"  drop  [{i}]  {session_dir(paths_stat[i]).name}  (poor overlap)")

    # ── 4. Rebuild with only the kept sessions ────────────────────────────────
    if len(keep) == n:
        # Nothing was dropped — avoid an unnecessary reload
        return data_all, keep

    data = roicat.data_importing.Data_suite2p(
        paths_statFiles=[paths_stat[i] for i in keep],
        paths_opsFiles=[paths_ops[i] for i in keep],
        um_per_pixel=[umpp[i] for i in keep],
        new_or_old_suite2p="new",
        type_meanImg="meanImgE",
        verbose=False,
    )
    assert data.check_completeness(verbose=False)["tracking"]
    return data, keep
