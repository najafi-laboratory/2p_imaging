"""
results_table.py
================
Tabular exports of ROICaT tracking results.

The pipeline saves labels as nested lists (``labels_bySession``) and quality
metrics as separate arrays with their own indexing conventions.  This module
flattens all of that into two pandas tables:

  build_roi_table(...)     long format — one row per tracked ROI
  build_match_matrix(...)  wide format — one row per UCID, one column per session

``export_tables`` writes both to CSV alongside the other tracking outputs.

Note on cs_sil indexing
-----------------------
``quality_metrics['cluster_silhouette']`` is aligned with
``quality_metrics['cluster_labels_unique']``, which starts at **-1** (the
unclustered label).  Indexing it directly by UCID is therefore off by one.
Use :func:`cs_sil_by_ucid` to get a properly UCID-indexed array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# --- quality-metric indexing ---


def cs_sil_by_ucid(quality_metrics) -> np.ndarray:
    """Return cluster silhouette scores indexed by UCID.

    ``quality_metrics['cluster_silhouette'][i]`` is the score for cluster
    ``quality_metrics['cluster_labels_unique'][i]``, and that label array
    starts at -1.  This re-indexes so that ``out[u]`` is the score for UCID
    ``u``, with NaN for any UCID that has no score.

    Parameters
    ----------
    quality_metrics : dict
        ``results_all['clusters']['quality_metrics']`` (or the equivalent
        from ``results_clusters.json`` / ``clusterer.quality_metrics``).

    Returns
    -------
    (n_ucid,) float array — NaN where unavailable.
    """
    scores = np.asarray(quality_metrics["cluster_silhouette"], dtype=float)
    labels = np.asarray(quality_metrics["cluster_labels_unique"])

    valid = labels >= 0
    labels_v = labels[valid].astype(int)
    out = np.full(labels_v.max() + 1 if len(labels_v) else 0, np.nan)
    out[labels_v] = scores[valid]
    return out


def _optional_per_roi(quality_metrics, key, n_roi) -> np.ndarray:
    """Per-ROI metric as a float array, or all-NaN when the run didn't make it.

    ``sample_probabilities`` is None for sequential-Hungarian runs (it is an
    HDBSCAN-only output), so this degrades to NaN rather than raising.
    """
    if quality_metrics is None or key not in quality_metrics:
        return np.full(n_roi, np.nan)
    arr = np.asarray(quality_metrics[key], dtype=object)
    if arr.ndim == 0 or arr.size != n_roi:
        return np.full(n_roi, np.nan)
    return arr.astype(float)


# --- session naming ---


def session_labels_from_paths(paths_stat) -> tuple[list[str], list[str]]:
    """Derive (session_name, date) from suite2p stat.npy paths.

    Assumes the lab layout ``.../<session_folder>/suite2p/plane0/stat.npy``,
    where the session folder looks like ``SA11_20250806``.  Falls back to the
    folder name itself when it doesn't split on '_'.
    """
    names, dates = [], []
    for p in paths_stat:
        name = Path(p).parts[-3]
        names.append(name)
        dates.append(name.split("_")[-1] if "_" in name else name)
    return names, dates


# --- centroids ---


def _centroids_for_session(rois, H, W) -> np.ndarray:
    """Intensity-weighted (row, col) centroid of every ROI in one session.

    Parameters
    ----------
    rois : (n_roi, H*W) sparse matrix, or dense (n_roi, H*W) / (n_roi, H, W)
    Returns
    -------
    (n_roi, 2) float array of (row, col).
    """
    if sp.issparse(rois):
        csr = rois.tocsr()
        n = csr.shape[0]
        out = np.full((n, 2), np.nan)
        for i in range(n):
            lo, hi = csr.indptr[i], csr.indptr[i + 1]
            w = csr.data[lo:hi].astype(float)
            total = w.sum()
            if total <= 0:
                continue
            flat = csr.indices[lo:hi]
            out[i] = ((flat // W) * w).sum() / total, ((flat % W) * w).sum() / total
        return out

    dense = np.asarray(rois, dtype=float)
    if dense.ndim == 2:
        dense = dense.reshape(dense.shape[0], H, W)
    total = dense.sum(axis=(1, 2))
    rr, cc = np.indices((H, W))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.stack(
            [
                (dense * rr).sum(axis=(1, 2)) / total,
                (dense * cc).sum(axis=(1, 2)) / total,
            ],
            axis=1,
        )
    out[total <= 0] = np.nan
    return out


# --- long table ---


def build_roi_table(
    labels_bySession,
    quality_metrics=None,
    paths_stat=None,
    session_names=None,
    stim_types=None,
    rois_aligned=None,
    H=None,
    W=None,
    include_unclustered=False,
) -> pd.DataFrame:
    """One row per ROI, with its UCID and quality scores.

    Parameters
    ----------
    labels_bySession : list of arrays
        ``results_all['clusters']['labels_bySession']``.  Entry ``s[i]`` is the
        UCID of ROI ``i`` in session ``s``, or -1 if unclustered.
    quality_metrics : dict or None
        ``results_all['clusters']['quality_metrics']``.  When None, the score
        columns are omitted.
    paths_stat : list of str or None
        ``results_all['input_data']['paths_stat']``, used to derive session
        names and dates when ``session_names`` isn't given.
    session_names, stim_types : list of str or None
        One entry per session, in filtered (post-overlap-screen) order.
    rois_aligned, H, W :
        Aligned footprints and frame shape.  When supplied, adds
        ``centroid_y`` / ``centroid_x`` in aligned coordinates.
    include_unclustered : bool
        Keep ROIs with UCID -1.  Default False.

    Returns
    -------
    DataFrame sorted by (ucid, session_idx), with columns:
        ucid, session_idx, session_name, date, stim_type,
        roi_idx, roi_idx_global, n_sessions_present, n_rois_in_cluster,
        cs_sil, sample_sil, sample_prob, centroid_y, centroid_x
    (session_name/date/stim_type/centroids appear only when derivable.)
    """
    labels_bySession = [np.asarray(x).astype(int) for x in labels_bySession]
    n_sessions = len(labels_bySession)
    n_roi_bySession = [len(x) for x in labels_bySession]
    n_roi_total = int(sum(n_roi_bySession))

    if session_names is None and paths_stat is not None:
        session_names, dates = session_labels_from_paths(paths_stat)
    else:
        dates = None
    if session_names is None:
        session_names = [f"S{s}" for s in range(n_sessions)]

    # Global ROI index = position in the session-concatenated label vector,
    # which is the indexing that sample_silhouette / sample_probabilities use.
    offsets = np.concatenate([[0], np.cumsum(n_roi_bySession)]).astype(int)

    sample_sil = _optional_per_roi(quality_metrics, "sample_silhouette", n_roi_total)
    sample_prob = _optional_per_roi(
        quality_metrics, "sample_probabilities", n_roi_total
    )
    cs_sil = cs_sil_by_ucid(quality_metrics) if quality_metrics is not None else None

    centroids = None
    if rois_aligned is not None and H is not None and W is not None:
        centroids = [
            _centroids_for_session(rois_aligned[s], H, W) for s in range(n_sessions)
        ]

    rows = []
    for s in range(n_sessions):
        labels_s = labels_bySession[s]
        for i, u in enumerate(labels_s):
            if u < 0 and not include_unclustered:
                continue
            g = offsets[s] + i
            row = {
                "ucid": int(u),
                "session_idx": s,
                "session_name": session_names[s],
                "roi_idx": i,
                "roi_idx_global": g,
                "sample_sil": sample_sil[g],
                "sample_prob": sample_prob[g],
            }
            if dates is not None:
                row["date"] = dates[s]
            if stim_types is not None:
                row["stim_type"] = stim_types[s]
            if cs_sil is not None:
                row["cs_sil"] = cs_sil[u] if 0 <= u < len(cs_sil) else np.nan
            if centroids is not None:
                row["centroid_y"], row["centroid_x"] = centroids[s][i]
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Cluster-level counts.  n_rois_in_cluster > n_sessions_present means a
    # session contributed two ROIs to one cluster — worth inspecting.
    grp = df.groupby("ucid")
    df["n_sessions_present"] = grp["session_idx"].transform("nunique")
    df["n_rois_in_cluster"] = grp["ucid"].transform("size")

    order = [
        "ucid",
        "session_idx",
        "session_name",
        "date",
        "stim_type",
        "roi_idx",
        "roi_idx_global",
        "n_sessions_present",
        "n_rois_in_cluster",
        "cs_sil",
        "sample_sil",
        "sample_prob",
        "centroid_y",
        "centroid_x",
    ]
    df = df[[c for c in order if c in df.columns]]
    return df.sort_values(["ucid", "session_idx"]).reset_index(drop=True)


# --- wide matrix ---


def build_match_matrix(df, value="roi_idx", session_key="session_name") -> pd.DataFrame:
    """Pivot the long table to one row per UCID, one column per session.

    Cells hold the ROI index for that UCID in that session, NaN where the
    neuron wasn't detected.  This is the table to join dF/F traces against:
    row = neuron, column = session, value = which suite2p ROI to pull.

    Duplicate (ucid, session) pairs — two ROIs from one session in the same
    cluster — are joined into a comma-separated string so nothing is silently
    dropped by the pivot.
    """
    if df.empty:
        return pd.DataFrame()

    def _join(x):
        vals = list(x)
        return vals[0] if len(vals) == 1 else ", ".join(str(v) for v in vals)

    wide = df.pivot_table(
        index="ucid", columns=session_key, values=value, aggfunc=_join
    )
    # Restore acquisition order; pivot_table sorts columns alphabetically.
    session_order = df.drop_duplicates("session_idx").sort_values("session_idx")
    wide = wide.reindex(columns=list(session_order[session_key]))
    wide.columns.name = None

    meta = df.drop_duplicates("ucid").set_index("ucid")
    for col in ("n_sessions_present", "n_rois_in_cluster", "cs_sil"):
        if col in meta.columns:
            wide.insert(0, col, meta[col])
    return wide.sort_index()


# --- export ---


def export_tables(df, path_prefix, matrix=True) -> dict[str, str]:
    """Write the long table (and optionally the wide matrix) to CSV.

    Returns a dict of {kind: path} for the files written.
    """
    written = {}
    p_long = f"{path_prefix}.roi_table.csv"
    df.to_csv(p_long, index=False)
    written["roi_table"] = p_long

    if matrix:
        p_wide = f"{path_prefix}.match_matrix.csv"
        build_match_matrix(df).to_csv(p_wide)
        written["match_matrix"] = p_wide

    for kind, p in written.items():
        print(f"wrote {kind} -> {p}")
    return written
