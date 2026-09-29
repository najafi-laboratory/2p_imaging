"""
dataset/crop.py

Utilities for extracting fixed-size ROI-centered image patches from
Suite2p outputs.

UPDATED DESIGN:
- ROI IDs are Suite2p-native (NO +1 / -1 shifts)
- masks values are used directly as ground truth IDs
- removes indexing ambiguity
"""

from __future__ import annotations

import numpy as np


# -------------------------------------------------------------------------
# ROI mask
# -------------------------------------------------------------------------

def roi_binary_mask(
    masks: np.ndarray,
    roi_id: int,
) -> np.ndarray:
    """
    Return binary mask for one ROI.

    Parameters
    ----------
    masks : (H, W)
        Suite2p label image where each ROI has a unique integer ID.

    roi_id : int
        Suite2p-native ROI ID (NO offset).

    Returns
    -------
    bool array
    """
    return masks == roi_id


# -------------------------------------------------------------------------
# centroid
# -------------------------------------------------------------------------

def roi_centroid(
    masks: np.ndarray,
    roi_id: int,
) -> tuple[float, float]:
    """
    Compute centroid of one ROI.

    Returns
    -------
    (y, x)
    """

    binary = roi_binary_mask(masks, roi_id)

    if not np.any(binary):
        raise ValueError(f"ROI {roi_id} not found in mask.")

    y, x = np.nonzero(binary)

    return float(np.mean(y)), float(np.mean(x))


# -------------------------------------------------------------------------
# cropping
# -------------------------------------------------------------------------

def _crop_with_padding(
    image: np.ndarray,
    cy: float,
    cx: float,
    patch_size: int,
    pad_value: float = 0,
) -> np.ndarray:
    """
    Extract square crop centered at (cy, cx) with padding.
    """

    half = patch_size // 2

    cy = int(round(cy))
    cx = int(round(cx))

    y0 = cy - half
    y1 = y0 + patch_size

    x0 = cx - half
    x1 = x0 + patch_size

    out = np.full(
        (patch_size, patch_size),
        pad_value,
        dtype=image.dtype,
    )

    src_y0 = max(0, y0)
    src_y1 = min(image.shape[0], y1)

    src_x0 = max(0, x0)
    src_x1 = min(image.shape[1], x1)

    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[
        dst_y0:dst_y1,
        dst_x0:dst_x1,
    ] = image[
        src_y0:src_y1,
        src_x0:src_x1,
    ]

    return out


# -------------------------------------------------------------------------
# normalization
# -------------------------------------------------------------------------

def normalize_patch(
    image: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Z-score normalize image patch.
    """

    image = image.astype(np.float32)

    mean = image.mean()
    std = image.std()

    return (image - mean) / (std + eps)


# -------------------------------------------------------------------------
# main extraction
# -------------------------------------------------------------------------

def extract_roi_patch(
    mean_img: np.ndarray,
    masks: np.ndarray,
    roi_id: int,
    patch_size: int = 64,
    normalize: bool = True,
):
    """
    Extract:
        - image patch
        - ROI mask patch
    """

    cy, cx = roi_centroid(masks, roi_id)

    image_patch = _crop_with_padding(
        mean_img,
        cy,
        cx,
        patch_size,
        pad_value=float(mean_img.min()),
    )

    binary = roi_binary_mask(masks, roi_id)

    mask_patch = _crop_with_padding(
        binary.astype(np.uint8),
        cy,
        cx,
        patch_size,
        pad_value=0,
    )

    if normalize:
        image_patch = normalize_patch(image_patch)

    return (
        image_patch.astype(np.float32),
        mask_patch.astype(np.uint8),
    )


# -------------------------------------------------------------------------
# final model input
# -------------------------------------------------------------------------

def make_two_channel_patch(
    mean_img: np.ndarray,
    masks: np.ndarray,
    roi_id: int,
    patch_size: int = 64,
):
    """
    Returns:
        (2, H, W)
    """

    img, mask = extract_roi_patch(
        mean_img,
        masks,
        roi_id,
        patch_size=patch_size,
        normalize=True,
    )

    return np.stack(
        [
            img,
            mask.astype(np.float32),
        ],
        axis=0,
    )


# -------------------------------------------------------------------------
# self-test
# -------------------------------------------------------------------------

if __name__ == "__main__":

    print("crop.py self-test")

    H = 200
    W = 300

    mean = np.random.rand(H, W)

    masks = np.zeros((H, W), dtype=np.int32)

    masks[80:95, 100:115] = 1
    masks[150:170, 220:240] = 2

    patch = make_two_channel_patch(
        mean,
        masks,
        roi_id=1,   # Suite2p-native ID
        patch_size=64,
    )

    print("Patch shape:", patch.shape)
    print("Image dtype:", patch.dtype)