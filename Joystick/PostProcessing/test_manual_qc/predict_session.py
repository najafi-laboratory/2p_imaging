# python predict_session.py /storage/project/r-fnajafi3-0/shared/2P_Imaging/SA18_LG/SA18_20260115

"""
predict_session.py

Runs the trained ROI QC model on a new Suite2p session and saves
its predictions for later review. It updates ROI_label.h5 in place.

# Output:
#     roi_predictions.h5

Contents:
    roi
    probability
    state
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from prediction_utils import predict_session
from constants import GOOD_THRESHOLD, BAD_THRESHOLD


# With Good = 0.9, Bad = 0.1:
# Good : 327
# Bad  : 351
# Gray : 723

# With Good = 0.85, Bad = 0.15:
# Good : 367
# Bad  : 462
# Gray : 572

MODEL_PATH = "./checkpoints/best_model.pt"
PREDICTION_DIR = "./predictions"
PATCH_SIZE = 64


def update_roi_labels(session_path, predictions):
    """
    Update ROI_label.h5 using high-confidence predictions.

    If ROI_label.h5 does not exist, it is created.

    Existing labels are never overwritten.
    Gray predictions are ignored.
    """

    label_path = os.path.join(
        session_path,
        "ROI_label.h5"
    )

    # Load existing labels if present
    if os.path.exists(label_path):

        with h5py.File(label_path, "r") as f:

            good_roi = set(
                f["good_roi"][:].astype(int)
            )

            bad_roi = set(
                f["bad_roi"][:].astype(int)
            )

    else:

        good_roi = set()
        bad_roi = set()

    # Add automatic labels
    for pred in predictions:

        roi = pred["roi"] - 1

        # Never overwrite existing labels
        if roi in good_roi or roi in bad_roi:
            continue

        if pred["state"] == "good":
            good_roi.add(roi)

        elif pred["state"] == "bad":
            bad_roi.add(roi)

        # gray -> ignore

    # Write file (creates if missing)
    with h5py.File(label_path, "w") as f:

        f.create_dataset(
            "good_roi",
            data=np.array(
                sorted(good_roi),
                dtype=np.int32
            )
        )

        f.create_dataset(
            "bad_roi",
            data=np.array(
                sorted(bad_roi),
                dtype=np.int32
            )
        )

    print(f"Updated {label_path}")


def save_predictions(session_path, predictions, output_dir):
    """
    Save CNN predictions for later review.

    Saves parallel arrays:
        roi[i]
        probability[i]
        state[i]
    """

    roi_ids = []
    probabilities = []
    states = []

    for pred in predictions:

        roi = pred["roi"] - 1   # convert to ROI_label.h5 indexing

        roi_ids.append(roi)
        probabilities.append(pred["probability"])
        states.append(pred["state"])


    session_name = os.path.basename(
        os.path.normpath(session_path)
    )

    session_output_dir = os.path.join(
        output_dir,
        session_name
    )

    os.makedirs(
        session_output_dir,
        exist_ok=True
    )

    out_path = os.path.join(
        session_output_dir,
        "roi_predictions.h5"
    )


    with h5py.File(out_path, "w") as f:

        f.create_dataset(
            "roi",
            data=np.array(roi_ids, dtype=np.int32),
        )

        f.create_dataset(
            "probability",
            data=np.array(probabilities, dtype=np.float32),
        )

        # store strings properly
        dt = h5py.string_dtype(
            encoding="utf-8"
        )

        f.create_dataset(
            "state",
            data=np.array(states, dtype=dt),
        )


    print(f"\nSaved predictions to:\n{out_path}")

    print(
        f"Good : {states.count('good')}"
    )
    print(
        f"Bad  : {states.count('bad')}"
    )
    print(
        f"Gray : {states.count('gray')}"
    )

def apply_predictions_to_labels(session_path, predictions):
    """
    Add high-confidence predictions to ROI_label.h5.

    Existing manual labels are preserved.
    Gray predictions are ignored.
    """

    label_path = os.path.join(
        session_path,
        "ROI_label.h5"
    )

    with h5py.File(label_path, "r") as f:

        good_roi = set(f["good_roi"][:].astype(int))
        bad_roi = set(f["bad_roi"][:].astype(int))

    for pred in predictions:

        roi = pred["roi"] - 1      # same indexing used elsewhere

        # Already manually labeled
        if roi in good_roi or roi in bad_roi:
            continue

        if pred["state"] == "good":
            good_roi.add(roi)

        elif pred["state"] == "bad":
            bad_roi.add(roi)

        # gray -> do nothing

    with h5py.File(label_path, "w") as f:

        f.create_dataset(
            "good_roi",
            data=np.array(sorted(good_roi), dtype=np.int32)
        )

        f.create_dataset(
            "bad_roi",
            data=np.array(sorted(bad_roi), dtype=np.int32)
        )

    print("\nUpdated ROI_label.h5 with automatic labels.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "session",
        help="Path to Suite2p session",
    )

    args = parser.parse_args()

    print("going to predict session")

    predictions = predict_session(
        session_path=args.session,
        model_path=MODEL_PATH,
        patch_size=PATCH_SIZE,
        good_threshold=GOOD_THRESHOLD,
        bad_threshold=BAD_THRESHOLD,
    )

    print("saving predictions")

    save_predictions(
        args.session,
        predictions,
        PREDICTION_DIR
    )

    update_roi_labels(
        args.session,
        predictions
    )

    apply_predictions_to_labels(
        args.session,
        predictions
    )


if __name__ == "__main__":
    main()