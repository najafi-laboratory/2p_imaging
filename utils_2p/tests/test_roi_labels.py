"""Tests for ROI reviewer label loading and export."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils_2p.roi_labels import (
    apply_label_export,
    load_reviewed_dff,
    map_qc_to_suite2p_rois,
    morphology_exclusion_reasons,
    roi_morphology_metrics,
    suite2p_stat_fingerprint,
)


def _stat_entry(pixels):
    ypix, xpix = zip(*pixels)
    return {"ypix": np.asarray(ypix), "xpix": np.asarray(xpix)}


class RoiLabelsTest(unittest.TestCase):
    def _write_processed_session(self, root):
        plane = root / "suite2p" / "plane0"
        plane.mkdir(parents=True)
        stat = np.asarray([_stat_entry([(i, i)]) for i in range(3)], dtype=object)
        np.save(plane / "stat.npy", stat)
        np.save(plane / "ops.npy", {"neucoeff": 0.0})
        traces = np.asarray(
            [
                [1.0, 2.0, 1.0, 2.0],
                [2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        )
        np.save(plane / "F.npy", traces)
        np.save(plane / "Fneu.npy", np.zeros_like(traces))
        np.save(plane / "iscell.npy", np.asarray([[1, 0.9], [0, 0.1], [1, 0.8]]))
        return stat

    def test_maps_filtered_qc_rois_to_original_suite2p_rows(self):
        suite2p_stat = np.asarray(
            [
                _stat_entry([(0, 0), (0, 1)]),
                _stat_entry([(4, 5)]),
                _stat_entry([(2, 2), (3, 2)]),
            ],
            dtype=object,
        )
        qc_stat = np.asarray([suite2p_stat[2], suite2p_stat[0]], dtype=object)
        np.testing.assert_array_equal(map_qc_to_suite2p_rois(qc_stat, suite2p_stat), [2, 0])

    def test_applies_json_labels_and_preserves_unreviewed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stat = np.asarray([_stat_entry([(i, i)]) for i in range(3)], dtype=object)
            np.save(root / "stat.npy", stat)
            original = np.asarray([[1.0, 0.8], [0.0, 0.2], [1.0, 0.9]])
            np.save(root / "iscell.npy", original)
            export = root / "labels.json"
            export.write_text(
                json.dumps(
                    {
                        "suite2p_roi_count": 3,
                        "suite2p_stat_fingerprint": suite2p_stat_fingerprint(stat),
                        "labels": [
                            {"summary_roi": 0, "suite2p_roi": 2, "label": 0},
                            {"summary_roi": 1, "suite2p_roi": 0, "label": 1},
                            {"summary_roi": 2, "suite2p_roi": 1, "label": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output = apply_label_export(export, root, backup=False)
            updated = np.load(root / "iscell_qc.npy")
            np.testing.assert_array_equal(updated[:, 0], [1, 0, 0])
            np.testing.assert_allclose(updated[:, 1], [1.0, 0.2, 0.0])
            np.testing.assert_array_equal(np.load(root / "iscell.npy"), original)
            self.assertEqual(output, (root / "iscell_qc.npy").resolve())

    def test_roi_morphology_metrics_match_qc_connectivity_semantics(self):
        stat = np.asarray(
            [
                {
                    **_stat_entry([(0, 0), (0, 1), (4, 4)]),
                    "skew": 1.2,
                    "aspect_ratio": 2.3,
                    "compact": 1.1,
                    "footprint": 1.5,
                }
            ],
            dtype=object,
        )
        metrics = roi_morphology_metrics(stat)
        self.assertEqual(metrics[0]["connect"], 2)
        self.assertEqual(metrics[0]["skew"], 1.2)

    def test_morphology_exclusion_reasons_lists_each_failed_condition(self):
        metrics = [{"skew": 3.0, "connect": 2, "aspect": 1.5, "compact": 0.5, "footprint": 1.5}]
        parameters = {
            "range_skew": [0.0, 2.0],
            "max_connect": 1,
            "range_aspect": [1.0, 2.0],
            "range_compact": [0.0, 1.0],
            "range_footprint": [1.0, 2.0],
        }
        reasons = morphology_exclusion_reasons(metrics, parameters)
        self.assertEqual(len(reasons[0]), 2)
        self.assertIn("skew", reasons[0][0])
        self.assertIn("connectivity", reasons[0][1])

    def test_load_reviewed_dff_uses_iscell_qc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_processed_session(root)
            plane = root / "suite2p" / "plane0"
            np.save(plane / "iscell_qc.npy", np.asarray([[0, 0.0], [1, 1.0], [1, 1.0]]))

            result = load_reviewed_dff(root, baseline_sigma=1.0)

            np.testing.assert_array_equal(result["roi_indices"], [1, 2])
            self.assertEqual(result["dff"].shape, (2, 4))
            self.assertEqual(result["label_path"].name, "iscell_qc.npy")

    def test_load_reviewed_dff_accepts_external_json_and_can_keep_unlabeled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stat = self._write_processed_session(root)
            export = root / "example_manual_roi_labels.json"
            export.write_text(
                json.dumps(
                    {
                        "suite2p_roi_count": 3,
                        "suite2p_stat_fingerprint": suite2p_stat_fingerprint(stat),
                        "labels": [
                            {"suite2p_roi": 0, "label": 0},
                            {"suite2p_roi": 1, "label": None},
                            {"suite2p_roi": 2, "label": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            strict = load_reviewed_dff(root, label_path=export, policy="good_only", baseline_sigma=1.0)
            permissive = load_reviewed_dff(root, label_path=export, policy="not_bad", baseline_sigma=1.0)

            np.testing.assert_array_equal(strict["roi_indices"], [2])
            np.testing.assert_array_equal(permissive["roi_indices"], [1, 2])
            self.assertEqual(strict["label_path"], export.resolve())

    def test_load_reviewed_dff_does_not_fall_back_to_original_iscell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_processed_session(root)

            with self.assertRaisesRegex(FileNotFoundError, "iscell_qc.npy"):
                load_reviewed_dff(root, baseline_sigma=1.0)


if __name__ == "__main__":
    unittest.main()
