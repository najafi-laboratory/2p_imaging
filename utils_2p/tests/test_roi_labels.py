"""Tests for ROI reviewer label loading and export."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils_2p.roi_labels import (
    autocorrelation_decay_tau,
    apply_label_export,
    dff_qc_metrics,
    load_reviewed_dff,
    map_qc_to_suite2p_rois,
    morphology_exclusion_reasons,
    roi_morphology_metrics,
    robust_event_snr,
    temporal_smoothness_snr,
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

    def test_applies_json_labels_to_manual_label_masks(self):
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
                            {"summary_roi": 0, "suite2p_roi": 2, "label": 0, "morphology_pass": False},
                            {"summary_roi": 1, "suite2p_roi": 0, "label": 1, "morphology_pass": True},
                            {"summary_roi": 2, "suite2p_roi": 1, "label": 2, "morphology_pass": True},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output = apply_label_export(export, root, backup=False)
            updated = np.load(root / "roi_manual_labels.npy")
            np.testing.assert_array_equal(updated[:, 0], [1, 0, 0])
            np.testing.assert_allclose(updated[:, 1], [1, 0, np.nan], equal_nan=True)
            np.testing.assert_allclose(updated[:, 2], [1, 1, np.nan], equal_nan=True)
            np.testing.assert_array_equal(np.load(root / "iscell.npy"), original)
            self.assertEqual(output, (root / "roi_manual_labels.npy").resolve())

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

    def test_load_reviewed_dff_uses_roi_manual_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_processed_session(root)
            plane = root / "suite2p" / "plane0"
            np.save(
                plane / "roi_manual_labels.npy",
                np.asarray(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [1, np.nan, np.nan],
                    ],
                    dtype=float,
                ),
            )

            result = load_reviewed_dff(root, baseline_sigma=1.0)
            permissive = load_reviewed_dff(root, policy="full_suite2p_good", baseline_sigma=1.0)

            np.testing.assert_array_equal(result["roi_indices"], [1])
            np.testing.assert_array_equal(permissive["roi_indices"], [1, 2])
            self.assertEqual(result["dff"].shape, (1, 4))
            self.assertEqual(result["label_path"].name, "roi_manual_labels.npy")

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
                            {"suite2p_roi": 2, "label": 2},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            strict = load_reviewed_dff(root, label_path=export, policy="good_only", baseline_sigma=1.0)
            permissive = load_reviewed_dff(root, label_path=export, policy="good_or_unsure", baseline_sigma=1.0)

            np.testing.assert_array_equal(strict["roi_indices"], [])
            np.testing.assert_array_equal(permissive["roi_indices"], [2])
            self.assertEqual(strict["label_path"], export.resolve())

    def test_load_reviewed_dff_does_not_fall_back_to_original_iscell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_processed_session(root)

            with self.assertRaisesRegex(FileNotFoundError, "roi_manual_labels.npy"):
                load_reviewed_dff(root, baseline_sigma=1.0)

    def test_dff_qc_metrics_capture_event_temporal_and_decay_metrics(self):
        trace = np.exp(-np.arange(20, dtype=float) / 4.0)
        event = robust_event_snr(trace, sigma=1.0, event_percentile=75.0, dilation=0)
        temporal = temporal_smoothness_snr(trace)
        decay = autocorrelation_decay_tau(trace, frame_rate=2.0, max_lag_seconds=5.0)
        metrics = dff_qc_metrics(np.asarray([trace], dtype=np.float32), frame_rate=2.0)

        self.assertTrue(np.isfinite(event["event_snr"]))
        self.assertGreater(event["event_snr"], 0.0)
        self.assertTrue(np.isfinite(temporal))
        self.assertTrue(np.isfinite(decay))
        self.assertGreater(decay, 0.0)
        self.assertEqual(len(metrics), 1)
        self.assertIn("event_snr", metrics[0])
        self.assertIn("temporal_snr", metrics[0])
        self.assertIn("decay_tau_seconds", metrics[0])
        self.assertTrue(np.isfinite(metrics[0]["temporal_snr"]))
        self.assertTrue(np.isfinite(metrics[0]["decay_tau_seconds"]))


if __name__ == "__main__":
    unittest.main()
