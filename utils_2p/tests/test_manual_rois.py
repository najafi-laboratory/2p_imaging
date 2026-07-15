"""Tests for Suite2p manual ROI cleanup utilities."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils_2p.manual_rois import (
    create_manual_roi_workspace,
    export_manual_roi_workspace,
    remove_all_manual_rois,
)


def _stat_entry(index, *, manual=0):
    entry = {
        "ypix": np.asarray([index, index + 1], dtype=np.int64),
        "xpix": np.asarray([index + 2, index + 3], dtype=np.int64),
    }
    if manual:
        entry["manual"] = manual
    return entry


class ManualRoisTest(unittest.TestCase):
    def test_remove_all_manual_rois_restores_row_aligned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            plane = Path(directory)
            stat_orig = np.asarray([_stat_entry(10), _stat_entry(20), _stat_entry(30)], dtype=object)
            stat = np.asarray([_stat_entry(1, manual=1), *stat_orig], dtype=object)

            np.save(plane / "stat_orig.npy", stat_orig)
            np.save(plane / "stat.npy", stat)
            np.save(plane / "iscell.npy", np.asarray([[1, 1.0], [1, 0.9], [0, 0.2], [1, 0.8]]))
            np.save(plane / "F.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
            np.save(plane / "Fneu.npy", np.arange(100, 116, dtype=np.float32).reshape(4, 4))
            np.save(plane / "spks.npy", np.arange(200, 216, dtype=np.float32).reshape(4, 4))

            result = remove_all_manual_rois(plane / "stat_orig.npy", plane / "stat.npy", backup=False)

            self.assertEqual(result["removed_indices"], [0])
            self.assertEqual(np.load(plane / "stat.npy", allow_pickle=True).shape, (3,))
            np.testing.assert_array_equal(np.load(plane / "iscell.npy"), [[1, 0.9], [0, 0.2], [1, 0.8]])
            np.testing.assert_array_equal(np.load(plane / "F.npy"), np.arange(16, dtype=np.float32).reshape(4, 4)[1:])
            np.testing.assert_array_equal(np.load(plane / "Fneu.npy"), np.arange(100, 116, dtype=np.float32).reshape(4, 4)[1:])
            np.testing.assert_array_equal(np.load(plane / "spks.npy"), np.arange(200, 216, dtype=np.float32).reshape(4, 4)[1:])

    def test_remove_all_manual_rois_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            plane = Path(directory)
            stat_orig = np.asarray([_stat_entry(10)], dtype=object)
            stat = np.asarray([_stat_entry(1, manual=1), stat_orig[0]], dtype=object)
            np.save(plane / "stat_orig.npy", stat_orig)
            np.save(plane / "stat.npy", stat)
            np.save(plane / "F.npy", np.asarray([[1, 2], [3, 4]], dtype=np.float32))

            result = remove_all_manual_rois(plane / "stat_orig.npy", plane / "stat.npy", dry_run=True)

            self.assertEqual(result["removed_indices"], [0])
            self.assertEqual(result["changed_files"], [])
            self.assertEqual(np.load(plane / "stat.npy", allow_pickle=True).shape, (2,))
            np.testing.assert_array_equal(np.load(plane / "F.npy"), [[1, 2], [3, 4]])

    def test_remove_all_manual_rois_refuses_misaligned_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            plane = Path(directory)
            stat_orig = np.asarray([_stat_entry(10)], dtype=object)
            stat = np.asarray([_stat_entry(1, manual=1), stat_orig[0]], dtype=object)
            np.save(plane / "stat_orig.npy", stat_orig)
            np.save(plane / "stat.npy", stat)
            np.save(plane / "F.npy", np.asarray([[1, 2]], dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "expected 2"):
                remove_all_manual_rois(plane / "stat_orig.npy", plane / "stat.npy", backup=False)

    def test_create_manual_roi_workspace_aliases_qc_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qc = root / "qc_results"
            plane = root / "suite2p" / "plane0"
            workspace = root / "manual_roi_workspace"
            qc.mkdir()
            plane.mkdir(parents=True)

            stat = np.asarray([_stat_entry(10), _stat_entry(20)], dtype=object)
            np.save(qc / "stat.npy", stat)
            np.save(qc / "fluo.npy", np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
            np.save(qc / "neuropil.npy", np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.float32))
            np.save(plane / "ops.npy", {"Ly": 512, "Lx": 512})
            (plane / "data.bin").write_bytes(b"binary")

            result = create_manual_roi_workspace(qc, plane, workspace)

            self.assertEqual(result["iscell_source"], "generated_all_cells")
            self.assertEqual(result["spks_source"], "generated_zeros")
            np.testing.assert_array_equal(np.load(workspace / "F.npy"), np.load(qc / "fluo.npy"))
            np.testing.assert_array_equal(np.load(workspace / "Fneu.npy"), np.load(qc / "neuropil.npy"))
            np.testing.assert_array_equal(np.load(workspace / "spks.npy"), np.zeros((2, 3), dtype=np.float32))
            np.testing.assert_array_equal(np.load(workspace / "iscell.npy"), np.ones((2, 2)))
            self.assertTrue((workspace / "data.bin").is_symlink())

    @unittest.skipIf(importlib.util.find_spec("scipy") is None, "derived regeneration requires scipy")
    def test_export_manual_roi_workspace_writes_qc_aliases_and_stale_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qc = root / "qc_results"
            workspace = root / "manual_roi_workspace"
            qc.mkdir()
            workspace.mkdir()

            stat = np.asarray([_stat_entry(1, manual=1), _stat_entry(10)], dtype=object)
            np.save(workspace / "stat.npy", stat)
            np.save(workspace / "F.npy", np.asarray([[1, 2], [3, 4]], dtype=np.float32))
            np.save(workspace / "Fneu.npy", np.asarray([[5, 6], [7, 8]], dtype=np.float32))
            (workspace / "manual_roi_workspace_source.txt").write_text("test workspace\n", encoding="ascii")

            np.save(qc / "stat.npy", np.asarray([_stat_entry(10)], dtype=object))
            np.save(qc / "fluo.npy", np.asarray([[3, 4]], dtype=np.float32))
            np.save(qc / "neuropil.npy", np.asarray([[7, 8]], dtype=np.float32))
            np.save(qc / "masks.npy", np.zeros((32, 32), dtype=float))
            (qc / "dff.h5").write_bytes(b"stale")
            (qc / "events.h5").write_bytes(b"do-not-touch")
            (qc / "onsets.h5").write_bytes(b"do-not-touch")

            result = export_manual_roi_workspace(workspace, qc, backup=False)

            self.assertEqual(np.load(qc / "stat.npy", allow_pickle=True).shape, (2,))
            np.testing.assert_array_equal(np.load(qc / "fluo.npy"), [[1, 2], [3, 4]])
            np.testing.assert_array_equal(np.load(qc / "neuropil.npy"), [[5, 6], [7, 8]])
            masks = np.load(qc / "masks.npy")
            self.assertEqual(masks[1, 3], 1)
            self.assertEqual(masks[2, 4], 1)
            self.assertEqual(masks[10, 12], 2)
            self.assertEqual(masks[11, 13], 2)
            self.assertNotIn(str(qc / "masks.npy"), result["stale_files"])
            self.assertIn(str(qc / "dff.h5"), result["regenerated_derived"])
            self.assertEqual(result["stale_files"], [])
            self.assertIsNone(result["stale_manifest"])
            self.assertEqual((qc / "events.h5").read_bytes(), b"do-not-touch")
            self.assertEqual((qc / "onsets.h5").read_bytes(), b"do-not-touch")

    def test_export_manual_roi_workspace_can_remove_workspace_after_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qc = root / "qc_results"
            workspace = root / "manual_roi_workspace"
            qc.mkdir()
            workspace.mkdir()

            stat = np.asarray([_stat_entry(10)], dtype=object)
            np.save(workspace / "stat.npy", stat)
            np.save(workspace / "F.npy", np.asarray([[1, 2]], dtype=np.float32))
            np.save(workspace / "Fneu.npy", np.asarray([[3, 4]], dtype=np.float32))
            (workspace / "manual_roi_workspace_source.txt").write_text("test workspace\n", encoding="ascii")
            np.save(qc / "stat.npy", stat)
            np.save(qc / "fluo.npy", np.asarray([[0, 0]], dtype=np.float32))
            np.save(qc / "neuropil.npy", np.asarray([[0, 0]], dtype=np.float32))
            np.save(qc / "masks.npy", np.zeros((32, 32), dtype=float))

            result = export_manual_roi_workspace(workspace, qc, backup=False, cleanup_workspace=True)

            self.assertTrue(result["workspace_removed"])
            self.assertFalse(workspace.exists())
            np.testing.assert_array_equal(np.load(qc / "fluo.npy"), [[1, 2]])


if __name__ == "__main__":
    unittest.main()
