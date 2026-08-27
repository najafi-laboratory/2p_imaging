import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from utils_2p import dff_traces


class DffTraceInputTest(unittest.TestCase):
    def test_run_prefers_native_suite2p_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plane = root / "suite2p" / "plane0"
            plane.mkdir(parents=True)
            np.save(plane / "F.npy", np.asarray([[10.0, 12.0, 14.0]], dtype=np.float32))
            np.save(plane / "Fneu.npy", np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32))

            qc = root / "qc_results"
            qc.mkdir()
            np.save(qc / "fluo.npy", np.asarray([[100.0, 100.0, 100.0]], dtype=np.float32))
            np.save(qc / "neuropil.npy", np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))

            dff_traces.run({"save_path0": str(root), "neucoeff": 0.5}, normalize=False)

            with h5py.File(root / "dff.h5", "r") as h5:
                np.testing.assert_array_equal(h5["fluo"][:], [[10.0, 12.0, 14.0]])

    def test_run_accepts_legacy_qc_results_without_suite2p_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qc = root / "qc_results"
            qc.mkdir()
            np.save(qc / "fluo.npy", np.asarray([[10.0, 12.0, 14.0]], dtype=np.float32))
            np.save(qc / "neuropil.npy", np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32))

            dff_traces.run({"save_path0": str(root), "neucoeff": 0.5}, normalize=False)

            with h5py.File(root / "dff.h5", "r") as h5:
                np.testing.assert_array_equal(h5["fluo"][:], [[10.0, 12.0, 14.0]])


if __name__ == "__main__":
    unittest.main()
