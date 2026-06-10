import os
import tempfile
import unittest
from pathlib import Path

from utils_2p.preprocessing_qc_pipeline import (
    PipelineConfig,
    SUITE2P_VERSIONED_PYTHONS,
    _suite2p_python_path,
)


class Suite2pEnvSelectionTest(unittest.TestCase):
    def test_versioned_suite2p_python_paths_are_distinct(self):
        self.assertIn("0.x", SUITE2P_VERSIONED_PYTHONS)
        self.assertIn("1.x", SUITE2P_VERSIONED_PYTHONS)
        self.assertNotEqual(_suite2p_python_path("0.x"), _suite2p_python_path("1.x"))

    def test_unknown_suite2p_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "suite2p_version"):
            _suite2p_python_path("2.x")

    def test_pipeline_config_defaults_to_1x_alias_when_python_is_unset(self):
        old_python = os.environ.pop("TWO_P_PYTHON", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                normalized = PipelineConfig().normalized(Path(tmp))
            self.assertEqual(normalized.suite2p_version, "1.x")
            self.assertEqual(normalized.python_bin, _suite2p_python_path("1.x"))
        finally:
            if old_python is not None:
                os.environ["TWO_P_PYTHON"] = old_python


if __name__ == "__main__":
    unittest.main()
