import os
import tempfile
import unittest
from pathlib import Path

from utils_2p.processing_pipeline import (
    PipelineConfig,
    SUITE2P_VERSIONED_PYTHON_CANDIDATES,
    _current_python_bin,
    _normalize_stages,
    _suite2p_python_path,
)


class Suite2pEnvSelectionTest(unittest.TestCase):
    def test_versioned_suite2p_python_paths_are_distinct(self):
        self.assertIn("0.x", SUITE2P_VERSIONED_PYTHON_CANDIDATES)
        self.assertIn("1.x", SUITE2P_VERSIONED_PYTHON_CANDIDATES)
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
            expected = _suite2p_python_path("1.x")
            if not expected.exists():
                expected = _current_python_bin()
            self.assertEqual(normalized.python_bin, expected)
        finally:
            if old_python is not None:
                os.environ["TWO_P_PYTHON"] = old_python


class PipelineStageSelectionTest(unittest.TestCase):
    def test_default_stages_skip_optional_and_legacy_qc(self):
        stages = _normalize_stages(
            None,
            run_label=False,
            run_oasis=False,
            run_roi_model_scores=False,
        )
        self.assertEqual(stages, ("prep", "suite2p", "dff", "summary"))

    def test_optional_stages_are_explicit(self):
        stages = _normalize_stages(
            None,
            run_label=True,
            run_oasis=True,
            run_roi_model_scores=True,
        )
        self.assertEqual(
            stages,
            ("prep", "suite2p", "roi_model_scores", "label", "dff", "spikes", "summary"),
        )

    def test_qc_stage_is_not_supported(self):
        with self.assertRaisesRegex(ValueError, "Unknown stages"):
            _normalize_stages(
                "prep,suite2p,qc,dff,summary",
                run_label=False,
                run_oasis=False,
                run_roi_model_scores=False,
            )


if __name__ == "__main__":
    unittest.main()
