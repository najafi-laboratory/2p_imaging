#!/usr/bin/env python3
"""Preferred entry point for staged two-photon processing jobs."""

from __future__ import annotations

from utils_2p.preprocessing_qc_pipeline import GeneratedRun
from utils_2p.preprocessing_qc_pipeline import PipelineConfig
from utils_2p.preprocessing_qc_pipeline import SessionSpec
from utils_2p.preprocessing_qc_pipeline import SlurmResources
from utils_2p.preprocessing_qc_pipeline import SUITE2P_VERSIONED_PYTHON_CANDIDATES
from utils_2p.preprocessing_qc_pipeline import _current_python_bin
from utils_2p.preprocessing_qc_pipeline import _normalize_stages
from utils_2p.preprocessing_qc_pipeline import _suite2p_python_path
from utils_2p.preprocessing_qc_pipeline import generate_processing_jobs
from utils_2p.preprocessing_qc_pipeline import generate_preprocessing_qc_jobs
from utils_2p.preprocessing_qc_pipeline import main
from utils_2p.preprocessing_qc_pipeline import run_stage
from utils_2p.preprocessing_qc_pipeline import submit_processing_jobs
from utils_2p.preprocessing_qc_pipeline import submit_preprocessing_qc_jobs


if __name__ == "__main__":
    main()
