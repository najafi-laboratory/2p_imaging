# `utils_2p` duplicate inventory

This repo contains many project directories that re-implement the same low-level helpers. This document records the most obvious shared utilities and the first extraction pass into `utils_2p`.

## Shared utility groups

- `utils_2p.stats`
  - `norm01`
  - `get_norm01_params`
  - `get_mean_sem`
  - Repeated in `passive_interval_oddball_202412/modeling/utils.py`, `passive_cerebellum/modeling/utils.py`, `joystick_cf_glm_202509/utils.py`, `single_interval_discrimination_202505/utils.py`, plus larger project `utils.py` files.
- `utils_2p.timing`
  - `get_frame_idx_from_time`
  - `get_sub_time_idx`
  - `get_trigger_time`
  - `correct_time_img_center`
  - Repeated across project `utils.py`, `Trialization.py`, `SpikeDeconv.py`, `PupilTime/ReadTime.py`, and `ebc_data_analysis/utils/functions.py`.
- `utils_2p.alignment`
  - `trim_seq`
  - `pad_seq`
  - `align_neu_seq_utils`
  - Repeated in `ebc_basic_202410/modules/Alignment.py`, `joystick_basic_202304/modules/Alignment.py`, `opto_pilot/modules/Alignment.py`, and `JoystickProcessing2026/PostProcessing/modules/Alignment.py`.
- `utils_2p.matlab`
  - `_check_keys`
  - `_todict`
  - `_tolist`
  - Generic loader `load_mat_struct`
  - Repeated inside multiple `ReadResults.py` modules while unpacking `bpod_session_data.mat`.
- `utils_2p.voltage`
  - `read_raw_voltages_basic`
  - `read_raw_voltages_memmap`
  - Derived from repeated raw-voltage readers in project `ReadResults.py` modules.

## Repos with clear duplication

- `2p_post_process_module_202404`
  - Mostly standalone post-processing logic; fewer generic utility duplicates than the behavior-analysis repos.
- `2p_processing_pipeline_202401`
  - Mostly config/script driven; not a primary duplication source.
- `ebc_basic_202410`
  - Duplicates alignment and trialization primitives.
- `ebc_data_analysis`
  - Duplicates timing and plotting helpers.
- `joystick_basic_202304`
  - Duplicates alignment, trialization, and reader helpers.
- `joystick_cf_glm_202509`
  - Duplicates stats/timing helpers in `utils.py` and trialization helpers.
- `JoystickProcessing2026`
  - Duplicates alignment/trialization and adds another postprocessing stack.
- `opto_pilot`
  - Duplicates alignment, timing, and `ReadResults` helpers.
- `passive_cerebellum`
  - Duplicates modeling `utils`, trialization helpers, and MAT readers.
- `passive_interval_oddball_202412`
  - Duplicates modeling `utils`, trialization helpers, and MAT readers.
- `PupilTime`
  - Duplicates timing helpers.
- `single_interval_discrimination_202505`
  - Duplicates stats/timing helpers in `utils.py` and trialization helpers.

## Initial adoption in this branch

- Centralized exact duplicates into `utils_2p`.
- Replaced several project-local duplicates with imports/wrappers where the signatures were stable.
- Left large plotting functions and project-coupled trial logic in place for a later pass.

## Deferred categories

- Plotting helpers such as `apply_colormap`, `plot_heatmap_neuron`, `get_roi_label_color`, and many figure-specific functions are duplicated too, but they carry more project-specific assumptions and are not included in this first extraction.
- `ReadResults.py` files still differ in memmap strategy, return shapes, and optional channels; only the common loader building blocks were centralized here.
