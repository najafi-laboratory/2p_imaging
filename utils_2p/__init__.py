_EXPORTS = {
    "align_neu_seq_utils": ("utils_2p.alignment", "align_neu_seq_utils"),
    "correct_time_img_center": ("utils_2p.timing", "correct_time_img_center"),
    "extract_bpod_session_data": ("utils_2p.matlab", "extract_bpod_session_data"),
    "get_frame_idx_from_time": ("utils_2p.timing", "get_frame_idx_from_time"),
    "get_mean_sem": ("utils_2p.stats", "get_mean_sem"),
    "get_norm01_params": ("utils_2p.stats", "get_norm01_params"),
    "get_sub_time_idx": ("utils_2p.timing", "get_sub_time_idx"),
    "get_trigger_time": ("utils_2p.timing", "get_trigger_time"),
    "infer_processing_status": ("utils_2p.slurm_pipeline", "infer_processing_status"),
    "load_mat_struct": ("utils_2p.matlab", "load_mat_struct"),
    "norm01": ("utils_2p.stats", "norm01"),
    "pad_seq": ("utils_2p.alignment", "pad_seq"),
    "read_raw_voltages_basic": ("utils_2p.voltage", "read_raw_voltages_basic"),
    "read_raw_voltages_memmap": ("utils_2p.voltage", "read_raw_voltages_memmap"),
    "trim_seq": ("utils_2p.alignment", "trim_seq"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
