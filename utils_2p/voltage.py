import os

import h5py
import numpy as np


RAW_VOLTAGE_FIELDS = [
    ("vol_time", "float32"),
    ("vol_start", "int8"),
    ("vol_stim_vis", "int8"),
    ("vol_hifi", "int8"),
    ("vol_img", "int8"),
    ("vol_stim_aud", "float32"),
    ("vol_flir", "int8"),
    ("vol_pmt", "int8"),
    ("vol_led", "int8"),
]

LEGACY_FIELD_ALIASES = {
    "vol_start": "vol_start_bin",
    "vol_stim_vis": "vol_stim_bin",
    "vol_img": "vol_img_bin",
}


def read_raw_voltages_basic(file_path):
    with h5py.File(file_path, "r") as handle:
        raw = handle["raw"]
        results = []
        for field_name, _dtype in RAW_VOLTAGE_FIELDS:
            if field_name in raw:
                results.append(np.array(raw[field_name]))
            elif field_name in LEGACY_FIELD_ALIASES and LEGACY_FIELD_ALIASES[field_name] in raw:
                results.append(np.array(raw[LEGACY_FIELD_ALIASES[field_name]]))
            else:
                results.append(np.zeros_like(results[0]) if results else np.array([], dtype=float))
    return results


def read_raw_voltages_memmap(file_path, mm_path, create_memmap):
    with h5py.File(file_path, "r") as handle:
        raw = handle["raw"]
        results = []
        for field_name, dtype in RAW_VOLTAGE_FIELDS:
            dataset = raw[field_name]
            mmap_path = os.path.join(mm_path, f"{field_name}.mmap")
            results.append(create_memmap(dataset, dtype, mmap_path))
    return results
