import numpy as np
import scipy.io as sio


def _ndarray_to_list(ndarray):
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, sio.matlab.mat_struct):
            elem_list.append(_mat_struct_to_dict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(_ndarray_to_list(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list


def _mat_struct_to_dict(matobj):
    data = {}
    for field_name in matobj._fieldnames:
        elem = matobj.__dict__[field_name]
        if isinstance(elem, sio.matlab.mat_struct):
            data[field_name] = _mat_struct_to_dict(elem)
        elif isinstance(elem, np.ndarray):
            data[field_name] = _ndarray_to_list(elem)
        else:
            data[field_name] = elem
    return data


def _check_keys(data):
    for key in data:
        if isinstance(data[key], sio.matlab.mat_struct):
            data[key] = _mat_struct_to_dict(data[key])
    return data


def load_mat_struct(mat_path, root_key=None):
    raw = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    raw = _check_keys(raw)
    if root_key is None:
        return raw
    return raw[root_key]


def extract_bpod_session_data(mat_path, field_map):
    session_data = load_mat_struct(mat_path, root_key="SessionData")
    return {output_key: np.array(session_data[input_key]) for output_key, input_key in field_map.items()}
