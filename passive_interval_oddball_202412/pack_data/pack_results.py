#!/usr/bin/env python3

import argparse
import os

import h5py
import numpy as np


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def write_dataset(dst, name, data):
    arr = np.array(data)
    kwargs = {}
    if arr.shape != () and arr.size > 1024 and arr.dtype.kind in 'biuf':
        kwargs = {'compression': 'gzip', 'compression_opts': 4, 'shuffle': True}
    dst.create_dataset(name, data=arr, **kwargs)


def copy_group(src, dst):
    copy_attrs(src, dst)
    for key in src.keys():
        if isinstance(src[key], h5py.Group):
            copy_group(src[key], dst.create_group(key))
        else:
            write_dataset(dst, key, src[key])
            copy_attrs(src[key], dst[key])


def pack_session(session_path, out_path):
    neural_path = os.path.join(session_path, 'neural_trials.h5')
    masks_path = os.path.join(session_path, 'masks.h5')
    if not os.path.exists(neural_path) or not os.path.exists(masks_path):
        return False
    os.makedirs(out_path, exist_ok=True)
    session_name = os.path.basename(session_path)
    packed_path = os.path.join(out_path, session_name + '.h5')
    if os.path.exists(packed_path):
        os.remove(packed_path)
    with h5py.File(packed_path, 'w') as fout:
        fout.attrs['source_session'] = session_name
        with h5py.File(masks_path, 'r') as fm:
            write_dataset(fout, 'labels', fm['labels'])
            masks_group = fout.create_group('masks')
            for key in fm.keys():
                if key != 'labels':
                    write_dataset(masks_group, key, fm[key])
                    copy_attrs(fm[key], masks_group[key])
        with h5py.File(neural_path, 'r') as fn:
            copy_group(fn['neural_trials'], fout.create_group('neural_trials'))
    return True


def pack_results(results_dir, output_dir):
    n_pack = 0
    for subject in sorted(os.listdir(results_dir)):
        subject_path = os.path.join(results_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        for session in sorted(os.listdir(subject_path)):
            session_path = os.path.join(subject_path, session)
            if not os.path.isdir(session_path):
                continue
            out_path = os.path.join(output_dir, subject, session)
            if pack_session(session_path, out_path):
                n_pack += 1
                print('Packed {}'.format(os.path.join(subject, session)))
    print('Packed {} sessions into {}'.format(n_pack, output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack labels, masks, and neural_trials into compact release h5 files.')
    parser.add_argument('--results_dir', default='results', help='Input results folder.')
    parser.add_argument('--output_dir', default='results_pack', help='Output packed results folder.')
    args = parser.parse_args()
    pack_results(args.results_dir, args.output_dir)
