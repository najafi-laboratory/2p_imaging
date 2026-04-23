#!/usr/bin/env python3

import os
import argparse
import numpy as np

from modules import QualControlDataIO
from modules import LabelExcInh
from modules import DffTraces


def get_qc_args(args):
    """Parse CLI QC argument strings into numeric arrays."""
    range_skew = np.array(args.range_skew.split(','), dtype='float32')
    max_connect = np.array(args.max_connect, dtype='float32')
    range_aspect = np.array(args.range_aspect.split(','), dtype='float32')
    range_compact = np.array(args.range_compact.split(','), dtype='float32')
    range_footprint = np.array(args.range_footprint.split(','), dtype='float32')
    return range_skew, max_connect, range_aspect, range_compact, range_footprint


def read_ops(session_data_path):
    """Load suite2p ops for the given session."""
    print(f'Processing {session_data_path}')
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0', 'ops.npy'),
        allow_pickle=True,
    ).item()
    ops['save_path0'] = session_data_path
    return ops


def process_session(session_data_path, args):
    range_skew, max_connect, range_aspect, range_compact, range_footprint = get_qc_args(args)
    ops = read_ops(session_data_path)
    QualControlDataIO.run(ops, range_skew, max_connect, range_aspect, range_compact, range_footprint)
    LabelExcInh.run(ops, args.diameter)
    DffTraces.run(ops, correct_pmt=False)


def main():
    parser = argparse.ArgumentParser(description='Run suite2p post-processing/QC utilities.')
    parser.add_argument('--session_data_path', required=True, type=str, help='Folder containing suite2p outputs.')
    parser.add_argument('--range_skew', required=True, type=str, help='Range of skew for quality control (e.g., -5,5).')
    parser.add_argument('--max_connect', required=True, type=str, help='Maximum connectivity for QC (e.g., 1).')
    parser.add_argument('--range_footprint', required=True, type=str, help='Range of footprint sizes (e.g., 1.0,2.0).')
    parser.add_argument('--range_aspect', required=True, type=str, help='Range of aspect ratios (e.g., 0.0,5.0).')
    parser.add_argument('--range_compact', required=True, type=str, help='Range of compactness values (e.g., 0,1.06).')
    parser.add_argument('--diameter', required=True, type=float, help='Cellpose diameter for LabelExcInh.')
    args = parser.parse_args()
    process_session(args.session_data_path, args)


if __name__ == "__main__":
    main()
