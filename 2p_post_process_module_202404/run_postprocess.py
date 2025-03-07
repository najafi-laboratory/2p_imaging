#!/usr/bin/env python3

import os
import argparse
import numpy as np

from modules import QualControlDataIO
from modules import LabelExcInh
from modules import DffTraces
COMMANDLINE_MODE = False

# parse arg input to number list.
def get_qc_args(args):
    range_skew = args.range_skew.split(',')
    range_skew = np.array(range_skew, dtype='float32')
    max_connect = np.array(args.max_connect, dtype='float32')
    max_aspect = np.array(args.max_aspect, dtype='float32')
    range_compact = args.range_compact.split(',')
    range_compact = np.array(range_compact, dtype='float32')
    range_footprint = args.range_footprint.split(',')
    range_footprint = np.array(range_footprint, dtype='float32')
    return [range_skew,
            max_connect,
            max_aspect,
            range_compact,
            range_footprint]

# read saved ops.npy given a folder in ./results.
def read_ops(session_data_path):
    print('Processing {}'.format(session_data_path))
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    return ops

if __name__ == "__main__":

    if COMMANDLINE_MODE:
        parser = argparse.ArgumentParser(description='Do not forget the everlasting love from Yicong!')
        parser.add_argument('--session_data_path', required=True, type=str, help='The name of folder to save suite2p results.')
        parser.add_argument('--range_skew',        required=True, type=str, help='The range of skew for quality control.')
        parser.add_argument('--max_connect',       required=True, type=str, help='The maximum number of connectivity for quality control.')
        parser.add_argument('--range_footprint',   required=True, type=str, help='The range of footprint for quality control')
        parser.add_argument('--max_aspect',        required=True, type=str, help='The maximum value of aspect ratio for quality control.')
        parser.add_argument('--range_compact',     required=True, type=str, help='The range of compact for quality control.')
        parser.add_argument('--diameter',          required=True, type=str, help='The diameter for cellpose on anatomical channel.')
        args = parser.parse_args()
        [range_skew,
         max_connect,
         max_aspect,
         range_compact,
         range_footprint] = get_qc_args(args)
        ops = read_ops(args.session_data_path)
    else:
        '''
        session_data_path_list = [
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250106_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250107_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250108_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250109_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250111_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250113_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250114_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250115_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250116_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250117_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250118_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250120_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250121_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250122_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250123_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250127_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250128_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250129_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250130_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250131_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250201_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250203_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250204_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250205_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250206_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250207_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250208_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250210_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250212_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250214_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250217_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250218_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250220_1451ShortLong',
            ]
        session_data_path_list = [
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250106_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250107_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250108_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250109_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250111_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250113_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250114_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250115_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250116_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250117_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250118_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250120_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250121_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250122_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250123_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250127_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250129_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250130_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250131_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250202_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250203_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250205_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250206_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250207_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250208_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250210_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250211_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250212_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250213_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250214_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250217_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250218_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250221_1451ShortLong',
            ]
        session_data_path_list = [
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250106_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250107_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250108_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250109_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250111_3331Random',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250113_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250114_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250116_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250117_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250118_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250120_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250121_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250122_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250123_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250129_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250130_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250131_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250201_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250203_4131FixJitterOdd',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250205_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250206_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250207_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250208_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250210_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250212_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250214_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250217_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250218_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250219_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250221_1451ShortLong',
            ]
        '''
        session_data_path_list = [
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250217_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250218_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH01VT/VTYH01_PPC_20250220_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250217_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250218_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH02VT/VTYH02_PPC_20250221_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250217_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250218_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250219_1451ShortLong',
            'C:/Users/yhuang887/Projects/passive_interval_oddball_202412/results/YH03VT/VTYH03_PPC_20250221_1451ShortLong',
            ]

        for session_data_path in session_data_path_list:
            ops = read_ops(session_data_path)
            '''
            range_skew = [1,2]
            max_connect = 1
            max_aspect = 55
            range_footprint = [1,1]
            range_compact = [1.05,5]
            diameter = 6
            '''
            range_skew = [-5,5]
            max_connect = 1
            max_aspect = 5
            range_footprint = [1,2]
            range_compact = [0,1.05]
            diameter = 6
            '''
            QualControlDataIO.run(
                ops, range_skew, max_connect, max_aspect, range_compact, range_footprint,
                run_qc=False)
            '''
        
            QualControlDataIO.run(
                ops,
                range_skew, max_connect, max_aspect, range_compact, range_footprint)
        
            LabelExcInh.run(ops, diameter)
        
            DffTraces.run(ops, correct_pmt=False)
