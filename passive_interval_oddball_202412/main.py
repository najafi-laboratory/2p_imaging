#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import gc
import os
import argparse
from datetime import datetime

from modules import Trialization
from modules.ReadResults import read_ops

def combine_session_config_list(session_config_list, results_dir='results'):
    list_session_data_path = []
    for sc in session_config_list['list_config']:
        list_session_data_path += [
            os.path.join(results_dir, sc['session_folder'], n)
            for n in sc['list_session_name'].keys()]
    list_session_name = {}
    for sc in session_config_list['list_config']:
        list_session_name.update({
            os.path.join(sc['session_folder'], k): v
            for k, v in sc['list_session_name'].items()})
    session_config_list['list_session_name'] = list_session_name
    session_config_list['list_session_data_path'] = list_session_data_path
    return session_config_list

import visualization1_FieldOfView
import visualization2_3331Random
import visualization3_1451ShortLong
import visualization4_4131FixJitterOdd
import visualization5_3331RandomExtended
from webpage import pack_webpage_main

def run(session_config_list, cate_list, do_trialization=False, use_packed=False):
    smooth = False
    results_dir = 'results_pack' if use_packed else 'results'
    session_config_list['use_packed'] = use_packed

    print('===============================================')
    print('============ Start Processing Data ============')
    print('===============================================')
    if not os.path.exists(os.path.join('results', 'temp_'+session_config_list['subject_name'])):
        os.makedirs(os.path.join('results', 'temp_'+session_config_list['subject_name']))
    print('Created canvas')
    session_config_list = combine_session_config_list(session_config_list, results_dir)
    print('Processing {} sessions'.format(
        len(session_config_list['list_session_data_path'])))
    for n in session_config_list['list_session_name']:
        print(n)
    if use_packed:
        print('Using packed h5 results')
        list_ops = []
    else:
        print('Reading ops.npy')
        list_ops = read_ops(session_config_list['list_session_data_path'])

    if do_trialization and not use_packed:
        print('===============================================')
        print('============= trials segmentation =============')
        print('===============================================')
        for i in range(len(list_ops)):
            print('Trializing {}'.format(
                list(session_config_list['list_session_name'].keys())[i]))
            Trialization.run(list_ops[i])
    elif use_packed:
        print('Skip trialization for packed h5 results')
    else:
        print('Skip trialization')

    print('===============================================')
    print('======== plotting representative masks ========')
    print('===============================================')
    fn1 = visualization1_FieldOfView.run(session_config_list, smooth, cate_list)
    #fn1 = []

    print('===============================================')
    print('========= plotting 3331Random results =========')
    print('===============================================')
    fn2 = visualization2_3331Random.run(session_config_list, smooth, cate_list)
    #fn2 = []

    print('===============================================')
    print('======= plotting 1451ShortLong results ========')
    print('===============================================')
    fn3 = visualization3_1451ShortLong.run(session_config_list, smooth, cate_list)
    #fn3 = []

    print('===============================================')
    print('====== plotting 4131FixJitterOdd results ======')
    print('===============================================')
    fn4 = visualization4_4131FixJitterOdd.run(session_config_list, smooth, cate_list)
    #fn4 = []

    print('===============================================')
    print('===== plotting 3331RandomExtended results =====')
    print('===============================================')
    fn5 = visualization5_3331RandomExtended.run(session_config_list, smooth, cate_list)
    #fn5 = []

    print('===============================================')
    print('============ saving session report ============')
    print('===============================================')
    print('Saving results')
    pack_webpage_main.run(
        session_config_list,
        [fn1, fn2, fn3, fn4, fn5],
        ['Field of View', 'the Random session', 'the Short-Long session', 'the Fix-Jitter-Oddball session', 'the Extended Random session'],
        ['random', 'fix_jitter_odd', 'short_long'])
    for i in range(len(list_ops)):
        print('Cleaning memory mapping files for {}'.format(
            list(session_config_list['list_session_name'].keys())[i]))
    print('Processing completed for all sessions')
    for n in session_config_list['list_session_name']:
        print(n)
    print('File saved as '+os.path.join('results', session_config_list['output_filename'] + '.html'))
    print('Finished at '+datetime.now().strftime("%Y.%m.%d %H:%M:%S"))
    del list_ops
    gc.collect()


if __name__ == "__main__":
    COMMANDLINE_MODE = 1
    cate_list = [[-1], [1], [2], [-1,1,2]]
    from session_configs import all_config_list
    
    if COMMANDLINE_MODE:
        parser = argparse.ArgumentParser(description='Experiments can go shit but Yicong will love you forever!')
        parser.add_argument('--config_list', required=True, type=str, help='Target subjects.')
        parser.add_argument('--do_trialization', action='store_true', help='Run trial segmentation before analysis.')
        parser.add_argument('--use_packed', action='store_true', help='Read compact h5 files from results_pack.')
        args = parser.parse_args()
        for subject, session_config_list in zip(
                ['YH01VT', 'YH02VT', 'YH03VT', 'YH14SC', 'YH16SC',
                 'YH17VT', 'YH18VT', 'YH19VT', 'YH20SC', 'YH21SC',
                 'PPC', 'V1'],
                all_config_list
            ):
            if subject in args.config_list:
                run(session_config_list, cate_list, args.do_trialization, args.use_packed)

    else:
        520
