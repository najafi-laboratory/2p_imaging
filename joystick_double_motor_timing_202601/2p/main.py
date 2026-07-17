#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import gc
import os
import argparse
from datetime import datetime

from modules import Trialization
from modules.ReadResults import read_ops

def combine_session_config_list(session_config_list):
    list_session_data_path = []
    for sc in session_config_list['list_config']:
        list_session_data_path += [
            os.path.join('results', sc['session_folder'], n)
            for n in sc['list_session_name'].keys()]
    list_session_name = [sc['list_session_name'] for sc in session_config_list['list_config']]
    list_session_name = {k: v for d in list_session_name for k, v in d.items()}
    session_config_list['list_session_name'] = list_session_name
    session_config_list['list_session_data_path'] = list_session_data_path
    return session_config_list

import visualization1_FieldOfView
import visualization2_Behavior
import visualization3_StateAlignments
from webpage import pack_webpage_main

def run(session_config_list, cate_list):
    smooth = False

    print('===============================================')
    print('============ Start Processing Data ============')
    print('===============================================')
    if not os.path.exists(os.path.join('results', 'temp_'+session_config_list['subject_name'])):
        os.makedirs(os.path.join('results', 'temp_'+session_config_list['subject_name']))
    print('Created canvas')
    session_config_list = combine_session_config_list(session_config_list)
    print('Processing {} sessions'.format(
        len(session_config_list['list_session_data_path'])))
    for n in session_config_list['list_session_name']:
        print(n)
    print('Reading ops.npy')
    list_ops = read_ops(session_config_list['list_session_data_path'])

    print('===============================================')
    print('============= trials segmentation =============')
    print('===============================================')
    for i in range(len(list_ops)):
        print('Trializing {}'.format(
            list(session_config_list['list_session_name'].keys())[i]))
        Trialization.run(list_ops[i])
        print('-------------------------')

    print('===============================================')
    print('======== plotting representative masks ========')
    print('===============================================')
    #fn1 = visualization1_FieldOfView.run(session_config_list, smooth, cate_list)
    fn1 = []
    
    print('===============================================')
    print('======== plotting behavior performance ========')
    print('===============================================')
    #fn2 = visualization2_Behavior.run(session_config_list, smooth, cate_list)
    fn2 = []
    
    print('===============================================')
    print('========== plotting state alignments ==========')
    print('===============================================')
    #fn3 = visualization3_StateAlignments.run(session_config_list, smooth, cate_list)
    fn3 = []
    
    print('===============================================')
    print('============== plotting modeling ==============')
    print('===============================================')
    #fn4 = visualization4_Model.run(session_config_list, smooth, cate_list)
    fn4 = []

    print('===============================================')
    print('============ saving session report ============')
    print('===============================================')
    print('Saving results')
    pack_webpage_main.run(
        session_config_list,
        [fn1, fn2, fn3, fn4],
        ['Field of View', 'Behavior', 'State Alignments', 'Modelings'],
        ['single'])
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
    COMMANDLINE_MODE = 0
    cate_list = [[-1,1,2]]
    from session_configs import all_config_list
    
    if COMMANDLINE_MODE:
        parser = argparse.ArgumentParser(description='Experiments can go shit but Yicong will love you forever!')
        parser.add_argument('--config_list', required=True, type=str, help='Whether run denoising.')
        args = parser.parse_args()
        for subject, session_config_list in zip(
                ['YH01VT', 'YH02VT', 'YH03VT', 'YH14SC', 'YH16SC',
                 'YH17VT', 'YH18VT', 'YH19VT', 'YH20SC', 'YH21SC',
                 'PPC', 'V1'],
                all_config_list
            ):
            if subject in args.config_list:
                run(session_config_list, cate_list)

    else:

        session_config_test = {
            'list_session_name' : {
                'YH30VT_20260219_joystick' : 'double',
                'YH30VT_20260224_joystick' : 'double',
                },
            'session_folder' : 'test',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        session_config_list_test = {
            'list_config': [
                session_config_test,
                ],
            'label_names' : {
                '-1':'Exc',
                '1':'Inh_VIP',
                '2':'Inh_SST',
                },
            'subject_name' : 'test',
            'output_filename' : 'test_double'
            }
        
        session_config_YH33 = {
            'list_session_name' : {
                '20260122' : 'single',
                '20260523' : 'single',
                '20260524' : 'single',
                '20260607' : 'single',
                #'20260612' : 'single',
                #'20260614' : 'single',
                },
            'session_folder' : 'YH33',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        session_config_YH39LG = {
            'list_session_name' : {
                '20260611' : 'double',
                #'20260612' : 'double',
                #'20260615' : 'double',
                #'20260617' : 'double',
                #'20260618' : 'double',
                },
            'session_folder' : 'YH39LG',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        session_config_YH40LG = {
            'list_session_name' : {
                '20260611' : 'double',
                #'20260612' : 'double',
                '20260615' : 'double',
                '20260617' : 'double',
                #'20260618' : 'double',
                },
            'session_folder' : 'YH40LG',
            'sig_tag' : 'all',
            'force_label' : None,
            }
        session_config_list_test = {
            'list_config': [
                session_config_YH33,
                #session_config_YH39LG,
                #session_config_YH40LG,
                ],
            'label_names' : {
                '-1':'Exc',
                '1':'Inh_VIP',
                '2':'Inh_SST',
                },
            'subject_name' : 'test_fp',
            'output_filename' : 'test_fp'
            }
        
        
        '''
        
        session_config_list=session_config_list_test
        # run(session_config_list_test, cate_list)
        
        import matplotlib.pyplot as plt
        session_config_list = combine_session_config_list(session_config_list_test)
        list_ops = read_ops(session_config_list['list_session_data_path'])
        ops = list_ops[0]

        from modules.ReadResults import read_all
        [list_labels, list_masks,
         list_neural_trials
         ] = read_all(session_config_list, smooth=False)
        neural_trials = list_neural_trials[0]
        dff = neural_trials['dff']
        trial_labels = neural_trials['trial_labels']
        label_names = {'-1':'Exc', '1':'Inh_VIP', '2':'Inh_SST'}
        cate = [-1,1,2]
        roi_id = None
        norm_mode='none'
        target_state='state_press1'
        ti=12
        
        cluster_cmap = plt.cm.hsv
        standard = 1
        oddball = 1
        block = 0
        mode = 'post'
        temp_folder = 'temp_'+session_config_list['subject_name']
        if not os.path.exists(os.path.join('results', 'temp_'+session_config_list['subject_name'])):
            os.makedirs(os.path.join('results', 'temp_'+session_config_list['subject_name']))


        '''
