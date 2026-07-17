#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

from modules.ReadResults import read_subject
from plot.fig1_beh import run

max_n_sess = 555

if __name__ == "__main__":

    subject_list = ['YH35WT']
    
    for subject_name in subject_list:
        list_sess_time, list_trial_labels = read_subject(subject_name)
        if len(list_trial_labels) > max_n_sess:
            list_sess_time = list_sess_time[-max_n_sess:]
            list_trial_labels = list_trial_labels[-max_n_sess:]
        run(subject_name, list_sess_time, list_trial_labels)