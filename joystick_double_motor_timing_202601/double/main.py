#!/usr/bin/env python3

import os
import fitz
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from modules.ReadResults import read_subject

max_n_sess = 10

if __name__ == "__main__":

    subject_list = ['YH30', 'YH31']
    
    subject_name = 'YH35WT'
    
    list_sess_time, list_trial_labels = read_subject(subject_name)
    if len(list_trial_labels) > max_n_sess:
        list_sess_time = list_sess_time[-max_n_sess:]
        list_trial_labels = list_trial_labels[-max_n_sess:]

'''
    session_data = DataIO.run(subject_list)
    subject_session_data = session_data[0]

    subject_report = fitz.open()
    for i in range(len(session_data)):
        fig = plt.figure(layout='constrained', figsize=(35, 20))
        gs = GridSpec(4, 7, figure=fig)
        #plot_decision_outcome.run(plt.subplot(gs[1, 6]), session_data[i])
        plt.suptitle(session_data[i]['subject'])
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(35, 20)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    subject_report.save('./figures/subject_report.pdf')
    subject_report.close()
'''
