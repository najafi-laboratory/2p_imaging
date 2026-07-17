#!/usr/bin/env python3

import numpy as np
from modules.Alignment import run_sess_alignment

alignment=run_sess_alignment(list_neural_trials)

list_states = [
    'state_vis1',
    'state_press1',
    'state_retract1',
    'state_delay',
    'state_press2',
    'state_retract2',
    'state_reward',
    'state_iti'
    ]

filter_outcome = [
    'reward',
    'early_2nd_press',
    ]

colors = ['mediumseagreen', 'coral']

fig, axs = plt.subplots(3, 8, figsize=(24, 8), layout='tight')

for i,state in enumerate(list_states):
    neu_time = alignment[state]['neu_time']
    if not np.isnan(np.sum(neu_time)):
        
        for ai, (out, color) in enumerate(zip(filter_outcome, colors)):
            
            list_js_rot = alignment[state]['list_js_rot']
            list_outcome = alignment[state]['list_outcome']
            list_js_rot = [j[o==out,:] for j,o in zip(list_js_rot, list_outcome)]
            js_rot = np.concatenate(list_js_rot,axis=0)
            m, s = get_mean_sem(js_rot)
            if not np.isnan(np.nansum(m)):
                plot_mean_sem(axs[0,i],neu_time, m, s, color)
                adjust_layout_neu(axs[0,i])
                axs[0,i].set_xlim([-2000,2000])
                axs[0,i].axvline(0, color='black', lw=1, linestyle='--')
                axs[0,i].set_ylabel('Joystick deflection (deg)')
                axs[0,i].set_title(state)
                
            list_js_rot = alignment[state]['list_js_rot']
            list_outcome = alignment[state]['list_outcome']
            list_js_rot = [j[o==out,:] for j,o in zip(list_js_rot, list_outcome)]
            js_rot = np.diff(np.concatenate(list_js_rot,axis=0), prepend=0)
            m, s = get_mean_sem(js_rot)
            if not np.isnan(np.nansum(m)):
                plot_mean_sem(axs[1,i],neu_time, m, s, color)
                adjust_layout_neu(axs[1,i])
                axs[1,i].set_xlim([-2000,2000])
                axs[1,i].axvline(0, color='black', lw=1, linestyle='--')
                axs[1,i].set_ylabel('Joystick deflection velocity (deg)')
                axs[1,i].set_title(state)
            
            list_neu_seq = alignment[state]['list_neu_seq']
            list_outcome = alignment[state]['list_outcome']
            list_neu_seq = [ns[o==out,:,:] for ns,o in zip(list_neu_seq, list_outcome)]
            neu_seq = np.concatenate([ns[:,:,:].reshape(-1, len(neu_time)) for ns in list_neu_seq],axis=0)
            m, s = get_mean_sem(neu_seq)
            if not np.isnan(np.nansum(m)):
                plot_mean_sem(axs[2,i],neu_time, m, s, color)
                adjust_layout_neu(axs[2,i])
                axs[2,i].set_xlabel('Time since state start (ms)')
                axs[2,i].set_xlim([-2000,2000])
                axs[2,i].axvline(0, color='black', lw=1, linestyle='--')
    
add_legend(axs[0,-1], colors, filter_outcome, None, None, None, 'upper right')