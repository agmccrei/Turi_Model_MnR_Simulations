#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:01:48 2017.

@author: spiros
"""
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

npath_x, npath_y = 200, 1
Nbins = 100

nTrials = 10
for icell in ['aacell', 'bcell', 'bscell', 'olm', 'vipcck', 'vipcr',
              'vipcrnvm']:
    numbers_ALL = {}
    numbers_rwd_ALL = {}
    for learning in ['prelearning', 'locomotion', 'reward']:
        print("\nLEARNING: ", learning)
        print
        print
        spec = 'data_analysis'

        os.system('mkdir -p '+spec+'/figures_final_interneurons/'+learning)

        back = '0'
        path_figs = spec+'/figures_final_interneurons/'
        file_load = spec+'/metrics_interneurons/'+learning
        trials = [str(i) for i in range(1, nTrials+1)]
        maindir = os.getcwd()

        my_list = ['Control']
        rateMaps = {}
        for case in my_list:

            for ntrial in trials:
                with open(file_load+'/pickled_sn_'+icell+'_'+case+'_'+ntrial+'.pkl', 'rb') as f:
                    loaded_data = pickle.load(f)

                if ntrial == '1':
                    rateMaps[case] = loaded_data['maps']
                else:
                    rateMaps[case] = np.concatenate(
                        (rateMaps[case], loaded_data['maps']), axis=0)

        nCells = rateMaps['Control'].shape[0]

        for case in my_list:

            idx = np.argmax(rateMaps[case].squeeze(), axis=1)
            new_idx = np.lexsort((range(nCells), idx))
            rtMaps = rateMaps[case][new_idx, :, :].squeeze()

            Max = np.max(rtMaps, axis=1).reshape(-1, 1)
            for i in range(Max.shape[0]):
                if Max[i, 0] == 0:
                    Max[i, 0] = 1e-12

            rtMaps = rtMaps / np.repeat(Max, Nbins, axis=1)

            fig, axes = plt.subplots(nrows=1, ncols=1)

            im0 = axes.imshow(rtMaps, cmap='jet', aspect='equal')
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            axes.set_xlim((0, 100))
            axes.set_xticks(range(0, 100+1, 25))
            axes.set_xticklabels(
                ['-0.5', '-0.25', '0', '0.25', '0.5'], fontsize=11)
            axes.set_title(case+'_' + icell, fontsize=14)
            axes.set_ylabel('# cell index', fontsize=12)
            axes.set_xlabel('position', fontsize=12)
            axes.tick_params(direction='out', width=1.5)
            # Hide the right and top spines
            axes.spines['right'].set_visible(False)
            axes.spines['top'].set_visible(False)

            # Only show ticks on the left and bottom spines
            axes.yaxis.set_ticks_position('left')
            axes.xaxis.set_ticks_position('bottom')
            plt.colorbar(im0, cax=cax)
            ax = plt.gca()
            plt.tight_layout()
            plt.savefig(path_figs+learning+'/'+case+'_'+icell +
                        '_heatmap.pdf', format='pdf', dpi=300)
            plt.cla()
            plt.clf()
            plt.close()
