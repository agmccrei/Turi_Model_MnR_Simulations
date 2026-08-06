#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 11:30:35 2018.

@author: spiros
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import matplotlib
from place_cell_metrics import field_size

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

nTrials = 10

folder1 = 'data_analysis'
fdname0 = folder1+'/metrics/'
fdname1 = folder1+'/metrics_interneurons/'
fdname2 = folder1+'/figures_interneurons/'

learning = ['prelearning_1', 'postlearning_1', 'postlearning_2']
my_list = ['Control', 'No_VIPcells', 'No_VIPCR',
           'No_VIPCCK', 'No_VIPPVM', 'No_VIPNVM']
cells = ['aacell', 'bcell', 'bscell', 'olm',
         'vipcr', 'vipcck', 'vipcrnvm', 'pvsoma']
trials = xrange(1, nTrials+1)
npath_x = 200
npath_y = 1
Nbins = 100


col = ['blue', 'green', 'orange', 'red']


for learn in ['postlearning_2']:
    means_all_cells = {}
    sems_all_cells = {}
    means_out_cells = {}
    sems_out_cells = {}
    means_rwd_cells = {}
    sems_rwd_cells = {}
    for cell in cells:
        means1 = []
        sems1 = []
        means2 = []
        sems2 = []
        means3 = []
        sems3 = []

        for case in my_list:
            print(case)

            rtmaps = []
            for ntrial in trials:
                if cell == 'pvsoma':
                    with open(fdname0+learn+'/pickled_sn_'+case+'_'+str(ntrial)+'.pkl', 'rb') as f:
                        loaded_data = pickle.load(f)
                    nCells = loaded_data['maps'].shape[0]
                    for i in range(nCells):
                        fmax = np.max(loaded_data['maps'][i, :, :].squeeze())
                        size = field_size(
                            loaded_data['maps'][i, :, :], relfreq=0.10*fmax)[0]
                        # Keep only Place cells
                        if fmax >= 3 and size >= 4:

                            rtmaps += list(loaded_data['maps'][i, :, :].T)

                else:
                    with open(fdname1+learn+'/pickled_sn_'+cell+'_'+case+'_'+str(ntrial)+'.pkl', 'rb') as f:
                        loaded_data = pickle.load(f)

                    nCells = loaded_data['maps'].shape[0]
                    if np.sum(loaded_data['maps']) != 0:
                        for i in range(nCells):
                            fmax = np.max(
                                loaded_data['maps'][i, :, :].squeeze())
                            size = field_size(
                                loaded_data['maps'][i, :, :], relfreq=0.10*fmax)[0]
                            # Keep all neurons.
                            if fmax >= 0 and size >= 0:

                                rtmaps += list(loaded_data['maps'][i, :, :].T)

            rtmaps = np.array(rtmaps).squeeze()
            if rtmaps.shape[0] != 0:
                maps_all = list(np.mean(rtmaps, axis=1))
                maps_rwd = list(np.mean(rtmaps[:, 40:56], axis=1))
                maps_out = list(
                    np.mean(np.delete(rtmaps, range(40, 56), axis=1), axis=1))

                means1.append(maps_all)
                means2.append(maps_rwd)
                means3.append(maps_out)
            else:
                means1.append(0)
                means2.append(0)
                means3.append(0)

        means_all_cells[cell] = means1
        means_rwd_cells[cell] = means2
        means_out_cells[cell] = means3

    for cell in cells:
        N = len(my_list)
        all_means = [np.mean(x) for x in means_all_cells[cell]]
        all_sems = [sem(x).item() for x in means_all_cells[cell]]

        ind = np.arange(N)  # the x locations for the groups
        width = 0.25       # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, all_means, width, color='r', yerr=all_sems)

        rwd_means = [np.mean(x) for x in means_rwd_cells[cell]]
        rwd_sems = [sem(x).item() for x in means_rwd_cells[cell]]
        rects2 = ax.bar(ind+width, rwd_means, width, color='y', yerr=rwd_sems)

        out_means = [np.mean(x) for x in means_out_cells[cell]]
        out_sems = [sem(x).item() for x in means_out_cells[cell]]
        rects3 = ax.bar(ind+2*width, out_means, width,
                        color='b', yerr=out_sems)

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Mean firing rate [Hz]')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(tuple(my_list))

        ax.legend((rects1[0], rects2[0], rects3[0]),
                  ('Track', 'Reward zone', 'Outside reward'))

        plt.savefig(fdname2+learn+'/'+cell+'.pdf', format='pdf', dpi=600)


# Figure 7 D - uncomment below
# N = len(cells)
# all_means=[]
# all_sems = []
# for cell in cells:
#    if cell != 'pvsoma':
#        all_means.append(np.mean([np.mean(x) for x in means_all_cells[cell][0]]))
#        all_sems.append(np.std([np.mean(x) for x in means_all_cells[cell][0]]))
#
#
# plt.figure()
# plt.bar(range(N-1), all_means, width,color='r', yerr=all_sems)
# plt.xticks(np.arange(N-1), tuple(cells[:-1]))
# plt.savefig(fdname2+learn+'/figure7.pdf', format='pdf', dpi=600)
