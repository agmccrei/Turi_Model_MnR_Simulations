#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:01:48 2017.

@author: spiros
"""
from place_cell_metrics import field_size, peak_frequency
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def bar_plots(mydict, metric, path_figs, learning):
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats

    my_list = ['Control', 'No_VIPcells', 'No_VIPCR', 'No_VIPCCK',
               'No_VIPPVM', 'No_VIPNVM', 'No_VIPCRtoOLM', 'No_VIPCRtoBC']
    A = mydict
    A_means = []
    A_sems = []
    for case in my_list:
        A_means.append(np.mean(A[case]))
        A_sems.append(scipy.stats.sem(A[case]))

    plt.figure(1, dpi=300)

    y = A_means
    labels = my_list
    N = len(y)
    x = range(N)

    p1, p2, p3, p4, p5, p6, p7, p8 = plt.bar(x, y, yerr=A_sems)

    p1.set_facecolor('blue')
    p2.set_facecolor('red')
    p3.set_facecolor('green')
    p4.set_facecolor('yellow')
    p5.set_facecolor('lightblue')
    p6.set_facecolor('olive')
    p7.set_facecolor('darkmagenta')
    p8.set_facecolor('darkorange')

    plt.xticks(x, labels, rotation='45')
    plt.ylabel(metric, fontsize=16)
    plt.title(metric)

    plt.savefig(path_figs+'/'+learning+'_'+metric +
                '_barplot.pdf', format='pdf', dpi=300)
    plt.cla()
    plt.clf()
    plt.close()

    # Make Boxplots
    A_list = []
    for case in my_list:
        A_list.append(list(mydict[case]))

    plt.figure(1, dpi=300)

    y = A_list
    labels = my_list
    N = len(y)
    x = range(1, N+1)

    # notch shape box plot
    # will be used to label x-ticks
    bplot = plt.boxplot(y, notch=True, vert=True,
                        patch_artist=True, labels=labels)

    # fill with colors
    colors = ['blue', 'red', 'green', 'yellow',
              'lightblue', 'olive', 'darkmagenta', 'darkorange']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    for element in ['fliers', 'means', 'medians', 'caps']:
        plt.setp(bplot[element], color='black')

    plt.xticks(x, labels, rotation='45')
    plt.ylabel(metric, fontsize=16)

    plt.savefig(path_figs+'/'+learning+'_'+metric +
                '_boxplot.pdf', format='pdf', dpi=300)
    plt.cla()
    plt.clf()
    plt.close()


Npyramidals = 130
Nperms = 500
Nbins = 100

npath_x, npath_y = 200, 1
xlim1, xlim2 = 80, 110
xrew1, xrew2 = xlim1/(npath_x/Nbins), xlim1/(npath_x/Nbins)+1

nTrials = 10
numbers_ALL = {}
numbers_rwd_ALL = {}
for learning in ['prelearning', 'locomotion', 'reward']:
    print("\nLEARNING: ", learning)
    print
    print
    spec = 'data_analysis'

    os.system('mkdir -p '+spec+'/figures_final/'+learning)

    path_figs = spec+'/figures_final/'
    file_load = spec+'/metrics/'+learning
    trials = [str(i) for i in range(1, nTrials+1)]
    maindir = os.getcwd()

    numbers_all = {}
    numbers_rwd_all = {}
    reward_zone_all = {}
    reward_zone_all2 = {}

    my_list = ['Control', 'No_VIPcells', 'No_VIPCR', 'No_VIPCCK',
               'No_VIPPVM', 'No_VIPNVM', 'No_VIPCRtoBC', 'No_VIPCRtoOLM']

    rateMaps = {}
    pvals = {}
    infors = {}
    for case in my_list:
        pval_infor = []
        infor = []
        for ntrial in trials:
            with open(file_load+'/pickled_sn_'+case+'_'+ntrial+'.pkl', 'rb') as f:
                loaded_data = pickle.load(f)

            if ntrial == '1':
                rateMaps[case] = loaded_data['maps']
            else:
                rateMaps[case] = np.concatenate(
                    (rateMaps[case], loaded_data['maps']), axis=0)

            fileload_perms = spec+'/metrics_permutations/'+learning + \
                '/perms_pickled_info_'+case+'_Mouse_'+ntrial+'.pkl'
            with open(fileload_perms, 'rb') as f:
                infor_per = pickle.load(f)

            for npyr in range(Npyramidals):
                info = list(infor_per[npyr, :])
                pval_infor.append(sum(info[1:] < info[0])/float(Nperms))
                infor.append(info[0])

        pvals[case] = pval_infor
        infors[case] = infor

    nCells = rateMaps['Control'].shape[0]

    numbers_plc_all = {}
    numbers_rwd_all = {}
    plc_idx_all = {}
    rate_maps_plc = {}
    rateIn = {}
    rateOut = {}
    fieldsize = {}
    peakfield = {}

    for case in my_list:
        numbersALL = 0
        numbersrwdALL = 0
        plc_idx = []
        fsize = []
        ratein = []
        rateout = []
        peaks = []
        for npyr in range(nCells):

            rate_map = rateMaps[case][npyr, :, :]
            if np.argmax(rate_map) != -1:
                maxpeak = np.max(rate_map)
                fsmr = field_size(rate_map, relfreq=0.1 *
                                  maxpeak, track_length=Nbins)
                sizetest1 = fsmr[0]
                peakrate = peak_frequency(rate_map)
                mean_in = fsmr[1]
                mean_out = fsmr[2]

                lim1 = 8/(npath_x/Nbins)
                lim2 = 40/(npath_x/Nbins)
                if maxpeak >= 3.0 and lim1 <= sizetest1:
                    numbersALL += 1
                    plc_idx.append(npyr)
                    fsize.append(sizetest1*(npath_x/Nbins))
                    ratein.append(mean_in)
                    rateout.append(mean_out)
                    peaks.append(maxpeak)
                    if xrew1 <= np.argmax(rate_map) <= xrew2:
                        numbersrwdALL += 1

        plc_idx_all[case] = plc_idx
        numbers_plc_all[case] = numbersALL
        numbers_rwd_all[case] = numbersrwdALL
        rateIn[case] = ratein
        peakfield[case] = peaks

        rateOut[case] = rateout
        fieldsize[case] = fsize
        rate_maps_plc[case] = rateMaps[case][plc_idx_all[case], :, :]

        idx = np.argmax(rateMaps[case].squeeze(), axis=1)
        new_idx = np.lexsort((range(nCells), idx))
        rtMaps = rateMaps[case][new_idx, :, :].squeeze()

        Max = np.max(rtMaps, axis=1).reshape(-1, 1)
        for i in range(Max.shape[0]):
            if Max[i, 0] == 0:
                Max[i, 0] = 1e-12

        rtMaps = np.divide(rtMaps, Max)

        L = rate_maps_plc[case].shape[0]
        mapsplc = rate_maps_plc[case].squeeze()
        idx = np.argmax(mapsplc, axis=1)
        new_idx = np.lexsort((range(L), idx))
        rtMapsplc = rate_maps_plc[case][new_idx, :, :].squeeze()
        Max = np.max(rtMapsplc, axis=1).reshape(-1, 1)
        for i in range(Max.shape[0]):
            if Max[i, 0] == 0:
                Max[i, 0] = 1e-12

        rtMapsplc_norm = np.divide(rtMapsplc, Max)

        fig, axes = plt.subplots(nrows=1, ncols=1)

        im0 = axes.imshow(rtMapsplc_norm, cmap='jet', aspect='equal')
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axes.set_xlim((0, Nbins))
        axes.set_xticks(range(0, Nbins+1, Nbins/4))
        axes.set_xticklabels(
            ['-0.5', '-0.25', '0', '0.25', '0.5'], fontsize=11)
        axes.set_title(case, fontsize=14)
        axes.set_ylabel('# cell index', fontsize=12)
        axes.set_xlabel('position relative to reward', fontsize=12)
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
        plt.savefig(path_figs+learning+'/'+case +
                    '_PlaceCellsALL_heatmap.pdf', format='pdf', dpi=300)
        plt.cla()
        plt.clf()
        plt.close()
