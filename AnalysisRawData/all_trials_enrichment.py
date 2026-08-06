#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:01:48 2017.

@author: spiros
"""
from place_cell_metrics import field_size
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def make_dicts(dict_all, dict_input, case):
    if isinstance(dict_input[case], int):
        if case in dict_input.keys():
            dict_all[case].append(dict_input[case])
        else:
            dict_all[case] = [dict_input[case]]

    if case in dict_all:
        dict_all[case] += dict_input[case]
    else:
        dict_all[case] = dict_input[case]

    return dict_all


def bar_plots(mydict, metric, learning, path_figs, baseline):
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    my_list = ['Control', 'No_VIPcells', 'No_VIPCR', 'No_VIPCCK',
               'No_VIPPVM', 'No_VIPNVM', 'No_VIPCRtoBC', 'No_VIPCRtoOLM']
    A = mydict
    A_means = [np.mean(A['Control']), np.mean(A['No_VIPcells']),
               np.mean(A['No_VIPCR']), np.mean(A['No_VIPCCK']),
               np.mean(A['No_VIPPVM']), np.mean(A['No_VIPNVM']),
               np.mean(A['No_VIPCRtoBC']), np.mean(A['No_VIPCRtoOLM'])]
    A_sems = [sem(A['Control']), sem(A['No_VIPcells']), sem(A['No_VIPCR']),
              sem(A['No_VIPCCK']), sem(A['No_VIPPVM']), sem(A['No_VIPNVM']),
              sem(A['No_VIPCRtoBC']), sem(A['No_VIPCRtoOLM'])]

    plt.figure(1, dpi=300)

    y = A_means
    labels = my_list
    N = len(y)
    x = range(N)

    pControl, pVIPcells, pVIPCR, pVIPCCK, pVIPPVM, pVIPNVM, pVIPtoBC, pVIPtoOLM = plt.bar(
        x, y, yerr=A_sems)

    pControl.set_facecolor('blue')
    pVIPcells.set_facecolor('green')
    pVIPCR.set_facecolor('yellow')
    pVIPCCK.set_facecolor('red')
    pVIPPVM.set_facecolor('lightblue')
    pVIPNVM.set_facecolor('lightgreen')
    pVIPtoBC.set_facecolor('yellowgreen')
    pVIPtoOLM.set_facecolor('darkred')

    plt.axhline(y=baseline, linestyle='--', linewidth=2)

    plt.plot()
    plt.xticks(x, labels)
    plt.ylabel(metric, fontsize=16)
    plt.title(learning)
    plt.ylim([0, 0.4])
    plt.savefig(path_figs+learning+'/'+metric +
                '_barplot.pdf', format='pdf', dpi=300)

    plt.cla()
    plt.clf()
    plt.close()

    # Make Boxplots
    A_list = [list(A['Control']), list(A['No_VIPcells']), list(A['No_VIPCR']),
              list(A['No_VIPCCK']), list(A['No_VIPPVM']),
              list(A['No_VIPNVM']), list(A['No_VIPCRtoBC']),
              list(A['No_VIPCRtoOLM'])]

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
    colors = ['blue', 'green', 'yellow', 'red',
              'lightblue', 'lightgreen', 'yellowgreen', 'darkred']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    for element in ['fliers', 'means', 'medians', 'caps']:
        plt.setp(bplot[element], color='black')

    plt.xticks(x, labels)
    plt.ylabel(metric, fontsize=16)
    plt.title(learning)

    plt.savefig(path_figs+learning+'/'+metric +
                '_boxplot.pdf', format='pdf', dpi=300)

    plt.cla()
    plt.clf()
    plt.close()


fnames = 'Simulation_Results/'

my_list = ['Control', 'No_VIPcells', 'No_VIPCR', 'No_VIPCCK',
           'No_VIPPVM', 'No_VIPNVM', 'No_VIPCRtoBC', 'No_VIPCRtoOLM']

npath_x, npath_y = 200, 1
xlim1, xlim2 = 80, 110
Nbins = 100

xrew1, xrew2 = xlim1/(npath_x/Nbins), xlim2/(npath_x/Nbins)+1

trialsAll = 10
Npyramidals = 130


everything = {}
for learning in ['prelearning', 'locomotion', 'reward']:

    print("\nLEARNING: ", learning)
    print
    print
    spec = 'data_analysis'
    path_figs = spec+'/figures/'
    file_load = spec+'/metrics/'+learning
    trials = [str(i) for i in range(1, trialsAll+1)]
    maindir = os.getcwd()

    numbers_all = {}
    numbers_rwd_all = {}

    for ntrial in trials:
        print("TRIAL:", ntrial)
        rateMaps = {}
        numbers_plc = {}
        numbers_rwd = {}

        for case in my_list:
            with open(file_load+'/pickled_sn_'+case+'_'+ntrial+'.pkl', 'rb') as f:
                loaded_data = pickle.load(f)

            rateMaps[case] = loaded_data['maps']

            numbersALL = 0
            numbersrwdALL = 0
            for npyr in range(Npyramidals):

                rate_map = rateMaps[case][npyr, :, :]
                maxpeak = np.max(rate_map)
                sizetest1 = field_size(
                    rate_map, relfreq=0.1*maxpeak, track_length=Nbins)[0]
                maxpeak = np.max(rate_map)

                lim1 = 8/(npath_x/Nbins)
                lim2 = 40/(npath_x/Nbins)
                if maxpeak >= 3.0 and lim1 <= sizetest1:
                    numbersALL += 1

                    if xrew1 <= np.argmax(rate_map) <= xrew2:
                        numbersrwdALL += 1

            numbers_plc[case] = numbersALL
            numbers_rwd[case] = numbersrwdALL

            # END of loop in different cases ####

            if case in numbers_all.keys():
                numbers_all[case].append(numbers_plc[case])
            else:
                numbers_all[case] = [numbers_plc[case]]

            if case in numbers_rwd_all.keys():
                numbers_rwd_all[case].append(
                    numbers_rwd[case]/float(numbers_plc[case]))
            else:
                numbers_rwd_all[case] = [
                    numbers_rwd[case]/float(numbers_plc[case])]

    mydict_all = {}
    mydict_all['numbers_plc'] = numbers_all
    mydict_all['numbers_rwd'] = numbers_rwd_all

    everything[learning] = mydict_all

baseline = 1.0/6.0
A1 = {}
A2 = {}
A3 = {}
for case in my_list:
    A = []
    for learning in ['prelearning', 'locomotion', 'reward']:

        A.append(everything[learning]['numbers_rwd'][case])

    A1[case] = [x for x in A[0]]
    A2[case] = [x for x in A[1]]
    A3[case] = [x for x in A[2]]

learning = ['prelearning', 'locomotion', 'reward']
N = len(my_list)
pre_means = [np.mean(A1[case]) for case in my_list]
pre_sems = [sem(A1[case]) for case in my_list]

ind = np.arange(N)  # the x locations for the groups
width = 0.25       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, pre_means, width, color='r', yerr=pre_sems)

loc_means = [np.mean(A2[case]) for case in my_list]
loc_sems = [sem(A2[case]) for case in my_list]
rects2 = ax.bar(ind+width, loc_means, width, color='y', yerr=loc_sems)

rwd_means = [np.mean(A3[case]) for case in my_list]
rwd_sems = [sem(A3[case]) for case in my_list]
rects3 = ax.bar(ind+2*width, rwd_means, width, color='b', yerr=rwd_sems)

# add some text for labels, title and axes ticks
ax.set_ylabel('Enrichment (%)')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(tuple(my_list), rotation='vertical')

ax.legend((rects1[0], rects2[0], rects3[0]),
          ('Before Learning', 'Locomotion', 'Reward'))

plt.axhline(y=baseline, linestyle='--', linewidth=2)
plt.savefig('All.pdf', format='pdf', dpi=600)

pre = [A1[case] for case in my_list]
loc = [A2[case] for case in my_list]
rwd = [A3[case] for case in my_list]
