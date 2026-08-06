#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:18:33 2018.

@author: spiros
"""
import pickle
import os
from place_cell_metrics import field_size
import numpy as np

Npyramidals = 130
fnames = 'Simulation_Results/'

npath_x, npath_y = 200, 1
Nbins = 100

Nperms = 500
xrew1, xrew2 = 80/(npath_x/Nbins), 110/(npath_x/Nbins)+1
runsAll = 5
nTrials = 1

learning = 'locomotion'
print("\nLEARNING: ", learning)
print
print
spec = 'data_analysis'
path_figs = spec+'/figures_plasticity/'
os.system('mkdir -p ' + path_figs)
file_load = spec+'/metrics/'+learning
trials = [str(i) for i in range(1, nTrials+1)]
maindir = os.getcwd()


my_list = ['Control', 'No_VIPcells', 'No_VIPCR', 'No_VIPCCK',
           'No_VIPNVM', 'No_VIPPVM', 'No_VIPCRtoBC', 'No_VIPCRtoOLM']
my_list = ['Control']

for case in my_list:
    print("CASE:", case)
    for ntrial in trials:
        theoretical_cells = np.loadtxt(
            '../'+fnames+learning+'/'+case+'/Trial_'+ntrial+'/Run_1/input_conv2.txt', delimiter=',')
        theoretical_fields = [int(x) for x in list(theoretical_cells[:, 1])]
        theoretical_cells = [int(x) for x in list(theoretical_cells[:, 0])]
        print("TRIAL:", ntrial)

        with open(file_load+'/pickled_sn_'+case+'_'+ntrial+'.pkl', 'rb') as f:
            loaded_data = pickle.load(f)

        rateMaps = loaded_data['maps']

        indices_cells = []
        for npyr in range(Npyramidals):

            if npyr in theoretical_cells:
                rate_map = rateMaps[npyr, :, :]
                fmean = np.mean(rate_map[xrew1:xrew2])
                fmax = np.max(rate_map[xrew1:xrew2])
                sizetest1 = field_size(
                    rate_map[xrew1:xrew2], relfreq=0.10*fmax)[0]

                if (fmean > 0.8) and (sizetest1 >= 4.0):
                    indices_cells.append(npyr)

        np.savetxt('../'+fnames+learning+'/'+case+'/Trial_'+ntrial +
                   '/cells_weights_up.txt', indices_cells, fmt='%u', delimiter=' ', newline='\n')
