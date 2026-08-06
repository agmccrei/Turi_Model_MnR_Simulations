#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 21:28:15 2017.

@author: spiros
"""

import numpy as np
import pickle
import sys
import os
import time
import scipy.ndimage.filters as flt
from functions_analysis import spike_map, binning


def analysis_path_cluster(ntrial, case, learning, inter):

    folder1 = 'data_analysis'

    os.system('mkdir -p '+folder1+'/figures_interneurons/')
    os.system('mkdir -p '+folder1+'/metrics_interneurons/')
    fdname1 = '/'+folder1+'/figures_interneurons/'
    fdname2 = '/'+folder1+'/metrics_interneurons/'

    cond = '../Simulation_Results/'+learning+'/'+case
    print("Analyse ... " + case + " trial "+ntrial + " " + learning)

    os.system('mkdir -p '+folder1+'/figures_interneurons/'+learning+'/')
    os.system('mkdir -p '+folder1+'/metrics_interneurons/'+learning+'/')
    # Define the map size!
    maindir = os.getcwd()

    # Give path dimensions
    # Give path dimensions
    npath_x = 200
    npath_y = 1
    # Number of pyramidal
    Ncells = 130
    Nbins = 100
    skernel = 3.0 / (npath_x/Nbins)
    runsAll = 5

    # Number of basket cells
    if inter == 'bcell':
        Ncells = 8
    elif inter == 'vipcck' or inter == 'vipcrnvm':
        Ncells = 1
    elif inter == 'vipcr':
        Ncells = 4
    elif inter == 'olm' or inter == 'aacell' or inter == 'bscell':
        Ncells = 2

    # 3-d matrix of all interneurons
    rateMaps = np.zeros((Ncells, Nbins, npath_y))
    rateMaps_unsmoothed = np.zeros((Ncells, Nbins, npath_y))
    time_array_in_bin = np.zeros((Ncells, Nbins, npath_y))

    # File location - pathfile
    fileload = folder1 + '/metrics_permutations/'+learning

    with open(fileload+'/path_all_trial_'+str(ntrial)+'.pkl', 'rb') as f:
        path_all = pickle.load(f)

    # Loop for all INs
    for ncell in range(Ncells):
        # A matrix for rate map
        Zall = np.zeros((Nbins, npath_y))
        time_array_all = np.zeros(Nbins*npath_y)

        for nrun in range(1, runsAll+1):

            # Load of path -- different for each run
            path = path_all[nrun-1]

            # make the time - space map
            time_array = np.bincount(path[:, 0])[1:]
            csum = np.cumsum(time_array)

            fileload = cond+'/Trial_' + \
                str(ntrial)+'/Run_'+str(nrun)+'/spiketimes_'+inter+'_.pkl'
            with open(fileload, 'rb') as f:
                spiketimes_all = pickle.load(f)
            spiketimes = spiketimes_all[ncell][1:]

            Z = spike_map(spiketimes, csum, npath_x, npath_y)
            # Take the sum over all runs given by total
            Zall += binning(Z, Nbins, 'summing')
            time_array_binned = binning(time_array, Nbins, 'summing').squeeze()

            # time spent in each bin in ms
            time_array_all += time_array_binned / 1000.0

        # Calculate the time spent in each bin
        time_array_smoothed = flt.gaussian_filter1d(
            time_array_all, sigma=skernel, mode='nearest', truncate=3.0)
        Zsmoothed = flt.gaussian_filter1d(
            Zall.squeeze(), sigma=skernel, mode='nearest', truncate=3.0)
        # convert to Hz, so divide with seconds
        # time ms/1000 (ms/sec) --> seconds
        Zmean = np.divide(Zsmoothed, time_array_smoothed)

        rateMaps_unsmoothed[int(ncell), :, :] = Zall
        rateMaps[int(ncell), :, :] = Zmean.reshape(-1, 1)
        time_array_in_bin[int(ncell), :, :] = time_array_all.reshape(-1, 1)

    print('\nDone with the rate maps')

    # ==============================================================================
    # ##################### RATE MAPS SAVING #################################
    # ==============================================================================
    filesave = maindir+fdname2+learning

    mydict = {}
    mydict['maps'] = rateMaps
    mydict['maps_unsmoothed'] = rateMaps_unsmoothed
    mydict['time_in_bin'] = time_array_in_bin
    if not os.path.exists(filesave):
        os.makedirs(filesave)

    with open(filesave+'/pickled_sn_'+inter+'_'+case+'_'+ntrial+'.pkl', 'wb') as handle:
        pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nDone with "+case+" analysis. Done with trial "+ntrial)


tic = time.time()
ntrial = sys.argv[1]
case = sys.argv[2]
learning = sys.argv[3]
inter = sys.argv[4]
results = analysis_path_cluster(ntrial, case, learning, inter)
toc = time.time()

print("\nTotal time: "+str(round(toc-tic, 3))+" seconds")
