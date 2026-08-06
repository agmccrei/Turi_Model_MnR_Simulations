#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 21:28:15 2017.

@author: spiros
"""

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import time
import scipy.ndimage.filters as flt
from functions_analysis import spike_map, binning
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('agg')


def analysis_path_cluster(ntrial, case, learning):
    """

    Parameters
    ----------
    ntrial : MOUSE ID
        DESCRIPTION.
    case : STRING
        DESCRIPTION.
    learning : STRING
        DESCRIPTION.

    Returns
    -------
    None.

    """
    folder1 = 'data_analysis'

    os.system('mkdir -p '+folder1+'/figures/')
    os.system('mkdir -p '+folder1+'/metrics/')
    fdname1 = '/'+folder1+'/figures/'
    fdname2 = '/'+folder1+'/metrics/'

    print("Analyse ... " + case + " trial "+ntrial + " " + learning)

    os.system('mkdir -p '+folder1+'/figures/'+learning+'/')
    os.system('mkdir -p '+folder1+'/metrics/'+learning+'/')
    maindir = os.getcwd()

    # Give path dimensions
    npath_x = 200
    npath_y = 1
    # Number of pyramidal
    Ncells = 130
    Nbins = 100
    skernel = 3.0 / (npath_x/Nbins)
    runsAll = 5

    # Define the map size!
    rate_maps_all = np.zeros((Ncells, Nbins))
    # 3-d matrix of all pyramidals
    rateMaps = np.zeros((Ncells, Nbins, npath_y))
    rateMaps_unsmoothed = np.zeros((Ncells, Nbins, npath_y))
    time_array_in_bin = np.zeros((Ncells, Nbins, npath_y))

    # File location - pathfile
    fileload = folder1 + '/metrics_permutations/'+learning

    with open(fileload+'/path_all_trial_'+str(ntrial)+'.pkl', 'rb') as f:
        path_all = pickle.load(f)

    with open(fileload+'/spiketimes_all_trial_'+str(ntrial)+'.pkl', 'rb') as f:
        spiketimes_all = pickle.load(f)

    # Loop for all pyramidals
    for npyr in range(Ncells):

        # A matrix for rate map
        Zall = np.zeros((Nbins, npath_y))
        time_array_all = np.zeros(Nbins*npath_y)

        for nrun in range(1, runsAll+1):

            # Load of path -- different for each run
            path = path_all[nrun-1]

            # Make the time -space map
            time_array = np.bincount(path[:, 0])[1:]
            csum = np.cumsum(time_array)

            spiketimes = spiketimes_all['Run' +
                                        str(nrun)]['Pyramidal'+str(npyr)][case]

            Z = spike_map(spiketimes, csum, npath_x, npath_y)
            # Take the sum over all runs given by total
            Zall += binning(Z, Nbins, 'summing')
            time_array_binned = binning(time_array, Nbins, 'summing').squeeze()

            # time spent in each bin in ms
            time_array_all += time_array_binned / 1000.0

        # Calculate the time spent in each bin
        time_smoothed = flt.gaussian_filter1d(
            time_array_all, sigma=skernel, mode='nearest', truncate=3.0)
        Zsmoothed = flt.gaussian_filter1d(
            Zall.squeeze(), sigma=skernel, mode='nearest', truncate=3.0)
        # convert to Hz, so divide with seconds,
        # time ms/1000 (ms/sec) --> seconds
        Zmean = np.divide(Zsmoothed, time_smoothed)

        rateMaps_unsmoothed[int(npyr), :, :] = Zall
        rateMaps[int(npyr), :, :] = Zmean.reshape(-1, 1)
        time_array_in_bin[int(npyr), :, :] = time_array_all.reshape(-1, 1)

    print('\nDone with the rate maps')

    fig, axes = plt.subplots(nrows=13, ncols=10, figsize=(20, 20))
    nn = 0
    for ax in axes.flat:
        Max = np.max(rateMaps[nn, :, :])
        im = ax.imshow(rateMaps[nn, :, :].T/Max,
                       origin='lower', cmap="jet", aspect='10')
        ax.tick_params(axis='y', which='both', right='off',
                       left='off', labelleft='off')
        ax.title.set_text('PC'+str(nn) + '  ' + str(np.round(Max, 1)) + ' Hz')
        nn += 1

    fig.colorbar(im, ax=axes.ravel().tolist())

    if not os.path.exists(maindir+fdname1+learning+'/'):
        os.makedirs(maindir+fdname1+learning+'/')

    plt.savefig(maindir+fdname1+learning+'/'+case+'_' +
                str(ntrial)+'_heatmap.pdf', format='pdf', dpi=300)

    idx = np.argmax(rateMaps.squeeze(), axis=1)
    new_idx = np.lexsort((range(Ncells), idx))
    rtMaps = rateMaps[new_idx, :, :].squeeze()

    rate_maps_all = rtMaps

    fig = plt.subplots(figsize=(15, 15))
    ax = plt.gca()
    im = ax.imshow(rate_maps_all, cmap="jet", aspect='equal')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xlim((0, Nbins))
    ax.set_xticks(range(0, Nbins+1, 50 / (npath_x/Nbins)))
    ax.set_xticklabels(['-0.5', '-0.25', '0', '0.25', '0.5'], fontsize=13)
    ax.set_yticks(range(0, Ncells+1, 20))
    ax.set_yticklabels([str(x) for x in range(0, Ncells+1, 20)], fontsize=13)
    ax.set_title(case, fontsize=14)
    plt.savefig(maindir+fdname1+learning+'/'+case+'_'+str(ntrial) +
                '_heatmap_all_cells.pdf', format='pdf', dpi=300)

    # ========================================================================
    # ##################### RATE MAPS SAVING #################################
    # ========================================================================

    mydict = {}
    mydict['maps'] = rateMaps
    mydict['maps_unsmoothed'] = rateMaps_unsmoothed
    mydict['time_in_bin'] = time_array_in_bin
    filesave = maindir+fdname2+learning
    if not os.path.exists(filesave):
        os.makedirs(filesave)

    with open(filesave+'/pickled_sn_'+case+'_'+ntrial+'.pkl', 'wb') as handle:
        pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nDone with "+case+" analysis. Done with trial "+ntrial)


tic = time.time()
ntrial = sys.argv[1]
case = sys.argv[2]
learning = sys.argv[3]
results = analysis_path_cluster(ntrial, case, learning)
toc = time.time()

print("\nTotal time: "+str(round(toc-tic, 3))+" seconds")
