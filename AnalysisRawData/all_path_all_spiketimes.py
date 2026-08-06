#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:18:16 2018.

@author: spiros
"""
import sys
import pickle
import os
import numpy as np


def spiketimes_path_load(learning):

    fsave = 'data_analysis/metrics_permutations/'+learning+'/'
    os.system('mkdir -p '+fsave)

    my_list = ['Control', 'No_VIPcells', 'No_VIPCR', 'No_VIPCCK',
               'No_VIPPVM', 'No_VIPNVM', 'No_VIPCRtoBC', 'No_VIPCRtoOLM']

    nruns = 5
    trials = 7
    Npyramidals = 130
    for ntrial in range(1, trials+1):
        print
        print("TRIAL: "+str(ntrial))
        print
        spiketimes_all = {}
        path_all = []
        for nrun in range(1, nruns+1):
            print("RUN: "+str(nrun))

            # Load PATH and SPIKETIMES of Pyramidal npyr
            # Load of path -- different for each run
            if learning == 'prelearning':
                pathd = '../make_inputs_linear_track/runs_produced_by_python_prelearning/run_' + \
                    str(nrun)
            else:
                pathd = '../make_inputs_linear_track/runs_produced_by_python_speed_pos/run_' + \
                    str(nrun)
            path = np.loadtxt(pathd+'/path.txt', 'int', delimiter=',')
            path_all.append(path)

            Aspiketimesdict = {}
            for npyr in range(Npyramidals):
                Aspiketimes = {}
                for case in my_list:
                    cond = '../Simulation_Results/'+learning+'/'+case
                    fileload = cond+'/Trial_' + \
                        str(ntrial)+'/Run_'+str(nrun)+'/spiketimes_pvsoma_.pkl'

                    if os.path.isfile(fileload):
                        with open(fileload, 'rb') as f:
                            spiketimes_load = pickle.load(f)
                        spiketimes = spiketimes_load[npyr]
                        # remove first entry -- aka pyramidal number
                        spiketimes = spiketimes[1:]
                        spiketimes = [
                            x for x in spiketimes if x < path.shape[0]]
                    else:
                        print("File does not exist.")
                        continue

                    Aspiketimes[case] = spiketimes
                Aspiketimesdict['Pyramidal'+str(npyr)] = Aspiketimes
            spiketimes_all['Run'+str(nrun)] = Aspiketimesdict

        with open(fsave+'path_all_trial_'+str(ntrial)+'.pkl', 'wb') as handle:
            pickle.dump(path_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(fsave+'spiketimes_all_trial_'+str(ntrial)+'.pkl', 'wb') as handle:
            pickle.dump(spiketimes_all, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


learning = sys.argv[1]
spiketimes_path_load(learning)
