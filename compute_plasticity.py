#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
compute_plasticity.py
---------------------
Reads the spikes.dat output from a completed simulation condition and
identifies which Pyramidal cells fired above a threshold mean firing rate.
Those cell GIDs are written to cells_potentiated.txt in the condition's
Simulation_Results directory. The NEXT condition then loads this file
to boost EC/CA3 synaptic weights for those potentiated cells.

Usage:
    python compute_plasticity.py <source_cond> <dest_cond> <n_neuron_str> <n_trials> <n_runs> [threshold_hz]

Example (Novelty -> CFC):
    python compute_plasticity.py 0 1 Control 0 0
Example (CFC -> Reintroduction):
    python compute_plasticity.py 1 2 Control 0 0
"""
import sys
import os
import numpy as np

# --- Args ---
src_cond     = int(sys.argv[1])    # source condition (0=Novelty, 1=CFC)
dst_cond     = int(sys.argv[2])    # destination condition (1=CFC, 2=Reintro)
n_neuron_str = sys.argv[3]         # e.g. 'Control'
n_trials     = int(sys.argv[4])
n_runs       = int(sys.argv[5])
n_runs_dst   = int(sys.argv[6])
threshold_hz = float(sys.argv[7]) if len(sys.argv) > 6 else 0.8   # Hz (matches original plasticity_indices.py: fmean > 0.8)

cond_labels = {0: '2D_box_Novelty', 1: '2D_box_CFC', 2: '2D_box_Reintroduction', 3: '2D_box_Neutral'}

src_dir = 'Simulation_Results/%s/%s/Trial_%d/Run_%d' % (
    cond_labels[src_cond], n_neuron_str, n_trials, n_runs)
dst_dir = 'Simulation_Results/%s/%s/Trial_%d/Run_%d' % (
    cond_labels[dst_cond], n_neuron_str, n_trials, n_runs_dst)

spike_file = src_dir + '/spikes.dat'
output_file = dst_dir + '/cells_potentiated.txt'

if not os.path.exists(spike_file):
    print("ERROR: Spike file not found: " + spike_file)
    sys.exit(1)

# Make sure dest dir exists (it may not yet since next condition hasn't run)
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# ---- Load spikes ----
# spikes.dat format: time  gid  (space separated)
data = np.loadtxt(spike_file)
if data.ndim == 1:
    data = data.reshape(1, -1)

# Separate into time and gid columns
spike_times = data[:, 0]
gids = data[:, 1].astype(int)

# Only count Pyramidal cells (GIDs 0..129 for npcell=130)
npcell = 130
pyr_mask = (gids >= 0) & (gids < npcell)
pyr_times = spike_times[pyr_mask]
pyr_gids  = gids[pyr_mask]

# Simulation duration in seconds (match HOC: TINIT=400, STARTDEL=500,
# THETA=125, duration=176 → SIMDUR ≈ 22.9 s active portion)
# Use total spikes / active window
sim_dur_s = 22.0

potentiated = []
for gid in range(npcell):
    n_spikes = np.sum(pyr_gids == gid)
    rate = n_spikes / sim_dur_s
    if rate >= threshold_hz:
        potentiated.append(gid)

print("Condition %d -> %d plasticity: %d / %d Pyr cells potentiated (threshold=%.2f Hz)" % (
    src_cond, dst_cond, len(potentiated), npcell, threshold_hz))

np.savetxt(output_file, potentiated, fmt='%d')
print("Saved potentiated cell list to: " + output_file)
