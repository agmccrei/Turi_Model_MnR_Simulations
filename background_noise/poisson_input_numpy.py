import numpy as np
import os
import sys

nrun = int(sys.argv[1])
freq = int(sys.argv[2])

np.random.seed(nrun)

print("RUN: %d" % nrun)
foldername = "rate%d/run_%d" % (freq, nrun)
if not os.path.exists(foldername):
    os.makedirs(foldername)

N = 1000
duration_ms = 23000
rate_hz = freq
dt_ms = 0.1 # ms resolution

# Prob of spike in dt
p = (rate_hz * dt_ms) / 1000.0

for s in range(N):
    # Generate random numbers for each time step
    spikes = np.random.rand(int(duration_ms / dt_ms)) < p
    # Convert indices to timestamps
    spiketimes = np.where(spikes)[0] * dt_ms
    
    np.savetxt("%s/noise_%d.txt" % (foldername, s), spiketimes, fmt="%.1f")

print("Done generating %d noise files." % N)
