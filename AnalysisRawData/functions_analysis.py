import numpy as np


def spike_map(spiketimes, csum, npath_x, npath_y):
    """
    Make the Spike Matrix from spiketimes, path and time_array
    """
    Z = np.zeros((npath_x, npath_y))

    if len(spiketimes) != 0:
        for spike in spiketimes:
            # Find the last positive index --> this is the mapping from
            # time to space

            if spike > csum[-1]:
                continue
            else:
                idxs = np.argwhere((spike - csum) > 0)
                if idxs.shape[0] == 0:
                    idx = 0
                else:
                    idx = idxs.shape[0]

                Z[idx, :] += 1

    if Z.shape[0] != npath_x or Z.shape[1] != npath_y:
        print('Error in Z dimensions')

    return Z


def binning(a, N, method):
    """
    Bin a vector a in N parts
    """
    if not isinstance(N, int):
        raise ValueError('Binning size must be integer.')

    a = a.squeeze()
    L = a.shape[0]
    rem = L % N

    if rem != 0:
        raise ValueError('Not a valid binning size.')

    # find the step
    step = int(L/N)

    b = np.zeros(N)

    cnt = 0
    for i in range(0, L, step):
        if method == 'summing':
            b[cnt] = np.sum(a[i:i+step])
        elif method == 'mean':
            b[cnt] = np.mean(a[i:i+step])
        cnt += 1

    return b.reshape(-1, 1)
