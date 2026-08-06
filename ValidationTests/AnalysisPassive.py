import os
import numpy as np
from brian2 import nA, mV
from peakdet import peakdet
import matplotlib.pyplot as plt
import pickle


namecells = ['Pyramidal', 'AxoAxonic', 'Basket',
             'Bistratified', 'OLM', 'VIP_CCK', 'VIP_CR']

foldername = 'FiguresValidation/'
os.system('mkdir -p FiguresValidation')

Sag_ratio_all = {}
R_input_all = {}
V_rest_all = {}
t_mem_all = {}
spike_ampl_all = {}
Vthreshold_all = {}

Freq_plots = {}
Vpeak_plots = {}

for iname in namecells:

    fname = iname+'/PassiveProperties/'

    print('ANALYSIS OF: ' + iname)
    print
    # time parameters
    dt = 1.0/10  # ms
    t1 = 0
    tend = 2000  # ms

    time_vec = range(int(tend/dt) + 1)
    time_vec = [t*dt for t in time_vec]

    # inp = float(raw_input('Give Current Amplitude: '))
    # for Figure S7E
    I = [0.001*x for x in range(0, 410, 10)]

    Rin = []
    sag_ratio = []
    frequency = []
    V_peak = []
    t_mem = []
    Iinj = []
    Iinj2 = []
    spike_ampl = []
    Vthreshold = []
    counter = 0
    for i in I:
        Itext = '%.3f' % i

        V_soma = np.loadtxt(fname+'somaV_'+Itext+'.txt')

        if counter % 5 == 0:
            fig = plt.figure(1)
            ax = plt.subplot(111)
            t1 = int(470/dt)
            t2 = int(1700/dt)
            plt.plot(time_vec[t1:t2], V_soma[t1:t2],
                     label='Iinjected= '+str(i)+' nA')
            plt.xlabel('Time [ms]', fontsize=14)
            plt.ylabel('Voltage @ soma [mV]', fontsize=14)
            plt.title(iname, fontsize=14)
            plt.axis([0, 1800, -200, 80])

            zed = [tick.label.set_fontsize(14)
                   for tick in ax.yaxis.get_major_ticks()]
            zed = [tick.label.set_fontsize(14)
                   for tick in ax.xaxis.get_major_ticks()]
            ax.plot([5, 5], [0, 20], linewidth=1.6, color='blue')
            ax.plot([5, 105], [00, 00], linewidth=1.6, color='blue')
            plt.legend()

        thres = -5
        if iname == 'Pyramidal':
            thres = 5

        maxtab, mintab = peakdet(V_soma, 1, thres)

        duration = 1000
        start = int(499/dt)
        end = int((499+duration)/dt)

        if i == -0.1:
            DV = (V_soma[end] - V_soma[start]) * mV
            DVmax = (min(V_soma[start:end]) - V_soma[start]) * mV
            sag_ratio.append(DV/DVmax)
            Rin.append(DV/(i*nA))  # mega Ohm
            V_peak.append(V_soma[end]-V_soma[start])
            Iinj.append(i)
        elif i > 0:
            frequency.append(len(maxtab))
            Iinj2.append(i)
            if len(maxtab) == 0:
                Voltage = V_soma[start:end]
                V_peak.append(np.max(Voltage) - Voltage[0])
                Iinj.append(i)

                Voltage = V_soma[end:]
                Vmax = Voltage[0]
                Vtar = (Voltage - min(Voltage))/(Vmax-min(Voltage))
                tm = []
                epsilon = 1e-2
                factor = 1-1/np.exp(1)
                for iv in range(len(Voltage)):
                    if (factor-epsilon <= Vtar[0]-Vtar[iv] <= factor+epsilon):
                        tm.append(dt*iv)
                tm = np.mean(tm)
                t_mem.append(tm)
            elif len(maxtab) != 0:
                Voltage = V_soma[start:]
                dV = np.diff(Voltage)
                deriv = np.mean(dV) - 2*np.std(dV)
                for ivn in range(len(dV)):
                    if dV[ivn] > deriv:
                        Vthreshold.append(Voltage[iv+1])
                        break
            else:
                spike_ampl.append(max(V_soma))

        else:
            V_rest = V_soma[end]
            V_peak.append(V_soma[end]-V_soma[start])
            Iinj.append(i)
            frequency.append(len(maxtab))
            Iinj2.append(i)

        counter += 1

    plt.savefig(foldername+iname+'_voltage_traces.eps', format='eps', dpi=1200)
    plt.savefig(foldername+iname+'_voltage_traces.png', format='png', dpi=600)
    plt.cla()
    plt.clf()
    plt.close()

    Sag_ratio_all[iname] = sag_ratio
    R_input_all[iname] = Rin
    V_rest_all[iname] = V_rest
    t_mem_all[iname] = t_mem
    Vthreshold_all[iname] = Vthreshold

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(Iinj, V_peak, '.')
    Vpeak_plots[iname] = V_peak
    plt.xlabel('Iinj [nA]', fontsize=14)
    plt.ylabel('Peak voltage [mV]', fontsize=14)
    plt.title(iname, fontsize=14)
    m, b = np.polyfit(Iinj, V_peak, 1)
    ax.plot(Iinj, [m*x + b for x in Iinj], '--')
    plt.ylim(-100, 50)
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.get_label().set_rotation('horizontal')

    plt.savefig(foldername+iname+'_VI.eps', format='eps',
                bbox_inches='tight', dpi=1200)
    plt.savefig(foldername+iname+'_VI.png', format='png',
                bbox_inches='tight', dpi=600)
    plt.cla()
    plt.clf()
    plt.close()

    fig = plt.figure()
    ax = plt.subplot(111)
    Freq_plots[iname] = frequency
    ax.plot(Iinj2, frequency, '.-')
    plt.xlabel('Iinj [nA]', fontsize=14)
    plt.ylabel('Firing frequency [Hz]', fontsize=14)
    plt.title(iname, fontsize=14)
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.savefig(foldername+iname+'_FI.eps', format='eps',
                bbox_inches='tight', dpi=1200)
    plt.savefig(foldername+iname+'_FI.png', format='png',
                bbox_inches='tight', dpi=600)
    plt.cla()
    plt.clf()
    plt.close()

big_list = [Sag_ratio_all, R_input_all, V_rest_all,
            t_mem_all, spike_ampl_all, Vthreshold_all]
with open('pickled_sn_passive_properties.txt', 'wb') as handle:
    pickle.dump(big_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig = plt.figure(3)
ax = plt.subplot(111)

nn = 0
colors = ['blue', 'lightblue', 'brown',
          'purple', 'yellow', 'red', 'darkorange']
for iname in namecells:
    frequency = Freq_plots[iname]
    ax.plot(Iinj2, frequency, '-', label=iname + 'Cell', color=colors[nn])
    plt.xlabel('Iinj [nA]', fontsize=14)
    plt.ylabel('Firing frequency [Hz]', fontsize=14)
    plt.title(iname, fontsize=14)
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    nn += 1
plt.legend()
plt.savefig(foldername+'all_FI.eps', format='eps',
            bbox_inches='tight', dpi=1200)
plt.savefig(foldername+'all_FI.png', format='png',
            bbox_inches='tight', dpi=600)
