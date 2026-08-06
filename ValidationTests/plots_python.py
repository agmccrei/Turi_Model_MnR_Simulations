import matplotlib.pyplot as plt
import numpy as np
import os

print("SEP plot")
print

my_list = ['AxoAxonic', 'Basket', 'Bistratified', 'OLM', 'VIP_CCK', 'VIP_CR']

fname = 'SynapticProperties'
os.system('mkdir -p '+fname+'/plots/')

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(
    16, 16), sharey=True, sharex=True)
nn = 0
for ax in axes.flat:
    # SEP cell
    cell = my_list[nn]
    print(cell)
    print

    dt = 0.1
    t1 = int(900/dt)
    t2 = int(1600/dt)

    casper1 = np.loadtxt(fname+'/SEP_GABAAto'+cell+'_i_rec_soma.txt')
    casper1 = casper1[t1:t2]
    casper2 = np.loadtxt(fname+'/SEP_GABABto'+cell+'_i_rec_soma.txt')
    casper2 = casper2[t1:t2]

    casper1 -= np.min(casper1)
    casper2 -= np.min(casper2)

    # Make strings as floats with list comprehension and convert in pA
    casper1_pA = [1000*float(x) for x in casper1]
    casper2_pA = [1000*float(x) for x in casper2]

    a = range(1, len(casper1_pA)+1)  # time bins
    tb = [x*dt for x in a]

    ax.plot(tb, casper1_pA, label='SEP to '+cell+' GABAA IPSC')
    ax.plot(tb, casper2_pA, label='SEP to '+cell+' GABAB IPSC')
    ax.set_xticks(range(0, 601, 200))
    ax.set_xticklabels([str(x) for x in range(0, 601, 200)])
    ax.set_ylim(0, 20)
    ax.set_ylabel('Current (pA)')
    ax.set_xlabel('Time (ms)')
    ax.set_title(' current to '+cell, fontsize=14, fontweight='bold')
    ax.legend(frameon=False)

    ax.text(290, 0.65, r'1.0 pA', fontweight='bold')
    ax.text(420, 0.2, r'100 ms', fontweight='bold')

    ax.plot([400, 400], [0.5, 1.5], linewidth=1.6,
            color='black')  # 5 pA vertical or pA
    ax.plot([400, 500], [0.5, 0.5], linewidth=1.6, color='black')  # 10 ms
    nn += 1


plt.tight_layout()
plt.savefig(fname+'/plots/SEPtoCells_currents.pdf', format='pdf', dpi=600)
