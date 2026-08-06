import numpy as np
import matplotlib.pyplot as plt
import sys
import os

dt = 0.1
t1 = 8970  # stimulation at time 900ms (tstim = 9000*dt )
# PC: 10500,  ACC: 10500, BC: 10500
# BSC: 10500, OLM: 10500, VIPCCK: 10500 , VIPCR: 10500
t2 = 10500

# select the target cell
# EC_CA3_to_PC or Pyramidal or Axoaxonic or Basket or Bistratified
# or OLM or VIPCCK or VIPCR
cell = sys.argv[1]

fname = 'SynapticProperties'

os.system('mkdir -p '+fname+'/plots/')
#               ___    ___
############## |___|  |    ################
############## |      |    #################
############## |      |___ #################

if cell == 'EC_CA3_to_PC':
    print("EC and CA3 ampa/nmda to Pyramidal")

    # PC cell
    with open(fname+'/ampaECtoPC_i_rec_soma.txt', 'r', 0) as f1:
        vec1 = f1.readlines()
    with open(fname+'/ampaCA3toapicalPC_i_rec_soma.txt', 'r', 0) as f2:
        vec2 = f2.readlines()
    with open(fname+'/ampaCA3tobasalPC_i_rec_soma.txt', 'r', 0) as f3:
        vec3 = f3.readlines()
    with open(fname+'/nmdaECtoPC_i_rec_soma.txt', 'r', 0) as f4:
        vec4 = f4.readlines()
    with open(fname+'/nmdaCA3toapicalPC_i_rec_soma.txt', 'r', 0) as f5:
        vec5 = f5.readlines()
    with open(fname+'/nmdaCA3tobasalPC_i_rec_soma.txt', 'r', 0) as f6:
        vec6 = f6.readlines()

    # Make strings as floats with list comprehension
    vec1 = [float(x1) for x1 in vec1]
    vec2 = [float(x2) for x2 in vec2]
    vec3 = [float(x3) for x3 in vec3]
    vec4 = [float(x4) for x4 in vec4]
    vec5 = [float(x5) for x5 in vec5]
    vec6 = [float(x6) for x6 in vec6]

    # To convert in pA
    vec1_pA = [1000*x1 for x1 in vec1]
    vec2_pA = [1000*x2 for x2 in vec2]
    vec3_pA = [1000*x3 for x3 in vec3]
    vec4_pA = [1000*x4 for x4 in vec4]
    vec5_pA = [1000*x5 for x5 in vec5]
    vec6_pA = [1000*x6 for x6 in vec6]

    t2 = t1+5000  # more time because of slow kinetics of NMDA

    M1 = vec1_pA[t1]
    M2 = vec2_pA[t1]
    M3 = vec3_pA[t1]

    M4 = vec4_pA[t1]
    M5 = vec5_pA[t1]
    M6 = vec6_pA[t1]

    vec1n_pA = [x1-M1 for x1 in vec1_pA]
    vec2n_pA = [x2-M2 for x2 in vec2_pA]
    vec3n_pA = [x3-M3 for x3 in vec3_pA]

    vec4n_pA = [x4-M4 for x4 in vec4_pA]
    vec5n_pA = [x5-M5 for x5 in vec5_pA]
    vec6n_pA = [x6-M6 for x6 in vec6_pA]

    a = range(1, len(vec1_pA)+1)  # time bins
    tb = [x*dt for x in a]

    fig = plt.figure(1, figsize=(10, 10))

    plt.plot(tb[t1:t2], vec1n_pA[t1:t2], label='EC AMPA EPSC')
    plt.plot(tb[t1:t2], vec2n_pA[t1:t2], label='CA3 AMPA EPSC (apical)')
    plt.plot(tb[t1:t2], vec3n_pA[t1:t2], label='CA3 AMPA EPSC (basal)')
    plt.plot(tb[t1:t2], vec4n_pA[t1:t2], label='EC NMDA EPSC')
    plt.plot(tb[t1:t2], vec5n_pA[t1:t2], label='CA3 NMDA EPSC (apical)')
    plt.plot(tb[t1:t2], vec6n_pA[t1:t2], label='CA3 NMDA EPSC (basal)')

    mn = min(vec3n_pA[t1:t2])  # min epsc/ipsc
    mx = max(vec1n_pA[t1:t2])  # max epsc/ipsc
    rg = int(mx-mn)/10
    rg = rg - rg % 10

    plt.plot([1030, 1030], [mn+5, mn+5+1], linewidth=1.6,
             color='black')  # 5 pA vertical
    plt.plot([1030, 1130], [mn+5, mn+5], linewidth=1.6, color='black')  # 10 ms

    plt.text(1050, mn+5.2, r'1 pA', fontweight='bold')
    plt.text(1052, mn+4.2, r'100 ms', fontweight='bold')

    plt.ylabel('Current (pA)')
    plt.xlabel('Time (ms)')
    plt.suptitle('Post-synaptic ampa & nmda currents from EC & CA3 to Pyramidal',
                 fontsize=14, fontweight='bold')
    plt.legend(frameon=False)

    plt.savefig(fname+'/plots/EC_CA3_AMPA_NMDAtoPC_currents.pdf',
                format='pdf', dpi=600)

    # plt.show()

elif cell == 'Pyramidal':
    print("Pyramidal Cell plot")

    # PC cell
    with open(fname+'/BCtoPC_i_rec_soma.txt', 'r', 0) as f1:
        vec1 = f1.readlines()
    with open(fname+'/VIPCCKtoPC_i_rec_soma.txt', 'r', 0) as f2:
        vec2 = f2.readlines()
    with open(fname+'/AACtoPC_i_rec_soma.txt', 'r', 0) as f3:
        vec3 = f3.readlines()
    with open(fname+'/gabaaOLMtoPC_i_rec_soma.txt', 'r', 0) as f4:
        vec4 = f4.readlines()
    with open(fname+'/gabaaBSCtoPC_i_rec_soma.txt', 'r', 0) as f5:
        vec5 = f5.readlines()
    with open(fname+'/gabaaBSCtobasalPC_i_rec_soma.txt', 'r', 0) as f6:
        vec6 = f6.readlines()
    with open(fname+'/PCtoPC_i_rec_soma.txt', 'r', 0) as f7:
        vec7 = f7.readlines()

    # Make strings as floats with list comprehension
    vec1 = [float(x1) for x1 in vec1]
    vec2 = [float(x2) for x2 in vec2]
    vec3 = [float(x3) for x3 in vec3]
    vec4 = [float(x4) for x4 in vec4]
    vec5 = [float(x5) for x5 in vec5]
    vec6 = [float(x6) for x6 in vec6]
    vec7 = [float(x7) for x7 in vec7]

    # To convert in pA
    vec1_pA = [1000*x1 for x1 in vec1]
    vec2_pA = [1000*x2 for x2 in vec2]
    vec3_pA = [1000*x3 for x3 in vec3]
    vec4_pA = [1000*x4 for x4 in vec4]
    vec5_pA = [1000*x5 for x5 in vec5]
    vec6_pA = [1000*x6 for x6 in vec6]
    vec7_pA = [1000*x7 for x7 in vec7]

    a = range(1, len(vec1_pA)+1)  # time bins
    tb = [x*dt for x in a]

    fig = plt.figure(1, figsize=(10, 10))

    plt.plot(tb[t1:t2], vec1_pA[t1:t2], label='BC IPSC')
    plt.plot(tb[t1:t2], vec2_pA[t1:t2], label='VIPCCK IPSC')
    plt.plot(tb[t1:t2], vec3_pA[t1:t2], label='ACC IPSC')
    plt.plot(tb[t1:t2], vec4_pA[t1:t2], label='GABAA OLM IPSC')
    plt.plot(tb[t1:t2], vec5_pA[t1:t2], label='GABAA BSC (apical) IPSC')
    plt.plot(tb[t1:t2], vec6_pA[t1:t2], label='GABAA BSC (basal) IPSC')
    plt.plot(tb[t1:t2], vec7_pA[t1:t2], label='PC EPSC')

    mn = min(vec7_pA[t1:t2])  # min epsc/ipsc
    mx = max(vec3_pA[t1:t2])  # max epsc/ipsc
    rg = int(mx-mn)/10
    rg = rg - rg % 10

    plt.plot([1000, 1000], [mn+150, mn+150+20],
             linewidth=1.6, color='black')  # 5 pA vertical
    plt.plot([1000, 1010], [mn+150, mn+150],
             linewidth=1.6, color='black')  # 10 ms

    plt.text(985, mn+154, r'20 pA', fontweight='bold')
    plt.text(1002, mn+140, r'10 ms', fontweight='bold')

    plt.ylabel('Current (pA)')
    plt.xlabel('Time (ms)')
    plt.suptitle('      Post-synaptic currents from all cells except EC & CA3 ',
                 fontsize=14, fontweight='bold')
    plt.legend(frameon=False)

    plt.savefig(fname+'/plots/all_except_ec_ca3_toPC_currents.pdf',
                format='pdf', dpi=600)

    # plt.sh1ow()


#                           ___
############## /\     /\   |    ##############
############# /__\   /__\  |    #################
############ /    \ /    \ |___ #################


elif cell == 'Axoaxonic':
    print("Axoaxonic Cell plot")

    # AAC cell
    with open(fname+'/ECtoAAC_i_rec_soma.txt', 'r', 0) as f1:
        vec1 = f1.readlines()
    with open(fname+'/CA3toAAC_i_rec_soma.txt', 'r', 0) as f2:
        vec2 = f2.readlines()
    with open(fname+'/PCtoAAC_i_rec_soma.txt', 'r', 0) as f3:
        vec3 = f3.readlines()
    with open(fname+'/BCtoAAC_i_rec_soma.txt', 'r', 0) as f4:
        vec4 = f4.readlines()
    with open(fname+'/BSCtoAAC_i_rec_soma.txt', 'r', 0) as f5:
        vec5 = f5.readlines()
    with open(fname+'/OLMtoAAC_i_rec_soma.txt', 'r', 0) as f6:
        vec6 = f6.readlines()
    with open(fname+'/VIPCCKtoAAC_i_rec_soma.txt', 'r', 0) as f7:
        vec7 = f7.readlines()

    # Make strings as floats with list comprehension
    vec1 = [float(x1) for x1 in vec1]
    vec2 = [float(x2) for x2 in vec2]
    vec3 = [float(x3) for x3 in vec3]
    vec4 = [float(x4) for x4 in vec4]
    vec5 = [float(x5) for x5 in vec5]
    vec6 = [float(x6) for x6 in vec6]
    vec7 = [float(x7) for x7 in vec7]

    # To convert in pA
    vec1_pA = [1000*x1 for x1 in vec1]
    vec2_pA = [1000*x2 for x2 in vec2]
    vec3_pA = [1000*x3 for x3 in vec3]
    vec4_pA = [1000*x4 for x4 in vec4]
    vec5_pA = [1000*x5 for x5 in vec5]
    vec6_pA = [1000*x6 for x6 in vec6]
    vec7_pA = [1000*x7 for x7 in vec7]

    a = range(1, len(vec1_pA)+1)  # time bins
    tb = [x*dt for x in a]

    fig = plt.figure(1, figsize=(10, 10))

    plt.plot(tb[t1:t2], vec1_pA[t1:t2], label='EC EPSC')
    plt.plot(tb[t1:t2], vec2_pA[t1:t2], label='CA3 EPSC')
    plt.plot(tb[t1:t2], vec3_pA[t1:t2], label='PC EPSC')
    plt.plot(tb[t1:t2], vec4_pA[t1:t2], label='BC IPSC')
    plt.plot(tb[t1:t2], vec5_pA[t1:t2], label='BSC IPSC')
    plt.plot(tb[t1:t2], vec6_pA[t1:t2], label='OLM IPSC')
    plt.plot(tb[t1:t2], vec7_pA[t1:t2], label='VIPCCK IPSC')

    mn = min(vec2_pA[t1:t2])
    mx = max(vec7_pA[t1:t2])
    rg = int(mx-mn)/10
    rg = rg - rg % 10

    plt.text(1025, mn+2.2, r'1 pA', fontweight='bold')
    plt.text(1042, mn+1.5, r'10 ms', fontweight='bold')

    plt.plot([1040, 1040], [mn+2, mn+2 + 1],
             linewidth=1.6, color='black')  # 1 pA vertical
    plt.plot([1040, 1050], [mn+2, mn+2], linewidth=1.6, color='black')  # 10 ms

    plt.ylabel('Current (pA)')
    plt.xlabel('Time (ms)')
    plt.suptitle('      Post-synaptic currents to AAC ',
                 fontsize=14, fontweight='bold')
    plt.legend(frameon=False)

    plt.savefig(fname+'/plots/toAAC_post_synaptic_currents.pdf',
                format='pdf', dpi=600)

    plt.show()

#              ___    ___
############# |   |  |    #################
############# |~~~   |    #################
############# |___|  |___ #################


elif cell == "Basket":
    print("Basket Cell plot")

    # BC cell
    with open(fname+'/ECtoBC_i_rec_soma.txt', 'r', 0) as f1:
        vec1 = f1.readlines()
    with open(fname+'/CA3toBC_i_rec_soma.txt', 'r', 0) as f2:
        vec2 = f2.readlines()
    with open(fname+'/PCtoBC_i_rec_soma.txt', 'r', 0) as f3:
        vec3 = f3.readlines()
    with open(fname+'/BCtoBC_i_rec_soma.txt', 'r', 0) as f4:
        vec4 = f4.readlines()
    with open(fname+'/BSCtoBC_i_rec_soma.txt', 'r', 0) as f5:
        vec5 = f5.readlines()
    with open(fname+'/OLMtoBC_i_rec_soma.txt', 'r', 0) as f6:
        vec6 = f6.readlines()
    with open(fname+'/VIPCRtoBC_i_rec_soma.txt', 'r', 0) as f7:
        vec7 = f7.readlines()

    # Make strings as floats with list comprehension
    vec1 = [float(x1) for x1 in vec1]
    vec2 = [float(x2) for x2 in vec2]
    vec3 = [float(x3) for x3 in vec3]
    vec4 = [float(x4) for x4 in vec4]
    vec5 = [float(x5) for x5 in vec5]
    vec6 = [float(x6) for x6 in vec6]
    vec7 = [float(x7) for x7 in vec7]

    # To convert in pA
    vec1_pA = [1000*x1 for x1 in vec1]
    vec2_pA = [1000*x2 for x2 in vec2]
    vec3_pA = [1000*x3 for x3 in vec3]
    vec4_pA = [1000*x4 for x4 in vec4]
    vec5_pA = [1000*x5 for x5 in vec5]
    vec6_pA = [1000*x6 for x6 in vec6]
    vec7_pA = [1000*x7 for x7 in vec7]

    a = range(1, len(vec1_pA)+1)  # time bins
    tb = [x*dt for x in a]

    fig = plt.figure(1, figsize=(10, 10))

    plt.plot(tb[t1:t2], vec1_pA[t1:t2], label='EC EPSC')
    plt.plot(tb[t1:t2], vec2_pA[t1:t2], label='CA3 EPSC')
    plt.plot(tb[t1:t2], vec3_pA[t1:t2], label='PC EPSC')
    plt.plot(tb[t1:t2], vec4_pA[t1:t2], label='BC IPSC')
    plt.plot(tb[t1:t2], vec5_pA[t1:t2], label='BSC IPSC')
    plt.plot(tb[t1:t2], vec6_pA[t1:t2], label='OLM IPSC')
    plt.plot(tb[t1:t2], vec7_pA[t1:t2], label='VIPCR IPSC')

    mn = min(vec3_pA[t1:t2])  # min PC
    mx = max(vec7_pA[t1:t2])  # max VIPCR
    rg = int(mx-mn)/10
    rg = rg - rg % 10

    plt.text(1015, mn+105, r'60 pA', fontweight='bold')
    plt.text(1032, mn+65, r'10 ms', fontweight='bold')

    plt.plot([1030, 1030], [mn+90, mn+90 + rg],
             linewidth=1.6, color='black')  # 60 pA vertical
    plt.plot([1030, 1040], [mn+90, mn+90],
             linewidth=1.6, color='black')  # 10 ms

    plt.ylabel('Current (pA)')
    plt.xlabel('Time (ms)')
    plt.suptitle('      Post-synaptic currents to BC ',
                 fontsize=14, fontweight='bold')
    plt.legend(frameon=False)

    plt.savefig(fname+'/plots/toBC_post_synaptic_currents.pdf',
                format='pdf', dpi=600)

    plt.show()

#              ___    __    ___
############# |   |  |__   |    ###########
############# |---      |  |    #################
############# |___|  ___|  |___ #################


elif cell == "Bistratified":
    print("Bistratified Cell plot")

    # BC cell
    with open(fname+'/ECtoBSC_i_rec_soma.txt', 'r', 0) as f1:
        vec1 = f1.readlines()
    with open(fname+'/CA3toBSC_i_rec_soma.txt', 'r', 0) as f2:
        vec2 = f2.readlines()
    with open(fname+'/PCtoBSC_i_rec_soma.txt', 'r', 0) as f3:
        vec3 = f3.readlines()
    with open(fname+'/BCtoBSC_i_rec_soma.txt', 'r', 0) as f4:
        vec4 = f4.readlines()
    with open(fname+'/BSCtoBSC_i_rec_soma.txt', 'r', 0) as f5:
        vec5 = f5.readlines()
    with open(fname+'/OLMtoBSC_i_rec_soma.txt', 'r', 0) as f6:
        vec6 = f6.readlines()
    with open(fname+'/VIPCCKtoBSC_i_rec_soma.txt', 'r', 0) as f7:
        vec7 = f7.readlines()

    # Make strings as floats with list comprehension
    vec1 = [float(x1) for x1 in vec1]
    vec2 = [float(x2) for x2 in vec2]
    vec3 = [float(x3) for x3 in vec3]
    vec4 = [float(x4) for x4 in vec4]
    vec5 = [float(x5) for x5 in vec5]
    vec6 = [float(x6) for x6 in vec6]
    vec7 = [float(x7) for x7 in vec7]

    # To convert in pA
    vec1_pA = [1000*x1 for x1 in vec1]
    vec2_pA = [1000*x2 for x2 in vec2]
    vec3_pA = [1000*x3 for x3 in vec3]
    vec4_pA = [1000*x4 for x4 in vec4]
    vec5_pA = [1000*x5 for x5 in vec5]
    vec6_pA = [1000*x6 for x6 in vec6]
    vec7_pA = [1000*x7 for x7 in vec7]

    a = range(1, len(vec1_pA)+1)  # time bins
    tb = [x*dt for x in a]

    fig = plt.figure(1, figsize=(10, 10))

    plt.plot(tb[t1:t2], vec1_pA[t1:t2], label='EC EPSC')
    plt.plot(tb[t1:t2], vec2_pA[t1:t2], label='CA3 EPSC')
    plt.plot(tb[t1:t2], vec3_pA[t1:t2], label='PC EPSC')
    plt.plot(tb[t1:t2], vec4_pA[t1:t2], label='BC IPSC')
    plt.plot(tb[t1:t2], vec5_pA[t1:t2], label='BSC IPSC')
    plt.plot(tb[t1:t2], vec6_pA[t1:t2], label='OLM IPSC')
    plt.plot(tb[t1:t2], vec7_pA[t1:t2], label='VIPCR IPSC')

    mn = min(vec3_pA[t1:t2])  # min PC
    mx = max(vec4_pA[t1:t2])  # max VIPCR
    rg = int(mx-mn)/10
    rg = rg - rg % 10

    plt.text(1005, mn+53, r'10 pA', fontweight='bold')
    plt.text(1023, mn+47, r'10 ms', fontweight='bold')

    plt.plot([1020, 1020], [mn+50, mn+50 + rg], linewidth=1.6,
             color='black')  # 10 pA vertical or pA
    plt.plot([1020, 1030], [mn+50, mn+50],
             linewidth=1.6, color='black')  # 10 ms

    plt.ylabel('Current (pA)')
    plt.xlabel('Time (ms)')
    plt.suptitle('      Post-synaptic currents to BSC ',
                 fontsize=14, fontweight='bold')
    plt.legend(frameon=False)

    plt.savefig(fname+'/plots/toBSC_post_synaptic_currents.pdf',
                format='pdf', dpi=600)

    plt.show()


#              ___
############# |   | |     |\  /| ###########
############# |   | |     | \/ | ##############
############# |___| |____ |    | #################


elif cell == "OLM":
    print("OLM Cell plot")

    # BC cell
    with open(fname+'/CA3toOLM_i_rec_soma.txt', 'r', 0) as f1:
        vec1 = f1.readlines()
    with open(fname+'/PCtoOLM_i_rec_soma.txt', 'r', 0) as f2:
        vec2 = f2.readlines()
    with open(fname+'/BSCtoOLM_i_rec_soma.txt', 'r', 0) as f3:
        vec3 = f3.readlines()
    with open(fname+'/OLMtoOLM_i_rec_soma.txt', 'r', 0) as f4:
        vec4 = f4.readlines()
    with open(fname+'/VIPCRtoOLM_i_rec_soma.txt', 'r', 0) as f5:
        vec5 = f5.readlines()

    # Make strings as floats with list comprehension
    vec1 = [float(x1) for x1 in vec1]
    vec2 = [float(x2) for x2 in vec2]
    vec3 = [float(x3) for x3 in vec3]
    vec4 = [float(x4) for x4 in vec4]
    vec5 = [float(x5) for x5 in vec5]

    # To convert in pA
    vec1_pA = [1000*x1 for x1 in vec1]
    vec2_pA = [1000*x2 for x2 in vec2]
    vec3_pA = [1000*x3 for x3 in vec3]
    vec4_pA = [1000*x4 for x4 in vec4]
    vec5_pA = [1000*x5 for x5 in vec5]

    a = range(1, len(vec1_pA)+1)  # time bins
    tb = [x*dt for x in a]

    fig = plt.figure(1, figsize=(10, 10))

    plt.plot(tb[t1:t2], vec1_pA[t1:t2], label='CA3 EPSC')
    plt.plot(tb[t1:t2], vec2_pA[t1:t2], label='PC EPSC')
    plt.plot(tb[t1:t2], vec3_pA[t1:t2], label='BSC IPSC')
    plt.plot(tb[t1:t2], vec4_pA[t1:t2], label='OLM IPSC')
    plt.plot(tb[t1:t2], vec5_pA[t1:t2], label='VIPCR IPSC')

    mn = min(vec2_pA[t1:t2])  # min PC
    mx = max(vec5_pA[t1:t2])  # max VIPCR
    rg = int(mx-mn)/10
    rg = rg - rg % 10

    plt.text(1011, mn+4.5, r'1 pA', fontweight='bold')
    plt.text(1028, mn+3.5, r'10 ms', fontweight='bold')

    plt.plot([1025, 1025], [mn+4, mn+4 + 1], linewidth=1.6,
             color='black')  # 1 pA vertical or pA
    plt.plot([1025, 1035], [mn+4, mn+4], linewidth=1.6, color='black')  # 10 ms

    plt.ylabel('Current (pA)')
    plt.xlabel('Time (ms)')
    plt.suptitle('      Post-synaptic currents to OLM ',
                 fontsize=14, fontweight='bold')
    plt.legend(frameon=False)

    plt.savefig(fname+'/plots/toOLM_post_synaptic_currents.pdf',
                format='pdf', dpi=600)

    plt.show()


#                    ___     ___   ___
########### \    / | |___|  |     |     |  / ###########
############ \  /  | |      |     |     |<   ##############
############# \/   | |      |___  |___  |  \ ################

elif cell == "VIPCCK":
    print("VIPCCK Cell plot")

    # VIPCCK cell
    with open(fname+'/ECtoVIPCCK_i_rec_soma.txt', 'r', 0) as f1:
        vec1 = f1.readlines()
    with open(fname+'/CA3toVIPCCK_i_rec_soma.txt', 'r', 0) as f2:
        vec2 = f2.readlines()
    with open(fname+'/PCtoVIPCCK_i_rec_soma.txt', 'r', 0) as f3:
        vec3 = f3.readlines()
    with open(fname+'/BCtoVIPCCK_i_rec_soma.txt', 'r', 0) as f4:
        vec4 = f4.readlines()
    with open(fname+'/BSCtoVIPCCK_i_rec_soma.txt', 'r', 0) as f5:
        vec5 = f5.readlines()
    with open(fname+'/OLMtoVIPCCK_i_rec_soma.txt', 'r', 0) as f6:
        vec6 = f6.readlines()
    with open(fname+'/VIPCCKtoVIPCCK_i_rec_soma.txt', 'r', 0) as f7:
        vec7 = f7.readlines()

    # Make strings as floats with list comprehension
    vec1 = [float(x1) for x1 in vec1]
    vec2 = [float(x2) for x2 in vec2]
    vec3 = [float(x3) for x3 in vec3]
    vec4 = [float(x4) for x4 in vec4]
    vec5 = [float(x5) for x5 in vec5]
    vec6 = [float(x6) for x6 in vec6]
    vec7 = [float(x7) for x7 in vec7]

    # To convert in pA
    vec1_pA = [1000*x1 for x1 in vec1]
    vec2_pA = [1000*x2 for x2 in vec2]
    vec3_pA = [1000*x3 for x3 in vec3]
    vec4_pA = [1000*x4 for x4 in vec4]
    vec5_pA = [1000*x5 for x5 in vec5]
    vec6_pA = [1000*x6 for x6 in vec6]
    vec7_pA = [1000*x7 for x7 in vec7]

    a = range(1, len(vec1_pA)+1)  # time bins
    tb = [x*dt for x in a]

    fig = plt.figure(1, figsize=(10, 10))

    plt.plot(tb[t1:t2], vec1_pA[t1:t2], label='EC EPSC')
    plt.plot(tb[t1:t2], vec2_pA[t1:t2], label='CA3 EPSC')
    plt.plot(tb[t1:t2], vec3_pA[t1:t2], label='PC EPSC')
    plt.plot(tb[t1:t2], vec4_pA[t1:t2], label='BC IPSC')
    plt.plot(tb[t1:t2], vec5_pA[t1:t2], label='BSC IPSC')
    plt.plot(tb[t1:t2], vec6_pA[t1:t2], label='OLM IPSC')
    plt.plot(tb[t1:t2], vec7_pA[t1:t2], label='VIPCR IPSC')

    mn = min(vec3_pA[t1:t2])  # min epsc (PC)
    mx = max(vec4_pA[t1:t2])  # max ipsc (VIPCR)
    rg = int(mx-mn)/10
    rg = rg - rg % 10

    plt.text(1027, mn+21.5, r'5 pA', fontweight='bold')
    plt.text(1047, mn+18.5, r'10 ms', fontweight='bold')

    plt.plot([1045, 1045], [mn+20, mn+20 + 5], linewidth=1.6,
             color='black')  # 5 pA vertical or pA
    plt.plot([1045, 1055], [mn+20, mn+20],
             linewidth=1.6, color='black')  # 10 ms

    plt.ylabel('Current (pA)')
    plt.xlabel('Time (ms)')
    plt.suptitle('      Post-synaptic currents to VIPCCK ',
                 fontsize=14, fontweight='bold')
    plt.legend(frameon=False)

    plt.savefig(fname+'/plots/toVIPCCK_post_synaptic_currents.pdf',
                format='pdf', dpi=600)

    plt.show()


#                    ___     ___   ___
########### \    / | |___|  |     |___| ###########
############ \  /  | |      |     | \    ##############
############# \/   | |      |___  |  \  ################


elif cell == "VIPCR":
    print("VIPCR Cell plot")

    # VIPCR cell
    with open(fname+'/ECtoVIPCR_i_rec_soma.txt', 'r', 0) as f1:
        vec1 = f1.readlines()
    with open(fname+'/CA3toVIPCR_i_rec_soma.txt', 'r', 0) as f2:
        vec2 = f2.readlines()
    with open(fname+'/PCtoVIPCR_i_rec_soma.txt', 'r', 0) as f3:
        vec3 = f3.readlines()

    # Make strings as floats with list comprehension
    vec1 = [float(x1) for x1 in vec1]
    vec2 = [float(x2) for x2 in vec2]
    vec3 = [float(x3) for x3 in vec3]

    # To convert in pA
    vec1_pA = [1000*x1 for x1 in vec1]
    vec2_pA = [1000*x2 for x2 in vec2]
    vec3_pA = [1000*x3 for x3 in vec3]

    a = range(1, len(vec1_pA)+1)  # time bins
    # print a
    tb = [x*dt for x in a]

    fig = plt.figure(1, figsize=(10, 10))

    plt.plot(tb[t1:t2], vec1_pA[t1:t2], label='EC EPSC')
    plt.plot(tb[t1:t2], vec2_pA[t1:t2], label='CA3 EPSC')
    plt.plot(tb[t1:t2], vec3_pA[t1:t2], label='PC EPSC')

    mn = min(vec3_pA[t1:t2])  # min PC
    mx = 0  # max 0
    rg = int(mx-mn)/10
    rg = rg - rg % 10
    # print mx,mn, rg

    plt.text(1016, mn+21.5, r'5 pA', fontweight='bold')
    plt.text(1033, mn+18.5, r'10 ms', fontweight='bold')

    plt.plot([1030, 1030], [mn+20, mn+20 + 5], linewidth=1.6,
             color='black')  # 5 pA vertical or pA
    plt.plot([1030, 1040], [mn+20, mn+20],
             linewidth=1.6, color='black')  # 10 ms

    plt.ylabel('Current (pA)')
    plt.xlabel('Time (ms)')
    plt.suptitle('      Post-synaptic currents to VIPCR ',
                 fontsize=14, fontweight='bold')
    plt.legend(frameon=False)

    plt.savefig(fname+'/plots/toVIPCR_post_synaptic_currents.pdf',
                format='pdf', dpi=600)

    plt.show()


#########################################################################
########################################################################
#########################################################################

else:
    print("oops!! something is going wrong! Not a valid Input argument.")
