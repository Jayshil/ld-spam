import numpy as np
import os
import mcspam as mcs

p1 = os.getcwd()
p1r = os.getcwd() + '/Results/'

# Getting data from our modelling
path1 = "/home/jayshil/Documents/Dissertation/ld-project-updated"

a_j, a_jp, a_jn = np.loadtxt(path1 + '/Data/results.dat', usecols = (22,23,24), unpack = True)
a_e = (a_jp + a_jn)/2
p_j, p_jp, p_jn = np.loadtxt(path1 + '/Data/results.dat', usecols = (28,29,30), unpack = True)
p_e = (p_jp + p_jn)/2
r_j, r_jp, r_jn = np.loadtxt(path1 + '/Data/results.dat', usecols = (4,5,6), unpack = True)
r_e = (r_jp + r_jn)/2
b_j, b_jp, b_jn = np.loadtxt(path1 + '/Data/results.dat', usecols = (7,8,9), unpack = True)
b_e = (b_jp + b_jn)/2


##### Run-1: For Atlas - EJ15 LDCs
nn = np.loadtxt(p1 + '/Atlas/code_us_nl_ata.dat', usecols=0, unpack=True, dtype=str)
c1, c2, c3, c4 = np.loadtxt(p1 + '/Atlas/code_us_nl_ata.dat', usecols=(1,2,3,4), unpack=True)

##### Saving Results
f1 = open(p1r + 'MCS_Atlas_code_cft.dat', 'w')
f1.write('#Name\t\t MCS_u1\t\t MCS_u1_err\t\t MCS_u2\t\t MCS_u2_err\n')

for i in range(len(a_j)):
    t1 = p_j[i]/(a_j[i]*np.pi)
    tt = np.linspace(-t1, t1, 1000)
    # For Atlas - EJ15
    u1_mcs, u2_mcs = mcs.mc_spam_cft(time=tt, per=p_j[i], per_err=p_e[i], rp=r_j[i], rp_err=r_e[i], \
                                 a=a_j[i], a_err=a_e[i], b=b_j[i], b_err=b_e[i], u=[c1[i],c2[i],c3[i], c4[i]])
    f1.write(nn[i] + '\t' + str(np.median(u1_mcs)) + '\t' + str(np.std(u1_mcs)) + '\t' + str(np.median(u2_mcs)) + '\t' + str(np.std(u2_mcs)) + '\n')
    print("--------   Completed " + str(i+1) + " systems / out of " + str(len(a_j)) + " system!")

f1.close()