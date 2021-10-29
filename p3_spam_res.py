import numpy as np
import matplotlib.pyplot as plt
import utils as utl
import os
import seaborn as sns
from matplotlib import rcParams
from pylab import *


sns.set_context("talk")
sns.set_style("ticks")

# Fonts:
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size':12})
plt.rc('legend', **{'fontsize':12})

# Ticks to the outside:
rcParams['axes.linewidth'] = 1.2 
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

# Path to the data (i.e., SPAM LDCs)
p1d = os.getcwd() + '/Results_New/'

# Getting data from our modelling
path1 = "/home/jayshil/Documents/Dissertation/ld-project-updated"

# juliet LDCs
u1_j, u1_jp, u1_jn, u2_j, u2_jp, u2_jn = np.loadtxt(path1 + '/Data/results.dat', usecols = (16,17,18,19,20,21), unpack = True)

# ATLAS EJ15 SPAM
u1a_ej15, u2a_ej15 = np.loadtxt(p1d + 'SPAM_ATLAS-EJ15.dat', usecols=(1,2), unpack=True)
# ATLAS C17 SPAM
u1a_c17, u2a_c17 = np.loadtxt(p1d + 'SPAM_ATLAS-C17.dat', usecols=(1,2), unpack=True)

# PHOENIX EJ15 SPAM
u1p_ej15, u2p_ej15 = np.loadtxt(p1d + 'SPAM_PHOENIX-EJ15.dat', usecols=(1,2), unpack=True)
# PHOENIX C17 SPAM
u1p_c17, u2p_c17 = np.loadtxt(p1d + 'SPAM_PHOENIX-C17.dat', usecols=(1,2), unpack=True)
# PHOENIX C17r SPAM
u1p_c17r, u2p_c17r = np.loadtxt(p1d + 'SPAM_PHOENIX-C17r.dat', usecols=(1,2), unpack=True)

errs = np.zeros(len(u1a_ej15))

l1, l2 = -0.4, 1.4

## u1_cla
u1_cla_pho, u1_cla_ata = utl.image_double(xdata1=u1_j, xdata2=u1_j, xerr1=u1_jp, xerr2=u1_jp,\
     ydata1=u1p_c17, ydata2=u1a_c17, yerr1=errs, yerr2=errs,\
     label1='PHOENIX SPAM LDCs', label2='ATLAS SPAM LDCs', xlabel=r'$u_1$ (Empirical)', ylabel=r'$u_1$ (Theoretical)',\
     ttl='Values from Claret (2017)', lim1=l1, lim2=l2)
plt.savefig(p1d + 'u1_cla_SPAM.pdf')

## u1_cla_r
u1_cla_pho1, u1_cla_pho_r = utl.image_double(xdata1=u1_j, xdata2=u1_j, xerr1=u1_jp, xerr2=u1_jp,\
     ydata1=u1p_c17, ydata2=u1p_c17r, yerr1=errs, yerr2=errs,\
     label1='PHOENIX (q-method) SPAM LDCs', label2='PHOENIX (r-method) SPAM LDCs', xlabel=r'$u_1$ (Empirical)', ylabel=r'$u_1$ (Theoretical)',\
     ttl='Values from Claret (2017)', lim1=l1, lim2=l2)
plt.savefig(p1d + 'u1_cla_SPAM_r.pdf')

## u1_code
u1_code_pho, u1_code_ata = utl.image_double(xdata1=u1_j, xdata2=u1_j, xerr1=u1_jp, xerr2=u1_jp,\
     ydata1=u1p_ej15, ydata2=u1a_ej15, yerr1=errs, yerr2=errs,\
     label1='PHOENIX SPAM LDCs', label2='ATLAS SPAM LDCs', xlabel=r'$u_1$ (Empirical)', ylabel=r'$u_1$ (Theoretical)',\
     ttl='Values from Espinoza \& Jordan (2015)', lim1=l1, lim2=l2)
plt.savefig(p1d + 'u1_code_SPAM.pdf')

## u2_cla
u2_cla_pho, u2_cla_ata = utl.image_double(xdata1=u2_j, xdata2=u2_j, xerr1=u2_jp, xerr2=u2_jp,\
     ydata1=u2p_c17, ydata2=u2a_c17, yerr1=errs, yerr2=errs,\
     label1='PHOENIX SPAM LDCs', label2='ATLAS SPAM LDCs', xlabel=r'$u_2$ (Empirical)', ylabel=r'$u_2$ (Theoretical)',\
     ttl='Values from Claret (2017)', lim1=l1, lim2=l2)
plt.savefig(p1d + 'u2_cla_SPAM.pdf')

## u2_cla_r
u2_cla_pho1, u2_cla_pho_r = utl.image_double(xdata1=u2_j, xdata2=u2_j, xerr1=u2_jp, xerr2=u2_jp,\
     ydata1=u2p_c17, ydata2=u2p_c17r, yerr1=errs, yerr2=errs,\
     label1='PHOENIX (q-method) SPAM LDCs', label2='PHOENIX (r-method) SPAM LDCs', xlabel=r'$u_2$ (Empirical)', ylabel=r'$u_2$ (Theoretical)',\
     ttl='Values from Claret (2017)', lim1=l1, lim2=l2)
plt.savefig(p1d + 'u2_cla_SPAM_r.pdf')

## u2_code
u2_code_pho, u2_code_ata = utl.image_double(xdata1=u2_j, xdata2=u2_j, xerr1=u2_jp, xerr2=u2_jp,\
     ydata1=u2p_ej15, ydata2=u2a_ej15, yerr1=errs, yerr2=errs,\
     label1='PHOENIX SPAM LDCs', label2='ATLAS SPAM LDCs', xlabel=r'$u_2$ (Empirical)', ylabel=r'$u_2$ (Theoretical)',\
     ttl='Values from Espinoza \& Jordan (2015)', lim1=l1, lim2=l2)
plt.savefig(p1d + 'u2_code_SPAM.pdf')


## Making tables for mean offset
f101 = open(p1d + 'mean_offset.dat', 'w')
f101.write('\t\t\t\tu1\t\t\t\tu2\n')
f101.write('Claret(2017), PHOENIX - q method\t\t' + str(u1_cla_pho[0]) + ' +/- ' + str(u1_cla_pho[1]) + '\t\t' + str(u2_cla_pho[0]) + ' +/- ' + str(u2_cla_pho[1]) + '\n')
f101.write('Claret(2017), PHOENIX - r method\t\t' + str(u1_cla_pho_r[0]) + ' +/- ' + str(u1_cla_pho_r[1]) + '\t\t' + str(u2_cla_pho_r[0]) + ' +/- ' + str(u2_cla_pho_r[1]) + '\n')
f101.write('Claret(2017), ATLAS\t\t' + str(u1_cla_ata[0]) + ' +/- ' + str(u1_cla_ata[1]) + '\t\t' + str(u2_cla_ata[0]) + ' +/- ' + str(u2_cla_ata[1]) + '\n')
f101.write('EJ(2015), PHOENIX\t\t' + str(u1_code_pho[0]) + ' +/- ' + str(u1_code_pho[1]) + '\t\t' + str(u2_code_pho[0]) + ' +/- ' + str(u2_code_pho[1]) + '\n')
f101.write('EJ(2015), ATLAS\t\t\t' + str(u1_code_ata[0]) + ' +/- ' + str(u1_code_ata[1]) + '\t\t' + str(u2_code_ata[0]) + ' +/- ' + str(u2_code_ata[1]) + '\n')
f101.close()