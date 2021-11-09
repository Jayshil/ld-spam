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

# Effective temperatures
teff1 = np.loadtxt(path1 + '/data_new.dat', usecols=9, unpack=True)

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

errs = np.zeros(len(teff1))

l1, l2 = -0.4, 1.4

# Just remember to remove TOI-157b from teff1 and u1_j, u2_j arrays

utl.teff_vs_resid(teff=teff1, xdata1=u1_j, xerr1=u1_jp, ydata1=u1a_c17, yerr1=errs, xdata2=u1_j,\
      xerr2=u1_jp, ydata2=u1p_c17, yerr2=errs, label1='PHOENIX LDCs', label2='ATLAS LDCs', ylabel=r'Residuals in $u_1$ from Claret (2017)')
plt.show()