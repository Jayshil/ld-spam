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
p1d = os.getcwd() + '/Results_new/'

# Getting data from our modelling
path1 = "/home/jayshil/Documents/Dissertation/ld-project-updated"

# juliet LDCs
u1_j, u1_jp, u1_jn, u2_j, u2_jp, u2_jn = np.loadtxt(path1 + '/Data/results.dat', usecols = (16,17,18,19,20,21), unpack = True)

# Let's first test our code!!
# Using Claret (2017) u1 data points

u1a = np.loadtxt(path1 + '/Atlas/claret_limiting_LDC_ata.dat', usecols=1, unpack=True)
u1p = np.loadtxt(path1 + '/Phoenix/claret_limiting_LDC_pho.dat', usecols=1, unpack=True)

errs = np.zeros(len(u1a))

ab, cd = utl.image_double(xdata1=u1_j, xdata2=u1_j, xerr1=u1_jp, xerr2=u1_jp, ydata1=u1a, ydata2=u1p, yerr1=errs, yerr2=errs,\
     label1='ATLAS-C17', label2='Phoenix-C17', xlabel=r'$u_1$ (Empirical)', ylabel=r'$u_1$ (Theoretical)', ttl='Claret (2017) LDCs')
plt.show()

print(ab)
print(cd)