import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec as gd
import os
import pickle as pc
#import utils1 as utl
from astropy.io import fits
from astroquery.mast import Observations as obs
from scipy import interpolate as inp
import matplotlib.cm as cm
import matplotlib.colors as cls
from scipy.optimize import curve_fit as cft
from astropy.table import Table
from astropy.io import ascii
import cdspyreadme
import astropy.units as u

path1 = '/home/jayshil/Documents/Dissertation/ld-spam/Results_New/'
path2 = '/home/jayshil/Documents/Dissertation/ld-spam'

name = np.loadtxt(path1 + 'SPAM_ATLAS-EJ15.dat', usecols = 0, unpack = True, dtype=str)

u1_code_ata, u2_code_ata = np.loadtxt(path1 + 'SPAM_ATLAS-EJ15.dat', usecols = (1,2), unpack = True)
u1_cla_ata, u2_cla_ata = np.loadtxt(path1 + 'SPAM_ATLAS-C17.dat', usecols = (1,2), unpack = True)

u1_code_pho, u2_code_pho = np.loadtxt(path1 + 'SPAM_PHOENIX-EJ15.dat', usecols = (1,2), unpack = True)
u1_cla_pho, u2_cla_pho = np.loadtxt(path1 + 'SPAM_PHOENIX-C17.dat', usecols = (1,2), unpack = True)
u1_cla_pho_r, u2_cla_pho_r = np.loadtxt(path1 + 'SPAM_PHOENIX-C17r.dat', usecols = (1,2), unpack = True)


"""
table1 = open(path2 + '/Table/ldc.dat','w')

for i in range(len(name)):
	str(np.format_float_positional(u1_cla_ata[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_ata[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_pho[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_pho[i], 2)) + ' & ' + str(np.format_float_positional(u1_cla_pho_r[i], 2)) + ' & ' + str(np.format_float_positional(u2_cla_pho_r[i], 2))

table1.close()
"""
# for name
name1 = []
for i in range(len(name)):
	name1.append(name[i][:-1])

tab1 = Table()
tab1['name'], tab1['name'].info.format, tab1['name'].description = np.asarray(name1), '%s', 'Name of the host star'
tab1['u1_code_ata'], tab1['u1_code_ata'].info.format, tab1['u1_code_ata'].description = u1_code_ata, '%1.2f', 'u1 from EJ15 ATLAS'
tab1['u2_code_ata'], tab1['u2_code_ata'].info.format, tab1['u2_code_ata'].description = u2_code_ata, '%1.2f', 'u2 from EJ15 ATLAS'
tab1['u1_code_pho'], tab1['u1_code_pho'].info.format, tab1['u1_code_pho'].description = u1_code_pho, '%1.2f', 'u1 from EJ15 PHOENIX'
tab1['u2_code_pho'], tab1['u2_code_pho'].info.format, tab1['u2_code_pho'].description = u2_code_pho, '%1.2f', 'u2 from EJ15 PHOENIX'
tab1['u1_cla_ata'], tab1['u1_cla_ata'].info.format, tab1['u1_cla_ata'].description = u1_cla_ata, '%1.2f', 'u1 from C17 ATLAS'
tab1['u2_cla_ata'], tab1['u2_cla_ata'].info.format, tab1['u2_cla_ata'].description = u2_cla_ata, '%1.2f', 'u2 from C17 ATLAS'
tab1['u1_cla_pho'], tab1['u1_cla_pho'].info.format, tab1['u1_cla_pho'].description = u1_cla_pho, '%1.2f', 'u1 from C17 PHOENIX'
tab1['u2_cla_pho'], tab1['u2_cla_pho'].info.format, tab1['u2_cla_pho'].description = u2_cla_pho, '%1.2f', 'u2 from C17 PHOENIX'
tab1['u1_cla_pho_r'], tab1['u1_cla_pho_r'].info.format, tab1['u1_cla_pho_r'].description = u1_cla_pho_r, '%1.2f', 'u1 from C17 PHOENIX, with r-method'
tab1['u2_cla_pho_r'], tab1['u2_cla_pho_r'].info.format, tab1['u2_cla_pho_r'].description = u2_cla_pho_r, '%1.2f', 'u2 from C17 PHOENIX, with r-method'

tab1.write(os.getcwd() + '/Table/spam_ldc_ascii_mrt.dat', format='ascii.mrt', overwrite=True, delimiter='\t\t')
