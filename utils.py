from sys import path
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy import stats
from matplotlib import gridspec as gd
from matplotlib import rcParams
import pickle as pc
from astropy.io import fits
from astroquery.mast import Observations as obs
from scipy import interpolate as inp
import matplotlib.cm as cm
import matplotlib.colors as cls
from scipy.optimize import curve_fit as cft
from pylab import *
import seaborn as sns



def freedman_diaconis(data, returnas="width"):
	"""
	Use Freedman Diaconis rule to compute optimal histogram bin width. 
	``returnas`` can be one of "width" or "bins", indicating whether
	the bin width or number of bins should be returned respectively.

	Parameters
	----------
	data: np.ndarray
		One-dimensional array.

	returnas: {"width", "bins"}
		If "width", return the estimated width for each histogram bin. 
		If "bins", return the number of bins suggested by rule.
	"""
	data = np.asarray(data, dtype=np.float_)
	IQR = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
	N = data.size
	bw = (2 * IQR) / np.power(N, 1/3)
	if returnas=="width":
		result = bw
	else:
		datmin, datmax = data.min(), data.max()
		datrng = datmax - datmin
		result = int((datrng / bw) + 1)
	return(result)


def image_double(xdata1, xdata2, xerr1, xerr2, ydata1, ydata2, yerr1, yerr2, label1, label2, xlabel, ylabel, ttl):
    """
    Function to make figures according
    to Patel & Espinoza (2021) pattern
    (Two plots on the same figure)
    ----------------------------------
    Parameters:
    -----------
    xdata1, xdata2 : numpy.ndarray
        data to be plotted on the x-axis
    xerr1, xerr2 : numpy.ndarray
        errors on xdata1 and xdata2
    ydata1, ydata2 : numpy.ndarray
        data to be plotted on the y-axis
    yerr1, yerr2 : numpy.ndarray
        errors on ydata1, ydata2
    save : bool
        if True, save the figure at Path
        default is False
    path : str
        location to save the image
    -----------
    return
    -----------
    matplotlib.figure
        showing figure object
    """
    # Setting up the limits of the figure
    # x-limit
    x1max, x1min = np.max(xdata1), np.min(xdata1)
    x2max, x2min = np.max(xdata2), np.min(xdata2)

    xmin, xmax = np.minimum(x1min, x2min), np.maximum(x1max, x2max)

    # y-limit
    y1max, y1min = np.max(ydata1), np.min(ydata1)
    y2max, y2min = np.max(ydata2), np.min(ydata2)

    ymin, ymax = np.minimum(y1min, y2min), np.maximum(y1max, y2max)

    # limits on the plot to make the figure square
    low_lim = np.minimum(xmin, ymin)
    upp_lim = np.maximum(xmax, ymax)

    # Making a line for xdata = ydata, and for
    # ydata = 0 for the bottom panel
    xlin = ylin = np.linspace(low_lim, upp_lim, 100)

    # Difference between xdata1 and ydata1
    diff_1, diff_1e = np.zeros(len(xdata1)), np.zeros(len(xdata1))

    for i in range(len(xdata1)):
        x11 = np.random.normal(xdata1[i], xerr1[i], 10000)
        y11 = np.random.normal(ydata1[i], yerr1[i], 10000)
        diff1 = x11 - y11
        d11_m = np.median(diff1)
        d11_e = np.std(diff1)
        diff_1[i], diff_1e[i] = d11_m, d11_e

    # Difference between xdata2 and ydata2
    diff_2, diff_2e = np.zeros(len(xdata1)), np.zeros(len(xdata1))

    for i in range(len(u1_j)):
        x22 = np.random.normal(xdata2[i], xerr2[i], 10000)
        y22 = np.random.normal(ydata2[i], yerr2[i], 10000)
        diff2 = x22 - y22
        d22_m = np.median(diff2)
        d22_e = np.std(diff2)
        diff_2[i], diff_2e[i] = d22_m, d22_e

    # Plotting figures
    fig1 = plt.figure(figsize=(8,10))
    gs1 = gd.GridSpec(2, 1, height_ratios = [4,1])

    # Upper Panel
    ax1 = plt.subplot(gs1[0])

    ax1.errorbar(xdata1, ydata1, xerr = xerr1, yerr = yerr1, fmt='.', elinewidth=1, alpha=0.5, color='orangered', zorder=5, label = label1)
    ax1.errorbar(xdata2, ydata2, xerr = xerr2, yerr = yerr2, fmt='.', elinewidth=1, alpha=0.5, color='cornflowerblue',zorder=5, label = label2)

    ax1.plot(xlin, ylin, 'k--')
    ax1.grid()

    ax1.set_xlim([low_lim, upp_lim])
    ax1.set_ylim([low_lim, upp_lim])

    plt.legend(loc='best')
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.title(ttl)

    # Bottom Panel
    ax2 = plt.subplot(gs1[1])#, sharex = ax_u1_c_p)

    ax2.hist(diff_1, bins=freedman_diaconis(data=diff_1, returnas="bins"), alpha=0.7, color='orangered', zorder=5)
    ax2.hist(diff_2, bins=freedman_diaconis(data=diff_2, returnas="bins"), alpha=0.7, color='cornflowerblue', zorder=5)

    plt.ylabel('Count')
    plt.xlabel('Residuals')

    plt.subplots_adjust(hspace = 0.3)


path1 = '/home/jayshil/Documents/Dissertation/ld-project-updated'

# Let's make ATLAS Clatet (2017) and Phoenix Claret (2017) comparison with juliet to test the function
u1_a = np.loadtxt(path1 + '/Atlas/claret_limiting_LDC_ata.dat', usecols=1, unpack=True)
u1_p = np.loadtxt(path1 + '/Phoenix/claret_limiting_LDC_pho_r.dat', usecols=1, unpack=True)
u1_j, u1_jp, u1_jn, u2_j, u2_jp, u2_jn = np.loadtxt(path1 + '/Data/results.dat', usecols = (16,17,18,19,20,21), unpack = True)

image_double(xdata1=u1_j, xdata2=u1_j, xerr1=u1_jp, xerr2=u1_jp, ydata1=u1_a, ydata2=u1_p, yerr1=np.zeros(len(u1_a)),\
     yerr2=np.zeros(len(u1_a)), label1='ATLAS LDCs', label2='Phoenix LDCs', xlabel='Observed LDCs', ylabel='Theoretical LDCs',\
     ttl='Claret (2017) LDCs')
plt.show()