import sys
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
    label1, label2 : str
        labels of two type of data
    xlabel, ylabel : str
        labels of axis
    ttl : str
        title of the image
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
        diff_1[i] = np.median(diff1)
        diff_1e[i] = np.std(diff1)

    # Difference between xdata2 and ydata2
    diff_2, diff_2e = np.zeros(len(xdata1)), np.zeros(len(xdata1))

    for i in range(len(xdata1)):
        x22 = np.random.normal(xdata2[i], xerr2[i], 10000)
        y22 = np.random.normal(ydata2[i], yerr2[i], 10000)
        diff2 = x22 - y22
        diff_2[i] = np.median(diff2)
        diff_2e[i] = np.std(diff2)

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


def single_image(xdata, xerr, ydata, yerr, xlabel, ylabel):
    """
    Function to make figures according
    to Patel & Espinoza (2021) pattern
    ----------------------------------
    Parameters:
    -----------
    xdata : numpy.ndarray
        data to be plotted on the x-axis
    xerr : numpy.ndarray
        errors on xdata
    ydata : numpy.ndarray
        data to be plotted on the y-axis
    yerr : numpy.ndarray
        errors on ydata
    xlabel, ylabel : str
        labels of x- and y-axis
    -----------
    return
    -----------
    matplotlib.figure
        showing figure object
    """
    # Setting up the limits of the plot
    # x-lim
    xmax = np.max(xdata)
    xmin = np.min(xdata)
    # y-lim
    ymax = np.max(ydata)
    ymin = np.min(ydata)

    # Limits of both axis to make figure square
    low_limit = np.minimum(xmin, ymin)
    upp_limit = np.maximum(xmax, ymax)

    # Making a straight line for xdata=ydata
    xlin = ylin = np.linspace(low_limit, upp_limit, 100)
    yzero = np.zeros(len(xlin))

    diff, diff_e = np.zeros(len(xdata)), np.zeros(len(xdata))

    for i in range(len(xdata)):
        x1 = np.random.normal(xdata[i],xerr[i],10000)
        y1 = np.random.normal(ydata[i], yerr[i], 10000)
        diff1 = x1 - y1
        diff[i] = np.median(diff1)
        diff_e[i] = np.std(diff1)

    # Making figure
    fig1 = plt.figure(figsize = (8,10))
    gs1 = gd.GridSpec(2, 1, height_ratios = [4,1])

    ax1 = plt.subplot(gs1[0])

    ax1.errorbar(xdata, ydata, xerr = xerr, yerr = yerr, fmt = '.', elinewidth=1, alpha=0.5, color='black', zorder=5)

    plt.xlim([low_limit, upp_limit])
    plt.ylim([low_limit, upp_limit])
    ax1.plot(xlin, ylin, 'k--')
    ax1.grid()
    plt.ylabel(ylabel)
    #plt.title('Comparison between literature values and calculated values of a/R*')

    ax2 = plt.subplot(gs1[1], sharex = ax1)

    ax2.errorbar(xdata, diff, xerr = xerr, yerr = diff_e, fmt = '.', elinewidth=1, alpha=0.5, color='black', zorder=5)

    plt.xlim([low_limit, upp_limit])
    #plt.ylim([-2,2])
    ax2.plot(xlin, yzero, 'k--')
    ax2.grid()
    plt.ylabel('Residuals')
    plt.xlabel(xlabel)

    plt.subplots_adjust(hspace = 0.2)


def limiting_ldcs(c1, c2, c3, c4):
    """
    To compute limiting LDCs from
    Espinoza & Jordan (2015)
    -----------------------------
    Parameters:
    -----------
    c1, c2, c3, c4 : float, or numpy.ndarray
        non-linear LDCs
    -----------
    return:
    -----------
    float, or numpy.ndarray
        limiting LDCs
    """
    u1 = (12./35.)*c1 + c2 + (164./105.)*c3 + 2.*c4
    u2 = (10./21.)*c1 - (34./63.)*c3 - c4
    return u1, u2