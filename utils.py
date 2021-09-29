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


def image_double(xdata1, xdata2, xerr1, xerr2, ydata1, ydata2, yerr1, yerr2, save=False, path=os.getcwd()):
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

    xlo = np.minimum(xlu1_c_p, xlu1_c_a)
    xup = np.maximum(xuu1_c_p, xuu1_c_a)

    xlo, xup = -0.4, 1.4

    x1u1_c_p = y1u1_c_p = np.linspace(xlo, xup, 100)
    y11u1_c_p = np.zeros(len(x1u1_c_p))

    diff_u1_c_p = np.array([])#--------------------------------------------------------------------------------------------
    diff_u1_c_pe = np.array([])#-------------------------------------------------------------------------------------------

    for i in range(len(u1_j)):
        u11_c_p = np.random.normal(u1_c_p[i], 0, 10000)
        u11_j = np.random.normal(u1_j[i], u1_jp[i], 10000)
        diff1 = u11_c_p - u11_j
        u11_m = np.median(diff1)
        u11_e = np.std(diff1)
        diff_u1_c_p = np.hstack((diff_u1_c_p, u11_m))
        diff_u1_c_pe = np.hstack((diff_u1_c_pe, u11_e))

    diff_u1_c_a = np.array([])#--------------------------------------------------------------------------------------------
    diff_u1_c_ae = np.array([])#-------------------------------------------------------------------------------------------

    for i in range(len(u1_j)):
        u11_c_a = np.random.normal(u1_c_a[i], 0, 10000)
        u11_j = np.random.normal(u1_j[i], u1_jp[i], 10000)
        diff1 = u11_c_a - u11_j
        u11_m = np.median(diff1)
        u11_e = np.std(diff1)
        diff_u1_c_a = np.hstack((diff_u1_c_a, u11_m))
        diff_u1_c_ae = np.hstack((diff_u1_c_ae, u11_e))


    fig_u1_c_p = plt.figure(figsize=(8,10))
    gs_u1_c_p = gd.GridSpec(2, 1, height_ratios = [4,1])

    ax_u1_c_p = plt.subplot(gs_u1_c_p[0])

    ax_u1_c_p.errorbar(u1_j, u1_c_p, xerr = [u1_jn, u1_jp], fmt='.', elinewidth=1, alpha=0.5, color='orangered', zorder=5, label = 'PHOENIX LDCs')
    ax_u1_c_p.errorbar(u1_j, u1_c_a, xerr = [u1_jn, u1_jp], fmt='.', elinewidth=1, alpha=0.5, color='cornflowerblue',zorder=5, label = 'ATLAS LDCs')

    ax_u1_c_p.plot(x1u1_c_p, y1u1_c_p, 'k--')
    ax_u1_c_p.grid()

    ax_u1_c_p.set_xlim([xlo, xup])
    ax_u1_c_p.set_ylim([xlo, xup])

    plt.legend(loc='best')
    plt.ylabel(r'$u_1$ (Theoretical)')
    plt.xlabel(r'$u_1$ (Empirical)')
    plt.title('Values from Claret(2017)')

    ax1_u1_c_p = plt.subplot(gs_u1_c_p[1])#, sharex = ax_u1_c_p)

    ax1_u1_c_p.hist(diff_u1_c_p, bins=utl.freedman_diaconis(data=diff_u1_c_p, returnas="bins"), alpha=0.7, color='orangered', zorder=5)
    ax1_u1_c_p.hist(diff_u1_c_a, bins=utl.freedman_diaconis(data=diff_u1_c_a, returnas="bins"), alpha=0.7, color='cornflowerblue', zorder=5)

    plt.ylabel('Count')
    plt.xlabel('Residuals')

    plt.subplots_adjust(hspace = 0.3)
    plt.savefig(path1 + '/Results/cal_us_and_evidance/u1_cla.pdf')
    plt.close(fig_u1_c_p)
