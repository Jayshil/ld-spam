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
	IQR = stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
	N = data.size
	bw = (2 * IQR) / np.power(N, 1/3)
	if returnas=="width":
		result = bw
	else:
		datmin, datmax = data.min(), data.max()
		datrng = datmax - datmin
		result = int((datrng / bw) + 1)
	return(result)

def average_resid(diff, diffe):
    """
    Upon giving array of residuals and
    errors in them, this function evaluates
    median residual and uncertainty in it.
    ---------------------------------------
    Parameters:
    -----------
    diff : numpy.ndarray
        Array of residuals
        Each point gives residual for one planet
    diffe : numpy.ndarray
        Array of errors in residuals
        Each point gives standard deviation in residuals for one planet
    -----------
    return
    -----------
    float, float :
        Median residual and error in it
    """
    diff_tmp = np.random.normal(0,0,10000)

    for i in range(len(diff)):
        diff1 = np.random.normal(diff[i], diffe[i], 10000)
        diff_tmp = np.vstack((diff_tmp, diff1))

    diffi = diff_tmp[1:]
    mean8 = np.mean(diffi, axis=0)

    med8 = np.median(mean8)
    std8 = np.std(mean8)
    return med8, std8


def image_double(xdata1, xdata2, xerr1, xerr2, ydata1, ydata2, yerr1, yerr2, label1, label2, xlabel, ylabel, ttl, lim1=None, lim2=None, diff_ret=False):
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
    tuple, tuple
        average difference (with standard deviation)
        b/w (xdata1 & ydata1) and (xdata2 & ydata2)
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
    if lim1 != None:
        xlin = ylin = np.linspace(lim1, lim2, 100)
    else:
        xlin = ylin = np.linspace(low_lim, upp_lim, 100)

    # Difference between xdata1 and ydata1
    diff_1, diff_1e = np.zeros(len(xdata1)), np.zeros(len(xdata1))

    for i in range(len(xdata1)):
        x11 = np.random.normal(xdata1[i], xerr1[i], 10000)
        y11 = np.random.normal(ydata1[i], yerr1[i], 10000)
        diff1 = y11 - x11
        diff_1[i] = np.median(diff1)
        diff_1e[i] = np.std(diff1)

    # Difference between xdata2 and ydata2
    diff_2, diff_2e = np.zeros(len(xdata1)), np.zeros(len(xdata1))

    for i in range(len(xdata1)):
        x22 = np.random.normal(xdata2[i], xerr2[i], 10000)
        y22 = np.random.normal(ydata2[i], yerr2[i], 10000)
        diff2 = y22 - x22
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

    if lim1 != None:
        ax1.set_xlim([lim1, lim2])
        ax1.set_ylim([lim1, lim2])
    else:
        ax1.set_xlim([low_lim, upp_lim])
        ax1.set_ylim([low_lim, upp_lim])

    plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ttl)

    # Bottom Panel
    ax2 = plt.subplot(gs1[1])#, sharex = ax_u1_c_p)

    ax2.hist(diff_1, bins=freedman_diaconis(data=diff_1, returnas="bins"), alpha=0.7, color='orangered', zorder=5)
    ax2.hist(diff_2, bins=freedman_diaconis(data=diff_2, returnas="bins"), alpha=0.7, color='cornflowerblue', zorder=5)

    plt.ylabel('Count')
    plt.xlabel('Residuals')

    plt.subplots_adjust(hspace = 0.3)

    if diff_ret:
        return diff_1, diff_1e, diff_2, diff_2e
    else:
        med1, std1 = average_resid(diff_1, diff_1e)
        med2, std2 = average_resid(diff_2, diff_2e)
        return (med1, std1), (med2, std2)
        


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

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

def lowest_bic(a):
    """
    Find the lowest BIC (Bayesian Inference Crterian) among
    three BICs.
    I know I can just look at them manually but I am a little-
    bit lazy to do so. Hence here I am with a function defined for it.
    Parameters
    ----------
    a: dict
        Values of BICs in dict.
    returns: dict
        Containing the best fit model, with its BIC.
        If two models have BIC difference less than 2,
        then the model with a fewer number of parameters
        would be selected.
    """
    b = np.array([])
    for i in a.values():
        b = np.hstack((b,i))
    def get_key(val, my_dict):
        for key, value in my_dict.items():
            if val == value:
                return key
        return "Key does not exist"
    xx = np.min(b)
    yy = get_key(xx, a)
    ret = {}
    con = np.abs(a[yy] - a['constant'])
    lin = np.abs(a[yy] - a['linear'])
    qua = np.abs(a[yy] - a['quadratic'])
    if yy == 'quadratic':
        if con < 2:
            ret['constant'] = a['constant']
        elif lin < 2 and lin < con:
            ret['linear'] = a['linear']
        else:
            ret['quadratic'] = a['quadratic']
    if yy == 'linear':
        if con < 2:
            ret['constant'] = a['constant']
        else:
            ret['linear'] = a['linear']
    if yy == 'constant':
        ret['constant'] = a['constant']
    return ret

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

def binned_data(datax, datay, nos=10, datax_err=None, datay_err=None):
    """
    This function creates binned array from the given array.
    Parameters
    ----------
    datax: np.ndarray
        One dimensional array. x-coordinate
    datax_err: np.ndarray
        One dimensional array. Error in x-coordinate
        If not provided then assumes to be a zero matrix
        of length of datax.
    datay: np.ndarray
        One dimensional array. y-coordinate
    datay_err: np.ndarray
        One dimensional array. Error in y-coordinate
        If not provided then assumes to be a zero matrix
        of length of datay.
    nos: float
        Number of binned data points you want to set.
        Default is 10.
    returns: {np.ndarray}
        numpy array of binned data
    """
    if datax_err is None:
        datax_err = np.zeros(len(datax))
    if datay_err is None:
        datay_err = np.zeros(len(datay))
    aa = []
    for i in range(len(datax)):
        xxx = (datax[i], datax_err[i], datay[i], datay_err[i])
        aa.append(xxx)
    bb = sorted(aa)
    aaa = bbb = ccc = ddd = np.array([])
    for i in range(len(bb)):
        aaa = np.hstack((aaa, bb[i][0]))
        bbb = np.hstack((bbb, bb[i][1]))
        ccc = np.hstack((ccc, bb[i][2]))
        ddd = np.hstack((ddd, bb[i][3]))
    rep = int((len(datax))/(nos-1))
    rem = len(datax) - ((nos-1)*rep)
    bin_datax = bin_datax_err = bin_datay = bin_datay_err = np.array([])
    k = 0
    for i in range(nos-1):
        du_t = np.zeros(1000)
        for j in range(rep):
            abc1 = np.random.normal(aaa[k], bbb[k], 1000)
            du_t = np.vstack((du_t, abc1))
            k = k+1
        bint = np.mean(du_t[1:], axis=0)
        bin_datax = np.hstack((bin_datax, np.median(bint)))
        bin_datax_err = np.hstack((bin_datax_err, np.std(bint)))
    rem_t = np.zeros(1000)
    for i in range(rem):
        abc1 = np.random.normal(aaa[k], bbb[k], 1000)
        rem_t = np.vstack((rem_t, abc1))
    remt = np.mean(rem_t[1:], axis=0)
    bin_datax = np.hstack((bin_datax, np.median(remt)))
    bin_datax_err = np.hstack((bin_datax_err, np.std(remt)))
    k1 = 0
    for i in range(nos-1):
        du_d = np.zeros(1000)
        for j in range(rep):
            abc1 = np.random.normal(ccc[k1], ddd[k1], 1000)
            du_d = np.vstack((du_d, abc1))
            k1 = k1+1
        bind = np.mean(du_d[1:], axis=0)
        bin_datay = np.hstack((bin_datay, np.median(bind)))
        bin_datay_err = np.hstack((bin_datay_err, np.std(bind)))
    rem_d = np.zeros(1000)
    for i in range(rem):
        abc1 = np.random.normal(ccc[k1], ddd[k1], 1000)
        rem_d = np.vstack((rem_d, abc1))
    remd = np.mean(rem_d[1:], axis=0)
    bin_datay = np.hstack((bin_datay, np.median(remd)))
    bin_datay_err = np.hstack((bin_datay_err, np.std(remd)))
    return bin_datax, bin_datax_err, bin_datay, bin_datay_err

def teff_vs_resid(teff, xdata1, xerr1, ydata1, yerr1, xdata2, xerr2, ydata2, yerr2, label1, label2, ylabel):
    """
    Function to make plots of effective temps
    vs residuals of Empirical LDCs - Theoretical LDCs
    -------------------------------------------------
    Parameters:
    -----------
    teff : numpy.ndarray
        Effective temperatures of the host stars
    xdata1, ydata1 : numpy.ndarray
        x-data and y-data of the first kind of residuals
    xerr1, yerr1 : numpy.ndarray
        errors on xdata1 and ydata1
    xdata2, ydata2 : numpy.ndarray
        x-data and y-data of the second kind of residuals
    xerr2, yerr2 : numpy.ndarray
        errors on xdata2 and ydata2
    label1, label2 : str
        Labels of both plots
    xlabel, ylabel : str
        xlabels and ylabels of the plot
    -----------
    return
    -----------
    matplotlib.figure
    """
    resid1, resid1_err, resid2, resid2_err = image_double(xdata1, xdata2, xerr1, xerr2, ydata1, ydata2, yerr1,\
         yerr2, label1, label2, xlabel, ylabel, ttl='', lim1=None, lim2=None, diff_ret=True)
    # Basics
    tmin = np.min(teff)
    tmax = np.max(teff)

    t11 = np.linspace(tmin, tmax, 1000)

    x = np.linspace(tmin, tmax, 100)
    y = np.zeros(len(x))

    def line(x,m,c):
        function = m*x + c
        return function

    def constant(x, c):
        function = c + x*0
        return function

    def quadratic(x, a, b, c):
        function = a*x*x + b*x + c
        return function

    model = ['constant', 'linear', 'quadratic']
    
    # Making a plot and plotting the points
    fig1 = plt.figure(figsize=(8,6))

    plt.errorbar(teff, resid1, yerr = resid1_err, fmt='.', elinewidth=1, alpha=0.35, color='orangered', zorder=5, label = label1)
    plt.errorbar(teff, resid2, yerr = resid2_err, fmt='.', elinewidth=1, alpha=0.35, color='cornflowerblue',zorder=5, label = label2)

    #-----Plotting binned data
    bin_teff, bin_teffe, bin_diff1, bin_diff1e = binned_data(datax=teff, datay=resid1, datay_err=resid1_err)
    plt.errorbar(bin_teff, bin_diff1, xerr=bin_teffe, yerr=bin_diff1e, fmt='o',\
        color='orangered', mew=2, ms=8, alpha=1, markerfacecolor='white', mec='orangered', zorder=10)

    bin_teff, bin_teffe, bin_diff2, bin_diff2e = binned_data(datax=teff, datay=resid2, datay_err=resid2_err)
    plt.errorbar(bin_teff, bin_diff2, xerr=bin_teffe, yerr=bin_diff2e, fmt='o',\
        color='cornflowerblue', mew=2, alpha=1, ms=8, markerfacecolor='white', mec='cornflowerblue', zorder=10)

    #----Constant fitting
    popt_c, pcov_c = cft(constant, teff, resid1)
    popt1_c, pcov1_c = cft(constant, teff, resid2)

    rss_c = rss1_c = 0
    for i in range(len(teff)):
        r111 = (resid1[i] - constant(teff[i], *popt_c))**2
        rss_c = rss_c + r111
        r222 = (resid2[i] - constant(teff[i], *popt1_c))**2
        rss1_c = rss1_c + r222

    bic_c = len(resid1)*np.log((rss_c)/(len(resid1))) + np.log(len(resid1))
    bic1_c = len(resid2)*np.log((rss1_c)/(len(resid2))) + np.log(len(resid2))

    #----Linear fitting
    popt_l, pcov_l = cft(line, teff, resid1)
    popt1_l, pcov1_l = cft(line, teff, resid2)

    rss_l = rss1_l = 0
    for i in range(len(teff)):
        r111 = (resid1[i] - line(teff[i], *popt_l))**2
        rss_l = rss_l + r111
        r222 = (resid2[i] - line(teff[i], *popt1_l))**2
        rss1_l = rss1_l + r222

    bic_l = len(resid1)*np.log((rss_l)/(len(resid1))) + 2*np.log(len(resid1))
    bic1_l = len(resid2)*np.log((rss1_l)/(len(resid2))) + 2*np.log(len(resid2))

    #----Quadratic fitting
    popt_q, pcov_q = cft(quadratic, teff, resid1)
    popt1_q, pcov1_q = cft(quadratic, teff, resid2)

    rss_q = rss1_q = 0
    for i in range(len(teff)):
        r111 = (resid1[i] - quadratic(teff[i], *popt_q))**2
        rss_q = rss_q + r111
        r222 = (resid2[i] - quadratic(teff[i], *popt1_q))**2
        rss1_q = rss1_q + r222

    bic_q = len(resid1)*np.log((rss_q)/(len(resid1))) + 3*np.log(len(resid1))
    bic1_q = len(resid2)*np.log((rss1_q)/(len(resid2))) + 3*np.log(len(resid2))

    #----Plotting models
    bic = [bic_c, bic_l, bic_q]
    bic1 = [bic1_c, bic1_l, bic1_q]

    dict_bic = {}
    dict_bic1 = {}
    for i in range(3):
        dict_bic[model[i]] = bic[i]
        dict_bic1[model[i]] = bic1[i]

    best_bic = lowest_bic(dict_bic)
    if 'constant' in best_bic:
        plt.plot(t11, constant(t11, *popt_c), color = 'orangered', ls = '-.', zorder=3, alpha=0.85)
    elif 'linear' in best_bic:
        plt.plot(t11, line(t11, *popt_l), color='orangered', ls='-.', zorder=3, alpha=0.85)
    elif 'quadratic' in best_bic:
        plt.plot(t11, quadratic(t11, *popt_q), color = 'orangered', ls='-.', zorder=3, alpha=0.85)

    best_bic1 = lowest_bic(dict_bic1)
    if 'constant' in best_bic1:
        plt.plot(t11, constant(t11, *popt1_c), color = 'cornflowerblue', ls = '-.', zorder=3, alpha=0.85)
    elif 'linear' in best_bic1:
        plt.plot(t11, line(t11, *popt1_l), color='cornflowerblue', ls='-.', zorder=3, alpha=0.85)
    elif 'quadratic' in best_bic1:
        plt.plot(t11, quadratic(t11, *popt1_q), color = 'cornflowerblue', ls='-.', zorder=3, alpha=0.85)

    #---------------------
    plt.plot(x, y, 'k--')
    plt.grid()
    plt.legend(loc='best')
    plt.ylabel(ylabel)
    plt.xlabel('Effective Temperature')