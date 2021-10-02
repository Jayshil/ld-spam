import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as mz
from scipy.optimize import curve_fit as cft
from scipy.optimize import leastsq as lsq
from tqdm import tqdm
import batman

def transit(time, t0, per, rp, a, b, ecc, w, u, law):
    """
    -----------
    Parameters:
    -----------
    time : numpy.ndarray
        time at which the flux is calculated
    t0 : float
        transit central time (in days/or the same as period)
    per : float
        orbital period of exoplanet (in days)
    rp : float
        square-root of tansit depth
    a : float
        scaled semi-major axis
    b : float
        impact parameter
    ecc : float
        eccentricity of the orbit
    w : float
        longitude of peri-astron passage (in deg)
    u : array like
        limb-darkening coefficients
    law : str
        name of the limb-darkening law
        linear, quadratic, nonlinear
    -----------
    return
    -----------
    numpy.ndarray
        array containing transit lightcurve
    """
    para = batman.TransitParams()
    para.t0 = t0
    para.per = per
    para.rp = rp
    para.a = a
    cosi = b/a
    in1 = 180*np.arccos(cosi)/np.pi
    para.inc = in1
    para.ecc = ecc
    para.w = w
    para.u = u
    para.limb_dark = law
    m = batman.TransitModel(para, time)
    fl = m.light_curve(para)
    return fl


def spam(time, per, rp, a, b, u, ecc=0., w=90., t0=0.):
    """
    -----------
    Parameters:
    -----------
    time : numpy.ndarray
        time at which the flux is calculated
    per : float
        orbital period of exoplanet (in days)
    rp : float
        square-root of tansit depth
    a : float
        scaled semi-major axis
    b : float
        impact parameter
    u : array like
        non-linear limb-darkening coefficients
    ecc : float
        eccentricity of the orbit
        default is 0.
    w : float
        longitude of peri-astron passage (in deg)
        default is 90 deg
    t0 : float
        transit central time (in days/or the same as period)
        default is 0.
    -----------
    return
    -----------
    numpy.ndarray
        array containing SPAM LDCs
    """
    synthetic_flux = transit(time, t0, per, rp, a, b, ecc, w, u, "nonlinear")
    def min_log_likelihood(x):
        model = transit(time, t0, per, rp, a, b, ecc, w, x, "quadratic")
        chi2 = np.sum((synthetic_flux-model)**2)
        return chi2
    u1_guess, u2_guess = (12./35.)*u[0] + u[1] + (164./105.)*u[2] + 2.*u[3], (10./21.)*u[0] - (34./63.)*u[2] - u[3]
    soln = mz(min_log_likelihood, x0=[u1_guess, u2_guess], method='L-BFGS-B')
    return soln.x

def mc_spam(time, per, per_err, rp, rp_err, a, a_err, b, b_err, u, ecc=0., w=90., t0=0.):
    """
    -----------
    Parameters:
    -----------
    time : numpy.ndarray
        time at which the flux is calculated
    per, per_err : float
        orbital period and error in it (in days)
    rp, rp_err : float
        square-root of tansit depth and its error
    a, a_err : float
        scaled semi-major axis and error in it
    b, b_err : float
        impact parameter and its error
    u : array like
        non-linear limb-darkening coefficients
    ecc : float
        eccentricity of the orbit
        default is 0.
    w : float
        longitude of peri-astron passage (in deg)
        default is 90 deg
    t0 : float
        transit central time (in days/or the same as period)
        default is 0.
    -----------
    return
    -----------
    numpy.ndarray, numpy.ndarray
        arrays containing distribution in MC-SPAM LDCs
    """
    period = np.random.normal(per, per_err, 1000)
    rp1 = np.random.normal(rp, rp_err, 1000)
    ar1 = np.random.normal(a, a_err, 1000)
    b1 = np.random.normal(b, b_err, 1000)
    u1_mcs, u2_mcs = np.zeros(1000), np.zeros(1000)
    for i in tqdm(range(len(period))):
        u1_mcs[i], u2_mcs[i] = spam(time, period[i], rp1[i], ar1[i], b1[i], u)
    return u1_mcs, u2_mcs

# SPAM and MC-SPAM LDCs using scipy.optimize.leastsq method

def spam_lsq(time, per, rp, a, b, u, ecc=0., w=90., t0=0.):
    """
    -----------
    Parameters:
    -----------
    time : numpy.ndarray
        time at which the flux is calculated
    per : float
        orbital period of exoplanet (in days)
    rp : float
        square-root of tansit depth
    a : float
        scaled semi-major axis
    b : float
        impact parameter
    u : array like
        non-linear limb-darkening coefficients
    ecc : float
        eccentricity of the orbit
        default is 0.
    w : float
        longitude of peri-astron passage (in deg)
        default is 90 deg
    t0 : float
        transit central time (in days/or the same as period)
        default is 0.
    -----------
    return
    -----------
    float, float
        SPAM LDCs
    """
    synthetic_flux = transit(time, t0, per, rp, a, b, ecc, w, u, "nonlinear")
    def resid(x):
        model = transit(time, t0, per, rp, a, b, ecc, w, x, "quadratic")
        residuals = synthetic_flux - model
        return residuals
    u1_guess, u2_guess = (12./35.)*u[0] + u[1] + (164./105.)*u[2] + 2.*u[3], (10./21.)*u[0] - (34./63.)*u[2] - u[3]
    soln = lsq(resid, x0 = [u1_guess, u2_guess])
    return soln[0][0], soln[0][1]

def mc_spam_lsq(time, per, per_err, rp, rp_err, a, a_err, b, b_err, u, ecc=0., w=90., t0=0.):
    """
    -----------
    Parameters:
    -----------
    time : numpy.ndarray
        time at which the flux is calculated
    per, per_err : float
        orbital period and error in it (in days)
    rp, rp_err : float
        square-root of tansit depth and its error
    a, a_err : float
        scaled semi-major axis and error in it
    b, b_err : float
        impact parameter and its error
    u : array like
        non-linear limb-darkening coefficients
    ecc : float
        eccentricity of the orbit
        default is 0.
    w : float
        longitude of peri-astron passage (in deg)
        default is 90 deg
    t0 : float
        transit central time (in days/or the same as period)
        default is 0.
    -----------
    return
    -----------
    numpy.ndarray, numpy.ndarray
        arrays containing distribution in MC-SPAM LDCs
    """
    period = np.random.normal(per, per_err, 1000)
    rp1 = np.random.normal(rp, rp_err, 1000)
    ar1 = np.random.normal(a, a_err, 1000)
    b1 = np.random.normal(b, b_err, 1000)
    u1_mcs, u2_mcs = np.zeros(1000), np.zeros(1000)
    for i in tqdm(range(len(period))):
        u1_mcs[i], u2_mcs[i] = spam_lsq(time, period[i], rp1[i], ar1[i], b1[i], u)
    return u1_mcs, u2_mcs