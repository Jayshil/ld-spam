import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as mz
import tqdm
import batman

def transit(time, t0, per, rp, a, b, ecc, w, u, law):
    """
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