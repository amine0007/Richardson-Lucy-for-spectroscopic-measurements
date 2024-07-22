import numpy as np
from scipy.interpolate import interp1d

def interpol_bandpass(lamb_axis, values, tol_equi=1e-5):
    if np.ptp(np.diff(lamb_axis)) > tol_equi:
        delta = np.min(np.diff(lamb_axis))
        newAx = np.arange(lamb_axis[0], lamb_axis[-1], delta)
        Intfunc = interp1d(lamb_axis, values, kind='cubic', bounds_error=False, fill_value=0)
        newVals = Intfunc(newAx)
        newVals[newVals < 0] = 0.0
    else:
        newAx = lamb_axis
        newVals = values

    return newAx, newVals


def interpol_Val(M, lambda_M, delta_b, tol_equi=1e-5):
    deltaM = np.min(np.diff(lambda_M))
    if abs(deltaM - delta_b) > tol_equi or np.ptp(np.diff(lambda_M)) > tol_equi:
        lm_interp = np.arange(lambda_M[0], lambda_M[-1], delta_b)
        spline_M = interp1d(lambda_M, M, bounds_error=False, kind='cubic', fill_value=0)
        M_interp = spline_M(lm_interp)
        M_interp[M_interp < 0] = 0.0
    else:
        lm_interp = lambda_M
        M_interp = M

    return lm_interp, M_interp


def fix_inputs(lambda_bandpass, bandpass, lambda_spectrum, spectrum):
    lb, b = interpol_bandpass(lambda_bandpass, bandpass)
    deltab = lb[1]-lb[0]
    intb = np.trapz(b, dx=deltab)
    if abs(1 - intb) > 1e-4:
        b = b / intb

    lM, M = interpol_Val(spectrum, lambda_spectrum, deltab)

    if len(M) < len(b):
        leng_diff = len(b) - len(M)
        padd0 = np.arange(lM[0] - ((leng_diff / 2) + 1) * deltab, lM[0], deltab)
        padd1 = np.arange(lM[-1] + deltab, lM[-1] + (leng_diff / 2) * deltab, deltab)
        M = np.hstack((np.zeros(int(leng_diff / 2)), M))
        M = np.hstack((M, np.zeros(len(b) - len(M))))
        lM = np.hstack((padd0, lM, padd1))

    return lb, b, lM, M
