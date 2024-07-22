from typing import Tuple
import numpy as np

def calcOptCurv(SH, signal, flipped_psf, autostop = True, orig_mode = False):
    '''
    Calculates the RMS progress in the Richardson-Lucy deconvolution and its 
    curvature to determine the best number of iterations.

    Parameters
    ----------
    SH : ndarray of shape (K, L)
        with K the number of iterations and L the wavelength.
    signal : ndarray of shape (S,)
        the measured signal values.
    flipped_psf : ndarray of shape (K,)
        the reversed bandpass function.
    autostop : boolean, optional
        whether to calculate the maximum curvature. The default is True.
    orig_mode : boolean, optional
        if True then the global maximum curvature is chosen, if False then
        the optimal point is chosen as the local maximum with the smallest
        error. The default is False.

    Returns
    -------
    RMS : ndarray of shape (K,)
        the RMS of progress.
    curv : ndarray of shape (K,)
        the curvature of the RMS of progress.
    indopt : integer
        the optimal stopping point.
    '''
    
    from scipy.interpolate import InterpolatedUnivariateSpline
    from scipy.signal import argrelmax
    
    def calcEsignalE(SHcandidates: np.ndarray) -> np.ndarray:
        '''
        Calculates the error of the estimated measurement for each candidate.

        Parameters
        ----------
        SHcandidates : ndarray of shape (m, L)
            a matrix of candidates with m the number of candidate solutions.

        Returns
        -------
        EstsignaleasDiff : ndarray of shape (m,)
            the error of the estimated measurement for each candidate.
        '''
        m = SHcandidates.shape[0]
        EstsignaleasDiff = np.zeros((m,))
        for r in range(m):
            signalest = np.convolve(SHcandidates[r], flipped_psf, 'same')
            signalest /= np.trapz(signalest, dx=signal[1] - signal[0])
            EstsignaleasDiff[r] = np.sum((signal - signal) ** 2 / signalest)
        return EstsignaleasDiff

    # fixed values
    min_curv_iter = 10
    num_iter, lsignal = SH.shape
    
    print('Calculating optimal stopping iteration using curvature information\n')
    
    # RMS progress calculation
    RMS = np.sqrt(np.mean(np.square(np.diff(SH, axis=0)), axis=1))
    print('done\n')

    if np.any(np.isnan(RMS)) or np.any(np.isinf(RMS)) or np.min(RMS) == np.max(RMS):
        return RMS, RMS, -1

    # getting the curvature of the RMS progress
    if num_iter > min_curv_iter:
        print('Calculating curvature of rms progress.\n')
        iters = np.arange(1, num_iter)
        q = np.linspace(np.log10(iters[0]), np.log10(iters[-1]), len(RMS))
        dq = q[1] - q[0]
        spline_qrms = InterpolatedUnivariateSpline(np.log10(iters), np.log10(RMS))
        qrms = spline_qrms(q)
        
        # check if any NaN values were produced during the spline interpolation
        num_nans = np.count_nonzero(np.isnan(qrms))
        if num_nans > 0:
            print(f'Warning: In calcOptCurv interpolation of rms values resulted in {num_nans} NaN entries!')
            print('Results may be inaccurate.\n')
        
        # calculate the curvature of the RMS progress using the second derivative
        # the equation is described in the paper "The L-curve and its use in the
        # numerical treatment of inverse problems (P. C. Hansen, 2000)"
        first_derivative = np.gradient(qrms, dq)
        snd_derivative = np.gradient(first_derivative, dq)
        curv = snd_derivative / (1 + first_derivative ** 2) ** 1.5
        qmin = np.nonzero(q >= np.log10(min_curv_iter))[0][0]

        if autostop:
            # calculate optimal q-value
            qmax = argrelmax(curv[:-1])[0]  # indices of relative maxima of curvature
            relevant_qmax = [qm for qm in qmax if qm >= qmin]  # indices of relevant maxima of curvature
            if len(relevant_qmax) > 0:
                if orig_mode:  # global maximum
                    # find index corresponding to maximum curvature
                    qopt = q[curv[qmin:].argmax()+qmin]
                    indopt = np.round(10**qopt)
                else:  # maximum with smallest error in estimated measurement
                    indopts = []
                    cmax = curv[relevant_qmax]
                    # find indices of the top 3 maxima of curvature
                    inds = cmax.argsort()[-3:]
                    for k in inds:
                        # find q-value corresponding to each of the top 3 maxima
                        qopt = q[curv == curv[relevant_qmax[k]]]
                        qopt = qopt[0]
                        # transform q-value to iteration number
                        iopt = np.flatnonzero(np.log10(iters) <= qopt)
                        indopts.append(iters[iopt[-1]])
                    # calculate error of estimated measurement for the candidates
                    EsignalE = calcEsignalE(SH[indopts, :])
                    print("potential stopping points are:\n " + repr(indopts) + "\n")
                    # select the candidate with the smallest error
                    indopt = indopts[EsignalE.argmin()]
            else:
                indopt = -1
        else:
            indopt = num_iter
    else:
        print('Warning: signalaximum number of iterations is too small for calculation of curvature. Automatic stopping not available.\n')
        if autostop:
            indopt = -1
        else:
            indopt = num_iter
        curv = np.zeros_like(RMS)

    if indopt < 0:
        print('Calculation of optimal stopping using the max curvature method failed.\n')

    return RMS, curv, indopt


def rlMethod(signal: np.ndarray, psf: np.ndarray, num_iter: int =1500, autostop: bool = True, initialShat: np.ndarray =None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Richardson Lucy deconvolution for two samples that are equidistant and
    have the same wavelength step size.

    Parameters
    ----------
    signal : ndarray of shape (S,)
        the measured spectrum values.
    psf : ndarray of shape (K,)
        the bandpass function values.
    num_iter : integer, optional
        the maximum number of iterations. The default is 1500.
    autostop : boolean, optional
        whether to automatically find the optimal number of iterations. The 
        default is True.
    initialShat : ndarray of shape (I,), optional
        an initial estimation of the measured spectrum. The default is None.

    Raises
    ------
    ValueError
        The measured or bandpass function have negative values.

    Returns
    -------
    Shat : ndarray of shape (S,)
        the estimated spectrum.
    critVals1 : ndarray of shape (num_iter,)
        the RMS of progress.
    critVals2 : ndarray of shape (num_iter,)
        the curvature of the RMS of progress.
    '''
    # Check for negative values in signal and psf
    if np.any(signal < 0) or np.any(psf < 0):
        raise ValueError('Measured spectrum and bandpass function must not have negative values.')

    # Adjust length of signal and psf for proper convolution
    if signal.size < psf.size:
        leng_diff = psf.size - signal.size
        signal = np.pad(signal, (leng_diff, leng_diff), 'constant')

    # Allocate computer memory
    SH = np.zeros((num_iter + 1, signal.size))

    # Set initial estimate
    SH[0] = initialShat if isinstance(initialShat, np.ndarray) and len(initialShat) == signal.size else signal

    # Detect relevant region of measured spectrum
    indsignal = signal >= np.max(signal) * 1e-6
    if indsignal.sum() < len(signal) and autostop:
        print(f"Data contains {len(signal) - indsignal.sum()} values which are relatively close to zero. ")
        print("These will not be considered for the determination of the stopping point.")

    # Flip psf for convolution
    flipped_psf = psf[::-1]

    # set initial estimate
    if isinstance(initialShat, np.ndarray) and len(initialShat) == signal.size:
        SH[0] = initialShat
    else:
        print("User-supplied initial estimate does not match measured spectrum and will be ignored.")
        SH[0] = signal

    if autostop:
        print(f"Richardson-Lucy method calculating optimal stopping criterion using Optimal Curvature\nStep 1: Carry out {int(num_iter)} iterations.")
    else:
        print(f"Richardson-Lucy method with {int(num_iter)} iterations")

    # actual iterations
    flipped_psf = psf[::-1]

   # Iterate to estimate spectrum
    for r in range(1, num_iter + 1):
        temp1 = np.convolve(SH[r - 1], flipped_psf, mode='same')
        temp1 = np.divide(signal, temp1, out=np.zeros_like(signal), where=temp1 != 0)
        
        # Check for zero division errors
        count_bad = np.count_nonzero(temp1 == 0)
        if count_bad > 0:
            print(f'After convolution in RL iteration {r}, {count_bad} estimated values were set to zero.')
        temp1[np.logical_or(np.isnan(temp1), np.isinf(temp1))] = 0

        temp2 = np.convolve(temp1, psf, mode='same')
        tempSH = SH[r - 1] * temp2
        tempSH[np.logical_or(np.isnan(tempSH), np.isinf(tempSH))] = 0

        SH[r] = tempSH

    print('Done')

    # Calculate optimal stopping using Optimal Curvature
    print('Calculating optimal stopping using Optimal Curvature')
    critVals1, critVals2, indopt = calcOptCurv(SH[:, indsignal], signal[indsignal], flipped_psf, autostop)

    # If calculation of optimal stopping failed, use measured spectrum as best estimate
    if indopt < 0:
        print('Warning: Calculation of optimal stopping failed. Using measured spectrum as best estimate.')
        Shat = signal
    else:
        # If autostop, print optimal number of iterations
        if autostop:
            print(f'Optimal number of iterations = {int(indopt)}')
            Shat = np.squeeze(SH[indopt-1, :])
        else:
            Shat = SH[-1, :]
    
    return Shat, critVals1, critVals2    



