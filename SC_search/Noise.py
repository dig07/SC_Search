import numpy as np 
import matplotlib.pyplot as plt 

import numpy as np 
import copy 

from  .Waveforms import Constants as const

def psd_AEX(f, Sdisp, Sopt, L=2.5e9):
    """

    Parameters
    -------

    f : float or array of floats
        frequencies [Hz]

    Sdisp :
        displacement noise

    Sopt :
        optical noise

    L :
        mean arm length [m]

    Returns
    -------
    float or array of floats
        analytical noise psd in AX,EX channel

    """
    taufL = 2 * np.pi * f * L /c
    sin2taufL = np.sin(taufL)**2
    costaufL = np.cos(taufL)
    out = (8 * sin2taufL * ((6 + 4 * costaufL + 2 * np.cos(2 * taufL)) * Sdisp
                            + (2 + costaufL) * Sopt))

    return out
def psd_TX(f, Sdisp, Sopt, L=2.5e9):
    """

    Parameters
    -------

    f : float or array of floats
        frequencies [Hz]

    Sdisp :
        displacement noise

    Sopt :
        optical noise

    L :
        mean arm length [m]

    Returns
    -------
    float or array of floats
        analytical noise psd in TX channel

    """
    halftaufL = np.pi * f * L / c
    sin2halftaufL = np.sin(halftaufL)**2
    out = (128 * (np.cos(halftaufL) * sin2halftaufL)**2
           * (4 * sin2halftaufL * Sdisp + Sopt))
    return out

def psd_AEM(f, Sdisp, Sopt, L=2.5e9):
    """

    Parameters
    -------

    f : float or array of floats
        frequencies [Hz]

    Sdisp :
        displacement noise

    Sopt :
        optical noise

    L :
        mean arm length [m]

    Returns
    -------
    float or array of floats
        analytical noise psd in AX,EX channel

    """
    taufL = 2.*np.pi*f*L/c
    costaufL = np.cos(taufL)
    out = 2. * ((6 + 4 * costaufL + 2 * np.cos(2 * taufL))
                 * Sdisp + (2 + costaufL) * Sopt)
    return out

def psd_TM(f, Sdisp, Sopt, L=2.5e9):
    """

    Parameters
    -------

    f : float or array of floats
        frequencies [Hz]

    Sdisp :
        displacement noise

    Sopt :
        optical noise

    L :
        mean arm length [m]

    Returns
    -------
    float or array of floats
        analytical noise psd in TX channel

    """
    halftaufL = np.pi * f * L / c
    sin2halftaufL = np.sin(halftaufL)**2
    return 8. * sin2halftaufL * (4 * sin2halftaufL * Sdisp + Sopt)




def Sdisp_SciRD(f, L=2.5e9):
    """

    Parameters
    -------

    f : float or array of floats
        frequencies [Hz]

    L :
        mean arm length [m]


    Returns
    -------
    float or array of floats
        analytical displacement noise in SciRD document
        (3 10^-15 m / s^2 / sqrt(Hz))

    """
    freqs = copy.deepcopy(f)
    out = 9.e-30 * (1 + (4.e-4 / freqs)**2) / (L**2 * (2 * np.pi * freqs)**4)
    return out
def Sopt_SciRD(f, L=2.5e9):
    """

    Parameters
    -------

    f : float or array of floats
        frequencies [Hz]

    L :
        mean arm length [m]


    Returns
    -------
    float or array of floats
        analytical optical noise in SciRD document (15 pm / sqrt(Hz))

    """
    return np.full(np.size(f), 2.25e-22/L**2)


def XYZ2AET(X, Y, Z):
    """
    Convert TDI from XYZ to AET

    Parameters
    ----------
    X, Y, Z: scalars or NumPy Arrays

    Returns
    -------
    A, E, T: float or NumPy Array (same dimensions as X, Y, Z)
    """

    A = (Z - X) / 2.0**0.5
    E = (X - 2.0*Y + Z) / 6.0**0.5
    T = (X + Y + Z) / 3.0**0.5

    return A, E, T


def noise_realization(psd, T, df=None, fs=None, FD=True):
    """
    Generates a random, Gaussian noise realisation from a known PSD. 

    Parameters
    -------

    psd : array of floats
        noise psd

    T : float
        observation time [s] ( = 1 / frequency spacing )

    df : float
        the sampling frequency [Hz] ( = 1 / LISA sampling cadence )
        this is only needed if FD=False
        
    fs : array of floats
        frequencies that the psd is computed over
        this is only needed if FD=False to check requirement that 
        fs[0]==0
        
    FD : bool
        if true, then return output in the frequency domain
        if false, then take inverse Fourier transform and return in 
                  the time domain. Requires fs[0]==0. 

    Returns
    -------
    array of floats
        noise realization for the given psd

    """
    amp = np.random.normal(loc=0, scale=np.sqrt(T*psd/2.))
    phase = 2.*np.pi*np.random.random(size=len(psd))
    Sn = amp*np.exp(1j*phase)
    if FD: 
        return Sn
    else: 
        assert df!=None, "when generating time domain noise user must \
                          specify df sampling frequency"
        assert fs[0] == 0, "the frequency needs to start at zero for \
                            FD=False"

        dt = 1. / df
        return np.fft.irfft(Sn)/dt
