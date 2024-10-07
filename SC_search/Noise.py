try: 
    import cupy as np
    import numpy as numpy

except Exception as e:
    print('Cupy not installed')
    import numpy as np
    import numpy as numpy


import matplotlib.pyplot as plt 

import copy 

from  .Waveforms import Constants as const

# TODO: Make the functions GPU/CPU agnostic, this isnt a priority since the noise functions are only every called ~once in inference/searches

def psd_AEX(f, Sdisp, Sopt, L=2.5e9):
    """
    Calculates the analytical noise psd in AX,EX channel.

    Args:
        f (float or array of floats): Frequencies [Hz]
        Sdisp: Displacement noise
        Sopt: Optical noise
        L (float, optional): Mean arm length [m]

    Returns:
        float or array of floats: Analytical noise psd in AX,EX channel
    """
    taufL = 2 * np.pi * f * L /const.clight
    sin2taufL = np.sin(taufL)**2
    costaufL = np.cos(taufL)
    out = (8 * sin2taufL * ((6 + 4 * costaufL + 2 * np.cos(2 * taufL)) * Sdisp
                            + (2 + costaufL) * Sopt))

    return out
def psd_TX(f, Sdisp, Sopt, L=2.5e9):
    """
    Calculates the analytical noise psd in the TX channel.

    Args:
        f (float or array of floats): Frequencies [Hz]
        Sdisp: Displacement noise
        Sopt: Optical noise
        L (float, optional): Mean arm length [m]

    Returns:
        float or array of floats: Analytical noise psd in the TX channel
    """
    halftaufL = np.pi * f * L / const.clight
    sin2halftaufL = np.sin(halftaufL)**2
    out = (128 * (np.cos(halftaufL) * sin2halftaufL)**2
           * (4 * sin2halftaufL * Sdisp + Sopt))
    return out

def Sdisp_SciRD(f, L=2.5e9):
    """
    Calculates the analytical displacement noise in SciRD document.

    Args:
        f (float or array of floats): Frequencies [Hz]
        L (float): Mean arm length [m]

    Returns:
        float or array of floats: Analytical displacement noise in SciRD document (3e-15 m/s^2/sqrt(Hz))
    """
    freqs = copy.deepcopy(f)
    out = 9.e-30 * (1 + (4.e-4 / freqs)**2) / (L**2 * (2 * np.pi * freqs)**4)
    return out
def Sopt_SciRD(f, L=2.5e9):
    """
    Calculates the analytical optical noise in SciRD document.

    Args:
        f (float or array of floats): Frequencies [Hz]
        L (float): Mean arm length [m]

    Returns:
        float or array of floats: Analytical optical noise in SciRD document (15 pm / sqrt(Hz))
    """
    return np.full(np.size(f), 2.25e-22/L**2)


def XYZ2AET(X, Y, Z):
    """
    Convert TDI from XYZ to AET.

    Args:
        X (scalar or NumPy Array): X component
        Y (scalar or NumPy Array): Y component
        Z (scalar or NumPy Array): Z component

    Returns:
        A (float or NumPy Array): A component (same dimensions as X, Y, Z)
        E (float or NumPy Array): E component (same dimensions as X, Y, Z)
        T (float or NumPy Array): T component (same dimensions as X, Y, Z)
    """

    A = (Z - X) / 2.0**0.5
    E = (X - 2.0*Y + Z) / 6.0**0.5
    T = (X + Y + Z) / 3.0**0.5

    return A, E, T


def noise_realization(psd, T):
    """
    Generates a random, Gaussian noise realization from a known PSD. 

    Args:
        psd (array of floats): Noise PSD
        T (float): Observation time [s] ( = 1 / frequency spacing )
        df (float): Sampling frequency [Hz] ( = 1 / LISA sampling cadence )
            This is only needed if FD=False.

    Returns:
        array of floats: Noise realization for the given PSD
    """
    # Generate a white noise realization in the frequency domain
    wgn_realization = white_noise_realization(T,psd.size)
    # Scale using PSD
    cgn_realization = psd**0.5 * wgn_realization
    return cgn_realization
    
    
def white_noise_realization(T, fs=None):
    """
    Generate a white noise realization.

    Args:
        T (float): The duration of the white noise realization.
        fs (float, optional): The size of the frequency array to associate with the realization. Default is None.

    Returns:
        white_noise (ndarray): The generated white noise realization.

    """
    norm1 = 0.5 * T**0.5
    re1, im1 = np.random.normal(loc=0, scale= norm1, size=(2, fs))
    white_noise = re1 + 1j * im1
    # set DC and Nyquist = 0
    white_noise[0] = 0
    # no Nyquist frequency when N=odd
    if np.mod(fs, 2) == 0:
        white_noise[-1] = 0
    # python: transpose for use with infft
    white_noise = np.transpose(white_noise)
    return white_noise


# Confusion noise - taken from Balrog code
# Updated to 2103.14598 + 2108.01167

# High frequency sensitivity factor
highFreqFactor = 0.6

# Confusion noise is estimated either using a running mean or median on the power spectrum of unresolvable GBs.
# The form of the confusion noise model is in equattion 6 of arxiv:2103.14598

# Confusion noise parameter from 2103.14598
Sconf_a1_thresh5_runningMean = -0.16
Sconf_ak_thresh5_runningMean = -0.34
Sconf_b1_thresh5_runningMean = -2.78
Sconf_bk_thresh5_runningMean = -2.53
Sconf_A_thresh5_runningMean = 1.15e-44
Sconf_f2_thresh5_runningMean = 0.59e-3
Sconf_alpha_thresh5_runningMean = 1.66

Sconf_a1_thresh7_runningMean = -0.25
Sconf_ak_thresh7_runningMean = -0.27
Sconf_b1_thresh7_runningMean = -2.70
Sconf_bk_thresh7_runningMean = -2.47
Sconf_A_thresh7_runningMean = 1.14e-44
Sconf_f2_thresh7_runningMean = 0.31e-3
Sconf_alpha_thresh7_runningMean = 1.80


# By default the running medians are used, not the running means. 
Sconf_a1_thresh5_runningMedian = -0.15
Sconf_ak_thresh5_runningMedian = -0.34
Sconf_b1_thresh5_runningMedian = -2.78
Sconf_bk_thresh5_runningMedian = -2.55
Sconf_A_thresh5_runningMedian = 1.14e-44
Sconf_f2_thresh5_runningMedian = 0.59e-3
Sconf_alpha_thresh5_runningMedian = 1.66

Sconf_a1_thresh7_runningMedian = -0.15
Sconf_ak_thresh7_runningMedian = -0.37
Sconf_b1_thresh7_runningMedian = -2.72
Sconf_bk_thresh7_runningMedian = -2.49
Sconf_A_thresh7_runningMedian = 1.15e-44
Sconf_f2_thresh7_runningMedian = 0.67e-3
Sconf_alpha_thresh7_runningMedian = 1.56


Sconf_a1_default = Sconf_a1_thresh5_runningMedian
Sconf_ak_default = Sconf_ak_thresh5_runningMedian
Sconf_b1_default = Sconf_b1_thresh5_runningMedian
Sconf_bk_default = Sconf_bk_thresh5_runningMedian
Sconf_A_default = Sconf_A_thresh5_runningMedian
Sconf_f2_default = Sconf_f2_thresh5_runningMedian
Sconf_alpha_default = Sconf_alpha_thresh5_runningMedian


# Not found in 2103.14598, assumed to be 1 year
Tref = const.YRSID_SI

## Note: the confusion noise fit that has been implemented here has been computed for a *specific* instrumental noise model, if you want to use a different instrumental 
## noise model, you will need to recompute the confusion noise fit for that model, you cannot just supply a different instrumental noise into the functions here. 

def SOMS_VarLength(f, L):
    """Instrumental Optical Metrology System (OMS) noise PSD, at a given frequency and armlength L.

    Equation 9 from arxiv:2108.01167,

    Note, we have an extra factor of 1/L^2 compared to the original equation in the paper, 
    we want a dimensionless quantity here, while Equation 9 is in units of displacement.

    Args:
        f (array): frequency in Hz
        L (float): armlength in light-seconds
    Returns:
        S_OMS (array): OMS noise PSD at the given frequencies and armlength
    """
    freqs = np.copy(f)
    
    # Set the minimum frequency to avoid division by zero
    freqs[freqs <= 1.e-8] = 1.e-8

    S_OMS = (15.e-12/L)**2*(1. + (2e-3/freqs)**4)

    return S_OMS


def SOMS(f):
    """Explicitly asks for the OMS noise PSD at the mean armlength L_mean, which is hardcoded at the top of this file.

    Args:
        f (array): frequency in Hz
    Returns:
        SOMS_VarLength (array): OMS noise  PSD at the given frequencies and mean armlength
    """
    return SOMS_VarLength(f, L_mean)
    
    
    
def Sacc_VarLength(f, L):
    """Acceleration noise PSD, at a given frequency and armlength L.

    Equation 12 from arxiv:2108.01167

    Note, we have an extra factor of 1/L^2 compared to the original equation in the paper, 
    we want a dimensionless quantity here, while Equation 12 is in units of displacement.

    Args:
        f (array): frequency in Hz
        L (float): armlength in light-seconds
    Returns:
        S_acc (array): Acceleration noise PSD at the given frequencies and armlength
    """
    freqs = np.copy(f)
    
    # Set the minimum frequency to avoid division by zero
    freqs[freqs <= 1.e-8] = 1.e-8

    S_acc =  (3.e-15/(L * (2.*np.pi*freqs)**2))**2 * (1. + (0.4e-3/freqs)**2) * (1. + (freqs/8.e-3)**4)

    return S_acc


def Sacc(f):
    """Explicitly asks for the acceleration noise PSD at the mean armlength L_mean, which is hardcoded at the top of this file.

    Args:
        f (array): frequency in Hz
    Returns:
        Sacc_VarLength (array): acceleration noise PSD at the given frequencies and mean armlength
    """
    return Sacc_VarLength(f, L_mean)


def Sinst_VarLength(f, L):
    """Combined instrumental noise PSD, at a given frequency and armlength L.

    Appendix B from arxiv: 2108.01167. 

    Args:
        f (array): frequency in Hz
        L (float): armlength in light-seconds
    Returns:
        S_n (array): Instrumental noise PSD at the given frequencies and armlength
    """

    freqs = np.copy(f)
    # Set the minimum frequency to avoid division by zero
    freqs[freqs <= 1.e-8] = 1.e-8
    
    S_n = (4.*Sacc_VarLength(f, L) + SOMS_VarLength(f, L)) * (1. + highFreqFactor*(2.*np.pi*freqs*L/const.clight)**2)

    return S_n

def Sinst(f):
    """Explicitly asks for the instrumental noise PSD at the mean armlength L_mean.

    Args:
        f (array): frequency in Hz
    Returns:
        Sinst_VarLength (array): instrumental noise PSD at the given frequencies and mean armlength
    """
    return Sinst_VarLength(f, L_mean)
    
    
def Sconf_f1(a1, b1, Tobs):
    """Calculate confusion noise PSD parameter f1 from equation 7 of arxiv:2103.14598.

    Function primarily of Tobs.
    
    Note that the normalisation wrt the reference time is assumed to be 1 year, hardcoded at top of file. 

    Args:
        a1 (float): parameter a1
        b1 (float): parameter b1
        Tobs (float): observation time in seconds
    Returns:
        f1 (float): confusion noise parameter f1
    """
    f1 = 10.**b1 * (Tobs/Tref)**a1
    return f1
    
def Sconf_fknee(ak, bk, Tobs):
    """Calculate confusion noise PSD parameter fk from equation 7 of arxiv:2103.14598. 
    
    Function primarily of Tobs.

    Note that the normalisation wrt the reference time is assumed to be 1 year, hardcoded at top of file. 

    Args:
        ak (float): parameter a1
        bk (float): parameter b1
        Tobs (float): observation time in seconds
    Returns:
        fknee (float): confusion noise parameter fk
    """
    fknee = 10.**bk * (Tobs/Tref)**ak
    return fknee


def Sconf_VarParams(f, A, f1, alpha, fknee, f2):
    """Confusion noise PSD, with given fitted parameters.

    Equation 6 of arxiv:2103.14598

    Args:
        f (array): frequency in Hz
        A (float): amplitude parameter in confusion noise model
        f1 (float): parameter f1 in confusion noise model
        alpha (float): parameter alpha in confusion noise model
        fknee (float): parameter fknee in confusion noise model
        f2 (float): parameter f2 in confusion noise model

    Returns:
        S_gal (array): confusion noise PSD at the given frequencies and parameters
    """
    freqs = np.copy(f)
    
    # Set the minimum frequency to avoid division by zero
    freqs[freqs <= 1.e-8] = 1.e-8

    S_gal = (0.5*A) * freqs**(-7./3.) * np.exp(-(freqs/f1)**alpha) * (1. + np.tanh((fknee - freqs) / f2))

    return S_gal



def Sconf_VarFit(f, A, a1, b1, alpha, ak, bk, f2, Tobs):
    """Confusion noise PSD, wihtout having computed f1 and fknee beforehand (Function of Tobs).

    Args:
        f (array): frequency in Hz
        A (float): amplitude parameter in confusion noise model
        a1 (float): parameter a1 in confusion noise model
        b1 (float): parameter b1 in confusion noise model
        alpha (float): parameter alpha in confusion noise model
        ak (float): parameter ak in confusion noise model
        bk (float): parameter bk in confusion noise model
        f2 (float): parameter f2 in confusion noise model
        Tobs (float): observation time in 
        
    Returns:
        S_conf (array): confusion noise PSD at the given frequencies and parameters
    
    """
    # Calculate f1 and fknee at T_obs
    f1_fit = Sconf_f1(a1, b1, Tobs)
    fknee_fit = Sconf_fknee(ak, bk, Tobs)

    # Compute confusion noise at this T_obs
    S_conf = Sconf_VarParams(f, A, f1_fit, alpha, fknee_fit, f2)
    
    return S_conf
    
def Sconf(f, Tobs):
    """Wraps Sconf_VarFit, explicitly asking for the confusion noise PSD. Using the default values of all confusion noise parameters from 2103.14598

    Args:
        f (array): frequency in Hz
        Tobs (float): observation time in seconds
    Returns:
        Sconf_VarFit (array): confusion noise PSD at the given frequencies and observation time    
    """
    return Sconf_VarFit(f, Sconf_A_default, Sconf_a1_default, Sconf_b1_default, Sconf_alpha_default, Sconf_ak_default, Sconf_bk_default, Sconf_f2_default, Tobs)

def Sconf_ratio(f, Tobs):
    """Ratio of the confusion noise to the instrumental noise, at a given frequency and observation time.

    Args:
        f (array): frequency in Hz
        Tobs (float): observation time in seconds
    Returns:
        Sconf (array): confusion noise to instrumental noise ratio at the given frequencies and observation time
    """
    return Sconf(f, Tobs)/Sinst(f)


# Add confusion noise to the psd in Sn with frequencies in freqs,
def Add_confusion(f, Sn, Tobs):
    """Add confusion noise to the instrumental noise PSD.

    Args:
        f (array): frequency in Hz
        Sn (array): instrumental noise PSD
        Tobs (float): observation time in seconds
    Returns:
        Sout (array): total noise PSD, sum of confusion and instrumental noise at the given frequencies and observation time
    
    """
    # Sc here is actually Sc/Sn allowing for S_out 
    Sc = Sconf_ratio(f, Tobs)
    Sout = Sn*(1. + Sc)
    return Sout
