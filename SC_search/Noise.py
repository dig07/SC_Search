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
