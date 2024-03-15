'''
Utility file contains useful utility functions for the code.
'''
from .Semi_Coherent_Functions import noise_weighted_inner_product
import numpy as np 
def component_masses_from_chirp_eta(mchirp, eta):
    """
    Calculate the component masses of a binary system from the chirp mass and symmetric mass ratio.

    Parameters:
        mchirp (float): The chirp mass of the binary system.
        eta (float): The symmetric mass ratio of the binary system.

    Returns:
        tuple: A tuple containing the component masses (m1, m2) of the binary system.
    """
    mtotal = mchirp / eta**(3/5)
    m1 = 0.5 * mtotal * (1.0 + (1.0 - 4.0 * eta)**0.5)
    m2 = 0.5 * mtotal * (1.0 - (1.0 - 4.0 * eta)**0.5)
    return m1, m2

def chirp_mass_eta_from_component_mass(m1, m2):
    """
    Calculate the chirp mass and symmetric mass ratio from the component masses.

    Parameters:
        m1 (float): Mass of the first component.
        m2 (float): Mass of the second component.

    Returns:
        tuple: A tuple containing the chirp mass and symmetric mass ratio.
        - mc (float): Chirp mass.
        - symmetric_mass_ratio (float): Symmetric mass ratio.
    """
    mc = ((m1 * m2) ** (3 / 5)) / (m1 + m2) ** (1 / 5)
    symmetric_mass_ratio = (m1 * m2) / (m1 + m2) ** 2
    return mc, symmetric_mass_ratio

def match(h1, h2, df, psd_array, phase_maximize=False):
    """
    Calculates the match between two waveforms using the noise-weighted inner product.

    Parameters:
        h1 (array-like): The first waveform.
        h2 (array-like): The second waveform.
        df (float): The frequency resolution.
        phase_maximize (bool, optional): Whether to maximize the phase. Defaults to False.

    Returns:
        float: The match between the two waveforms.

    """
    numerator = noise_weighted_inner_product(h1, h2, df, psd_array, phase_maximize=phase_maximize)
    denominator = np.sqrt(noise_weighted_inner_product(h1, h1, df, psd_array) * noise_weighted_inner_product(h2, h2, df, psd_array))

    overlap = numerator / denominator
    
    return np.abs(overlap)

def TaylorF2Ecc_mc_eta_to_m1m2(parameters):
    '''
    Parameter transforms are hardcoded in to:
    - Polarization shift to match Balrog convention
    - Mc,eta->m1,m2

    Args:
        parameters (array): Waveform parameters. 
            parameters[0]: Mc
            parameters[1]: eta
            parameters[6]: polarization (BBHx convention)
    
    Returns:
        parameters (array): Waveform parameters transformed to match Balrog convetion
            parameters[0]: m1
            parameters[1]: m2
            parameters[6]: polarization (Balrog convention)
    
    '''

    # Polarization convention (We are sticking to Balrog)
    parameters[6] = -(parameters[6]-np.pi/2)
    
    # Mc,eta->m1,m2
    parameters[0],parameters[1] = component_masses_from_chirp_eta(parameters[0],parameters[1])

    return(parameters)
