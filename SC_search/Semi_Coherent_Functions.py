try: 
    # Look for cupy, if no Cupy, use numpy
    import cupy as np 
except ImportError:
    import numpy as np
import math
import warnings
warnings.simplefilter('ignore', RuntimeWarning)


def noise_weighted_inner_product(a, b, df, psd, phase_maximize=False):
    """
    Calculates the noise-weighted inner product between two frequency domain series.

    Note: Phase maximisation assumes one harmonic per signal, and is not yet generalised.

    Args:
        a (array-like): First array (complex) Shape: (3,#FFTgrid).
        b (array-like): Second array (complex) Shape: (3,#FFTgrid).
        df (float): Frequency step size (1/Tobs).
        psd (array-like): Power Spectral Density Shape: (3,#FFTgrid). 
        phase_maximize (bool, optional): If True, the inner product is phase maximized. 
                                        If False, the inner product is not phase maximized. 
                                        Defaults to False.

    Returns:
        float: The noise-weighted inner product (optionally phase maximised).

    """
    if phase_maximize == False:
        inner_prod = 4*np.real(np.sum((a*b.conj()/psd)*df))
    else:
        inner_prod = 4*np.abs(np.sum((a*b.conj()/psd)*df))
    return(inner_prod)


def equal_SNR_segmentation(model,psd,segment_number):
    """
    Calculates equal network SNRsq chunks in the Fourier domain
    
    Args:   
        model (array): Timeseries in each TDI channel to calculate equal SNR chunks for. Shape: (3, #data_points)
        psd (array): The PSD in each channel. Shape: Same as signals.
        segment_number (int): Number of segments to split the signal up into.
    
    Returns:
        boundary_indexes (array): Boundary indices for the data chunks.
    """
    # the SNRsq at each frequency (summed across the network)
    net_SRNsq_array = np.sum( (np.abs(model)**2) /psd, axis=0)
    
    # the cumulative network SNRsq with frequency
    net_SNRsq_cum = np.cumsum(net_SRNsq_array)
    
    # the total network SNRsq
    net_SNRsq = net_SNRsq_cum[-1]
    
    # the desired SNRsq chunk for each segment
    SNRsq_chunks = net_SNRsq / segment_number
    
    # # find the boundary indices for the segments
    SNRsq_breaks = SNRsq_chunks * np.arange(segment_number+1)

    # Search sorted finds the indexes 
    boundary_indexes = np.searchsorted(net_SNRsq_cum,SNRsq_breaks)

    boundary_indexes[-1] = model.shape[1] # this a weird thing with numpy indices
    return boundary_indexes


def semi_coherent_logl(signal,data,psd_array,df,d_inner_d,num_segments=1):
    """
    Semi-coherent log likelihood function for a given nunmber of segments

    As defined in arXiv:2305.18048 eqn 9.

    Args:
        signal (array-like): The signal model. Shape: (3,#FFTgrid).
        data (array-like): The data. Shape: (3,#FFTgrid).
        psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).
        df (float): Frequency step size (1/Tobs).
        d_inner_d (float): The inner product of the data with itself.
        num_segments (int, optional): The number of segments to split the signal into. Defaults to 1.
    Returns:
        float: The semi-coherent log likelihood
    """
    segment_indices = equal_SNR_segmentation(signal,psd_array,num_segments)

    signal_split = [signal[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]

    psd_split = [psd_array[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]

    data_split = [data[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]    

    logl = 0 

    for segment_index in range(num_segments):

        signal_segment = signal_split[segment_index]
        psd_segment = psd_split[segment_index]
        data_segment = data_split[segment_index]

        #h_inner_d
        logl += noise_weighted_inner_product(signal_segment,data_segment,df,psd_segment,phase_maximize=True) 

    logl += -1/2*(noise_weighted_inner_product(signal,signal,df,psd_array,phase_maximize=False) + d_inner_d)

    return(logl.item())


def upsilon_func(signal,data,psd_array,df,num_segments=1):
    """
    Semi-coherent search statistic Upsilon from arXiv:1705.04259v2 eqn 67.

    Args:
        signal (array-like): The signal model. Shape: (3,#FFTgrid).
        data (array-like): The data. Shape: (3,#FFTgrid).
        psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).
        df (float): Frequency step size (1/Tobs).
        num_segments (int, optional): The number of segments to split the signal into. Defaults to 1.
    Returns:
        float: The semi-coherent quantity upsilon evaluated with given num_segments
    """
    segment_indices = equal_SNR_segmentation(signal,psd_array,num_segments)

    signal_split = [signal[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]

    psd_split = [psd_array[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]

    data_split = [data[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]    

    upsilon = 0 

    for segment_index in range(num_segments):

        signal_segment = signal_split[segment_index]
        psd_segment = psd_split[segment_index]
        data_segment = data_split[segment_index]

        #h_inner_d/sqrt(h_inner_h)
        upsilon += (noise_weighted_inner_product(signal_segment,data_segment,df,psd_segment,phase_maximize=True)/(np.sqrt(
                        noise_weighted_inner_product(signal_segment,signal_segment,df,psd_segment,phase_maximize=True))))**2

    return(upsilon.item())

def semi_coherent_match(signal,data,psd_array,df,num_segments=1):
    """
    Semi-coherent match, using semi-coherent inner product. 

    Args:
        signal (array-like): The signal model. Shape: (3,#FFTgrid).
        data (array-like): The data. Shape: (3,#FFTgrid).
        psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).
        df (float): Frequency step size (1/Tobs).
        num_segments (int, optional): The number of segments to split the signal into. Defaults to 1.
    Returns:
        float: The semi-coherent log likelihood
    """
    segment_indices = equal_SNR_segmentation(signal,psd_array,num_segments)

    signal_split = [signal[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]

    psd_split = [psd_array[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]

    data_split = [data[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]    

    smatch = 0 

    for segment_index in range(num_segments):

        signal_segment = signal_split[segment_index]
        psd_segment = psd_split[segment_index]
        data_segment = data_split[segment_index]

        #h_inner_d/sqrt(h_inner_h)
        smatch += noise_weighted_inner_product(signal_segment,data_segment,df,psd_segment,phase_maximize=True)
    
    smatch *= 1/np.sqrt(noise_weighted_inner_product(signal,signal,df,psd_array,phase_maximize=True)*noise_weighted_inner_product(data,data,df,psd_array,phase_maximize=True))

    return(smatch.item())

def coherent_match(signal,data,psd_array,df):
    """
    Standard coherent normalised match quantity

    Args:
        signal (array-like): The signal model. Shape: (3,#FFTgrid).
        data (array-like): The data. Shape: (3,#FFTgrid).
        psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).
        df (float): Frequency step size (1/Tobs).
    Returns:
        float: The coherent log likelihood
    """

    smatch = noise_weighted_inner_product(signal,data,df,psd_array,phase_maximize=False)/np.sqrt(noise_weighted_inner_product(signal,signal,df,psd_array,phase_maximize=True)*noise_weighted_inner_product(data,data,df,psd_array,phase_maximize=True))

    return(smatch.item())

def vanilla_log_likelihood(signal,data,df,psd_array):
    """
    Standard Gaussian likelihood function. 

    Args:
        signal (array-like): The signal model. Shape: (3,#FFTgrid).
        data (array-like): The data. Shape: (3,#FFTgrid).
        psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).

    Returns:
        float: The standard Gaussian log likelihood
    
    """

    res = data - signal 

    logl = -1/2*noise_weighted_inner_product(res,res,df,psd_array,phase_maximize=False)#

    return(logl.item())


def upsilon_func_masking(signal,data,psd_array,df,num_segments=1):
    """
    Semi-coherent search statistic Upsilon from arXiv:1705.04259v2 eqn 67.
    Using array masking to avoid for loops. 

    Note: This function becomes faster than the for loop implementation when the number of segments is large.
        At low number of segments this is actually slower than the for loop implementation: upislon_func.

    Warning: Very memory expensive at times.

    Args:
        signal (array-like): The signal model. Shape: (3,#FFTgrid).
        data (array-like): The data. Shape: (3,#FFTgrid).
        psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).
        df (float): Frequency step size (1/Tobs).
        num_segments (int, optional): The number of segments to split the signal into. Defaults to 1.
    Returns:
        float: The semi-coherent quantity upsilon evaluated with given num_segments
    """
    segment_indices = equal_SNR_segmentation(signal,psd_array,num_segments)

    signal_split = [signal[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]

    psd_split = [psd_array[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]

    data_split = [data[:,segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]    

    # Length of each segment in frequency points
    lengths = np.diff(segment_indices)

    # Maximum length of any segment in the freq domain, this is what we have to pad to
    max_length = np.max(lengths).item()

    # Fill in padded arrays
    signal_array = np.zeros((3,num_segments,max_length),complex)
    data_array = np.zeros((3,num_segments,max_length),complex)
    psd_array= np.zeros((3,num_segments,max_length),float)

    # Frequency mask for every segments filling in 
    mask = np.arange(max_length) < lengths[:,None]

    # Maybe we can use the mask directly in the sum? Not sure

    # fill in padded arrays
    signal_array[:,mask] = np.concatenate(signal_split,axis=1)
    data_array[:,mask] = np.concatenate(data_split,axis=1)
    psd_array[:,mask] = np.concatenate(psd_split,axis=1)

    # # Fill extra padded values with infs for psd, 0+0.j/np.inf = 0 in the inner product
    # psd_array[:,~mask] = np.inf # Cannot do this with cupy, for some reason it doesn't like infs in the fractions when it involes complex numbers

    
    # np.nan_to_num works in both cupy and numpy, converts all the nans for the complex divide by 0 to 0.

    # j: segment, i: channel, k: frequency point
    # The nan to num function deals with the cases where psd = 0 and sets these values in the inner product to 0. #
    # What is returned here is the components of upsilon for each segment. (j is the segment index)
    h_inner_d = 4*np.abs(np.einsum('ijk,ijk->j',signal_array,np.nan_to_num(data_array.conj()/psd_array)))*df

    h_inner_h = 4*np.abs(np.einsum('ijk,ijk->j',signal_array,np.nan_to_num(signal_array.conj()/psd_array)))*df
    
    # Compute upsilon
    upsilon = np.sum((h_inner_d/np.sqrt(h_inner_h))**2)

    return(upsilon.item())
