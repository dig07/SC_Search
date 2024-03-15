try: 
    # Look for cupy, if no Cupy, use numpy
    import cupy as np 
except ImportError:
    import numpy as np


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


def semi_coherent_logl(signal,data,psd_array,d_inner_d,num_segments=1):
    """
    Semi-coherent log likelihood function for a given nunmber of segments

    Args:
        signal (array-like): The signal model. Shape: (3,#FFTgrid).
        data (array-like): The data. Shape: (3,#FFTgrid).
        psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).
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
        float: The semi-coherent log likelihood
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
        upsilon += (noise_weighted_inner_product(signal_segment,data_segment,df,psd_segment,phase_maximize=True)/(cp.sqrt(
                        noise_weighted_inner_product(signal_segment,signal_segment,df,psd_segment,phase_maximize=True))))**2

    return(upsilon.item())

