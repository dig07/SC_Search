import numpy as np 
import math 

def false_alarm_rate(upsilon,template_SNR,num_segments):
    """
    Calculate the false alarm rate for a given threshold and number of segments. 

    Uses analytical fit from arXiv:1705.04259 for false alarm rate.
    Note: This is using an analytical fit to the false alarm rate, using the central limit theorem. 
          and is accurate for N>70.
    
    Args:
        upsilon (float): The value of the upsilon statistic.
        template_SNR (float): The SNR of the template.
        num_segments (int): The number of segments used to calculate the upsilon statistic.

    Returns:
        float: The false alarm rate.
    """


    mu_k = 2.00
    sigma_k = 1.45

    mu_0 = num_segments*mu_k
    sigma_0 = num_segments*mu_k+template_SNR**2


    pf = 1/2*(math.erf((mu_0-upsilon)/(np.sqrt(2*sigma_0**2))) + 1)

    return(pf)