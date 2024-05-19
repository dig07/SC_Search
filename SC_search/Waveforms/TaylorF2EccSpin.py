import logging
# Imports both CuPy and Numpy for waveform generation. If CuPy is not found, Numpy is used. 
try:
    import cupy as np 
    logging.warning(' WAVEFORMS: CuPy found! Using CuPy for waveform generation')
    import numpy as numpy 
    use_GPU = True
    from pyWaveformBuild import direct_sum_wrap as direct_sum_wrap_gpu


except(ImportError, ModuleNotFoundError) as e:
    logging.warning(' WAVEFORMS: No CuPy, using Numpy for waveform generation')
    import numpy as np 
    import numpy as numpy
    use_GPU = False
    from pyWaveformBuild_cpu import direct_sum_wrap as direct_sum_wrap_cpu

try: 
    import alb_TDI_WFs as alb_TDI
    from albertos.TDI_functions import XYZ2AET
    logging.warning(' WAVEFORMS: Balrog found! Importing Balrog response as an option')
except(ImportError, ModuleNotFoundError) as e:
    logging.warning(' WAVEFORMS: No Balrog, Balrog response not available')
    

import scipy 
from numba import jit

# LISA imports (Balrog)
from . import Constants as const


# LISA imports (BBHx)
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.utils.interpolate import CubicSplineInterpolant
from bbhx.waveformbuild import TemplateInterpFD



### Constants used for waveform building

# Euler Mascheroni constant
gamma_e = const.gamma_e 
c = const.clight
G = const.G
MTsun = const.MTsun
pc = const.pc

# Armlength of LISA in seconds
Armlength = 2.5e+9/c

# BBHx prefactor to get strains into same units as Balrog (See PSD comparison, compare the PSDs from LISAtools, equivalent to frequency_mode == True from Balrog)
bbhx_pre_factor = 1/(2j*np.pi*Armlength)

@jit(nopython=True)
def calculate_T(v, v0, e0, eta, beta):
    '''
    Calculates the value of T using the (TaylorT2) Equation 6.7b from arXiv:1605.00304v2.

    Parameters:
      v (float or array of floats): (pi*M*f)^{1/3} with f as frequencies of the binary system.
      v0 (float): (pi*M*f_0)^{1/3}, with f_0 as the initial GW frequency of the binary
      e0 (float): The initial eccentricity of the binary system.
      eta (float): The symmetric mass ratio of the binary system.
      beta (float): The 1.5 PN term.

    Returns:
      T (float or array of floats): The calculated value(s) of T at fs provided into v.

    '''

    term1 = (1 + ((743 / 252) + (11 / 3) * eta) * v**2 - (32 / 5) * numpy.pi * v**3 + 
        ((3058673 / 508032) + (5429 / 504) * eta + (617 / 72) * eta**2) * v**4 + 
        (-(7729 / 252) + (13 / 3) * eta) * numpy.pi * v**5)
    
    term2 = ((-10052469856691 / 23471078400) + 
         (6848 / 105) * gamma_e + 
         (128 / 3) * numpy.pi**2 + 
         ((3147553127 / 3048192) - 
          (451 / 12) * numpy.pi**2) * eta - 
         (15211 / 1728) * eta**2 + 
         (25565 / 1296) * eta**3 + 
         (3424 / 105) * numpy.log(16 * v**2)) * v**6
    
    term3 = ((-15419335 / 127008) - (75703 / 756) * eta + (14809 / 378) * eta**2) * numpy.pi * v**7

    term4 = -(157 / 43) * e0**2 * (v0 / v)**(19/3) * (1 + ((17592719 / 5855472) + (1103939 / 209124) * eta) * v**2 + \
        ((2833 / 1008) - (197 / 36) * eta) * v0**2 - (2819123 / 384336) * numpy.pi * v**3 + \
        (377 / 72) * numpy.pi * v0**3 + ((955157839 / 302766336) + (1419591809 / 88306848) * eta + \
        (91918133 / 6307632) * eta**2) * v**4 + ((49840172927 / 5902315776) - (42288307 / 26349624) * eta - \
        (217475983 / 7528464) * eta**2) * v**2 * v0**2 + (-(1193251 / 3048192) - (66317 / 9072) * eta + \
        (18155 / 1296) * eta**2)*v0**4 -((166558393 / 12462660) + (679533343 / 28486080) * eta) * numpy.pi * v**5 + \
        (-(7986575459 / 387410688) + (555367231 / 13836096) * eta) * numpy.pi * v**3 * v0**2 + \
        ((6632455063 / 421593984) + (416185003 / 15056928) * eta) * numpy.pi * v**2 * v0**3 + \
        ((764881 / 90720) - (949457 / 22680) * eta) * numpy.pi * v0**5 +(-(2604595243207055311 / 16582316889600000) + (31576663 / 2472750) * gamma_e + \
        (924853159 / 40694400) * numpy.pi**2 + ((17598403624381 / 86141905920) - (886789 / 180864) * numpy.pi**2) * eta + \
        (203247603823 / 5127494400) * eta**2 + (2977215983 / 109874880) * eta**3 + \
        (226088539 / 7418250) * numpy.log(2) - (65964537 / 2198000) * numpy.log(3) + \
        (31576663 / 4945500) * numpy.log(16 * v**2)) * v**6 + ((2705962157887 / 305188466688) + (14910082949515 / 534079816704) * eta - \
        (99638367319 / 2119364352) * eta**2 - (18107872201 / 227074752) * eta**3) * v**4 * v0**2 -(1062809371 / 27672192) * numpy.pi**2 * v**3 * v0**3 + \
        (-(20992529539469 / 17848602906624) - (15317632466765 / 637450103808) * eta + \
        (8852040931 / 2529563904) * eta**2 + (20042012545 / 271024704) * eta**3) * v**2 * v0**4 + \
        ((26531900578691 / 168991764480) - (3317 / 126) * gamma_e + (122833 / 10368) * numpy.pi**2 + \
         ((9155185261 / 548674560) - (3977 / 1152) * numpy.pi**2) * eta - (5732473 / 1306368) * eta**2 - \
         (3090307 / 139968) * eta**3 + (87419 / 1890) * numpy.log(2) - (26001 / 560) * numpy.log(3) - \
         (3317 / 252) * numpy.log(16 * v0**2)) * v0**6)

    spin_aligned_term = 8*v**3*beta/5

    T = term1+term2+term3+term4+spin_aligned_term
    return T


@jit(nopython=True)
def F2EccPhase(freqs,eta,v,e0,v0,coallesence_phase,time_to_merger,beta):
    '''
    Calculates the value of Lambda_f using the (TaylorT2) Equation 6.26 from arXiv:1605.00304v2. Waveform Fourier phase for TaylorF2Ecc.

    Parameters:
      v (float or array of floats): (pi*M*f)^{1/3} with f as frequencies of the binary system.
      v0 (float): (pi*M*f_0)^{1/3}, with f_0 as the initial GW frequency of the binary
      e0 (float): The initial eccentricity of the binary system.
      eta (float): The symmetric mass ratio of the binary system.
      beta (float): The 1.5 PN term.

    Returns:
      psi (float or array of floats): The calculated value(s) of psi at fs provided into v.

    '''
    term1 = 3 / (128 * eta * v**5)
    
    term2 = (1 + (3715 / 756 + 55 / 9 * eta) * v**2 - 16 * numpy.pi * v**3 +
             (15293365 / 508032 + 27145 / 504 * eta + 3085 / 72 * eta**2) * v**4)
    
    term3 = (1 + numpy.log(v**3)) * (38645 / 756 - 65 / 9 * eta) * numpy.pi * v**5
    
    term4 = (11583231236531 / 4694215680 - 6848 / 21 * gamma_e - 640 / 3 * numpy.pi**2 +
             (-15737765635 / 3048192 + 2255 / 12 * numpy.pi**2) * eta +
             76055 / 1728 * eta**2 - 127825 / 1296 * eta**3 - 3424 / 21 * numpy.log(16 * v**2)) * v**6
    
    term5 = (77096675 / 254016 + 378515 / 1512 * eta - 74045 / 756 * eta**2) * numpy.pi * v**7
    
    term6 = -(2355 / 1462) * e0**2 * (v0 / v)**(19/3) * (
        1 + (299076223 / 81976608 + 18766963 / 2927736 * eta) * v**2 +
        (2833 / 1008 - 197 / 36 * eta) * v0**2 - 2819123 / 282600 * numpy.pi * v**3 +
        377 / 72 * numpy.pi * v0**3 +
        (16237683263 / 3330429696 + 24133060753 / 971375328 * eta + 1562608261 / 69383952 * eta**2) * v**4 +
        (847282939759 / 82632420864 - 718901219 / 368894736 * eta - 3697091711 / 105398496 * eta**2) * v**2 * v0**2 +
        (-1193251 / 3048192 - 66317 / 9072 * eta + 18155 / 1296 * eta**2) * v0**4 -
        (2831492681 / 118395270 + 11552066831 / 270617760 * eta) * numpy.pi * v**5 +
        (-7986575459 / 284860800 + 555367231 / 10173600 * eta) * numpy.pi * v**3 * v0**2 +
        (112751736071 / 5902315776 + 7075145051 / 210796992 * eta) * numpy.pi * v**2 * v0**3 +
        (764881 / 90720 - 949457 / 22680 * eta) * numpy.pi * v0**5 +
        (-43603153867072577087 / 132658535116800000 + 536803271 / 19782000 * gamma_e +
         15722503703 / 325555200 * numpy.pi**2 +
         (299172861614477 / 689135247360 - 15075413 / 1446912 * numpy.pi**2) * eta +
         3455209264991 / 41019955200 * eta**2 + 50612671711 / 878999040 * eta**3 +
         3843505163 / 59346000 * numpy.log(2) - 1121397129 / 17584000 * numpy.log(3) +
         536803271 / 39564000 * numpy.log(16 * v**2)) * v**6 + 
        (46001356684079 / 3357073133568 + 253471410141755 / 5874877983744 * eta -
        1693852244423 / 23313007872 * eta**2 - 307833827417 / 2497822272 * eta**3) * v**4 * v0**2 -
        (1062809371 / 20347200) * numpy.pi**2 * v**3 * v0**3 + 
        (-356873002170973/249880440692736 - 260399751935005/8924301453312 * eta + 
         150484695827/35413894656 * eta**2 + 340714213265/3794345856* eta**3) * v**2*v0**4 + 
        (26531900578691 / 168991764480 - 3317 / 126 * gamma_e + 122833 / 10368 * numpy.pi**2 +
        (9155185261 / 548674560 - 3977 / 1152 * numpy.pi**2) * eta -
        5732473 / 1306368 * eta**2 - 3090307 / 139968 * eta**3 +
        87419 / 1890 * numpy.log(2) - 26001 / 560 * numpy.log(3) -
        3317 / 252 * numpy.log(16 * v0**2)) * v0**6)
    
    spin_aligned_term = 4*beta*v**3

    psi_0 = -2*coallesence_phase + 2*numpy.pi*freqs*time_to_merger - numpy.pi/4
        
    psi = psi_0 + term1 * (term2 + term3 + term4 + term5 + term6 + spin_aligned_term)
    
    return(psi)

@jit(nopython=True)
def time_to_merger(m1,m2,inc,e0,f_0,beta):

    '''
    Calculates time to merger using TaylorT2 Eqn 6.6a from arXiv:1605.00304v2

    Args:
        m1 (float): Mass of the first object in solar masses.
        m2 (float): Mass of the second object in solar masses.
        inc (float): Inclination angle in radians.
        e0 (float): Initial eccentricity.
        f_0 (float): Initial GW frequency in Hz.
        beta (float): The 1.5 PN term.

    Returns:
        float: Time to merger in seconds.
    '''
    m1 = m1*MTsun 
    m2 = m2*MTsun 
    
    M = m1+m2 
    eta = (m1*m2)/(M**2)# # Reduced mass ratio
    
    v_init = (numpy.pi*M*f_0)**(1/3)
    
    tm = (5/256* M/eta * 1/(v_init)**8*calculate_T(v_init, v_init, e0,eta,beta))
    
    return(tm)

@jit(nopython=True)
def waveform_construct(m1,m2,inc,e0,D,freqs,s1,s2,f_low,f_high,coallesence_phase=0,logging=False):
    '''
    Constructs an instance of TaylorF2Ecc waveform and returns the amplitude, phase and time-frequency map. 

    Note: This function and the waveform construction is forced to be done in Numpy, not CuPy. Since the waveform 
    is only every directly expected to be evaluated on a sparse frequency grid, its faster to do this on a CPU with 
    numba acceleration I think. 

    Args:
        m1 (float): Mass of the first object (solar masses).
        m2 (float): Mass of the second object (solar masses)..
        inc (float): Inclination angle (rads).
        e0 (float): Initial eccentricity.
        D (float): Distance to the source (pc).
        freqs (array): Array of frequencies at which to evaluate the GW at.
        s1 (float): Spin of the first object.
        s2 (float): Spin of the second object.
        f_low (float): Lower frequency limit.
        f_high (float): Upper frequency limit (set by mission lifetimee).
        coallesence_phase (float, optional): coallesence_phase (default is 0)
        logging (bool, optional): Flag to enable logging. Defaults to False.

    Returns:
        tuple: A tuple containing the amplitude, phase, and time-frequency map.
    '''
    # Unit conversions 
    m1 = m1*MTsun 
    m2 = m2*MTsun
    D = (D)*pc/c

    M = m1+m2 
    eta = (m1*m2)/(M**2)# # Reduced mass ratio #m1m2/M**2

    # Compute the 1.5 PN term from s1 and s2
    beta = s1*(113/12*(m1**2)/(M**2)+25/4*eta) + s2*(113/12*(m2**2)/(M**2)+25/4*eta)

    # Only compute the waveform for frequencies > the initial GW frequency
    freq_mask = freqs>=f_low
    masked_freqs = numpy.asarray(freqs[freq_mask])

    # initial v 
    v0 = (numpy.pi*M*f_low)**(1/3) #(pi M f_0)**(1/3)
    # final v 
    v1 = (numpy.pi*M*f_high)**(1/3)
    # All vs
    v = (numpy.pi*M*masked_freqs)**(1/3)

    # Calculate tc (time to merger)
    time_to_merger = 5/256* M/eta * 1/(v0)**8*calculate_T(v0, v0, e0, eta, beta)

    if logging==True:
        # Time to merger from the time the source exits the frequency band specified, no eccentricity evolution assumed
        time_to_merger_from_f_high = 5/256* M/eta * 1/(v1)**8*calculate_T(v1, v1, e0, eta, beta)    
        print('Time to merger is: ',(time_to_merger)/(const.YRSID_SI),' years')
        print('Upper bound on time in band: ',(time_to_merger-time_to_merger_from_f_high)/(const.YRSID_SI),' years (no eccentricity evolution assumed)')

    # Calculate amplitude (not accounting for eccentricity here (straight line in log log space))
    Amp = M*numpy.sqrt(5*numpy.pi/96)*(M/D)*numpy.sqrt(eta)*(numpy.pi*(M)*masked_freqs)**(-7/6)

    # Waveform phase 
    Phi = F2EccPhase(masked_freqs,eta,v,e0,v0,coallesence_phase,time_to_merger,beta)

    # Calculate t-f map using 6.7a from arXiv:1605.00304v2
    time_to_merger_minus_time = 5/256* M/eta * 1/(v)**8*calculate_T(v, v0, e0, eta,beta)

    #t-f map
    times = time_to_merger - time_to_merger_minus_time

    return(Amp,Phi,times)

def balrog_response(params, freqs, f_high, T_obs, engine, TDIType, logging=False):
    '''
    Computes the response of a gravitational wave detector to a binary black hole waveform using the Balrog response.

    Args:
        params (list): List of parameters for the binary black hole waveform.
        freqs (numpy.ndarray): Array of frequencies.
        f_high (float): Upper frequency limit.
        T_obs (float): Observation time.
        engine (str): Engine used for the response calculation.
        TDIType (str): Type of Time Delay Interferometry (TDI) channel.
        logging (bool, optional): Whether to enable logging. Defaults to False.

    Returns:
        numpy.ndarray: Array representing the response of the detector to the waveform.

    Raises:
        ValueError: If TDIType is not one of 'Michelson', 'Sagnac', 'Single', 'LowFrequency', or 'Acceleration'.

    Parameter order: 
        m1, m2, D, beta, lam, inc, psis, final_orbital_phase, f_low, e0, s1, s2
        Note: f_low inputted into the code is the initial orbital frequency, it is converted to initial GW frequency 
            in the code. 
    '''
    
    # Set up to match Balrog 
    m1 = params[0]
    m2 = params[1]
    D = params[2]
    beta = params[3]
    lam = params[4]
    inc = params[5]
    psis = params[6] # Polarization
    final_orbital_phase = params[7]
    f_low = params[8]*2 # Factor of 2 as f_low in balrog defined as the initial ORBITAL frequency, converting to initial GW frequency
    e0 = params[9]
    s1 = params[10]
    s2 = params[11]    

    waveform_amp,waveform_phase,waveform_times = waveform_construct(m1,m2,inc,e0,D,freqs,s1,s2,
                                                        f_low,f_high,coallesence_phase=final_orbital_phase,logging=logging)
    
    ## Response
    # Ensures we are masking out the frequencies below which f(t=0)
    freq_mask = freqs>=f_low

    C = numpy.cos(inc)

    # Seperating into waveform polarization (Correct splitting, does not agree with TaylorF2 in Balrog)
    h_plus_positive = -waveform_amp*(1+C**2)
    h_cross_positive = waveform_amp*2*C*(1j)

    # Currently only works for one source
    # Only generate response for f>f_low
    N = freqs[freq_mask].size
    n_harmonics = numpy.zeros([1],dtype=numpy.int32, order='C')
    n_points = N* numpy.ones([1, 1], dtype=numpy.int32, order='C')
    frequency = numpy.zeros([1, 1, N], dtype=float, order='C')
    time = numpy.zeros([1, 1,N], dtype=float, order='C')

    hplus = numpy.zeros([1, 1, h_plus_positive.size], dtype=complex, order='C')
    hcross = numpy.zeros([1, 1, h_cross_positive.size], dtype=complex, order='C')
    phase = numpy.zeros([1, 1, waveform_phase.size], dtype=float, order='C')

    n_harmonics[0] = 1 
    frequency[0,0,:] = freqs[freq_mask].copy()
    time[0,0,:] = waveform_times.copy()
    hplus[0,0,:] = h_plus_positive.copy()
    hcross[0,0,:] = h_cross_positive.copy()

    # Negative waveform phase due to convention of FT used by arXiv:1605.00304v2
    phase[0,0,:] = -waveform_phase.copy()

    # Run response through selected TDI channel
    # Note we dont use any corrections, I think this is what BBHx does, can add them in if we need to. 
    if TDIType == "Michelson":

         Xf,Yf,Zf = alb_TDI.XYZ_General_FD(freqs, 
                           freqs.size, 
                           0., 
                           T_obs, 
                           numpy.array([numpy.sin(beta)],dtype=numpy.double), 
                           numpy.array([lam],dtype=numpy.double), 
                           numpy.array([psis],dtype=numpy.double), 
                           n_harmonics, 
                           n_points, 
                           frequency, 
                           time, 
                           hplus, 
                           hcross, 
                           phase, 
                           SUATime=None,#SUATime, 
                           TA1=None,#TA1,   /____So far we have not used any of the corrections, can add them in if needed____/
                           fdot=None,#fdot, 
                           TA2sq=None,#TA2sq, 
                           fddot=None,#fddot, 
                           engine_params=engine)
    elif TDIType == 'Sagnac':
    
         Xf,Yf,Zf = alb_TDI.alphabetagamma_General_FD(freqs, freqs.size, 0., T_obs, numpy.array([numpy.sin(beta)],dtype=numpy.double), numpy.array([lam],dtype=numpy.double), numpy.array([psis],dtype=numpy.double), n_harmonics, n_points, frequency, time, hplus, hcross, phase, SUATime=None, TA1=None, fdot=None, TA2sq=None, fddot=None, engine_params=engine)
                           
    elif TDIType == "Single":
    
         Xf, Yf, Zf =  alb_TDI.MXMYMZ_General_FD(freqs, freqs.size, 0., T_obs, numpy.array([numpy.sin(beta)],dtype=numpy.double), numpy.array([lam],dtype=numpy.double), numpy.array([psis],dtype=numpy.double), n_harmonics, n_points, frequency, time, hplus, hcross, phase, SUATime=None, TA1=None, fdot=None, TA2sq=None, fddot=None, engine_params=engine)
         
    elif TDIType == "LowFrequency":
        
         Xf, Yf, = alb_TDI.LowFreq_General_FD(freqs, freqs.size, 0., T_obs, numpy.array([numpy.sin(beta)],dtype=numpy.double), numpy.array([lam],dtype=numpy.double), numpy.array([psis],dtype=numpy.double), n_harmonics, n_points, frequency, time, hplus, hcross, phase, SUATime=None, TA1=None, fdot=None, TA2sq=None, fddot=None, engine_params=engine)
         Zf = numpy.zeros(Yf.shape, dtype = Yf.dtype)
         
    elif TDIType == "Acceleration":
    
         Xf, Yf, Zf = alb_TDI.MXMYMZ_General_FD(freqs, freqs.size, 0., T_obs, numpy.array([numpy.sin(beta)],dtype=numpy.double), numpy.array([lam],dtype=numpy.double), numpy.array([psis],dtype=numpy.double), n_harmonics, n_points, frequency, time, hplus, hcross, phase, SUATime=None, TA1=None, fdot=None, TA2sq=None, fddot=None, engine_params=engine)
         accfactor = (2*numpy.pi*freqs)**2
         Xf *= accfactor
         Yf *= accfactor
         Zf *= accfactor
         
    else:
         raise ValueError("TDIType must be 'Michelson', 'Sagnac', 'Single', 'LowFrequency', or 'Acceleration'")

    signal= numpy.array(XYZ2AET(Xf,Yf,Zf))

    return(signal)

def BBHx_response_direct(params,freqs,f_high,T_obs,TDIType,logging=False):
    '''
    Computes waveform and runs it through the BBHx response. The response model is the same as that in arxiv:1806.10734v1 (Marsat and Baker).
    Response is pretty close to Balrog, but not exactly the same. 

    Args:
        params (list): List of parameters for the binary black hole waveform.
        freqs (numpy.ndarray): Array of frequencies.
        f_high (float): Upper frequency limit.
        T_obs (float): Observation time.
        TDIType (str): Type of Time Delay Interferometry (TDI) channel. ('XYZ' or 'AET' for BBHx response)
        logging (bool, optional): Whether to enable logging. Defaults to False.

    Returns:
        numpy.ndarray: Array representing the response of the detector to the waveform.

    Raises:
        ValueError: If TDIType is not one of 'Michelson', 'Sagnac', 'Single', 'LowFrequency', or 'Acceleration'.

    Parameter order: 
        m1, m2, D, beta, lam, inc, psis, final_orbital_phase, f_low, e0, s1, s2
        Note: f_low inputted into the code is the initial orbital frequency, it is converted to initial GW frequency 
            in the code. 

    This function computes it directly on a frequency grid with no interpolation. Checked against Balrog response.
    Uses GPU to accelerate computation of the response (if available), but still direclty on the full FFT grid.  

    '''
    

    m1 = params[0]
    m2 = params[1]
    D = params[2]
    beta = params[3]
    lam = params[4]
    inc = params[5]
    psis = params[6] # Polarization
    final_orbital_phase = params[7]
    f_low = params[8]*2
    e0 = params[9]
    s1 = params[10]
    s2 = params[11]

    waveform_amp,waveform_phase,waveform_times = waveform_construct(m1,m2,inc,e0,D,freqs,s1,s2,
                                                                    f_low,f_high,coallesence_phase=final_orbital_phase,logging=logging)
    ## Response
    # Ensures we are masking out the frequencies below which f(t=0)
    freq_mask = freqs>=f_low

    masked_freqs = np.asarray(freqs[freq_mask])
    # phi ref is an additional phase rotation which is not needed since we have applied all the rotations inside the waveform phase
    phi_ref = 0
    beta = beta
    lamda = lam
    # Remember the psi in BBHx follows a different convention to that in Balrog
    psi = psis
    
    # Run response through selected TDI channel
    if TDIType == "AET":
       TDITag = 'AET'
    elif TDIType == 'XYZ':
       TDITag = 'XYZ'    
    else:
       raise ValueError("TDIType must be 'AET', 'XYZ' for BBHx reponse ")
    
    response = LISATDIResponse(TDItag=TDITag,use_gpu=use_GPU)

    C = np.cos(inc)

    # Converting from h_plus to h_lm. 
    BBHx_amp = -waveform_amp/numpy.sqrt(5/(64*np.pi))

    # Notice the lack of negative sign, I think this is internally handeled inside Balrog. 
    BBHx_phase = waveform_phase

    # Number of binaries currently hard set 1 
    num_bin_all = 1

    # Length of frequency array for computation 
    length = freqs[freq_mask].shape[0]

    # Only one harmonic for now 
    num_modes = 1
    
    # params are amp, phase, tf, transferL1re, transferL1im, transferL2re, transferL2im, transferL3re, transferL3im
    num_interp_params = 9 

    # Dump everything in here, this is what is being used inside all the response functions 
    out_buffer = np.zeros(num_interp_params*length*num_modes*num_bin_all)
    out_buffer = out_buffer.reshape(num_interp_params, num_bin_all, num_modes, length)

    # Fill in out buffer with amplitude, phase, t-f in SSB frame
    out_buffer[0,0,0,:]= np.asarray(BBHx_amp)
    out_buffer[1,0,0,:]= np.asarray(BBHx_phase)
    out_buffer[2,0,0,:]= np.asarray(waveform_times)
    
    out_buffer = out_buffer.flatten().copy()
    # Generate response 

    response(masked_freqs,
                 float(inc),
                 float(lam),
                 float(beta),
                 float(psi),
                 float(phi_ref),
                 length,
                 out_buffer=out_buffer,
                 modes = [(2,2)])


    # Computing the waveform directly with no interpolation on the CPU. 
    if use_GPU == True:
        waveform_gen = direct_sum_wrap_gpu
    else:
        waveform_gen = direct_sum_wrap_cpu

    # setup template (this will contain the waveform in AET/XYZ Channels)
    templateChannels = np.zeros(
        (num_bin_all * 3 * length), dtype=complex)
    
    # direct computation of 3 channel waveform
    waveform_gen(
        templateChannels,
        out_buffer,
        num_bin_all,
        length,
        3,
        1,
        np.asarray(0),
        np.asarray(T_obs),
    )
    # Reshape into 3D array (extra dimension is number of binaries)
    out = templateChannels.reshape(num_bin_all, 3, length)

    # Push into XYZ array (Probably can be optimized alot more)
    XYZ = np.zeros((3,freqs.size),dtype=complex)

    # Squeeze just collapses the dimensions with size one, in our case the dimension is num_bin as we only have one binary
    # Multiplies by frequency mode factor to get waveform into same convention as Balrog. 
    XYZ[:,freq_mask] = out.squeeze()*bbhx_pre_factor*1/masked_freqs
    
    return(XYZ)


def BBHx_response_interpolate_CPU(params,freqs_sparse,freqs_dense,f_high,T_obs,TDIType,logging=False):
    '''
    Computes waveform and runs it through the BBHx response. Same as the function BBHx_response_direct, 
        but uses interpolation to speed up the response calculation.


    Args:
        params (list): List of parameters for the binary black hole waveform.
        freqs_sparse (array): Sparse array of frequencies at which to evaluate the amplitude,phase,t-f map and response at. 
        freqs_dense (array): Array of frequencies on the full FFT grid to interpolate to.
        f_high (float): Upper frequency limit.
        T_obs (float): Observation time.
        TDIType (str): Type of Time Delay Interferometry (TDI) channel. ('XYZ' or 'AET' for BBHx response)
        logging (bool, optional): Whether to enable logging. Defaults to False.

    Returns:
        numpy.ndarray: Array representing the response of the detector to the waveform.

    Raises:
        ValueError: If TDIType is not one of 'Michelson', 'Sagnac', 'Single', 'LowFrequency', or 'Acceleration'.

    Parameter order: 
        m1, m2, D, beta, lam, inc, psis, final_orbital_phase, f_low, e0, s1, s2
        Note: f_low inputted into the code is the initial orbital frequency, it is converted to initial GW frequency 
            in the code. 
    Checked against Balrog response. Interpolates on CPU, onto the full FFT grid. 
    '''
    

    m1 = params[0]
    m2 = params[1]
    D = params[2]
    beta = params[3]
    lam = params[4]
    inc = params[5]
    psis = params[6] # Polarization
    final_orbital_phase = params[7]
    f_low = params[8]*2
    e0 = params[9]
    s1 = params[10]
    s2 = params[11]

    waveform_amp,waveform_phase,waveform_times = waveform_construct(m1,m2,inc,e0,D,freqs_sparse,s1,s2,
                                                                    f_low,f_high,coallesence_phase=final_orbital_phase,logging=logging)

    ## Response
    # Ensures we are masking out the frequencies below which f(t=0)
    freq_mask = freqs_sparse>=f_low

    # phi ref is an additional phase rotation which is not needed since we have applied all the rotations inside the waveform phase
    phi_ref = 0.
    beta = beta
    lamda = lam
    # Remember the psi in BBHx follows a different convention to that in Balrog
    psi = psis
    
    # Run response through selected TDI channel
    if TDIType == "AET":
       TDITag = 'AET'
    elif TDIType == 'XYZ':
       TDITag = 'XYZ'    
    else:
       raise ValueError("TDIType must be 'AET', 'XYZ' for BBHx reponse ")
    
    response = LISATDIResponse(TDItag=TDITag,use_gpu=False)

    C = numpy.cos(inc)

    # Converting from h_plus to h_lm. 
    BBHx_amp = np.asarray(-waveform_amp)/numpy.sqrt(5/(64*np.pi))

    # Notice the lack of negative sign, I think this is internally handeled inside Balrog. 
    BBHx_phase = waveform_phase


    # Number of binaries currently hard set 1 
    num_bin_all = 1

    # Length of frequency array for computation 
    length = freqs_sparse[freq_mask].shape[0]

    # Only one harmonic for now 
    num_modes = 1
    
    # params are amp, phase, tf, transferL1re, transferL1im, transferL2re, transferL2im, transferL3re, transferL3im
    num_interp_params = 9 

    # Dump everything in here, this is what is being used inside all the response functions 
    out_buffer = numpy.zeros(num_interp_params*length*num_modes*num_bin_all)
    out_buffer = out_buffer.reshape(num_interp_params, num_bin_all, num_modes, length)

    # Fill in out buffer with amplitude, phase, t-f in SSB frame
    out_buffer[0,0,0,:]= BBHx_amp
    out_buffer[1,0,0,:]= np.asarray(BBHx_phase)
    out_buffer[2,0,0,:]= np.asarray(waveform_times)
    
    out_buffer = out_buffer.flatten().copy()

    dense_frequency_mask = freqs_dense>=f_low
    dense_freqs_length = freqs_dense[dense_frequency_mask].size
    
    # Generate response 
    response(freqs_sparse[freq_mask],
                 inc,
                 lam,
                 beta,
                 psi,
                 phi_ref,
                 length,
                 out_buffer=out_buffer,
                 modes = [(2,2)])

    # setup interpolant
    spline = CubicSplineInterpolant(
        freqs_sparse[freq_mask],
        out_buffer,
        length=length,
        num_interp_params=num_interp_params,
        num_modes=num_modes,
        num_bin_all=num_bin_all,
        use_gpu=False,
    )

    interp_response = TemplateInterpFD(use_gpu=False)    
    template_channels = interp_response(freqs_dense[dense_frequency_mask],spline.container,numpy.array([0]),numpy.array([T_obs]),freqs_sparse[freq_mask].size,num_modes,3)

    # combine into one data stream
    data_out = numpy.zeros((3, dense_freqs_length), dtype=complex)
    for temp, start_i, length_i in zip(
        template_channels,
        interp_response.start_inds,
        interp_response.lengths,
    ):
        data_out[:, start_i : start_i + length_i] = temp

    
    XYZ = numpy.zeros((3,freqs_dense.size),dtype=complex)
    XYZ[:,dense_frequency_mask] = data_out.squeeze()*1/(2j*numpy.pi*Armlength)*1/(freqs_dense[dense_frequency_mask])

    return(XYZ)

def BBHx_response_interpolate(params,freqs_sparse,freqs_dense,freqs_sparse_on_CPU,f_high,T_obs,TDIType,logging=False):
    '''
    Computes waveform and runs it through the BBHx response. Same as the function BBHx_response_interpolate_CPU, but if GPU is available  
        it will use that.

    Args:
        params (list): List of parameters for the binary black hole waveform.
        freqs_sparse (array): Sparse array of frequencies at which to evaluate the amplitude,phase,t-f map and response at. 
        freqs_dense (array): Array of frequencies on the full FFT grid to interpolate to.
        f_high (float): Upper frequency limit.
        T_obs (float): Observation time.
        TDIType (str): Type of Time Delay Interferometry (TDI) channel. ('XYZ' or 'AET' for BBHx response)
        logging (bool, optional): Whether to enable logging. Defaults to False.

    Returns:
        numpy.ndarray: Array representing the response of the detector to the waveform.

    Raises:
        ValueError: If TDIType is not one of 'Michelson', 'Sagnac', 'Single', 'LowFrequency', or 'Acceleration'.

    Parameter order: 
        m1, m2, D, beta, lam, inc, psis, final_orbital_phase, f_low, e0, s1, s2
        Note: f_low inputted into the code is the initial orbital frequency, it is converted to initial GW frequency 
            in the code. 
    Checked against Balrog response. Waveform computed on a CPU on sparse grid, response, interpolated all done on GPU (when available). 
    '''
    
    
    m1 = params[0]
    m2 = params[1]
    D = params[2]
    beta = params[3]
    lam = params[4]
    inc = params[5]
    psis = params[6] # Polarization
    final_orbital_phase = params[7]
    f_low = params[8]*2
    e0 = params[9]
    s1 = params[10]
    s2 = params[11]

    waveform_amp,waveform_phase,waveform_times = waveform_construct(m1,m2,inc,e0,D,freqs_sparse_on_CPU,s1,s2,
                                                                    f_low,f_high,coallesence_phase=final_orbital_phase,logging=logging)
    ## Response
    # Ensures we are masking out the frequencies below which f(t=0)
    freq_mask_sparse = freqs_sparse>=f_low
    freq_mask_dense = freqs_dense>=f_low

    freqs_sparse_masked = freqs_sparse[freq_mask_sparse]
    freqs_dense_masked = freqs_dense[freq_mask_dense]

    # phi ref is an additional phase rotation which is not needed since we have applied all the rotations inside the waveform phase
    phi_ref = 0.
    beta = beta
    lamda = lam
    # Remember the psi in BBHx follows a different convention to that in Balrog
    psi = psis
    
    # Run response through selected TDI channel
    if TDIType == "AET":
       TDITag = 'AET'
    elif TDIType == 'XYZ':
       TDITag = 'XYZ'    
    else:
       raise ValueError("TDIType must be 'AET', 'XYZ' for BBHx reponse ")
    
    response = LISATDIResponse(TDItag=TDITag,use_gpu=use_GPU)

    C = np.cos(inc)

    # Converting from h_plus to h_lm. 
    BBHx_amp = np.asarray(-waveform_amp)/np.sqrt(5/(64*np.pi))

    # Notice the lack of negative sign, I think this is internally handeled inside Balrog. 
    BBHx_phase = waveform_phase


    # Number of binaries currently hard set 1 
    num_bin_all = 1

    # Length of sparse frequency array for computation 
    length = freqs_sparse_masked.shape[0]

    # Only one harmonic for now 
    num_modes = 1
    
    # params are amp, phase, tf, transferL1re, transferL1im, transferL2re, transferL2im, transferL3re, transferL3im
    num_interp_params = 9 

    # Dump everything in here, this is what is being used inside all the response functions 
    out_buffer = np.zeros(num_interp_params*length*num_modes*num_bin_all)
    out_buffer = out_buffer.reshape(num_interp_params, num_bin_all, num_modes, length)

    # Fill in out buffer with amplitude, phase, t-f in SSB frame
    out_buffer[0,0,0,:]= BBHx_amp
    out_buffer[1,0,0,:]= np.asarray(BBHx_phase)
    out_buffer[2,0,0,:]= np.asarray(waveform_times)
    
    
    out_buffer = out_buffer.flatten().copy()


    dense_freqs_length = freqs_dense_masked.shape[0]

    # Generate response 
    response(freqs_sparse_masked,
                 inc,
                 lam,
                 beta,
                 psi,
                 phi_ref,
                 length,
                 out_buffer=out_buffer,
                 modes = [(2,2)])

    # setup interpolant
    spline = CubicSplineInterpolant(
        freqs_sparse_masked,
        out_buffer,
        length=length,
        num_interp_params=num_interp_params,
        num_modes=num_modes,
        num_bin_all=num_bin_all,
        use_gpu=use_GPU,
    )
    
    interp_response = TemplateInterpFD(use_gpu=use_GPU)    

    template_channels = interp_response(freqs_dense_masked,spline.container,numpy.array([0]),numpy.array([T_obs]),length,num_modes,3)


    # combine into one data stream
    data_out = np.zeros((3, dense_freqs_length), dtype=complex)
    for temp, start_i, length_i in zip(
        template_channels,
        interp_response.start_inds,
        interp_response.lengths,
    ):
        data_out[:, start_i : start_i + length_i] = temp

    XYZ = np.zeros((3,freqs_dense.size),dtype=complex)
    XYZ[:,freq_mask_dense] = data_out.squeeze()*bbhx_pre_factor*1/freqs_dense_masked

    return(XYZ)
