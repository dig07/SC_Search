'''
Utility file contains useful utility functions for the code.
'''
from .Semi_Coherent_Functions import noise_weighted_inner_product
from .Waveforms import TaylorF2Ecc, TaylorF2EccSpin
import numpy as np 

# Corner functions
from scipy import stats
import scipy
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

try: 
    import seaborn as sns
except:
    pass

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

def chirp_mass_q_from_component_mass(m1, m2):
    """
    Calculate the chirp mass and mass ratio from the component masses.

    Assumes m1>m2

    Parameters:
        m1 (float): Mass of the first component.
        m2 (float): Mass of the second component.

    Returns:
        tuple: A tuple containing the chirp mass and symmetric mass ratio.
        - mc (float): Chirp mass.
        - q (float): Symmetric mass ratio.
    """
    mc = ((m1 * m2) ** (3 / 5)) / (m1 + m2) ** (1 / 5)
    q = m2/m1
    return mc, q

def component_masses_from_chirp_q(mchirp, q):
    """
    Calculate the component masses of a binary system from the chirp mass and mass ratio.

    Assumes m1>m2

    
    Parameters:
        mchirp (float): The chirp mass of the binary system.
        q (float): The mass ratio of the binary system.

    Returns:
        tuple: A tuple containing the component masses (m1, m2) of the binary system.
    """
    eta = q/(1+q)**2
    mtotal = mchirp / eta**(3/5)
    m1 = mtotal/(1+q)
    m2 = mtotal*q/(1+q)
    return m1, m2


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

def TaylorF2Ecc_mc_q_to_m1m2(parameters):
    '''
    Parameter transforms are hardcoded in to:
    - Polarization shift to match Balrog convention
    - Mc,q->m1,m2

    Args:
        parameters (array): Waveform parameters. 
            parameters[0]: Mc
            parameters[1]: q (m2/m1)
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
    parameters[0],parameters[1] = component_masses_from_chirp_q(parameters[0],parameters[1])

    return(parameters)

def TaylorF2EccSpin_s1_s2_to_spin_params(m1,m2,s1,s2):
    '''
    Used to convert the aligned spin parameters to the quantities used for the TaylorF2EccSpin model

    This function is mainly used in postprocessing steps.

    Dont need to convert m1 and m2 to SI units as we only need ratios of masses, so the conversion factor drops out. 

    Args:
        m1 (float): mass of the first component [solar masses]
        m2 (float): mass of the second component [solar masses]
        s1 (float): spin of the first component [dimensionless]
        s2 (float): spin of the second component [dimensionless]
    
    Returns:
        beta_15 (float): 1.5 PN spin-orbit term
        beta_25 (float): 2.5 PN spin-orbit term
        sigma (float): 2 PN spin-spin term
    '''
    M = m1+m2
    eta = (m1*m2)/(M**2)

    # Compute the 1.5 PN term from s1 and s2 (Spin-orbit)
    beta_15 = s1*(113/12*(m1**2)/(M**2)+25/4*eta) + s2*(113/12*(m2**2)/(M**2)+25/4*eta)
    
    # Compute the 2.5 PN term from s1 and s2 (Spin-orbit)
    beta_25 = s1*((m1**2)/(M**2)*(-31319/1008+1159/24*eta)+eta*(-809/84+281/8*eta))+s2*((m2**2)/(M**2)*(-31319/1008+1159/24*eta)+eta*(-809/84+281/8*eta))

    # Compute the 2 PN term from s1 and s2 (Spin-spin) (sigma)

    # Standard spin-spin term
    simga_s1s2 = 474/48*eta*s1*s2

    # Quadrupole - monopole term
    sigma_qm = 5*(s1**2*(m1**2)/(M**2)+s2**2*(m2**2)/(M**2))

    # Self-spin interaction term 
    sigma_self_spin = 1/16*(s1**2*(m1**2)/(M**2)+s2**2*(m2**2)/(M**2)) 

    # Add them all together to get the 2PN term
    sigma = simga_s1s2 + sigma_qm + sigma_self_spin

    return(beta_15,beta_25,sigma)

def chi_effective_from_spins(m1,m2,s1,s2):
    '''
    Compute the effective spin parameter from the component masses and spins
    
    NOTE: Assumes spins are aligned spins to orbital angular momentum vector. 

    Args:
        m1 (float): mass of the first component [solar masses]
        m2 (float): mass of the second component [solar masses]
        s1 (float): spin of the first component [dimensionless]
        s2 (float): spin of the second component [dimensionless]

    Returns:
        chi_eff (float): effective spin parameter
    '''

    chi_eff = (s1*m1 + s2*m2)/(m1+m2)
    return(chi_eff)

def reconstruct_higherst_snr_from_waveform_posterior(posterior,logls,psd,waveform_object,df,waveform_args):
    '''
    Reconstruct the waveform with the highest SNR from the posterior samples. Use waveform with highest logl to reconstruct waveform.

    Args:
        posterior (array): posterior samples
        logls (array): log likelihoods of the posterior samples
        psd (array): power spectral density of the detector
        waveform_object (object): waveform object
        T_obs (float): observation time
        waveform_args (dictionary): dictionary containing the waveform arguments
    
    Returns:
        SNR (float): signal-to-noise ratio of the waveform
        wf (float): best matching waveform
    '''
    # Find max logl 
    max_logl_index = np.argmax(logls)
    #Find posterior samples that correspond to max logl
    max_logl_posterior = posterior[max_logl_index,:].copy()

    # Artificially add a orbital phase of 0 to the waveform, does not affect SNR but need it to compute the waveform 
    max_logl_posterior = np.insert(max_logl_posterior,7,0)

    # Reconstruct waveform at max logl
    source_params_transformed = TaylorF2Ecc_mc_eta_to_m1m2(max_logl_posterior.copy())
    wf= waveform_object(source_params_transformed,**waveform_args) 

    # Compute the SNR of the waveform
    SNR = np.sqrt(noise_weighted_inner_product(wf, wf, df, psd, phase_maximize=True))

    return(SNR,wf)

    

def generate_tc_prior_samples(priors,nsamples=1000000): 
    '''
    Generate samples from the prior for the time of coalescence

    Args:
        priors (array): array containing the priors
            Structure: priors[0] = [Chirp mass prior]
                       priors[1] = [Symmetric mass ratio prior]
                       priors[2] = [Eccentricity prior]
                       priors[3] = [Initial GW frequency prior]
        nsamples (int): number of samples to generate

    Returns:
        tc_samples (array): samples from the prior
    '''

    # Generate samples from the prior    
    initial_sampler = stats.qmc.LatinHypercube(4,strength=1)
    samples = initial_sampler.random(n=nsamples)*(np.ptp(priors,axis=1)) + np.array(priors)[:,0]

    # Convert samples from (mc,eta) -> (m1,m2)
    m1,m2= component_masses_from_chirp_eta(samples[:,0],samples[:,1])

    # Calculate time to merger for each prior sample
    tcs = TaylorF2Ecc.time_to_merger(m1,m2,
                                                3,#irrelevant (inc, need to fix this bug)
                                                samples[:,-1],#e0 
                                                samples[:,-2])# flow
    print('Maximal prior width: ',np.ptp(tcs)/(365.25*24*60*60),' years')
    
    return(tcs)
    
def compute_monte_carlo_estimate_of_sky_area(posterior_samples, KDE_downsampling=100, KDE_eval_points=10000, q=0.9):
    '''
    Compute the monte carlo estimate of the sky area for a given posterior at a given quantile level 

    Args:
        posterior_samples (array): 
            posterior samples over sky 
            array shape must be (num_samples, 2)
            with the 0 coloumn being lambda [rad] and the 1 column being sin(beta) [dim. less]
        KDE_downsampling (int): 
            downsampling factor for the KDE fitting
            for speed, instead of making giant KDE with all posterior samples, we downsample
        KDE_eval_points (int): 
            number of points to evaluate the KDE at
        q (float): quantile level
            in range 0<q<1, e.g. 0.9 means area of 90% credible region
    
    Returns:
        sky_area (float): 
            monte carlo estimate of the sky area [stradians]
    '''

    # KDE on sky posterior 
    sky_KDE = stats.gaussian_kde(posterior_samples[::KDE_downsampling].T)

    # Draw random points from the biggest box that encompasses the sky posterior

    lower_left_box_corner = np.min(posterior_samples, axis=0).reshape((2,1))

    box_dimensions = np.ptp(posterior_samples, axis=0).reshape((2,1))

    random_draws = np.random.uniform(size=(2,KDE_eval_points)) * box_dimensions + lower_left_box_corner

    # Area of this box
    total_area_of_box = box_dimensions[0,0] * box_dimensions[1,0]

    # Evaluate the KDE at these points 
    p = sky_KDE.pdf(random_draws)

    # Evaluate the contour that each point sits on, using a monte-carlo estimate 
    contour = np.zeros(KDE_eval_points)
    for i, prob in enumerate(p): 
        contour[i] = np.sum(p>prob) / KDE_eval_points
    
    # What fraction of the total area is within the quantile
    sky_area = total_area_of_box * np.sum(contour<q) / KDE_eval_points  

    return(sky_area)

def setup_tf_grid(T,
                  dt, 
                  dT):
    '''
    Sets up time-frequency grid for time-frequency domain calculations. 

    Args:
        T (float): Total observation time (s) 
        dt (float): Sampling time (s)
        dT (float): Time segment duration (s)
    Returns:
        t_grid (numpy.array): Time grid (s)
        f_grid (numpy.array): Frequency grid (Hz)
    '''

    print("Setting up time-frequency grid...")
    print(f"Total observation time: {T} s")

    # int -> Rounding down 
    nT = int(T/dT) # length of each time chunk
    
    print(f"Number of time segments: {nT}")

    t_grid = np.arange(nT+1)*dT # nT+1 as we want nT time segments, which means nT+1 time points
    # Frequency resolution 
    dF =f_min= (1/dT)
    f_max = 1/(dt)/ 2 # Nyquist frequency 

    nF = int((f_max - f_min) / dF) + 1 # frequency bins per segment
    print(f"Number of frequency bins per segment: {nF}")

    # Frequency grid (neglecting DC component)
    f_grid = np.arange(1,nF+1) * dF  # segment frequencies
    # Time grid
    print('NOTE: The time grid calculates the number of segments as int(T/dT), which means it rounds down the number of segments.')
    return(t_grid,f_grid)
        
def SFT_data(data,
             times,
             t_grid,
             nT,
             nF,
             window_alpha = 0.01):
    '''
    Generate SFT data from time-domain data. 

    Args:
        data (numpy.array): Time-domain data (3,#time samples)
        times (numpy.array): Time stamps corresponding to the time-domain data (shape: #time samples)
        t_grid (numpy.array): SFT time grid 
        nT (int): Number of time segments
        nF (int): Number of frequency bins per segment
        window_alpha (float): Alpha parameter for the Tukey window
    Returns:
        SFT_data (numpy.array): SFT data
    '''
    print("Generating SFT data...")

    SFT_data = np.zeros((3,nT,nF),dtype=complex)

    for i in range(nT):
        start_time = t_grid[i]
        end_time = t_grid[i+1]

        times_mask = (times>=start_time) & (times<end_time)

        channel_1_segment_data = data[0,:][times_mask] 
        channel_2_segment_data = data[1,:][times_mask]
        channel_3_segment_data = data[2,:][times_mask]
        # Window function 
        win = scipy.signal.windows.tukey(np.sum((times >= start_time) & (times < end_time)), alpha=window_alpha)


        SFT_data[0,i,:] = np.fft.rfft(channel_1_segment_data*win)[1:] 
        SFT_data[1,i,:] = np.fft.rfft(channel_2_segment_data*win)[1:] 
        SFT_data[2,i,:] = np.fft.rfft(channel_3_segment_data*win)[1:] 

    return(SFT_data)

def corner_mine(posteriors,
                quantiles=[],
                num_kde=50,
                num_1d_hist=[100],
                colors = ['k'],
                legend = None,
                figsize=(10,10),
                tick_fontsize=17,
                labels = [],
                label_fontsize=17,
                pad = 0.2,
                renormalize = True,
                truths = [],
                truth_color='m',
                inset = [],
                axes_adjust = False,
                special_1d_param_index = None,
                special_1d_hist_param_bins = [],
                line_width = 1.,
                legend_fontsize=20):
    ''' 
    Custom corner plot implementation 
    
    Args:
        posteriors: list of posteriors, [numpy array (npoints, ndimensions)]
            posterior to plot
            
        quantiles: list (defaults to empty list)
            if empty list, them plots a scatter plot, else plots this many quantiles on the hist2d plots
            
        num_kde: int (defaults to 50)
            number of gridpoints in the 2d hist, passed to seaborn
            
        num_1d_hist: list (defaults to 100)
            number of bins for 1d histograms, for each posterior
            
        colors: list (defaults to ['k'])
            list of colours for each posterior 
            
        legend: list (defaults to None)
            list of legend titles for each of the posteriors being plotted
            
        figsize: tuple (defaults to (10,10)
            size of figure to construct
            
        tick_fontsize: int (defaults to 17)
            fontsize for ticks
            
        labels: list (defaults to [])
            axis label for each dimension 
        
        label_fontsize: int (defaults to 20)
            size of axis labels
        
        pad: padding (defaults to -0.2)
            shift of the label from the axis boundary 
        
        renormalize: boolean (defaults to True)
            renormalize 1d histograms so the peaks are at the same height 
        
        truths: list (defaults to [])
            truth point
        
        truth_color: color (defaults to 'm')
            colour for truth lines
        
        inset: list (defaults to [])
            indices of plots on gridspec that should be made into an axis for an inset
        
        axes_adjust: Boolean (defaults to False)
            wether to return the axes assuming further editing or to run plt.show() inside function 
        
        special_1d_param_index: int (Defaults to None)
            1d histogram in which I want more control on the number of bins 
        
        special_1d_hist_param_bins: list (Defaults to [])
            list of bin nums for 1d histogram on special param
        
        line_width: float (Defaults to 1.)
            Thickness of the lines in the plot

        legend_fontsize: int (Defaults to 20)
            Fontsize of the legend
    
    '''
    # Set up the figure and gridspec
    ndim = posteriors[0].shape[1]
    
    num_posteriors = len(posteriors)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ndim, ndim, figure=fig)
    
    mins = np.zeros((num_posteriors,ndim))
    maxs = np.zeros((num_posteriors,ndim))
    
    # Generate diagonal subplots
    ax_diag = [fig.add_subplot(gs[i, i]) for i in range(ndim)]
    
    # Generate off diagonal plots
    ax_offdiag = [[fig.add_subplot(gs[i, j]) for j in range(i)] for i in range(1, ndim)]
    
    # Make inset axes if requested
    if inset != []:
        a,b = inset
        ax_inset = fig.add_subplot(gs[a,b])   

    # Plot the diagonal histograms
    
    for posterior_index,data in enumerate(posteriors):

        for i in range(ndim):
            
            if i == special_1d_param_index: 
                
                counts, _ = np.histogram(data[:,i],bins=special_1d_hist_param_bins[posterior_index])
                weights = np.repeat(1/np.max(counts),data[:,i].size)
                if renormalize == True:
                    ax_diag[i].hist(data[:, i], bins=special_1d_hist_param_bins[posterior_index], color=colors[posterior_index], histtype='step',weights=weights,linewidth=line_width)
                else:
                    ax_diag[i].hist(data[:, i], bins=special_1d_hist_param_bins[posterior_index], color=colors[posterior_index], histtype='step',linewidth=line_width,density=True)

            else:
                
                counts, _ = np.histogram(data[:,i],bins=num_1d_hist[posterior_index])
                weights = np.repeat(1/np.max(counts),data[:,i].size)
                if renormalize == True:
                    ax_diag[i].hist(data[:, i], bins=num_1d_hist[posterior_index], color=colors[posterior_index], histtype='step',weights=weights,linewidth=line_width)
                else:
                    ax_diag[i].hist(data[:, i], bins=num_1d_hist[posterior_index], color=colors[posterior_index], histtype='step',linewidth=line_width,density=True)

                
            ax_diag[i].set_yticks([])
            # Last diagonal
            if i != ndim-1:
                ax_diag[i].set_xticks([])
            
            mins[posterior_index,i] = np.min(data[:,i])
            maxs[posterior_index,i] = np.max(data[:,i])


        # Plot the off-diagonal plots
        for i in range(ndim-1):
            
            for j in range(0,i+1):

                if len(quantiles)!=0:

                    sns.kdeplot(x=data[:, j],y=data[:, i+1],gridsize=num_kde,ax=ax_offdiag[i][j],levels=quantiles,
                               color=colors[posterior_index],linewidths=line_width)

                # If not quantiles, draw a scatter 
                if len(quantiles)==0:
                    ax_offdiag[i][j].scatter(data[:, j], data[:, i+1], s=2, color=colors[posterior_index], alpha=0.5)
                    ax_offdiag[i][j].set_xlim(np.min(data[:, j]), np.max(data[:, j]))
                    ax_offdiag[i][j].set_ylim(np.min(data[:, i+1]), np.max(data[:, i+1]))

                # Bottom 
                if i!=ndim-2:

                    ax_offdiag[i][j].set_xticks([])

                if j!=0:

                    ax_offdiag[i][j].set_yticks([])

    # Adjust the spacing
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    
    if legend != None:
        legend_elements = []

        for posterior_index,posterior in enumerate(posteriors):

                legend_elements.append( Line2D([0], [0], color=colors[posterior_index], 
                                               label=legend[posterior_index]))
        plt.legend(handles=legend_elements,bbox_to_anchor=(0.9, 0.9),
          bbox_transform=fig.transFigure,prop={'size': legend_fontsize})

    global_axis_array = np.zeros((ndim,ndim), dtype=object)    
    
    # Put diagonals into global axis array 
    for axis_index,ax in enumerate(ax_diag):
        ax.tick_params(labelsize=tick_fontsize,rotation=45)
        global_axis_array[axis_index,axis_index] = ax
    
    for ax in sum(ax_offdiag,[]):
        ax.tick_params(labelsize=tick_fontsize,rotation=45)

    
    # Put off diagonals in global axis array 
    for i in range(ndim-1):

        for j in range(0,i+1):    
            
            global_axis_array[i+1,j] = ax_offdiag[i][j]
    
    # Now we can do all operations on this global axis array instead!
    
    # # For all columns : set x lim, set x labels
    for j in range(ndim):
        minimum_,maximum_ = np.min(mins[:,j],axis=0),np.max(maxs[:,j],axis=0)
        
        for row_index,ax in enumerate(global_axis_array[:,j]):
            # Checking if there actually is an axis there, ie that we are in the lower triangle
            if type(ax)!= int:
            
                ax.set_xlim(minimum_,maximum_)

                # If last row, set label
                if labels!= None and row_index==(ndim-1):
                    ax.set_xlabel(labels[j],fontsize=label_fontsize)
                    
                    # Shift the labels out a bit 
                    ax.get_xaxis().set_label_coords(0.5,-pad)
                
                # If there are truths plot them
                if len(truths)!= 0:
                    
                    ax.axvline(truths[j],color=truth_color,linewidth=line_width)
                
                    
    # # For all rows : set y lim, set y labels
    for j in range(ndim):
        # j =0 is the (0,0) component which is a 1d histogram, y lim for this does not make snese 
        if j!=0:
            minimum_,maximum_ = np.min(mins[:,j],axis=0),np.max(maxs[:,j],axis=0)

            for column_index,ax in enumerate(global_axis_array[j,:]):
                # Checking if there actually is an axis there, ie that we are in the lower triangle
                # Check to make sure we dont mess with the histograms ie the diagonal
                if type(ax)!= int and column_index!=j:

                    ax.set_ylim(minimum_,maximum_)

                    # If first row, set label
                    if labels!= None and column_index==(0):
                        ax.set_ylabel(labels[j],fontsize=label_fontsize)
                        
                        # Shift the labels out a bit 
                        ax.get_yaxis().set_label_coords(-pad,0.5)
                    
                    if len(truths)!= 0:
                    
                        ax.axhline(truths[j],color=truth_color,linewidth=line_width)
                        
    # If we want an inset axis, add it to the global axis array at the VERY end
    if inset!=[]:
        
        global_axis_array[a,b] = ax_inset
        
        # Switch tick side from axis
        global_axis_array[a,b].xaxis.tick_top()
        global_axis_array[a,b].yaxis.tick_right()

    if inset== [] and axes_adjust==False:
        plt.show()
    
        
    return(fig,global_axis_array)
