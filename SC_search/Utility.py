'''
Utility file contains useful utility functions for the code.
'''
from .Semi_Coherent_Functions import noise_weighted_inner_product
import numpy as np 

# Corner functions
from scipy import stats
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
                line_width = 1.):
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
                    ax_diag[i].hist(data[:, i], bins=special_1d_hist_param_bins[posterior_index], color=colors[posterior_index], histtype='step',linewidth=line_width)

            else:
                
                counts, _ = np.histogram(data[:,i],bins=num_1d_hist[posterior_index])
                weights = np.repeat(1/np.max(counts),data[:,i].size)
                if renormalize == True:
                    ax_diag[i].hist(data[:, i], bins=num_1d_hist[posterior_index], color=colors[posterior_index], histtype='step',weights=weights,linewidth=line_width)
                else:
                    ax_diag[i].hist(data[:, i], bins=num_1d_hist[posterior_index], color=colors[posterior_index], histtype='step',linewidth=line_width)

                
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
          bbox_transform=fig.transFigure,prop={'size': 20})

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