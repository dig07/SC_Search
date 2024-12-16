try: 
    import cupy as cp 
except ImportError:
    print('Cupy not installed, search (on full FFT grid) wont work')

try: 
    from ldc.lisa.noise import get_noise_model
except ImportError:
    print('LDC not installed')

import matplotlib.pyplot as plt
import os
import scipy.stats as stats


from .Swarm_class import Semi_Coherent_Model
from .Utility import TaylorF2Ecc_mc_eta_to_m1m2
from .Semi_Coherent_Functions import upsilon_func, semi_coherent_match, coherent_match
from .Noise import *
from .Waveforms import TaylorF2Ecc, TaylorF2EccSpin
from .Waveforms import Constants as const
import numpy as np 

class Q_look:
    '''
    Class designed to evaluate a bunch of waveforms in every tile of the search grid, to see if theres something statistically signfificant in there.     
    '''

    def __init__(self,
                frequency_series_dict,
                search_prior_mc_f_low,
                search_priors_in_other_parameters,
                num_points_per_tile,
                data_file_name,
                segment=100,
                include_spin=False,
                LDC_PSD = False,
                LDC_PSD_TDI_version =1,
                tiling_scheme='Log',
                mc_tiles_number = 20,
                f_low_tiles_number = 20,
                response_TDI_version=1):
        '''
        Initialises new instance of quick look class. 

        Parameters:
            frequency_series_dict (dict): A dictionary containing frequency series data. Also contains information about the LISA mission such as
                time of observation etc. 
            search_prior_mc_f_low (arraylike): A dictionary containing the search prior in chirp mass and lower frequency.
            search_priors_in_other_parameters (arraylike): A dictionary containing the search priors in parameters we are pretty much agnostic to. 
            num_points_per_tile (int): The number of points in each tile of the search grid to sample.
            data_file_name (str): The name of the file containing the data to be searched over.
            segment (int, optional): The number of segments to use for the quick look algorithm. [Defaults to 100]
            include_spin (bool, optional): Whether to include spin in the search waveform. [Defaults to False]
            LDC_PSD (bool, optional): Whether to use the LDC PSD in the search. [Defaults to False]
            LDC_PSD_TDI_version (int, optional):  Wether to use the LDC TDI-1 PSD or TDI-2 PSD. [Defaults is 1]
            tiling_scheme (str, optional): The tiling scheme to use for the search in f_low. [Defaults to 'Log']
            mc_tiles_number (int,optional): The number of tiles to use in the chirp mass direction. [Defaults to 20]
            f_low_tiles_number (int,optional): The number of tiles to use in the lower frequency direction. [Defaults to 20]
        '''

        self.frequency_series_dict = frequency_series_dict

        self.prior_mc = search_prior_mc_f_low[0]
        
        self.prior_f_low = search_prior_mc_f_low[1]

        self.other_priors = search_priors_in_other_parameters

        self.num_points_per_tile = num_points_per_tile

        self.generate_search_tiles(mc_tiles_number,f_low_tiles_number)

        self.segment = segment 

        self.data_file_name = data_file_name

        if include_spin == True:
                self.waveform_func = TaylorF2EccSpin.BBHx_response_interpolate
                self.Ndim = 12 # 12D parameter space (TaylorF2+e0+chi1+chi2)
                self.spin_waveform = True
        else:
                self.waveform_func = TaylorF2Ecc.BBHx_response_interpolate
                self.Ndim = 10 # 12D parameter space (TaylorF2+e0+chi1+chi2)
                self.spin_waveform = False
        
        self.LDC_PSD = LDC_PSD

        self.LDC_PSD_TDI_version = LDC_PSD_TDI_version

        self.response_TDI_version = response_TDI_version

    def generate_search_tiles(self,mc_tiles_number,f_low_tiles_number):

        '''
        Generates the search tiles for the quick look algorithm.

        NOTE: Only log scheme for initial frequency currently implemented 
        '''

        # Boundary points for each tile in chirp mass
        mc_points = np.linspace(self.prior_mc[0],self.prior_mc[1],mc_tiles_number,endpoint=True)

        # Boundary points for each tile in lower frequency
        f_low_points = np.logspace(np.log10(self.prior_f_low[0]),np.log10(self.prior_f_low[1]),num=f_low_tiles_number)

        mc_segments = [[mc_points[i],mc_points[i+1]] for i in range(mc_points.size-1)]

        f_low_segments = [[f_low_points[i],f_low_points[i+1]] for i in range(f_low_points.size-1)]

        # Generate the search tiles by combinging the two grids
        self.global_search_tiles = []

        for f_low_segment in f_low_segments: 
            for mc_segment in mc_segments:
                self.global_search_tiles.append([f_low_segment,mc_segment])

        print('Number of search tiles: ',len(self.global_search_tiles))

        self.global_search_tiles=np.array(self.global_search_tiles)


    def generate_frequency_grids(self,f_min,mc_prior,f_low_prior):
        '''
        Generates the frequency grids FOR A GIVEN TILE.
        '''

        # Initialising values for frequency grid
        fmin = f_min
        fmax = self.frequency_series_dict['fmax']
        self.T_obs = self.frequency_series_dict['T_obs']

        # Downsampling factor is used for the sparse frequency grid for interpolation
        self.downsampling_factor = self.frequency_series_dict['downsampling_factor']
        
        # If frequencies are already generated and stored in a file, load them in
        if 'pregenerated_frequencies' in self.frequency_series_dict:
            if self.frequency_series_dict['pregenerated_frequencies'] == True:
                freqs = cp.asarray(np.load('freqs_filtered.npy'))
                df = cp.diff(freqs)[1]

            else:
                df = 1/T_obs
                freqs = cp.arange(fmin,fmax,df) # On GPU
        else:
                df = 1/self.T_obs
                freqs = cp.arange(fmin,fmax,df) # On GPU

        # Option to compute the maximum frequency for integration based on the search tile. 
        if 'compute_f_max_for_tile' in self.frequency_series_dict:
            if self.frequency_series_dict['compute_f_max_for_tile'] == True:

                eta_prior = self.other_priors[0]#
                e0_prior = self.other_priors[6]#

                search_tile_prior = np.array([mc_prior,
                                              eta_prior,
                                              f_low_prior,
                                              e0_prior])
                # Maximum frequency of integration for whole search 
                fmax = TaylorF2Ecc.f_high_tile_compute(search_tile_prior,
                                                   self.T_obs,
                                                   f_psd_high=fmax, # set default value for f_high in case we are merging within observation time to be whatever the user sets
                                                   safety_factor=1.1)
                print('f_max for search for this tile:',fmax)

                # Frequency mask to cut off the frequency grid at the maximum frequency for integration
                # Used below and when importing data. 
                frequency_mask = ((freqs<=fmax) & (freqs>=fmin))

                freqs = freqs[frequency_mask].copy() # On GPU

        # If not just use the whole frequency grid
        freqs_on_CPU = freqs.get() # On CPU

        freqs_sparse = freqs[::50]  # On GPU
        print('Sparse frequency grid size:',freqs_sparse.size)

        freqs_sparse_on_CPU = freqs_sparse.get() # On CPU (Used to compute A,f,phase on small number of points)

        return(freqs,df,freqs_on_CPU,freqs_sparse,freqs_sparse_on_CPU,fmax,frequency_mask)

    def generate_PSD(self,freqs,LDC=False,confusion=False,LDC_PSD_TDI_version=1):
        '''
        Generates the PSD for the search.

        - Harcoded to Michelson PSD for now 

        Args:
            freqs (array): The frequency grid to generate the PSD on.
            LDC (bool, optional): A flag indicating whether to use the LDC PSD. Defaults to False.
            confusion (bool, optional): A flag indicating whether to include confusion noise in the search for the PSD . Defaults to False.
            LDC_PSD_TDI_version (int, optional):  Wether to use the LDC TDI-1 PSD or TDI-2 PSD.
        
        Returns:
            psd_array (array): An array containing the PSD for each of the TDI channels.

        '''
        # Generate the PSD

        if LDC == True:
            # Conventions
            c = const.clight
            L = 2.5e+9/c # Armlength in seconds
            prefactor = (2*np.pi*1j*freqs*L)
            
            if LDC_PSD_TDI_version == 1:
                tdi2 = False
            elif LDC_PSD_TDI_version == 2:
                tdi2 = True

            noise = get_noise_model("sangria", freqs, wd=0)
            psd_A = noise.psd(freqs, option='A', tdi2 = tdi2)*1/np.abs(prefactor)**2
            psd_E = noise.psd(freqs, option='E', tdi2 = tdi2)*1/np.abs(prefactor)**2
            psd_T = noise.psd(freqs, option='T', tdi2 = tdi2)*1/np.abs(prefactor)**2

        else:
            Sdisp = Sdisp_SciRD(freqs)
            Sopt = Sopt_SciRD(freqs)
            psd_A = psd_AEX(freqs,Sdisp,Sopt)
            psd_E = psd_AEX(freqs,Sdisp,Sopt)
            psd_T = psd_TX(freqs,Sdisp,Sopt)

        if confusion == True:
            # Adding in confusion noise wont work with LDC psd 
            psd_A  = Add_confusion(freqs,psd_A,T_obs)
            psd_E  = Add_confusion(freqs,psd_E,T_obs)
            psd_T  = Add_confusion(freqs,psd_T,T_obs)

        psd_array = cp.array([psd_A,psd_E,psd_T])

        return(psd_array)

    def generate_initial_positions(self,priors,num_points):
        '''
        Latin hypercube sampling

        Parameters:
            priors:
            num_points: 

        Returns:
            initial_positions

        '''
        num_dimensions = priors.shape[0] 

        # Generate samples from the prior   
        initial_sampler = stats.qmc.LatinHypercube(num_dimensions,strength=1)
        initial_positions = initial_sampler.random(n=nsamples)*(np.ptp(priors,axis=1)) + np.array(priors)[:,0]

        return(initial_positions)

    def run_quick_look(self,constant_initial_phase=0):


        self.data = cp.asarray(np.load(self.data_file_name))

    
        for tile in self.global_search_tiles:

            f_low_prior = tile[0]
            mc_prior = tile[1]

            # Generate the frequency grids for the tile
            freqs,df,freqs_on_CPU,freqs_sparse,freqs_sparse_on_CPU,fmax,frequency_mask = self.generate_frequency_grids(f_low_prior[0],mc_prior,f_low_prior)

            # Generate the PSD for the tile
            psd_array = self.generate_PSD(freqs,LDC=self.LDC_PSD,LDC_PSD_TDI_version=self.LDC_PSD_TDI_version)

            waveform_args = {'freqs_sparse':freqs_sparse,
                                    'freqs_dense':freqs,
                                    'freqs_sparse_on_CPU':freqs_sparse_on_CPU,
                                    'f_high':fmax,
                                    'T_obs':self.T_obs,
                                    'TDIType':'AET',
                                    'logging': False,
                                    'TDIversion':self.response_TDI_version}
            # Generate the priors for this search tile
            priors = self.other_priors.copy()

            priors = np.insert(priors,0,mc_prior,axis=0)
            
            priors = np.insert(priors,7,f_low_prior/2,axis=0)# GW->Orbital frequency since thats what the waveforms take

            # Generate the initial positions for the tile # TODO FILL IN PRIORS
            initial_positions = self.generate_initial_positions(priors,self.num_points_per_tile)

            upsilons = []

            for source_index in range(self.num_points_per_tile):

                source_params = initial_positions[source_index].copy()

                # Add in orbital phase fixed so we can generate the waveform
                source_params.insert(7,constant_initial_phase)

                # Transform input source parameters to those expected in TaylorF2Ecc (mc,eta)->(m1,m2) + polarization shift
                source_parameters_transformed = TaylorF2Ecc_mc_eta_to_m1m2(initial_positions[source_index].copy())

                source_params = source_parameters_transformed[source_index]
            
                # Generate noiseless signal
                signal= self.waveform_func(source_parameters_transformed,**injection_waveform_args)

                upsilons.append(upsilon_func(signal,self.data[frequency_mask],psd_array,df,num_segments=self.segment))

            print('Maximum upsilon from quick-look for this tile: ',max(upsilons))
