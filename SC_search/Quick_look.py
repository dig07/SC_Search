try: 
    import cupy as cp 
except ImportError:
    print('Cupy not installed, search (on full FFT grid) wont work')

try: 
    from ldc.lisa.noise import get_noise_model
except ImportError:
    print('LDC not installed')

# Try importing torch multiprocessing, if it fails, use the default multiprocessing
try: 
    from torch.multiprocessing import Pool, set_start_method
    set_start_method('spawn',force=True)
    parallel=True
except ImportError: 
    print('Torch multiprocessing not installed, no parallelisation')
    parallel = False

import matplotlib.pyplot as plt
import os
import scipy.stats as stats


#from .Swarm_class import Semi_Coherent_Model
from .Utility import TaylorF2Ecc_mc_eta_to_m1m2
from .Semi_Coherent_Functions import upsilon_func, semi_coherent_match, coherent_match
from .Noise import *
from .Waveforms import TaylorF2Ecc, TaylorF2EccSpin
from .Waveforms import Constants as const
import numpy as np 

import traceback
from functools import partial
from itertools import repeat



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
                response_TDI_version=1,
                constant_distance=100.e+6,
                Nthreads=1):
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
            response_TDI_version (int, optional): The TDI version to use in the response. [Defaults to 1]
            constant_distance (float, optional): The constant distance to use in the search. [Defaults to 100e+6]
                (Note this is used just to avoid numerical errors, the distance terms drops out of the waveform in the search statistic)
            Nthreads (int, optional): The number of parallel to use in the search. [Defaults to 1] 
                (Note this is only used if torch.multiprocessing is available, i.e. we can parallelise over the GPU)
        '''

        self.frequency_series_dict = frequency_series_dict

        self.prior_mc = search_prior_mc_f_low[0]
        
        self.prior_f_low = search_prior_mc_f_low[1]

        self.other_priors = search_priors_in_other_parameters

        self.num_points_per_tile = num_points_per_tile

        self.generate_search_tiles(mc_tiles_number,f_low_tiles_number)

        self.segment = segment 

        self.constant_distance = constant_distance

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

        self.Nthreads=Nthreads

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
        
        eta_prior = self.other_priors[0]#
        e0_prior = self.other_priors[5]#

        search_tile_prior = np.array([mc_prior,
                                        eta_prior,
                                        f_low_prior,
                                        e0_prior])
        # Maximum frequency of integration for whole search 
        fmax = TaylorF2Ecc.f_high_tile_compute(search_tile_prior,
                                            self.T_obs,
                                            f_psd_high=fmax, # set default value for f_high in case we are merging within observation time to be whatever the user sets
                                            safety_factor=1.1)


        frequency_mask = ((self.freqs<=fmax) & (self.freqs>=fmin))

        # Downsampling to target number of frequency points
        downsampling_factor = self.freqs.size//self.frequency_series_dict['target_number_of_frequency_points']

        print('f_gw min (set by tile): ',fmin)
        print('f_gw max (for integration): ',fmax)

        return(frequency_mask,downsampling_factor,fmax)

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
        initial_positions = initial_sampler.random(n=num_points)*(np.ptp(priors,axis=1)) + np.array(priors)[:,0]

        return(initial_positions)

    def run_quick_look(self,constant_initial_phase=0):

        self.freqs = cp.asarray(np.load('freqs.npy'))
        
        self.df = self.freqs[1]-self.freqs[0]

        self.freqs_on_CPU = self.freqs.get()
        
        self.psd_array = self.generate_PSD(self.freqs,LDC=self.LDC_PSD,LDC_PSD_TDI_version=self.LDC_PSD_TDI_version)        

        self.data = cp.asarray(np.load(self.data_file_name))

        self.max_upsilons = []
    
        for tile in self.global_search_tiles:

            f_low_prior = tile[0]
            mc_prior = tile[1]

            try:
                # Generate the frequency grids for the tile
                self.frequency_mask,downsampling_factor,fmax = self.generate_frequency_grids(f_low_prior[0],mc_prior,f_low_prior)

                self.waveform_args = {'freqs_sparse':self.freqs[self.frequency_mask][::downsampling_factor],
                                        'freqs_dense':self.freqs[self.frequency_mask],
                                        'freqs_sparse_on_CPU':self.freqs_on_CPU[self.frequency_mask.get()][::downsampling_factor],
                                        'f_high':fmax,
                                        'T_obs':self.T_obs,
                                        'TDIType':'AET',
                                        'logging': False,
                                        'TDIversion':self.response_TDI_version}
                # Generate the priors for this search tile
                priors = self.other_priors.copy()

                priors = np.insert(priors,0,mc_prior,axis=0)
                
                priors = np.insert(priors,6,f_low_prior/2,axis=0)# GW->Orbital frequency since thats what the waveforms take

                # Generate the initial positions for the tile # TODO FILL IN PRIORS
                initial_positions = self.generate_initial_positions(priors,self.num_points_per_tile)

                transformed_waveform_parameters = []

                for source_index in range(self.num_points_per_tile):

                    source_params = initial_positions[source_index].copy()
                    
                    source_params = list(source_params)

                    # Add in distance fixed so we can generate the waveform
                    source_params.insert(2,self.constant_distance)

                    # Add in orbital phase fixed so we can generate the waveform
                    source_params.insert(7,constant_initial_phase)

                    # Transform input source parameters to those expected in TaylorF2Ecc (mc,eta)->(m1,m2) + polarization shift
                    source_parameters_transformed = TaylorF2Ecc_mc_eta_to_m1m2(source_params.copy())

                    transformed_waveform_parameters.append(source_parameters_transformed)

                # If parallel try and run multiple computations across the GPU at once 
                if parallel == True and self.Nthreads>1:

                    # Create multiprocessing pool
                    self.Pool = Pool(self.Nthreads)
                    print('before upsilons computation')
                    upsilons = list(self.Pool.starmap(generate_waveform_and_compute_upsilon, zip(transformed_waveform_parameters,repeat((self.waveform_args,self.data[:,self.frequency_mask],
                                                      self.psd_array[:,self.frequency_mask],self.df,self.segment)))))
                    self.Pool.close()
                    self.Pool.join()
                    print('bbbbbbbbbbbb')
                # If not parallelisable, just do it linearly. 
                else:
                    upsilons = list(map(self.generate_waveform_and_compute_upsilon,transformed_waveform_parameters))
  
                print('Maximum upsilon from quick-look for this tile: ',max(upsilons))
                print('Maximum upsilon point: ',initial_positions[np.argmax(upsilons)])

                self.max_upsilons.append(max(upsilons))
            except Exception as e:
                # -1 is for us an error code that we can remove in postprocessing and we can try and figure out if there is something wrong with a tile
                self.max_upsilons.append(-1)
                print(e)
                print(traceback.format_exc())

        self.save_results()



    def save_results(self):
        '''
        Output results of quick look to a file
        '''

        tiles = np.array(self.global_search_tiles).reshape(len(self.global_search_tiles),4)

        upsilons_results = np.array(self.max_upsilons).reshape(len(self.max_upsilons),1)

        results = np.hstack((tiles,upsilons_results))

        np.savetxt('quick_look_results.txt',results)

def generate_waveform_and_compute_upsilon(source_params,a):
    '''
    Generate a waveform and compute the upsilon value for that waveform.
    Wrapped into its own function to allow for parallelisation over GPU. 

    Args:
    source_params (array): The source parameters to generate the waveform for (transformed into their correct form).

    Returns:
    upsilon (float): The upsilon value for the waveform

    '''
    waveform_args,data,psd_array,df,num_segments= a
    signal= TaylorF2Ecc.BBHx_response_interpolate(source_params,**waveform_args)

    upsilon = upsilon_func(signal,data,psd_array,df,num_segments=num_segments)

    return(upsilon)
