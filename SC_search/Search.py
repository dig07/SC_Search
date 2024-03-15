try: 
    import cupy as cp 
except ImportError:
    print('Cupy not installed, search')
import numpy as np 

import matplotlib.pyplot as plt

from .Swarm_class import Semi_Coherent_Model
from Utility import TaylorF2Ecc_mc_eta_to_m1m2
from .Semi_Coherent_Functions import upsilon_func
from Noise import *
from Waveforms import TaylorF2Ecc
import PySO

class Search:
    '''
    - Frequency series is the full FFT grid
    - Hardcoded to TaylorF2Ecc waveform for now
    - Hardcoded to Michelson PSD for now
    '''
    def __init__(self, 
                 frequency_series_dict, 
                 source_parameters, 
                 segment_ladder, 
                 prior_bounds,
                 PySO_num_particles,  
                 PySO_kwargs, 
                 include_noise=True):
        '''
        Initializes a new instance of the Search class.

        Parameters:
            frequency_series_dict (dict): A dictionary containing frequency series data. Also contains information about the LISA mission such as
                time of observation etc. 
            source_parameters (dict): A dictionary containing source parameters for the true injection.
            segment_ladder (list): A list of segment ladder values for the semi-coherent search.
            prior_bounds (list): A list of prior bounds for the search
            PySO_num_particles (int): The number of particles to be used in the PySO search.
            PySO_kwargs (dict): A dictionary containing PySO keyword arguments.
            include_noise (bool, optional): A flag indicating whether to include noise. Defaults to True.
        '''

        self.frequency_series_dict = frequency_series_dict

        self.source_parameters = source_parameters
        
        self.segment_ladder = segment_ladder
        
        self.prior_bounds = prior_bounds
        
        self.PySO_num_particles = PySO_num_particles

        self.PySO_kwargs = PySO_kwargs

        # Generate CPU and GPU frequency grids
        self.generate_frequency_grids()

        # Generate PSD
        self.generate_psd()

        # TODO: Change the function being injected to the direct FFT grid (no interpolation) one just to be rigorous 

        # Search is being tuned for these so hardcoded for now
        self.waveform_func = TaylorF2Ecc.BBHx_response_interpolate
        self.waveform_args = {'freqs_sparse':self.freqs_sparse,
                              'freqs_dense':self.freqs,
                              'freqs_sparse_on_CPU':self.freqq_sparse_on_CPU,
                              'f_high':self.fmax,
                              'T_obs':self.T_obs,
                              'TDIType':'AET',
                              'logging': False}

        # Generate signal we will be searching for 
        self.generate_injection_data(include_noise)

        # Check upsilons values for the injection 
        self.check_upsilons()


    def generate_frequency_grids(self,):
        '''
        Generates the dense and sparse frequency grids for search. 
        Stores both on CPU and GPU.         
        '''

        # Initialising values for frequency grid
        self.fmin = self.frequency_series_dict['fmin']
        self.fmax = self.frequency_series_dict['fmax']
        self.T_obs = self.frequency_series_dict['T_obs']

        # Downsampling factor is used for the sparse frequency grid for interpolation
        self.downsampling_factor = self.frequency_series_dict['downsampling_factor']

        # Generating frequency grid (dense)
        self.df = 1/self.T_obs

        self.freqs = cp.arrange(self.fmin,self.fmax,self.df) # On GPU
        self.freqs_on_CPU = self.freqs.get() # On CPU

        self.freqs_sparse = self.freqs[::self.downsampling_factor]  # On GPU

        self.freqs_sparse_on_CPU = self.freqs_sparse.get() # On CPU (Used to compute A,f,phase on small number of points)

    def generate_noise_realisation(self,):
        '''
        Generates a noise realisation for injecting into data

        - Harcoded to Michelson PSD for now 

        Returns:
            noise_: Noise realization (3,#FFTgrid)
        '''
        # Generate noise in each channel
        noise_A = noise_realization(self.psd_A,self.T_obs,self.df,FD=True)
        noise_E = noise_realization(self.psd_E,self.T_obs,self.df,FD=True)
        noise_T = noise_realization(self.psd_T,self.T_obs,self.df,FD=True)

        noise_ = cp.array([noise_A,noise_E,noise_T]) # On GPU

        return noise_        

    def generate_psd(self,):
        '''
        Generates the PSD for the search.

        - Harcoded to Michelson PSD for now 
        '''
        # Generate the PSD
        Sdisp = Sdisp_SciRD(self.freqs_on_CPU)
        Sopt = Sopt_SciRD(self.freqs_on_CPU)
        self.psd_A = psd_AEX(self.freqs_on_CPU,Sdisp,Sopt)
        self.psd_E = psd_AEX(self.freqs_on_CPU,Sdisp,Sopt)
        self.psd_T = psd_TX(self.freqs_on_CPU,Sdisp,Sopt)

        self.psd_array = cp.array([self.psd_A,self.psd_E,self.psd_T]) # On GPU
    
    def generate_injection_data(self,include_noise=True):
        '''
        Generates the injection data for the search. 
        Saves the data to a file (after conversion to numpy array).
        Optionally adds noise. 

        Args:
            include_noise (bool, optional): A flag indicating whether to include noise. Defaults to True.
        
        '''
        
        # Transform input source parameters to those expected in TaylorF2Ecc (mc,eta)->(m1,m2) + polarization shift
        source_parameters_transformed = TaylorF2Ecc_mc_eta_to_m1m2(self.source_parameters)
        
        # Turn logging on for the injection waveform so we can debug statements 
        injection_waveform_args = self.waveform_args.copy()
        injection_waveform_args['logging'] = True
        
        # Generate noiseless signal
        self.injection_model = self.waveform_func(source_parameters_transformed,**injection_waveform_args)

        # Print SNR of injection signal
        self.injection_SNR = cp.sqrt(4*cp.real(cp.sum((self.injection_model*self.injection_model.conj()/self.psd_array*self.df)))).item()
        print('SNR of injection signal:',self.injection_SNR)

        # Include noise if asked for
        if include_noise == True:
            noise = self.generate_noise_realisation 
            self.data = noise + self.injection_model
        else: 
            self.data = self.injection_model.copy()

        # Save data 
        self.cupy_to_numpy_save(self.data,'data.npy')

    def cupy_to_numpy_save(self,array,filename):
        '''
        Converts array to numpy from cupy and saves it to a file

        Args:
            array (array): The array to be saved.
            filename (str): The filename to save the array to.
        '''
        np.save(filename,array.get())

    def check_upsilons(self,):
        '''
        Check the value of the upsilon function for different segment values. for the injection 
        '''

        for segment in self.segment_ladder:
	        print('Sense check log upsilon: ',upsilon_func(self.injection_model,self.data,self.psd_array,num_segments=segment),
               'N at ',segment)

    
    def initialize_and_run_search(self,):
        """
        Initializes the hierarchical search, via the PySO package, for the semi-coherent search.
        """
        # Initialise classes at each segment for the semi-coherent search for the semi-coherent search
        self.Semi_Coherent_classes = [Semi_Coherent_Model(segment_number,
                                                            self.prior_bounds,
                                                            self.data,
                                                            self.psd_array,
                                                            self.waveform_func,
                                                            self.waveform_args) for segment_number in self.segment_ladder]
        
        PySO_search = PySO.HierarchicalSearch(self.Semi_Coherent_classes,
                                1,# Number of initial swarms
                                self.PySO_num_particles,# Number of particles
                                **self.PySO_kwargs)

        PySO_search.Run()

# Need to incorporate Vetos into this function and into PySO

if __name__=='__main__':

    year_in_seconds = 365.25*24*60*60

    
    # Frequency series parameters    
    frequency_series_dict = {'fmin':0.018,
                             'fmax':0.03,
                             'T_obs':3*year_in_seconds}

    # Injection parameters
    source_parameters = [28.09555579546043,#mc [sm]
                            0.24710059171597634,#eta
                            150.1*(1.e+6),#D
                            np.pi/4,#beta [rads]
                            2.01,#lambda [rads]
                            2.498091544796509,#inc [rads]
                            -1.851592653589793,#polarization [rads],
                            0,#phi0 [rads]
                            0.018/2,
                            0.01]#e0 

    # Segment ladder for hierarchical search 
    segment_ladder = [100,80,60,40,20,1]

    prior_bounds = [[28.05,28.15],#mc
                   [0.2,0.249999999999],#eta
                   [10.e+6,250.e+6],#D
                   [0.7,0.9],#beta
                   [1.8,2.2],#lambda
                   [1.5,3.0],#inc
                   [-3.1,-0.2],#polarization
                   [-2.5,2.5],#phi0
                   [0.0089999,0.0090001],#flow
                   [0.007,0.02]]#e0
    

    Omega = [0.4,0.4,0.3,0.3,0.2,0.2]
    Phip = [0.2,0.2,0.2,0.2,0.2,0.2]
    Phig = [0.4,0.4,0.5,0.5,0.6,0.6]
    Mh_fraction = [0.0,0.0,0.0,0.0,0.0,0.0]

    # Common arguments for every swarm in the hierarchical PySO

    	#False here is just disabling each swarms own logging, we have master logging in the hierarchical swarm 
    Swarm_kwargs = {'Verbose':False, 'Periodic':[0,
		                                     0,
		                                     0,
		                                     1,#beta
		                                     1,#lambda
		                                     0,
		                                     1,#pol
		                                     1,#initorbitphase
		                                     0,
		                                     0]} 


    # Minimum velocities for the hierarchical PySO



    v_min_array_100 = np.array([0.001,#Mc
		           0.01,#eta
		           5.e+6,#D
		           0.01,#beta
		           0.01,#lambda
		           0.1,#i
		           1.,#psi
		           1.,#phi0
		           1.e-7,#f_low
		           0.01]) #e0

    minimum_velocities = np.array([v_min_array_100,
		                       v_min_array_100,
		                       v_min_array_100*0.5,
		                       v_min_array_100*0.25,
		                       v_min_array_100*0.1,
		                       v_min_array_100*0.1])



    PySO_kwargs = {'num_particles':500,
                    'Omega':Omega,
                    'Phip':Phip,
                    'Phig':Phig,
                    'Mh_fraction':Mh_fraction,
                    'Swarm_kwargs':Swarm_kwargs,
                    'Output':'hierarchical_results/',
                    'Nperiodiccheckpoint':1,
                    'Verbose':True,
                    'Saveevolution':True, 
                    'Minimum_exploration_iterations':70,
                    'Initial_exploration_limit':30,
                    'Maximum_number_of_iterations_per_step':150,
                    'Kick_velocities':False, 
                    'Use_func_vals_in_clustering': False,
                    'Max_particles_per_swarm':50,
                    'Minimum_velocities':minimum_velocities,
                    'Nthreads':2}


    Search_object = Search(frequency_series_dict, 
                 source_parameters, 
                 segment_ladder, 
                 prior_bounds, 
                 PySO_kwargs, 
                 include_noise=True)
    
    print('Search setup complete. Running search...')
    
    Search_object.initialize_and_run_search()
    