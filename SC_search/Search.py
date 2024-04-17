try: 
    import cupy as cp 
except ImportError:
    print('Cupy not installed, search (on full FFT grid) wont work')
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os

from .Swarm_class import Semi_Coherent_Model, Semi_Coherent_Model_Inference, Coherent_Model_inference
from .Utility import TaylorF2Ecc_mc_eta_to_m1m2
from .Semi_Coherent_Functions import upsilon_func
from .Noise import *
from .Waveforms import TaylorF2Ecc
import PySO
from .Veto import *

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
                 PySO_num_swarms,
                 PySO_num_particles,  
                 PySO_kwargs, 
                 include_noise=True,
                 load_data_file = False,
                 data_file_name = None,
                 noise_only_injection = False,
                 masking = None):
        '''
        Initializes a new instance of the Search class.

        Parameters:
            frequency_series_dict (dict): A dictionary containing frequency series data. Also contains information about the LISA mission such as
                time of observation etc. 
            source_parameters (list): A dictionary containing source parameters for the true injection. Should be a nested list. Every item in this list
                is a new source. 
            segment_ladder (list): A list of segment ladder values for the semi-coherent search.
            prior_bounds (list): A list of prior bounds for the search
            PySO_num_particles (int): The number of particles to be used in the PySO search.
            PySO_num_swarms (int): The initial number of swarms to be used in the PySO search.
            PySO_kwargs (dict): A dictionary containing PySO keyword arguments.
            include_noise (bool, optional): A flag indicating whether to include noise. Defaults to True.
            load_data (bool, optional): A flag indicating whether to load the data to be searched over. 
            data_file_name (str, optional): The name of the file containing the data to be searched over.
            noise_only_injection (bool, optional): A flag indicating whether to inject noise only. Defaults to False.  
            masking (list, optional): A list of booleans indicating wether to use the masked upsilon function at each segment. Defaults to [False]*len(segment_ladder).
        '''

        self.frequency_series_dict = frequency_series_dict

        self.source_parameters = source_parameters
        
        self.segment_ladder = segment_ladder
        
        self.prior_bounds = prior_bounds
        
        self.PySO_num_particles = PySO_num_particles

        self.PySO_num_swarms = PySO_num_swarms

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
                              'freqs_sparse_on_CPU':self.freqs_sparse_on_CPU,
                              'f_high':self.fmax,
                              'T_obs':self.T_obs,
                              'TDIType':'AET',
                              'logging': False}

        # Generate signal or load the signal we will be searching for 
        if load_data_file == True:
            # Load in data
            self.data = cp.asarray(np.load(data_file_name))
        elif noise_only_injection == True and load_data_file == False:
            # Generate data containing only noise
            self.data = self.generate_noise_realisation()
        elif noise_only_injection == False and load_data_file == False:
            # Generate injection data
            self.generate_injection_data(include_noise)
            # Check upsilons values for the injection 
                # Only do this for not-loaded in data as we can isolate the noise and the signal
            self.check_upsilons()

        # Default to using the non masked function for the upsilon statistic
        if masking == None:
            self.masking_ladder = [False]*len(self.segment_ladder)
        else: 
            self.masking_ladder = masking

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

        self.freqs = cp.arange(self.fmin,self.fmax,self.df) # On GPU
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
        num_sources = len(self.source_parameters)

        print('Data contains '+ str(num_sources)+' sources!')

        self.injection = cp.zeros((3,self.freqs.size),dtype=cp.complex)

        for source in self.source_parameters:

            # Transform input source parameters to those expected in TaylorF2Ecc (mc,eta)->(m1,m2) + polarization shift
            source_parameters_transformed = TaylorF2Ecc_mc_eta_to_m1m2(source)
            
            # Turn logging on for the injection waveform so we can debug statements 
            injection_waveform_args = self.waveform_args.copy()
            injection_waveform_args['logging'] = True
            
            # Generate noiseless signal
            signal= self.waveform_func(source_parameters_transformed,**injection_waveform_args)

            # Print SNR of injection signal
            injection_SNR = cp.sqrt(4*cp.real(cp.sum((signal*signal.conj()/self.psd_array*self.df)))).item()
            print('SNR of signal:',injection_SNR)

            self.injection += signal

        # Include noise if asked for
        if include_noise == True:
            noise = self.generate_noise_realisation()
            self.data = noise + self.injection
        else: 
            self.data = self.injection.copy()

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
	        print('Sense check log upsilon: ',upsilon_func(self.injection,self.data,self.psd_array,self.df,num_segments=segment),
               'N at ',segment)

    def veto_function(self,parameters,best_upsilon_value,segment_number):
        '''
        Veto function for the search.  

        FUNCTION NOT YET USED. 

        Args:
            parameters (array-like): The parameters for the model at the location to be veto-checked.
            best_upsilon_value (float): The value of the search statistic at this location in parameter space.
            segment_number (int): The segment number for the model.
        Returns:
            veto (Boolean): A flag indicating whether the peak is vetoed or not. 
        '''
                
        # Transform input source parameters to those expected in TaylorF2Ecc (mc,eta)->(m1,m2) + polarization shift
        source_parameters_transformed = TaylorF2Ecc_mc_eta_to_m1m2(parameters)
    
        # Generate waveform
        veto_waveform = self.waveform_func(source_parameters_transformed,**self.waveform_args)

        # Print SNR of injection signal
        veto_SNR = cp.sqrt(4*cp.real(cp.sum((veto_waveform*veto_waveform.conj()/self.psd_array*self.df)))).item()

        theoretical_pf = false_alarm_rate(best_upsilon_value,veto_SNR,segment_number)
        
        pass
    
    def initialize_and_run_search(self,):
        """
        Initializes the hierarchical search, via the PySO package, for the semi-coherent search.
        """
        # Initialise classes at each segment for the semi-coherent search for the semi-coherent search

        self.Semi_Coherent_classes = [Semi_Coherent_Model(segment_number,
                                                            self.prior_bounds,
                                                            self.data,
                                                            self.psd_array,
                                                            self.df,
                                                            self.waveform_func,
                                                            waveform_args=self.waveform_args,masking=self.masking_ladder[segment_index]) for segment_index,segment_number in enumerate(self.segment_ladder)]
        
        PySO_search = PySO.HierarchicalSwarmHandler(self.Semi_Coherent_classes,
                                self.PySO_num_swarms,# Number of initial swarms
                                self.PySO_num_particles,# Number of particles
                                **self.PySO_kwargs)

        PySO_search.Run()

    def postprocess_seperate_inferences(self,):
        '''
        Read in the results of the PySO_search and seperate into multiple folders where inferences for each will be done. 
        '''
        # Directory where all the PySO results are dumped
        Swarm_results_file =  pd.read_csv(self.PySO_kwargs['Output']+'/EnsembleEvolutionHistory.dat')

        # Find interation number of final iteration
        final_iteration = np.sort(np.unique(Swarm_results_file['IterationNumber']))[-1]

        # Find final state
        df_subset_final_iteration = Swarm_results_file[Swarm_results_file['IterationNumber'] == final_iteration]

        # Unique swarms in the final state of the search
        unique_swarm_numbers = np.unique(df_subset_final_iteration['swarm_number'])

        # For each swarm, create a directory and dump the results of the swarms final positions in there
        for swarm_num in unique_swarm_numbers: 
            os.mkdir('Swarm_'+str(swarm_num+1)+'_inference')
        
            # Find the final positions of the swarm
            final_positions = df_subset_final_iteration[Swarm_results_file['swarm_number'] == swarm_num]

            # Save the final positions of the swarm
            final_positions.to_csv('Swarm_'+str(swarm_num+1)+'_inference/final_positions.csv')


class Post_Search_Inference:
    '''
    Class to perform inference on the results of the search.
        Each swarm from the search is loaded in and inference is performed on it using the 
        Affine-invariance MCMC sampling algorithm implemented within PySO.

    Harcoded to the N-1, phase maximised coherent log likelihood. 

    '''        
    def __init__(self, 
                 frequency_series_dict, 
                 prior_bounds, 
                 data_file_name,
                 swarm_directory,
                 PySO_MCMC_kwargs,
                 coherent_or_N_1='N_1',
                 Spread_multiplier=None):
        '''
        Initializes a new instance of the Post Search Inference class.

        Parameters:
            frequency_series_dict (dict): A dictionary containing frequency series data. Also contains information about the LISA mission such as
                time of observation etc. 
            prior_bounds (list): A list of prior bounds for the inference
            data_file_name (str): The name of the file containing the data.
            swarm_directory (str): The directory containing the results of the search for the swarm to be inferred over.
            PySO_MCMC_kwargs (dict): A dictionary containing PySO MMCMC keyword arguments.
            coherent_or_N_1 (str, optional): A flag indicating whether to perform coherent or N-1 PE. Defaults to 'N_1'.
            Spread_multiplier (float, optional): A multiplier for the spread of the initial positions for the MCMC. Defaults to None.
                Role is to make the particles in the swarm spread out a bit more before inference. 
        '''

        self.frequency_series_dict = frequency_series_dict
        
        self.prior_bounds = prior_bounds

        self.PySO_MCMC_kwargs = PySO_MCMC_kwargs

        # Generate CPU and GPU frequency grids
        self.generate_frequency_grids()

        # Generate PSD
        self.generate_psd()

        # Search is being tuned for these so hardcoded for now
        self.waveform_func = TaylorF2Ecc.BBHx_response_interpolate
        self.waveform_args = {'freqs_sparse':self.freqs_sparse,
                              'freqs_dense':self.freqs,
                              'freqs_sparse_on_CPU':self.freqs_sparse_on_CPU,
                              'f_high':self.fmax,
                              'T_obs':self.T_obs,
                              'TDIType':'AET',
                              'logging': False}

        # Load in data
        self.data = cp.asarray(np.load(data_file_name))

        self.swarm_directory = swarm_directory

        # Load positions from final iteration of the search for one swarm
            # Note this does not include distances!!! Since the search statistic does not search over that
        self.initial_positions = pd.read_csv(self.swarm_directory +'/final_positions.csv').to_numpy()[:,3:-3]


        if Spread_multiplier != None:
            # Increase the spread of the initial positions from the means
            self.increase_initial_position_spread(Spread_multiplier)

        # Draw distances from prior and insert into initial positions
        self.draw_distances_from_prior()

        if coherent_or_N_1 == 'Coherent':

            # Draw initial orbital phases for the coherent PE if requested
            self.draw_initial_orbital_phases()

        # If not coherent, ie N_1 no need to generate initial orbital phases as we do a phase maximisation anyway 
        

    def increase_initial_position_spread(self,Spread_multiplier):
        '''
        Multiply the distance of each particle from the mean of the swarm by a factor of the spread multiplier.
        '''

        # Mean across whole swarm of positions across each dimension 
        axis_means = np.mean(self.initial_positions,axis=0)
        self.initial_positions = axis_means + Spread_multiplier*(self.initial_positions - axis_means)



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

        self.freqs = cp.arange(self.fmin,self.fmax,self.df) # On GPU
        self.freqs_on_CPU = self.freqs.get() # On CPU

        self.freqs_sparse = self.freqs[::self.downsampling_factor]  # On GPU

        self.freqs_sparse_on_CPU = self.freqs_sparse.get() # On CPU (Used to compute A,f,phase on small number of points)

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
    
    def draw_distances_from_prior(self,):
        '''
        Draws a distances from the prior and fills it into the initial guesses for the inference. 
            As the search does not search over distance, this is a necessary step. 
        '''
        distance_draws = np.random.uniform(self.prior_bounds[2][0],self.prior_bounds[2][1],size=(self.initial_positions.shape[0]))
        
        # Insert distances into the correct index of the initial positions
        self.initial_positions = np.insert(self.initial_positions,2,distance_draws,axis=1)

    def draw_initial_orbital_phases(self,):
        '''
        Draws initial orbital phases from a uniform distribution and fills it into the initial guesses for the inference. 
            As the search does not search over initial orbital phase, this is a necessary step. 

        Only used for coherent post search PE. 
        '''
        initial_orbital_phase_draws = np.random.uniform(self.prior_bounds[7][0],self.prior_bounds[7][1],size=(self.initial_positions.shape[0]))
        
        # Insert distances into the correct index of the initial positions
        self.initial_positions = np.insert(self.initial_positions,7,initial_orbital_phase_draws,axis=1) 

    def initialize_and_run_inference_N_1(self,):
        '''
        Initializes and runs the inference on the results of the search
        '''
        Coherent_phase_maximised_inference_model = Semi_Coherent_Model_Inference(
                                                            self.prior_bounds,
                                                            self.data,
                                                            self.psd_array,
                                                            self.df,
                                                            self.waveform_func,
                                                            segment_number = 1,
                                                            waveform_args=self.waveform_args)
        
        sampler = PySO.Swarm(Coherent_phase_maximised_inference_model,
                        self.initial_positions.shape[0], # Num particless
                        Initialguess = self.initial_positions, # Initial guess
                        Output = self.swarm_directory,
                        **self.PySO_MCMC_kwargs)

        sampler.Run()

        # Load in swarm history 
        posterior_samples_from_search = pd.read_csv(self.swarm_directory+'/SwarmEvolutionHistory.dat').to_numpy()[:,1:-1]

        # Save samples 
        np.savetxt(self.swarm_directory+'/posterior_samples.dat',posterior_samples_from_search) 

    def initialize_and_run_inference_Coherent(self,):
        '''
        Initializes and runs the inference on the results of the search
        '''
        Coherent_phase_maximised_inference_model = Coherent_Model_inference(
                                                            self.prior_bounds,
                                                            self.data,
                                                            self.psd_array,
                                                            self.df,
                                                            self.waveform_func,
                                                            waveform_args=self.waveform_args)
        
        sampler = PySO.Swarm(Coherent_phase_maximised_inference_model,
                        self.initial_positions.shape[0], # Num particless
                        Initialguess = self.initial_positions, # Initial guess
                        Output = self.swarm_directory,
                        **self.PySO_MCMC_kwargs)

        sampler.Run()

        # Load in swarm history 
        posterior_samples_from_search = pd.read_csv(self.swarm_directory+'/SwarmEvolutionHistory.dat').to_numpy()[:,1:-1]

        # Save samples 
        np.savetxt(self.swarm_directory+'/posterior_samples.dat',posterior_samples_from_search) 




if __name__=='__main__':

    year_in_seconds = 365.25*24*60*60

    
    # Frequency series parameters    
    frequency_series_dict = {'fmin':0.018,
                             'fmax':0.03,
                             'T_obs':3*year_in_seconds,
                             'downsampling_factor':1000}

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



    PySO_kwargs = {'Omega':Omega,
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
                 500, 
                 PySO_kwargs, 
                 include_noise=True)
    
    print('Search setup complete. Running search...')
    
    Search_object.initialize_and_run_search()
    

    PySO_MCMC_kwargs= {'Verbose':True,
                        'Nperiodiccheckpoint':1, # Final two args mean evolution is saved at every iteration. Only necessary if running myswarm.Plot()
                        'Saveevolution':True,    ############
                        'Nthreads':5,
                        'Tol':1.0e-2,
                        'Omega':0., 
                        'Phip':0., 
                        'Phig':0., 
                        'Mh_fraction':1.,
                        'Maxiter':20,}