try: 
    import cupy as cp 
except ImportError:
    print('Cupy not installed, search (on full FFT grid) wont work')

try: 
    import zeus
except ImportError:
    print('Zeus not installed')

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os

from .Swarm_class import Semi_Coherent_Model, Semi_Coherent_Model_Inference, Coherent_Model_inference
from .Utility import TaylorF2Ecc_mc_eta_to_m1m2
from .Semi_Coherent_Functions import upsilon_func, semi_coherent_match, coherent_match
from .Noise import *
from .Waveforms import TaylorF2Ecc
import PySO
from .Veto import *
from .Inference import dynesty_inference
import networkx as nx

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
            source_parameters_transformed = TaylorF2Ecc_mc_eta_to_m1m2(source.copy())
            
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

    def postprocess_seperate_inferences(self,crashed=False):
        '''
        Read in the results of the PySO_search and seperate into multiple folders where inferences for each will be done. 

        Args:
            crashed (bool, optional): A flag indicating whether the search crashed/timed out, if so use the final iteration -1 as the 
            real final iteration, as this will be the last iteration that is complete in the search data file. Defaults to False.
        '''
        # Directory where all the PySO results are dumped
        Swarm_results_file =  pd.read_csv(self.PySO_kwargs['Output']+'/EnsembleEvolutionHistory.dat')

        # Find interation number of final iteration
        final_iteration = np.sort(np.unique(Swarm_results_file['IterationNumber']))[-1]

        if crashed == True:
            final_iteration = final_iteration - 1

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

    def combine_swarms_on_match(self,match_threshold=0.98,crashed=False):
        '''
        Combine swarms based on the match threshold. 

        If swarm i and swarm j have best particles which have a match(i,j) above the threshold then combine them. 

        Args:
            match_threshold (float, optional): The threshold above which to combine swarms. Defaults to 0.98.
            crashed (bool, optional): A flag indicating whether the search crashed/timed out, if so use the final iteration -1 as the 
            real final iteration, as this will be the last iteration that is complete in the search data file. Defaults to False.
        '''
    
        injection_waveform_args = self.waveform_args.copy()
    
        # Directory where all the PySO results are dumped
        Swarm_results_file =  pd.read_csv(self.PySO_kwargs['Output']+'/EnsembleEvolutionHistory.dat')

        # Find interation number of final iteration
        final_iteration = np.sort(np.unique(Swarm_results_file['IterationNumber']))[-1]

        if crashed == True:
            final_iteration = final_iteration - 1

        # Find final state
        df_subset_final_iteration = Swarm_results_file[Swarm_results_file['IterationNumber'] == final_iteration]

        # Unique swarms in the final state of the search
        unique_swarm_numbers = np.unique(df_subset_final_iteration['swarm_number'])

        # Best positions for each swarm
        best_positions = np.zeros((unique_swarm_numbers.size,10))# 10 D parameter space (adding in artificial orbital phase and distance for waveform generation)

        for swarm_index,swarm_num in enumerate(unique_swarm_numbers): 

            # Find the final positions of the swarm
            final_positions = df_subset_final_iteration[Swarm_results_file['swarm_number'] == swarm_num]

            # Extract final positions of the swarm into numpy array, [:,2,-3] filters down just to the parameter space locations 
            parameter_space_positions = final_positions.to_numpy()[:,2:-3]

            # Find particle position in this swarm with max upsilon
            function_values = final_positions.to_numpy()[:,-3]

            best_particle_index = np.argmax(function_values)

            best_particle_position = list(parameter_space_positions[best_particle_index,:])

            # Add in distance fixed here so we can generate the waveform
            #   Note this does not affect overlap at all, this is just a practicality to generate the waveform. 
            best_particle_position.insert(2,100.e+6)

            # Add in orbital phase fixed so we can generate the waveform
            #   Note the value we pick here will not affect the phase maximised overlap as it is one segment and phase maximised. 
            best_particle_position.insert(7,np.random.uniform(low=-np.pi,high=np.pi))

            best_positions[swarm_index,:]  = np.array(best_particle_position)

        # Matrix to hold best match between swarm i and swarm j
        matches = np.zeros((unique_swarm_numbers.size,unique_swarm_numbers.size))

        for i in range(best_positions.shape[0]):
            for j in range(best_positions.shape[0]):
                # Generate i and j waveforms 
                # wf 1
                source_params_transformed = TaylorF2Ecc_mc_eta_to_m1m2(best_positions[i,:].copy())
                wf_1= self.waveform_func(source_params_transformed,**injection_waveform_args)

                # wf 2
                source_params_transformed = TaylorF2Ecc_mc_eta_to_m1m2(best_positions[j,:].copy())
                wf_2= self.waveform_func(source_params_transformed,**injection_waveform_args)

                # compute coherent phase maximised match between them 
                matches[i,j] = semi_coherent_match(wf_1,wf_2,self.psd_array,self.df,num_segments=1)
    
        print('Matches between swarms: \n')
        print(matches)

        # Set anything less than the match threshold to be 0, ie make them "not connected"
        matches[matches<match_threshold] = 0
        
        # Only combine swarms if there are any similar swarms
        if np.any(matches)>match_threshold:

            # Use graph networks to try to find unique combinations of swarms from the matches
            G = nx.from_numpy_array(matches, create_using=nx.DiGraph)
            H = G.to_undirected()
            combined_swarms = list(nx.find_cliques(H))

            for combined_swarm_index, swarm_indices in enumerate(combined_swarms):

                # Make sure the new swarm has more than one swarm in it 

                if len(swarm_indices)>1:
                    
                    print('Combining swarms: ',str(swarm_indices))
                
                    os.mkdir('Combined_swarm_'+str(combined_swarm_index+1)+'_inference')

                    dataframes_to_be_combined = []

                    # Concatenate the final swarm positions of all the swarms in the combined swarm
                    for swarm_num in swarm_indices: 

                        # Find the final positions of the swarm
                        final_positions = df_subset_final_iteration[Swarm_results_file['swarm_number'] == swarm_num]
                        dataframes_to_be_combined.append(final_positions)

                    # Save the final positions of the swarm
                    combined_df = pd.concat(dataframes_to_be_combined, ignore_index=True, axis=0)
                    combined_df.to_csv('Combined_swarm_'+str(combined_swarm_index+1)+'_inference/final_positions.csv')



    def final_match_against_truth(self,source_params,crashed=False):
        '''
        Compute the best match for each swarm against waveform generation by given set of params:

        Args:
            source_params (array): The source parameters to be used for the match computation. 
            crashed (bool, optional): A flag indicating whether the search crashed/timed out, if so use the final iteration -1 as the
            real final iteration, as this will be the last iteration that is complete in the search data file. Defaults to False.       
        '''
        # Inject noiseless source with source params  
        # Transform input source parameters to those expected in TaylorF2Ecc (mc,eta)->(m1,m2) + polarization shift
        source_parameters_transformed = TaylorF2Ecc_mc_eta_to_m1m2(source_params.copy())
        
        injection_waveform_args = self.waveform_args.copy()
        
        # Generate noiseless signal to compute matches against
        injection= self.waveform_func(source_parameters_transformed,**injection_waveform_args)

        # Directory where all the PySO results are dumped
        Swarm_results_file =  pd.read_csv(self.PySO_kwargs['Output']+'/EnsembleEvolutionHistory.dat')

        # Find interation number of final iteration
        final_iteration = np.sort(np.unique(Swarm_results_file['IterationNumber']))[-1]

        if crashed == True:
            final_iteration = final_iteration - 1

        # Find final state
        df_subset_final_iteration = Swarm_results_file[Swarm_results_file['IterationNumber'] == final_iteration]

        # Unique swarms in the final state of the search
        unique_swarm_numbers = np.unique(df_subset_final_iteration['swarm_number'])
        
        # List used for results html file
        self.max_matches_per_swarm = []

        print('---Maximum fitness values for each swarm---')
        for swarm_num in unique_swarm_numbers: 

            # Find the final positions of the swarm
            final_positions = df_subset_final_iteration[Swarm_results_file['swarm_number'] == swarm_num]

            # Extract final positions of the swarm into numpy array, [:,3,-3] filters down just to the parameter space locations
            parameter_space_positions = final_positions.to_numpy()[:,2:-3]

            Semi_coherent_matches = []
            
            # For each location compute the coherent and N=1 coherent phase maximised overlap against the source parameter waveform. 
            for parameter_space_position in parameter_space_positions:

                parameter_space_position = list(parameter_space_position.copy())

                # Add in distance fixed here so we can generate the waveform
                #   Note this does not affect overlap at all, this is just a praciticality to generate the waveform. 
                parameter_space_position.insert(2,100.e+6)

                # Add in orbital phase fixed so we can generate the waveform
                #   Note the value we pick here will not affect the phase maximised overlap as it is one segment and phase maximised. 
                parameter_space_position.insert(7,np.random.uniform(low=-np.pi,high=np.pi))
                
                # Transform input source parameters to those expected in TaylorF2Ecc (mc,eta)->(m1,m2) + polarization shift
                source_parameters_transformed = TaylorF2Ecc_mc_eta_to_m1m2(parameter_space_position.copy())
                
                # Generate noiseless signal
                signal= self.waveform_func(source_parameters_transformed,**injection_waveform_args)
    
                # Check coherent-phase maximised overlap with the injection
                Semi_coherent_matches.append(semi_coherent_match(signal,injection,self.psd_array,self.df,num_segments=1))

            
            self.max_matches_per_swarm.append(np.max(Semi_coherent_matches))

            print('Swarm: ',swarm_num+1)
            print('Max Semi-Coherent match: ',np.max(Semi_coherent_matches))

    def search_results_to_html(self):
        '''
        Convert the results of the search to a HTML file for viewing. 
        '''
        PSO_stages = np.arange(len(self.segment_ladder))

        N = self.segment_ladder

        Omegas = self.PySO_kwargs['Omega']

        Phip = self.PySO_kwargs['Phip']

        Phig = self.PySO_kwargs['Phig']

        Swarm_results_file =  pd.read_csv(self.PySO_kwargs['Output']+'/EnsembleEvolutionHistory.dat')

        max_upsilons = []

        for stage in PSO_stages:
            # Filter down to this df 
            df = Swarm_results_file[Swarm_results_file['HierarchicalModelNumber']==stage]

            unique_swarm_numbers = np.unique(df['swarm_number'])
            
            upsilons_stage = []
            # For each swarm extract the best upsilon value
            for swarm in unique_swarm_numbers:
                
                # Filter down to this swarm
                df_swarm = df[df['swarm_number']==swarm]

                upsilons_stage.append(str(np.round(np.max(df_swarm['function_value']),1)))

            max_upsilons.append(', '.join(upsilons_stage))

        # Create a dataframe to store the results
        df = pd.DataFrame({'Stage':PSO_stages,'N':N,'Omega':Omegas,'Phip':Phip,'Phig':Phig,'Max_Upsilons':max_upsilons})

        main_html_string = df.to_html()
        
        # Add on another table for minimum velocities 
        minimum_velocities = np.array(self.PySO_kwargs['Minimum_velocities'])

        min_v_df = pd.DataFrame({'Stage':PSO_stages,'N':N,
                                'Mc':minimum_velocities[:,0],
                                'eta':minimum_velocities[:,1],
                                'beta': minimum_velocities[:,2],
                                'lambda': minimum_velocities[:,3],
                                'i': minimum_velocities[:,4],
                                'psi': minimum_velocities[:,5],
                                'f_low': minimum_velocities[:,6],
                                'e0': minimum_velocities[:,7]})
        
        min_v_html_string = min_v_df.to_html()

        # If we have computed matches for the results against some truth waveform, add these onto the results 
        try:
            # Check if it exists, ie if we are computing the match at the end result against some truth 
            self.max_matches_per_swarm
            # Convert each match to string and round to 2 decimal places
            self.max_matches_per_swarm = [str(np.round(match,4)) for match in self.max_matches_per_swarm]
            matches_string = '<br> <h2>Best matches against injection: </h2> <br>'+', '.join(self.max_matches_per_swarm)
        except NameError:
            # If it does not exist ie we are not computing the match at the end result against some truth, just set to empty string
            matches_string = ''

        html_string = '<h1>Search results</h1> <br>'+main_html_string + '<br> <h1>Minimum Velocities</h1>'+ min_v_html_string + matches_string

        # Dump html string to file
        with open(self.PySO_kwargs['Output']+'/search_results.html','x') as f:
            f.write(html_string)

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

class Post_Search_Inference_Dynesty:
    '''
    Class to perform inference on the results of the search.
        Very simple class that uses the results of the search to generate some uniform prior bounds

    Harcoded to the N-1, phase maximised coherent log likelihood. 

    '''        
    def __init__(self, 
                 frequency_series_dict, 
                 data_file_name,
                 swarm_directory,
                 distance_prior_bounds,
                 nlive=1000):
        '''
        Initializes a new instance of the Post Search Inference class using dynesty for the inference

        Parameters:
            frequency_series_dict (dict): A dictionary containing frequency series data. Also contains information about the LISA mission such as
                time of observation etc. 
            prior_bounds (list): A list of prior bounds for the inference
            data_file_name (str): The name of the file containing the data.
            swarm_directory (str): The directory containing the results of the search for the swarm to be inferred over.
            distance_prior_bounds (array): [min,max] bounds for the distance prior.
            coherent_or_N_1 (str, optional): A flag indicating whether to perform coherent or N-1 PE. Defaults to 'N_1'.
            Spread_multiplier (float, optional): A multiplier for the spread of the initial positions for the MCMC. Defaults to None.
                Role is to make the particles in the swarm spread out a bit more before inference.
            nlive (int, optional): Number of live points for dynesty. Defaults to 1000.
        '''

        self.frequency_series_dict = frequency_series_dict

        # Load in data
        self.data = cp.asarray(np.load(data_file_name))

        self.swarm_directory = swarm_directory

        #  Generate priors from the search results
        # Load positions from final iteration of the search, use these to compute priors. 
            # Note this does not include distances!!! Since the search statistic does not search over that
        self.initial_positions = pd.read_csv(self.swarm_directory +'/final_positions.csv').to_numpy()[:,3:-3]

        # Generate priors for all params from end of search (-Distance,initial orbital phase)
        self.priors = np.zeros((8,2))
        for i in range(8):
            self.priors[i,0] = np.min(self.initial_positions[:,i])
            self.priors[i,1] = np.max(self.initial_positions[:,i])
        
        # Insert distance prior bounds manually
        self.priors = np.insert(self.priors,2,distance_prior_bounds,axis=0)

        # Insert physical initial orbital phase prior bounds manually
        self.priors = np.insert(self.priors,7,np.array([-np.pi,np.pi]),axis=0)

        # Reset priors for inclination + polarization to be the whole physical range
        self.priors[5] = np.array([0,np.pi])# inc
        self.priors[6] = np.array([-np.pi,0])# pol (not sure about the limits here)

        print('Final prior setup for sampling: \n')
        print(self.priors)

        self.nlive = nlive
        
        self.inference_object = dynesty_inference(self.frequency_series_dict,
                                                  self.priors,self.nlive,load_data_file=True,data_file_name=data_file_name)
        
    def run_sampler(self,):
        '''
        Run the dynesty sampler
        '''
        sampler_results = self.inference_object.run_sampler()
        equally_weighted_samples = self.inference_object.resample_and_save(sampler_results)


class Post_Search_Inference_Zeus:
    '''
    Class to perform inference on the results of the search.
        Each swarm from the search is loaded in and inference is performed on it using Zeus.

    Harcoded to the N-1, phase maximised coherent log likelihood. 

    '''        
    def __init__(self, 
                 frequency_series_dict, 
                 prior_bounds, 
                 data_file_name,
                 swarm_directory,
                 redraw_eta = False,
                 number_of_walkers=100,
                 num_steps=1000,
                 Zeus_kwargs= {},
                 coherent_or_N_1='N_1',
                 Spread_multiplier=None,
                 terminate_on_max_iter_or_IAT = 'max_iter'):
        '''
        Initializes a new instance of the Post Search Inference class.

        Parameters:
            frequency_series_dict (dict): A dictionary containing frequency series data. Also contains information about the LISA mission such as
                time of observation etc. 
            prior_bounds (list): A list of prior bounds for the inference
            data_file_name (str): The name of the file containing the data.
            swarm_directory (str): The directory containing the results of the search for the swarm to be inferred over.
            redraw_eta (bool, optional): A flag indicating whether to redraw the eta parameter from the prior. Defaults to False.
            number_of_walkers (int): The number of walkers to be used in the MCMC. This is the number of particles from the end of the swarm used. 
                The best particles are selected based on the upsilon value and used in the MCMC. 
            num_steps (int): The number of steps to run the MCMC for.
            Zeus_kwargs (dict): A dictionary containing Zeus keyword arguments.
            coherent_or_N_1 (str, optional): A flag indicating whether to perform coherent or N-1 PE. Defaults to 'N_1'.
            Spread_multiplier (float, optional): A multiplier for the spread of the initial positions for the MCMC. Defaults to None.
                Role is to make the particles in the swarm spread out a bit more before inference. 
            terminate_on_max_iter_or_IAT (str, optional): A flag indicating whether to terminate the MCMC 
                on the maximum number of iterations or when the integrated autocorrelation time passes the default 10 (zeus internal).
                Defaults to 'max_iter', can also be 'IAT'.
        '''

        self.frequency_series_dict = frequency_series_dict
        
        self.prior_bounds = prior_bounds

        self.num_steps = num_steps

        self.zeus_kwargs = Zeus_kwargs

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
        swarm_final_positions = pd.read_csv(self.swarm_directory +'/final_positions.csv').to_numpy()
        final_upsilon_values = swarm_final_positions[:,-3]
        self.initial_positions = swarm_final_positions[np.argsort(final_upsilon_values)[-number_of_walkers:],3:-3]


        if Spread_multiplier != None:
            # Increase the spread of the initial positions from the means
            self.increase_initial_position_spread(Spread_multiplier)

        # Draw distances from prior and insert into initial positions
        self.draw_distances_from_prior()

        if redraw_eta == True:
            # Redraw eta from prior
            self.draw_redraw_eta()


        if coherent_or_N_1 == 'Coherent':

            # Draw initial orbital phases for the coherent PE if requested
            self.draw_initial_orbital_phases()

        # If not coherent, ie N_1 no need to generate initial orbital phases as we do a phase maximisation anyway 
        
        self.terminate_on_max_iter_or_IAT = terminate_on_max_iter_or_IAT

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

    def draw_redraw_eta(self,):
        '''
        Redraws eta from the prior and fills it into the initial guesses for the inference. 
            As the search does not search over eta, this is a necessary step. 
        '''
        eta_draws = np.random.uniform(self.prior_bounds[1][0],self.prior_bounds[1][1],size=(self.initial_positions.shape[0]))

        # Overwrite eta parameter for initial positions
        self.initial_positions[:,1] = eta_draws 

    def initialize_and_run_inference_N_1(self,):
        '''
        Initializes and runs the inference on the results of the search
        '''
        Semi_Coherent_model = Semi_Coherent_Model_Inference(
                                                            self.prior_bounds,
                                                            self.data,
                                                            self.psd_array,
                                                            self.df,
                                                            self.waveform_func,
                                                            segment_number = 1,
                                                            waveform_args=self.waveform_args)
        
        nwalkers = self.initial_positions.shape[0]
        ndim = self.initial_positions.shape[1]

        start = self.initial_positions

        sampler = zeus.EnsembleSampler(nwalkers, 
                                       ndim, 
                                       Semi_Coherent_model.log_likelihood,**self.zeus_kwargs)
        

        if self.terminate_on_max_iter_or_IAT == 'max_iter':
            sampler.run_mcmc(start,self.num_steps)
        elif self.terminate_on_max_iter_or_IAT == 'IAT':
            # Set a max of 200,000 steps for the IAT to reach 10
            sampler.run_mcmc(start,200000,callbacks=[zeus.callbacks.AutocorrelationCallback()])

        chain = sampler.get_chain(flat=True)
        logl = sampler.get_log_prob(flat=True)

        print('Max logl:',np.max(logl))

        # Save samples
        np.savetxt(self.swarm_directory+'/posterior_samples.dat',chain)
        np.savetxt(self.swarm_directory+'/logl.dat',logl)



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
        
        nwalkers = self.initial_positions.shape[0]
        ndim = self.initial_positions.shape[1]

        start = self.initial_positions

        sampler = zeus.EnsembleSampler(nwalkers, 
                                       ndim, 
                                       Coherent_phase_maximised_inference_model.log_likelihood,**self.zeus_kwargs)
   
        if self.terminate_on_max_iter_or_IAT == 'max_iter':
            sampler.run_mcmc(start,self.num_steps)
        elif self.terminate_on_max_iter_or_IAT == 'IAT':
            # Set a max of 200,000 steps for the IAT to reach 10
            sampler.run_mcmc(start,200000,callbacks=[zeus.callbacks.AutocorrelationCallback()])

        chain = sampler.get_chain(flat=True)
        logl = sampler.get_log_prob(flat=True)

        print('Max logl:',np.max(logl))

        # Save samples
        np.savetxt(self.swarm_directory+'/posterior_samples.dat',chain)
        np.savetxt(self.swarm_directory+'/logl.dat',logl)

        