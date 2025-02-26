try: 
    import zeus
except ImportError:
    print('Zeus not installed')

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os

from .Swarm_class import Semi_Coherent_Model
import PySO
from scipy.interpolate import CubicSpline


from SmBBHTF.waveforms.time_frequency import TaylorF2EccTF


class Search:
    def __init__(self, 
                 time_frequency_series_dict, 
                 segment_ladder, 
                 prior_bounds,
                 PySO_num_swarms,
                 PySO_num_particles,  
                 PySO_kwargs, 
                 data_file_name = 'data.npy',
                 use_GPU = True,
                 fresnel_kernel_width=5,
                 include_spin = False):
        '''
        Initializes a new instance of the Search class.

        Parameters:
            time_frequency_series_dict (dict): A dictionary containing time-frequency series data. Also contains information about the LISA mission such as
                time of observation etc. 
            source_parameters (list): A dictionary containing source parameters for the true injection. Should be a nested list. Every item in this list
                is a new source. 
            segment_ladder (list): A list of segment ladder values for the semi-coherent search.
            prior_bounds (list): A list of prior bounds for the search
            PySO_num_particles (int): The number of particles to be used in the PySO search.
            PySO_num_swarms (int): The initial number of swarms to be used in the PySO search.
            PySO_kwargs (dict): A dictionary containing PySO keyword arguments.
            data_file_name (str, optional): The name of the file containing the data to be searched over.
            noise_only_injection (bool, optional): A flag indicating whether to inject noise only. Defaults to False.  
            include_spin (bool, optional): A flag indicating whether to include spin in the search (Wether waveform contains the 1.5PN spin compoent). Defaults to False.   '''

        self.frequency_series_dict = time_frequency_series_dict

        self.segment_ladder = segment_ladder
        
        self.prior_bounds = prior_bounds
        
        self.PySO_num_particles = PySO_num_particles

        self.PySO_num_swarms = PySO_num_swarms

        self.PySO_kwargs = PySO_kwargs

        # Generate CPU and GPU frequency grids
        self.generate_tf_grid()

        # Generate PSD (For now just read in the spline and evaluate it)
        noise_arr = np.load("sangria_psd_info.npy")
        psd = CubicSpline(noise_arr[0], noise_arr[1:], axis=1)(self.f_seg)
        self.psd_arr = np.tile(psd[:,None,:], (1, self.nT, 1))

        self.data = np.load(data_file_name)

        # Setup waveform function 
        self.waveform_generator = TaylorF2EccTF(
                                                self.nT,
                                                self.dT,
                                                self.fmax,
                                                self.nF,
                                                self.dF,
                                                self.dt,
                                                compute_TDI=True,
                                                use_gpu=use_GPU,
                                                data = self.data,
                                                psd=self.psd_arr,
                                                use_fresnel_kernel=True,
                                                fresnel_kernel_width=fresnel_kernel_width)


    def generate_tf_grid(self,):
        '''
        Generates the time-frequency grid over which the search is performed.s
        '''

        # Initialising values for frequency grid
        self.fmin = self.frequency_series_dict['fmin']
        self.fmax = self.frequency_series_dict['fmax']
        self.T_obs = self.frequency_series_dict['T_obs']
        # cadence 
        self.dt = self.frequency_series_dict['dt']


        # Length of one tf segment 
        self.dT = self.frequency_series_dict['dT']

        # Frequency spacing
        self.dF = 1/self.dT 

        # Number of frequency bins 
        self.nF = int((self.fmax-self.fmin)/self.dF) + 1 

        # Number of time bins
        self.nT = int(self.T_obs/self.dT)
        
        # Time and frequency segments
        self.f_seg = np.arange(1,self.nF+1)*self.dF
        self.t_seg = np.arange(self.nT)*self.dT
    
    def initialize_and_run_search(self,):
        """
        Initializes the hierarchical search, via the PySO package, for the semi-coherent search.
        """
        # Initialise classes at each segment for the semi-coherent search for
        # the semi-coherent search

        self.Semi_Coherent_classes = [Semi_Coherent_Model(segment_number,
                                                            self.prior_bounds,
                                                            self.data,
                                                            self.waveform_generator)
                                                            for segment_index,segment_number in enumerate(self.segment_ladder)]
        
        PySO_search = PySO.HierarchicalSwarmHandler(self.Semi_Coherent_classes,
                                self.PySO_num_swarms,# Number of initial swarms
                                self.PySO_num_particles,# Number of particles
                                **self.PySO_kwargs)

        PySO_search.Run()



# class Post_Search_Inference_Zeus:
#     '''
#     Class to perform inference on the results of the search.
#         Each swarm from the search is loaded in and inference is performed on it using Zeus.

#     Harcoded to the N-1, phase maximised coherent log likelihood. 

#     '''        
#     def __init__(self, 
#                  frequency_series_dict, 
#                  prior_bounds, 
#                  data_file_name,
#                  swarm_directory,
#                  redraw_eta = False,
#                  number_of_walkers=100,
#                  num_steps=1000,
#                  Zeus_kwargs= {},
#                  coherent_or_N_1='N_1',
#                  Spread_multiplier=None,
#                  terminate_on_max_iter_or_IAT = 'max_iter',
#                  include_spin = False,
#                  confusion = False,
#                  LDC_PSD = False,
#                  LDC_PSD_TDI_version = 1,
#                  response_TDI_version = 1):
#         '''
#         Initializes a new instance of the Post Search Inference class.

#         Parameters:
#             frequency_series_dict (dict): A dictionary containing frequency series data. Also contains information about the LISA mission such as
#                 time of observation etc. 
#             prior_bounds (list): A list of prior bounds for the inference
#             data_file_name (str): The name of the file containing the data.
#             swarm_directory (str): The directory containing the results of the search for the swarm to be inferred over.
#             redraw_eta (bool, optional): A flag indicating whether to redraw the eta parameter from the prior. Defaults to False.
#             number_of_walkers (int): The number of walkers to be used in the MCMC. This is the number of particles from the end of the swarm used. 
#                 The best particles are selected based on the upsilon value and used in the MCMC. 
#             num_steps (int): The number of steps to run the MCMC for.
#             Zeus_kwargs (dict): A dictionary containing Zeus keyword arguments.
#             coherent_or_N_1 (str, optional): A flag indicating whether to perform coherent or N-1 PE. Defaults to 'N_1'.
#             Spread_multiplier (float, optional): A multiplier for the spread of the initial positions for the MCMC. Defaults to None.
#                 Role is to make the particles in the swarm spread out a bit more before inference. 
#             terminate_on_max_iter_or_IAT (str, optional): A flag indicating whether to terminate the MCMC 
#                 on the maximum number of iterations or when the integrated autocorrelation time passes the default 10 (zeus internal).
#                 Defaults to 'max_iter', can also be 'IAT'.
#             include_spin (bool, optional): A flag indicating whether to include spin parameters in the inference. Defaults to False.
#             confusion (bool, optional): A flag indicating whether to include confusion noise in the search for the PSD . Defaults to False.
#             LDC_PSD (bool, optional): A flag indicating whether to use the LDC PSD. Defaults to False.
#             LDC_PSD_TDI_version (int, optional): Wether to use the LDC TDI-1 PSD or TDI-2 PSD.
#             response_TDI_version (int, optional): If the response should use TDI-1 or TDI-2 (using an approximation to go from TDI-1 to TDI-2). Defaults to 1.
#         '''

#         self.frequency_series_dict = frequency_series_dict
        
#         self.prior_bounds = prior_bounds

#         self.num_steps = num_steps

#         self.zeus_kwargs = Zeus_kwargs

#         # Generate CPU and GPU frequency grids
#         self.generate_frequency_grids()

#         # Generate PSD
#         self.generate_psd(confusion=confusion,LDC=LDC_PSD,LDC_PSD_TDI_version=LDC_PSD_TDI_version)

#         # Search is being tuned for these so hardcoded for now
#         if include_spin == True:
#             self.waveform_func = TaylorF2EccSpin.BBHx_response_interpolate
#             self.Ndim = 12 # 12D parameter space (TaylorF2+e0+chi1+chi2)
#             self.spin_waveform = True
#         else:
#             self.waveform_func = TaylorF2Ecc.BBHx_response_interpolate
#             self.Ndim = 10 # 10D parameter space (TaylorF2+e0)
#             self.spin_waveform = False

#         self.waveform_args = {'freqs_sparse':self.freqs_sparse,
#                               'freqs_dense':self.freqs,
#                               'freqs_sparse_on_CPU':self.freqs_sparse_on_CPU,
#                               'f_high':self.fmax,
#                               'T_obs':self.T_obs,
#                               'TDIType':'AET',
#                               'logging': False,
#                               'TDIversion':response_TDI_version}

#         # Load in data
#         self.data = cp.asarray(np.load(data_file_name))
    
#         self.swarm_directory = swarm_directory
#         # Load positions from final iteration of the search for one swarm   
#             # Note this does not include distances!!! Since the search statistic does not search over that
#         swarm_final_positions = pd.read_csv(self.swarm_directory +'/final_positions.csv').to_numpy()
#         final_upsilon_values = swarm_final_positions[:,-3]
#         self.initial_positions = swarm_final_positions[np.argsort(final_upsilon_values)[-number_of_walkers:],3:-3]


#         if Spread_multiplier != None:
#             # Increase the spread of the initial positions from the means
#             self.increase_initial_position_spread(Spread_multiplier)

#         # Draw distances from prior and insert into initial positions
#         self.draw_distances_from_prior()

#         if redraw_eta == True:
#             # Redraw eta from prior
#             self.draw_redraw_eta()


#         if coherent_or_N_1 == 'Coherent':

#             # Draw initial orbital phases for the coherent PE if requested
#             self.draw_initial_orbital_phases()

#         # If not coherent, ie N_1 no need to generate initial orbital phases as we do a phase maximisation anyway 
        
#         self.terminate_on_max_iter_or_IAT = terminate_on_max_iter_or_IAT

#     def increase_initial_position_spread(self,Spread_multiplier):
#         '''
#         Multiply the distance of each particle from the mean of the swarm by a factor of the spread multiplier.
#         '''

#         # Mean across whole swarm of positions across each dimension 
#         axis_means = np.mean(self.initial_positions,axis=0)
#         self.initial_positions = axis_means + Spread_multiplier*(self.initial_positions - axis_means)

#     def generate_frequency_grids(self,):
#         '''
#         Generates the dense and sparse frequency grids for search. 
#         If provided data file and frequency series is pregenerated, load in the frequencies.

#         Stores both on CPU and GPU.         
#         '''

#         # Initialising values for frequency grid
#         self.fmin = self.frequency_series_dict['fmin']
#         self.fmax = self.frequency_series_dict['fmax']
#         self.T_obs = self.frequency_series_dict['T_obs']

#         # Downsampling factor is used for the sparse frequency grid for interpolation
#         self.downsampling_factor = self.frequency_series_dict['downsampling_factor']
        
#         # If frequencies are already generated and stored in a file, load them in
#         if 'pregenerated_frequencies' in self.frequency_series_dict:
#             if self.frequency_series_dict['pregenerated_frequencies'] == True:
#                 self.freqs = cp.asarray(np.load('../freqs_filtered.npy')) # Assumes in above directory
#                 self.df = cp.diff(self.freqs)[1]
#                 self.fmax = self.freqs[-1].get()

#             else:
#                 self.df = 1/self.T_obs
#                 self.freqs = cp.arange(self.fmin,self.fmax,self.df) # On GPU
#         else:
#                 self.df = 1/self.T_obs
#                 self.freqs = cp.arange(self.fmin,self.fmax,self.df) # On GPU

#         # If not just use the whole frequency grid
#         self.freqs_on_CPU = self.freqs.get() # On CPU

#         self.freqs_sparse = self.freqs[::self.downsampling_factor]  # On GPU

#         self.freqs_sparse_on_CPU = self.freqs_sparse.get() # On CPU (Used to compute A,f,phase on small number of points)    
    
#     def generate_psd(self,LDC=False,confusion=False,LDC_PSD_TDI_version=1):
#         '''
#         Generates the PSD for the search.

#         - Harcoded to Michelson PSD for now 

#         Args:
#             LDC (bool, optional): A flag indicating whether to use the LDC PSD. Defaults to False.
#             confusion (bool, optional): A flag indicating whether to include confusion noise in the search for the PSD . Defaults to False.
#             LDC_PSD_TDI_version (int, optional):  Wether to use the LDC TDI-1 PSD or TDI-2 PSD.
#         '''
#         # Generate the PSD

#         if LDC == True:
#             # Conventions
#             c = const.clight
#             L = 2.5e+9/c # Armlength in seconds
#             prefactor = (2*np.pi*1j*self.freqs*L)
            
#             if LDC_PSD_TDI_version == 1:
#                 tdi2 = False
#             elif LDC_PSD_TDI_version == 2:
#                 tdi2 = True

#             noise = get_noise_model("sangria", self.freqs, wd=0)
#             self.psd_A = noise.psd(self.freqs, option='A', tdi2 = tdi2)*1/np.abs(prefactor)**2
#             self.psd_E = noise.psd(self.freqs, option='E', tdi2 = tdi2)*1/np.abs(prefactor)**2
#             self.psd_T = noise.psd(self.freqs, option='T', tdi2 = tdi2)*1/np.abs(prefactor)**2

#         else:
#             Sdisp = Sdisp_SciRD(self.freqs)
#             Sopt = Sopt_SciRD(self.freqs)
#             self.psd_A = psd_AEX(self.freqs,Sdisp,Sopt)
#             self.psd_E = psd_AEX(self.freqs,Sdisp,Sopt)
#             self.psd_T = psd_TX(self.freqs,Sdisp,Sopt)

#         if confusion == True:
#             # Adding in confusion noise wont work with LDC psd 
#             self.psd_A  = Add_confusion(self.freqs,self.psd_A,self.T_obs)
#             self.psd_E  = Add_confusion(self.freqs,self.psd_E,self.T_obs)
#             self.psd_T  = Add_confusion(self.freqs,self.psd_T,self.T_obs)

#         self.psd_array = cp.array([self.psd_A,self.psd_E,self.psd_T])

#     def draw_distances_from_prior(self,):
#         '''
#         Draws a distances from the prior and fills it into the initial guesses for the inference. 
#             As the search does not search over distance, this is a necessary step. 
#         '''
#         distance_draws = np.random.uniform(self.prior_bounds[2][0],self.prior_bounds[2][1],size=(self.initial_positions.shape[0]))
        
#         # Insert distances into the correct index of the initial positions
#         self.initial_positions = np.insert(self.initial_positions,2,distance_draws,axis=1)

#     def draw_initial_orbital_phases(self,):
#         '''
#         Draws initial orbital phases from a uniform distribution and fills it into the initial guesses for the inference. 
#             As the search does not search over initial orbital phase, this is a necessary step. 

#         Only used for coherent post search PE. 
#         '''
#         initial_orbital_phase_draws = np.random.uniform(self.prior_bounds[7][0],self.prior_bounds[7][1],size=(self.initial_positions.shape[0]))
        
#         # Insert distances into the correct index of the initial positions
#         self.initial_positions = np.insert(self.initial_positions,7,initial_orbital_phase_draws,axis=1) 

#     def draw_redraw_eta(self,):
#         '''
#         Redraws eta from the prior and fills it into the initial guesses for the inference. 
#             As the search does not search over eta, this is a necessary step. 
#         '''
#         eta_draws = np.random.uniform(self.prior_bounds[1][0],self.prior_bounds[1][1],size=(self.initial_positions.shape[0]))

#         # Overwrite eta parameter for initial positions
#         self.initial_positions[:,1] = eta_draws 

#     def initialize_and_run_inference_N_1(self,):
#         '''
#         Initializes and runs the inference on the results of the search
#         '''
#         Semi_Coherent_model = Semi_Coherent_Model_Inference(
#                                                             self.prior_bounds,
#                                                             self.data,
#                                                             self.psd_array,
#                                                             self.df,
#                                                             self.waveform_func,
#                                                             segment_number = 1,
#                                                             waveform_args=self.waveform_args,
#                                                             spin_waveform=self.spin_waveform)
        
#         nwalkers = self.initial_positions.shape[0]
#         ndim = self.initial_positions.shape[1]

#         start = self.initial_positions

#         sampler = zeus.EnsembleSampler(nwalkers, 
#                                        ndim, 
#                                        Semi_Coherent_model.log_likelihood,**self.zeus_kwargs)
        

#         if self.terminate_on_max_iter_or_IAT == 'max_iter':
#             sampler.run_mcmc(start,self.num_steps)
#         elif self.terminate_on_max_iter_or_IAT == 'IAT':
#             # Set a max of 200,000 steps for the IAT to reach 10
#             sampler.run_mcmc(start,200000,callbacks=[zeus.callbacks.AutocorrelationCallback()])

#         chain = sampler.get_chain(flat=True)
#         logl = sampler.get_log_prob(flat=True)

#         print('Max logl:',np.max(logl))

#         # Save samples
#         np.savetxt(self.swarm_directory+'/posterior_samples.dat',chain)
#         np.savetxt(self.swarm_directory+'/logl.dat',logl)



#     def initialize_and_run_inference_Coherent(self,):
#         '''
#         Initializes and runs the inference on the results of the search
#         '''
#         Coherent_phase_maximised_inference_model = Coherent_Model_inference(
#                                                             self.prior_bounds,
#                                                             self.data,
#                                                             self.psd_array,
#                                                             self.df,
#                                                             self.waveform_func,
#                                                             waveform_args=self.waveform_args,
#                                                             spin_waveform=self.spin_waveform)
        
#         nwalkers = self.initial_positions.shape[0]
#         ndim = self.initial_positions.shape[1]

#         start = self.initial_positions

#         sampler = zeus.EnsembleSampler(nwalkers, 
#                                        ndim, 
#                                        Coherent_phase_maximised_inference_model.log_likelihood,**self.zeus_kwargs)
   
#         if self.terminate_on_max_iter_or_IAT == 'max_iter':
#             sampler.run_mcmc(start,self.num_steps)
#         elif self.terminate_on_max_iter_or_IAT == 'IAT':
#             # Set a max of 200,000 steps for the IAT to reach 10
#             sampler.run_mcmc(start,200000,callbacks=[zeus.callbacks.AutocorrelationCallback()])

#         chain = sampler.get_chain(flat=True)
#         logl = sampler.get_log_prob(flat=True)

#         print('Max logl:',np.max(logl))

#         # Save samples
#         np.savetxt(self.swarm_directory+'/posterior_samples.dat',chain)
#         np.savetxt(self.swarm_directory+'/logl.dat',logl)

        