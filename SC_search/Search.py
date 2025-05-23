try: 
    import zeus
except ImportError:
    print('Zeus not installed')

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os

from .Noise import *
from .Swarm_class import Semi_Coherent_Model
import PySO
from scipy.interpolate import CubicSpline

from ldc.lisa.noise import get_noise_model

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
                 include_spin = False,
                 use_estimated_PSD = False,
                 PSD_file_path = 'PSD_interpolator.npy',
                 generate_noise_realisation = False,
                 gap_mask = None,):
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
            include_spin (bool, optional): A flag indicating whether to include spin in the search (Wether waveform contains the 1.5PN spin compoent). Defaults to False.  
            use_estimated_PSD (str, optional): A flag to let the user load in
                an estimated PSD, the estimated PSD is either assumed to be in
                the format (3,#T,#F) OR an interpolator. #T is the number of
                time points used to estimate the PSD, #F is the number of
                frequencies. We assume no interpolation in time (sorr I have explained this badly)
            PSD_file_path (str, optional): The path to the file containing the
                estimated PSD. Defaults to 'PSD_interpolator.npy'.
            generate_noise_realisation (bool, optional): A flag indicating
            whether to generate a noise realization. Defaults to False. NOTE
            THIS ASSUMES THE DATA WE ARE LOADING IN IS NOISE FREE !!!!!!
            gap_mask (bool or Arraylike, optional): A flag indicating where the
            data is gapped. When ArrayLike, it is an array mask for the coloumns
            out of nT that are dropped. 
             '''

        self.frequency_series_dict = time_frequency_series_dict

        self.segment_ladder = segment_ladder
        
        self.prior_bounds = prior_bounds
        
        self.PySO_num_particles = PySO_num_particles

        self.PySO_num_swarms = PySO_num_swarms

        self.PySO_kwargs = PySO_kwargs

        self.data = np.load(data_file_name)

        # Generate CPU and GPU frequency grids
        self.generate_tf_grid()

        self.psd_arr = np.zeros(self.data.shape)


        if use_estimated_PSD == True:
            print('Using estimated PSD...')

            # Load in the PSD object
            psd_object = np.load(PSD_file_path,allow_pickle=True).item()

            if psd_object['Type'] == 'Interpolator':

                # Extract interpolators in three channels 
                interpolant_A,interpolant_E,interpolant_T = psd_object['A'],psd_object['E'],psd_object['T']
                
                # Generate query points as a 2D grid
                T, F = np.meshgrid(self.t_seg, self.f_seg, indexing='ij')
                tf_points = np.column_stack((T.ravel(), F.ravel()))

                # Interpolate
                psd_A = interpolant_A(tf_points).reshape(T.shape)  # Reshape to match the grid shape
                psd_E = interpolant_E(tf_points).reshape(T.shape)  
                psd_T = interpolant_T(tf_points).reshape(T.shape)  

                self.psd_arr = np.array([psd_A,psd_E,psd_T])

            # Use a directly estimated PSD without interpolating 
            elif psd_object['Type'] == 'Constant':

                # Extract PSD in three channels 
                psd_A,psd_E,psd_T = psd_object['A'],psd_object['E'],psd_object['T']
                psd_ = np.array([psd_A,psd_E,psd_T])

                # Extract frequencies times over which this psd is estimated
                time_points= psd_object['Times']
                frequency_points = psd_object['Frequencies']

                # time_points index that each t_seg falls into 
                t_seg_indices_to_match_time_points = np.searchsorted(time_points,self.t_seg) 
                
                self.psd_arr = np.zeros((3,self.t_seg.size,self.f_seg.size))

                for t_index,t in enumerate(self.t_seg):
                    # Which PSD bin should I be extracting 
                    PSD_file_time_index = t_seg_indices_to_match_time_points[t_index]
                    
                    # Edge case, i.e self.t_seg > time_points[-1], just asusme
                    # it remains constant
                    if PSD_file_time_index==psd_.shape[1]:
                        self.psd_arr[:,t_index,:] = np.array([self.interpolate_PSD(frequency_points,psd_[i,-1,:]) for i in range(3)])
                    else:
                        self.psd_arr[:,t_index,:] = np.array([self.interpolate_PSD(frequency_points,psd_[i,PSD_file_time_index,:]) for i in range(3)])

        else:   
            print('Using analytic PSD...')
            noise = get_noise_model("sangria", self.f_seg, wd=self.T_obs/(365.25*24*60*60))
            psd_A = noise.psd(self.f_seg, option='A', tdi2 = True)
            psd_E = noise.psd(self.f_seg, option='E', tdi2 = True)
            psd_T = noise.psd(self.f_seg, option='T', tdi2 = True)

            psd_ = np.array([psd_A,psd_E,psd_T]).reshape(3,self.data.shape[2])


            for i in range(self.nT):
                self.psd_arr[:,i,:] = psd_.copy()
        print('PSD shape vs data shape (sanity check): ',self.psd_arr.shape,self.data.shape)
        # # Generate PSD (For now just read in the spline and evaluate it)
        # noise_arr = np.load("sangria_psd_info.npy")
        # psd = CubicSpline(noise_arr[0], noise_arr[1:], axis=1)(self.f_seg)
        # self.psd_arr = np.tile(psd[:,None,:], (1, self.nT, 1))

        # Temporary bodge to clip out the 0s in the PSD array

        f_seg_clip_start = 0.029
        f_seg_clip_end = 0.031
        f_seg_clip_start_ind = int(np.argmin(np.abs(self.f_seg - f_seg_clip_start)))
        f_seg_clip_end_ind = int(np.argmin(np.abs(self.f_seg - f_seg_clip_end)))

        for stupid_ind in range(f_seg_clip_start_ind, f_seg_clip_end_ind):
            self.psd_arr[:,:,stupid_ind] = self.psd_arr[:,:,f_seg_clip_start_ind]

        f_seg_clip_start = 0.059
        f_seg_clip_end = 0.061
        f_seg_clip_start_ind = int(np.argmin(np.abs(self.f_seg - f_seg_clip_start)))
        f_seg_clip_end_ind = int(np.argmin(np.abs(self.f_seg - f_seg_clip_end)))

        for stupid_ind in range(f_seg_clip_start_ind, f_seg_clip_end_ind):
            self.psd_arr[:,:,stupid_ind] = self.psd_arr[:,:,f_seg_clip_start_ind]
        
        f_seg_clip_start = 0.0897
        f_seg_clip_end = 0.0902
        f_seg_clip_start_ind = int(np.argmin(np.abs(self.f_seg - f_seg_clip_start)))
        f_seg_clip_end_ind = int(np.argmin(np.abs(self.f_seg - f_seg_clip_end)))

        for stupid_ind in range(f_seg_clip_start_ind, f_seg_clip_end_ind):
            self.psd_arr[:,:,stupid_ind] = self.psd_arr[:,:,f_seg_clip_start_ind]    

        print('IS WHOLE PSD POSITIVE: ',np.all(self.psd_arr>0))
        # Generate tf noise realisation if noise is to be indjected 
        if generate_noise_realisation == True:
            psd_to_generate_noise_from = self.psd_arr.copy()
            #if use_estimated_PSD == True:
            #    psd_to_generate_noise_from[:,:,self.f_seg<1.e-3] = 0
            noise_tf = self.generate_noise_realisation(psd_to_generate_noise_from)
            self.data += noise_tf

            # If gaps are present, we need to set the noise to zero in those segments
            if self.gap_mask is not None:

                total_indices = np.arange(self.nT)
                dropped_indices= np.setdiff1d(total_indices,gap_mask)
                self.data[:,dropped_indices,:] = 0.0

        # Bodge for avoiding nans 
        # self.psd_arr[:,:,self.f_seg<1.e-3] = np.inf

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

        # Simulating gaps 
        if gap_mask is not None: 
            self.waveform_generator.apply_segment_mask(gap_mask)

    def generate_noise_realisation(self,psd_to_generate_noise_from):
        '''
        Generates a noise realisation for injecting into data

        Returns:
            noise: Noise realization 
        '''
        # Generate noise in each channel (for each time segment)

        noise = np.zeros((3,self.nT,self.nF),dtype=complex)
    
        for t_index,t in enumerate(self.t_seg):
            # Important thing here is that it is dT not T_obs as that is the size of each segment   
            noise_A = noise_realization(psd_to_generate_noise_from[0,t_index,:],self.dT)
            noise_E = noise_realization(psd_to_generate_noise_from[1,t_index,:],self.dT)
            noise_T = noise_realization(psd_to_generate_noise_from[2,t_index,:],self.dT)

            noise[:,t_index,:] = np.array([noise_A,noise_E,noise_T])

        return noise       


    def interpolate_PSD(self,f_sparse,PSD):
        '''
        Interpolates the PSD over the sparse frequency grid using cubic splines,
        onto the full frequency grid.
        '''
        # Interpolate the PSD over the sparse frequency grid
        psd_interpolator = CubicSpline(f_sparse, PSD)
        psd_dense = psd_interpolator(self.f_seg)
        return psd_dense

    def generate_tf_grid(self,):
        '''
        Generates the time-frequency grid over which the search is performed.s
        '''

        # Initialising values for frequency grid
        self.fmin = self.frequency_series_dict['fmin'] # NOT ACTUALLY TRUE
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

        
