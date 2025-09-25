import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os

from .Noise import *
from .Swarm_class import Coherent_model_inference
import PySO
from scipy.interpolate import CubicSpline

from ldc.lisa.noise import get_noise_model

from SmBBHTF.waveforms.time_frequency import TaylorF2EccTF

from nessai.plot import corner_plot
from nessai.flowsampler import FlowSampler
from nessai.utils import setup_logger

class Inference:
    def __init__(self, 
            time_frequency_series_dict, 
            prior_bounds,
            sampler = 'nessai',
            sampler_kwargs = {},
            data_file_name = 'data.npy',
            use_GPU = True,
            fresnel_kernel_width=5,
            use_estimated_PSD = False,
            PSD_file_path = 'PSD_interpolator.npy',
            generate_noise_realisation = False,
            gap_mask = None,
            outdir = './output/'):
                 
        '''
        Initializes a new instance of the Search class.

        Parameters:
            time_frequency_series_dict (dict): A dictionary containing time-frequency series data. Also contains information about the LISA mission such as
                time of observation etc. 
            prior_bounds (list): A list of prior bounds for the search
            sampler (str, optional):  Name of sampler to be used. Defaults to 'nessai'.
            sampler_kwargs (dict): A dictionary containing the sampler keyword arguments.
            data_file_name (str, optional): The name of the file containing the data to be searched over.
            use_GPU (boolean, optional): Wether to use GPU for the search, defaults to true.
            fresnel_kernel_width (int, optional): Width of fresnel kernel used for summation, defaults to 5
            noise_only_injection (bool, optional): A flag indicating whether to inject noise only. Defaults to False.  
            use_estimated_PSD (str, optional): A flag to let the user load in
                an estimated PSD, the estimated PSD is either assumed to be in
                the format (3,#T,#F) OR an interpolator. #T is the number of
                time points used to estimate the PSD, #F is the number of
                frequencies. We assume no interpolation in time (sorr I have explained this badly)
            PSD_file_path (str, optional): The path to the file containing the
                estimated PSD. Defaults to 'PSD_interpolator.npy'.
            generate_noise_realisation (bool, optional): A flag indicating
                whether to generate a noise realization. Defaults to False.
                NOTE: THIS ASSUMES THE DATA WE ARE LOADING IN IS NOISE FREE !!!!!!
            gap_mask (bool or Arraylike, optional): A flag indicating where the
                data is gapped. When ArrayLike, it is an array mask for the coloumns
                out of nT that are dropped.
            outdir (str, optional): The output directory for the sampler. Defaults to './output/'.
                          
        '''        

        self.frequency_series_dict = time_frequency_series_dict

        self.prior_bounds = prior_bounds
                
        self.data = np.load(data_file_name)

        # Generate CPU and GPU frequency grids
        self.generate_tf_grid()

        self.psd_arr = np.zeros(self.data.shape)

        self.sample_kwargs = sampler_kwargs
        self.outdir = outdir 

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

            # Use a directly estimated PSD
            #       This uses a PSD that is computed over usually a number of week segments, and then interpolates this onto a finer time grid
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
        ## TODO: Should this not be only for the analytic PSD ??? Think about this

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
            if gap_mask is not None:

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

    def initialize_and_run_nessai_inference(self,nlive=100):
        """
        Initializes the inference
        """
        self.inference_class = Coherent_model_inference(self.prior_bounds,
                                                            self.data,
                                                            self.waveform_generator)

        logger = setup_logger(output=self.outdir)

        self.sampler = FlowSampler(self.inference_class,
                                    output=self.outdir,
                                    nlive=nlive,
                                    **self.sampler_kwargs)

        self.sampler.run()