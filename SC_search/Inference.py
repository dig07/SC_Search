try: 
    import cupy as cp 
except ImportError:
    print('Cupy not installed, Inference (on full FFT grid) wont work')
import numpy as np 

import matplotlib.pyplot as plt

from .Swarm_class import Semi_Coherent_Model
from .Utility import TaylorF2Ecc_mc_eta_to_m1m2
from .Semi_Coherent_Functions import vanilla_log_likelihood,semi_coherent_logl,noise_weighted_inner_product
from .Noise import *
from .Waveforms import TaylorF2Ecc, TaylorF2EccSpin

import dynesty
from dynesty import utils as dyfunc

class dynesty_inference():
    '''
    Performs inference on the vanilla likelihood 
    - Harcoded to all the same thing the Search is for now
    '''

    def __init__(self, 
                    frequency_series_dict, 
                    prior_bounds,
                    nlive,
                    source_parameters=None,
                    segment = None,
                    load_data_file=True,
                    data_file_name=None, 
                    include_noise=False,
                    include_spin=False):
        '''
        Initialize the Inference object.

        Args:
            frequency_series_dict (dict): A dictionary containing frequency series data.
            prior_bounds (numpy.ndarray): An array of prior bounds.
            source_parameters (numpy.ndarray): An array of source parameters (Defaults to None, only needed if not loading in datafile).
            nlive (int): The number of live points for the nested sampling algorithm.
            segment (int): Semi-coherent segment number for the log likelihood, if None then use coherent likelihood. 
            load_data_file (bool): Whether to load a data file or generate injection data.
            data_file_name (str): The name of the data file to load.
            include_noise (bool): Whether to include noise in the generated injection data.
            include_spin (bool): Whether to include spin in the parameter estimation. 
        '''

        self.frequency_series_dict = frequency_series_dict

        self.source_parameters = source_parameters
                
        self.prior_bounds = prior_bounds
        self.prior_widths = np.ptp(self.prior_bounds,axis=1)
        
        self.nlive = nlive

        # Generate CPU and GPU frequency grids
        self.generate_frequency_grids()

        # Generate PSD
        self.generate_psd()

        # TODO: Change the function being injected to the direct FFT grid (no interpolation) one just to be rigorous 
        # Search is being tuned for these so hardcoded for now
        if include_spin == False:
            self.waveform_func = TaylorF2Ecc.BBHx_response_interpolate
        else: 
            self.waveform_func = TaylorF2EccSpin.BBHx_response_interpolate

        # Waveform arguments (same between both spin-aligned and no spin waveforms)
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
        else:
            # Generate injection data
            self.generate_injection_data(include_noise)

        # Set the likelihood function
        if segment==None:
            #If no segment specified use the standard likelihood
            self.likelihood = self.standard_likelihood
        if segment!=None:
            #If segment specified use the semi-coherent likelihood
            self.likelihood = self.semi_coherent_likelihood
            # Compute the inner product of the data with itself for the likelihood (computed once and stored)
            self.d_inner_d = noise_weighted_inner_product(self.data, self.data, self.df, self.psd_array, phase_maximize=False).item()
            self.segment_number = segment

        if source_parameters is not None:
            # Check injection values for the injection 
            print('Log likelihood at injection: ',self.likelihood(self.source_parameters.copy()))

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
        
        # Transform input source parameters to those expected in TaylorF2Ecc (mc,eta)->(m1,m2) + polarization shift
        source_parameters_transformed = TaylorF2Ecc_mc_eta_to_m1m2(self.source_parameters.copy())
        
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
            noise = self.generate_noise_realisation()
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

    def standard_likelihood(self,parameters):
        '''
        The likelihood function for the search. 

        Args:
            parameters (array): The parameters for the model. 

        Returns:
            float: The log likelihood. 
        '''
        params_transformed = TaylorF2Ecc_mc_eta_to_m1m2(parameters)

        # Generate model
        model = self.waveform_func(params_transformed,**self.waveform_args)

        # Compute likelihood
        logl = vanilla_log_likelihood(model,self.data,self.df,self.psd_array)

        return logl
    
    def semi_coherent_likelihood(self,parameters):
        '''
        Semi-coherent likelihood function. 
        
        Args:
            parameters (array): The parameters for the model. 

        Returns:
            float: The log likelihood. 
        
        '''
        params_transformed = TaylorF2Ecc_mc_eta_to_m1m2(parameters)


        # Generate model
        model = self.waveform_func(params_transformed,**self.waveform_args)

        # Compute likelihood
        logl = semi_coherent_logl(model,self.data,self.psd_array,self.df,self.d_inner_d,num_segments=self.segment_number)

        return logl
    
    def prior_transform(self,u):
        '''
        The prior transform for dynesty from the unit cube to the parameter space.

        Args:
            u (array): The unit cube parameters. 

        Returns:
            array: The transformed parameters. 
        '''
        x = u*self.prior_widths + self.prior_bounds[:,0]
        return(x)
    
    def run_sampler(self):
        '''
        Runs the search using dynesty.
        '''
        # Set up the sampler
        sampler = dynesty.NestedSampler(self.likelihood, self.prior_transform, len(self.prior_bounds),nlive=self.nlive)
        sampler.run_nested()
        results = sampler.results

        return results
    
    def resample_and_save(self,results):
        '''
        Resamples the inference run to equal weighted samples and saves the results to a file.
        
        Args:
            results (dynesty.results): The results from the inference run. 
        
        Returns:
            array: The resampled samples.
        '''
        # Extract sampling results.
        samples = results.samples  # samples
        weights = np.exp(results.logwt - results.logz[-1])  # normalized weights


        # Resample weighted samples.
        samples_equal = dyfunc.resample_equal(samples, weights)

        # Save samples
        np.savetxt('samples.txt',samples_equal)

        logls = []
        for sample in samples_equal:
            logls.append(self.likelihood(sample.copy()))
        
        # Save log likelihoods
        np.savetxt('logls.txt',logls)
        
        print('Maximum log likelihood: ',np.max(logls))

        return(samples_equal)