import matplotlib.pyplot as plt

from .Utility import TaylorF2Ecc_mc_eta_to_m1m2
from .Semi_Coherent_Functions import upsilon_func
from .Semi_Coherent_Functions import noise_weighted_inner_product
from .Noise import *
from .Waveforms import TaylorF2Ecc

try: 
    import cupy as np
    import numpy as numpy
    use_GPU = True
except Exception as e:
    print('Cupy not installed')
    import numpy as np
    import numpy as numpy
    use_GPU = False

class Distributions:
    '''
    Used to generate the distributions of the search statistic for the semi-coherent search.
    '''
    def __init__(self, 
                 frequency_series_dict, 
                 source_parameters
                 ):
        '''
        Initializes a new instance of the Distributions class.

        Parameters:
            frequency_series_dict (dict): A dictionary containing frequency series data. Also contains information about the LISA mission such as
                time of observation etc. 
            source_parameters (list): A dictionary containing source parameters for the true injection. Should be a nested list. Every item in this list
                is a new source. 
        '''

        self.frequency_series_dict = frequency_series_dict

        self.source_parameters = source_parameters
        
        # Generate CPU and GPU frequency grids
        self.generate_frequency_grids()

        # Generate PSD
        self.generate_psd()

        # TODO: Change the function being injected to the direct FFT grid (no interpolation) one just to be rigorous 

        # Search is being tuned for these so hardcoded for now

        if use_GPU == False:
            self.waveform_func = TaylorF2Ecc.BBHx_response_interpolate_CPU
            self.waveform_args = {'freqs_sparse':self.freqs_sparse,
                                  'freqs_dense':self.freqs,
                                  'f_high':self.fmax,
                                  'T_obs':self.T_obs,
                                  'TDIType':'AET',
                                  'logging': False}
                                  
        else:
            self.waveform_func = TaylorF2Ecc.BBHx_response_interpolate
            self.waveform_args = {'freqs_sparse':self.freqs_sparse,
                                'freqs_dense':self.freqs,
                                'freqs_sparse_on_CPU':self.freqs_sparse_on_CPU,
                                'f_high':self.fmax,
                                'T_obs':self.T_obs,
                                'TDIType':'AET',
                                'logging': False}
            
        # Generate the noiseless injection model 
        self.generate_injection_model()

        # For plotting the theoretical distribution of the search statistic (obtained from arXiv:1705.04259v2)
        self.mu_k = 2 
        self.sigma_k = np.sqrt(2)

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

        self.freqs = np.arange(self.fmin,self.fmax,self.df) # Agnostic to CPU or GPU
            
        self.freqs_sparse = self.freqs[::self.downsampling_factor] # Agnostic to CPU or GPU


        if use_GPU: 
            # If we are using the GPU, need to get a numpy version of the frequency grids for the CPU
            self.freqs_on_CPU = self.freqs.get() # On CPU
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

        noise_ = np.array([noise_A,noise_E,noise_T]) # Agnostic to CPU or GPU

        return noise_        

    def generate_psd(self,):
        '''
        Generates the PSD for the search.

        - Harcoded to Michelson PSD for now 
        '''

        Sdisp = Sdisp_SciRD(self.freqs)
        Sopt = Sopt_SciRD(self.freqs)
        self.psd_A = psd_AEX(self.freqs,Sdisp,Sopt)
        self.psd_E = psd_AEX(self.freqs,Sdisp,Sopt)
        self.psd_T = psd_TX(self.freqs,Sdisp,Sopt)

        self.psd_array = np.array([self.psd_A,self.psd_E,self.psd_T]) ## Agnostic to CPU or GPU
    
    def generate_injection_model(self):
        '''
        Generates the injection model for the search. 
        '''
        num_sources = len(self.source_parameters)

        print('Data contains '+ str(num_sources)+' sources!')

        self.injection = np.zeros((3,self.freqs.size),dtype=complex)

        for source_index,source in enumerate(self.source_parameters):

            # Transform input source parameters to those expected in TaylorF2Ecc (mc,eta)->(m1,m2) + polarization shift
            source_parameters_transformed = TaylorF2Ecc_mc_eta_to_m1m2(source)
            
            # Turn logging on for the injection waveform so we can debug statements 
            injection_waveform_args = self.waveform_args.copy()
            injection_waveform_args['logging'] = True
            
            # Generate noiseless signal
            signal= self.waveform_func(source_parameters_transformed,**injection_waveform_args)

            # Print SNR of injection signal
            self.injection_SNR = np.sqrt(4*np.real(np.sum((signal*signal.conj()/self.psd_array*self.df)))).item()## Agnostic to CPU or GPU
            print('SNR of signal ',source_index+1,' :',self.injection_SNR)

            self.injection += signal

    def generate_upsilon_statistic_height_plot(self,num_realisations=1000,num_segments=10,ssn_filename='ssn.txt',sn_filename='sn.txt',sn=True):
        '''
        Generates the distribution of the search statistic for the semi-coherent search.
            Generates both <s|n> and <s|s+n>
        
        Args:
            num_realisations (int, optional): Number of noise realisations to generate. Defaults to 1000.
            num_segments (int, optional): Number of segments to use for the semi-coherent approach. Defaults to 10.
            ssn_filename (str, optional): Filename to save the <s|s+n> search statistic values. Defaults to 'ssn.txt'.
            sn_filename (str, optional): Filename to save the <s|n> search statistic values. Defaults to 'sn.txt'.
            sn (bool, optional): If True, generates the <s|n> search statistic. Defaults to True.
        '''
        if use_GPU==False:
            print('Computing Upsilons over the FFT grid on many noise realisations is expensive on CPU. Please be patient.')
        # Lists to store the inner products 
        s_s_n = []
        s_n =  []

        for i in range(num_realisations):
            
            # Generate noise realisation
            noise_ = self.generate_noise_realisation()
            data = self.injection + noise_
            
            if num_segments!= 0:
                # Semi-coherent
                # Generate the upsilon statistic <s|s+n>
                s_s_n.append(upsilon_func(self.injection,data,self.psd_array,self.df,num_segments=num_segments))

                if sn == True:
                    # Generate the upsilon statistic <s|n>
                    s_n.append(upsilon_func(self.injection,noise_,self.psd_array,self.df,num_segments=num_segments))

            else:
                # Coherent
                s_s_n.append(noise_weighted_inner_product(self.injection,data,self.df,self.psd_array,phase_maximize=False)/
                                1/np.sqrt(noise_weighted_inner_product(self.injection,self.injection,self.df,self.psd_array,phase_maximize=False)))
                if sn == True:
                    s_n.append(noise_weighted_inner_product(self.injection,noise_,self.df,self.psd_array,phase_maximize=False)/
                                    1/np.sqrt(noise_weighted_inner_product(self.injection,self.injection,self.df,self.psd_array,phase_maximize=False)))

        s_s_n = np.array(s_s_n)

        # Save the search statistic values to a file
        np.savetxt(ssn_filename,s_s_n)

        if sn == True:
            s_n = np.array(s_n)
            np.savetxt(sn_filename,s_n)

        # Compute theoretical distribution of the search statistic that should agree with this  (See arXiv:1705.04259v2)
        
        # x_range_s_s_n = np.linspace(np.min(s_s_n),np.max(s_s_n),1000)

        # mu_1 = num_segments*self.mu_k+self.injection_SNR**2
        # sigma_1 = np.sqrt(2*num_segments*self.sigma_k**2+4*self.injection_SNR**2)

        # theoretical_s_s_n= scipy.stats.norm.pdf(x_range_s_s_n,loc = mu_1,scale = sigma_1)


        # x_range_s_n = np.linspace(np.min(s_n),np.max(s_n),1000)

        # mu_0 = num_segments*self.mu_k
        # sigma_0 = np.sqrt(2*num_segments*self.sigma_k**2)
        
        # theoretical_s_n= scipy.stats.norm.pdf(x_range_s_n,loc = mu_0,scale = sigma_0)


        # # Plot the distribution of the search statistic
        # plt.figure(figsize=(12,7))
        
        # #  <signal| signal + noise>
        # plt.hist(s_s_n,density=True,histtype='step',color='r',label=r'$\Upsilon(s,s+n)$ N='+str(num_segments),bins=50)
        # plt.plot(x_range_s_s_n,theoretical_s_s_n,linestyle='--',label=r'$\mathcal{N}(N\mu_k+\rho^2,2N\sigma_k^2+4\rho^2)$',lw=2,color='r')
        
        # # <signal| noise>
        # plt.hist(s_n,density=True,histtype='step',color='g',label=r'$\Upsilon(s|n)$ N='+str(num_segments),bins=50)
        # plt.plot(x_range_s_n,theoretical_s_n,linestyle='--',label=r'$\mathcal{N}(N \mu_k,2N\sigma^2_k)$',lw=2,color='g')    
    
        # plt.xlabel(r'$\Upsilon$')
        # plt.yticks([])
        # plt.legend(loc='upper left')
        # plt.show()


