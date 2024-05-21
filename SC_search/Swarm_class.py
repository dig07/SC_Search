import numpy as np
import PySO
from .Semi_Coherent_Functions import upsilon_func, semi_coherent_logl, noise_weighted_inner_product, vanilla_log_likelihood,upsilon_func_masking
from .Utility import TaylorF2Ecc_mc_eta_to_m1m2

class Semi_Coherent_Model(PySO.Model):
    '''
    Model class for *one* semi-coherent segment, to be used by the PySO package. 
    For now this is hardcoded to TaylorF2Ecc waveform. 
    '''

    names = ['Mc',
    'eta',
    # 'D',
    'beta',
    'lambda',
    'inc',#cos(i)
    'polarization',
    # 'Initial orbital phase',
    'f_low',
    'e0']

    def __init__(self,segment_number,priors,data,psd_array,df,waveform_function,
                 constant_initial_orbital_phase= 0, constant_distance=100.e+6, waveform_args=None,masking=False,spin_waveform=False):
        '''
        Args:
            segment_number (int): The segment number of the semi-coherent search.
            priors (dict): The priors bounds for the model. 
            data (array-like): The data. Shape: (3,#FFTgrid).
            psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).
            df (float): Frequency step size (1/Tobs).
            waveform_function (function): The waveform function to be used.
            constant_initial_orbital_phase (float, optional): The constant initial orbital phase. Defaults to 0.
            constant_distance (float, optional): The constant distance. Defaults to 100.e+6.
            waveform_args (dict, optional): The arguments for the waveform function. Defaults to None.
            masking (bool, optional): Whether to use the masked version of the upsilon statistic (Only really good at high N). Defaults to False
            spin_waveform (bool, optional): Whether to use the spin waveform. Defaults to False.
        '''
        self.segment_number = segment_number
        self.bounds = priors
        self.data = data
        self.psd_array = psd_array
        self.df = df 
        self.waveform = waveform_function
        self.waveform_args = waveform_args

        if spin_waveform == True:
            # If using spin waveform also search over the spin parameters
            self.names.append('chi1')
            self.names.append('chi2')

        # We hold  initial orbital phase and distance fixed as distance factors out in the search statistic,
        #    and initial orbital phase is unmeasured due to the semi-coherent phase maximisation 
        self.constant_initial_orbital_phase = constant_initial_orbital_phase
        self.constant_distance = constant_distance

        # Set the objective function to be used 
        if masking == True:
            self.objective_function = upsilon_func_masking
        if masking == False:
            self.objective_function = upsilon_func

    def log_likelihood(self, params):
        '''
        Log likelihood/optimisation function for PySO. Set to the upsilon statistic for the semi-coherent search.
        The fact this is called Log likelihood is an artifact of the way PySO is set up. Can be any 
        quantity to be maximised. 

        Parameter transforms are hardcoded in to:
            - Polarization shift to match Balrog convention
            - Mc,eta->m1,m2

        Args:
            params (dict): Waveform parameters.
        
        Returns:
            float: The log likelihood (Any quantity to be optimised).
        
        '''
        # Convert parameters from dict to list 
        parameters_array = [params[key] for key in list(params.keys())]

        # Add in distance and orbital phase so we can generate the waveform
        parameters_array.insert(2,self.constant_distance)
        parameters_array.insert(7,self.constant_initial_orbital_phase)
        
        parameters_array = TaylorF2Ecc_mc_eta_to_m1m2(parameters_array)
        
        model = self.waveform(parameters_array,**self.waveform_args)

        func_vals = self.objective_function(model,self.data,self.psd_array,self.df,num_segments=self.segment_number)

        return(func_vals)

class Semi_Coherent_Model_Inference(PySO.Model):
    '''
    Model class for *one* semi-coherent segment, to be used by the PySO package for inference 
        at the end of the search . 
        For now this is hardcoded to TaylorF2Ecc waveform. 
    '''

    names = ['Mc',
    'eta',
    'D',
    'beta',
    'lambda',
    'inc',#cos(i)
    'polarization',
    # 'Initial orbital phase',
    'f_low',
    'e0']

    def __init__(self,priors,data,psd_array,df,waveform_function,segment_number=1,
                 constant_initial_orbital_phase= 0, waveform_args=None,spin_waveform=False):
        '''
        Args:
            priors (list): The priors bounds for the inference. 
            data (array-like): The data. Shape: (3,#FFTgrid).
            psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).
            df (float): Frequency step size (1/Tobs).
            waveform_function (function): The waveform function to be used.
            segment_number (int): The segment number of the semi-coherent search. Defaults to 1. 
            constant_initial_orbital_phase (float, optional): The constant initial orbital phase. Defaults to 0.
            waveform_args (dict, optional): The arguments for the waveform function. Defaults to None.
        
        '''
        self.segment_number = segment_number
        self.bounds = np.array(priors)
        self.data = data
        self.psd_array = psd_array
        self.df = df 
        self.waveform = waveform_function
        self.waveform_args = waveform_args

        if spin_waveform == True:
            # If using spin waveform also search over the spin parameters
            self.names.append('chi1')
            self.names.append('chi2')


        # Inner product of data with itself for the inference log likelihood
        self.d_inner_d = noise_weighted_inner_product(self.data, self.data, self.df, self.psd_array, phase_maximize=False).item()

        # Hold the initial orbital phase fixed as it is unmeasured due to the semi-coherent phase maximisation at N=1
        self.constant_initial_orbital_phase = constant_initial_orbital_phase

    def log_likelihood(self, params):
        '''
        Log likelihood/optimisation function for PySO. 
        The fact this is called Log likelihood is an artifact of the way PySO is set up. Can be any 
        quantity to be maximised. 

        Here it is actually the semi-coherent log likelihood (defaulting to N=1)

        Parameter transforms are hardcoded in to:
            - Polarization shift to match Balrog convention
            - Mc,eta->m1,m2

        Args:
            params (dict): Waveform parameters.
        
        Returns:
            float: The log likelihood (Any quantity to be optimised).
        
        '''
        in_bounds = np.all(self.bounds[:,0]<params) and np.all(params<self.bounds[:,1])

        if not in_bounds:
            return -np.inf
        
        else:
            # Convert parameters from dict to list 
            parameters_array = list(params) #[params[key] for key in list(params.keys())]

            # Add in orbital phase fixed so we can generate the waveform
            parameters_array.insert(7,self.constant_initial_orbital_phase)
            
            parameters_array = TaylorF2Ecc_mc_eta_to_m1m2(parameters_array)
            
            model = self.waveform(parameters_array,**self.waveform_args)

            func_vals = semi_coherent_logl(model,self.data,self.psd_array,self.df,self.d_inner_d,num_segments=self.segment_number)

            return(func_vals)
    
class Coherent_Model_inference(PySO.Model):
    '''
    Model class for *one* coherent, to be used by the PySO package for inference 
        at the end of the search . 
        For now this is hardcoded to TaylorF2Ecc waveform. 
    '''

    names = ['Mc',
    'eta',
    'D',
    'beta',
    'lambda',
    'inc',#cos(i)
    'polarization',
    'Initial orbital phase',
    'f_low',
    'e0']

    def __init__(self,priors,data,psd_array,df,waveform_function,waveform_args=None,spin_waveform=False):
        '''
        Args:
            priors (list): The priors bounds for the inference. 
            data (array-like): The data. Shape: (3,#FFTgrid).
            psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).
            df (float): Frequency step size (1/Tobs).
            waveform_function (function): The waveform function to be used.
            waveform_args (dict, optional): The arguments for the waveform function. Defaults to None.
        '''
        self.bounds = np.array(priors)
        self.data = data
        self.psd_array = psd_array
        self.df = df 
        self.waveform = waveform_function
        self.waveform_args = waveform_args

        if spin_waveform == True:
            # If using spin waveform also search over the spin parameters
            self.names.append('chi1')
            self.names.append('chi2')


    def log_likelihood(self, params):
        '''
        Log likelihood/optimisation function for PySO. 
        The fact this is called Log likelihood is an artifact of the way PySO is set up. Can be any 
        quantity to be maximised. 

        Here it is actually the vanilla log likelihood (defaulting to N=1)

        Parameter transforms are hardcoded in to:
            - Polarization shift to match Balrog convention
            - Mc,eta->m1,m2

        Args:
            params (dict): Waveform parameters.
        
        Returns:
            float: The log likelihood (Any quantity to be optimised).
        
        '''
        in_bounds = np.all(self.bounds[:,0]<params) and np.all(params<self.bounds[:,1])

        if not in_bounds:

            return -np.inf

        else:
            # Convert parameters from dict to list 
            parameters_array = list(params) # [params[key] for key in list(params.keys())]

            parameters_array = TaylorF2Ecc_mc_eta_to_m1m2(parameters_array)
            
            model = self.waveform(parameters_array,**self.waveform_args)

            func_vals = vanilla_log_likelihood(model,self.data,self.df,self.psd_array)

        return(func_vals)