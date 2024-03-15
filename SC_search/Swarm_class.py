import numpy as np
import PySO
from .Semi_Coherent_Functions import upsilon_func
from .Utility import TaylorF2Ecc_mc_eta_to_m1m2

class Semi_Coherent_Model(PySO.Model):
    '''
    Model class for *one* semi-coherent segment, to be used by the PySO package. 
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

    def __init__(self,segment_number,priors,data,psd_array,waveform_function,waveform_args=None):
        '''
        Args:
            segment_number (int): The segment number of the semi-coherent search.
            priors (dict): The priors bounds for the model. 
            data (array-like): The data. Shape: (3,#FFTgrid).
            psd_array (array-like): The PSD in each channel. Shape: (3,#FFTgrid).
            waveform_function (function): The waveform function to be used.
            waveform_args (dict, optional): The arguments for the waveform function. Defaults to None.
        
        '''
        self.segment_number = segment_number
        self.bounds = priors
        self.data = data
        self.psd_array = psd_array
        self.waveform = waveform_function
        self.waveform_args = waveform_args

    def log_likelihood(self, params):
        '''
        Log likelihood/optimisation function for PySO. 
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
        # Convert parameters from dict to array 
        parameters_array = np.array([params[key] for key in list(params.keys())])

        parameters_array = TaylorF2Ecc_mc_eta_to_m1m2(parameters_array)
        
        model = self.waveform(parameters_array,**self.waveform_args)

        func_vals = upsilon_func(model,self.data,self.psd_array,num_segments=self.segment_number)

        return(func_vals)