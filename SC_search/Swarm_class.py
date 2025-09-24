import numpy as np
import PySO
from nessai.model import Model

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

    def __init__(self,segment_number,priors,data,waveform_generator,
                 constant_final_orbital_phase= 0, constant_distance=100.e+6):
        '''
        Args:
            segment_number (int): The segment number of the semi-coherent search.
            priors (dict): The priors bounds for the model. 
            data (array-like): The data.
            waveform_generator (function): The waveform generator to be used
            (from SMBBHTF). 
            constant_initial_orbital_phase (float, optional): The constant initial orbital phase. Defaults to 0.
            constant_distance (float, optional): The constant distance. Defaults
            to 100.e+6.
        '''
        self.segment_number = segment_number
        self.bounds = priors
        self.data = data
        self.waveform_generator = waveform_generator
        # D and phi_coal maximised over in the search
        self.names = ['Mc',
                        'q',
                        'cosinc',
                        'e0',
                        #'D',
                        'f0',
                        #'phi_coal',
                        'lam',
                        'beta',
                        'psi']
        


        # We hold  final orbital phase and distance fixed as distance factors out in the search statistic,
        #    and final orbital phase is unmeasured due to the semi-coherent phase maximisation 
        self.constant_final_orbital_phase = constant_final_orbital_phase
        self.constant_distance = constant_distance


    def log_likelihood(self, params):
        '''
        Log likelihood/optimisation function for PySO. Set to the upsilon statistic for the semi-coherent search.
        The fact this is called Log likelihood is an artifact of the way PySO is set up. Can be any 
        quantity to be maximised. 

        Args:
            params (dict): Waveform parameters. Can be arrays for batched evaluations.
        
        Returns:
            loglike (array): The log likelihood (Any quantity to be optimised).
        
        '''

        batchsize = params['Mc'].shape[0]
        
        loglike = self.waveform_generator.get_log_likelihood(
            params['Mc'], 
            params['q'], 
            params['cosinc'], 
            params['e0'], 
            [self.constant_distance]*batchsize, 
            params['f0'], 
            [self.constant_final_orbital_phase]*batchsize, 
            params['lam'],
            params['beta'],
            params['psi'],
            True,
            self.segment_number
        )
        try:
            return loglike.get()
        except AttributeError:
            return loglike
        

class Coherent_model_inference(Model):
    '''
    Coherent standard model inference. 
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

    def __init__(self,priors,data,waveform_generator,likelihood_chunksize = 10000):
        '''
        Args:
            priors (list): The priors bounds for the inference. 
            data (array-like): The data. Shape: (3,#FFTgrid).
            waveform_function (function): The waveform function to be used.
            likelihood_chunksize (int, optional): The chunk size for the likelihood evaluation batching.
        '''
        self.bounds = priors
        self.data = data
        self.waveform_generator = waveform_generator

        self.likelihood_chunksize = likelihood_chunksize
        self._vectorised_likelihood = True

        self.names = ['Mc',
                        'q',
                        'cosinc',
                        'e0',
                        'D',
                        'f0',
                        'phi_coal',
                        'lam',
                        'beta',
                        'psi']

    def log_prior(self, x):
        """Uniform prior"""
        log_p = np.log(self.in_bounds(x), dtype="float")
        for bounds in self.bounds.values():
            log_p -= np.log(bounds[1] - bounds[0])
        return log_p

    def to_unit_hypercube(self, x):
        """Map to the unit hyper-cube"""
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (x[n] - self.bounds[n][0]) / (
                self.bounds[n][1] - self.bounds[n][0]
            )
        return x_out

    def from_unit_hypercube(self, x):
        """Map from the unit hyper-cube"""
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (self.bounds[n][1] - self.bounds[n][0]) * x[
                n
            ] + self.bounds[n][0]
        return x_out


    def log_likelihood(self, params):
        '''
        Log likelihood to be accessed by a sampler

        Args:
            params (dict): Waveform parameters. (arrays)
        
        Returns:
            float (array): The log likelihood 
        
        '''
        loglike = self.waveform_generator.get_log_likelihood(
            params['Mc'], 
            params['q'], 
            params['cosinc'], 
            params['e0'], 
            params['D'], 
            params['f0'], 
            params['phi_coal'], 
            params['lam'],
            params['beta'],
            params['psi'],
            False)
        try:
            return loglike.get()
        except AttributeError:
            return loglike
            