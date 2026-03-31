import numpy as np
import PySO
from math import ceil 



from time import perf_counter
from numba import cuda 

from nessai.model import Model


class Semi_Coherent_Model(PySO.Model):
    """PySO model for a single segment of the semi-coherent search ladder.

    Wraps the waveform generator's log-likelihood (upsilon statistic) so
    that ``PySO.HierarchicalSwarmHandler`` can optimise over it.

    Distance and final orbital phase are held fixed: distance factors out
    of the search statistic, and the final orbital phase is analytically
    maximised over in the semi-coherent framework.

    Parameters
    ----------
    segment_number : int
        Which segment this model represents in the
        semi-coherent ladder.
    priors : dict
        Prior bounds for every search parameter, keyed by name.
    data : ndarray
        The time-frequency data array (shape ``(3, nT, nF)``).
    waveform_generator : object
        Waveform generator instance (e.g. ``TaylorF2EccTF``)
    nT: int
        Number of time bins in the data (used for pre-allocating arrays for the GPU kernel)
    psd: array
        Power spectral density array (shape (nT,nF,3)) for the data.  Used for computing the search statistics.
    total_number_of_particles : int, optional
        Total number of particles to be evaluated in the hierarchical PySO.  Defaults to 100000. 
    batch_size : int, 
        Batch size to use for the GPU-accelerated objective function.  Defaults to 10000.
    constant_final_orbital_phase : float, optional
        Fixed value used for the final orbital phase.  Defaults to 0.
    constant_distance : float, optional
        Fixed luminosity distance in parsecs.  Defaults to 1e8.
    use_GPU : bool, optional
        Whether to use the GPU-accelerated version of the objective function.  Defaults to False

    

    """

    names = [
        "Mc",
        "q",
        "cosinc",
        # "D",
        "f0",
        "s1",
        "s2",
        #phicoal,
        "psi",
        "lam",
        "beta",
    ]

    def __init__(
        self,
        segment_number,
        priors,
        data,
        waveform_generator,
        nT, 
        psd,
        total_number_of_particles = 100000,
        constant_final_orbital_phase=0,
        constant_distance=100.0e6,
        use_GPU = False,
    ):
        self.segment_number = segment_number
        self.bounds = priors
        self.waveform_generator = waveform_generator
        self.constant_final_orbital_phase = constant_final_orbital_phase
        self.constant_distance = constant_distance
        self.nT = nT


        if use_GPU:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        # Move data and psd to the selected backend (NumPy or CuPy) once, so that we don't have to keep transferring them for each batch of particles in the objective function.
        self.data = self.xp.asarray(data)
        self.psd = self.xp.asarray(psd)
        self.results_array = self.xp.zeros((total_number_of_particles,), dtype=np.float64) # Pre-allocate array for results

        self.total_number_of_particles = total_number_of_particles

        self.statistic_array = self.xp.zeros((total_number_of_particles,)) # self.xp.zeros((self.batch_size,), dtype=np.float64) # Pre-allocate array for the search statistic values for each batch of particles

        self._wf_params = self.xp.zeros((total_number_of_particles, 8), dtype=np.float64) 
        self._resp_params = self.xp.zeros((total_number_of_particles, 4), dtype=np.float64)

    def objective_function(self, params):
        """Evaluate the semi-coherent search statistic (upsilon) for a batch of particles.

        Parameters
        ----------
        params : dict of ndarray
            Parameter arrays keyed by name.  Each array has shape
            ``(batch_size,)``.

        Returns
        -------
        statistic : ndarray
            Search statistic value for each particle.
        """

        # t_0 = perf_counter()  
        
        nparticles = params["Mc"].shape[0]

        # Move particle arrays to the selected backend once (NumPy or CuPy).
        Mc = self.xp.asarray(params["Mc"])
        q = self.xp.asarray(params["q"])
        cosinc = self.xp.asarray(params["cosinc"])
        f0 = self.xp.asarray(params["f0"])
        s1 = self.xp.asarray(params["s1"])
        s2 = self.xp.asarray(params["s2"])
        psi = self.xp.asarray(params["psi"])
        lam = self.xp.asarray(params["lam"])
        beta = self.xp.asarray(params["beta"])

        M = Mc * (q / (1 + q)**2)**(-3/5)

        eta = (Mc / M)**(5/3)
        
        # Subset to the number of particles in the current batch.  This way we can keep the GPU kernel's output array allocated to the maximum batch size, and just fill it with the current batch's results for each call to the objective function.
        wf_params = self._wf_params[:nparticles]
        resp_params = self._resp_params[:nparticles]

        wf_params[:, 0] = M
        wf_params[:, 1] = eta
        wf_params[:, 2] = cosinc
        wf_params[:, 3] = self.constant_distance
        wf_params[:, 4] = f0
        wf_params[:, 5] = s1
        wf_params[:, 6] = s2
        wf_params[:, 7] = self.constant_final_orbital_phase

        resp_params[:, 0] = cosinc
        resp_params[:, 1] = psi
        resp_params[:, 2] = lam
        resp_params[:, 3] = beta

        if  self.statistic_array.shape[0] != nparticles:
            print("Re-Allocating output array for the GPU kernel with batch size:", nparticles)
            self.statistic_array = self.xp.zeros((nparticles,), dtype=np.float64) # Allocate array for the search statistic values for each batch of particles if it hasn't been allocated yet or if the batch size has changed

        self.waveform_generator(parameters=wf_params, 
                                        channels=self.data,
                                        psds=self.psd,
                                        parameters_response=resp_params,
                                        out = None,
                                        compute_statistic=True,
                                        search_statistic = self.statistic_array,
                                        N_seg = self.segment_number)
        # # CuPy arrays expose .get(); NumPy arrays do not.
        try:
            return self.statistic_array.get()
        
        except AttributeError:
            return self.statistic_array
    
class Model_inference(Model):
    """Coherent inference model

    Parameters
    ----------
    segment_number : int
        Which segment this model represents in the
        semi-coherent ladder.
    priors : dict
        Prior bounds for every search parameter, keyed by name.
    data : ndarray
        The time-frequency data array (shape ``(nT,nF,3)``).
    waveform_generator : object
        Waveform generator instance (e.g. ``TaylorF2EccTF``)
    nT: int
        Number of time bins in the data (used for pre-allocating arrays for the GPU kernel)
    psd: array
        Power spectral density array (shape (nT,nF,3)) for the data.  Used for computing the search statistics.
    use_GPU : bool, optional
        Whether to use the GPU-accelerated version of the objective function.  Defaults to False
    """

    names = [
        "Mc",
        "q",
        "cosinc",
        "D",
        "f0",
        "s1",
        "s2",
        "phicoal",
        "psi",
        "lam",
        "beta",
    ]

    def __init__(
        self,
        priors,
        data,
        waveform_generator,
        nT, 
        psd,
        dF, 
        use_GPU = False,
        nlive = 1000,
        likelihood_chunksize = 10000,
    ):
        self.bounds = priors
        self.waveform_generator = waveform_generator
        self.nT = nT

        # Use the fact that nessai can batch likelihood computations
        self._vectorised_likelihood = True


        if use_GPU:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        # Move data and psd to the selected backend (NumPy or CuPy) once, so that we don't have to keep transferring them for each batch of particles in the objective function.
        self.data = self.xp.asarray(data)
        self.psd = self.xp.asarray(psd)
        # self.results_array = self.xp.zeros((total_number_of_particles,), dtype=np.float64) # Pre-allocate array for results

        # self.total_number_of_particles = total_number_of_particles

        # self.statistic_array = self.xp.zeros((likelihood_chunksize,self.nT,2),dtype=complex) # self.xp.zeros((self.batch_size,), dtype=np.float64) # Pre-allocate array for the search statistic values for each batch of particles

        self._wf_params = self.xp.zeros((likelihood_chunksize, 8), dtype=np.float64) 
        self._resp_params = self.xp.zeros((likelihood_chunksize, 4), dtype=np.float64)


        # Storing d_d by computing it once at the beginning of the inference. 
        # Both data and PSD are shaped as (nT,nF,3)
        self.d_d = 4*self.xp.abs(self.xp.sum(self.data.conjugate() * self.data / self.psd * dF))

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
        """Evaluate the semi-coherent search statistic (upsilon) for a batch of particles.

        Parameters
        ----------
        params : dict of ndarray
            Parameter arrays keyed by name.  Each array has shape
            ``(batch_size,)``.

        Returns
        -------
        statistic : ndarray
            Search statistic value for each particle.
        """

        # t_0 = perf_counter()  
        
        nlive = params["Mc"].shape[0]

        # Move particle arrays to the selected backend once (NumPy or CuPy).
        Mc = self.xp.asarray(params["Mc"])
        q = self.xp.asarray(params["q"])
        D = self.xp.asarray(params["D"])*1.e+6 # Convert distance from Mpc to pc
        phi_coal = self.xp.asarray(params["phicoal"])
        cosinc = self.xp.asarray(params["cosinc"])
        f0 = self.xp.asarray(params["f0"])
        s1 = self.xp.asarray(params["s1"])
        s2 = self.xp.asarray(params["s2"])
        psi = self.xp.asarray(params["psi"])
        lam = self.xp.asarray(params["lam"])
        beta = self.xp.asarray(params["beta"])

        M = Mc * (q / (1 + q)**2)**(-3/5)

        eta = (Mc / M)**(5/3)
        
        # Subset to the number of particles in the current batch.  This way we can keep the GPU kernel's output array allocated to the maximum batch size, and just fill it with the current batch's results for each call to the objective function.
        wf_params = self._wf_params[:nlive]
        resp_params = self._resp_params[:nlive]

        wf_params[:, 0] = M
        wf_params[:, 1] = eta
        wf_params[:, 2] = cosinc
        wf_params[:, 3] = D
        wf_params[:, 4] = f0
        wf_params[:, 5] = s1
        wf_params[:, 6] = s2
        wf_params[:, 7] = phi_coal

        resp_params[:, 0] = cosinc
        resp_params[:, 1] = psi
        resp_params[:, 2] = lam
        resp_params[:, 3] = beta

        # if  self.statistic_array.shape[0] != nlive:
        #     print("Re-Allocating output array for the GPU kernel with batch size:", nlive)
        #     self.statistic_array = self.xp.zeros((nlive,self.nT,2), dtype=complex) # Allocate array for the search statistic values for each batch of particles if it hasn't been allocated yet or if the batch size has changed
                
        statistic_array = self.waveform_generator(parameters=wf_params, 
                                        channels=self.data,
                                        psds=self.psd,
                                        parameters_response=resp_params,
                                        out = None,
                                        compute_statistic=True)
        
        # self.statistic_array is shaped as (nlive, nT, 2)
        d_h_per_source = self.xp.sum(statistic_array[:,:,0], axis=1) 
        h_h_per_source = self.xp.sum(statistic_array[:,:,1], axis=1)
        log_likelihoods = -0.5 * (self.d_d + h_h_per_source - 2*d_h_per_source)

        # # CuPy arrays expose .get(); NumPy arrays do not.
        try:
            return log_likelihoods.get()
        
        except AttributeError:
            return log_likelihoods
    