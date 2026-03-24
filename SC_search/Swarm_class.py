import numpy as np
import PySO
from math import ceil 



from time import perf_counter
from numba import cuda 


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
        "eta",
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
        eta = self.xp.asarray(params["eta"])
        cosinc = self.xp.asarray(params["cosinc"])
        f0 = self.xp.asarray(params["f0"])
        s1 = self.xp.asarray(params["s1"])
        s2 = self.xp.asarray(params["s2"])
        psi = self.xp.asarray(params["psi"])
        lam = self.xp.asarray(params["lam"])
        beta = self.xp.asarray(params["beta"])


        M = self.xp.asarray(params["Mc"]) / (eta ** (3 / 5))

        distance = self.xp.full((nparticles,), self.constant_distance, dtype=M.dtype)

        final_orbital_phase = self.xp.full((nparticles,), self.constant_final_orbital_phase, dtype=M.dtype)

        wf_parameters_all = self.xp.column_stack(
            (M, eta, cosinc, distance, f0, s1, s2, final_orbital_phase)
        )
        response_parameters_all = self.xp.column_stack((cosinc, psi, lam, beta)) 

        if  self.statistic_array.shape[0] != nparticles:
            print("Allocating output array for the GPU kernel with batch size:", self.batch_size)
            self.statistic_array = self.xp.zeros((nparticles,), dtype=np.float64) # Allocate array for the search statistic values for each batch of particles if it hasn't been allocated yet or if the batch size has changed

        self.waveform_generator(parameters=wf_parameters_all, 
                                        channels=self.data,
                                        psds=self.psd,
                                        parameters_response=response_parameters_all,
                                        out = None,
                                        compute_statistic=True,
                                        search_statistic = self.statistic_array,
                                        N_seg = self.segment_number)
        # # CuPy arrays expose .get(); NumPy arrays do not.
        try:
            return self.statistic_array.get()
        
        except AttributeError:
            return self.statistic_array
    