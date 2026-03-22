import numpy as np
import PySO


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
    constant_final_orbital_phase : float, optional
        Fixed value used for the final orbital phase.  Defaults to 0.
    constant_distance : float, optional
        Fixed luminosity distance in parsecs.  Defaults to 1e8.

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
        constant_final_orbital_phase=0,
        constant_distance=100.0e6,
    ):
        self.segment_number = segment_number
        self.bounds = priors
        self.data = data
        self.waveform_generator = waveform_generator
        self.constant_final_orbital_phase = constant_final_orbital_phase
        self.constant_distance = constant_distance
        self.nT = nT
        self.psd = psd 

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
        batch_size = params["Mc"].shape[0]



        # Transform from Mc, eta to M, eta. 

        M = params["Mc"] / (params["eta"] ** (3 / 5))

        wf_parameters = np.array([M, 
                                params["eta"], 
                                params["cosinc"],
                                [self.constant_distance] * batch_size,
                                params["f0"], 
                                params["s1"], 
                                params["s2"], 
                                [self.constant_final_orbital_phase] * batch_size]).T
        
        
        response_parameters = np.array([params["cosinc"],
                                        params["psi"],
                                        params["lam"],
                                        params["beta"]]).T
                
        statistic_array = np.zeros((batch_size,self.nT,2),dtype=complex)

        search_statistics = self.waveform_generator(parameters=wf_parameters, 
                                    channels=self.data,
                                    psds=self.psd,
                                    parameters_response=response_parameters,
                                    out = statistic_array,
                                    compute_statistic=True,
                                    N_seg = self.segment_number)

        # CuPy arrays expose .get(); NumPy arrays do not.
        try:
            return search_statistics.get()
        except AttributeError:
            return search_statistics
        
