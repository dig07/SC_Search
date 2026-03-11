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
        Waveform generator instance (e.g. ``TaylorF2EccTF``) exposing a
        ``get_log_likelihood`` method.
    constant_final_orbital_phase : float, optional
        Fixed value used for the final orbital phase.  Defaults to 0.
    constant_distance : float, optional
        Fixed luminosity distance in parsecs.  Defaults to 1e8.
    """

    names = [
        "Mc",
        "q",
        "cosinc",
        "e0",
        "f0",
        "lam",
        "beta",
        "psi",
    ]

    def __init__(
        self,
        segment_number,
        priors,
        data,
        waveform_generator,
        constant_final_orbital_phase=0,
        constant_distance=100.0e6,
    ):
        self.segment_number = segment_number
        self.bounds = priors
        self.data = data
        self.waveform_generator = waveform_generator
        self.constant_final_orbital_phase = constant_final_orbital_phase
        self.constant_distance = constant_distance

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

        loglike = self.waveform_generator.get_log_likelihood(
            params["Mc"],
            params["q"],
            params["cosinc"],
            params["e0"],
            [self.constant_distance] * batch_size,
            params["f0"],
            [self.constant_final_orbital_phase] * batch_size,
            params["lam"],
            params["beta"],
            params["psi"],
            True,
            self.segment_number,
        )

        # CuPy arrays expose .get(); NumPy arrays do not.
        try:
            return loglike.get()
        except AttributeError:
            return loglike
        
