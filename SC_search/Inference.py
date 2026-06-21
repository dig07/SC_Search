import numpy as np
import os

from .Noise import *
from .Swarm_class import Model_inference
 
from pygwtf.models import TaylorT2Ecc, TaylorT3Spin
from pygwtf.generator import AnalyticTimeFrequencyWaveform
from pygwtf.response.orbits import generate_mojito_orbit_splines_resample

from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid

from ldc.lisa.noise import get_noise_model

from lisaconstants import c as clight

from nessai.flowsampler import FlowSampler
from nessai.utils import setup_logger

# Default frequency bands where the PSD is flattened to suppress
# spectral artefacts (e.g. transfer-function zeroes).
DEFAULT_PSD_CLIP_BANDS = [
    (0.029, 0.031),
    (0.059, 0.061),
    (0.0897, 0.0902),
]


class Inference:
    """Inference for slowly-chirping signals in LISA data.

    The typical workflow is::

        inference = Inference(tf_dict, segment_ladder, prior_bounds,
                              PySO_num_swarms, PySO_num_particles, PySO_kwargs,
                        datafile_path='path/to/data')

        inference.compute_psd(use_estimated_PSD=False)
        inference.clip_psd()                          # optional
        inference.inject_noise(gap_mask=gap_mask)      # optional
        inference.setup_waveform_generator(use_GPU=True)
        inference.initialize_and_run_inference()

    Parameters
    ----------
    time_frequency_series_dict : dict
        Contains the time-frequency grid parameters.  Expected keys:
        ``'T_obs'``, ``'dT'``.
    prior_bounds : dict
        Parameter-space prior for the inference.
    datafile_path : str, optional
        Directory containing the data, time-grid, and frequency-grid
        ``.npy`` files.  Defaults to ``'.'``.
    data_file_name : str, optional
        Filename of the SFT data array.  Defaults to ``'data.npy'``.
    t_grid_key : str, optional
        Filename for the time-segment grid.  Defaults to ``'t_grid.npy'``.
    f_grid_key : str, optional
        Filename for the frequency grid.  Defaults to ``'f_grid.npy'``.
    """

    def __init__(
        self,
        time_frequency_series_dict,
        datafile_path=".",
        data_file_name="data.npy",
        t_grid_key="t_grid.npy",
        f_grid_key="f_grid.npy",
    ):
        # Store search configuration
        self.frequency_series_dict = time_frequency_series_dict

        # Load data and build time-frequency grid
        self._load_data_and_generate_tf_grid(data_file_name, datafile_path, t_grid_key=t_grid_key, f_grid_key=f_grid_key)


    def _load_data_and_generate_tf_grid(
        self,
        data_file_name,
        datafile_path,
        t_grid_key="t_grid.npy",
        f_grid_key="f_grid.npy",
    ):
        """Load the SFT data array and construct the time-frequency grid.

        Reads ``data.npy``, ``t_grid.npy``, and ``f_grid.npy`` from
        *datafile_path*.

        Parameters
        ----------
        data_file_name : str
            Filename of the SFT data array.
        datafile_path : str
            Directory containing all ``.npy`` files.
        t_grid_key : str, optional
            Filename for the time-segment grid.
        f_grid_key : str, optional
            Filename for the frequency grid.
        """
        self.data = np.load(os.path.join(datafile_path, data_file_name)) # shape (3, nT, nF)
        self.t_seg = np.load(os.path.join(datafile_path, t_grid_key))
        self.f_seg = np.load(os.path.join(datafile_path, f_grid_key))

        # Unpack grid parameters
        self.T_obs = self.frequency_series_dict["T_obs"]
        self.dT = self.frequency_series_dict["dT"]
        self.dF = 1.0 / self.dT

        # Trim frequency grid to search range
        self.nT = self.t_seg.size - 1
        self.nF = self.f_seg.size

        print(f"Final f_grid has size: {self.nF}")
        print(f"Frequency range on TF grid: [{self.f_seg[0]:.6f}, {self.f_seg[-1]:.6f}]")

        # Reshaping data for ingestion by the kenrel which expects (nT, nF, 3) shape. 
        self.data = self.data.transpose(1,2,0).copy() # shape (nT, nF, 3)

    def compute_psd(self, use_estimated_PSD=False, PSD_file_path="PSD_interpolator.npy"):
        """Build the PSD array.

        Either loads an empirically-estimated PSD from file, or evaluates
        the analytic *Sangria* noise model at each frequency bin.

        Parameters
        ----------
        use_estimated_PSD : bool, optional
            If ``True``, load the PSD from *PSD_file_path*.  The file is
            expected to be a dictionary saved with ``np.save`` containing
            keys ``'A'``, ``'E'``, ``'T'`` (each shaped
            ``(n_time_bins, n_freq_bins)``), ``'Times'``, and
            ``'Frequencies'``.  If ``False`` (default), the analytic
            *Sangria* noise model is used (stationary across segments).
        PSD_file_path : str, optional
            Path to the estimated PSD ``.npy`` file.
        """
        self.psd_arr = np.zeros((3, self.nT, self.nF))

        if use_estimated_PSD:
            self._load_estimated_psd(PSD_file_path)
        else:
            self._compute_analytic_psd()

        # Reshaping PSD for ingestion by the kernel which expects (nT, nF, 3) shape. 
        self.psd_arr = self.psd_arr.transpose(1,2,0).copy()

        print(f"PSD shape: {self.psd_arr.shape}  |  Data shape: {self.data.shape}")
        print(f"PSD entirely positive: {np.all(self.psd_arr > 0)}")

    def _load_estimated_psd(self, PSD_file_path):
        """Populate ``self.psd_arr`` from an empirically-estimated PSD file.

        For each time segment the PSD is looked up by nearest earlier
        time bin and interpolated onto ``self.f_seg`` via log-space cubic
        splines.  If a segment time exceeds the last estimation point
        the final PSD bin is re-used.
        """
        print("Using directly estimated PSD...")

        psd_object = np.load(PSD_file_path, allow_pickle=True).item()

        psd_channels = np.array([psd_object["A"], psd_object["E"], psd_object["T"]])
        time_points = psd_object["Times"]
        frequency_points = psd_object["Frequencies"]

        # Map each t_seg edge to the nearest PSD time bin
        time_indices = np.searchsorted(time_points, self.t_seg)
        for t_idx in range(self.nT):
            # Which psd from the welch estimation to use for this segment?  Look up the nearest time bin.
            psd_time_idx = time_indices[t_idx]
            # Edge case, i.e self.t_seg > time_points[-1] (when this happens psd_time_idx == psd_channels.shape[1] from the searchsorted above), just asusme it remains constant
            if psd_time_idx == psd_channels.shape[1]:
                psd_time_idx = -1
            self.psd_arr[:, t_idx, :] = np.array(
                [self._interpolate_psd_onto_grid(frequency_points, psd_channels[channel, psd_time_idx, :]) for channel in range(3)]
            )

    def _compute_analytic_psd(self):
        """Populate ``self.psd_arr`` using the analytic Sangria noise model.

        The model is evaluated once (stationary) and tiled across all
        time segments.
        """
        print("Using analytic PSD...")

        wd_years = self.T_obs / (365.25 * 24 * 60 * 60)
        noise = get_noise_model("sangria", self.f_seg, wd=wd_years)

        psd_per_channel = np.array([
            noise.psd(self.f_seg, option="A", tdi2=True),
            noise.psd(self.f_seg, option="E", tdi2=True),
            noise.psd(self.f_seg, option="T", tdi2=True),
        ])  # shape (3, nF)

        # Broadcast the stationary PSD to every time segment
        self.psd_arr[:] = psd_per_channel[:, np.newaxis, :]

    def clip_psd(self, clip_bands=None):
        """Flatten the PSD inside specified frequency bands.

        This suppresses narrow spectral artefacts (e.g. transfer-function
        zeroes at ~30 mHz harmonics) by replacing the PSD values in each
        band with the value at the band's lower edge.

        Parameters
        ----------
        clip_bands : list of (float, float), optional
            Each tuple gives ``(f_low, f_high)`` in Hz.  Defaults to
            ``DEFAULT_PSD_CLIP_BANDS``.
        """
        if clip_bands is None:
            clip_bands = DEFAULT_PSD_CLIP_BANDS

        for f_low, f_high in clip_bands:
            # Beginning index of dip 
            idx_low = int(np.argmin(np.abs(self.f_seg - f_low)))
            # Ending index of dip
            idx_high = int(np.argmin(np.abs(self.f_seg - f_high)))
            # For all indexes in the dip, set the PSD to the value at the lower edge
            for k in range(idx_low, idx_high):
                self.psd_arr[:, k, :] = self.psd_arr[:, idx_low, :]

          
        print(f"PSD entirely positive: {np.all(self.psd_arr > 0)}")

    def inject_noise(self, gap_mask=None):
        """Generate a noise realisation from the current PSD and add it to the data.

        **Assumes the loaded data is noise-free** (signal only).

        Parameters
        ----------
        gap_mask : array-like of int or None, optional
            Indices of time segments that are *kept* (i.e. not gapped).
            Segments not in *gap_mask* are zeroed after noise injection.
        """
        noise_tf = self._generate_noise_realisation(self.psd_arr)
        self.data += noise_tf

        if gap_mask is not None:
            kept = np.asarray(gap_mask)
            all_indices = np.arange(self.nT)
            dropped = np.setdiff1d(all_indices, kept)
            self.data[dropped, :, :] = 0.0

    def setup_waveform_generator(self, mojito_orbit_filepath='./mojito_orbits.h5',
                                 mojito_ltt_filepath='./mojito_ltts.h5',
                                 use_GPU=True, fresnel_kernel_width=5, gap_mask=None,
                                 spin_only_waveform=True,
                                 block_vectorised = False):
        """Initialise the waveform generator. 
        
        Two main elements to this: 
            - F2Ecc waveform model functions: amplitude, time to coalescence, and phase evolution.
            - LISA response function: AET transfer functions. 
         
        Group waveform + response together. 
            At the end of the day they both combine to produce the GW model which enters the statistic/likelihood.

        Parameters
        ----------
        mojito_orbit_filepath : str, optional
            Containing ESA orbits for the spacecraft (used to setup the response function).  Defaults to './mojito_orbits.h5'.
        mojito_ltt_filepath : str, optional
            Containing ESA light travel times for the spacecraft (used to setup the response function).
        use_GPU : bool, optional
            Whether to run the waveform model on CUDA.  Defaults to True.
        fresnel_kernel_width : int, optional
            Width of the Fresnel-kernel summation.  Defaults to 5.
        gap_mask : array-like of int or None, optional
            If provided, the corresponding segment mask is applied to the
            waveform generator so that gapped segments are excluded.
            NOTE: Not implemented yet
        spin_only_waveform : bool, optional
            If True, use a waveform model that includes only spin effects and no eccentricity. (T3)
            If False, use a waveform model that includes both spin and eccentricity effects. (F2Ecc)  Defaults to True.
        """
        # Positions of spacecraft (3,3,nT)
        p, Ls = self.setup_response_function(mojito_orbit_filepath=mojito_orbit_filepath, mojito_ltt_filepath=mojito_ltt_filepath)

        self.Ls = Ls*clight # Convert to seconds for the waveform generator.

        # Needs to transform this to (nT,3,3) for the gwtf kernel 
        self.p = p.transpose(2,0,1).copy()

        # Setup config that waveform generator needs
        config = {'nT':self.nT,
                  'nF':self.nF,
                  'dT':self.dT,
                  'dF':self.dF,
                  'kernel_width':fresnel_kernel_width}    
        
        # If using block PE sampling, use the kernel width of 16
        if block_vectorised:
            config['kernel_width'] = 16   
        
        if use_GPU:
            backend = 'gpu'
        else:
            backend = 'cpu'

        print(f"Setting up waveform generator with {backend} backend...")

        if spin_only_waveform:
            # No eccentricity, spin algined. 
            wf_model_class = TaylorT3Spin
        else:
            # Eccentricity only no spin. 
            wf_model_class = TaylorT2Ecc
        self.waveform_generator = AnalyticTimeFrequencyWaveform(model_class=wf_model_class, 
                                                                config=config,
                                                                tdi_type=2,
                                                                backend=backend,
                                                                channels=self.data,
                                                                spacecraft_orbits=self.p,
                                                                spacecraft_ltts=self.Ls,
                                                                block_vectorised_gpu = block_vectorised)
        # This returns a function which is the kernel that directly takes in waveform parameters and outputs search statistics.         
        # self.statistic_generator = self.waveform_generator.statistic_kernel
        
        # Waveform generator object, fills in array provided to it with waveform, useful for debugging, constructed using the same methods as the statistic generator 
        # self.debugging_waveform_generator = self.waveform_generator.waveform_kernel
        

        # if gap_mask is not None:
        #     self.waveform_generator.apply_segment_mask(gap_mask)

    def setup_response_function(self,mojito_orbit_filepath='./mojito_orbits.h5', mojito_ltt_filepath='./mojito_ltts.h5'):
        """
        Setup the LISA response funtion. 

        - Reads in the mojito orbit file which contains position data for each spacecraft.
        - Interpolates (using Cubicspline) to SFT fixed time array to the SFT segment times. 
        - Outputs positions of all spacecraft at central times within each SFT segment. 

        Ran once at the beginning of analysis. 

        Feeds into the AET_TFs_func within the kernel. 

        Parameters
        ----------
        mojito_orbit_filepath : str, optional
            Containing ESA orbits.  Defaults to './mojito_orbits.h5'.
        mojito_ltt_filepath : str, optional
            Containing ESA light travel times for the spacecraft (used to setup the response function).  Defaults to './mojito_ltts.h5'.

        Returns
        -------
        p : array of shape (3,3,nT)
            Positions of 3 spacecraft in SSB frame at central SFT times. 
        Ls : array of shape (nT,3)
            Light travel times for each link at central SFT times.
        """
        
        p, Ls = generate_mojito_orbit_splines_resample(mojito_orbit_filepath=mojito_orbit_filepath, 
                                                    mojito_ltt_filepath=mojito_ltt_filepath,
                                                              t_tranches=self.t_seg)

        return(p,Ls)

    def initialize_and_run_inference(self,
                                     priors,
                                     nlive=1000,
                                     use_GPU=False,
                                     outdir="./Inference_output",
                                     mass_parameterisation="m1m2",
                                     sampler_kwargs = {}):
        """Run *coherent* inference.

        Parameters
        ----------
        use_GPU : bool, optional
            Whether to run the search on CUDA.  Defaults to True.
        nlive: int, optional
            Number of live points to use in the nested sampling inference.  Defaults to 1000
        outdir: str, optional
            Output directory for the inference results.  Defaults to "./Inference_output".
        mass_parameterisation : {"m1m2", "mceta"}, optional
            Which mass coordinates the sampler varies. ``"m1m2"`` (default) ->
            ``priors`` must supply ``m1`` and ``m2``; ``"mceta"`` -> it must
            supply ``Mc`` and ``eta`` instead. See
            :class:`~SC_search.Swarm_class.Model_inference`.
        sampler_kwargs: dict, optional
            Additional keyword arguments to pass to the FlowSampler.  Defaults to an empty dictionary.

        """
        self.inference_class = Model_inference(priors,
                                                self.data,
                                                self.waveform_generator,
                                                self.nT,
                                                self.psd_arr,
                                                self.dF,
                                                use_GPU=use_GPU,
                                                mass_parameterisation=mass_parameterisation) # This is the batch size for the likelihood evaluation, we set it to nlive so that the GPU kernel can process all particles in one batch.

        logger = setup_logger(output=outdir)

        self.sampler = FlowSampler(self.inference_class, # This is the batch size for the likelihood evaluation, we set it to nlive so that the GPU kernel can process all particles in one batch.
                                    output=outdir,
                                    nlive=nlive,
                                    **sampler_kwargs)

        self.sampler.run()

    def initialize_and_run_inference_eryn(self,
                                          priors,
                                          nwalkers=1000,
                                          ntemps=1,
                                          nsteps=5000,
                                          burn=None,
                                          thin_by=1,
                                          use_GPU=False,
                                          outdir="./Inference_output",
                                          mass_parameterisation="m1m2",
                                          injection_params=None,
                                          injection_scatter=1e-3,
                                          injection_std=None,
                                          temper_seed_width=True,
                                          periodic=None,
                                          progress=True,
                                          sampler_kwargs={}):
        """Run *coherent* inference with the eryn ensemble (PT-)MCMC sampler.

        This is the eryn analogue of :meth:`initialize_and_run_inference` (which
        uses nessai).  It wraps the vectorised likelihood of
        :class:`~SC_search.Swarm_class.Model_inference` so eryn evaluates a whole
        batch of walkers in a single GPU kernel call.

        Parameters
        ----------
        priors : dict
            Prior bounds keyed by parameter name, e.g.
            ``dict(m1=[lo, hi], m2=[lo, hi], cosinc=[-1, 1], ...)``.  Must contain
            every name the model expects for the chosen ``mass_parameterisation``
            (i.e. ``m1``/``m2`` or ``Mc``/``eta`` for the two masses, plus the
            shared parameters), same dict format as the nessai path.  Bounds are
            interpreted as uniform priors.
        nwalkers : int, optional
            Number of walkers *per temperature*.  Defaults to 100.
        ntemps : int, optional
            Number of parallel-tempering temperatures.  eryn builds its own
            default (geometric) inverse-temperature ladder for ``ntemps`` rungs
            and **adapts** it during the run (eryn's standard adaptive
            tempering).  ``ntemps = 1`` (default) is a single cold chain, i.e.
            plain ensemble MCMC with no tempering.  The cold chain (``beta = 1``)
            is index 0, the convention the analysis/plotting code assumes.
        nsteps : int, optional
            Number of stored MCMC iterations.  Defaults to 5000.
        burn : int or None, optional
            Number of burn-in iterations run before storing (not written to the
            backend).  Defaults to ``None`` (no burn-in).
        thin_by : int, optional
            Store only every ``thin_by``-th iteration.  Defaults to 1.
        use_GPU : bool, optional
            Whether the likelihood runs on CUDA.  Defaults to False.
        outdir : str, optional
            Output directory.  The chain is written to
            ``<outdir>/eryn_state.h5`` via an :class:`eryn.backends.HDFBackend`.
        mass_parameterisation : {"m1m2", "mceta"}, optional
            Which mass coordinates the walkers sample. ``"m1m2"`` (default) ->
            ``priors`` (and ``injection_params`` / ``injection_std``) use ``m1``
            and ``m2``; ``"mceta"`` -> they use ``Mc`` and ``eta`` instead. The
            likelihood converts to ``(Mc, eta, M)`` internally either way. See
            :class:`~SC_search.Swarm_class.Model_inference`.
        injection_params : dict, array-like, or None, optional
            If provided, the walkers are seeded in a tight Gaussian ball around
            these values (the "injection") and evolve from there, instead of
            being drawn from the prior.  May be a dict keyed by parameter name
            (a subset is allowed -- names not supplied are drawn from the prior)
            or a full-length array in ``Model_inference.names`` order.
        injection_scatter : float, optional
            Std of the seeding ball as a fraction of each parameter's prior
            width.  Defaults to ``1e-3`` (0.1 %).  Walkers are clipped to stay
            inside the prior so every starting point has finite prior.  Used as
            the fallback for any dimension not covered by ``injection_std``.
        injection_std : float, dict, array-like, or None, optional
            Per-dimension **absolute** standard deviation of the seeding ball,
            in each parameter's own units (overrides ``injection_scatter`` for
            the dimensions it covers).  May be:

            - a scalar applied to every dimension;
            - a dict keyed by parameter name (a subset is allowed -- dimensions
              not supplied fall back to ``injection_scatter * prior_width``);
            - a full-length array/list in ``Model_inference.names`` order
              (entries that are ``NaN`` fall back to the scatter default).

            ``None`` (default) reproduces the original behaviour where every
            dimension uses ``injection_scatter * prior_width``.
        temper_seed_width : bool, optional
            When seeding around an injection in a parallel-tempered run
            (``ntemps > 1``), widen each temperature's seeding ball by
            ``sqrt(T) = 1/sqrt(beta)`` so every rung starts at its own
            tempered-posterior width and no chain has to slowly expand. The
            cold chain (``beta = 1``) is left exactly as ``injection_std``
            specifies. Rungs whose widened ball would already be as broad as
            the prior fall back to a prior draw. ``True`` by default; set
            ``False`` to seed every temperature with the same (cold) ball.
        periodic : dict or None, optional
            Periodic parameters as ``{name: period}``.  ``None`` (default)
            auto-detects the standard angles present (``phicoal`` -> 2*pi,
            ``psi`` -> pi).  Pass ``{}`` to disable periodic boundaries.
        progress : bool, optional
            Show a tqdm progress bar.  Defaults to True.
        sampler_kwargs : dict, optional
            Extra keyword arguments forwarded to ``EnsembleSampler``.

        Returns
        -------
        sampler : eryn.ensemble.EnsembleSampler
            The sampler after running.  The chain is available via
            ``sampler.get_chain()["model_0"]`` with shape
            ``(nsteps, ntemps, nwalkers, 1, ndim)``; the cold-chain posterior is
            ``[..., 0, :, 0, :]``.  Columns follow ``Model_inference.names``.
        """
        import numpy as np
        # eryn 1.2.6 still calls np.in1d, which NumPy 2.0 removed in favour of
        # np.isin. They are equivalent for the 1-D inputs eryn uses, so restore
        # the alias rather than editing the installed package.
        if not hasattr(np, "in1d"):
            np.in1d = np.isin
            
        from eryn.ensemble import EnsembleSampler
        from eryn.prior import ProbDistContainer, uniform_dist, spline_prior
        from eryn.state import State
        from eryn.backends import HDFBackend
        from eryn.utils import PlotContainer
        from eryn.utils.periodic import PeriodicContainer

        self.inference_class = Model_inference(priors,
                                               self.data,
                                               self.waveform_generator,
                                               self.nT,
                                               self.psd_arr,
                                               self.dF,
                                               use_GPU=use_GPU,
                                               vectorize=True,
                                               mass_parameterisation=mass_parameterisation)

        # Parameter order is fixed by the model; the eryn coordinate array and
        # all priors below are built in exactly this order.
        names = self.inference_class.names
        ndim = len(names)

        missing = [n for n in names if n not in priors]
        if missing:
            raise ValueError(f"priors is missing bounds for parameters: {missing}")

        lows = np.array([priors[n][0] for n in names], dtype=np.float64)
        highs = np.array([priors[n][1] for n in names], dtype=np.float64)

        # Setting up a volumetric prior spline over the distances p(d) \proptp d^2

        low_distance = lows[names.index("D")]
        high_distance = highs[names.index("D")]

        # Analytical normalisation factor 
        fac = (high_distance**3)/3 - (low_distance**3)/3
        Amp_D = 1/fac

        # spline 
        distances = np.linspace(low_distance, high_distance, 1000)
        CDF = Amp_D * ((distances**3)/3 - (low_distance)**3/3)
        
        inv_cdf_spline = CubicSpline(CDF, distances)
        pdf_spline = CubicSpline(distances, Amp_D * distances**2)

        # Setting up a prior on the chirp mass and symmetric mass ratio, which is uniform in m1 and m2.

        low_mc = lows[names.index("Mc")]
        high_mc = highs[names.index("Mc")]

        low_eta = lows[names.index("eta")]
        high_eta = highs[names.index("eta")]

        # p(Mc) \propto Mc 

        fac_mc = (high_mc**2 - low_mc**2)/2
        Amp_Mc = 1/fac_mc

        Mcs = np.linspace(low_mc, high_mc, 1000)
        CDF_Mc = Amp_Mc * ((Mcs**2)/2 - (low_mc**2)/2)

        inv_cdf_spline_Mc = CubicSpline(CDF_Mc, Mcs)
        pdf_spline_Mc = CubicSpline(Mcs, Amp_Mc * Mcs)

        # p(eta) \propto etaˆ{-6/5}*(1-4*eta)ˆ{-1/2}
        
        # Normalise + build the CDF by trapezoidal integration of the PDF over
        # the allowed eta range. cumulative_trapezoid(..., initial=0) anchors the
        # CDF at exactly 0 at low_eta (so the inverse spline covers u in [0, 1])
        # and its final value is the normalisation integral.
        etas = np.linspace(low_eta, high_eta, 1000)
        pdf_eta = etas**(-6/5) * (1 - 4*etas)**(-1/2)
    
        cum_eta = cumulative_trapezoid(pdf_eta, etas, initial=0.0)
        fac_eta = cum_eta[-1]
        Amp_eta = 1/fac_eta

        CDF_eta = Amp_eta * cum_eta

        inv_cdf_spline_eta = CubicSpline(CDF_eta, etas)
        pdf_spline_eta = CubicSpline(etas, Amp_eta * pdf_eta)




        # Uniform prior per parameter, keyed by integer index (== column index).
        priors_in = {i: uniform_dist(lows[i], highs[i]) for i in range(ndim)}

        # Override the three astrophysically-motivated parameters with their
        # spline-backed priors built above: volumetric distance p(D) ∝ D^2,
        # mass-uniform chirp mass p(Mc) ∝ Mc, and p(eta) ∝ eta^(-6/5)(1-4eta)^(-1/2).
        priors_in[names.index("D")] = spline_prior(
            pdf_spline, inv_cdf_spline, low_distance, high_distance)
        priors_in[names.index("Mc")] = spline_prior(
            pdf_spline_Mc, inv_cdf_spline_Mc, low_mc, high_mc)
        priors_in[names.index("eta")] = spline_prior(
            pdf_spline_eta, inv_cdf_spline_eta, low_eta, high_eta)

        prior_container = ProbDistContainer(priors_in)

        # Periodic boundaries for angular parameters (improves mixing a lot).
        if periodic is None:
            periodic_names = {} 
            if "phicoal" in names: # Coalescence phase
                periodic_names["phicoal"] = 2 * np.pi
            if "psi" in names: # Polarisation angle
                periodic_names["psi"] = np.pi
            if "lam" in names: # Ecliptic Longitude
                periodic_names["lam"] = 2 * np.pi
        else:
            periodic_names = periodic
            
        # NB: build the PeriodicContainer ourselves rather than handing
        # EnsembleSampler a plain dict. This installed eryn expects the nested
        # `{branch_name: {param: period}}` shape paired with a `key_order`
        # (periodic.py:27-39): the str param names are resolved to column
        # indices via key_order. Handing EnsembleSampler our flat
        # `{param: period}` dict instead makes it call PeriodicContainer with no
        # key_order, which crashes ('float' has no .items()). With key_order
        # supplied here, ONLY phicoal/psi/lam (their column indices) are wrapped;
        # every other parameter is left untouched.
        # periodic_eryn = (
        #     PeriodicContainer(
        #         {"model_0": dict(periodic_names)},
        #         key_order={"model_0": names},
        #     )
        #     if periodic_names else None
        # )
        # periodic_eryn = (
        #     {"model_0": {names.index(n): p for n, p in periodic_names.items()}}
        #     if periodic_names else None
        # )
        
        periodic_dict = {"model_0":  {names.index(n): p for n, p in periodic_names.items()}}
        periodic_keys = {"model_0": names}
        
        periodic_eryn = PeriodicContainer(periodic_dict,key_order=periodic_keys)
        
        print("Periodic parameters (eryn):", periodic_dict, periodic_keys)


        os.makedirs(outdir, exist_ok=True)
        backend_path = os.path.join(outdir, "eryn_state.h5")
        # Always start fresh: a pre-existing backend would otherwise be silently
        # continued by eryn (and trips an eryn key-order check on reload).
        if os.path.exists(backend_path):
            os.remove(backend_path)
        backend = HDFBackend(backend_path)

        # plotter = PlotContainer(
        #     plots='base',
        #     parent_folder=outdir,
        #     tempering_palette="icefire",
        #     discard=0.5,
        # )



        # Let eryn build and ADAPT its own temperature ladder. Passing ntemps
        # to EnsembleSampler's tempering_kwargs makes it construct a default
        # (geometric) inverse-temperature ladder for ntemps rungs and adapt it
        # during the run (adaptive=True is eryn's default) -- the standard
        # adaptive tempering. The cold chain (beta = 1) stays index 0.
        ntemps = int(ntemps)
        if ntemps < 1:
            raise ValueError(f"ntemps must be >= 1 (1 = plain ensemble); got {ntemps}")
        print(f"Adaptive temperature ladder requested: ntemps = {ntemps}")

        # ntemps == 1 is a plain ensemble (no tempering); >1 hands eryn the
        # rung count and lets it build + adapt the ladder.
        tempering_kwargs = dict(ntemps=ntemps) if ntemps > 1 else {}

        sampler = EnsembleSampler(
            nwalkers,
            ndim,
            self.inference_class.log_likelihood,  # vectorised: (n_points, ndim) -> (n_points,)
            prior_container,
            tempering_kwargs=tempering_kwargs,
            vectorize=True,
            backend=backend,
            periodic=periodic_eryn,
            **sampler_kwargs
        )

        betas0 = sampler.temperature_control.betas.copy()   # inverse-temp ladder, shape (ntemps,)
        print(f"Initial beta ladder: {betas0}")
        print(f"Initial temperature ladder: {1/betas0}")
            # **sampler_kwargs
        # )
        #     plot_generator=plotter,
        #     plot_iterations=100,
        # )

            # plot_generator=plotter,
            # plot_iterations=100,

        # Build the initial walker coordinates: (ntemps, nwalkers, ndim).
        coords = prior_container.rvs(size=(ntemps, nwalkers))

        if injection_params is not None:
            injection_vec = self._eryn_injection_vector(injection_params, names)
            # Per-dimension absolute seeding std; NaN entries fall back to the
            # injection_scatter * prior_width default below.
            if injection_std is None:
                std_vec = np.full(ndim, np.nan)
            elif np.isscalar(injection_std):
                std_vec = np.full(ndim, float(injection_std))
            else:
                std_vec = self._eryn_injection_vector(injection_std, names)

            # Per-temperature widening of the seeding ball. Tempering raises the
            # likelihood to power beta = 1/T, which divides the variance of a
            # locally-Gaussian peak by beta -> each posterior 1-sigma grows by
            # 1/sqrt(beta) = sqrt(T). (NOTE: sqrt(T), not T -- T scales the
            # log-likelihood / the variance, so the *std* scales as its sqrt.)
            # Seeding every rung at its own equilibrium width means no chain,
            # cold or hot, has to slowly expand; the cold chain (beta = 1) is
            # left exactly as injection_std specifies. To use literal T instead,
            # replace np.sqrt(betas) with betas below.
            if temper_seed_width and ntemps > 1:
                betas = np.asarray(sampler.temperature_control.betas, dtype=np.float64)
                with np.errstate(divide="ignore"):
                    temp_scale = np.where(betas > 0, 1.0 / np.sqrt(betas), np.inf)
            else:
                temp_scale = np.ones(ntemps)

            rng = np.random.default_rng()
            for i in range(ndim):
                if np.isnan(injection_vec[i]):
                    continue  # parameter not supplied -> keep the prior draw
                width = highs[i] - lows[i]
                # Absolute std if supplied for this dim, else the scatter default.
                std_i = std_vec[i] if not np.isnan(std_vec[i]) else injection_scatter * width
                # Widen per temperature: std_t = std_i * sqrt(T_t).
                std_t = std_i * temp_scale                       # (ntemps,)
                ball = injection_vec[i] + std_t[:, None] * rng.standard_normal((ntemps, nwalkers))
                # Clip just inside the prior so every walker starts with finite log-prior.
                eps = 1e-10 * width
                ball = np.clip(ball, lows[i] + eps, highs[i] - eps)
                # Where the widened ball is already as broad as the prior (the
                # hottest rungs, or an infinite-T chain), the tempered posterior
                # ~ the prior: keep the original prior draw rather than a clipped
                # pile-up at the bounds.
                use_prior = std_t >= width                       # (ntemps,)
                ball[use_prior, :] = coords[use_prior, :, i]
                coords[:, :, i] = ball
            print("Seeding eryn walkers around the injection; hot chains widened by "
                  "sqrt(T) per temperature." if (temper_seed_width and ntemps > 1)
                  else "Seeding eryn walkers around the supplied injection parameters.")

        initial_state = State(coords)

        sampler.run_mcmc(initial_state, nsteps, burn=burn, thin_by=thin_by, progress=progress)

        self.sampler = sampler
        return sampler

    @staticmethod
    def _eryn_injection_vector(injection_params, names):
        """Build a length-``ndim`` injection vector ordered to match ``names``.

        Accepts either a dict keyed by parameter name (a subset is allowed --
        names not present are returned as ``NaN`` and later drawn from the prior)
        or a full-length array/list already in ``names`` order.
        """
        if isinstance(injection_params, dict):
            return np.array(
                [float(injection_params[n]) if n in injection_params else np.nan
                 for n in names],
                dtype=np.float64,
            )
        injection_vec = np.asarray(injection_params, dtype=np.float64).ravel()
        if injection_vec.size != len(names):
            raise ValueError(
                f"injection_params has {injection_vec.size} values but the model "
                f"expects {len(names)} ({names})."
            )
        return injection_vec

    def _generate_noise_realisation(self, psd):
        """Draw a coloured-noise realisation from the given PSD array.

        Generates independent Gaussian noise for each TDI channel
        (A, E, T) and each time segment, using the per-segment PSD and
        the segment duration ``dT``.

        Parameters
        ----------
        psd : ndarray, shape (nT, nF, 3)
            One-sided PSD for each channel and time segment.

        Returns
        -------
        noise : ndarray, shape (nT, nF, 3), complex
            Frequency-domain noise realisation.
        """
        noise = np.zeros((self.nT, self.nF, 3), dtype=complex)

        for t_idx in range(self.nT):
            for ch in range(3):
                noise[t_idx, :, ch] = noise_realization(psd[t_idx, :, ch], self.dT)

        return noise

    def _interpolate_psd_onto_grid(self, f_sparse, psd_sparse):
        """Interpolate a sparse PSD onto ``self.f_seg`` via log-space cubic splines.

        Operates in log-space so that the interpolated PSD is guaranteed
        to remain strictly positive.

        Parameters
        ----------
        f_sparse : ndarray
            Frequency sample points of the sparse PSD.
        psd_sparse : ndarray
            PSD values at *f_sparse*.

        Returns
        -------
        psd_dense : ndarray
            Interpolated PSD evaluated on ``self.f_seg``.
        """
        positive = psd_sparse > 0
        spline = CubicSpline(f_sparse[positive], np.log(psd_sparse[positive]))
        return np.exp(spline(self.f_seg))