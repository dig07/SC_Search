import numpy as np
import os

from .Noise import *
from .Swarm_class import Semi_Coherent_Model
import PySO

import pygwtf 
from pygwtf.models import TaylorT2Ecc, TaylorT3Spin
from pygwtf.generator import AnalyticTimeFrequencyWaveform
from pygwtf.response.orbits import generate_mojito_orbit_splines_resample

from scipy.interpolate import CubicSpline

from ldc.lisa.noise import get_noise_model

from lisaconstants import c as clight

# Default frequency bands where the PSD is flattened to suppress
# spectral artefacts (e.g. transfer-function zeroes).
DEFAULT_PSD_CLIP_BANDS = [
    (0.029, 0.031),
    (0.059, 0.061),
    (0.0897, 0.0902),
]


class Search:
    """Semi-coherent hierarchical PSO search for slowly-chirping signals in LISA data.

    The typical workflow is::

        search = Search(tf_dict, segment_ladder, prior_bounds,
                        PySO_num_swarms, PySO_num_particles, PySO_kwargs,
                        datafile_path='path/to/data')

        search.compute_psd(use_estimated_PSD=False)
        search.clip_psd()                          # optional
        search.inject_noise(gap_mask=gap_mask)      # optional
        search.setup_waveform_generator(use_GPU=True)
        search.initialize_and_run_search()

    Parameters
    ----------
    time_frequency_series_dict : dict
        Contains the time-frequency grid parameters.  Expected keys:
        ``'T_obs'``, ``'dT'``.
    segment_ladder : list of int
        Number of segments at each step of the semi-coherent hierarchy.
    prior_bounds : list
        Parameter-space prior for the search.
    PySO_num_swarms : int
        Initial number of swarms in the hierarchical PSO.
    PySO_num_particles : int
        Number of particles per swarm.
    PySO_kwargs : dict
        Additional keyword arguments forwarded to
        ``PySO.HierarchicalSwarmHandler``.
    datafile_path : str, optional
        Directory containing the data, time-grid, and frequency-grid
        ``.npy`` files.  Defaults to ``'.'``.
    data_file_name : str, optional
        Filename of the SFT data array.  Defaults to ``'data.npy'``.
    """

    def __init__(
        self,
        time_frequency_series_dict,
        segment_ladder,
        prior_bounds,
        PySO_num_swarms,
        PySO_num_particles,
        PySO_kwargs,
        datafile_path=".",
        data_file_name="data.npy",
    ):
        # Store search configuration
        self.frequency_series_dict = time_frequency_series_dict
        self.segment_ladder = segment_ladder
        self.prior_bounds = prior_bounds
        self.PySO_num_particles = PySO_num_particles
        self.PySO_num_swarms = PySO_num_swarms
        self.PySO_kwargs = PySO_kwargs

        # Load data and build time-frequency grid
        self._load_data_and_generate_tf_grid(data_file_name, datafile_path)


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
        self.data = self.data.transpose(1,2,0) # shape (nT, nF, 3)

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
        self.psd_arr = self.psd_arr.transpose(1,2,0) 

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
                                 spin_only_waveform=True):
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
        self.p = p.transpose(2,0,1)

        # Setup config that waveform generator needs
        config = {'nT':self.nT,
                  'nF':self.nF,
                  'dT':self.dT,
                  'dF':self.dF,
                  'kernel_width':fresnel_kernel_width}        
        
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
                                                                spacecraft_ltts=self.Ls)

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
                                                              t_sft=self.t_seg)

        return(p,Ls)

    def initialize_and_run_search(self):
        """Run the hierarchical semi-coherent PSO search.

        Creates a ``Semi_Coherent_Model`` for each rung in the segment
        ladder and passes them to ``PySO.HierarchicalSwarmHandler``.
        """
        self.Semi_Coherent_classes = [
            Semi_Coherent_Model(
                segment_number,
                self.prior_bounds,
                self.data,
                self.waveform_generator,
            )
            for segment_number in self.segment_ladder
        ]

        PySO_search = PySO.HierarchicalSwarmHandler(
            self.Semi_Coherent_classes,
            self.PySO_num_swarms,
            self.PySO_num_particles,
            **self.PySO_kwargs,
        )
        PySO_search.Run()

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