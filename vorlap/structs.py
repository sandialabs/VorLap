"""
Data structures for the VorLap package.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import os
from scipy import interpolate


class AirfoilFFT:
    """
    Holds multidimensional FFT data for unsteady aerodynamic forces and moments
    as a function of Reynolds number and angle of attack.

    Attributes:
        name (str): Airfoil identifier name
        Re (np.ndarray): Reynolds number axis (1D)
        AOA (np.ndarray): Angle of attack axis in degrees (1D)
        Thickness (float): Relative airfoil thickness
        
        CL_ST (np.ndarray): Strouhal numbers for lift coefficient [Re × AOA × freq]
        CD_ST (np.ndarray): Strouhal numbers for drag coefficient
        CM_ST (np.ndarray): Strouhal numbers for moment coefficient
        CF_ST (np.ndarray): Strouhal numbers for total force coefficient magnitude
        
        CL_Amp (np.ndarray): FFT amplitudes of CL [Re × AOA × freq]
        CD_Amp (np.ndarray): FFT amplitudes of CD
        CM_Amp (np.ndarray): FFT amplitudes of CM
        CF_Amp (np.ndarray): FFT amplitudes of CF
        
        CL_Pha (np.ndarray): FFT phases of CL (radians)
        CD_Pha (np.ndarray): FFT phases of CD
        CM_Pha (np.ndarray): FFT phases of CM
        CF_Pha (np.ndarray): FFT phases of CF
        
        # Cached interpolators for performance optimization
        _interpolators_cached (bool): Whether interpolators are pre-computed
        _cl_st_interps (List): Pre-computed ST interpolators for CL
        _cl_amp_interps (List): Pre-computed amplitude interpolators for CL
        _cl_pha_interps (List): Pre-computed phase interpolators for CL
        _cd_st_interps (List): Pre-computed ST interpolators for CD
        _cd_amp_interps (List): Pre-computed amplitude interpolators for CD
        _cd_pha_interps (List): Pre-computed phase interpolators for CD
        _cf_st_interps (List): Pre-computed ST interpolators for CF
        _cf_amp_interps (List): Pre-computed amplitude interpolators for CF
        _cf_pha_interps (List): Pre-computed phase interpolators for CF
        
    Notes:
        - The frequency axis is implicit in the third dimension and must be consistent across all arrays.
        - All arrays must share shape `[length(Re), length(AOA), length(freq)]`.
        - Phases are in radians, and FFT data assumes periodic unsteady oscillations (e.g., vortex shedding).
    """
    
    def __init__(self, 
                 name: str,
                 Re: np.ndarray,
                 AOA: np.ndarray,
                 Thickness: float,
                 CL_ST: np.ndarray,
                 CD_ST: np.ndarray,
                 CM_ST: np.ndarray,
                 CF_ST: np.ndarray,
                 CL_Amp: np.ndarray,
                 CD_Amp: np.ndarray,
                 CM_Amp: np.ndarray,
                 CF_Amp: np.ndarray,
                 CL_Pha: np.ndarray,
                 CD_Pha: np.ndarray,
                 CM_Pha: np.ndarray,
                 CF_Pha: np.ndarray):
        """Initialize AirfoilFFT with the provided data."""
        self.name = name
        self.Re = Re
        self.AOA = AOA
        self.Thickness = Thickness
        
        self.CL_ST = CL_ST
        self.CD_ST = CD_ST
        self.CM_ST = CM_ST
        self.CF_ST = CF_ST
        
        self.CL_Amp = CL_Amp
        self.CD_Amp = CD_Amp
        self.CM_Amp = CM_Amp
        self.CF_Amp = CF_Amp
        
        self.CL_Pha = CL_Pha
        self.CD_Pha = CD_Pha
        self.CM_Pha = CM_Pha
        self.CF_Pha = CF_Pha
        
        # Initialize cached interpolators
        self._interpolators_cached = False
        self._cl_st_interps = []
        self._cl_amp_interps = []
        self._cl_pha_interps = []
        self._cd_st_interps = []
        self._cd_amp_interps = []
        self._cd_pha_interps = []
        self._cf_st_interps = []
        self._cf_amp_interps = []
        self._cf_pha_interps = []
    
    def _cache_interpolators(self):
        """Pre-compute and cache all interpolation objects for performance optimization."""
        if self._interpolators_cached:
            return
            
        n_freq = self.CL_ST.shape[2]
        
        # Pre-allocate lists
        self._cl_st_interps = [None] * n_freq
        self._cl_amp_interps = [None] * n_freq
        self._cl_pha_interps = [None] * n_freq
        self._cd_st_interps = [None] * n_freq
        self._cd_amp_interps = [None] * n_freq
        self._cd_pha_interps = [None] * n_freq
        self._cf_st_interps = [None] * n_freq
        self._cf_amp_interps = [None] * n_freq
        self._cf_pha_interps = [None] * n_freq
        
        # Create interpolators for each frequency
        for k in range(n_freq):
            # CL interpolators
            self._cl_st_interps[k] = interpolate.RegularGridInterpolator(
                (self.Re, self.AOA), self.CL_ST[:, :, k], bounds_error=False, fill_value=None)
            self._cl_amp_interps[k] = interpolate.RegularGridInterpolator(
                (self.Re, self.AOA), self.CL_Amp[:, :, k], bounds_error=False, fill_value=None)
            self._cl_pha_interps[k] = interpolate.RegularGridInterpolator(
                (self.Re, self.AOA), self.CL_Pha[:, :, k], bounds_error=False, fill_value=None)
            
            # CD interpolators
            self._cd_st_interps[k] = interpolate.RegularGridInterpolator(
                (self.Re, self.AOA), self.CD_ST[:, :, k], bounds_error=False, fill_value=None)
            self._cd_amp_interps[k] = interpolate.RegularGridInterpolator(
                (self.Re, self.AOA), self.CD_Amp[:, :, k], bounds_error=False, fill_value=None)
            self._cd_pha_interps[k] = interpolate.RegularGridInterpolator(
                (self.Re, self.AOA), self.CD_Pha[:, :, k], bounds_error=False, fill_value=None)
            
            # CF interpolators
            self._cf_st_interps[k] = interpolate.RegularGridInterpolator(
                (self.Re, self.AOA), self.CF_ST[:, :, k], bounds_error=False, fill_value=None)
            self._cf_amp_interps[k] = interpolate.RegularGridInterpolator(
                (self.Re, self.AOA), self.CF_Amp[:, :, k], bounds_error=False, fill_value=None)
            self._cf_pha_interps[k] = interpolate.RegularGridInterpolator(
                (self.Re, self.AOA), self.CF_Pha[:, :, k], bounds_error=False, fill_value=None)
        
        self._interpolators_cached = True


class Component:
    """
    Represents a single physical component in the full rotating structure. Each component includes
    a global transformation and local shape definition, segmented for per-section force calculations.

    Attributes:
        id (str): Identifier name for the component
        translation (np.ndarray): Translation vector applied to the entire component
        rotation (np.ndarray): Euler angle rotation vector [deg] around X, Y, Z axes
        pitch (np.ndarray): Pitch angle for the segment [deg], vectorized to avoid mutability
        shape_xyz (np.ndarray): Nx3 matrix of local segment positions (untransformed)
        shape_xyz_global (np.ndarray): Nx3 matrix of global segment positions (transformed)
        chord (np.ndarray): Chord length per segment
        twist (np.ndarray): Twist angle per segment [deg]
        thickness (np.ndarray): Relative thickness per segment (scales airfoil height), fraction of chord
        offset (np.ndarray): Offset values per segment
        airfoil_ids (List[str]): Airfoil data identifier for each segment (defaults to "default" if missing)
        chord_vector (np.ndarray): Chord vector for each segment
        normal_vector (np.ndarray): Normal vector for each segment
    """
    
    def __init__(self,
                 id: str,
                 translation: np.ndarray,
                 rotation: np.ndarray,
                 pitch: np.ndarray,
                 shape_xyz: np.ndarray,
                 shape_xyz_global: np.ndarray,
                 chord: np.ndarray,
                 twist: np.ndarray,
                 thickness: np.ndarray,
                 offset: np.ndarray,
                 airfoil_ids: List[str],
                 chord_vector: np.ndarray,
                 normal_vector: np.ndarray):
        """Initialize Component with the provided data."""
        self.id = id
        self.translation = translation
        self.rotation = rotation
        self.pitch = pitch
        self.shape_xyz = shape_xyz
        self.shape_xyz_global = shape_xyz_global
        self.chord = chord
        self.twist = twist
        self.thickness = thickness
        self.offset = offset
        self.airfoil_ids = airfoil_ids
        self.chord_vector = chord_vector
        self.normal_vector = normal_vector


class VIV_Params:
    """
    Encapsulates all top-level user-defined configuration inputs required for vortex-induced vibration analysis.

    Attributes:
        fluid_density (float): Air density [kg/m³]
        fluid_dynamicviscosity (float): Dynamic viscosity of air [Pa·s]
        rotation_axis (np.ndarray): Axis of rotation as a 3-element vector
        rotation_axis_offset (np.ndarray): Origin of the rotation axis (used in torque calculations and visualization)
        inflow_vec (np.ndarray): Direction of inflow velocity (typically [1, 0, 0] for +X)
        plot_cycle (List[str]): List of hex colors used to differentiate components in visualization
        azimuths (np.ndarray): Azimuthal angles [deg] swept by the rotor or structure
        inflow_speeds (np.ndarray): Freestream inflow speeds [m/s]
        output_time (np.ndarray): Output time points [s]
        freq_min (float): Minimum frequency [Hz] to consider in overlap comparison
        freq_max (float): Maximum frequency [Hz] to consider in overlap comparison
        airfoil_folder (str): Path to the airfoil folder
        n_harmonic (int): Number of harmonics to check against
        amplitude_coeff_cutoff (float): Lower threshold on what amplitudes are of interest
        n_freq_depth (int): How deep to go in the Strouhaul number spectrum
        output_azimuth_vinf (Tuple[float, float]): Used to limit the case where the relatively expensive output signal reconstruction is done
    """
    
    def __init__(self,
                 fluid_density: float = 1.225,
                 fluid_dynamicviscosity: float = 1.81e-5,
                 rotation_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
                 rotation_axis_offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
                 inflow_vec: np.ndarray = np.array([1.0, 0.0, 0.0]),
                 plot_cycle: Optional[List[str]] = None,
                 azimuths: Optional[np.ndarray] = None,
                 inflow_speeds: Optional[np.ndarray] = None,
                 output_time: Optional[np.ndarray] = None,
                 freq_min: float = 0.0,
                 freq_max: float = float('inf'),
                 airfoil_folder: Optional[str] = None,
                 n_harmonic: int = 5,
                 amplitude_coeff_cutoff: float = 0.01,
                 n_freq_depth: int = 3,
                 output_azimuth_vinf: Tuple[float, float] = (5.0, 2.0)):
        """Initialize VIV_Params with the provided data."""
        self.fluid_density = fluid_density
        self.fluid_dynamicviscosity = fluid_dynamicviscosity
        self.rotation_axis = rotation_axis
        self.rotation_axis_offset = rotation_axis_offset
        self.inflow_vec = inflow_vec
        
        if plot_cycle is None:
            self.plot_cycle = ["#348ABD", "#A60628", "#009E73", "#7A68A6", "#D55E00", "#CC79A7"]
        else:
            self.plot_cycle = plot_cycle
            
        if azimuths is None:
            self.azimuths = np.arange(0, 360, 5)
        else:
            self.azimuths = azimuths
            
        if inflow_speeds is None:
            self.inflow_speeds = np.arange(2.0, 11.0, 1.0)
        else:
            self.inflow_speeds = inflow_speeds
            
        if output_time is None:
            self.output_time = np.arange(0.0, 10.001, 0.001)
        else:
            self.output_time = output_time
            
        self.freq_min = freq_min
        self.freq_max = freq_max
        
        if airfoil_folder is None:
            module_path = os.path.dirname(os.path.abspath(__file__))
            self.airfoil_folder = os.path.join(module_path, "airfoils/")
        else:
            self.airfoil_folder = airfoil_folder
            
        self.n_harmonic = n_harmonic
        self.amplitude_coeff_cutoff = amplitude_coeff_cutoff
        self.n_freq_depth = n_freq_depth
        self.output_azimuth_vinf = output_azimuth_vinf
