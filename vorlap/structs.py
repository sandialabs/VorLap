"""
Data structures for the VorLap package.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Tuple, Optional
import numpy as np
import os


def _array1d(value, name: str, *, dtype=float, copy: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    return arr.copy() if copy else arr


def _array2d(value, name: str, *, columns: int, dtype=float, copy: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 2 or arr.shape[1] != columns:
        raise ValueError(f"{name} must have shape [n, {columns}], got {arr.shape}.")
    return arr.copy() if copy else arr


def _vector3(value, default: List[float], name: str) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    arr = _array1d(value, name)
    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return arr


@dataclass
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

    Notes:
        - The frequency axis is implicit in the third dimension and must be consistent across all arrays.
        - All arrays must share shape `[length(Re), length(AOA), length(freq)]`.
        - Phases are in radians, and FFT data assumes periodic unsteady oscillations (e.g., vortex shedding).
    """
    
    name: str
    Re: np.ndarray
    AOA: np.ndarray
    Thickness: float
    CL_ST: np.ndarray
    CD_ST: np.ndarray
    CM_ST: np.ndarray
    CF_ST: np.ndarray
    CL_Amp: np.ndarray
    CD_Amp: np.ndarray
    CM_Amp: np.ndarray
    CF_Amp: np.ndarray
    CL_Pha: np.ndarray
    CD_Pha: np.ndarray
    CM_Pha: np.ndarray
    CF_Pha: np.ndarray
    reynolds_model: str = ""

    _FIELDS: ClassVar[Tuple[str, ...]] = ("CL", "CD", "CM", "CF")

    def __post_init__(self):
        self.name = str(self.name)
        self.reynolds_model = str(self.reynolds_model or "").strip()
        self.Re = _array1d(self.Re, "Re")
        self.AOA = _array1d(self.AOA, "AOA")
        thickness_arr = _array1d(self.Thickness, "Thickness")
        if thickness_arr.size != 1:
            raise ValueError("Thickness must contain exactly one scalar thickness ratio.")
        self.Thickness = float(thickness_arr[0])

        self.CL_ST = np.asarray(self.CL_ST, dtype=float)
        self.CD_ST = np.asarray(self.CD_ST, dtype=float)
        self.CM_ST = np.asarray(self.CM_ST, dtype=float)
        self.CF_ST = np.asarray(self.CF_ST, dtype=float)

        self.CL_Amp = np.asarray(self.CL_Amp, dtype=float)
        self.CD_Amp = np.asarray(self.CD_Amp, dtype=float)
        self.CM_Amp = np.asarray(self.CM_Amp, dtype=float)
        self.CF_Amp = np.asarray(self.CF_Amp, dtype=float)

        self.CL_Pha = np.asarray(self.CL_Pha, dtype=float)
        self.CD_Pha = np.asarray(self.CD_Pha, dtype=float)
        self.CM_Pha = np.asarray(self.CM_Pha, dtype=float)
        self.CF_Pha = np.asarray(self.CF_Pha, dtype=float)
        self.validate_schema()

    def _field_arrays(self, field: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return Strouhal, amplitude, and phase arrays for a field."""
        return (
            getattr(self, f"{field}_ST"),
            getattr(self, f"{field}_Amp"),
            getattr(self, f"{field}_Pha"),
        )

    def validate_schema(self) -> None:
        """
        Validate the FFT schema required by the VorLap solvers.

        Spectral index 0 is a mandatory DC/mean entry for every field. It must
        have Strouhal number zero; its amplitude is the signed mean coefficient
        and its phase is ignored. Indices 1..K are oscillatory peaks with
        nonnegative amplitudes and positive Strouhal numbers whenever the
        corresponding amplitude is nonzero.
        """
        if self.Re.size == 0 or self.AOA.size == 0:
            raise ValueError("Re and AOA axes must be non-empty.")
        if np.any(~np.isfinite(self.Re)) or np.any(~np.isfinite(self.AOA)):
            raise ValueError("Re and AOA axes must contain only finite values.")
        if np.any(self.Re <= 0.0):
            raise ValueError("Re axis values must be positive.")
        if self.Re.size > 1 and np.any(np.diff(self.Re) <= 0.0):
            raise ValueError("Re axis values must be strictly increasing and unique.")
        if self.AOA.size > 1 and np.any(np.diff(self.AOA) <= 0.0):
            raise ValueError("AOA axis values must be strictly increasing and unique.")
        if not np.isfinite(self.Thickness) or self.Thickness < 0.0:
            raise ValueError("Thickness must be a finite, non-negative scalar.")

        expected_prefix = (self.Re.size, self.AOA.size)
        common_depth = None
        tol = 1.0e-12
        for field in self._FIELDS:
            st, amp, pha = self._field_arrays(field)
            for suffix, arr in (("ST", st), ("Amp", amp), ("Pha", pha)):
                name = f"{field}_{suffix}"
                if arr.ndim != 3:
                    raise ValueError(f"{name} must be 3D [Re, AOA, freq], got shape {arr.shape}.")
                if arr.shape[:2] != expected_prefix:
                    raise ValueError(
                        f"{name} shape prefix {arr.shape[:2]} does not match Re/AOA axes {expected_prefix}."
                    )
                if arr.shape[2] < 1:
                    raise ValueError(f"{name} must contain at least one spectral entry.")
                if np.any(~np.isfinite(arr)):
                    raise ValueError(f"{name} contains non-finite values.")
                common_depth = arr.shape[2] if common_depth is None else common_depth
                if arr.shape[2] != common_depth:
                    raise ValueError("All airfoil FFT arrays must share the same frequency depth.")

            if not np.allclose(st[:, :, 0], 0.0, rtol=0.0, atol=tol):
                raise ValueError(f"{field}_ST spectral index 0 must be DC with Strouhal number 0.")
            if st.shape[2] > 1:
                harmonic_st = st[:, :, 1:]
                harmonic_amp = amp[:, :, 1:]
                if np.any(harmonic_st < -tol):
                    raise ValueError(f"{field}_ST harmonic Strouhal numbers must be non-negative.")
                if np.any(harmonic_amp < -tol):
                    raise ValueError(f"{field}_Amp harmonic amplitudes must be non-negative.")
                active = harmonic_amp > tol
                if np.any(active & (harmonic_st <= tol)):
                    raise ValueError(
                        f"{field} harmonic entries with nonzero amplitude must have positive Strouhal numbers."
                    )

        if self.reynolds_model == "replicated_reynolds_independent":
            if self.Re.size < 2:
                raise ValueError(
                    "reynolds_model='replicated_reynolds_independent' requires at least two Reynolds planes."
                )
            for field in self._FIELDS:
                for suffix, array in zip(("ST", "Amp", "Pha"), self._field_arrays(field)):
                    if not np.array_equal(array, np.broadcast_to(array[0:1, :, :], array.shape)):
                        raise ValueError(
                            f"{field}_{suffix} Reynolds planes are not exact replicas, but "
                            "reynolds_model='replicated_reynolds_independent' was declared."
                        )
    
@dataclass
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
    id: str
    translation: np.ndarray
    rotation: np.ndarray
    pitch: np.ndarray
    shape_xyz: np.ndarray
    shape_xyz_global: np.ndarray
    chord: np.ndarray
    twist: np.ndarray
    thickness: np.ndarray
    offset: np.ndarray
    airfoil_ids: List[str]
    chord_vector: np.ndarray
    normal_vector: np.ndarray

    def __post_init__(self):
        self.id = str(self.id)
        self.translation = _vector3(self.translation, [0.0, 0.0, 0.0], "translation")
        self.rotation = _vector3(self.rotation, [0.0, 0.0, 0.0], "rotation")
        self.pitch = _array1d(self.pitch, "pitch")
        self.shape_xyz = _array2d(self.shape_xyz, "shape_xyz", columns=3)
        n_nodes = self.shape_xyz.shape[0]
        if n_nodes == 0:
            raise ValueError("Component shape_xyz must contain at least one node.")
        self.shape_xyz_global = _array2d(self.shape_xyz_global, "shape_xyz_global", columns=3)
        if self.shape_xyz_global.shape[0] != n_nodes:
            raise ValueError("shape_xyz_global row count must match shape_xyz.")
        self.chord = _array1d(self.chord, "chord")
        self.twist = _array1d(self.twist, "twist")
        self.thickness = _array1d(self.thickness, "thickness")
        self.offset = _array1d(self.offset, "offset")
        for name, arr in (
            ("chord", self.chord),
            ("twist", self.twist),
            ("thickness", self.thickness),
            ("offset", self.offset),
        ):
            if arr.size != n_nodes:
                raise ValueError(f"{name} length must match shape_xyz rows.")
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values.")
        raw_airfoil_ids = [] if self.airfoil_ids is None else list(self.airfoil_ids)
        self.airfoil_ids = [str(airfoil_id) for airfoil_id in raw_airfoil_ids]
        if len(self.airfoil_ids) != n_nodes:
            raise ValueError("airfoil_ids length must match shape_xyz rows.")
        self.chord_vector = _array2d(self.chord_vector, "chord_vector", columns=3)
        self.normal_vector = _array2d(self.normal_vector, "normal_vector", columns=3)
        if self.chord_vector.shape[0] != n_nodes or self.normal_vector.shape[0] != n_nodes:
            raise ValueError("chord_vector and normal_vector row counts must match shape_xyz.")
        for name, arr in (
            ("translation", self.translation),
            ("rotation", self.rotation),
            ("pitch", self.pitch),
            ("shape_xyz", self.shape_xyz),
            ("shape_xyz_global", self.shape_xyz_global),
            ("chord_vector", self.chord_vector),
            ("normal_vector", self.normal_vector),
        ):
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values.")

    @property
    def node_count(self) -> int:
        return int(self.shape_xyz.shape[0])


@dataclass
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
    fluid_density: float = 1.225
    fluid_dynamicviscosity: float = 1.81e-5
    rotation_axis: Optional[np.ndarray] = None
    rotation_axis_offset: Optional[np.ndarray] = None
    inflow_vec: Optional[np.ndarray] = None
    plot_cycle: Optional[List[str]] = None
    azimuths: Optional[np.ndarray] = None
    inflow_speeds: Optional[np.ndarray] = None
    output_time: Optional[np.ndarray] = None
    freq_min: float = 0.0
    freq_max: float = float('inf')
    airfoil_folder: Optional[str] = None
    n_harmonic: int = 5
    amplitude_coeff_cutoff: float = 0.01
    n_freq_depth: int = 3
    output_azimuth_vinf: Tuple[float, float] = (5.0, 2.0)
    # Azimuth already applied to the supplied global section bases. Subsequent
    # controller rotations are relative to this angle, avoiding double rotation.
    geometry_reference_azimuth_deg: float = 0.0

    def __post_init__(self):
        self.fluid_density = float(self.fluid_density)
        self.fluid_dynamicviscosity = float(self.fluid_dynamicviscosity)
        self.rotation_axis = _vector3(self.rotation_axis, [0.0, 0.0, 1.0], "rotation_axis")
        self.rotation_axis_offset = _vector3(
            self.rotation_axis_offset,
            [0.0, 0.0, 0.0],
            "rotation_axis_offset",
        )
        self.inflow_vec = _vector3(self.inflow_vec, [1.0, 0.0, 0.0], "inflow_vec")
        
        if self.plot_cycle is None:
            self.plot_cycle = ["#348ABD", "#A60628", "#009E73", "#7A68A6", "#D55E00", "#CC79A7"]
        else:
            self.plot_cycle = list(self.plot_cycle)
            
        if self.azimuths is None:
            self.azimuths = np.arange(0, 360, 5, dtype=float)
        else:
            self.azimuths = _array1d(self.azimuths, "azimuths")
            
        if self.inflow_speeds is None:
            self.inflow_speeds = np.arange(2.0, 11.0, 1.0, dtype=float)
        else:
            self.inflow_speeds = _array1d(self.inflow_speeds, "inflow_speeds")
            
        if self.output_time is None:
            self.output_time = np.arange(0.0, 10.001, 0.001, dtype=float)
        else:
            self.output_time = _array1d(self.output_time, "output_time")
            
        self.freq_min = float(self.freq_min)
        self.freq_max = float(self.freq_max)
        
        if self.airfoil_folder is None:
            module_path = os.path.dirname(os.path.abspath(__file__))
            self.airfoil_folder = os.path.join(module_path, "airfoils/")
        else:
            self.airfoil_folder = str(self.airfoil_folder)
            
        self.n_harmonic = int(self.n_harmonic)
        self.amplitude_coeff_cutoff = float(self.amplitude_coeff_cutoff)
        self.n_freq_depth = int(self.n_freq_depth)
        output_pair = tuple(float(value) for value in self.output_azimuth_vinf)
        if len(output_pair) != 2:
            raise ValueError("output_azimuth_vinf must contain exactly two values.")
        self.output_azimuth_vinf = output_pair


@dataclass
class VorLapAnalysisResult:
    """Offline GUI analysis output with dict-compatible access for existing plots."""

    percdiff_matrix: np.ndarray
    percdiff_info: Any
    total_global_force_vector: np.ndarray
    total_global_moment_vector: np.ndarray
    global_force_vector_nodes_selected_case: np.ndarray
    global_force_vector_nodes: np.ndarray
    inflow_time: np.ndarray
    time_varying_total_global_force_vector: np.ndarray
    time_varying_total_global_moment_vector: np.ndarray
    qblade_node_ids: Optional[Tuple[str, ...]]
    qblade_sim_path: str
    viv_params: VIV_Params
    components: Tuple[Component, ...] = ()
    warnings: Tuple[str, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    execution_time_s: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    _MAPPING_KEYS: ClassVar[Tuple[str, ...]] = (
        "percdiff_matrix",
        "percdiff_info",
        "total_global_force_vector",
        "total_global_moment_vector",
        "global_force_vector_nodes_selected_case",
        "global_force_vector_nodes",
        "inflow_time",
        "time_varying_total_global_force_vector",
        "time_varying_total_global_moment_vector",
        "qblade_node_ids",
        "qblade_sim_path",
        "viv_params",
        "components",
        "warnings",
        "artifacts",
        "execution_time_s",
        "metadata",
    )

    def __post_init__(self):
        self.qblade_node_ids = None if self.qblade_node_ids is None else tuple(self.qblade_node_ids)
        self.components = tuple(self.components)
        self.warnings = tuple(str(message) for message in self.warnings)
        self.artifacts = {str(key): str(value) for key, value in self.artifacts.items()}
        if self.execution_time_s is not None:
            self.execution_time_s = float(self.execution_time_s)
        self.metadata = {str(key): value for key, value in self.metadata.items()}

    def as_dict(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in self._MAPPING_KEYS}

    def get(self, key: str, default: object = None) -> object:
        return self.as_dict().get(key, default)

    def __getitem__(self, key: str) -> object:
        return self.as_dict()[key]

    def __contains__(self, key: object) -> bool:
        return key in self._MAPPING_KEYS


@dataclass
class InflowTimeSeries:
    """
    Time-varying inflow profile used for nonstationary force reconstruction.

    Attributes:
        time (np.ndarray): Time stamps [s], strictly increasing.
        inflow_speeds (np.ndarray): Inflow speed magnitude [m/s] per time sample.
        inflow_directions (np.ndarray): Unit inflow direction vectors [ntime x 3].
    """

    time: np.ndarray
    inflow_speeds: np.ndarray
    inflow_directions: np.ndarray

    def __post_init__(self):
        self.time = _array1d(self.time, "time")
        self.inflow_speeds = _array1d(self.inflow_speeds, "inflow_speeds")
        self.inflow_directions = np.asarray(self.inflow_directions, dtype=float)

        if self.time.size < 2:
            raise ValueError("Inflow profile must contain at least two time samples.")
        if self.inflow_speeds.shape[0] != self.time.shape[0]:
            raise ValueError("inflow_speeds length must match time length.")
        if self.inflow_directions.shape != (self.time.shape[0], 3):
            raise ValueError("inflow_directions must have shape [ntime, 3].")
        if np.any(~np.isfinite(self.time)) or np.any(~np.isfinite(self.inflow_speeds)) or np.any(~np.isfinite(self.inflow_directions)):
            raise ValueError("Inflow profile contains non-finite values.")

        dt = np.diff(self.time)
        if np.any(dt <= 0.0):
            raise ValueError("Inflow profile time values must be strictly increasing.")

        if np.any(self.inflow_speeds < 0.0):
            raise ValueError("Inflow speeds must be non-negative.")

        norms = np.linalg.norm(self.inflow_directions, axis=1)
        if np.any(norms <= 1.0e-12):
            raise ValueError("Inflow direction vectors must have non-zero magnitude.")
        self.inflow_directions = self.inflow_directions / norms[:, None]
