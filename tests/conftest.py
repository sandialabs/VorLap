import numpy as np

from vorlap.structs import AirfoilFFT, Component, VIV_Params


def make_constant_airfoil_fft(
    name: str = "default",
    cl_dc: float = 1.0,
    cd_dc: float = 2.0,
    cf_dc: float = 0.0,
    cf_h1_amp: float = 0.1,
    st_h1: float = 0.5,
    n_freq: int = 2,
) -> AirfoilFFT:
    """Create a small synthetic airfoil FFT dataset for deterministic tests."""
    re_grid = np.array([1.0, 3.0], dtype=float)
    aoa_grid = np.array([-10.0, 10.0], dtype=float)
    shape = (re_grid.size, aoa_grid.size, n_freq)

    def _zeros() -> np.ndarray:
        return np.zeros(shape, dtype=float)

    cl_st = _zeros()
    cd_st = _zeros()
    cm_st = _zeros()
    cf_st = _zeros()
    if n_freq > 1:
        cl_st[:, :, 1] = st_h1
        cd_st[:, :, 1] = st_h1
        cm_st[:, :, 1] = st_h1
        cf_st[:, :, 1] = st_h1

    cl_amp = _zeros()
    cd_amp = _zeros()
    cm_amp = _zeros()
    cf_amp = _zeros()
    cl_amp[:, :, 0] = cl_dc
    cd_amp[:, :, 0] = cd_dc
    cf_amp[:, :, 0] = cf_dc
    if n_freq > 1:
        cf_amp[:, :, 1] = cf_h1_amp

    cl_pha = _zeros()
    cd_pha = _zeros()
    cm_pha = _zeros()
    cf_pha = _zeros()

    return AirfoilFFT(
        name=name,
        Re=re_grid,
        AOA=aoa_grid,
        Thickness=0.12,
        CL_ST=cl_st,
        CD_ST=cd_st,
        CM_ST=cm_st,
        CF_ST=cf_st,
        CL_Amp=cl_amp,
        CD_Amp=cd_amp,
        CM_Amp=cm_amp,
        CF_Amp=cf_amp,
        CL_Pha=cl_pha,
        CD_Pha=cd_pha,
        CM_Pha=cm_pha,
        CF_Pha=cf_pha,
    )


def make_linear_airfoil_fft() -> AirfoilFFT:
    """Create an FFT grid where each value is linear in Re and AOA for interpolation tests."""
    re_grid = np.array([1.0, 3.0], dtype=float)
    aoa_grid = np.array([0.0, 10.0], dtype=float)
    n_freq = 2
    rr, aa = np.meshgrid(re_grid, aoa_grid, indexing="ij")
    base = rr + 2.0 * aa
    base = base[:, :, None]
    harmonic = (0.1 * rr - 0.05 * aa)[:, :, None]
    data = np.concatenate([base, harmonic], axis=2)

    zeros = np.zeros_like(data)
    return AirfoilFFT(
        name="linear",
        Re=re_grid,
        AOA=aoa_grid,
        Thickness=0.12,
        CL_ST=data,
        CD_ST=data + 1.0,
        CM_ST=data + 2.0,
        CF_ST=data + 3.0,
        CL_Amp=data + 4.0,
        CD_Amp=data + 5.0,
        CM_Amp=data + 6.0,
        CF_Amp=data + 7.0,
        CL_Pha=zeros + 0.01,
        CD_Pha=zeros + 0.02,
        CM_Pha=zeros + 0.03,
        CF_Pha=zeros + 0.04,
    )


def make_component(n_nodes: int = 2, span: float = 2.0, airfoil_id: str = "default") -> Component:
    """Create a straight component aligned with the z-axis."""
    z = np.linspace(0.0, span, n_nodes, dtype=float)
    shape_xyz = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
    chord = np.ones(n_nodes, dtype=float)
    twist = np.zeros(n_nodes, dtype=float)
    thickness = np.ones(n_nodes, dtype=float) * 0.12
    offset = np.zeros(n_nodes, dtype=float)
    airfoil_ids = [airfoil_id] * n_nodes
    chord_vector = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=float), (n_nodes, 1))
    normal_vector = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=float), (n_nodes, 1))

    return Component(
        id="blade",
        translation=np.zeros(3, dtype=float),
        rotation=np.zeros(3, dtype=float),
        pitch=np.array([0.0], dtype=float),
        shape_xyz=shape_xyz,
        shape_xyz_global=shape_xyz.copy(),
        chord=chord,
        twist=twist,
        thickness=thickness,
        offset=offset,
        airfoil_ids=airfoil_ids,
        chord_vector=chord_vector,
        normal_vector=normal_vector,
    )


def make_viv_params() -> VIV_Params:
    """Create deterministic VIV parameters for unit tests."""
    return VIV_Params(
        fluid_density=1.0,
        fluid_dynamicviscosity=1.0,
        rotation_axis=np.array([0.0, 0.0, 1.0], dtype=float),
        rotation_axis_offset=np.array([0.0, 0.0, 0.0], dtype=float),
        inflow_vec=np.array([1.0, 1.0, 0.0], dtype=float),
        azimuths=np.array([0.0], dtype=float),
        inflow_speeds=np.array([2.0], dtype=float),
        output_time=np.array([0.0, 0.25], dtype=float),
        output_azimuth_vinf=(0.0, 2.0),
        n_harmonic=2,
        amplitude_coeff_cutoff=0.05,
        n_freq_depth=2,
    )
