"""Minimal VorLap example using a time-varying inflow profile CSV."""

from pathlib import Path

import numpy as np

from vorlap.computations import compute_time_varying_force_history_optimized
from vorlap.fileio import load_inflow_time_series
from vorlap.structs import AirfoilFFT, Component, VIV_Params


def build_airfoil_fft() -> AirfoilFFT:
    re_grid = np.array([1.0, 2.0], dtype=float)
    aoa_grid = np.array([-10.0, 10.0], dtype=float)
    shape = (2, 2, 2)

    zeros = np.zeros(shape)
    cl_amp = zeros.copy()
    cd_amp = zeros.copy()
    cf_amp = zeros.copy()
    cl_amp[:, :, 0] = 1.0
    cd_amp[:, :, 0] = 2.0
    cl_amp[:, :, 1] = 0.2
    cd_amp[:, :, 1] = 0.1
    cf_amp[:, :, 1] = 0.1

    st = zeros.copy()
    st[:, :, 1] = 0.5

    return AirfoilFFT(
        name="default",
        Re=re_grid,
        AOA=aoa_grid,
        Thickness=0.12,
        CL_ST=st.copy(),
        CD_ST=st.copy(),
        CM_ST=st.copy(),
        CF_ST=st.copy(),
        CL_Amp=cl_amp,
        CD_Amp=cd_amp,
        CM_Amp=zeros.copy(),
        CF_Amp=cf_amp,
        CL_Pha=zeros.copy(),
        CD_Pha=zeros.copy(),
        CM_Pha=zeros.copy(),
        CF_Pha=zeros.copy(),
    )


def build_component() -> Component:
    shape_xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=float)
    return Component(
        id="blade",
        translation=np.zeros(3),
        rotation=np.zeros(3),
        pitch=np.array([0.0]),
        shape_xyz=shape_xyz,
        shape_xyz_global=shape_xyz.copy(),
        chord=np.ones(2),
        twist=np.zeros(2),
        thickness=np.ones(2) * 0.12,
        offset=np.zeros(2),
        airfoil_ids=["default", "default"],
        chord_vector=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        normal_vector=np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
    )


def run(profile_csv: str = ""):
    repo = Path(__file__).resolve().parents[1]
    profile_path = Path(profile_csv) if profile_csv else (repo / "data" / "inflow_profile.csv")
    inflow_profile = load_inflow_time_series(str(profile_path))

    component = build_component()
    affts = {"default": build_airfoil_fft()}
    viv = VIV_Params(
        fluid_density=1.0,
        fluid_dynamicviscosity=1.0,
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        rotation_axis_offset=np.array([0.0, 0.0, 0.0]),
        inflow_vec=np.array([1.0, 0.0, 0.0]),
        azimuths=np.array([0.0]),
        inflow_speeds=np.array([2.0]),
        output_time=np.array([0.0, 0.25]),
        output_azimuth_vinf=(0.0, 2.0),
        n_freq_depth=2,
        amplitude_coeff_cutoff=0.05,
    )

    time, total_force, total_moment, node_forces = compute_time_varying_force_history_optimized(
        [component],
        affts,
        viv,
        inflow_profile,
        azimuth_deg=0.0,
        smoothing_cycles=0.25,
    )

    print(f"Inflow profile file: {profile_path}")
    print(f"Time samples: {time.size}")
    print("Total force at first sample:", total_force[0, :])
    print("Total moment at first sample:", total_moment[0, :])
    print("Node 1 x-force (first 5 samples):", node_forces[:5, 0, 0])


if __name__ == "__main__":
    run()
