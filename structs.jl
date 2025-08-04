"""
    AirfoilFFT

Holds multidimensional FFT data for unsteady aerodynamic forces and moments
as a function of Reynolds number and angle of attack.

# Fields
- `name::String`: Airfoil identifier name
- `Re::Vector{Float64}`: Reynolds number axis (1D)
- `AOA::Vector{Float64}`: Angle of attack axis in degrees (1D)
- `Thickness::Vector{Float64}`: Relative airfoil thicknesses (1D)

- `CL_ST::Array{Float64,3}`: Strouhal numbers for lift coefficient [Re × AOA × freq]
- `CD_ST::Array{Float64,3}`: Strouhal numbers for drag coefficient
- `CM_ST::Array{Float64,3}`: Strouhal numbers for moment coefficient
- `CF_ST::Array{Float64,3}`: Strouhal numbers for total force coefficient magnitude

- `CL_Amp::Array{Float64,3}`: FFT amplitudes of CL [Re × AOA × freq]
- `CD_Amp::Array{Float64,3}`: FFT amplitudes of CD
- `CM_Amp::Array{Float64,3}`: FFT amplitudes of CM
- `CF_Amp::Array{Float64,3}`: FFT amplitudes of combined force magnitude

- `CL_Pha::Array{Float64,3}`: FFT phases of CL (radians)
- `CD_Pha::Array{Float64,3}`: FFT phases of CD
- `CM_Pha::Array{Float64,3}`: FFT phases of CM
- `CF_Pha::Array{Float64,3}`: FFT phases of combined force magnitude

# Notes
- The frequency axis is implicit in the third dimension and must be consistent across all arrays.
- All arrays must share shape `[length(Re), length(AOA), length(freq)]`.
- Phases are in radians, and FFT data assumes periodic unsteady oscillations (e.g., vortex shedding).
"""
struct AirfoilFFT
    name::String
    Re::Vector{Float64}
    AOA::Vector{Float64}
    Thickness::Float64

    CL_ST::Array{Float64, 3}
    CD_ST::Array{Float64, 3}
    CM_ST::Array{Float64, 3}
    CF_ST::Array{Float64, 3}

    CL_Amp::Array{Float64, 3}
    CD_Amp::Array{Float64, 3}
    CM_Amp::Array{Float64, 3}
    CF_Amp::Array{Float64, 3}

    CL_Pha::Array{Float64, 3}
    CD_Pha::Array{Float64, 3}
    CM_Pha::Array{Float64, 3}
    CF_Pha::Array{Float64, 3}
end


"""
    Component

Represents a single physical component in the full rotating structure. Each component includes
a global transformation and local shape definition, segmented for per-section force calculations.

# Fields
- `id::String`: Identifier name for the component
- `translation::Vector{Float64}`: Translation vector applied to the entire component
- `rotation::Vector{Float64}`: Euler angle rotation vector [deg] around X, Y, Z axes
- `shape_xyz::Matrix{Float64}`: Nx3 matrix of local segment positions (untransformed)
- `pitch::Vector{Float64}`: Pitch angle for the segment [deg], vectorized to avoid mutability
- `chord::Vector{Float64}`: Chord length per segment
- `twist::Vector{Float64}`: Twist angle per segment [deg]
- `thickness::Vector{Float64}`: Relative thickness per segment (scales airfoil height), fraction of chord
- `airfoil_ids::Vector{String}`: Airfoil data identifier for each segment (defaults to \"default\" if missing)
"""
struct Component
    id::String
    translation::Vector{Float64}
    rotation::Vector{Float64}
    pitch::Vector{Float64}
    shape_xyz::Matrix{Float64}
    shape_xyz_global::Matrix{Float64}
    chord::Vector{Float64}
    twist::Vector{Float64}
    thickness::Vector{Float64}
    offset::Vector{Float64}
    airfoil_ids::Vector{String}
    chord_vector::Matrix{Float64}
    normal_vector::Matrix{Float64}
end
# Component(id,translation,rotation,pitch,shape_xyz,chord,twist,thickness,offset,airfoil_ids) = Component(id,translation,rotation,pitch,shape_xyz,chord,twist,thickness,offset,airfoil_ids,nothing)
# === Top-Level Configuration Struct ===
"""
    VIVInputs

Encapsulates all top-level user-defined configuration inputs required for vortex-induced vibration analysis.

# Fields
- `fluid_density`: Air density [kg/m³]
- `fluid_dynamicviscosity`: Dynamic viscosity of air [Pa·s]
- `rotation_axis`: Axis of rotation as a 3-element vector
- `rotation_axis_offset`: Origin of the rotation axis (used in torque calculations and visualization)
- `inflow_vec`: Direction of inflow velocity (typically [1, 0, 0] for +X)
- `plot_cycle`: List of hex colors used to differentiate components in visualization
- `azimuths`: Azimuthal angles [deg] swept by the rotor or structure
- `inflow_speeds`: Freestream inflow speeds [m/s]
- `freq_min`: Minimum frequency [Hz] to consider in overlap comparison
- `freq_max`: Maximum frequency [Hz] to consider in overlap comparison
- `airfoil_folder`: path to the airfoil folder
- `n_harmonic`: number of harmonics to check against
- `amplitude_coeff_cutoff`: lower threshold on what amplitudes (normalized by q=0.5*rho*Vinf^2*chord*1) are of interest for inducing modal frequencies
- `n_freq_depth`: how deep to go in the Strouhaul number spectrum (which is sorted by amplitude, above the cutoff frequency which is 10hz in the process_naluout script), since a given airfoil may have multiple shedding frequencies not just the dominant one
"""
struct VIV_Params
    fluid_density::Float64
    fluid_dynamicviscosity::Float64
    rotation_axis::Vector{Float64}
    rotation_axis_offset::Vector{Float64}
    inflow_vec::Vector{Float64}
    plot_cycle::Vector{String}
    azimuths::Vector{Float64}
    inflow_speeds::Vector{Float64}
    output_time::Vector{Float64}
    freq_min::Float64
    freq_max::Float64
    airfoil_folder::String
    n_harmonic::Int64
    amplitude_coeff_cutoff::Float64
    n_freq_depth::Int64
    output_azimuth_vinf
end

function VIV_Params(;
    fluid_density=1.225,
    fluid_dynamicviscosity=1.81e-5,
    rotation_axis=[0.0, 0.0, 1.0],
    rotation_axis_offset=[0.0, 0.0, 0.0],
    inflow_vec=[1.0, 0.0, 0.0],
    plot_cycle=["#348ABD", "#A60628", "#009E73", "#7A68A6", "#D55E00", "#CC79A7"],
    azimuths=0:5:355,
    inflow_speeds=2.0:1.0:10.0,
    output_time = collect(0.0:0.001:10.0), #s
    freq_min=0.0,
    freq_max=Inf,
    airfoil_folder="$path/airfoils/",
    n_harmonic=5,
    amplitude_coeff_cutoff=0.01,
    n_freq_depth = 3,
    output_azimuth_vinf = (5.0, 2.0)) #used to limit the case where the relatively expensive output signal reconstruction is done)

    return VIV_Params(fluid_density,fluid_dynamicviscosity,rotation_axis,rotation_axis_offset,inflow_vec,plot_cycle,azimuths,inflow_speeds,output_time,freq_min,freq_max,airfoil_folder,n_harmonic,amplitude_coeff_cutoff,n_freq_depth,output_azimuth_vinf)
end