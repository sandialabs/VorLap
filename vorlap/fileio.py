"""
File input/output operations for the VorLap package.
"""

import os
import numpy as np
import pandas as pd
import h5py
from typing import List
import warnings
from .structs import AirfoilFFT, Component


def load_components_from_csv(dir_path: str) -> List[Component]:
    """
    Loads all component geometry and metadata from CSV files in the given directory.

    Args:
        dir_path: Path to a directory containing CSV files. Each file must follow a two-header format.

    Returns:
        List of parsed Component objects containing all geometry and configuration data.

    Expected CSV Format:
        1. First data row contains: `id`, `translation_x`, `translation_y`, `translation_z`, `rotation_x`, `rotation_y`, `rotation_z`
        2. Second header row: column names for vectors — must include `x`, `y`, `z`, `chord`, `twist`, `thickness`, and optional `airfoil_id`
        3. Remaining rows: vector data for each blade segment or shape point

    Notes:
        - If `airfoil_id` is missing, `"default"` will be used for all segments in that component
        - All transformations are centered at the origin and adjusted by top-level translation/rotation
        - All components are assumed to have the span oriented in the z-direction
    """
    import glob
    
    files = glob.glob(os.path.join(dir_path, "*.csv"))
    files.sort()  # Sort files for consistent ordering
    components = []
    
    for file in files:
        # Read the raw CSV data
        with open(file, 'r') as f:
            lines = f.readlines()
        
        # Extract top-level metadata (row 2)
        metadata = lines[1].strip().split(',')
        id_str = metadata[0]
        tx = float(metadata[1])
        ty = float(metadata[2])
        tz = float(metadata[3])
        rx = float(metadata[4])
        ry = float(metadata[5])
        rz = float(metadata[6])
        pitch = float(metadata[7])
        
        # Get column names (row 3)
        colnames = [col.strip() for col in lines[2].strip().split(',')]
        
        # Read the data rows (from row 4 onwards)
        data = []
        for line in lines[3:]:
            if line.strip():  # Skip empty lines
                data.append(line.strip().split(','))
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=colnames)
        
        # Extract vectors
        xyz = np.column_stack([
            df['x'].astype(float).values,
            df['y'].astype(float).values,
            df['z'].astype(float).values
        ])
        
        chord = df['chord'].astype(float).values
        twist = df['twist'].astype(float).values
        thickness = df['thickness'].astype(float).values
        offset = df['offset'].astype(float).values
        
        # Handle optional airfoil_id column
        if 'airfoil_id' in df.columns:
            airfoil_ids = df['airfoil_id'].values.tolist()
        else:
            airfoil_ids = ['default'] * len(df)
        
        # Create placeholders for vectors that will be filled later
        chord_vec = np.zeros((xyz.shape[0], 3))
        norm_vec = np.zeros((xyz.shape[0], 3))
        xyz_global = np.zeros_like(xyz)
        
        # Create and add the component
        component = Component(
            id=id_str,
            translation=np.array([tx, ty, tz]),
            rotation=np.array([rx, ry, rz]),
            pitch=np.array([pitch]),
            shape_xyz=xyz,
            shape_xyz_global=xyz_global,
            chord=chord,
            twist=twist,
            thickness=thickness,
            offset=offset,
            airfoil_ids=airfoil_ids,
            chord_vector=chord_vec,
            normal_vector=norm_vec
        )
        
        components.append(component)
    
    return components


def load_airfoil_fft(path: str) -> AirfoilFFT:
    """
    Loads a processed airfoil unsteady FFT dataset from an HDF5 file.

    Args:
        path: Path to the HDF5 file containing the airfoil FFT data.

    Returns:
        AirfoilFFT object containing the loaded data.

    Expected HDF5 File Format:
        The file must contain the following datasets:
        - `Airfoilname` :: String — Name of the airfoil (e.g., "NACA0012")
        - `Re` :: Vector{Float64} — Reynolds number values (assumed constant across all entries)
        - `Thickness` :: Vector{Float64} — Thickness ratio(s) used
        - `AOA` :: Vector{Float64} — Angle of attack values in degrees
        - `CL_ST`, `CD_ST`, `CM_ST`, `CF_ST` :: 3D Arrays [Re x AOA x freq] — Strouhal numbers for each force/moment
        - `CL_Amp`, `CD_Amp`, `CM_Amp`, `CF_Amp` :: 3D Arrays [Re x AOA x freq] — FFT amplitudes for lift, drag, moment, and combined force
        - `CL_Pha`, `CD_Pha`, `CM_Pha`, `CF_Pha` :: 3D Arrays [Re x AOA x freq] — FFT phases in radians for each quantity

    Assumptions:
        - All arrays must share dimensions [Re, AOA, NFreq], where the frequency dimension is sorted by the amplitude
        - Phase data is in radians.
        - Struhaul data represents unsteady aerodynamics due to vortex shedding.
        - No ragged or missing data is allowed.
    """
    with h5py.File(path, 'r') as h5:
        name = h5['Airfoilname'][()] if 'Airfoilname' in h5 else os.path.basename(path)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
            
        Re = h5['Re'][()]
        Thickness = h5['Thickness'][()]
        AOA = h5['AOA'][()]
        
        CL_ST = h5['CL_ST'][()]
        CD_ST = h5['CD_ST'][()]
        CM_ST = h5['CM_ST'][()]
        CF_ST = h5['CF_ST'][()]
        
        CL_Amp = h5['CL_Amp'][()]
        CD_Amp = h5['CD_Amp'][()]
        CM_Amp = h5['CM_Amp'][()]
        CF_Amp = h5['CF_Amp'][()]
        
        CL_Pha = h5['CL_Pha'][()]
        CD_Pha = h5['CD_Pha'][()]
        CM_Pha = h5['CM_Pha'][()]
        CF_Pha = h5['CF_Pha'][()]
        
        # Check and fix dimension mismatch
        expected_shape = (len(Re), len(AOA), CL_ST.shape[-1])
        
        def fix_dimensions(arr, name):
            """Fix dimension mismatch in FFT arrays"""
            if arr.shape != expected_shape:
                # Common issue: arrays stored as [freq, AOA, Re] instead of [Re, AOA, freq]
                if arr.shape == (arr.shape[0], len(AOA), len(Re)):
                    warnings.warn(f"Transposing {name} from shape {arr.shape} to {expected_shape}")
                    return np.transpose(arr, (2, 1, 0))  # [freq, AOA, Re] -> [Re, AOA, freq]
                else:
                    warnings.warn(f"Unexpected shape for {name}: {arr.shape}, expected {expected_shape}")
                    # Try to reshape if possible
                    if arr.size == np.prod(expected_shape):
                        return arr.reshape(expected_shape)
                    else:
                        raise ValueError(f"Cannot fix dimension mismatch for {name}: {arr.shape} vs {expected_shape}")
            return arr
        
        CL_ST = fix_dimensions(CL_ST, "CL_ST")
        CD_ST = fix_dimensions(CD_ST, "CD_ST")
        CM_ST = fix_dimensions(CM_ST, "CM_ST")
        CF_ST = fix_dimensions(CF_ST, "CF_ST")
        
        CL_Amp = fix_dimensions(CL_Amp, "CL_Amp")
        CD_Amp = fix_dimensions(CD_Amp, "CD_Amp")
        CM_Amp = fix_dimensions(CM_Amp, "CM_Amp")
        CF_Amp = fix_dimensions(CF_Amp, "CF_Amp")
        
        CL_Pha = fix_dimensions(CL_Pha, "CL_Pha")
        CD_Pha = fix_dimensions(CD_Pha, "CD_Pha")
        CM_Pha = fix_dimensions(CM_Pha, "CM_Pha")
        CF_Pha = fix_dimensions(CF_Pha, "CF_Pha")
        
        return AirfoilFFT(
            name=name,
            Re=Re,
            AOA=AOA,
            Thickness=Thickness[0] if isinstance(Thickness, np.ndarray) and len(Thickness) > 0 else Thickness,
            CL_ST=CL_ST,
            CD_ST=CD_ST,
            CM_ST=CM_ST,
            CF_ST=CF_ST,
            CL_Amp=CL_Amp,
            CD_Amp=CD_Amp,
            CM_Amp=CM_Amp,
            CF_Amp=CF_Amp,
            CL_Pha=CL_Pha,
            CD_Pha=CD_Pha,
            CM_Pha=CM_Pha,
            CF_Pha=CF_Pha
        )


def load_airfoil_coords(afpath: str = "") -> np.ndarray:
    """
    Loads an airfoil shape from a 2-column text file (x, z), normalized to unit chord length.
    If no file is specified, or if loading fails, returns a built-in 200-point Clark Y airfoil shape.

    Args:
        afpath: Optional path to a text file with two columns: x and z coordinates.

    Returns:
        xy: Nx2 matrix of normalized (x, y) coordinates representing the airfoil surface.

    Notes:
        - If loading from file, x-coordinates are normalized to span [0, 1].
        - The default fallback airfoil is a symmetric approximation of the Clark Y shape.
        - This airfoil is primarily used for visualization, not aerodynamic calculations.
    """
    if afpath and os.path.isfile(afpath):
        try:
            xy = np.loadtxt(afpath, delimiter=',')
            xy[:, 0] -= np.min(xy[:, 0])
            xy[:, 0] /= np.max(xy[:, 0])
            xy[:, 1] /= (np.max(xy[:, 1]) - np.min(xy[:, 1]))
            return xy
        except Exception as e:
            warnings.warn(f"Could not load airfoil file: {e}. Falling back to default Clark Y profile for plotting.")
    else:
        warnings.warn("Could not load airfoil file used for plotting. Falling back to default Clark Y profile for plotting.")
    
    # Fallback Clark Y airfoil coordinates (from airfoiltools.com)
    xy = np.array([
        [1.0000000, 0.0],
        [0.9900000, 0.0029690],
        [0.9800000, 0.0053335],
        [0.9700000, 0.0076868],
        [0.9600000, 0.0100232],
        [0.9400000, 0.0146239],
        [0.9200000, 0.0191156],
        [0.9000000, 0.0235025],
        [0.8800000, 0.0277891],
        [0.8600000, 0.0319740],
        [0.8400000, 0.0360536],
        [0.8200000, 0.0400245],
        [0.8000000, 0.0438836],
        [0.7800000, 0.0476281],
        [0.7600000, 0.0512565],
        [0.7400000, 0.0547675],
        [0.7200000, 0.0581599],
        [0.7000000, 0.0614329],
        [0.6800000, 0.0645843],
        [0.6600000, 0.0676046],
        [0.6400000, 0.0704822],
        [0.6200000, 0.0732055],
        [0.6000000, 0.0757633],
        [0.5800000, 0.0781451],
        [0.5600000, 0.0803480],
        [0.5400000, 0.0823712],
        [0.5200000, 0.0842145],
        [0.5000000, 0.0858772],
        [0.4800000, 0.0873572],
        [0.4600000, 0.0886427],
        [0.4400000, 0.0897175],
        [0.4200000, 0.0905657],
        [0.4000000, 0.0911712],
        [0.3800000, 0.0915212],
        [0.3600000, 0.0916266],
        [0.3400000, 0.0915079],
        [0.3200000, 0.0911857],
        [0.3000000, 0.0906804],
        [0.2800000, 0.0900016],
        [0.2600000, 0.0890840],
        [0.2400000, 0.0878308],
        [0.2200000, 0.0861433],
        [0.2000000, 0.0839202],
        [0.1800000, 0.0810687],
        [0.1600000, 0.0775707],
        [0.1400000, 0.0734360],
        [0.1200000, 0.0686204],
        [0.1000000, 0.0629981],
        [0.0800000, 0.0564308],
        [0.0600000, 0.0487571],
        [0.0500000, 0.0442753],
        [0.0400000, 0.0391283],
        [0.0300000, 0.0330215],
        [0.0200000, 0.0253735],
        [0.0120000, 0.0178581],
        [0.0080000, 0.0137350],
        [0.0040000, 0.0089238],
        [0.0020000, 0.0058025],
        [0.0010000, 0.0037271],
        [0.0005000, 0.0023390],
        [0.0000000, 0.0000000],
        [0.0005000, -0.0046700],
        [0.0010000, -0.0059418],
        [0.0020000, -0.0078113],
        [0.0040000, -0.0105126],
        [0.0080000, -0.0142862],
        [0.0120000, -0.0169733],
        [0.0200000, -0.0202723],
        [0.0300000, -0.0226056],
        [0.0400000, -0.0245211],
        [0.0500000, -0.0260452],
        [0.0600000, -0.0271277],
        [0.0800000, -0.0284595],
        [0.1000000, -0.0293786],
        [0.1200000, -0.0299633],
        [0.1400000, -0.0302404],
        [0.1600000, -0.0302546],
        [0.1800000, -0.0300490],
        [0.2000000, -0.0296656],
        [0.2200000, -0.0291445],
        [0.2400000, -0.0285181],
        [0.2600000, -0.0278164],
        [0.2800000, -0.0270696],
        [0.3000000, -0.0263079],
        [0.3200000, -0.0255565],
        [0.3400000, -0.0248176],
        [0.3600000, -0.0240870],
        [0.3800000, -0.0233606],
        [0.4000000, -0.0226341],
        [0.4200000, -0.0219042],
        [0.4400000, -0.0211708],
        [0.4600000, -0.0204353],
        [0.4800000, -0.0196986],
        [0.5000000, -0.0189619],
        [0.5200000, -0.0182262],
        [0.5400000, -0.0174914],
        [0.5600000, -0.0167572],
        [0.5800000, -0.0160232],
        [0.6000000, -0.0152893],
        [0.6200000, -0.0145551],
        [0.6400000, -0.0138207],
        [0.6600000, -0.0130862],
        [0.6800000, -0.0123515],
        [0.7000000, -0.0116169],
        [0.7200000, -0.0108823],
        [0.7400000, -0.0101478],
        [0.7600000, -0.0094133],
        [0.7800000, -0.0086788],
        [0.8000000, -0.0079443],
        [0.8200000, -0.0072098],
        [0.8400000, -0.0064753],
        [0.8600000, -0.0057408],
        [0.8800000, -0.0050063],
        [0.9000000, -0.0042718],
        [0.9200000, -0.0035373],
        [0.9400000, -0.0028028],
        [0.9600000, -0.0020683],
        [0.9700000, -0.0017011],
        [0.9800000, -0.0013339],
        [0.9900000, -0.0009666],
        [1.0, 0]
    ])
    
    xy[:, 0] -= np.min(xy[:, 0])
    xy[:, 0] /= np.max(xy[:, 0])
    xy[:, 1] /= (np.max(xy[:, 1]) - np.min(xy[:, 1]))
    
    return xy


def write_force_time_series(filename: str, output_time: np.ndarray, global_force_vector_nodes: np.ndarray) -> None:
    """
    Writes force time series data to a CSV file.

    Args:
        filename: Path to the output CSV file.
        output_time: Vector of time points.
        global_force_vector_nodes: Array of force vectors for each node at each time point.

    Returns:
        None
    """
    ntime, _, nnodes = global_force_vector_nodes.shape
    
    with open(filename, "w") as f:
        # Write header
        header = ["time"]
        for n in range(nnodes):
            header.extend([f"node{n+1}x", f"node{n+1}y", f"node{n+1}z"])
        f.write(", ".join(header) + "\n")
        
        # Write each time row
        for t in range(ntime):
            row = [str(output_time[t])]
            for n in range(nnodes):
                fx = global_force_vector_nodes[t, 0, n]
                fy = global_force_vector_nodes[t, 1, n]
                fz = global_force_vector_nodes[t, 2, n]
                row.extend([str(fx), str(fy), str(fz)])
            f.write(", ".join(row) + "\n")
