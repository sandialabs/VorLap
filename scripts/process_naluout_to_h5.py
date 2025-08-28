"""
Process NALU output data to HDF5 format for use with VorLap.

This script processes data from .dat files containing time-series data for forces and moments,
computes FFT for lift, drag, moment, and combined force coefficients, calculates Strouhal numbers,
sorts the data by amplitude, and writes the processed data to an HDF5 file.
"""

import os
import numpy as np
import h5py
import warnings
# Use numpy's FFT functions
from numpy.fft import fft
import glob
import argparse


def compute_fft(signal, dt, chord, aoa, Vinf):
    """
    Compute the FFT of a signal.
    
    Args:
        signal: Time-domain signal
        dt: Time step
        chord: Chord length
        aoa: Angle of attack in degrees
        Vinf: Freestream velocity
        
    Returns:
        freqs: Frequency vector
        amps: Amplitude spectrum
        phases: Phase spectrum
        ST_sorted: Sorted Strouhal numbers
        amps_sorted: Sorted amplitudes
        phases_sorted: Sorted phases
    """
    N = len(signal)
    signal_used = signal  # No need to subtract mean, maximum difference in 2:end is 1.6e-16
    fft_vals = fft(signal_used)
    half_N = N // 2
    
    # One-sided frequency axis
    freqs = np.arange(half_N) / (N * dt)
    
    # Amplitude spectrum (one-sided)
    amps = np.abs(fft_vals[:half_N]) / N
    amps[1:half_N-1] *= 2  # Double non-DC, non-Nyquist components
    
    # Phase spectrum
    phases = np.angle(fft_vals[:half_N])
    
    # Strouhal number calculation
    # Sort by the largest to smallest amplitudes
    perm_peakamp = np.argsort(amps)[::-1]  # Largest to smallest
    freqs_sorted = freqs[perm_peakamp]
    amps_sorted = amps[perm_peakamp]
    phases_sorted = phases[perm_peakamp]
    STlength = chord * abs(np.sin(np.radians(aoa)))  # NOTE: ST seems to be dominated by the major axis length, not planform length
    ST_sorted = freqs_sorted * STlength / Vinf
    
    return freqs, amps, phases, ST_sorted, amps_sorted, phases_sorted


def process_dat_folder(dat_folder, h5_path, 
                      Vinf=2.0,
                      chord=1.0,
                      span=4.0,
                      fluid_density=1.2,
                      fluid_viscosity=9.0e-6,
                      NFreq=200,
                      minFreq=0.0,
                      genplots=True,
                      sampledT_startcutoff=70.0):
    """
    Process a folder of .dat files to create an HDF5 file with FFT data.
    
    Args:
        dat_folder: Path to folder containing .dat files
        h5_path: Path to output HDF5 file
        Vinf: Freestream velocity
        chord: Chord length
        span: Span length
        fluid_density: Fluid density
        fluid_viscosity: Fluid viscosity
        NFreq: Number of frequencies to store
        minFreq: Minimum frequency cutoff for Strouhal number calculation
        genplots: Whether to generate plots
        sampledT_startcutoff: Time cutoff for plotting
        
    Returns:
        None
    """
    # Find all .dat files in the folder
    files = glob.glob(os.path.join(dat_folder, "*.dat"))
    N_AOA = len(files)
    
    thickness = 0.0  # placeholder, is set elsewhere
    airfoil_name = "placeholder"
    
    # Calculate Reynolds number
    reynolds = fluid_density * Vinf * chord / fluid_viscosity
    
    # Figure out the number of AOAs
    Re = np.array([reynolds])  # TODO: other reynolds?
    NRe = len(Re)
    single_Re = False
    if NRe == 1:
        Re = np.array([Re[0], Re[0] + 1e-6])
        NRe = 2
        single_Re = True
    
    AOA = np.zeros(N_AOA)
    CL_ST = np.zeros((NRe, N_AOA, NFreq))
    CD_ST = np.zeros((NRe, N_AOA, NFreq))
    CM_ST = np.zeros((NRe, N_AOA, NFreq))
    CF_ST = np.zeros((NRe, N_AOA, NFreq))
    CL_Amp = np.zeros((NRe, N_AOA, NFreq))
    CD_Amp = np.zeros((NRe, N_AOA, NFreq))
    CM_Amp = np.zeros((NRe, N_AOA, NFreq))
    CF_Amp = np.zeros((NRe, N_AOA, NFreq))
    CL_Pha = np.zeros((NRe, N_AOA, NFreq))
    CD_Pha = np.zeros((NRe, N_AOA, NFreq))
    CM_Pha = np.zeros((NRe, N_AOA, NFreq))
    CF_Pha = np.zeros((NRe, N_AOA, NFreq))
    
    iRe = 0  # TODO: other Reynolds
    for iaoa, file in enumerate(files):
        print(f"Processing {file}")
        filename = os.path.basename(file).split('.')[0]
        AOA_str = ''.join(filter(lambda x: x.isdigit() or x == '-', filename.split('_')[-1]))
        AOA[iaoa] = float(AOA_str)
        airfoil_name = '_'.join(filename.split('_')[:-1])
        
        try:
            thickness_str = ''.join(filter(lambda x: x.isdigit() or x == '-', airfoil_name.split('_')[-1]))
            thickness = float(thickness_str) / 1000
        except:
            thickness = 0.0
        
        # Read data (skip headers)
        data = np.loadtxt(file, skiprows=1)
        timefull = data[:, 0]
        dt = timefull[1] - timefull[0]  # NOTE: assumes that the CFD solver uses a fixed time step
        fpx, fpy = data[:, 1], data[:, 2]
        fvx, fvy = data[:, 4], data[:, 5]
        mty = data[:, 8]
        
        q = 0.5 * fluid_density * Vinf**2 * chord * span
        CL_data = (fpy + fvy) / q  # pressure and viscous forces. NOTE: mesh x is assumed to be in the same direction as the flow, +x
        CD_data = (fpx + fvx) / q
        CF_data = np.sqrt(CD_data**2 + CL_data**2)
        CM_data = mty / q
        
        # Calculate the number of flow-throughs, and cut out the first
        time_flowthrough = chord / Vinf  # chord_m/m*s = second/chord
        total_flowthroughs = timefull[-1] / time_flowthrough  # seconds * chord/seconds = chords
        
        # Compute FFT
        freqs_cl, amps_cl, phases_cl, ST_sorted_cl, amps_sorted_cl, phases_sorted_cl = compute_fft(CL_data, dt, chord, AOA[iaoa], Vinf)
        freqs_cd, amps_cd, phases_cd, ST_sorted_cd, amps_sorted_cd, phases_sorted_cd = compute_fft(CD_data, dt, chord, AOA[iaoa], Vinf)
        freqs_cm, amps_cm, phases_cm, ST_sorted_cm, amps_sorted_cm, phases_sorted_cm = compute_fft(CM_data, dt, chord, AOA[iaoa], Vinf)
        freqs_cf, amps_cf, phases_cf, ST_sorted_cf, amps_sorted_cf, phases_sorted_cf = compute_fft(CF_data, dt, chord, AOA[iaoa], Vinf)
        
        if genplots:
            try:
                import matplotlib.pyplot as plt
                
                idx_start = max(1, int(round(sampledT_startcutoff / dt)))
                
                # Time plot
                plt.figure()
                plt.plot(timefull[idx_start:], CF_data[idx_start:])
                plt.xlabel('Time (s)')
                plt.ylabel('CF')
                plt.title(f'(CF), AOA: {AOA[iaoa]}')
                plt.legend(True)
                plt.show()
                
                # Frequency plot
                plt.figure()
                plt.plot(freqs_cf[1:NFreq], amps_cf[1:NFreq], 'x-')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Magnitude (CF)')
                plt.title(f'(CF), AOA: {AOA[iaoa]}')
                plt.legend(False)
                plt.show()
            except ImportError:
                warnings.warn("Matplotlib not available, skipping plots")
                genplots = False
        
        # Store Contents From File, Only Save Frequency up to NFreq
        if NFreq > len(ST_sorted_cl):
            warnings.warn("Number of frequencies exceeds those available")
            NFreq = len(ST_sorted_cl)
        
        CL_ST[iRe, iaoa, :NFreq] = ST_sorted_cl[:NFreq]
        CD_ST[iRe, iaoa, :NFreq] = ST_sorted_cd[:NFreq]
        CM_ST[iRe, iaoa, :NFreq] = ST_sorted_cm[:NFreq]
        CF_ST[iRe, iaoa, :NFreq] = ST_sorted_cf[:NFreq]
        CL_Amp[iRe, iaoa, :NFreq] = amps_sorted_cl[:NFreq]
        CD_Amp[iRe, iaoa, :NFreq] = amps_sorted_cd[:NFreq]
        CM_Amp[iRe, iaoa, :NFreq] = amps_sorted_cm[:NFreq]
        CF_Amp[iRe, iaoa, :NFreq] = amps_sorted_cf[:NFreq]
        CL_Pha[iRe, iaoa, :NFreq] = phases_sorted_cl[:NFreq]
        CD_Pha[iRe, iaoa, :NFreq] = phases_sorted_cd[:NFreq]
        CM_Pha[iRe, iaoa, :NFreq] = phases_sorted_cm[:NFreq]
        CF_Pha[iRe, iaoa, :NFreq] = phases_sorted_cf[:NFreq]
    
    if single_Re:
        CL_ST[1, :, :] = CL_ST[0, :, :]
        CD_ST[1, :, :] = CD_ST[0, :, :]
        CM_ST[1, :, :] = CM_ST[0, :, :]
        CF_ST[1, :, :] = CF_ST[0, :, :]
        CL_Amp[1, :, :] = CL_Amp[0, :, :]
        CD_Amp[1, :, :] = CD_Amp[0, :, :]
        CM_Amp[1, :, :] = CM_Amp[0, :, :]
        CF_Amp[1, :, :] = CF_Amp[0, :, :]
        CL_Pha[1, :, :] = CL_Pha[0, :, :]
        CD_Pha[1, :, :] = CD_Pha[0, :, :]
        CM_Pha[1, :, :] = CM_Pha[0, :, :]
        CF_Pha[1, :, :] = CF_Pha[0, :, :]
    
    # Sort the AOA vector and its dependencies
    aoa_sort_idx = np.argsort(AOA)
    
    AOA_sort = AOA[aoa_sort_idx]
    CL_ST_sort = CL_ST[:, aoa_sort_idx, :]
    CD_ST_sort = CD_ST[:, aoa_sort_idx, :]
    CM_ST_sort = CM_ST[:, aoa_sort_idx, :]
    CF_ST_sort = CF_ST[:, aoa_sort_idx, :]
    CL_Amp_sort = CL_Amp[:, aoa_sort_idx, :]
    CD_Amp_sort = CD_Amp[:, aoa_sort_idx, :]
    CM_Amp_sort = CM_Amp[:, aoa_sort_idx, :]
    CF_Amp_sort = CF_Amp[:, aoa_sort_idx, :]
    CL_Pha_sort = CL_Pha[:, aoa_sort_idx, :]
    CD_Pha_sort = CD_Pha[:, aoa_sort_idx, :]
    CM_Pha_sort = CM_Pha[:, aoa_sort_idx, :]
    CF_Pha_sort = CF_Pha[:, aoa_sort_idx, :]
    
    if genplots:
        try:
            import matplotlib.pyplot as plt
            
            # Plot ST vs AOA
            plt.figure()
            for ist in range(1, 22, 5):
                plt.plot(AOA_sort, CF_ST_sort[0, :, ist], 'x-', label=f'{ist}')
            plt.xlabel('AOA (deg)')
            plt.ylabel('ST (CF)')
            plt.title('ST')
            plt.legend()
            plt.show()
            
            # Plot Amp vs AOA
            plt.figure()
            for ist in range(1, 22, 5):
                plt.plot(AOA_sort, CF_Amp_sort[0, :, ist], 'x-', label=f'{ist}')
            plt.xlabel('AOA (deg)')
            plt.ylabel('Amp (CF)')
            plt.title('Amp')
            plt.legend()
            plt.show()
        except ImportError:
            warnings.warn("Matplotlib not available, skipping plots")
    
    # Write to HDF5 file
    with h5py.File(h5_path, 'w') as file:
        file.create_dataset('Airfoilname', data=airfoil_name)
        file.create_dataset('Re', data=Re)
        file.create_dataset('Thickness', data=thickness)
        file.create_dataset('AOA', data=AOA_sort)
        file.create_dataset('CL_ST', data=CL_ST_sort)
        file.create_dataset('CD_ST', data=CD_ST_sort)
        file.create_dataset('CM_ST', data=CM_ST_sort)
        file.create_dataset('CF_ST', data=CF_ST_sort)
        file.create_dataset('CL_Amp', data=CL_Amp_sort)
        file.create_dataset('CD_Amp', data=CD_Amp_sort)
        file.create_dataset('CM_Amp', data=CM_Amp_sort)
        file.create_dataset('CF_Amp', data=CF_Amp_sort)
        file.create_dataset('CL_Pha', data=CL_Pha_sort)
        file.create_dataset('CD_Pha', data=CD_Pha_sort)
        file.create_dataset('CM_Pha', data=CM_Pha_sort)
        file.create_dataset('CF_Pha', data=CF_Pha_sort)


if __name__ == "__main__":
    # Get the module path
    MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
    
    # Default parameters
    Vinf = 2.0
    chord = 1.0
    span = 4.0
    fluid_density = 1.2
    fluid_viscosity = 9.0e-6
    NFreq = 200
    minFreq = 0.0
    genplots = True
    sampledT_startcutoff = 70.0
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process NALU output data to HDF5 format for use with VorLap.')
    parser.add_argument('--dat_folder', type=str, default=os.path.join(MODULE_PATH, "airfoils/cylinder/data_files"),
                        help='Path to folder containing .dat files')
    parser.add_argument('--h5_path', type=str, default=os.path.join(MODULE_PATH, "airfoils/cylinder_fft.h5"),
                        help='Path to output HDF5 file')
    parser.add_argument('--Vinf', type=float, default=Vinf,
                        help='Freestream velocity')
    parser.add_argument('--chord', type=float, default=chord,
                        help='Chord length')
    parser.add_argument('--span', type=float, default=span,
                        help='Span length')
    parser.add_argument('--fluid_density', type=float, default=fluid_density,
                        help='Fluid density')
    parser.add_argument('--fluid_viscosity', type=float, default=fluid_viscosity,
                        help='Fluid viscosity')
    parser.add_argument('--NFreq', type=int, default=NFreq,
                        help='Number of frequencies to store')
    parser.add_argument('--minFreq', type=float, default=minFreq,
                        help='Minimum frequency cutoff for Strouhal number calculation')
    parser.add_argument('--genplots', type=bool, default=genplots,
                        help='Whether to generate plots')
    parser.add_argument('--sampledT_startcutoff', type=float, default=sampledT_startcutoff,
                        help='Time cutoff for plotting')
    
    args = parser.parse_args()
    
    # Process the data
    process_dat_folder(args.dat_folder, args.h5_path,
                      Vinf=args.Vinf,
                      chord=args.chord,
                      span=args.span,
                      fluid_density=args.fluid_density,
                      fluid_viscosity=args.fluid_viscosity,
                      NFreq=args.NFreq,
                      minFreq=args.minFreq,
                      genplots=args.genplots,
                      sampledT_startcutoff=args.sampledT_startcutoff)
