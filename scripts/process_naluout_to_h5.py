# build_airfoil_fft.py
import os
import re
import math
import numpy as np
import h5py
from numpy.fft import fft
from scipy.signal.windows import hann  # pip install scipy
import matplotlib.pyplot as plt

# ------------------------ config / knobs ------------------------
localpath = os.path.dirname(os.path.abspath(__file__))

# dat_folder points to AIRFOIL directory that contains RE* subfolders
dat_folder = os.path.join(localpath, "../airfoils", "randata", "NACA0018")   # CHANGED (root for RE* dirs)
h5_path    = os.path.join(localpath, "../airfoils", "NACA0018_fft.h5")

Vinf_fallback = 2.0   # used only if no RE subfolders are found
chord = 1.0
span  = 4.0
fluid_density   = 1.2
fluid_viscosity = 9.0e-06
NFreq   = 200
minFreq = 0.0
genplots = True
sampledT_startcutoff = 10.3

# make/ensure figs directory
figs_dir = os.path.join(localpath, "figs")
os.makedirs(figs_dir, exist_ok=True)

def _slug(s: str) -> str:
    """Safe filename piece."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))

def savefig_and_close(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)             # free memory right away


# ------------------------ helpers ------------------------
def parse_re_from_dirname(name: str) -> float:
    m = re.match(r"^RE(\d+)_([0-9]+)E(\d+)$", name)
    if not m:
        return float("nan")
    base, frac, exp = m.group(1), m.group(2), int(m.group(3))
    return float(f"{base}.{frac}e{exp}")


def dat_files_in(directory: str):
    """List .dat files in a directory, or in its ./data_files subdir if none at top level."""
    fs = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dat")]
    if not fs:
        d2 = os.path.join(directory, "data_files")
        if os.path.isdir(d2):
            fs = [os.path.join(d2, f) for f in os.listdir(d2) if f.endswith(".dat")]
    return sorted(fs)

def safe_load_dat(path: str):
    """Load .dat with one header row (skiprows=1). Returns np.ndarray or None."""
    try:
        arr = np.loadtxt(path, skiprows=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        print(f"Warning: Skipping {path}, unreadable or no data.")
        return None

def compute_fft(signal: np.ndarray, dt: float, chord: float, aoa_deg: float, Vinf: float):
    """
    Assumptions / conventions:
      - DC from mean(signal)
      - Demean, windowed PSD for power-based sorting
      - Unwindowed FFT for amplitudes/phases (cosine convention)
      - One-sided scaling: double bins 2..end-1 (exclude DC and Nyquist)
      - Sort non-DC bins by windowed per-bin power (descending)
      - Convert frequencies -> Strouhal via ST = f * (chord * |sin(aoa)|) / Vinf
    Returns:
      freqs_unsorted, amps_unsorted, phases_unsorted, power_pos, ST_sorted, amps_sorted, phases_sorted
    """
    N = signal.size
    fs = 1.0 / dt
    half_N = N // 2

    # DC and demean for spectral parts
    mean_amp = float(np.mean(signal))
    x = signal - mean_amp

    # Windowed path (for PSD + power sorting)
    w = hann(N, sym=False)
    U = np.sum(w**2) / N
    xw = w * x
    Xw = fft(xw)

    freqs = np.arange(half_N) / (N * dt)  # 0..(N/2-1)
    Xpos_w = Xw[:half_N]
    # One-sided PSD (power/Hz)
    S = (np.abs(Xpos_w)**2) / (fs * N * U)
    if half_N > 2:
        S[1:-1] *= 2.0
    df = fs / N
    power = S * df  # per-bin power (for sorting)

    # Unwindowed path (for reconstruction amps & phases)
    X = fft(x)
    Xpos = X[:half_N]

    # Peak amplitude per cosine
    amps = np.abs(Xpos) / N
    if half_N > 2:
        amps[1:-1] *= 2.0

    phases = np.angle(Xpos)

    # DC convention: put mean at 0 Hz, phase 0
    amps[0] = mean_amp
    phases[0] = 0.0

    # Sort all except DC by POWER (from windowed PSD)
    if power.size > 1:
        perm_peakpow = np.argsort(power[1:])[::-1] + 1  # skip DC bin
        freqs_sorted  = np.concatenate(([0.0], freqs[perm_peakpow]))
        amps_sorted   = np.concatenate(([mean_amp], amps[perm_peakpow]))
        phases_sorted = np.concatenate(([0.0],    phases[perm_peakpow]))
    else:
        freqs_sorted  = np.array([0.0])
        amps_sorted   = np.array([mean_amp])
        phases_sorted = np.array([0.0])

    STlength = chord * abs(math.sin(math.radians(aoa_deg)))
    ST_sorted = freqs_sorted * STlength / Vinf

    return freqs, amps, phases, power, ST_sorted, amps_sorted, phases_sorted

# ------------------------ discover Re directories ------------------------
all_entries = [os.path.join(dat_folder, f) for f in os.listdir(dat_folder)] if os.path.isdir(dat_folder) else []
re_dirs = [p for p in all_entries if os.path.isdir(p) and re.match(r"^RE\d+_\d+E\d+$", os.path.basename(p))]

# If no RE* subfolders, treat dat_folder as a single-Re directory
if not re_dirs:
    print(f"Info: No RE* subfolders found; using {dat_folder} as a single-Re directory.")
    re_dirs = [dat_folder]

Re = []
for d in re_dirs:
    r = parse_re_from_dirname(os.path.basename(d))
    if np.isnan(r):
        r = fluid_density * Vinf_fallback * chord / fluid_viscosity
        print(f"Warning: Folder {os.path.basename(d)} didn't match RE* pattern; using fallback Re={r}")
    Re.append(r)
Re = np.array(Re, dtype=float)

# Use first RE folder to define the ordered AOA file set
base_files = dat_files_in(re_dirs[0])
N_AOA = len(base_files)
if N_AOA == 0:
    raise RuntimeError(f"No .dat files found in {re_dirs[0]} (or its data_files subdir).")

# ------------------------ alloc outputs ------------------------
NRe = len(Re)
single_Re = (NRe == 1)

AOA = np.zeros(N_AOA, dtype=float)
thickness = 0.0  # placeholder
airfoil_name = "placeholder"

CL_ST = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CD_ST = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CM_ST = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CF_ST = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CL_Amp = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CD_Amp = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CM_Amp = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CF_Amp = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CL_Pha = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CD_Pha = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CM_Pha = np.zeros((NRe, N_AOA, NFreq), dtype=float)
CF_Pha = np.zeros((NRe, N_AOA, NFreq), dtype=float)

# ------------------------ main loops: Re folders, then AOA files ------------------------
for iRe, re_dir in enumerate(re_dirs):
    Vinf = Re[iRe] * fluid_viscosity / (fluid_density * chord)

    files_i = dat_files_in(re_dir)
    filemap = {os.path.basename(f): f for f in files_i}

    for iaoa, f0 in enumerate(base_files):
        b0 = os.path.basename(f0)
        file = filemap.get(b0, None)
        if file is None:
            print(f"Warning: AOA file '{b0}' not found in {re_dir}; skipping (iRe={iRe}, iaoa={iaoa}).")
            continue

        print(f"Processing (Re idx={iRe}) {file}")
        filename = os.path.splitext(os.path.basename(file))[0]
        AOA_str = re.sub(r"[^\d\-]+", "", filename.split("_")[-1])

        if iRe == 1-1:  # first Re pass
            try:
                AOA[iaoa] = float(AOA_str)
            except Exception:
                AOA[iaoa] = np.nan
            airfoil_name = "_".join(filename.split("_")[:-1])
            try:
                thickness_str = re.sub(r"[^\d\-]+", "", airfoil_name.split("_")[-1])
                thickness = float(thickness_str) / 1000.0
            except Exception:
                thickness = 0.0
        else:
            try:
                aoa_chk = float(AOA_str)
            except Exception:
                aoa_chk = np.nan
            if not np.isnan(aoa_chk) and abs(aoa_chk - AOA[iaoa]) > 1e-9:
                print(f"Warning: AOA mismatch at iaoa={iaoa} between base set ({AOA[iaoa]}) and {re_dir} ({aoa_chk}).")

        # Read data
        data = safe_load_dat(file)
        if data is None:
            continue

        timefull = data[:, 0]
        if timefull.size < 6001:
            print(f"Warning: Skipping {file}: timefull length {timefull.size} < 6001")
            continue

        dt = float(timefull[1] - timefull[0])  # assumes fixed dt

        # Columns (0-based): time=0, fpx=1, fpy=2, fvx=4, fvy=5, mty=8
        try:
            fpx, fpy = data[:, 1], data[:, 2]
            fvx, fvy = data[:, 4], data[:, 5]
            mty      = data[:, 8]
        except Exception:
            print(f"Warning: {file} does not have expected columns; skipping.")
            continue

        q = 0.5 * fluid_density * Vinf**2 * chord * span
        CL = (fpy + fvy) / q      # +x is inflow direction
        CD = (fpx + fvx) / q
        CF = np.sqrt(CD**2 + CL**2)
        CM = (mty) / q

        # FFT / PSD per-channel
        freqs_cl, amps_cl, phases_cl, power_cl, ST_sorted_cl, amps_sorted_cl, phases_sorted_cl = \
            compute_fft(CL, dt, chord, AOA[iaoa], Vinf)
        freqs_cd, amps_cd, phases_cd, power_cd, ST_sorted_cd, amps_sorted_cd, phases_sorted_cd = \
            compute_fft(CD, dt, chord, AOA[iaoa], Vinf)
        freqs_cm, amps_cm, phases_cm, power_cm, ST_sorted_cm, amps_sorted_cm, phases_sorted_cm = \
            compute_fft(CM, dt, chord, AOA[iaoa], Vinf)
        freqs_cf, amps_cf, phases_cf, power_cf, ST_sorted_cf, amps_sorted_cf, phases_sorted_cf = \
            compute_fft(CF, dt, chord, AOA[iaoa], Vinf)

        # --- Plots (optional) ---
        if genplots:
            idx_start = max(0, int(round(sampledT_startcutoff / dt)) - 1)
            aoa_str = _slug(f"{AOA[iaoa]:.4g}")
            re_str  = _slug(f"{Re[iRe]:.5g}")

            # CF time series with reconstruction
            try:
                from vorlap import reconstruct_signal   # import your package function
            except ImportError:
                raise RuntimeError("Could not import vorlap.reconstruct_signal. Make sure VorLap is installed and on PYTHONPATH.")

            signal = reconstruct_signal(freqs_cf, amps_cf, phases_cf, timefull)

            fig = plt.figure()
            plt.plot(timefull[idx_start:], signal[idx_start:], label="Reconstructed", linewidth=2)
            plt.plot(timefull[idx_start:], CF[idx_start:], label="Original", linewidth=2)
            plt.xlabel("Time (s)")
            plt.ylabel("CF")
            plt.title(f"(CF), AOA: {AOA[iaoa]} (Re={Re[iRe]:.5g})")
            plt.xlim(timefull[idx_start], timefull[-1]*0.98)         # cover only the plotted time span
            plt.ylim(np.min(CF[idx_start:]) * 1.1, np.max(CF[idx_start:]) * 1.1)  # padded range for visibility
            plt.legend()
            savefig_and_close(fig, os.path.join(figs_dir, f"CF_time_AOA{aoa_str}_Re{re_str}.png"))

            # CL time series
            fig = plt.figure()
            plt.plot(timefull[idx_start:], CL[idx_start:], linewidth=2)
            plt.xlabel("Time (s)"); plt.ylabel("CL")
            plt.title(f"(CL), AOA: {AOA[iaoa]} (Re={Re[iRe]:.5g})")
            savefig_and_close(fig, os.path.join(figs_dir, f"CL_time_AOA{aoa_str}_Re{re_str}.png"))

            # CD time series
            fig = plt.figure()
            plt.plot(timefull[idx_start:], CD[idx_start:], linewidth=2)
            plt.xlabel("Time (s)"); plt.ylabel("CD")
            plt.title(f"(CD), AOA: {AOA[iaoa]} (Re={Re[iRe]:.5g})")
            savefig_and_close(fig, os.path.join(figs_dir, f"CD_time_AOA{aoa_str}_Re{re_str}.png"))

            # Bode-like PSDs (skip DC)
            if freqs_cf.size >= 2:
                fig = plt.figure()
                plt.plot(freqs_cf[1:NFreq], power_cf[1:NFreq], marker='x', linewidth=2)
                plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD")
                plt.title(f"(CF), AOA: {AOA[iaoa]} (Re={Re[iRe]:.5g})")
                savefig_and_close(fig, os.path.join(figs_dir, f"CF_psd_AOA{aoa_str}_Re{re_str}.png"))

            if freqs_cl.size >= 2:
                fig = plt.figure()
                plt.plot(freqs_cl[1:NFreq], power_cl[1:NFreq], marker='x', linewidth=2)
                plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD")
                plt.title(f"(CL), AOA: {AOA[iaoa]} (Re={Re[iRe]:.5g})")
                savefig_and_close(fig, os.path.join(figs_dir, f"CL_psd_AOA{aoa_str}_Re{re_str}.png"))

            if freqs_cd.size >= 2:
                fig = plt.figure()
                plt.plot(freqs_cd[1:NFreq], power_cd[1:NFreq], marker='x', linewidth=2)
                plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD")
                plt.title(f"(CD), AOA: {AOA[iaoa]} (Re={Re[iRe]:.5g})")
                savefig_and_close(fig, os.path.join(figs_dir, f"CD_psd_AOA{aoa_str}_Re{re_str}.png"))


        # --- Store (cap by available length) ---
        if NFreq > ST_sorted_cl.size:
            print(f"Warning: Number of frequencies ({NFreq}) exceeds available ({ST_sorted_cl.size})")
            NFreq = ST_sorted_cl.size
            # Also shrink already-allocated arrays along last axis if needed
            CL_ST = CL_ST[..., :NFreq]; CD_ST = CD_ST[..., :NFreq]; CM_ST = CM_ST[..., :NFreq]; CF_ST = CF_ST[..., :NFreq]
            CL_Amp = CL_Amp[..., :NFreq]; CD_Amp = CD_Amp[..., :NFreq]; CM_Amp = CM_Amp[..., :NFreq]; CF_Amp = CF_Amp[..., :NFreq]
            CL_Pha = CL_Pha[..., :NFreq]; CD_Pha = CD_Pha[..., :NFreq]; CM_Pha = CM_Pha[..., :NFreq]; CF_Pha = CF_Pha[..., :NFreq]

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

# ------------------------ single-Re duplication ------------------------
if single_Re:
    # Duplicate the first slice so downstream code expecting 2 Re slices still works
    def dup_first(arr):
        return np.concatenate([arr, arr[:1, ...]], axis=0)
    CL_ST = dup_first(CL_ST); CD_ST = dup_first(CD_ST); CM_ST = dup_first(CM_ST); CF_ST = dup_first(CF_ST)
    CL_Amp = dup_first(CL_Amp); CD_Amp = dup_first(CD_Amp); CM_Amp = dup_first(CM_Amp); CF_Amp = dup_first(CF_Amp)
    CL_Pha = dup_first(CL_Pha); CD_Pha = dup_first(CD_Pha); CM_Pha = dup_first(CM_Pha); CF_Pha = dup_first(CF_Pha)

# ------------------------ sort by AOA ------------------------
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

# ------------------------ sort by Re (ascending) ------------------------
Re_sort_idx = np.argsort(Re)
Re_sort = Re[Re_sort_idx]

CL_ST_sort = CL_ST_sort[Re_sort_idx, :, :]
CD_ST_sort = CD_ST_sort[Re_sort_idx, :, :]
CM_ST_sort = CM_ST_sort[Re_sort_idx, :, :]
CF_ST_sort = CF_ST_sort[Re_sort_idx, :, :]

CL_Amp_sort = CL_Amp_sort[Re_sort_idx, :, :]
CD_Amp_sort = CD_Amp_sort[Re_sort_idx, :, :]
CM_Amp_sort = CM_Amp_sort[Re_sort_idx, :, :]
CF_Amp_sort = CF_Amp_sort[Re_sort_idx, :, :]

CL_Pha_sort = CL_Pha_sort[Re_sort_idx, :, :]
CD_Pha_sort = CD_Pha_sort[Re_sort_idx, :, :]
CM_Pha_sort = CM_Pha_sort[Re_sort_idx, :, :]
CF_Pha_sort = CF_Pha_sort[Re_sort_idx, :, :]

if genplots:
    ire_indices = [0] if single_Re else list(range(CF_ST_sort.shape[0]))
    for iRe in ire_indices:
        re_str = _slug(f"{Re_sort[iRe]:.5g}")

        # ST vs AOA, multiple ist
        fig = plt.figure()
        for ist in range(2, min(20, NFreq)):
            plt.plot(AOA_sort, CF_ST_sort[iRe, :, ist], marker='x', linewidth=2, label=f"ist={ist}")
        plt.xlabel("AOA (deg)"); plt.ylabel("ST (CF)")
        plt.ylim(0.0, 0.5)
        plt.title(f"ST Re={Re_sort[iRe]:.5g}")
        plt.legend()
        savefig_and_close(fig, os.path.join(figs_dir, f"ST_summary_Re{re_str}.png"))

        # Amp vs AOA, multiple ist
        fig = plt.figure()
        for ist in range(2, min(20, NFreq)):
            plt.plot(AOA_sort, CF_Amp_sort[iRe, :, ist], marker='x', linewidth=2, label=f"ist={ist}")
        plt.xlabel("AOA (deg)"); plt.ylabel("Amp (CF)")
        plt.title(f"Amp Re={Re_sort[iRe]:.5g}")
        plt.legend()
        savefig_and_close(fig, os.path.join(figs_dir, f"Amp_summary_Re{re_str}.png"))


# ------------------------ write HDF5 ------------------------
with h5py.File(h5_path, "w") as f:
    # Strings: store as fixed-length or variable-length ASCII/UTF-8
    f.create_dataset("Airfoilname", data=np.array(airfoil_name, dtype=h5py.string_dtype("utf-8")))
    f.create_dataset("Re", data=Re_sort)
    f.create_dataset("Thickness", data=np.array(thickness))
    f.create_dataset("AOA", data=AOA_sort)

    f.create_dataset("CL_ST", data=CL_ST_sort)
    f.create_dataset("CD_ST", data=CD_ST_sort)
    f.create_dataset("CM_ST", data=CM_ST_sort)
    f.create_dataset("CF_ST", data=CF_ST_sort)

    f.create_dataset("CL_Amp", data=CL_Amp_sort)
    f.create_dataset("CD_Amp", data=CD_Amp_sort)
    f.create_dataset("CM_Amp", data=CM_Amp_sort)
    f.create_dataset("CF_Amp", data=CF_Amp_sort)

    f.create_dataset("CL_Pha", data=CL_Pha_sort)
    f.create_dataset("CD_Pha", data=CD_Pha_sort)
    f.create_dataset("CM_Pha", data=CM_Pha_sort)
    f.create_dataset("CF_Pha", data=CF_Pha_sort)

print(f"Wrote HDF5: {h5_path}")
