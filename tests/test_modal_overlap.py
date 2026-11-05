from conftest import make_component, make_constant_airfoil_fft
from vorlap.modal_overlap import assess_peak_coverage, rank_local_peaks, screen_modal_overlap


def test_modal_overlap_dimensionalizes_stcf_and_skips_dc():
    component = make_component(n_nodes=1, span=1.0)
    component.thickness[:] = 0.2
    airfoil = make_constant_airfoil_fft(st_h1=0.2, cf_h1_amp=0.5)

    rows = screen_modal_overlap(
        [component],
        {"default": airfoil},
        natural_frequencies_hz=[1.0],
        wind_speeds_mps=[1.0],
        azimuths_deg=[0.0],
        node_ids=["BLD_1_0.500000"],
        rpm=0.0,
        fluid_density_kg_m3=1.0,
        fluid_kinematic_viscosity_m2_s=1.0,
        n_freq_depth=2,
        amplitude_coefficient_cutoff=0.01,
        source_frequency_cap_hz=10.0,
        overlap_band_fraction=0.05,
    )

    row = rows[0]
    best = row["best_force_overlap"]
    assert best["source_depth_index"] == 1
    assert best["forcing_frequency_hz"] == 1.0  # St=0.2, U=1, L=chord*t/c=0.2.
    assert best["natural_frequency_hz"] == 1.0
    assert best["detuning_percent"] == 0.0
    assert best["section_force_amplitude_n"] == 0.25  # AmpCF * 0.5*rho*U^2*c*Lnode.
    assert row["overlap_score_n"] == 0.25
    assert row["within_overlap_band_candidate_count"] == 1


def test_peak_ranking_and_matrix_coverage_use_frequency_band_tolerance():
    rows = [
        {"wind_speed_mps": 5.0, "score": 1.0},
        {"wind_speed_mps": 6.0, "score": 4.0},
        {"wind_speed_mps": 7.0, "score": 2.0},
        {"wind_speed_mps": 8.0, "score": 5.0},
        {"wind_speed_mps": 9.0, "score": 1.0},
    ]
    peaks = rank_local_peaks(
        rows,
        "score",
        max_peaks=3,
        minimum_relative_score=0.2,
        minimum_spacing_mps=0.5,
    )
    assert [row["wind_speed_mps"] for row in peaks] == [8.0, 6.0]

    missing = assess_peak_coverage(peaks, [5.0, 8.0], minimum_tolerance_mps=0.5)
    assert missing["passed"] is False
    assert missing["uncovered_peak_wind_speeds_mps"] == [6.0]

    covered = assess_peak_coverage(peaks, [5.5, 8.0], minimum_tolerance_mps=0.5)
    assert covered["passed"] is True
