from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "paper" / "2025-nawea" / "reproduce.py"


def load_reproduction_driver():
    spec = importlib.util.spec_from_file_location("nawea_2025_reproduce", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_corrected_paper_spectrum_changes_only_known_metadata(tmp_path):
    driver = load_reproduction_driver()
    source = REPO / "data" / "airfoils" / "NACA0018.h5"
    corrected, summary = driver.corrected_legacy_naca(tmp_path)

    assert summary["legacy_h5_sha256"] == driver.LEGACY_NACA_H5_SHA256
    assert summary["old_thickness_ratio"] == 0.018
    assert summary["new_thickness_ratio"] == 0.18
    with h5py.File(source, "r") as old, h5py.File(corrected, "r") as new:
        assert float(new["Thickness"][()]) == 0.18
        for name in ("CL_ST", "CD_ST", "CM_ST", "CF_ST"):
            np.testing.assert_array_equal(new[name][..., 1:], old[name][..., 1:])
            np.testing.assert_array_equal(new[name][..., 0], 0.0)
        for name in ("CL_Amp", "CD_Amp", "CM_Amp", "CF_Amp", "CL_Pha", "CD_Pha", "CM_Pha", "CF_Pha"):
            np.testing.assert_array_equal(new[name][...], old[name][...])


def test_release_acceptance_rejects_reviewer_metric_drift():
    driver = load_reproduction_driver()
    summary = {
        "verification": {
            "pearson_r_reported_2dp": "0.91",
            "modal_frequency_hz": 15.0,
            "minimum_percent_difference": 0.5,
        },
        "reference_turbine": {
            "inflow_speeds_m_s": list(np.arange(1.0, 17.0, 1.0)),
            "minimum_percent_difference": 0.2,
            "torque_max_nm": -1.0,
        },
        "corrected_legacy_spectral_input": {
            "new_thickness_ratio": 0.18,
            "new_dc_strouhal": 0.0,
            "legacy_h5_sha256": driver.LEGACY_NACA_H5_SHA256,
        },
        "naca_processor": {"archive_sha256": driver.NACA_ARCHIVE_SHA256},
        "naca_histories": {"archive_sha256": driver.PAPER_NACA_HISTORIES_SHA256},
    }

    try:
        driver.validate_release_results(summary)
    except RuntimeError as error:
        assert "paper_pearson_r_is_0.90" in str(error)
    else:
        raise AssertionError("reviewer-sensitive correlation drift was accepted")
