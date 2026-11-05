"""Actual upstream solver checks; receipts/fake API success cannot satisfy these.

The oracles are beam theory and QBlade's independent LOADINGFILE path. No
retained response curves or paper-private inputs are used or regenerated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from vorlap.qblade_sil import read_qblade_result_numeric_columns

pytestmark = pytest.mark.requires_qblade
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_cases(tmp_path_factory):
    evidence = os.environ.get("VORLAP_QBLADE_PHYSICS_EVIDENCE")
    base = Path(evidence) if evidence else tmp_path_factory.mktemp("qblade-physics")
    base.mkdir(parents=True, exist_ok=True)
    cache = {}

    def run(mode, *, azimuth=0, execution="step-loop", harmonic=False, dt=.002, elements=20, fault="none"):
        key = (mode, azimuth, execution, harmonic, dt, elements, fault)
        if key not in cache:
            case = base / f"{len(cache):02d}-{mode}-{azimuth}-{fault}"
            command = [sys.executable, str(ROOT / "scripts/ci/qblade_physics.py"), str(case),
                       "--mode", mode, "--azimuth", str(azimuth), "--execution", execution,
                       "--dt", str(dt), "--elements", str(elements), "--fault", fault]
            if harmonic:
                command.append("--harmonic")
            log = case.with_suffix(".log")
            with log.open("w") as output:
                process = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                         env=os.environ | {"QT_QPA_PLATFORM": "offscreen"}, timeout=300)
            assert process.returncode == 0, f"Real QBlade worker failed; inspect {log}"
            receipt = json.loads((case / "receipt.json").read_text())
            summary = json.loads(Path(receipt["summary"]).read_text())
            assert summary["completed_steps"] == round(2 / dt)
            assert summary["failed"] is False and summary["stopped"] is False
            assert summary["exported_results"] is True
            exported = read_qblade_result_numeric_columns(Path(summary["result_file"]))
            required = ["Time~[s]"] + [
                f"{axis}_{frame}~{quantity}~BLD_1~pos~{station}~[{unit}]"
                for axis in "XYZ"
                for frame, quantity, station, unit in (
                    ("g", "Pos.", "1.000", "m"),
                    ("b", "Trl.Def.", "1.000", "m"),
                    ("b", "Mom.", "0.000", "Nm"))
            ]
            # Rotor power coefficients are undefined for this zero-RPM,
            # aerodynamics-disabled fixture. Every physical oracle channel
            # must exist and be finite, including all three force directions.
            data = {name: np.asarray(exported[name]) for name in required}
            assert all(np.isfinite(v).all() for v in data.values())
            time = data["Time~[s]"]
            expected_samples = round(2 / dt) + (execution == "run-full-simulation")
            np.testing.assert_allclose(time, np.arange(expected_samples) * dt, atol=1e-10, rtol=0)
            assert receipt["inputs"] and receipt["source"]
            if mode == "qblade-external":
                native = summary["native_callback_evidence"]
                assert native["runtime_initialized_message_seen"] is True
                assert native["update_init_count"] == 1
                # QBlade initializes/samples once more in its full scheduler;
                # SIL explicit controller advancement starts after step zero.
                assert native["update_count_observed_max"] == round(2 / dt) + (
                    1 if execution == "run-full-simulation" else -1)
                assert native["vorlap_qblade_runtime_file"] == str(ROOT / "vorlap/qblade_runtime.py")
            elif mode == "sil-vorlap":
                assert summary["sil_vorlap_updates"] == round(2 / dt)
                assert summary["native_callback_evidence"]["update_init_count"] == 0
            cache[key] = data, summary
        return cache[key]
    return run


def tip_displacement(data):
    # QBlade's *_b displacement components do not rotate with initial rotor
    # azimuth. Read the actual global positions and subtract the initial state.
    return np.array([data[f"{axis}_g~Pos.~BLD_1~pos~1.000~[m]"]
                     - data[f"{axis}_g~Pos.~BLD_1~pos~1.000~[m]"][0] for axis in "XYZ"]).T


def assert_static_response(data, azimuth):
    angle = np.deg2rad(azimuth)
    # EI=120, L=1, q=.5*1.225*4²*.1=.98 N/m. Timoshenko shear
    # correction <6e-9 m. 20-element nodal quadrature is O(h²), <0.1%.
    expected = .98 / (8 * 120) * np.array([1, .5*np.cos(angle), .5*np.sin(angle)])
    displacement = tip_displacement(data)[-100:].mean(axis=0)
    np.testing.assert_allclose(displacement, expected, rtol=.002, atol=2e-6)
    moment = np.mean(data["Y_b~Mom.~BLD_1~pos~0.000~[Nm]"][-100:])
    assert moment == pytest.approx(.98 / 2, rel=.002)  # root reaction, independent balance


@pytest.mark.parametrize("mode,execution", [("sil-vorlap", "step-loop"),
    ("qblade-external", "step-loop"), ("qblade-external", "run-full-simulation")])
@pytest.mark.parametrize("azimuth", [0, 90])
def test_real_coupling_static_load_sign_scale_frame_and_reaction(real_cases, mode, execution, azimuth):
    data, _ = real_cases(mode, execution=execution, azimuth=azimuth)
    reference, _ = real_cases("reference", azimuth=azimuth)
    assert_static_response(data, azimuth)
    assert_static_response(reference, azimuth)
    np.testing.assert_allclose(tip_displacement(data)[-100:].mean(axis=0),
                               tip_displacement(reference)[-100:].mean(axis=0), atol=2e-6, rtol=.002)


def phasor(data):
    time = data["Time~[s]"]
    selected = (time >= 1) & (time < 2)
    fit = np.column_stack([np.ones(sum(selected)), np.cos(4*np.pi*time[selected]),
                           np.sin(4*np.pi*time[selected])])
    coefficients = np.linalg.lstsq(fit, tip_displacement(data)[selected, 0], rcond=None)[0]
    return coefficients[1] - 1j*coefficients[2]


def exact_beam_phasor():
    # Solve EI*(1+i*w*beta) y'''' - m*w²*y=q with the four independent
    # cantilever boundary conditions. This is not QBlade's FE discretization.
    omega = 4*np.pi
    rigidity = 120*(1 + .01j*omega)
    roots = (omega**2/rigidity)**.25 * np.array([1, -1, 1j, -1j])
    exponent = np.exp(roots)
    particular = -.98 / omega**2
    coefficients = np.linalg.solve(np.array([np.ones(4), roots, roots**2*exponent,
                                              roots**3*exponent]), [-particular, 0, 0, 0])
    return particular + exponent @ coefficients


@pytest.mark.parametrize("mode,execution", [("sil-vorlap", "step-loop"),
    ("qblade-external", "step-loop"), ("qblade-external", "run-full-simulation")])
def test_real_harmonic_coupling_amplitude_phase_and_time_grid(real_cases, mode, execution):
    data, _ = real_cases(mode, execution=execution, harmonic=True)
    reference, _ = real_cases("reference", harmonic=True)
    # The tiny two-way velocity feedback is absent from the imposed-load oracle.
    # 0.5% complex error admits it, but rejects a single dt=.002 step delay
    # (2.51% at 2 Hz), sign/axis errors, disabled callbacks and wrong frequency.
    assert abs(phasor(data)/phasor(reference) - 1) < .005
    assert abs(abs(phasor(data)/exact_beam_phasor()) - 1) < .005


def test_real_beam_time_refinement_against_continuum(real_cases):
    coarse, _ = real_cases("reference", harmonic=True, elements=40)
    fine, _ = real_cases("reference", harmonic=True, dt=.001, elements=40)
    exact = exact_beam_phasor()
    coarse_error = abs(phasor(coarse)/exact - 1)
    fine_error = abs(phasor(fine)/exact - 1)
    assert fine_error < .7 * coarse_error
    assert fine_error < .01  # Includes QBlade integration/time-reporting phase error.


def test_real_beam_space_refinement_against_static_continuum(real_cases):
    coarse, _ = real_cases("reference")
    fine, _ = real_cases("reference", elements=40)
    exact = .98 / (8 * 120)
    coarse_error = abs(tip_displacement(coarse)[-100:, 0].mean() / exact - 1)
    fine_error = abs(tip_displacement(fine)[-100:, 0].mean() / exact - 1)
    assert fine_error < .5 * coarse_error
    assert fine_error < .0004


@pytest.mark.parametrize("fault", ["drop", "sign", "scale", "axis"])
def test_real_native_physics_rejects_corrupted_exchanged_loads(real_cases, fault):
    broken, _ = real_cases("qblade-external", execution="run-full-simulation", fault=fault)
    # The worker and every native callback must succeed above; only the
    # independent physical oracle is allowed to detect the injected fault.
    with pytest.raises(AssertionError):
        assert_static_response(broken, 0)


def test_real_native_phase_oracle_rejects_one_step_force_delay(real_cases):
    broken, _ = real_cases("qblade-external", execution="run-full-simulation", harmonic=True, fault="delay")
    reference, _ = real_cases("reference", harmonic=True)
    assert abs(phasor(broken)/phasor(reference) - 1) > .005


@pytest.mark.parametrize("body,message", [
    ("@pytest.mark.requires_qblade\ndef test_probe(): pytest.skip('intentional guard probe')",
     "Required QBlade tests did not all execute and pass"),
    ("def test_probe(): pass", "selected no QBlade tests"),
], ids=["skip", "empty-selection"])
def test_required_qblade_job_cannot_pass_by_skipping_or_selecting_nothing(tmp_path, body, message):
    probe = tmp_path / "test_probe.py"
    probe.write_text("import pytest\n" + body + "\n")
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(ROOT / "tests"), str(ROOT)]),
        "VORLAP_QBLADE_TEST_RECEIPT": str(tmp_path / "receipt.json"),
    }
    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "conftest",
                             "--require-qblade", str(probe)], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert message in result.stdout + result.stderr


def test_required_qblade_inventory_rejects_removed_coverage(tmp_path):
    probe = tmp_path / "test_probe.py"
    probe.write_text("import pytest\n@pytest.mark.requires_qblade\ndef test_probe(): pass\n")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"required_nodeids": ["test_probe.py::test_deleted"]}))
    result = subprocess.run([
        sys.executable, "-m", "pytest", "-p", "conftest", "--require-qblade",
        "--qblade-inventory", str(inventory), str(probe),
    ], cwd=tmp_path, env=os.environ | {
        "PYTHONPATH": os.pathsep.join([str(ROOT / "tests"), str(ROOT)]),
        "VORLAP_QBLADE_TEST_RECEIPT": str(tmp_path / "receipt.json"),
    }, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "Required QBlade coverage is missing" in result.stdout + result.stderr
