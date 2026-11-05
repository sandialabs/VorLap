"""Four-step native QBlade API/input check with synthetic DC coefficients.

This checks explicit nodal inputs, not the pending automatic ambient/local
sampling choice, response calibration, or fatigue convergence.
"""
from __future__ import annotations

import argparse
from ctypes import POINTER, byref, c_double
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vorlap.qblade_case_prep import QBladeCasePrepRequest, prepare_qblade_external_case
from vorlap.qblade_files import replace_qblade_keyword_text
from vorlap.qblade_run_support import load_qblade_api
from vorlap.qblade_sil import SilVorLapCoupler
from vorlap.turbsim import write_turbsim_input


def patch(path, settings):
    text = path.read_text()
    for key, value in settings.items():
        text, _, count = replace_qblade_keyword_text(text, key, value)
        if count != 1:
            raise ValueError(f"Missing {key} in staged {path}")
    path.write_text(text)


def array_sample(qblade, positions):
    coordinates = [np.ascontiguousarray(positions[:, axis], dtype=np.float64) for axis in range(3)]
    velocities = [np.full(len(positions), np.nan, dtype=np.float64) for _ in range(3)]
    pointers = [array.ctypes.data_as(POINTER(c_double))
                for array in [*coordinates, *velocities]]
    qblade.getWindspeedArray(*pointers, len(positions))
    wind = np.column_stack(velocities)
    assert np.isfinite(wind).all(), "Native array sampler left missing/nonfinite vectors"
    return wind


def sample(qblade, positions, *, native_aerodynamics):
    wind = array_sample(qblade, positions)
    reversed_wind = array_sample(qblade, positions[::-1])[::-1]
    np.testing.assert_allclose(wind, reversed_wind, rtol=1e-11, atol=1e-11)
    scalar = np.empty_like(wind)
    for i, point in enumerate(positions):
        buffer = (c_double * 3)(math.nan, math.nan, math.nan)
        qblade.getWindspeed(*map(float, point), byref(buffer))
        scalar[i] = buffer[:]
    # The CPU array getter adds bound-blade induction, which the scalar
    # getter omits. Equality is valid only for the induction-free case.
    if not native_aerodynamics:
        np.testing.assert_allclose(wind, scalar, rtol=1e-11, atol=1e-11)
    return wind, float(np.max(np.abs(wind - scalar))), float(np.max(np.abs(wind - reversed_wind)))


def expected_dc_force(coupler, wind):
    controller = coupler.runtime.controller
    angle = math.radians(float(coupler.swap[coupler._azimuth_offset]))
    axis = controller.rotation_axis
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) * math.cos(angle) + (1 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * cross
    velocity_spec = coupler._velocity_spec
    velocity = coupler.swap[velocity_spec.offset:velocity_spec.offset + velocity_spec.size].reshape(-1, 3)
    controller_index = {node: index for index, node in enumerate(controller.node_ids)}
    expected = []
    for i, node in enumerate(coupler.node_ids):
        j = controller_index[node]
        chord = rotation @ controller._base_chord[j]
        normal = rotation @ controller._base_normal[j]
        relative = wind[i] - velocity[i]
        speed_squared = np.dot(relative, chord)**2 + np.dot(relative, normal)**2
        scale = 0.5 * controller.fluid_density * speed_squared * controller._chord[j] * controller._local_length[j]
        expected.append(scale * (0.5 * chord + normal))
    return np.asarray(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qblade-root", type=Path, default=ROOT / "run_headless/QBladeCE_2.0.9.7")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-aerodynamics", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="vorlap-nodal-native-") as temporary:
        stage = Path(temporary)
        archive = Path(os.environ.get("VORLAP_PAPER_SOURCE_ARCHIVE", str(ROOT / "run_headless/config/hvawt_pilot_hpc_inputs/U05p00ms_source_case.tar.gz")))
        with tarfile.open(archive) as source:
            source.extractall(stage, filter="data")
        case = stage / "U05p00ms"
        sim = case / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
        trb = case / "10m_HVAWT_reference_turbine/10m_HVAWT_reference_turbine.trb"
        wind_input = stage / "verification_wind.inp"
        write_turbsim_input(wind_input, template_path=ROOT / "run_headless/templates/turbsim_openfast_iec.inp", settings={
            "NumGrid_Z": 5, "NumGrid_Y": 5, "TimeStep": 0.05,
            "AnalysisTime": 16, "UsableTime": '"ALL"', "IECturbc": '"5"',
            "RandSeed1": 4433456, "GridHeight": 26, "GridWidth": 26,
        })
        turbsim = stage / "TurbSim64"
        shutil.copyfile(args.qblade_root / "Binaries/TurbSim64", turbsim)
        turbsim.chmod(0o755)
        with (output / "turbsim.log").open("w") as log:
            subprocess.run([str(turbsim), str(wind_input)],
                           cwd=stage, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        wind_path = wind_input.with_suffix(".bts")
        assert wind_path.is_file()
        patch(sim, {"WNDTYPE": 1, "WNDNAME": str(wind_path), "HORANGLE": 0,
                    "VERTANGLE": 0, "NUMTIMESTEPS": 4, "TIMESTEP": 0.0075,
                    "RAMPUP": 0, "RELAXSTEPS": 0, "RPMPRESCRIBED": 1,
                    "INCLUDEAERO": int(args.native_aerodynamics),
                    "STOREFROM": 0, "FILTERFILE": ""})
        if not args.native_aerodynamics:
            # INCLUDEAERO disables structural aero loads, but QBlade still
            # calculates circulation. Zero native polar coefficients give
            # a genuine zero-induction limit for the getter comparison.
            zeroed_rows = 0
            for polar in (case / "10m_HVAWT_reference_turbine/Aero/Polars").glob("*.plr"):
                lines = []
                for line in polar.read_text().splitlines():
                    fields = line.split()
                    try:
                        values = [float(field) for field in fields]
                    except ValueError:
                        values = []
                    if len(values) > 3 and (len(values) - 1) % 3 == 0 and -180 <= values[0] <= 180:
                        line = " ".join([fields[0], *(["0.0"] * (len(fields) - 1))])
                        zeroed_rows += 1
                    lines.append(line)
                polar.write_text("\n".join(lines) + "\n")
            assert zeroed_rows > 0
        airfoils = case / "10m_HVAWT_reference_turbine/VorLapAirfoils"
        for airfoil in airfoils.glob("*.h5"):
            with h5py.File(airfoil, "r") as original:
                name = original["Airfoilname"][()]
            with h5py.File(airfoil, "w") as data:
                data["Airfoilname"] = name
                data["Re"] = [0.01, 1e9]
                data["AOA"] = [-180., 180.]
                data["Thickness"] = [0.18]
                for channel in ("CL", "CD", "CM", "CF"):
                    data[channel + "_ST"] = np.zeros((2, 2, 1))
                    data[channel + "_Pha"] = np.zeros((2, 2, 1))
                    data[channel + "_Amp"] = np.full((2, 2, 1), {"CL": 1., "CD": 0.5}.get(channel, 0.))
        prepare_qblade_external_case(QBladeCasePrepRequest(
            sim_path=sim, airfoil_dir=airfoils, use_qblade_local_inflow=True,
            include_mean_load=True, n_freq_depth=1, include_default_python_path=False,
            write_geometry_audit=False,
        ))
        # Only the explicit Python SIL forces are applied in this probe.
        patch(trb, {"LIBFILE_1": "", "LIBFUNCTION_1": ""})
        coupler = SilVorLapCoupler(sim, record_history=True)
        qblade, library = load_qblade_api(args.qblade_root)
        rows = []
        try:
            qblade.setLogFile(str(output / "qblade.log").encode())
            qblade.createInstance(1, 32)
            qblade.setOmpNumThreads(4)
            qblade.setUseOpenCl(False)
            os.chdir(case)
            qblade.loadSimDefinition(sim.name.encode())
            qblade.initializeSimulation()
            coupler.validate_channels(qblade)
            spec = coupler._local_inflow_position_spec
            for step in range(4):
                coupler._fill_inputs(qblade, step)
                positions = coupler.swap[spec.offset:spec.offset + spec.size].reshape(-1, 3).copy()
                assert np.isfinite(positions).all() and np.max(np.ptp(positions, axis=0)) > 1.
                wind, scalar_difference, permutation_error = sample(
                    qblade, positions, native_aerodynamics=args.native_aerodynamics,
                )
                assert np.max(np.ptp(wind, axis=0)) > 1e-4, "Wind did not vary across nodes"
                assert np.max(np.abs(wind[:, 1:])) > 1e-4, "No lateral/vertical turbulent components"
                coupler.step(qblade, step, ambient_velocity_global_mps=wind)
                force_spec = coupler._force_spec
                forces = coupler.swap[force_spec.offset:force_spec.offset + force_spec.size].reshape(-1, 3).copy()
                expected = expected_dc_force(coupler, wind)
                np.testing.assert_allclose(forces, expected, rtol=1e-10, atol=1e-10)
                assert np.max(np.abs(forces)) > 1e-3
                rows.append({"step": step, "time_s": float(qblade.getCustomData_at_num(b"Time [s]", 0., 0)),
                             "positions_m": positions.tolist(), "sampled_local_wind_mps": wind.tolist(),
                             "array_vs_scalar_max_difference_mps": scalar_difference,
                             "array_permutation_max_error_mps": permutation_error,
                             "force_vs_analytic_max_error_n": float(np.max(np.abs(forces - expected))),
                             "maximum_force_n": float(np.max(np.abs(forces)))})
                assert qblade.advanceTurbineSimulation(), f"QBlade failed at step {step}"
        finally:
            qblade.closeInstance()
        position_change = float(np.max(np.abs(np.asarray(rows[-1]["positions_m"]) - np.asarray(rows[0]["positions_m"]))))
        assert position_change > 1e-5, "Nodes did not move"
        report = {"status": "passed_native_api_and_explicit_nodal_input_smoke",
                  "steps": 4, "nodes": len(coupler.node_ids), "node_ids": coupler.node_ids,
                  "wind": "new 16-s, 5x5 TurbSim field, seed 4433456; QBlade samples include induction",
                  "coefficient_probe": "synthetic constant CL=1, CD=0.5; analytic dimensional-force check",
                  "native_aerodynamics": args.native_aerodynamics,
                  "native_polar_coefficients_zeroed": not args.native_aerodynamics,
                  "scalar_array_equality_required": not args.native_aerodynamics,
                  "automatic_sampler_verified": False, "position_change_m": position_change,
                  "qblade_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(), "samples": rows}
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key not in {"samples", "node_ids"}}, indent=2))


if __name__ == "__main__":
    main()
