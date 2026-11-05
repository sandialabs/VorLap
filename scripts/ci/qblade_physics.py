"""Real QBlade worker for independent beam/load CI tests (no solver doubles).

Each invocation owns a new QBlade process and writes its inputs and provenance.
The generic beam uses SI units: L=1 m, EI=120 Nm², m=1 kg/m, c=0.1 m.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from vorlap.qblade_bridge_config import QBladeBridgeConfig
from vorlap.qblade_bridge_service import QBladeBridgeService
from vorlap.qblade_run import QBladeRunRequest, run_qblade_case
from vorlap.qblade_run_support import file_sha256


def patch(text, values):
    for key, value in values.items():
        text, count = re.subn(r"^.*?(?<!\S)" + re.escape(key) + r"(?!\S)(.*)$",
                             lambda m: f"{value} {key}{m[1]}", text, flags=re.M)
        if count != 1:
            raise ValueError(f"Expected exactly one {key}, found {count}")
    return text


def make_case(case, *, azimuth, dt, elements, harmonic):
    template = ROOT / "examples/vlastos2026_cantilever/generated_case"
    model = case / "model"
    shutil.copytree(template / "vlastos2026_sc5p18_cantilever_u5p00", model)
    sim = case / "beam.sim"
    sim.write_text(patch((template / "vlastos2026_sc5p18_cantilever_u5p00.sim").read_text(), dict(
        OBJECTNAME="ci_beam", TURBNAME="ci_beam", TURBFILE="model/vlastos2026_sc5p18_cantilever.trb",
        MEANINF=4, HORANGLE=0, GRAVITY=0, INITIAL_AZIMUTH=azimuth,
        TIMESTEP=dt, NUMTIMESTEPS=round(2 / dt), LOADINGFILE="")), encoding="utf-8")
    fractions = np.linspace(0, 1, elements + 1)
    beam = model / "Structure/cantilever_beam.str"
    header = ("0.010 RAYLEIGHDMP\n1 STIFFTUNER\n1 MASSTUNER\n0 INTPTYPE\n1 BEAMTYPE\n"
              f"2 DISCTYPE\n{elements + 1} DISC\n"
              "LENFRACT MASSD EIx EIy EA GJ GA STRPIT KSX KSY RGX RGY XCM YCM XCE YCE XCS YCS\n")
    beam.write_text(header + "\n".join(" ".join(map(str, [s, 1, 120, 120, 1e8, 100, 1e8,
                    0, .85, .85, .001, .001, 0, 0, 0, 0, 0, 0])) for s in fractions) + "\n")
    bld = model / "Aero/vlastos2026_sc5p18_cantilever.bld"
    header = bld.read_text().split("POS_[m]")[0]
    bld.write_text(header + "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] TAXIS_[-] POLAR_FILE\n"
                   + "\n".join(f"{s:.12g} 0.1 0 0 0 0.5 Polars/CYLINDER.plr" for s in fractions) + "\n")
    main = model / "Structure/cantilever_main.str"
    # These independent monitor stations include root reaction and tip response.
    main.write_text(main.read_text().split("BLD_1_0.0000000000")[0]
                    + "\n".join(f"BLD_1_{s:.10f}" for s in fractions) + "\nTWR_0.0\nTWR_1.0\n")
    trb = model / "vlastos2026_sc5p18_cantilever.trb"
    trb.write_text(trb.read_text().split("----------------------------------------External Libraries")[0])
    airfoils = case / "airfoils"
    airfoils.mkdir()
    # Independent synthetic spectrum: q=.5*rho*U²*c=.98 N/m. CL is section
    # normal (-global X); CD is chordwise (+global Y at azimuth zero).
    with h5py.File(airfoils / "CYLINDER.h5", "w") as h:
        for key, value in dict(Airfoilname=np.bytes_("CYLINDER"), Re=[100, 1e8],
                               AOA=[-180, 180], Thickness=[1.0]).items():
            h[key] = value
        for channel, coefficient in dict(CL=-1, CD=.5, CM=0, CF=0).items():
            for field in ("ST", "Amp", "Pha"):
                data = np.zeros((2, 2, 2))
                if field == "ST":
                    data[:, :, 1] = .05  # 2 Hz at U=4 m/s and c=.1 m
                if field == "Amp":
                    data[:, :, int(harmonic)] = abs(coefficient) if harmonic else coefficient
                if field == "Pha" and harmonic:
                    data[:, :, 1] = np.pi if coefficient < 0 else 0
                h[f"{channel}_{field}"] = data
    return sim, airfoils, fractions


def inject_force_fault(fault):
    """Corrupt actual exchanged forces while retaining the actual solver/runtime.

    Deliberately keeps successful callbacks, load computation, and provenance.
    A validator that merely checked those receipts would falsely accept these.
    """
    if fault == "none":
        return
    from vorlap.qblade_runtime import VorLapQBladeRuntime
    original = VorLapQBladeRuntime.update

    def broken(self, swap, **kwargs):
        original(self, swap, **kwargs)
        force = self._adapter_for(swap).block(self.force_block)
        if fault == "drop":
            force[...] = 0
        elif fault == "sign":
            force[...] *= -1
        elif fault == "scale":
            force[...] *= 2
        elif fault == "axis":
            force[...] = force[..., [1, 0, 2]]
        elif fault == "delay":
            current = force.copy()
            force[...] = getattr(self, "_ci_previous_force", np.zeros_like(force))
            self._ci_previous_force = current
        return force.copy()

    VorLapQBladeRuntime.update = broken


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--mode", choices=("reference", "sil-vorlap", "qblade-external"), required=True)
    parser.add_argument("--execution", default="step-loop", choices=("step-loop", "run-full-simulation"))
    parser.add_argument("--azimuth", type=float, default=0)
    parser.add_argument("--dt", type=float, default=.002)
    parser.add_argument("--elements", type=int, default=20)
    parser.add_argument("--harmonic", action="store_true")
    parser.add_argument("--fault", default="none", choices=("none", "drop", "sign", "scale", "axis", "delay"))
    args = parser.parse_args()
    qroot = ROOT / "run_headless/QBladeCE_2.0.9.7"
    # Set the loader before constructing inputs: environment setup can re-exec.
    from vorlap.qblade_run_support import ensure_qblade_loader_env
    ensure_qblade_loader_env(qroot)
    case = args.case.resolve()
    case.mkdir(parents=True, exist_ok=False)
    sim, airfoils, fractions = make_case(case, azimuth=args.azimuth, dt=args.dt,
                                        elements=args.elements, harmonic=args.harmonic)
    channels = ["Time [s]"]
    for axis in "XYZ":
        channels.extend((f"{axis}_g Pos. BLD_1 pos 1.000 [m]",
                         f"{axis}_b Trl.Def. BLD_1 pos 1.000 [m]",
                         f"{axis}_b Mom. BLD_1 pos 0.000 [Nm]"))
    result_filter = case / "results.filter"
    result_filter.write_text("\n".join(channels) + "\n")
    inject_force_fault(args.fault)
    if args.mode == "reference":
        # Literal independent QBlade load file, bypassing every VorLap force,
        # frame, tributary-length and loading-file writer implementation.
        angle = np.deg2rad(args.azimuth)
        direction = np.array([1, .5 * np.cos(angle), .5 * np.sin(angle)])
        times = np.arange(round(2 / args.dt) + 1) * args.dt if args.harmonic else [0, 2]
        with (case / "loads.txt").open("w") as output:
            for i, fraction in enumerate(fractions):
                weight = (.5 if i in (0, len(fractions) - 1) else 1) / args.elements
                output.write(f"BLD_1_{fraction:.10f}\n")
                for t in times:
                    force = .98 * weight * direction * (np.cos(4 * np.pi * t) if args.harmonic else 1)
                    output.write(f"{t:.12g} {force[0]:.12g} {force[1]:.12g} {force[2]:.12g} 0 0 0\n")
                output.write("\n")
        sim.write_text(patch(sim.read_text(), {"LOADINGFILE": "loads.txt"}))
        result = run_qblade_case(QBladeRunRequest(sim_path=sim, qblade_root=qroot,
                                                output_dir=case / "out", result_filter=result_filter))
    else:
        config = QBladeBridgeConfig(sim_path=sim, qblade_root=qroot, output_dir=case / "out",
            airfoil_dir=airfoils, include_struts=False, include_mean_load=not args.harmonic,
            default_airfoil_id="CYLINDER", coupling_mode=args.mode, qblade_execution_mode=args.execution,
            record_sil_vorlap_history=args.mode == "sil-vorlap", debug=True,
            python_executable=sys.executable, fluid_density=1.225)
        result = QBladeBridgeService().run_case(config, check_paths=True).run_result
    receipt = {"arguments": vars(args) | {"case": str(case)}, "summary": str(result.summary_path),
               "inputs": {str(p.relative_to(case)): file_sha256(p)["sha256"]
                          for p in case.rglob("*") if p.is_file() and "out" not in p.parts},
               "source": {str(p.relative_to(ROOT)): file_sha256(p)["sha256"]
                          for p in list((ROOT / "vorlap").rglob("*.py"))
                          + list((ROOT / "qblade_external_bridge").rglob("*.cpp"))
                          + [Path(__file__).resolve()]}}
    (case / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
