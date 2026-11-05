#!/usr/bin/env python3
"""Run a parked VorLap/QBlade production-readiness campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vorlap.turbsim import (  # noqa: E402
    estimate_full_field_bytes,
    generate_turbsim_wind_file,
    quote_turbsim_string,
)
from vorlap.qblade_batch import (  # noqa: E402
    WAKE_PROFILE_CHOICES,
    WAKE_PROFILE_REFERENCE,
    completed_filtered_export_ready,
    reference_case_dir_for_root,
    reference_sim_name_for_root,
    vorlap_airfoil_dir_for_sim,
    wake_profile_settings,
)
from vorlap.qblade_case_prep import build_airfoil_source_manifest  # noqa: E402
from vorlap.dlc_manifest import sha256_file  # noqa: E402

COUPLING_CHOICES = ("sil-vorlap", "qblade-external", "none")
DEFAULT_POSTPROCESS_LOCATIONS = "0.230 0.500 0.770 0.940"
DEFAULT_POSTPROCESS_BLADE = "BLD_1"
DEFAULT_WOHLER_EXPONENT = 10.0
DEFAULT_EQUIVALENT_CYCLES = 600.0
SCENARIO_NAMES = ("baseline", "identity", "vorlap")


def atomic_write_json(path: Path, payload: object) -> None:
    """Publish shared campaign metadata without exposing a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_headless_root() -> Path:
    return Path(__file__).resolve().parent


def default_qblade_root(root: Path) -> Path:
    if "QBLADE_ROOT" in os.environ:
        return Path(os.environ["QBLADE_ROOT"])
    for candidate in (run_headless_root() / "QBladeCE_2.0.9.7", root / "QBladeCE_2.0.9.7"):
        if candidate.is_dir():
            return candidate
    return run_headless_root() / "QBladeCE_2.0.9.7"


def default_vorlap_root(root: Path) -> Path:
    return Path(os.environ.get("VORLAP_ROOT", root))


def campaign_airfoil_source_manifest(vorlap_root: Path) -> tuple[list[dict[str, object]], str]:
    try:
        reference_dir = reference_case_dir_for_root(vorlap_root)
        reference_sim = reference_dir / reference_sim_name_for_root(vorlap_root)
        airfoil_dir = vorlap_airfoil_dir_for_sim(reference_sim)
        return build_airfoil_source_manifest(airfoil_dir), ""
    except Exception as exc:
        return [], str(exc)


def speed_label(speed: float) -> str:
    return f"U{float(speed):05.2f}ms".replace(".", "p").replace("-", "m")


def azimuth_label(azimuth_deg: float) -> str:
    return f"A{float(azimuth_deg):06.2f}deg".replace(".", "p").replace("-", "m")


def parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {text!r}.")


def parse_float_spec(spec: str, *, name: str) -> list[float]:
    text = str(spec).strip()
    if not text:
        raise ValueError(f"{name} specification is empty.")
    if text.startswith("linspace:"):
        parts = text.split(":")
        if len(parts) != 4:
            raise ValueError(f"Use linspace:start:stop:count for {name} sweeps.")
        start, stop, count = float(parts[1]), float(parts[2]), int(parts[3])
        if count < 2:
            return [start]
        step = (stop - start) / (count - 1)
        return [start + i * step for i in range(count)]
    if text.startswith("range:"):
        parts = text.split(":")
        if len(parts) != 4:
            raise ValueError(f"Use range:start:stop:step for {name} sweeps.")
        start, stop, step = float(parts[1]), float(parts[2]), float(parts[3])
        if step == 0.0:
            raise ValueError(f"{name} range step cannot be zero.")
        if (stop - start) * step < 0.0:
            raise ValueError(f"{name} range step has the wrong sign for the requested stop value.")
        values: list[float] = []
        value = start
        tolerance = abs(step) * 1.0e-9 + 1.0e-12
        if step > 0.0:
            while value <= stop + tolerance:
                values.append(value)
                value += step
        else:
            while value >= stop - tolerance:
                values.append(value)
                value += step
        return values
    return [float(part) for part in text.replace(",", " ").split() if part]


def parse_wind_speeds(spec: str) -> list[float]:
    return parse_float_spec(spec, name="wind-speed")


def parse_azimuths(spec: str) -> list[float]:
    azimuths = parse_float_spec(spec, name="azimuth")
    for azimuth in azimuths:
        if not math.isfinite(azimuth) or azimuth < 0.0 or azimuth >= 360.0:
            raise ValueError(f"Azimuths must be finite values in [0, 360) deg; got {azimuth!r}.")
    return azimuths


def run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def validate_force_scale(name: str, force_scale: float, diagnostic_force_scale: bool = False) -> None:
    """Require explicit labeling for force scales above the nominal physical value."""
    if not math.isfinite(force_scale) or force_scale < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    if force_scale > 1.0 and not diagnostic_force_scale:
        raise ValueError(
            f"{name} values greater than 1.0 require --diagnostic-force-scale "
            "so exaggerated forcing cannot be mistaken for a nominal physical run."
        )


def batch_command(
    args: argparse.Namespace,
    scenario: str,
    force_scale: float,
    timesteps: int,
    *,
    coupling: str | None = None,
    command_name: str = "run-all",
    case_index: int | None = None,
    force_setup: bool | None = None,
) -> list[str]:
    script_dir = Path(__file__).resolve().parent
    coupling_mode = coupling or args.coupling
    command = [
        args.python,
        str(script_dir / "vorlap_qblade_batch.py"),
        command_name,
        "--run-root",
        str(args.run_root / scenario),
        "--qblade-root",
        str(args.qblade_root),
        "--vorlap-root",
        str(args.vorlap_root),
        "--python",
        args.python,
        "--wind-speeds",
        args.wind_speeds,
        "--azimuths",
        args.azimuths,
        "--timesteps",
        str(timesteps),
        "--timestep",
        f"{args.timestep:.9g}",
        "--store-from",
        f"{args.store_from:.9g}",
        "--store-replay",
        str(args.store_replay).lower(),
        "--rpm",
        f"{args.rpm:.9g}",
        "--horizontal-inflow-angle",
        f"{getattr(args, 'horizontal_inflow_angle', 0.0):.9g}",
        "--vertical-inflow-angle",
        f"{getattr(args, 'vertical_inflow_angle', 0.0):.9g}",
        "--shear-exponent",
        f"{getattr(args, 'shear_exponent', 0.23):.9g}",
        "--reference-height",
        f"{getattr(args, 'reference_height', 18.0):.9g}",
        "--air-density",
        f"{getattr(args, 'air_density', 1.0296):.9g}",
        "--air-kinematic-viscosity",
        f"{getattr(args, 'air_kinematic_viscosity', 1.647e-5):.9g}",
        "--wake-profile",
        str(getattr(args, "wake_profile", WAKE_PROFILE_REFERENCE)),
        "--rampup",
        f"{args.rampup:.9g}",
        "--add-damp",
        f"{args.add_damp:.9g}",
        "--force-scale",
        f"{force_scale:.9g}",
        "--source-bounds-policy",
        args.source_bounds_policy,
        "--source-frequency-truncation",
        str(getattr(args, "source_frequency_truncation", "none")),
        "--threads",
        str(args.threads),
        "--progress-every",
        str(args.progress_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--coupling",
        coupling_mode,
    ]
    if getattr(args, "result_filter", None) is not None:
        command.extend(["--result-filter", str(args.result_filter)])
    if getattr(args, "full_structure_strain_output", False):
        command.append("--full-structure-strain-output")
    if args.use_opencl:
        command.extend(
            [
                "--use-opencl",
                "--opencl-device",
                str(args.opencl_device),
                "--opencl-group-size",
                str(args.opencl_group_size),
            ]
        )
    if args.resume_initialize:
        command.append("--resume-initialize")
    if case_index is not None:
        command.extend(["--case-index", str(case_index)])
    if args.str_iterations is not None:
        command.extend(["--str-iterations", str(args.str_iterations)])
    if args.wind_name:
        command.extend(["--wind-type", "1", "--wind-name", args.wind_name])
    else:
        command.extend(["--wind-type", "0"])
    if args.qblade_steps is not None and command_name != "setup":
        command.extend(["--qblade-steps", str(args.qblade_steps)])
    if args.diagnostic_force_scale:
        command.append("--diagnostic-force-scale")
    if args.allow_partial and command_name != "setup":
        command.append("--allow-partial")
    if getattr(args, "native_run_full_simulation", False) and command_name != "setup":
        command.append("--native-run-full-simulation")
    if getattr(args, "store_project", False) and command_name != "setup":
        command.append("--store-project")
    if args.record_sil_vorlap_history and command_name != "setup":
        command.append("--record-sil-vorlap-history")
    if force_setup is None:
        force_setup = not args.reuse_setup
    if force_setup:
        command.append("--force-setup")
    if args.force_bridge_build:
        command.append("--force-bridge-build")
    if args.skip_vorlap_install:
        command.append("--skip-vorlap-install")
    if args.skip_bridge_build:
        command.append("--skip-bridge-build")
    if args.skip_stable_support_patch:
        command.append("--skip-stable-support-patch")
    if args.allow_coarse_timestep:
        command.append("--allow-coarse-timestep")
    return command


def case_count(run_root: Path) -> int:
    manifest_path = run_root / "cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return len(manifest["cases"])


def run_parallel(commands: list[list[str]], max_workers: int, *, continue_on_failure: bool = False) -> None:
    active: list[tuple[list[str], subprocess.Popen]] = []
    pending = list(commands)
    failures: list[tuple[list[str], int]] = []

    while pending or active:
        while pending and len(active) < max_workers:
            command = pending.pop(0)
            print("+ " + " ".join(command), flush=True)
            active.append((command, subprocess.Popen(command)))

        next_active: list[tuple[list[str], subprocess.Popen]] = []
        for command, process in active:
            return_code = process.poll()
            if return_code is None:
                next_active.append((command, process))
            elif return_code != 0:
                if continue_on_failure:
                    failures.append((command, return_code))
                    print(
                        f"WARNING: command exited {return_code}; continuing because --continue-on-failure is set.",
                        flush=True,
                    )
                else:
                    for _, active_process in active:
                        if active_process.poll() is None:
                            active_process.terminate()
                    for _, active_process in active:
                        if active_process is not process:
                            active_process.wait()
                    raise subprocess.CalledProcessError(return_code, command)
        active = next_active

        if active:
            time.sleep(1.0)

    if failures:
        print(f"WARNING: {len(failures)} case command(s) failed; postprocessing available outputs.", flush=True)


def postprocess_command(args: argparse.Namespace) -> list[str]:
    script_dir = Path(__file__).resolve().parent
    scenarios = campaign_scenarios(args)
    scenario_names = [str(scenario["name"]) for scenario in scenarios]
    if "identity" in scenario_names and "vorlap" in scenario_names:
        comparison_baseline = "identity"
        comparison_coupled = "vorlap"
    elif "baseline" in scenario_names and "vorlap" in scenario_names:
        comparison_baseline = "baseline"
        comparison_coupled = "vorlap"
    elif len(scenario_names) >= 2:
        comparison_baseline, comparison_coupled = scenario_names[:2]
    else:
        raise ValueError("Postprocessing requires at least two selected scenarios.")
    command = [
        args.python,
        str(script_dir / "postprocess_parked_demo.py"),
        "--run-root",
        str(args.run_root),
        "--out-dir",
        str(args.run_root / "postprocess"),
        "--scenarios",
        " ".join(scenario_names),
        "--comparison-baseline",
        comparison_baseline,
        "--comparison-coupled",
        comparison_coupled,
        "--locations",
        args.postprocess_locations,
        "--blade",
        args.postprocess_blade,
        "--wohler-exponent",
        f"{args.wohler_exponent:.9g}",
        "--equivalent-cycles",
        f"{float(getattr(args, 'equivalent_cycles', DEFAULT_EQUIVALENT_CYCLES)):.9g}",
        "--min-fatigue-duration",
        f"{float(getattr(args, 'min_fatigue_duration', 600.0)):.9g}",
    ]
    if args.ultimate_allowables is not None:
        command.extend(["--ultimate-allowables", str(args.ultimate_allowables)])
    if getattr(args, "include_smoke_in_del", False):
        command.append("--include-smoke-in-del")
    return command


def maybe_postprocess_command(args: argparse.Namespace) -> list[str] | None:
    if args.skip_postprocess:
        return None
    if len(campaign_scenarios(args)) < 2:
        return None
    return postprocess_command(args)


def apply_postprocess_defaults(args: argparse.Namespace, settings: dict | None = None) -> None:
    settings = settings or {}
    if args.postprocess_locations is None:
        args.postprocess_locations = str(settings.get("postprocess_locations", DEFAULT_POSTPROCESS_LOCATIONS))
    if args.postprocess_blade is None:
        args.postprocess_blade = str(settings.get("postprocess_blade", DEFAULT_POSTPROCESS_BLADE))
    if args.wohler_exponent is None:
        args.wohler_exponent = float(settings.get("wohler_exponent", DEFAULT_WOHLER_EXPONENT))
    if getattr(args, "equivalent_cycles", None) is None:
        args.equivalent_cycles = float(
            settings.get("equivalent_cycles", DEFAULT_EQUIVALENT_CYCLES)
        )
    if getattr(args, "min_fatigue_duration", None) is None:
        args.min_fatigue_duration = float(settings.get("min_fatigue_duration_s", 600.0))
    if args.ultimate_allowables is None and settings.get("ultimate_allowables"):
        args.ultimate_allowables = Path(str(settings["ultimate_allowables"])).expanduser().resolve()


def planned_cases(args: argparse.Namespace) -> list[dict[str, float | str]]:
    cases: list[dict[str, float | str]] = []
    azimuths = parse_azimuths(args.azimuths)
    for speed in parse_wind_speeds(args.wind_speeds):
        speed_text = speed_label(speed)
        for azimuth in azimuths:
            azimuth_text = azimuth_label(azimuth)
            label = speed_text if len(azimuths) == 1 else f"{speed_text}_{azimuth_text}"
            cases.append(
                {
                    "wind_speed_mps": speed,
                    "parked_azimuth_deg": azimuth,
                    "speed_label": speed_text,
                    "azimuth_label": azimuth_text,
                    "label": label,
                }
            )
    return cases


def completed_scenario_can_be_preserved(args: argparse.Namespace, scenario: str) -> bool:
    """Return true only when every planned result can be reused without setup."""
    if not getattr(args, "preserve_completed_scenarios", False) or args.result_filter is None:
        return False
    manifest_path = args.run_root / scenario / "cases.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = manifest.get("cases")
        if not isinstance(cases, list):
            return False
        expected_labels = [str(case["label"]) for case in planned_cases(args)]
        observed_labels = [str(case.get("label", "")) for case in cases]
        if observed_labels != expected_labels:
            return False
        return bool(cases) and all(
            completed_filtered_export_ready(case, args.result_filter) for case in cases
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def preserve_completed_scenario_or_run(
    args: argparse.Namespace,
    scenario: str,
    command: list[str],
) -> None:
    if completed_scenario_can_be_preserved(args, scenario):
        print(
            f"Preserving completed {scenario} scenario without setup or QBlade timestep advancement.",
            flush=True,
        )
        return
    run_checked(command)


def format_case_template(template: str, case: dict[str, float | str]) -> str:
    speed = float(case["wind_speed_mps"])
    azimuth = float(case["parked_azimuth_deg"])
    label = str(case.get("speed_label") or speed_label(speed))
    azi_label = str(case.get("azimuth_label") or azimuth_label(azimuth))
    case_label = str(case.get("label") or label)
    return template.format(
        speed=speed,
        speed_label=label,
        azimuth=azimuth,
        azimuth_label=azi_label,
        label=case_label,
    )


def turbsim_settings_for_case(args: argparse.Namespace, case: dict[str, float | str], index: int) -> dict[str, object]:
    analysis_time = args.turbsim_analysis_time if args.turbsim_analysis_time is not None else args.duration
    usable_time: str | float = args.turbsim_usable_time if args.turbsim_usable_time is not None else "ALL"
    wind_speed = float(case["wind_speed_mps"])
    seed = int(args.turbsim_seed_base) + int(index)
    return {
        "RandSeed1": seed,
        "RandSeed2": quote_turbsim_string(args.turbsim_seed2),
        "NumGrid_Z": args.turbsim_num_grid_z,
        "NumGrid_Y": args.turbsim_num_grid_y,
        "TimeStep": args.turbsim_time_step,
        "AnalysisTime": analysis_time,
        "UsableTime": quote_turbsim_string(str(usable_time)) if isinstance(usable_time, str) else usable_time,
        "HubHt": args.turbsim_hub_height,
        "GridHeight": args.turbsim_grid_height,
        "GridWidth": args.turbsim_grid_width,
        "TurbModel": quote_turbsim_string(args.turbsim_turb_model),
        "IECstandard": quote_turbsim_string(args.turbsim_iec_standard)
        if not str(args.turbsim_iec_standard).isdigit()
        else int(args.turbsim_iec_standard),
        "IECturbc": quote_turbsim_string(args.turbsim_iec_turbc),
        "IEC_WindType": quote_turbsim_string(args.turbsim_iec_wind_type),
        "RefHt": args.turbsim_ref_height,
        "URef": wind_speed,
    }


def turbsim_wind_records(args: argparse.Namespace, cases: list[dict[str, float | str]], *, dry_run: bool) -> list[dict]:
    if not args.generate_turbsim:
        return []
    records: list[dict] = []
    for index, case in enumerate(cases):
        wind_file = Path(format_case_template(args.wind_name, case)).expanduser()
        settings = turbsim_settings_for_case(args, case, index)
        estimate = estimate_full_field_bytes(
            analysis_time_s=float(settings["AnalysisTime"]),
            time_step_s=float(settings["TimeStep"]),
            num_grid_y=int(settings["NumGrid_Y"]),
            num_grid_z=int(settings["NumGrid_Z"]),
        )
        if dry_run:
            records.append(
                {
                    "wind_file": str(wind_file),
                    "input_file": str(wind_file.with_suffix(".inp")),
                    "settings": settings,
                    "estimated_full_field_bytes": estimate,
                    "dry_run": True,
                }
            )
            continue
        run = generate_turbsim_wind_file(
            wind_file=wind_file,
            settings=settings,
            executable=args.turbsim_exe,
            template_path=args.turbsim_template,
            vorlap_root=args.vorlap_root,
            force=args.force_generate_turbsim,
            adopt_existing=getattr(args, "adopt_existing_turbsim", False),
        )
        record = run.as_dict()
        record["estimated_full_field_bytes"] = estimate
        record["wind_file_bytes"] = run.wind_file.stat().st_size
        record["wind_file_sha256"] = sha256_file(run.wind_file)
        record["input_file_sha256"] = sha256_file(run.input_file)
        record["metadata_file_sha256"] = sha256_file(run.metadata_file)
        records.append(record)
    return records


def parse_scenarios(spec: str) -> list[str]:
    scenarios = [part.strip() for part in spec.replace(",", " ").split() if part.strip()]
    if not scenarios:
        raise ValueError("At least one scenario must be selected.")
    unknown = [scenario for scenario in scenarios if scenario not in SCENARIO_NAMES]
    if unknown:
        raise ValueError(f"Unknown scenario(s): {', '.join(unknown)}. Valid scenarios: {', '.join(SCENARIO_NAMES)}.")
    duplicates = sorted({scenario for scenario in scenarios if scenarios.count(scenario) > 1})
    if duplicates:
        raise ValueError(f"Duplicate scenario(s): {', '.join(duplicates)}.")
    return scenarios


def campaign_scenarios(args: argparse.Namespace) -> list[dict[str, float | str]]:
    """Return the explicit control/identity/nominal scenarios for each parked case."""
    scenario_defs = {
        "baseline": {
            "name": "baseline",
            "role": "no_sil_vorlap_force",
            "force_scale": args.baseline_force_scale,
            "coupling": "none",
        },
        "identity": {
            "name": "identity",
            "role": "bridge_active_zero_force",
            "force_scale": args.identity_force_scale,
            "coupling": args.identity_coupling,
        },
        "vorlap": {
            "name": "vorlap",
            "role": "nominal_vorlap_force",
            "force_scale": args.vorlap_force_scale,
            "coupling": args.coupling,
        },
    }
    return [scenario_defs[name] for name in parse_scenarios(getattr(args, "scenarios", "baseline vorlap"))]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_duration = float(os.environ.get("QBLADE_DURATION", "660.0"))
    duration_minutes_env = os.environ.get("QBLADE_DURATION_MINUTES", os.environ.get("SIM_MINUTES", ""))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=run_headless_root() / "runs" / "vorlap_parked_demo")
    parser.add_argument("--qblade-root", type=Path, default=default_qblade_root(root))
    parser.add_argument("--vorlap-root", type=Path, default=default_vorlap_root(root))
    parser.add_argument("--python", default=os.environ.get("PYTHON_EXE", sys.executable))
    parser.add_argument("--wind-speeds", default=os.environ.get("WIND_SPEEDS", "5 10 30 50"))
    parser.add_argument(
        "--azimuths",
        default=os.environ.get("QBLADE_AZIMUTHS", "270"),
        help="Initial azimuths in deg. Accepts lists, linspace:start:stop:count, or range:start:stop:step.",
    )
    parser.add_argument("--duration", type=float, default=default_duration)
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=None if not duration_minutes_env else float(duration_minutes_env),
        help="Simulation duration in minutes. Overrides --duration when provided.",
    )
    parser.add_argument("--timestep", type=float, default=float(os.environ.get("QBLADE_TIMESTEP", "0.0075")))
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument(
        "--scenarios",
        default=os.environ.get("CAMPAIGN_SCENARIOS", "baseline vorlap"),
        help="Whitespace/comma-separated scenarios to run. Valid values: baseline, identity, vorlap.",
    )
    parser.add_argument("--store-from", type=float, default=float(os.environ.get("QBLADE_STORE_FROM", "0.0")))
    parser.add_argument(
        "--store-replay",
        type=parse_bool,
        default=parse_bool(os.environ.get("QBLADE_STORE_REPLAY", "false")),
        help="Set QBlade STOREREPLAY in generated .sim files.",
    )
    parser.add_argument(
        "--result-filter",
        type=Path,
        default=None
        if not os.environ.get("QBLADE_RESULT_FILTER", "").strip()
        else Path(os.environ["QBLADE_RESULT_FILTER"]),
        help=(
            "Compact QBlade result-export filter. SIL setup augments the generated .sim "
            "storage filter with the live bridge input channels."
        ),
    )
    parser.add_argument(
        "--full-structure-strain-output",
        action="store_true",
        default=parse_bool(os.environ.get("QBLADE_FULL_STRUCTURE_STRAIN_OUTPUT", "false")),
        help=(
            "Generate and retain Fz/Mx/My at dense blade, strut, upper/lower "
            "tower, and torque-tube stations for surface normal-strain DEL mapping."
        ),
    )
    parser.add_argument(
        "--store-project",
        action="store_true",
        default=parse_bool(os.environ.get("QBLADE_STORE_PROJECT", "false")),
        help="Store a completed QBlade .qpr project for each successful case.",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="Defaults to 0 in the low-level driver; production DLC6.x wrappers pass 1 for idling cases.",
    )
    parser.add_argument(
        "--horizontal-inflow-angle",
        type=float,
        default=float(os.environ.get("QBLADE_HORIZONTAL_INFLOW_ANGLE", "0.0")),
        help="QBlade horizontal inflow angle [deg]; explicitly defaults to aligned inflow.",
    )
    parser.add_argument(
        "--vertical-inflow-angle",
        type=float,
        default=float(os.environ.get("QBLADE_VERTICAL_INFLOW_ANGLE", "0.0")),
        help="QBlade vertical inflow angle [deg].",
    )
    parser.add_argument(
        "--shear-exponent",
        type=float,
        default=float(os.environ.get("QBLADE_SHEAR_EXPONENT", "0.23")),
        help=(
            "Power-law shear exponent written to QBlade. Imported TurbSim fields retain "
            "the profile encoded by their immutable .inp/.bts provenance."
        ),
    )
    parser.add_argument(
        "--reference-height",
        type=float,
        default=float(os.environ.get("QBLADE_REFERENCE_HEIGHT", "18.0")),
        help="QBlade boundary-layer reference height [m].",
    )
    parser.add_argument(
        "--air-density",
        type=float,
        default=float(os.environ.get("QBLADE_AIR_DENSITY", "1.0296")),
        help="Air density [kg/m^3]; the default preserves the reference-model basis.",
    )
    parser.add_argument(
        "--air-kinematic-viscosity",
        type=float,
        default=float(os.environ.get("QBLADE_AIR_KINEMATIC_VISCOSITY", "1.647e-5")),
        help="Air kinematic viscosity [m^2/s].",
    )
    parser.add_argument(
        "--wake-profile",
        choices=WAKE_PROFILE_CHOICES,
        default=os.environ.get("QBLADE_WAKE_PROFILE", WAKE_PROFILE_REFERENCE),
        help=(
            "Generated-case QBlade aerodynamic/wake profile. Reduced profiles are frozen to the "
            "qualified 1 RPM and 0.0075 s parked-study basis."
        ),
    )
    parser.add_argument(
        "--quasi-static-third-rev",
        action="store_true",
        help="Use the older diagnostic convention of one third revolution over --duration when --rpm is omitted.",
    )
    parser.add_argument("--rampup", type=float, default=float(os.environ.get("QBLADE_RAMPUP", "0.0")))
    parser.add_argument("--add-damp", type=float, default=float(os.environ.get("QBLADE_ADD_DAMP", "0.0")))
    parser.add_argument(
        "--str-iterations",
        type=int,
        default=None if "QBLADE_STR_ITERATIONS" not in os.environ else int(os.environ["QBLADE_STR_ITERATIONS"]),
    )
    parser.add_argument(
        "--baseline-force-scale",
        type=float,
        default=0.0,
        help="Must remain zero; the baseline scenario disables SIL-applied VorLap forces.",
    )
    parser.add_argument(
        "--identity-force-scale",
        type=float,
        default=0.0,
        help="Must remain zero; the identity scenario keeps the bridge active with zero returned force.",
    )
    parser.add_argument("--vorlap-force-scale", type=float, default=float(os.environ.get("VORLAP_FORCE_SCALE", "1.0")))
    parser.add_argument(
        "--source-bounds-policy",
        choices=("warn", "raise", "clip"),
        default=os.environ.get("VORLAP_SOURCE_BOUNDS_POLICY", "warn"),
        help=(
            "Runtime policy for VorLap airfoil FFT source-grid excursions. "
            "'warn' clips and records source-coverage diagnostics; 'raise' fails fast."
        ),
    )
    parser.add_argument(
        "--source-frequency-truncation",
        choices=("none", "nyquist"),
        default=os.environ.get("VORLAP_SOURCE_FREQUENCY_TRUNCATION", "none"),
        help=(
            "Set to 'nyquist' to discard VorLap oscillatory tones above the QBlade "
            "Nyquist frequency 0.5/TIMESTEP. The default leaves the spectrum unfiltered."
        ),
    )
    parser.add_argument(
        "--diagnostic-force-scale",
        action="store_true",
        help="Allow force scales above 1.0 for explicit diagnostic/amplitude-sensitivity runs.",
    )
    parser.add_argument("--wind-name", default=os.environ.get("QBLADE_WIND_NAME", ""))
    parser.add_argument(
        "--generate-turbsim",
        action="store_true",
        default=parse_bool(os.environ.get("GENERATE_TURBSIM", "false")),
        help="Generate missing TurbSim wind files before QBlade setup/run.",
    )
    parser.add_argument(
        "--force-generate-turbsim",
        action="store_true",
        default=parse_bool(os.environ.get("FORCE_GENERATE_TURBSIM", "false")),
        help="Regenerate TurbSim wind files even when the expected output already exists.",
    )
    parser.add_argument(
        "--adopt-existing-turbsim",
        action="store_true",
        default=parse_bool(os.environ.get("ADOPT_EXISTING_TURBSIM", "false")),
        help=(
            "Create immutable provenance for a legacy wind file only when its existing .inp "
            "exactly matches the requested TurbSim settings. This is non-default migration behavior."
        ),
    )
    parser.add_argument("--turbsim-exe", default=os.environ.get("TURBSIM_EXE", ""))
    parser.add_argument(
        "--turbsim-template",
        type=Path,
        default=None if "TURBSIM_TEMPLATE" not in os.environ else Path(os.environ["TURBSIM_TEMPLATE"]),
    )
    parser.add_argument(
        "--wind-root",
        type=Path,
        default=None if "TURBSIM_WIND_ROOT" not in os.environ else Path(os.environ["TURBSIM_WIND_ROOT"]),
        help="Directory used for generated TurbSim wind files when --wind-name is relative or omitted.",
    )
    parser.add_argument(
        "--turbsim-time-step",
        type=float,
        default=None if "TURBSIM_TIMESTEP" not in os.environ else float(os.environ["TURBSIM_TIMESTEP"]),
    )
    parser.add_argument(
        "--allow-coarse-wind-timestep",
        action="store_true",
        default=parse_bool(os.environ.get("ALLOW_COARSE_WIND_TIMESTEP", "false")),
        help="Allow TurbSim TimeStep to be larger than the QBlade timestep.",
    )
    parser.add_argument(
        "--turbsim-analysis-time",
        type=float,
        default=None if "TURBSIM_ANALYSIS_TIME" not in os.environ else float(os.environ["TURBSIM_ANALYSIS_TIME"]),
    )
    parser.add_argument(
        "--turbsim-usable-time",
        default=None if "TURBSIM_USABLE_TIME" not in os.environ else os.environ["TURBSIM_USABLE_TIME"],
    )
    parser.add_argument("--turbsim-num-grid-y", type=int, default=int(os.environ.get("TURBSIM_NUM_GRID_Y", "26")))
    parser.add_argument("--turbsim-num-grid-z", type=int, default=int(os.environ.get("TURBSIM_NUM_GRID_Z", "38")))
    parser.add_argument("--turbsim-grid-width", type=float, default=float(os.environ.get("TURBSIM_GRID_WIDTH", "26")))
    parser.add_argument("--turbsim-grid-height", type=float, default=float(os.environ.get("TURBSIM_GRID_HEIGHT", "26")))
    parser.add_argument("--turbsim-hub-height", type=float, default=float(os.environ.get("TURBSIM_HUB_HEIGHT", "18")))
    parser.add_argument(
        "--turbsim-ref-height",
        type=float,
        default=float(os.environ.get("TURBSIM_REF_HEIGHT", os.environ.get("QBLADE_REFERENCE_HEIGHT", "18"))),
    )
    parser.add_argument("--turbsim-turb-model", default=os.environ.get("TURBSIM_TURB_MODEL", "IECKAI"))
    parser.add_argument("--turbsim-iec-standard", default=os.environ.get("TURBSIM_IEC_STANDARD", "1"))
    parser.add_argument("--turbsim-iec-turbc", default=os.environ.get("TURBSIM_IEC_TURBC", "A"))
    parser.add_argument("--turbsim-iec-wind-type", default=os.environ.get("TURBSIM_IEC_WIND_TYPE", "NTM"))
    parser.add_argument("--turbsim-seed-base", type=int, default=int(os.environ.get("TURBSIM_SEED_BASE", "4433456")))
    parser.add_argument("--turbsim-seed2", default=os.environ.get("TURBSIM_SEED2", "RanLux"))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    parser.add_argument(
        "--use-opencl",
        action="store_true",
        default=parse_bool(os.environ.get("QBLADE_USE_OPENCL", "false")),
        help="Pass OpenCL acceleration flags through to QBlade case runs.",
    )
    parser.add_argument("--opencl-device", type=int, default=int(os.environ.get("QBLADE_OPENCL_DEVICE", "1")))
    parser.add_argument("--opencl-group-size", type=int, default=int(os.environ.get("QBLADE_OPENCL_GROUP_SIZE", "32")))
    parser.add_argument("--case-workers", type=int, default=int(os.environ.get("CASE_WORKERS", "2")))
    parser.add_argument(
        "--case-index",
        type=int,
        default=None,
        help="Run only one planned case index for each scenario. Intended for one-node-per-case Slurm arrays.",
    )
    parser.add_argument("--qblade-steps", type=int, default=None, help="Short-run debug override.")
    parser.add_argument(
        "--native-run-full-simulation",
        action="store_true",
        help=(
            "Diagnostic qblade-external mode: pass --native-run-full-simulation to "
            "the lower-level runner so QBlade owns the full simulation loop."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=int(os.environ.get("QBLADE_PROGRESS_EVERY", "10")))
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(os.environ.get("QBLADE_CHECKPOINT_EVERY", "0")),
        help="Store/reuse QBlade project restart checkpoints every N completed timesteps; 0 disables.",
    )
    parser.add_argument(
        "--resume-initialize",
        action="store_true",
        help="Call initializeSimulation() after loading a restart checkpoint project.",
    )
    parser.add_argument(
        "--coupling",
        choices=COUPLING_CHOICES,
        default=os.environ.get("VORLAP_QBLADE_COUPLING", "sil-vorlap"),
        help="Coupling path for the nominal vorlap scenario.",
    )
    parser.add_argument(
        "--identity-coupling",
        choices=("sil-vorlap", "qblade-external"),
        default=os.environ.get("VORLAP_QBLADE_IDENTITY_COUPLING", "sil-vorlap"),
        help="Bridge-active coupling path for the zero-force identity scenario.",
    )
    parser.add_argument(
        "--allow-coarse-timestep",
        action="store_true",
        help="Allow timesteps above 0.01 s despite observed QBlade structural divergence.",
    )
    parser.add_argument("--postprocess-locations", default=None)
    parser.add_argument("--postprocess-blade", default=None)
    parser.add_argument("--wohler-exponent", type=float, default=None)
    parser.add_argument(
        "--equivalent-cycles",
        type=float,
        default=(
            None
            if "POSTPROCESS_EQUIVALENT_CYCLES" not in os.environ
            else float(os.environ["POSTPROCESS_EQUIVALENT_CYCLES"])
        ),
        help="Fixed reference cycle count passed to fatigue postprocessing.",
    )
    parser.add_argument(
        "--min-fatigue-duration",
        type=float,
        default=(
            None
            if "POSTPROCESS_MIN_FATIGUE_DURATION" not in os.environ
            else float(os.environ["POSTPROCESS_MIN_FATIGUE_DURATION"])
        ),
        help="Minimum retained result duration admitted to paper-facing fatigue products.",
    )
    parser.add_argument(
        "--include-smoke-in-del",
        action="store_true",
        default=parse_bool(os.environ.get("POSTPROCESS_INCLUDE_SMOKE_IN_DEL", "false")),
        help="Include short smoke-run rows in postprocessed DEL comparison products.",
    )
    parser.add_argument(
        "--ultimate-allowables",
        type=Path,
        default=None,
        help="Optional allowables CSV passed through to postprocess_parked_demo.py.",
    )
    parser.add_argument("--reuse-setup", action="store_true", help="Reuse existing generated case directories.")
    parser.add_argument(
        "--preserve-completed-scenarios",
        action="store_true",
        default=parse_bool(os.environ.get("PRESERVE_COMPLETED_SCENARIOS", "false")),
        help=(
            "Skip setup and execution for a scenario only when every planned result is a complete, "
            "successful, hash-verified filtered export. Intended for narrow recovery submissions."
        ),
    )
    parser.add_argument("--force-bridge-build", action="store_true")
    parser.add_argument("--skip-vorlap-install", action="store_true")
    parser.add_argument("--skip-bridge-build", action="store_true")
    parser.add_argument("--skip-stable-support-patch", action="store_true")
    parser.add_argument(
        "--record-sil-vorlap-history",
        action="store_true",
        help="For SIL VorLap scenario runs, write per-step velocity-input and force-output history CSVs.",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Prepare all scenario case directories/manifests and exit before launching QBlade runs.",
    )
    parser.add_argument(
        "--wind-only",
        action="store_true",
        default=parse_bool(os.environ.get("WIND_ONLY", "false")),
        help="Generate and archive requested TurbSim wind files, then exit before QBlade setup.",
    )
    parser.add_argument("--only-postprocess", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write campaign_settings.json and print the planned case matrix/commands without preparing or running QBlade.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Ask the QBlade runner to export completed timesteps after a QBlade timestep failure.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Keep launching remaining cases if one parallel QBlade process exits nonzero.",
    )
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.qblade_root = args.qblade_root.resolve()
    args.vorlap_root = args.vorlap_root.resolve()
    if args.result_filter is not None:
        args.result_filter = args.result_filter.expanduser().resolve()
        if not args.result_filter.is_file():
            raise FileNotFoundError(f"QBlade result filter does not exist: {args.result_filter}")
    if args.wind_root is None:
        args.wind_root = (args.run_root / "wind_fields").resolve()
    else:
        args.wind_root = args.wind_root.expanduser().resolve()
    if args.turbsim_template is not None:
        args.turbsim_template = args.turbsim_template.expanduser().resolve()
    if args.turbsim_time_step is None:
        args.turbsim_time_step = args.timestep
    if args.generate_turbsim:
        if args.turbsim_time_step > args.timestep and not args.allow_coarse_wind_timestep:
            raise ValueError(
                f"Refusing TurbSim TimeStep {args.turbsim_time_step:g} s because it is coarser than "
                f"QBlade timestep {args.timestep:g} s. Use --allow-coarse-wind-timestep only after "
                "a wind-field interpolation sensitivity check."
            )
        if args.turbsim_analysis_time is not None and args.turbsim_analysis_time <= 0.0:
            raise ValueError("--turbsim-analysis-time must be positive when provided.")
        if args.turbsim_num_grid_y < 2 or args.turbsim_num_grid_z < 2:
            raise ValueError("TurbSim grid dimensions must both be at least 2.")
        if not args.wind_name:
            args.wind_name = str(args.wind_root / "{speed_label}" / "dlc6p4_{speed_label}.bts")
        elif not Path(args.wind_name).expanduser().is_absolute():
            args.wind_name = str(args.wind_root / args.wind_name)
    if args.ultimate_allowables is not None:
        args.ultimate_allowables = args.ultimate_allowables.resolve()
    parse_scenarios(args.scenarios)
    if args.duration_minutes is not None:
        args.duration = args.duration_minutes * 60.0
    validate_force_scale("--baseline-force-scale", args.baseline_force_scale, args.diagnostic_force_scale)
    validate_force_scale("--identity-force-scale", args.identity_force_scale, args.diagnostic_force_scale)
    validate_force_scale("--vorlap-force-scale", args.vorlap_force_scale, args.diagnostic_force_scale)
    if not math.isclose(args.baseline_force_scale, 0.0, abs_tol=1.0e-12):
        raise ValueError("--baseline-force-scale must remain 0.0 because baseline disables VorLap force application.")
    if not math.isclose(args.identity_force_scale, 0.0, abs_tol=1.0e-12):
        raise ValueError("--identity-force-scale must remain 0.0 because identity is the zero-force bridge check.")
    if args.rpm is None and args.quasi_static_third_rev:
        args.rpm = 20.0 / args.duration
    elif args.rpm is None:
        args.rpm = 0.0
    return args


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.timestep <= 0.0:
        raise ValueError("--timestep must be positive.")
    if args.air_density <= 0.0 or not math.isfinite(args.air_density):
        raise ValueError("--air-density must be finite and positive.")
    if args.air_kinematic_viscosity <= 0.0 or not math.isfinite(args.air_kinematic_viscosity):
        raise ValueError("--air-kinematic-viscosity must be finite and positive.")
    if args.reference_height <= 0.0 or not math.isfinite(args.reference_height):
        raise ValueError("--reference-height must be finite and positive.")
    if args.timestep > 0.01 and not args.allow_coarse_timestep:
        raise ValueError(
            f"Refusing coarse QBlade timestep {args.timestep:g} s. The 10 s Docker "
            "campaign was unstable at 0.1 s; use 0.0075 s or pass --allow-coarse-timestep "
            "for an explicit diagnostic run."
        )
    if args.case_workers < 1:
        raise ValueError("--case-workers must be positive.")
    if args.case_index is not None and args.case_index < 0:
        raise ValueError("--case-index must be non-negative.")
    if args.setup_only and args.only_postprocess:
        raise ValueError("--setup-only and --only-postprocess are mutually exclusive.")
    if args.wind_only and (args.setup_only or args.only_postprocess):
        raise ValueError("--wind-only is mutually exclusive with --setup-only and --only-postprocess.")
    if args.wind_only and not args.generate_turbsim:
        raise ValueError("--wind-only requires --generate-turbsim.")
    if args.dry_run and args.only_postprocess:
        raise ValueError("--dry-run and --only-postprocess are mutually exclusive.")
    settings_path = args.run_root / "campaign_settings.json"
    existing_settings: dict = {}
    if args.only_postprocess:
        if not settings_path.is_file():
            raise FileNotFoundError(f"--only-postprocess requires an existing campaign settings file: {settings_path}")
        existing_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    apply_postprocess_defaults(args, existing_settings)
    timesteps = args.timesteps or int(math.ceil(args.duration / args.timestep))
    cases = planned_cases(args)
    scenarios = campaign_scenarios(args)
    if args.case_index is not None and args.case_index >= len(cases):
        raise IndexError(f"--case-index {args.case_index} is outside 0..{len(cases) - 1}.")
    args.run_root.mkdir(parents=True, exist_ok=True)
    turbsim_records = [] if args.only_postprocess else turbsim_wind_records(args, cases, dry_run=args.dry_run)
    airfoil_source_manifest, airfoil_source_manifest_error = campaign_airfoil_source_manifest(
        args.vorlap_root
    )
    dlc_manifest_row_archive = str(os.environ.get("DLC_MANIFEST_ROW_ARCHIVE", "") or "").strip()
    dlc_manifest_row: dict | None = None
    dlc_manifest_row_error = ""
    if dlc_manifest_row_archive:
        try:
            dlc_manifest_row = json.loads(
                Path(dlc_manifest_row_archive).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            dlc_manifest_row_error = str(exc)

    settings = {
        "campaign_settings_version": 2,
        "wind_speeds": parse_wind_speeds(args.wind_speeds),
        "azimuths_deg": parse_azimuths(args.azimuths),
        "planned_case_count_per_scenario": len(cases),
        "planned_case_count_total": len(scenarios) * len(cases),
        "scenario_names": [scenario["name"] for scenario in scenarios],
        "scenario_roles": {scenario["name"]: scenario["role"] for scenario in scenarios},
        "scenario_force_scales": {scenario["name"]: scenario["force_scale"] for scenario in scenarios},
        "scenario_couplings": {scenario["name"]: scenario["coupling"] for scenario in scenarios},
        "duration_s": args.duration,
        "timestep_s": args.timestep,
        "timesteps": timesteps,
        "qblade_steps_override": args.qblade_steps,
        "native_run_full_simulation": args.native_run_full_simulation,
        "store_from_s": args.store_from,
        "store_replay": args.store_replay,
        "result_filter": None if args.result_filter is None else str(args.result_filter),
        "result_filter_sha256": None
        if args.result_filter is None
        else sha256_file(args.result_filter),
        "store_project": args.store_project,
        "checkpoint_every": args.checkpoint_every,
        "resume_initialize": args.resume_initialize,
        "rpm": args.rpm,
        "rpm_prescribe_type": 1,
        "rpm_audit_tolerance_rpm": 0.01,
        "rpm_audit_startup_allowance_s": 0.1,
        "horizontal_inflow_angle_deg": args.horizontal_inflow_angle,
        "vertical_inflow_angle_deg": args.vertical_inflow_angle,
        "shear_exponent": args.shear_exponent,
        "reference_height_m": args.reference_height,
        "air_density_kg_m3": args.air_density,
        "air_kinematic_viscosity_m2_s": args.air_kinematic_viscosity,
        "wake_profile": args.wake_profile,
        "wake_settings": wake_profile_settings(args.wake_profile),
        "quasi_static_third_rev": args.quasi_static_third_rev,
        "baseline_force_scale": args.baseline_force_scale,
        "identity_force_scale": args.identity_force_scale,
        "vorlap_force_scale": args.vorlap_force_scale,
        "source_bounds_policy": args.source_bounds_policy,
        "source_frequency_truncation": getattr(args, "source_frequency_truncation", "none"),
        "source_frequency_cap_hz": (
            None
            if getattr(args, "source_frequency_truncation", "none") == "none"
            else 0.5 / args.timestep
        ),
        "baseline_coupling": "none",
        "identity_coupling": args.identity_coupling,
        "vorlap_coupling": args.coupling,
        "diagnostic_force_scale": args.diagnostic_force_scale,
        "rampup_s": args.rampup,
        "add_damp_s": args.add_damp,
        "str_iterations": args.str_iterations,
        "qblade_root": str(args.qblade_root),
        "vorlap_root": str(args.vorlap_root),
        "use_opencl": args.use_opencl,
        "opencl_device": args.opencl_device,
        "opencl_group_size": args.opencl_group_size,
        "stable_support_patch": not args.skip_stable_support_patch,
        "wind_name": args.wind_name,
        "generate_turbsim": args.generate_turbsim,
        "wind_only": args.wind_only,
        "force_generate_turbsim": args.force_generate_turbsim,
        "adopt_existing_turbsim": getattr(args, "adopt_existing_turbsim", False),
        "turbsim_exe": args.turbsim_exe,
        "turbsim_template": None if args.turbsim_template is None else str(args.turbsim_template),
        "turbsim_time_step_s": args.turbsim_time_step,
        "turbsim_analysis_time_s": args.turbsim_analysis_time,
        "turbsim_usable_time_s": args.turbsim_usable_time,
        "turbsim_num_grid_y": args.turbsim_num_grid_y,
        "turbsim_num_grid_z": args.turbsim_num_grid_z,
        "turbsim_grid_width_m": args.turbsim_grid_width,
        "turbsim_grid_height_m": args.turbsim_grid_height,
        "turbsim_hub_height_m": args.turbsim_hub_height,
        "turbsim_ref_height_m": args.turbsim_ref_height,
        "turbsim_turb_model": args.turbsim_turb_model,
        "turbsim_iec_standard": args.turbsim_iec_standard,
        "turbsim_iec_turbc": args.turbsim_iec_turbc,
        "turbsim_iec_wind_type": args.turbsim_iec_wind_type,
        "turbsim_seed_base": args.turbsim_seed_base,
        "turbsim_seed2": args.turbsim_seed2,
        "turbsim_wind_files": turbsim_records,
        "threads": args.threads,
        "case_workers": args.case_workers,
        "allow_partial": args.allow_partial,
        "continue_on_failure": args.continue_on_failure,
        "reuse_setup": args.reuse_setup,
        "preserve_completed_scenarios": args.preserve_completed_scenarios,
        "coupling": args.coupling,
        "allow_coarse_timestep": args.allow_coarse_timestep,
        "postprocess_locations": args.postprocess_locations,
        "postprocess_blade": args.postprocess_blade,
        "wohler_exponent": args.wohler_exponent,
        "equivalent_cycles": args.equivalent_cycles,
        "min_fatigue_duration_s": args.min_fatigue_duration,
        "include_smoke_in_del": args.include_smoke_in_del,
        "ultimate_allowables": None if args.ultimate_allowables is None else str(args.ultimate_allowables),
        "airfoil_source_manifest": airfoil_source_manifest,
        "airfoil_source_manifest_error": airfoil_source_manifest_error,
        "dlc_manifest_row_archive": dlc_manifest_row_archive,
        "dlc_manifest_row": dlc_manifest_row,
        "dlc_manifest_row_error": dlc_manifest_row_error,
        "planned_cases": cases,
    }
    if not args.only_postprocess:
        atomic_write_json(settings_path, settings)

    if args.dry_run:
        dry_run = {
            "run_root": str(args.run_root),
            "timesteps_per_case": timesteps,
            "duration_s": args.duration,
            "timestep_s": args.timestep,
            "use_opencl": args.use_opencl,
            "opencl_device": args.opencl_device,
            "opencl_group_size": args.opencl_group_size,
            "rpm": args.rpm,
            "cases_per_scenario": cases,
            "case_index": args.case_index,
            "scenarios": scenarios,
            "scenario_setup_commands": [
                batch_command(
                    args,
                    str(scenario["name"]),
                    float(scenario["force_scale"]),
                    timesteps,
                    coupling=str(scenario["coupling"]),
                    command_name="setup",
                    force_setup=not args.reuse_setup,
                )
                for scenario in scenarios
            ],
            "scenario_case_commands": None
            if args.case_index is None
            else [
                batch_command(
                    args,
                    str(scenario["name"]),
                    float(scenario["force_scale"]),
                    timesteps,
                    coupling=str(scenario["coupling"]),
                    command_name="run",
                    case_index=args.case_index,
                    force_setup=False,
                )
                for scenario in scenarios
            ],
            "turbsim_wind_files": turbsim_records,
            "postprocess_command": maybe_postprocess_command(args),
        }
        print(json.dumps(dry_run, indent=2, sort_keys=True))
        return

    if args.wind_only:
        return

    if not args.only_postprocess:
        if args.setup_only:
            for scenario in scenarios:
                name = str(scenario["name"])
                preserve_completed_scenario_or_run(
                    args,
                    name,
                    batch_command(
                        args,
                        name,
                        float(scenario["force_scale"]),
                        timesteps,
                        coupling=str(scenario["coupling"]),
                        command_name="setup",
                        force_setup=not args.reuse_setup,
                    )
                )
            return
        if args.case_index is not None:
            for scenario in scenarios:
                name = str(scenario["name"])
                preserve_completed_scenario_or_run(
                    args,
                    name,
                    batch_command(
                        args,
                        name,
                        float(scenario["force_scale"]),
                        timesteps,
                        coupling=str(scenario["coupling"]),
                        command_name="run",
                        case_index=args.case_index,
                        force_setup=False,
                    )
                )
            return
        if args.case_workers == 1:
            for scenario in scenarios:
                name = str(scenario["name"])
                preserve_completed_scenario_or_run(
                    args,
                    name,
                    batch_command(
                        args,
                        name,
                        float(scenario["force_scale"]),
                        timesteps,
                        coupling=str(scenario["coupling"]),
                    )
                )
        else:
            for scenario in scenarios:
                name = str(scenario["name"])
                preserve_completed_scenario_or_run(
                    args,
                    name,
                    batch_command(
                        args,
                        name,
                        float(scenario["force_scale"]),
                        timesteps,
                        coupling=str(scenario["coupling"]),
                        command_name="setup",
                        force_setup=not args.reuse_setup,
                    )
                )
            commands: list[list[str]] = []
            for scenario in scenarios:
                name = str(scenario["name"])
                if completed_scenario_can_be_preserved(args, name):
                    continue
                for index in range(case_count(args.run_root / name)):
                    commands.append(
                        batch_command(
                            args,
                            name,
                            float(scenario["force_scale"]),
                            timesteps,
                            coupling=str(scenario["coupling"]),
                            command_name="run",
                            case_index=index,
                            force_setup=False,
                        )
                    )
            run_parallel(commands, args.case_workers, continue_on_failure=args.continue_on_failure)

    if not args.skip_postprocess:
        run_checked(postprocess_command(args))


if __name__ == "__main__":
    main()
