#!/usr/bin/env python3
"""Matched fixed-azimuth U11.25 QBlade / Spectral / Hybrid Flight experiment.

Clone reviewed physical inputs, never the previous solutions. The old response
basis remains honestly labelled 1 RPM; its use at 0 RPM is explicitly gated by
the archived 0/1 RPM eigensolutions and recorded in the campaign manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vorlap import hvawt_complete_dlc_campaign as previous
from vorlap.qblade_files import read_qblade_keyword, replace_qblade_keyword_file
from vorlap.qblade_run_support import atomic_write_json, canonical_fingerprint
from vorlap.qblade_sil import read_qblade_result_numeric_columns, normalize_qblade_header

SEED = 4433456
ROW = "dlc6p4_U11p25ms_s4433456"
DT = 0.0075
STEPS = 88000
AZIMUTH_TOL_DEG = 0.01
RPM_TOL = 0.001
WIND_SHA = "dd1608a5bf2dc5415858fe89e97bcecc332eb515b3497dd7b66fb04d735d74b2"
SCHEMA = "vorlap-fixed-azimuth-trio-20260922-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def basis_transfer() -> dict:
    """Quantify the existing 0/1 RPM comparison rather than relabel the basis."""
    path = ROOT / "run_headless/config/hvawt_response_mode_identification.json"
    identification = load(path)
    evidence = []
    for mode in identification["modes"]:
        comparison = mode["comparison_to_zero_rpm"]
        assert comparison["comparison_rpm"] == 0.0
        if (abs(comparison["frequency_shift_fraction"]) > 1e-3
                or comparison["action_translation_mac"] < 0.999):
            raise ValueError("0/1 RPM modal transfer exceeds the stated engineering bounds")
        evidence.append({"tracked_mode": mode["tracked_mode"], **comparison})
    return {
        "basis_rpm": 1.0,
        "simulation_rpm": 0.0,
        "identification_sha256": sha(path),
        "mode_comparisons": evidence,
        "frequency_shift_fraction_limit": 0.001,
        "minimum_action_translation_mac": 0.999,
        "policy": "Retain the reviewed 1 RPM shapes, duals, frequencies, masses and damping. "
                  "Use them at prescribed 0 RPM only; do not relabel as a zero-RPM basis. "
                  "The archived zero-RPM eigenmodes bound this engineering approximation. "
                  "The full QBlade structural solver itself runs at actual zero RPM.",
        "maximum_abs_basis_rpm_error_rpm": 1.0 + RPM_TOL,
        "actual_rpm_tolerance": RPM_TOL,
    }


def restore_source(case_dir: Path, original: dict, source: Path) -> None:
    """Rebuild clean archived inputs; never copy a previous run's mutable state."""
    archive = ROOT / "run_headless/config/hvawt_pilot_hpc_inputs/U05p00ms_source_case.tar.gz"
    if sha(archive) != "6f6daf91d026087bb5f5acec6b313c7cead1a70faa1deded1819e15c60d7d403":
        raise ValueError("Reviewed source archive changed")
    with tempfile.TemporaryDirectory() as temp:
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts or not (member.isfile() or member.isdir()):
                    raise ValueError("Unexpected member in reviewed source archive")
            stream.extractall(temp)
        shutil.copytree(Path(temp) / "U05p00ms", case_dir)
    from vorlap.hvawt_modal_decay_campaign import apply_substructure_variant
    from vorlap.hvawt_response_pilot import _apply_engineering_rayleigh_inputs
    selection = load(source / "inputs/hvawt_documented_rayleigh_selection.json")
    rayleigh = {**selection, **selection["source_case"]}
    apply_substructure_variant(case_dir / rayleigh["selected_substructure_relative_path"], "C20")
    _apply_engineering_rayleigh_inputs(case_dir, rayleigh)
    sim = case_dir / Path(original["sim"]).name
    for key, value in original["common_physics_contract"]["simulation_keywords"].items():
        replace_qblade_keyword_file(sim, key, value)
    previous._apply_wake_profile(sim)


def prepare(args) -> dict:
    root = args.campaign_root.resolve()
    source = args.source_campaign.resolve()
    if root.exists():
        raise FileExistsError(f"Use a fresh output root: {root}")
    if not math.isfinite(args.azimuth_deg) or not 0 <= args.azimuth_deg < 360:
        raise ValueError("Azimuth must lie in [0, 360) degrees")
    parent = load(source / "campaign_manifest.json")
    selected = [c for c in parent["cases"] if c["row_id"] == ROW]
    if tuple(c["scenario"] for c in selected) != previous.SCENARIOS:
        raise ValueError("Source campaign must contain the reviewed matched U11.25 trio")
    if len({c["common_physics_fingerprint"] for c in selected}) != 1:
        raise ValueError("Source trio does not have matched common physics")
    wind = Path(selected[0]["wind_file"])
    expected = selected[0]["common_physics_contract"]["wind_file_sha256"]
    if expected != WIND_SHA:
        raise ValueError("Source wind is not the displayed U11.25/seed4433456 realization")
    if not args.smoke_uniform and sha(wind) != WIND_SHA:
        raise ValueError("Flight wind bytes differ from the original simulation")
    transfer = basis_transfer()
    root.mkdir(parents=True)
    (root / "inputs").mkdir()
    result_filter = root / "inputs/fixed_azimuth_results.filter"
    base_filter = ROOT / previous.RESULT_FILTER_SOURCE_RELATIVE
    result_filter.write_text(base_filter.read_text().rstrip() + "\nLSS Azimuthal Pos. [deg]\n")
    response_model = load(ROOT / previous.RESPONSE_MODEL_RELATIVE)
    response_model["kinematics"]["maximum_abs_basis_rpm_error_rpm"] = 1.0 + RPM_TOL
    atomic_write_json(root / "inputs/modal_basis_transfer.json", transfer)
    model_source = ROOT / previous.RESPONSE_MODEL_RELATIVE
    cases = []
    for index, original in enumerate(selected):
        case_dir = root / "cases" / original["scenario"]
        restore_source(case_dir, original, source)
        # Hash all physical files using the same normalization as the old campaign.
        sim_rel = Path(original["sim"]).relative_to(original["case_dir"])
        sim = case_dir / sim_rel
        observed = previous._physics_assets(case_dir, sim)
        if observed != original["common_physics_contract"]["physical_assets"]:
            raise ValueError(f"Source physical assets changed: {original['scenario']}")
        patches = {
            "PRESCRIBETYPE": 1, "RPMPRESCRIBED": 0.0,
            "INITIAL_AZIMUTH": args.azimuth_deg, "RELAXSTEPS": 0,
            "TIMESTEP": DT, "NUMTIMESTEPS": STEPS,
            # Record startup as well, so fixed azimuth can be checked for the
            # entire calculation; fatigue/steady comparisons start at 60 s.
            "STOREFROM": 0.0, "STOREREPLAY": "false",
        }
        if args.smoke_uniform:
            patches.update({"WNDTYPE": 0, "WNDNAME": "", "MEANINF": 11.25})
        for name, value in patches.items():
            replace_qblade_keyword_file(sim, name, value)
        case = {
            "index": index, "scenario": original["scenario"],
            "sim": sim.relative_to(root).as_posix(),
            "case_dir": case_dir.relative_to(root).as_posix(),
            "source_case_id": original["case_id"],
            "force_contract": original["force_contract"],
            "result_dir": f"results/{original['scenario']}",
        }
        previous._prepare_scenario(
            runtime_repository=ROOT, case=case, case_dir=case_dir, sim_path=sim,
            response_model=response_model, model_source=model_source,
            result_filter=result_filter,
        )
        case["storage_filter_channel_count"] = len(
            (case_dir / case["storage_filter"]).read_text().splitlines())
        # The case is path-bound and cannot be moved after preparation.
        case["input_hashes"] = {
            p.relative_to(root).as_posix(): sha(p)
            for p in sorted(case_dir.rglob("*")) if p.is_file()
        }
        cases.append(case)
    manifest = {
        "schema": SCHEMA, "source_campaign": str(source),
        "source_manifest_sha256": sha(source / "campaign_manifest.json"),
        "source_campaign_fingerprint": parent["fingerprint"],
        "campaign_root": str(root), "smoke_uniform_only": args.smoke_uniform,
        "wind_speed_mps": 11.25, "seed": SEED, "wind_file": str(wind),
        "wind_sha256": WIND_SHA, "initial_azimuth_deg": args.azimuth_deg,
        "rotor_speed_rpm": 0.0, "timestep_s": DT,
        "duration_s": 660.0, "total_steps": STEPS, "analysis_start_s": 60.0,
        "retained_duration_s": 600.0, "store_from_s": 0.0,
        "modal_basis_transfer": transfer, "cases": cases,
        "result_filter": result_filter.relative_to(root).as_posix(),
        "result_filter_sha256": sha(result_filter),
        "generator_sha256": sha(Path(__file__)),
        "physical_basis": "same structural/aero inputs, damping, wake model, startup and "
                          "turbulent wind realization as source; rotor fixed from t=0",
        "claim_boundary": "Fixed-angle follow-up at a high-excitation angle selected from "
                          "the idling record. It does not prove a global fixed-angle maximum.",
    }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    atomic_write_json(root / "campaign_manifest.json", manifest)
    validate(root, check_wind=not args.smoke_uniform)
    return manifest


def validate(root: Path, *, check_wind: bool = True) -> dict:
    root = root.resolve()
    manifest = load(root / "campaign_manifest.json")
    fingerprint = manifest.pop("fingerprint")
    if fingerprint != canonical_fingerprint(manifest) or manifest["schema"] != SCHEMA:
        raise ValueError("Campaign manifest identity changed")
    manifest["fingerprint"] = fingerprint
    if str(root) != manifest["campaign_root"]:
        raise ValueError("Case contains absolute source paths; prepare it on its execution filesystem")
    if sha(Path(__file__)) != manifest["generator_sha256"]:
        raise ValueError("Submission driver changed after preparation; use a fresh campaign")
    if sha(root / manifest["result_filter"]) != manifest["result_filter_sha256"]:
        raise ValueError("Result filter changed")
    if check_wind and not manifest["smoke_uniform_only"]:
        if sha(Path(manifest["wind_file"])) != manifest["wind_sha256"]:
            raise ValueError("Wind file changed")
    for case in manifest["cases"]:
        for relative, expected in case["input_hashes"].items():
            if sha(root / relative) != expected:
                raise ValueError(f"Prepared input changed: {relative}")
        sim = root / case["sim"]
        for name, expected in {
            "RPMPRESCRIBED": 0, "PRESCRIBETYPE": 1,
            "INITIAL_AZIMUTH": manifest["initial_azimuth_deg"],
            "TIMESTEP": DT, "NUMTIMESTEPS": STEPS, "STOREFROM": 0,
        }.items():
            if float(read_qblade_keyword(sim, name)) != expected:
                raise ValueError(f"Fixed-rotor constraint changed: {name}")
    return manifest


def command(root: Path, manifest: dict, index: int, args, output: Path) -> list[str]:
    case = manifest["cases"][index]
    result = [sys.executable, str(ROOT / "run_headless/run_qblade_sim.py"),
              "--sim", str(root / case["sim"]), "--qblade-root", str(args.qblade_root.resolve()),
              "--output-dir", str(output), "--result-name", "loads",
              "--result-filter", str(root / manifest["result_filter"]),
              "--threads", str(args.threads), "--coupling-mode", case["force_contract"]["coupling_mode"],
              "--checkpoint-every", "0" if args.smoke_steps else "24000",
              "--progress-every", "1000", "--case-metadata", str(output / "case_metadata.json")]
    if index:
        result += ["--sil-vorlap", "--sil-vorlap-history-mode", "compact"]
    if args.smoke_steps:
        result += ["--steps", str(args.smoke_steps)]
    return result


def audit_columns(columns: dict, *, azimuth: float, min_last_time: float,
                  expected_samples: int | None = None) -> dict:
    values = {normalize_qblade_header(k): v for k, v in columns.items()}
    t, rpm, angle = [values[key] for key in (
        "Time [s]", "LSS Rpm [rpm]", "LSS Azimuthal Pos. [deg]")]
    if not t or len(t) != len(rpm) or len(t) != len(angle):
        raise ValueError("Missing full rotor-state histories")
    if expected_samples is not None and len(t) != expected_samples:
        raise ValueError("Export does not contain every requested timestep")
    if not all(math.isfinite(float(x)) for v in (t, rpm, angle) for x in v):
        raise ValueError("Nonfinite rotor-state history")
    # QBlade exports may round time to milliseconds: compare with the ideal
    # grid, not adjacent differences of 0.007 versus 0.008 s.
    export_rounding = 0.00051
    if abs(t[0]) > export_rounding or max(t) < min_last_time - export_rounding:
        raise ValueError("Rotor-state histories do not cover the requested simulation")
    if (any(b <= a for a, b in zip(t, t[1:]))
            or any(abs(value - i * DT) > export_rounding for i, value in enumerate(t))):
        raise ValueError("Missing or duplicate rotor-state timesteps")
    rpm_error = max(abs(x) for x in rpm)
    angle_error = max(abs(math.remainder(x - azimuth, 360.0)) for x in angle)
    if rpm_error > RPM_TOL or angle_error > AZIMUTH_TOL_DEG:
        raise ValueError(f"Rotor was not fixed: max RPM={rpm_error}, azimuth error={angle_error} deg")
    return {"samples": len(t), "first_time_s": min(t), "last_time_s": max(t),
            "maximum_abs_rpm": rpm_error, "maximum_azimuth_error_deg": angle_error}


def audit_required_channels(columns: dict, requested: list[str], steps: int) -> dict:
    """A finite partial export must not masquerade as complete load evidence."""
    normalized = {normalize_qblade_header(k): v for k, v in columns.items()}
    names = [normalize_qblade_header(k) for k in requested if k.strip()]
    missing = sorted(set(names) - set(normalized))
    if missing:
        raise ValueError(f"Missing requested result channels: {missing}")
    for name in names:
        data = normalized[name]
        if len(data) != steps:
            raise ValueError(f"Incomplete result channel {name}: {len(data)} of {steps} samples")
        if not all(math.isfinite(float(value)) for value in data):
            raise ValueError(f"Nonfinite exported result channel: {name}")
    return {"required_channel_count": len(names), "samples_per_channel": steps,
            "all_required_channels_finite_and_complete": True}


def audit(root: Path, manifest: dict, index: int, output: Path, steps: int) -> dict:
    summaries = list(output.glob("*_run_summary.json"))
    if len(summaries) != 1:
        raise ValueError("Expected one QBlade run summary")
    summary = load(summaries[0])
    if summary["failed"] or summary["stopped"] or summary["completed_steps"] != steps:
        raise ValueError("QBlade did not finish all requested steps")
    result = output / "loads.txt"
    if not result.is_file():
        candidates = list(output.glob("loads*.txt"))
        if len(candidates) != 1:
            raise ValueError("Could not uniquely locate exported QBlade results")
        result = candidates[0]
    columns = read_qblade_result_numeric_columns(result)
    channel_check = audit_required_channels(
        columns, (root / manifest["result_filter"]).read_text().splitlines(), steps)
    state = audit_columns(columns, azimuth=manifest["initial_azimuth_deg"],
                          min_last_time=(steps - 1) * DT, expected_samples=steps)
    if index and summary["sil_vorlap_updates"] != steps:
        raise ValueError("SIL did not update every structural timestep")
    receipt = {"passed": True, "scenario": manifest["cases"][index]["scenario"],
               "completed_steps": steps, "rotor_state": state,
               "load_channels": channel_check,
               "campaign_fingerprint": manifest["fingerprint"],
               "summary_sha256": sha(summaries[0]), "results_sha256": sha(result),
               "scope": "short runtime verification only" if steps != STEPS else "660 s production run"}
    atomic_write_json(output / "fixed_azimuth_audit.json", receipt)
    return receipt


def run(args) -> dict:
    root = args.campaign_root.resolve()
    manifest = validate(root)
    if manifest["smoke_uniform_only"] and not args.smoke_steps:
        raise ValueError("A uniform smoke campaign cannot be used for production")
    if not 0 <= args.case_index < 3:
        raise ValueError("Case index must be 0, 1 or 2")
    output = root / ("smoke" if args.smoke_steps else "results") / manifest["cases"][args.case_index]["scenario"]
    if output.exists():
        raise FileExistsError(f"Do not overwrite an existing run: {output}")
    output.mkdir(parents=True)
    atomic_write_json(output / "case_metadata.json", {"campaign": manifest,
                      "case": manifest["cases"][args.case_index]})
    cmd = command(root, manifest, args.case_index, args, output)
    atomic_write_json(output / "command.json", cmd)
    subprocess.run(cmd, check=True)
    return audit(root, manifest, args.case_index, output, args.smoke_steps or STEPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-campaign", type=Path, required=True)
    prep.add_argument("--campaign-root", type=Path, required=True)
    prep.add_argument("--azimuth-deg", type=float, required=True)
    prep.add_argument("--smoke-uniform", action="store_true", help="Local runtime smoke only, never production")
    check = sub.add_parser("validate")
    check.add_argument("--campaign-root", type=Path, required=True)
    exe = sub.add_parser("run")
    exe.add_argument("--campaign-root", type=Path, required=True)
    exe.add_argument("--qblade-root", type=Path, required=True)
    exe.add_argument("--case-index", type=int, required=True)
    exe.add_argument("--threads", type=int, default=112)
    exe.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "validate":
        result = validate(args.campaign_root)
    else:
        result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
