#!/usr/bin/env python3
"""Run a short synthetic-map QBlade smoke for the work-map adapter.

The map is a deterministic software-verification fixture, never calibration or
validation evidence.  It exercises the complete campaign generation, native
QBlade/SIL launch, work-map query, startup seed, energy-conjugate force path,
history export, and summary provenance without using Ehrmann response values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Sequence

from . import campaign
from vorlap.airfoil_work_map_identification import assemble_airfoil_work_map_payload
from vorlap.qblade_run_support import atomic_write_json


def synthetic_map_payload() -> dict[str, object]:
    records = []
    for incidence in (-90.0,):
        for reynolds in (1.0e4, 2.0e6):
            for reduced_frequency in (0.01, 10.0):
                for amplitude, entrainment, power in (
                    (0.0, 0.0, 0.0),
                    (0.02, 1.0, 0.012),
                    (0.5, 0.5, -0.006),
                ):
                    records.append(
                        {
                            "incidence_deg": incidence,
                            "reynolds_number": reynolds,
                            "reduced_frequency_k": reduced_frequency,
                            "amplitude_ratio": amplitude,
                            "direction_deg_mod_180": 0.0,
                            "entrainment_fraction": entrainment,
                            "reference_available_force_coefficient": 0.2,
                            "signed_power_coefficient": power,
                            "power_coefficient_standard_uncertainty": 0.0,
                        }
                    )
    return assemble_airfoil_work_map_payload(
        records=records,
        profile="NACA0018",
        sources=(
            {
                "role": "prescribed_motion_high_fidelity",
                "label": "synthetic software-verification force history",
                "sha256": "1" * 64,
            },
            {
                "role": "exact_runtime_base",
                "label": "synthetic software-verification runtime base",
                "sha256": "2" * 64,
            },
            {
                "role": "stationary_force_spectrum",
                "label": "tracked NACA0018 stationary spectrum",
                "sha256": campaign.TRACKED_STATIONARY_SPECTRUM_SHA256,
            },
        ),
    )


def run(
    root: Path,
    *,
    qblade_root: Path,
    runtime_vorlap_root: Path,
    python_executable: str,
    threads: int,
    steps: int,
) -> Path:
    if not 1 <= threads <= 8:
        raise ValueError("threads must be in [1, 8].")
    if steps < 2:
        raise ValueError("steps must be at least two.")
    root = Path(root).expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    map_path = root / "synthetic_work_map.json"
    atomic_write_json(map_path, synthetic_map_payload())
    campaign_root = root / "campaign"
    campaign.generate(
        campaign_root,
        runtime_vorlap_root=runtime_vorlap_root,
        work_map=map_path,
    )
    campaign.run_case(
        campaign=campaign_root,
        case_index=1,
        qblade_root=qblade_root,
        python_executable=python_executable,
        threads=threads,
        resume=False,
        qblade_steps=steps,
    )
    row = campaign.validate(campaign_root)["manifest"]["cases"][1]
    summary_path = (
        campaign_root
        / row["result_dir"]
        / f"{Path(row['sim']).stem}_run_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    response = summary.get("sil_vorlap_response_diagnostics_summary")
    if (
        summary.get("failed") is not False
        or summary.get("stopped") is not False
        or summary.get("completed_steps") != steps
        or not isinstance(response, dict)
        or response.get("model_type")
        != "single_mode_airfoil_force_motion_work_map_v1"
        or response.get("samples") != 1
        or response.get("residual_source_scale") != 0.0
        or response.get("work_map_fingerprint")
        != campaign.validate(campaign_root)["manifest"]["work_map_identity"][
            "fingerprint"
        ]
    ):
        raise RuntimeError("QBlade work-map smoke summary failed its exact contract.")
    report = {
        "schema": "vorlap-airfoil-work-map-qblade-smoke-v1",
        "status": "passed_software_verification_only",
        "threads": threads,
        "steps": steps,
        "campaign_fingerprint": campaign.validate(campaign_root)["manifest"][
            "fingerprint"
        ],
        "run_summary": str(summary_path),
        "response": response,
    }
    report_path = root / "work_map_smoke_report.json"
    atomic_write_json(report_path, report)
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qblade-root", type=Path, required=True)
    parser.add_argument("--runtime-vorlap-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args(argv)
    print(
        run(
            args.root,
            qblade_root=args.qblade_root,
            runtime_vorlap_root=args.runtime_vorlap_root,
            python_executable=args.python,
            threads=args.threads,
            steps=args.steps,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
