#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE="${VORLAP_QBLADE_IMAGE:-vorlap-qblade-headless:amd64}"
PLATFORM="${VORLAP_QBLADE_PLATFORM:-linux/amd64}"

exec /opt/homebrew/bin/docker run --rm \
  --platform "${PLATFORM}" \
  --cpus 4 \
  --volume "${REPO_ROOT}:/work/VorLap" \
  --workdir /work/VorLap \
  --env QT_QPA_PLATFORM=offscreen \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=4 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    export VORLAP_ROOT=/work/VorLap
    export PYTHON_EXE=/usr/bin/python3
    export QBLADE_ROOT=/work/VorLap/run_headless/QBladeCE_2.0.9.7
    export PREP_ROOT=/tmp/hvawt-complete-hybrid-prep-v8
    export PREFLIGHT_CAMPAIGN_ROOT=/tmp/hvawt-complete-hybrid-preflight-v8
    export FULL_CAMPAIGN_ROOT=/tmp/hvawt-complete-hybrid-full-v8
    export PREPARE_FULL_CAMPAIGN=0
    export PYTHONPATH="${VORLAP_ROOT}:${QBLADE_ROOT}/SIL_Interface"
    export LD_LIBRARY_PATH="${QBLADE_ROOT}:${QBLADE_ROOT}/Libraries:${QBLADE_ROOT}/ControllerFiles:${QBLADE_ROOT}/Binaries:${QBLADE_ROOT}/Binaries/IPCBridge:${LD_LIBRARY_PATH:-}"
    export VORLAP_QBLADE_LOADER_ENV=1

    bash run_headless/prepare_hvawt_response_pilot_native.sh

    python3 - "${PREFLIGHT_CAMPAIGN_ROOT}" "${QBLADE_ROOT}" <<'"'"'PY'"'"'
import csv
import json
from pathlib import Path
import subprocess
import sys

from vorlap.hvawt_response_pilot import (
    SCENARIOS,
    command_for_case,
    validate_campaign,
)

campaign = Path(sys.argv[1]).resolve()
qblade_root = Path(sys.argv[2]).resolve()
state = validate_campaign(campaign)
manifest = state["manifest"]
assert manifest["case_count"] == 2
assert manifest["matrix"]["duration_s"] == 120.0
assert manifest["matrix"]["assessment_tail_duration_s"] == 60.0
assert [row["scenario"] for row in manifest["cases"]] == list(SCENARIOS)

smoke_root = Path("/tmp/hvawt-response-four-step-results")
reports = []
for case_index, case in enumerate(manifest["cases"]):
    output = smoke_root / str(case["case_id"])
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = output / "case_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "vorlap-hvawt-response-pilot-case-metadata-v1",
                "campaign_fingerprint": manifest["fingerprint"],
                "case": case,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    command = command_for_case(
        campaign,
        case_index=case_index,
        qblade_root=qblade_root,
        python_executable="/usr/bin/python3",
        threads=4,
        checkpoint_every=0,
        progress_every=0,
    )
    for option, value in (
        ("--output-dir", str(output)),
        ("--case-metadata", str(metadata_path)),
    ):
        command[command.index(option) + 1] = value
    command.extend(("--steps", "4"))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    sim_stem = Path(str(case["sim"])).stem
    summary_path = output / f"{sim_stem}_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["failed"] is False and summary["stopped"] is False
    assert summary["exported_results"] is True
    assert summary["requested_steps"] == summary["completed_steps"] == 4
    assert summary["sil_vorlap_updates"] == 4
    assert summary["sil_vorlap_history_mode"] == "compact"
    assert summary["restart"]["checkpoint_every"] == 0
    assert summary["restart"]["resume_start_step"] == 0
    assert summary["reproducibility"]["qblade"]["library"]["sha256"] == (
        "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
    )
    assert summary["reproducibility"]["thread_settings"] == {
        "threads": 4,
        "omp_num_threads_env": "4",
        "use_opencl": False,
    }
    history_path = Path(summary["sil_vorlap_history"]["path"])
    if not history_path.is_absolute():
        history_path = output / history_path
    rows = list(csv.DictReader(history_path.open(encoding="utf-8")))
    assert len(rows) == 4
    assert all(
        row["response_model_type"] == "hvawt_airfoil_engineering_lockin_hybrid"
        for row in rows
    )
    assert max(
        float(row["response_spectral_source_energy_sum"]) for row in rows
    ) > 0.0
    if case["scenario"] == SCENARIOS[0]:
        assert float(summary["sil_vorlap_max_abs_force"]) <= 1.0e-12
        assert all(
            float(row["response_spectral_residual_source_scale"]) == 0.0
            and float(
                row[
                    "response_spectral_applied_residual_maximum_nodal_force_n"
                ]
            )
            == 0.0
            and float(row["response_total_maximum_nodal_force_n"]) == 0.0
            for row in rows
        )
    else:
        assert all(
            float(row["response_spectral_residual_source_scale"]) == 1.0
            for row in rows
        )
        assert max(
            float(
                row[
                    "response_spectral_applied_residual_maximum_nodal_force_n"
                ]
            )
            for row in rows
        ) > 0.0
        assert 0.0 < max(
            float(row["response_total_maximum_nodal_force_n"]) for row in rows
        ) < 100.0
        assert max(
            abs(float(row["generated_max_abs_force_component_n"]))
            for row in rows
        ) > 0.0
        assert int(float(rows[-1]["applied_source_step_index"])) == 2
        assert int(float(rows[-1]["applied_force_lag_steps"])) == 1
    reports.append(
        {
            "case_id": case["case_id"],
            "scenario": case["scenario"],
            "completed_steps": summary["completed_steps"],
            "maximum_abs_force_n": summary["sil_vorlap_max_abs_force"],
            "history_rows": len(rows),
        }
    )

print(
    json.dumps(
        {
            "status": "passed_four_step_submission_input_smoke_only",
            "campaign_fingerprint": manifest["fingerprint"],
            "qblade_library_sha256": (
                "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
            ),
            "cases": reports,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
  '
