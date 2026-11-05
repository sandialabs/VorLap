"""Per-job execution environment provenance for sampled DLC rows."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

from .dlc_manifest import DLCManifestError, sha256_file, sha256_payload


_SCHEDULER_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_NTASKS",
)
_RUNTIME_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "LD_LIBRARY_PATH",
    "VORLAP_SOURCE_BOUNDS_POLICY",
    "VORLAP_SOURCE_FREQUENCY_TRUNCATION",
    "QBLADE_WAKE_PROFILE",
    "QBLADE_RESULT_FILTER",
    "CASE_WORKERS",
)


def _file_record(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _first_file(root: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def _resolve_executable(value: str | None, fallback: Path | None = None) -> Path | None:
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        found = shutil.which(value)
        if found:
            return Path(found).resolve()
    if fallback is not None and fallback.is_file():
        return fallback.resolve()
    return None


def _git_record(root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"root": str(root.resolve()), "commit": None, "tracked_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        ).stdout.strip()
        record.update({"commit": commit, "tracked_dirty": bool(dirty)})
    except (OSError, subprocess.SubprocessError):
        pass
    return record


def build_execution_provenance(
    *,
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index: int,
    phase: str,
    environment: Mapping[str, str] | None = None,
    vorlap_root: Path,
) -> dict[str, Any]:
    if phase not in {"wind", "qblade"}:
        raise DLCManifestError(f"Unsupported execution provenance phase: {phase}")
    env = dict(os.environ if environment is None else environment)
    qblade_root = Path(
        env.get("QBLADE_ROOT", str(vorlap_root / "run_headless" / "QBladeCE_2.0.9.7"))
    ).expanduser()
    qblade_library = _first_file(qblade_root, ("*QBlade*.so*", "*QBlade*.dll", "*QBlade*.dylib"))
    bridge_library = _first_file(
        qblade_root / "ControllerFiles",
        ("*vorlap*qblade*.so*", "*vorlap*qblade*.dylib", "*vorlap*qblade*.dll"),
    )
    turbsim = _resolve_executable(
        env.get("TURBSIM_EXE"),
        vorlap_root / "run_headless" / "openfast" / "install" / "bin" / "turbsim",
    )
    openfast_root = Path(
        env.get("OPENFAST_SOURCE_ROOT", str(vorlap_root / "run_headless" / "openfast"))
    ).expanduser()
    airfoil_source_dir = (
        vorlap_root
        / "examples"
        / "ReferenceHVAWTQBladeModel"
        / "10m_HVAWT_reference_turbine"
        / "VorLapAirfoils"
    )
    airfoil_source_files = [
        record
        for path in sorted(airfoil_source_dir.glob("*.h5"))
        if (record := _file_record(path)) is not None
    ]
    loaded_modules = [item for item in env.get("LOADEDMODULES", "").split(":") if item]
    python_executable = _resolve_executable(env.get("PYTHON_EXE")) or Path(sys.executable).resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": phase,
        "created_unix": time.time(),
        "manifest_sha256": manifest["manifest_sha256"],
        "row_index": int(row_index),
        "row_id": row["row_id"],
        "dlc": row["dlc"],
        "wind_speed_mps": row["wind_speed_mps"],
        "seed": row["seed"],
        "repository": _git_record(vorlap_root),
        "python": {
            "executable": str(python_executable),
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "modules": {
            "loaded": loaded_modules,
            "loadedmodules_raw": env.get("LOADEDMODULES", ""),
            "modulepath": env.get("MODULEPATH", ""),
            "lmod_system_name": env.get("LMOD_SYSTEM_NAME", ""),
        },
        "scheduler": {key: env.get(key) for key in _SCHEDULER_ENV_KEYS},
        "runtime_environment": {key: env.get(key) for key in _RUNTIME_ENV_KEYS},
        "qblade": {
            "root": str(qblade_root.resolve()) if qblade_root.exists() else str(qblade_root),
            "library": _file_record(qblade_library),
            "result_filter": _file_record(
                Path(env["QBLADE_RESULT_FILTER"]).expanduser()
                if env.get("QBLADE_RESULT_FILTER")
                else None
            ),
        },
        "bridge_library": _file_record(bridge_library),
        "source_data": {
            "manifest_airfoil_source_sha256": (
                manifest.get("basis", {}).get("airfoil_source_sha256")
                if isinstance(manifest.get("basis"), Mapping)
                else None
            ),
            "runtime_airfoil_files": airfoil_source_files,
        },
        "turbsim_executable": _file_record(turbsim),
        "turbsim_source": {
            "repository": _git_record(openfast_root),
            "explicit_ref": env.get("OPENFAST_SOURCE_REF"),
        },
    }
    payload["provenance_sha256"] = sha256_payload(payload)
    return payload


def validate_execution_provenance(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index: int,
    phase: str,
    require_runtime_artifacts: bool = True,
) -> None:
    archived = dict(payload)
    digest = str(archived.pop("provenance_sha256", "") or "")
    if digest != sha256_payload(archived):
        raise DLCManifestError(f"{phase} execution provenance digest is invalid for {row['row_id']}.")
    if (
        payload.get("phase") != phase
        or payload.get("manifest_sha256") != manifest["manifest_sha256"]
        or payload.get("row_index") != row_index
        or payload.get("row_id") != row["row_id"]
        or payload.get("seed") != row["seed"]
    ):
        raise DLCManifestError(f"{phase} execution provenance does not match row {row['row_id']}.")
    repository = payload.get("repository")
    if not isinstance(repository, Mapping) or not str(repository.get("commit", "") or ""):
        raise DLCManifestError(f"{phase} execution provenance is missing the repository commit.")
    python = payload.get("python")
    if not isinstance(python, Mapping) or not str(python.get("version", "") or ""):
        raise DLCManifestError(f"{phase} execution provenance is missing Python details.")
    if not require_runtime_artifacts:
        return
    if phase == "wind":
        if not isinstance(payload.get("turbsim_executable"), Mapping):
            raise DLCManifestError(f"Wind execution provenance is missing the TurbSim executable hash.")
        turbsim_source = payload.get("turbsim_source")
        source_repository = (
            turbsim_source.get("repository")
            if isinstance(turbsim_source, Mapping)
            else None
        )
        source_commit = (
            source_repository.get("commit")
            if isinstance(source_repository, Mapping)
            else None
        )
        explicit_ref = (
            turbsim_source.get("explicit_ref")
            if isinstance(turbsim_source, Mapping)
            else None
        )
        if not str(source_commit or explicit_ref or ""):
            raise DLCManifestError(
                f"Wind execution provenance is missing the TurbSim/OpenFAST source revision."
            )
    else:
        qblade = payload.get("qblade")
        if not isinstance(qblade, Mapping) or not isinstance(qblade.get("library"), Mapping):
            raise DLCManifestError(f"QBlade execution provenance is missing the QBlade library hash.")
        result_filter = qblade.get("result_filter") if isinstance(qblade, Mapping) else None
        if not isinstance(result_filter, Mapping):
            raise DLCManifestError(f"QBlade execution provenance is missing the result-filter hash.")
        execution = manifest.get("execution_settings")
        expected_filter_sha256 = (
            execution.get("result_filter_sha256")
            if isinstance(execution, Mapping)
            else None
        )
        if result_filter.get("sha256") != expected_filter_sha256:
            raise DLCManifestError(
                f"QBlade execution provenance result filter does not match row {row['row_id']}."
            )
        if not isinstance(payload.get("bridge_library"), Mapping):
            raise DLCManifestError(f"QBlade execution provenance is missing the bridge-library hash.")
