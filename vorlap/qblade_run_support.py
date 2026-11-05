"""Reusable support helpers for one-case QBlade runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

from .qblade_files import read_qblade_keyword, resolve_qblade_path
from .qblade_sil import SilVorLapCoupler, find_qblade_external_parameter_file

_FILE_SHA256_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}
MODAL_NATIVE_BRIDGE_RUNTIME_IDENTITY_SCHEMA = (
    "vorlap-qblade-modal-native-bridge-runtime-identity-v1"
)
MODAL_NATIVE_BRIDGE_LOGGING_CONTRACT_SCHEMA = (
    "vorlap-qblade-modal-native-bridge-logging-contract-v1"
)
MODAL_NATIVE_BRIDGE_RUNTIME_AUDIT_SCHEMA = (
    "vorlap-qblade-modal-native-bridge-runtime-audit-v1"
)
_MODAL_NATIVE_REQUIRED_SYMBOLS = ["update", "update_init", "update_message"]
_MODAL_NATIVE_BRIDGE_IDENTITY_KEYS = frozenset(
    {
        "schema",
        "resolved_library_filename",
        "resolved_library_relative_path",
        "sha256",
        "size_bytes",
        "required_symbols",
        "logging_contract",
    }
)
_MODAL_NATIVE_LOGGING_CONTRACT = {
    "schema": MODAL_NATIVE_BRIDGE_LOGGING_CONTRACT_SCHEMA,
    "environment_override": "VORLAP_QBLADE_BRIDGE_LOG_DIR",
    "environment_override_policy": "raw_nonempty_value_prohibited",
    "log_directory_source": "source_parameter_dir",
    "log_filename": "vorlap_qblade_bridge_cpp.log",
    "required_callback_activity": "exactly_zero_update_init_and_update_callbacks",
}


def file_sha256(path: Path) -> dict[str, object] | None:
    """Return a stable file identity block for reproducibility metadata."""
    if not path.is_file():
        return None
    for attempt in range(3):
        stat = path.stat()
        cache_key = (
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_dev,
            stat.st_ino,
        )
        digest_text = _FILE_SHA256_CACHE.get(cache_key)
        if digest_text is not None:
            break
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        final_stat = path.stat()
        changed = (
            final_stat.st_size != stat.st_size
            or final_stat.st_mtime_ns != stat.st_mtime_ns
            or final_stat.st_ctime_ns != stat.st_ctime_ns
            or final_stat.st_dev != stat.st_dev
            or final_stat.st_ino != stat.st_ino
        )
        if not changed:
            digest_text = digest.hexdigest()
            _FILE_SHA256_CACHE[cache_key] = digest_text
            break
        if attempt == 2:
            raise RuntimeError(f"File changed while hashing runtime input: {path}")
        # Docker Desktop can publish a large QBlade export one metadata update
        # after QBlade closes.  Rehash only after the file has settled; never
        # accept the changed snapshot.
        time.sleep(0.1)
    return {
        "path": str(path),
        "sha256": digest_text,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


_QBL_RUNTIME_ASSET_SUFFIXES = {
    ".afl",
    ".bld",
    ".filter",
    ".flt",
    ".h5",
    ".json",
    ".plr",
    ".sim",
    ".str",
    ".trb",
}


def _stable_file_identity(
    identity: dict[str, object] | None,
) -> dict[str, object] | None:
    if identity is None:
        return None
    return {
        "path": identity.get("path"),
        "sha256": identity.get("sha256"),
        "size_bytes": identity.get("size_bytes"),
    }


def canonical_fingerprint(payload: object) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible data."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qblade_runtime_asset_manifest(sim_path: Path) -> list[dict[str, object]]:
    """Hash the generated case assets and external wind file QBlade/VorLap can load."""
    sim = Path(sim_path).resolve()
    case_root = sim.parent
    rows: list[dict[str, object]] = []
    for path in sorted(case_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _QBL_RUNTIME_ASSET_SUFFIXES:
            continue
        identity = _stable_file_identity(file_sha256(path))
        if identity is None:
            continue
        rows.append(
            {
                "role": "case_asset",
                "relative_path": path.relative_to(case_root).as_posix(),
                **identity,
            }
        )

    raw_wind_type = read_qblade_keyword(sim, "WNDTYPE")
    raw_wind_name = read_qblade_keyword(sim, "WNDNAME")
    try:
        wind_type = int(float(raw_wind_type)) if raw_wind_type is not None else 0
    except ValueError:
        wind_type = -1
    if wind_type in (1, 2) and raw_wind_name:
        wind_path = resolve_qblade_path(sim, raw_wind_name)
        identity = _stable_file_identity(file_sha256(wind_path))
        rows.append(
            {
                "role": "wind_input",
                "relative_path": None,
                "path": str(wind_path),
                "sha256": None if identity is None else identity["sha256"],
                "size_bytes": None if identity is None else identity["size_bytes"],
            }
        )
    return rows


def qblade_runtime_asset_fingerprint(
    sim_path: Path,
) -> tuple[list[dict[str, object]], str]:
    assets = qblade_runtime_asset_manifest(sim_path)
    fingerprint_rows = [
        {
            "role": row.get("role"),
            "relative_path": row.get("relative_path"),
            "path": row.get("path") if row.get("role") == "wind_input" else None,
            "sha256": row.get("sha256"),
            "size_bytes": row.get("size_bytes"),
        }
        for row in assets
    ]
    return assets, canonical_fingerprint(fingerprint_rows)


def vorlap_python_runtime_manifest() -> tuple[list[dict[str, object]], str]:
    """Hash the Python implementation used by SIL coupling and post-run diagnostics."""
    package_root = Path(__file__).resolve().parent
    rows: list[dict[str, object]] = []
    for path in sorted(package_root.rglob("*.py")):
        identity = _stable_file_identity(file_sha256(path))
        if identity is None:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(package_root).as_posix(),
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
            }
        )
    return rows, canonical_fingerprint(rows)


def qblade_runtime_dependency_manifest(
    qblade_root: Path,
) -> tuple[list[dict[str, object]], str]:
    """Hash QBlade's loaded dependency directory and Python SIL wrapper."""

    root = Path(qblade_root).resolve()
    libraries = root / "Libraries"
    wrapper = root / "SIL_Interface" / "QBladeLibrary.py"
    if not libraries.is_dir() or not wrapper.is_file() or wrapper.is_symlink():
        raise FileNotFoundError(
            f"QBlade runtime dependency tree is incomplete below {root}"
        )
    dependency_symlinks = [
        path for path in sorted(libraries.rglob("*")) if path.is_symlink()
    ]
    if dependency_symlinks:
        raise ValueError(
            "QBlade runtime dependency tree contains symlink(s): "
            + ", ".join(
                path.relative_to(root).as_posix() for path in dependency_symlinks
            )
        )
    paths = [path for path in sorted(libraries.rglob("*")) if path.is_file()]
    paths.append(wrapper)
    rows: list[dict[str, object]] = []
    for path in paths:
        identity = _stable_file_identity(file_sha256(path))
        if identity is None:  # pragma: no cover - protected by is_file above
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": identity["sha256"],
                "bytes": identity["size_bytes"],
            }
        )
    return rows, canonical_fingerprint(rows)


def require_modal_qblade_runtime_dependency_contract(
    qblade_root: Path,
    expected_fingerprint: object,
) -> dict[str, object]:
    """Re-hash modal QBlade/Chrono dependencies at a publication boundary."""

    dependencies, fingerprint = qblade_runtime_dependency_manifest(qblade_root)
    if not isinstance(expected_fingerprint, str) or fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Modal-burst QBlade runtime dependency fingerprint changed after preflight: "
            f"expected={expected_fingerprint!r}, observed={fingerprint!r}."
        )
    return {
        "schema": "vorlap-qblade-modal-runtime-dependency-audit-v1",
        "passed": True,
        "fingerprint": fingerprint,
        "dependencies": dependencies,
    }


def build_checkpoint_identity(
    *,
    sim_path: Path,
    qblade_root: Path,
    coupling_mode: str,
    requested_steps: int,
    threads: int,
    use_opencl: bool,
    device: int,
    group_size: int,
    constant_force_actions: list[dict[str, object]] | None = None,
    qblade_execution_mode: str = "step-loop",
    resume_initialize: bool = False,
    allow_partial: bool = False,
    modal_burst_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the complete compatibility identity for a QBlade restart project."""
    modal_burst = modal_burst_identity is not None
    assets, asset_fingerprint = qblade_runtime_asset_fingerprint(sim_path)
    python_assets, python_asset_fingerprint = vorlap_python_runtime_manifest()
    qblade_library = find_qblade_library(qblade_root)
    parameter_file, config, _ = load_vorlap_config_for_sim(sim_path)
    library_stem = None if config is None else config.get("library_stem")
    bridge_library = find_bridge_library(
        qblade_root,
        None if library_stem is None else str(library_stem),
    )
    response_model = None if config is None else config.get("response_model")
    response_schema_v2 = (
        isinstance(response_model, dict) and response_model.get("schema_version") == 2
    )
    if response_schema_v2 or modal_burst:
        qblade_dependencies, qblade_dependency_fingerprint = (
            qblade_runtime_dependency_manifest(qblade_root)
        )
    else:
        qblade_dependencies = None
        qblade_dependency_fingerprint = None
    payload: dict[str, object] = {
        "version": 1,
        "runtime_asset_fingerprint": asset_fingerprint,
        "runtime_assets": assets,
        "vorlap_python_runtime_fingerprint": python_asset_fingerprint,
        "vorlap_python_runtime_assets": python_assets,
        "qblade_library": _stable_file_identity(file_sha256(qblade_library)),
        "bridge_library": (
            _stable_file_identity(file_sha256(bridge_library))
            if bridge_library is not None
            else None
        ),
        "external_parameter_file": (
            None if parameter_file is None else str(parameter_file)
        ),
        "coupling_mode": str(coupling_mode),
        "requested_steps": int(requested_steps),
        "threads": int(threads),
        "use_opencl": bool(use_opencl),
        "device": int(device),
        "group_size": int(group_size),
        "constant_force_actions": constant_force_actions or [],
        "qblade_execution_mode": str(qblade_execution_mode),
        "resume_initialize": bool(resume_initialize),
        "allow_partial": bool(allow_partial),
    }
    if modal_burst_identity is not None:
        payload["modal_burst_identity"] = modal_burst_identity
    if response_schema_v2 or modal_burst:
        payload["qblade_runtime_dependencies"] = qblade_dependencies
        payload["qblade_runtime_dependency_fingerprint"] = qblade_dependency_fingerprint
    payload["fingerprint"] = canonical_fingerprint(payload)
    return payload


def require_modal_runtime_identity_contract(
    *,
    sim_path: Path,
    qblade_root: Path,
    plan_path: Path,
    expected_checkpoint_identity: dict[str, object],
) -> dict[str, object]:
    """Reparse every modal artifact and re-hash the complete live runtime.

    This check deliberately reconstructs the coupler from disk instead of trusting
    its in-memory identity.  It therefore covers the plan and all three plan-bound
    artifacts, while the rebuilt checkpoint identity additionally covers the staged
    case, main QBlade library, bridge, QBlade/Chrono dependencies, and Python source.
    """

    from .qblade_modal_decay import SilModalBurstCoupler

    resolved_plan = Path(plan_path).expanduser().resolve()
    live_modal_identity = SilModalBurstCoupler.from_plan_file(
        resolved_plan
    ).checkpoint_identity()
    expected_modal_identity = expected_checkpoint_identity.get("modal_burst_identity")
    if live_modal_identity != expected_modal_identity:
        raise RuntimeError(
            "Modal-burst plan or plan-bound artifact identity changed after preflight."
        )

    try:
        current_checkpoint_identity = build_checkpoint_identity(
            sim_path=Path(sim_path),
            qblade_root=Path(qblade_root),
            coupling_mode=str(expected_checkpoint_identity["coupling_mode"]),
            requested_steps=int(expected_checkpoint_identity["requested_steps"]),
            threads=int(expected_checkpoint_identity["threads"]),
            use_opencl=bool(expected_checkpoint_identity["use_opencl"]),
            device=int(expected_checkpoint_identity["device"]),
            group_size=int(expected_checkpoint_identity["group_size"]),
            constant_force_actions=list(
                expected_checkpoint_identity["constant_force_actions"]
            ),
            qblade_execution_mode=str(
                expected_checkpoint_identity["qblade_execution_mode"]
            ),
            resume_initialize=bool(expected_checkpoint_identity["resume_initialize"]),
            allow_partial=bool(expected_checkpoint_identity["allow_partial"]),
            modal_burst_identity=live_modal_identity,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Modal-burst expected checkpoint identity is incomplete or malformed."
        ) from exc
    if current_checkpoint_identity != expected_checkpoint_identity:
        changed = sorted(
            key
            for key in set(current_checkpoint_identity)
            | set(expected_checkpoint_identity)
            if current_checkpoint_identity.get(key)
            != expected_checkpoint_identity.get(key)
        )
        raise RuntimeError(
            "Modal-burst complete checkpoint runtime identity changed after preflight: "
            + ", ".join(changed)
        )
    return {
        "schema": "vorlap-qblade-modal-runtime-identity-audit-v1",
        "passed": True,
        "plan_file": str(resolved_plan),
        "modal_burst_identity_fingerprint": live_modal_identity.get("fingerprint"),
        "checkpoint_identity_fingerprint": current_checkpoint_identity.get(
            "fingerprint"
        ),
    }


def _run_capture(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command, check=False, text=True, capture_output=True, timeout=10
        )
    except Exception as exc:  # pragma: no cover - depends on local git availability.
        return 127, str(exc)
    text = (completed.stdout + completed.stderr).strip()
    return completed.returncode, text


def git_metadata(repo: Path) -> dict[str, object]:
    """Best-effort git provenance for a repository root."""
    repo = repo.resolve()
    metadata: dict[str, object] = {"path": str(repo), "available": False}
    if not (repo / ".git").exists():
        metadata["error"] = "not a git checkout"
        return metadata
    code, commit = _run_capture(["git", "-C", str(repo), "rev-parse", "HEAD"])
    if code != 0:
        metadata["error"] = commit
        return metadata
    _, branch = _run_capture(["git", "-C", str(repo), "branch", "--show-current"])
    _, status = _run_capture(["git", "-C", str(repo), "status", "--porcelain"])
    status_lines = [line for line in status.splitlines() if line]
    metadata.update(
        {
            "available": True,
            "commit": commit,
            "branch": branch,
            "dirty": bool(status_lines),
            "status_count": len(status_lines),
            "status_porcelain": status_lines,
        }
    )
    return metadata


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def vorlap_package_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "version": package_versions(["vorlap"]).get("vorlap"),
        "import_path": None,
        "git": None,
        "error": None,
    }
    try:
        import vorlap  # type: ignore

        package_path = Path(vorlap.__file__).resolve()
        metadata["import_path"] = str(package_path)
        metadata["version"] = getattr(vorlap, "__version__", metadata["version"])
        metadata["git"] = git_metadata(package_path.parents[1])
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        metadata["error"] = str(exc)
        env_root = os.environ.get("VORLAP_ROOT")
        if env_root:
            metadata["git"] = git_metadata(Path(env_root))
    return metadata


def load_json_file(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_vorlap_config_for_sim(
    sim_path: Path,
) -> tuple[Path | None, dict | None, str | None]:
    try:
        path = find_qblade_external_parameter_file(sim_path)
        return path, json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, None, str(exc)


def native_bridge_log_path(sim_path: Path) -> Path | None:
    try:
        parameter_file = find_qblade_external_parameter_file(sim_path)
    except Exception:
        return None
    return parameter_file.parent / "vorlap_qblade_bridge_cpp.log"


def clear_native_bridge_log(sim_path: Path) -> None:
    log_path = native_bridge_log_path(sim_path)
    if log_path is not None and log_path.is_file():
        log_path.unlink()


def qblade_external_library_candidate_names(library_stem: str | None) -> list[str]:
    """Return extension-expanded library names for a QBlade `LIBFILE_1` stem."""
    if library_stem is None or not str(library_stem).strip():
        return []
    stem = str(library_stem).strip()
    return [f"{stem}.so", f"{stem}.dylib", f"{stem}.dll"]


def qblade_external_required_symbols(function_name: str | None) -> list[str]:
    """Return QBlade's expected native callback symbols for `LIBFUNCTION_1`."""
    if function_name is None or not str(function_name).strip():
        return []
    name = str(function_name).strip()
    return [name, f"{name}_init", f"{name}_message"]


def qblade_external_library_symbol_diagnostics(
    library_path: Path | None,
    function_name: str | None,
) -> dict[str, object]:
    """Check whether a native callback library exports QBlade's expected symbols."""
    required = qblade_external_required_symbols(function_name)
    diagnostics: dict[str, object] = {
        "library_required_symbols": required,
        "library_symbol_check_available": False,
        "library_symbol_check_tool": "nm -D --defined-only",
        "library_exported_required_symbols": [],
        "library_missing_required_symbols": required,
        "library_symbol_check_error": None,
    }
    if library_path is None or not required:
        return diagnostics
    path = Path(library_path)
    if not path.is_file():
        diagnostics["library_symbol_check_error"] = f"library file not found: {path}"
        return diagnostics
    try:
        completed = subprocess.run(
            ["nm", "-D", "--defined-only", str(path)],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (
        Exception
    ) as exc:  # pragma: no cover - depends on local binutils availability.
        diagnostics["library_symbol_check_error"] = str(exc)
        return diagnostics
    if completed.returncode != 0:
        diagnostics["library_symbol_check_error"] = (
            completed.stderr or completed.stdout
        ).strip()
        return diagnostics
    exported: set[str] = set()
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        # `--defined-only` is the primary guard.  Retain an explicit symbol-type
        # check as defense in depth for nm implementations that ignore it.
        symbol_type = fields[-2]
        if symbol_type in {"U", "u"}:
            continue
        exported.add(fields[-1])
    present = [symbol for symbol in required if symbol in exported]
    missing = [symbol for symbol in required if symbol not in exported]
    diagnostics.update(
        {
            "library_symbol_check_available": True,
            "library_exported_required_symbols": present,
            "library_missing_required_symbols": missing,
        }
    )
    return diagnostics


def qblade_external_library_search_roots(qblade_root: Path | None) -> list[Path]:
    """Return QBlade roots that can contain external controller libraries."""
    if qblade_root is None:
        return []
    root = qblade_root.resolve()
    return [root / "ControllerFiles", root]


def resolve_qblade_external_library(
    qblade_root: Path | None,
    library_stem: str | None,
    function_name: str | None = None,
) -> dict[str, object]:
    """Resolve the configured QBlade external library against ControllerFiles."""
    candidate_names = qblade_external_library_candidate_names(library_stem)
    search_roots = qblade_external_library_search_roots(qblade_root)
    result: dict[str, object] = {
        "library_candidate_names": candidate_names,
        "library_search_roots": [str(root) for root in search_roots],
        "library_file_resolved": None,
        "library_file_exists": None if qblade_root is None else False,
        "library_file_hash": None,
        **qblade_external_library_symbol_diagnostics(None, function_name),
    }
    for root in search_roots:
        if not root.is_dir():
            continue
        for name in candidate_names:
            candidate = root / name
            if candidate.is_file():
                resolved = candidate.resolve()
                result["library_file_resolved"] = str(resolved)
                result["library_file_exists"] = True
                result["library_file_hash"] = file_sha256(resolved)
                result.update(
                    qblade_external_library_symbol_diagnostics(resolved, function_name)
                )
                return result
    return result


def native_callback_configuration(
    sim_path: Path, qblade_root: Path | None = None
) -> dict[str, object]:
    """Return the `.trb` callback block QBlade should use for native bridge runs."""
    info: dict[str, object] = {
        "turbine_file": None,
        "turbine_file_resolved": None,
        "turbine_file_exists": False,
        "library_stem": None,
        "function_name": None,
        "array_size": None,
        "parameter_file_from_turbine": None,
        "parameter_file_resolved": None,
        "parameter_file_exists": False,
        "callback_block_present": False,
        "callback_block_error": None,
        **resolve_qblade_external_library(qblade_root, None),
    }
    try:
        turbine_file = read_qblade_keyword(sim_path, "TURBFILE")
        info["turbine_file"] = turbine_file
        if turbine_file is None:
            raise ValueError(f"Could not find TURBFILE in {sim_path}")
        turbine_path = resolve_qblade_path(sim_path, turbine_file)
        info["turbine_file_resolved"] = str(turbine_path)
        info["turbine_file_exists"] = turbine_path.is_file()
        if not turbine_path.is_file():
            raise FileNotFoundError(
                f"QBlade turbine definition not found: {turbine_path}"
            )

        library_stem = read_qblade_keyword(turbine_path, "LIBFILE_1")
        function_name = read_qblade_keyword(turbine_path, "LIBFUNCTION_1")
        array_size = read_qblade_keyword(turbine_path, "LIBARRAYSIZE_1")
        parameter_file = read_qblade_keyword(turbine_path, "LIBPARAMETERFILE_1")
        info.update(
            {
                "library_stem": library_stem,
                "function_name": function_name,
                "array_size": array_size,
                "parameter_file_from_turbine": parameter_file,
                **resolve_qblade_external_library(
                    qblade_root, library_stem, function_name
                ),
            }
        )
        if parameter_file is not None:
            parameter_path = resolve_qblade_path(turbine_path, parameter_file)
            info["parameter_file_resolved"] = str(parameter_path)
            info["parameter_file_exists"] = parameter_path.is_file()
        info["callback_block_present"] = all(
            str(value or "").strip()
            for value in (library_stem, function_name, array_size, parameter_file)
        )
    except Exception as exc:
        info["callback_block_error"] = str(exc)
    return info


def _reject_duplicate_json_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Modal-burst external configuration contains duplicate JSON key {key!r}."
            )
        result[key] = value
    return result


def load_strict_modal_external_config(
    sim_path: Path,
) -> tuple[Path, dict[str, object]]:
    """Load the modal mapping config under the exact native string contract.

    The C++ bridge uses a deliberately small first-match string extractor for
    ``source_parameter_dir``.  Canonical JSON, duplicate-key rejection, and an
    unescaped POSIX path make that extractor agree exactly with Python's JSON
    decoder instead of merely agreeing for ordinary inputs.
    """

    parameter_file = find_qblade_external_parameter_file(Path(sim_path).resolve())
    data = parameter_file.read_bytes()
    try:
        text = data.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Modal-burst external configuration is not strict JSON: {parameter_file}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("Modal-burst external configuration must be a JSON object.")
    config = dict(parsed)
    canonical = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if text != canonical:
        raise ValueError(
            "Modal-burst external configuration must use canonical sorted JSON so the "
            "native C++ and Python parsers consume identical bytes."
        )

    source_parameter_dir = config.get("source_parameter_dir")
    if not isinstance(source_parameter_dir, str) or not source_parameter_dir:
        raise ValueError(
            "Modal-burst source_parameter_dir must be one nonempty string."
        )
    if (
        not source_parameter_dir.isascii()
        or "\\" in source_parameter_dir
        or '"' in source_parameter_dir
        or any(ord(ch) < 0x20 for ch in source_parameter_dir)
    ):
        raise ValueError(
            "Modal-burst source_parameter_dir must be an unescaped ASCII POSIX path."
        )
    expected_member = f'  "source_parameter_dir": "{source_parameter_dir}",'
    if canonical.splitlines().count(expected_member) != 1:
        raise ValueError(
            "Modal-burst source_parameter_dir must have exactly one unescaped canonical "
            "JSON member."
        )
    source_dir = Path(source_parameter_dir)
    if (
        not source_dir.is_absolute()
        or source_dir.resolve() != parameter_file.parent.resolve()
    ):
        raise ValueError(
            "Modal-burst source_parameter_dir must be the canonical absolute parameter-file "
            "parent used by the native bridge."
        )
    return parameter_file, config


def build_modal_native_bridge_runtime_identity(
    library_path: Path,
) -> dict[str, object]:
    """Build the exact config block that pins the approved modal bridge bytes."""

    path = Path(library_path).resolve()
    if path.name != "libvorlap_qblade_bridge.so" or not path.is_file():
        raise ValueError(
            "Modal-burst approved bridge must be libvorlap_qblade_bridge.so."
        )
    identity = file_sha256(path)
    if identity is None:
        raise FileNotFoundError(path)
    return {
        "schema": MODAL_NATIVE_BRIDGE_RUNTIME_IDENTITY_SCHEMA,
        "resolved_library_filename": path.name,
        "resolved_library_relative_path": f"ControllerFiles/{path.name}",
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
        "required_symbols": list(_MODAL_NATIVE_REQUIRED_SYMBOLS),
        "logging_contract": dict(_MODAL_NATIVE_LOGGING_CONTRACT),
    }


def require_modal_native_bridge_runtime_contract(
    sim_path: Path,
    qblade_root: Path,
) -> dict[str, object]:
    """Require the exact approved logging bridge bytes and native call contract."""

    raw_log_override = os.environ.get("VORLAP_QBLADE_BRIDGE_LOG_DIR")
    if raw_log_override is not None and raw_log_override != "":
        raise ValueError(
            "Modal-burst runs prohibit any raw nonempty "
            "VORLAP_QBLADE_BRIDGE_LOG_DIR value."
        )
    parameter_file, config = load_strict_modal_external_config(sim_path)
    expected = config.get("native_bridge_runtime_identity")
    if not isinstance(expected, dict) or set(expected) != set(
        _MODAL_NATIVE_BRIDGE_IDENTITY_KEYS
    ):
        raise ValueError(
            "Modal-burst external configuration must bind the exact native bridge runtime "
            "identity schema."
        )
    if expected.get("schema") != MODAL_NATIVE_BRIDGE_RUNTIME_IDENTITY_SCHEMA:
        raise ValueError(
            "Modal-burst native bridge runtime identity schema is invalid."
        )
    if expected.get("required_symbols") != _MODAL_NATIVE_REQUIRED_SYMBOLS:
        raise ValueError(
            "Modal-burst native bridge identity must require update, update_init, and "
            "update_message."
        )
    if expected.get("logging_contract") != _MODAL_NATIVE_LOGGING_CONTRACT:
        raise ValueError("Modal-burst native bridge logging contract is invalid.")

    filename = expected.get("resolved_library_filename")
    relative_path = expected.get("resolved_library_relative_path")
    if (
        not isinstance(filename, str)
        or filename != "libvorlap_qblade_bridge.so"
        or relative_path != f"ControllerFiles/{filename}"
    ):
        raise ValueError(
            "Modal-burst native bridge must be the approved "
            "ControllerFiles/libvorlap_qblade_bridge.so artifact."
        )
    approved_library = Path(qblade_root).resolve() / str(relative_path)
    if approved_library.is_symlink() or not approved_library.is_file():
        raise ValueError(
            "Modal-burst approved native bridge must be a regular non-symlink file."
        )
    approved_identity = file_sha256(approved_library)
    if approved_identity is None:
        raise FileNotFoundError(approved_library)
    if expected.get("sha256") != approved_identity.get("sha256") or expected.get(
        "size_bytes"
    ) != approved_identity.get("size_bytes"):
        raise ValueError(
            "Modal-burst live native bridge bytes do not match the approved hash-bound "
            "runtime identity."
        )

    callback = native_callback_configuration(sim_path, qblade_root=qblade_root)
    callback_expected = {
        "library_stem": "libvorlap_qblade_bridge",
        "function_name": "update",
        "array_size": "237",
        "parameter_file_resolved": str(parameter_file.resolve()),
        "library_file_resolved": str(approved_library.resolve()),
    }
    callback_mismatches = {
        key: {"expected": value, "observed": callback.get(key)}
        for key, value in callback_expected.items()
        if str(callback.get(key)) != str(value)
    }
    if callback_mismatches:
        raise ValueError(
            "Modal-burst turbine native callback contract mismatch: "
            + json.dumps(callback_mismatches, sort_keys=True)
        )
    if callback.get("callback_block_present") is not True:
        raise ValueError("Modal-burst turbine callback block is incomplete.")
    if (
        callback.get("library_symbol_check_available") is not True
        or callback.get("library_missing_required_symbols") != []
        or callback.get("library_exported_required_symbols")
        != _MODAL_NATIVE_REQUIRED_SYMBOLS
    ):
        raise ValueError(
            "Modal-burst approved native bridge must define update, update_init, and "
            "update_message."
        )
    return {
        "schema": MODAL_NATIVE_BRIDGE_RUNTIME_AUDIT_SCHEMA,
        "passed": True,
        "parameter_file": str(parameter_file.resolve()),
        "source_parameter_dir": str(parameter_file.parent.resolve()),
        "expected_identity": expected,
        "observed_identity": {
            "resolved_library_filename": filename,
            "resolved_library_relative_path": relative_path,
            "sha256": approved_identity["sha256"],
            "size_bytes": approved_identity["size_bytes"],
        },
        "required_symbols_defined": list(_MODAL_NATIVE_REQUIRED_SYMBOLS),
        "log_file": str(
            parameter_file.parent / str(_MODAL_NATIVE_LOGGING_CONTRACT["log_filename"])
        ),
        "raw_log_environment_override_absent_or_empty": True,
    }


def native_callback_evidence(
    sim_path: Path, qblade_root: Path | None = None
) -> dict[str, object]:
    """Summarize the C++ bridge callback log produced by native QBlade runs."""
    parameter_file, _, config_error = load_vorlap_config_for_sim(sim_path)
    log_path = None if parameter_file is None else native_bridge_log_path(sim_path)
    log_entry_exists = bool(log_path is not None and os.path.lexists(log_path))
    log_is_regular_nonsymlink = bool(
        log_path is not None
        and log_entry_exists
        and not log_path.is_symlink()
        and log_path.is_file()
    )
    callback_configuration = native_callback_configuration(
        sim_path, qblade_root=qblade_root
    )
    evidence: dict[str, object] = {
        "parameter_file": None if parameter_file is None else str(parameter_file),
        "config_error": config_error,
        "log_file": None if log_path is None else str(log_path),
        # Keep the legacy key name, but use lexists semantics so a directory,
        # FIFO, or dangling symlink cannot masquerade as an absent callback log.
        "log_file_exists": log_entry_exists,
        "log_file_hash": (file_sha256(log_path) if log_is_regular_nonsymlink else None),
        **callback_configuration,
        "update_init_seen": False,
        "update_init_count": 0,
        "update_log_entry_count": 0,
        "update_cache_observation_count": 0,
        "update_count_observed_max": 0,
        "runtime_initialized_message_seen": False,
        "python_env_before_import_seen": False,
        "python_env_after_import_seen": False,
        "python_env_before_import": None,
        "python_env_after_import": None,
        "python_env_parse_errors": [],
        "python_path_entries_added": [],
        "numpy_version": None,
        "numpy_file": None,
        "vorlap_file": None,
        "vorlap_qblade_runtime_file": None,
    }
    if not log_is_regular_nonsymlink:
        return evidence
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    update_counts = [
        int(match) for match in re.findall(r"update entered; count=(\d+)", text)
    ]
    update_cache_observed_counts = [
        int(hits) + int(misses)
        for hits, misses in re.findall(r"\bcache=(\d+)/(\d+)\b", text)
    ]
    update_init_count = len(re.findall(r"update_init entered", text))
    python_env_parse_errors: list[str] = []
    python_env_before = None
    python_env_after = None
    for label in ("python_env_before_import", "python_env_after_import"):
        matches = re.findall(rf"^{re.escape(label)}=(.*)$", text, flags=re.MULTILINE)
        for raw in matches:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                python_env_parse_errors.append(f"{label}: {exc}")
                continue
            if label == "python_env_before_import":
                python_env_before = parsed
            else:
                python_env_after = parsed
    evidence.update(
        {
            "update_init_seen": update_init_count > 0,
            "update_init_count": update_init_count,
            "update_log_entry_count": len(update_counts),
            "update_cache_observation_count": len(update_cache_observed_counts),
            "update_count_observed_max": max(
                [0, *update_counts, *update_cache_observed_counts]
            ),
            "runtime_initialized_message_seen": "VorLap QBlade runtime initialized."
            in text,
            "python_env_before_import_seen": python_env_before is not None,
            "python_env_after_import_seen": python_env_after is not None,
            "python_env_before_import": python_env_before,
            "python_env_after_import": python_env_after,
            "python_env_parse_errors": python_env_parse_errors,
            "python_path_entries_added": (
                []
                if not isinstance(python_env_before, dict)
                else python_env_before.get("python_path_entries_added", [])
            ),
            "numpy_version": (
                None
                if not isinstance(python_env_after, dict)
                else python_env_after.get("numpy_version")
            ),
            "numpy_file": (
                None
                if not isinstance(python_env_after, dict)
                else python_env_after.get("numpy_file")
            ),
            "vorlap_file": (
                None
                if not isinstance(python_env_after, dict)
                else python_env_after.get("vorlap_file")
            ),
            "vorlap_qblade_runtime_file": (
                None
                if not isinstance(python_env_after, dict)
                else python_env_after.get("vorlap_qblade_runtime_file")
            ),
        }
    )
    return evidence


def require_zero_native_callback_evidence(
    evidence: dict[str, object],
    *,
    context: str,
    reject_existing_log: bool = False,
) -> dict[str, object]:
    """Fail closed unless native callback evidence is exactly zero activity."""

    init_count = int(evidence.get("update_init_count", 0) or 0)
    update_count = max(
        int(evidence.get("update_log_entry_count", 0) or 0),
        int(evidence.get("update_cache_observation_count", 0) or 0),
        int(evidence.get("update_count_observed_max", 0) or 0),
    )
    if reject_existing_log and evidence.get("log_file_exists") is True:
        raise RuntimeError(
            f"{context}: a native bridge log already exists; refusing to clear or reuse "
            "stale callback evidence."
        )
    if (
        evidence.get("log_file_exists") is True
        and evidence.get("log_file_hash") is None
    ):
        raise RuntimeError(
            f"{context}: native bridge log target exists but is not a regular "
            "non-symlink file."
        )
    if init_count or update_count:
        raise RuntimeError(
            f"{context}: native callback activity must be exactly zero; "
            f"update_init={init_count}, update={update_count}."
        )
    return evidence


def find_bridge_library(
    qblade_root: Path, library_stem: str | None = None
) -> Path | None:
    names = []
    if library_stem:
        names.extend(
            [
                f"{library_stem}.so",
                f"{library_stem}.dylib",
                f"{library_stem}.dll",
            ]
        )
    names.extend(
        [
            "libvorlap_qblade_bridge.so",
            "libvorlap_qblade_bridge.dylib",
            "libvorlap_qblade_bridge.dll",
        ]
    )
    search_roots = [qblade_root / "ControllerFiles", qblade_root]
    for root in search_roots:
        if not root.is_dir():
            continue
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate.resolve()
    return None


def qblade_case_settings(sim_path: Path) -> dict[str, str | None]:
    keys = [
        "TURBFILE",
        "NUMTIMESTEPS",
        "TIMESTEP",
        "STOREFROM",
        "STOREREPLAY",
        "FILTERFILE",
        "MEANINF",
        "INITIAL_AZIMUTH",
        "PRESCRIBETYPE",
        "RPMPRESCRIBED",
        "WNDTYPE",
        "WNDNAME",
        "HORANGLE",
        "VERTANGLE",
        "SHEAREXP",
        "REFHEIGHT",
        "DENSITYAIR",
        "VISCOSITYAIR",
        "RELAXSTEPS",
        "RELAXTIMESTEP",
        "RAMPUP",
        "ADDDAMP",
        "STRITERATIONS",
    ]
    return {key: read_qblade_keyword(sim_path, key) for key in keys}


def preflight_qblade_wind_input(sim_path: Path) -> dict[str, object]:
    """Validate wind-file availability before QBlade emits a generic import failure."""
    raw_wind_type = read_qblade_keyword(sim_path, "WNDTYPE")
    raw_wind_name = read_qblade_keyword(sim_path, "WNDNAME")
    diagnostics: dict[str, object] = {
        "wind_type": raw_wind_type,
        "wind_name": raw_wind_name,
        "requires_file": False,
        "resolved_wind_file": None,
        "exists": None,
    }
    if raw_wind_type is None:
        return diagnostics
    try:
        wind_type = int(float(raw_wind_type))
    except ValueError as exc:
        raise ValueError(
            f"Invalid WNDTYPE value in {sim_path}: {raw_wind_type!r}"
        ) from exc
    diagnostics["wind_type"] = wind_type
    if wind_type not in (1, 2):
        return diagnostics

    wind_name = "" if raw_wind_name is None else str(raw_wind_name).strip()
    diagnostics["requires_file"] = True
    if not wind_name:
        raise FileNotFoundError(
            f"QBlade WNDTYPE={wind_type} in {sim_path} requires a non-empty WNDNAME."
        )
    wind_path = resolve_qblade_path(sim_path, wind_name)
    diagnostics["resolved_wind_file"] = str(wind_path)
    diagnostics["exists"] = wind_path.is_file()
    if not wind_path.is_file():
        raise FileNotFoundError(
            f"QBlade WNDTYPE={wind_type} requires WNDNAME file {wind_name!r}, "
            f"but it was not found at {wind_path}."
        )
    return diagnostics


def selected_environment() -> dict[str, str | None]:
    keys = [
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "PATH",
        "QBLADE_ROOT",
        "QBLADE_EXTRA_LIBRARY_PATH",
        "QBLADE_EXTRA_LIBRARY_PATHS",
        "VORLAP_ROOT",
        "VORLAP_QBLADE_LOADER_ENV",
        "VORLAP_HVAWT_PROFILE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    return {key: os.environ.get(key) for key in keys}


def numpy_blas_metadata() -> dict[str, object]:
    """Return the active NumPy build's BLAS declaration for run receipts."""
    try:
        import numpy as np

        try:
            configuration = np.__config__.show(mode="dicts")
        except TypeError:
            # NumPy <2 has no structured ``mode``.  Preserve its complete
            # human-readable declaration rather than dropping BLAS evidence.
            from contextlib import redirect_stdout

            stream = io.StringIO()
            with redirect_stdout(stream):
                np.__config__.show()
            return {
                "numpy_version": np.__version__,
                "configuration_text": stream.getvalue(),
            }
        dependencies = configuration.get("Build Dependencies", {})
        return {
            "numpy_version": np.__version__,
            "blas": dependencies.get("blas"),
            "lapack": dependencies.get("lapack"),
        }
    except Exception as exc:  # pragma: no cover - version/environment dependent.
        return {"available": False, "error": str(exc)}


def build_reproducibility_metadata(
    *,
    command_line: list[str],
    sim_path: Path,
    output_dir: Path,
    qblade_root: Path,
    qblade_library: Path,
    coupling_mode: str,
    requested_steps: int,
    completed_steps: int,
    threads: int,
    use_opencl: bool,
    qblade_execution_mode: str = "step-loop",
    case_metadata: dict | None = None,
    result_filter: Path | None = None,
    modal_burst: bool = False,
) -> dict[str, object]:
    """Collect per-run provenance needed to reproduce a paper figure."""
    parameter_file, config, config_error = load_vorlap_config_for_sim(sim_path)
    library_stem = None if config is None else config.get("library_stem")
    bridge_library = find_bridge_library(
        qblade_root, None if library_stem is None else str(library_stem)
    )
    headless_root = Path(__file__).resolve().parents[1]
    input_files = {
        "sim": file_sha256(sim_path),
        "vorlap_config": (
            None if parameter_file is None else file_sha256(parameter_file)
        ),
        "result_filter": None if result_filter is None else file_sha256(result_filter),
    }
    turbine_file = read_qblade_keyword(sim_path, "TURBFILE")
    if turbine_file:
        turbine_path = resolve_qblade_path(sim_path, turbine_file)
        input_files["turbine"] = file_sha256(turbine_path)
    runtime_assets, runtime_asset_fingerprint = qblade_runtime_asset_fingerprint(
        sim_path
    )
    python_runtime_assets, python_runtime_asset_fingerprint = (
        vorlap_python_runtime_manifest()
    )
    response_model = None if config is None else config.get("response_model")
    response_schema_v2 = (
        isinstance(response_model, dict) and response_model.get("schema_version") == 2
    )
    dependency_manifest_required = response_schema_v2 or bool(modal_burst)
    if dependency_manifest_required:
        qblade_dependencies, qblade_dependency_fingerprint = (
            qblade_runtime_dependency_manifest(qblade_root)
        )
    if response_schema_v2:
        runner_path = (
            Path(command_line[1]).expanduser().resolve()
            if len(command_line) > 1
            else None
        )

    metadata: dict[str, object] = {
        "command_line": command_line,
        "working_directory": str(Path.cwd()),
        "output_dir": str(output_dir),
        "created_unix": time.time(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "packages": package_versions(
            (
                ["numpy", "scipy", "pandas", "matplotlib", "h5py", "vorlap"]
                if response_schema_v2
                else ["numpy", "pandas", "matplotlib", "h5py", "vorlap"]
            )
        ),
        "repositories": {
            "Headless": git_metadata(headless_root),
            "VorLap": vorlap_package_metadata(),
        },
        "qblade": {
            "root": str(qblade_root),
            "version_hint": qblade_root.name,
            "library": file_sha256(qblade_library),
        },
        "bridge_library": (
            None if bridge_library is None else file_sha256(bridge_library)
        ),
        "coupling_mode": coupling_mode,
        "qblade_execution_mode": qblade_execution_mode,
        "vorlap_config": {
            "path": None if parameter_file is None else str(parameter_file),
            "error": config_error,
            "force_scale": None if config is None else config.get("force_scale"),
            "include_mean_load": (
                None if config is None else config.get("include_mean_load", False)
            ),
            "n_freq_depth": None if config is None else config.get("n_freq_depth"),
            "source_frequency_cap_hz": (
                None if config is None else config.get("source_frequency_cap_hz")
            ),
            "library_stem": library_stem,
            "function_name": None if config is None else config.get("function_name"),
            "python_path_entries": (
                None if config is None else config.get("python_path_entries")
            ),
            "swap_size": None if config is None else config.get("swap_size"),
            "node_count": None if config is None else len(config.get("node_ids", [])),
        },
        "qblade_case_settings": qblade_case_settings(sim_path),
        "input_files": input_files,
        "runtime_assets": runtime_assets,
        "runtime_asset_fingerprint": runtime_asset_fingerprint,
        "vorlap_python_runtime_assets": python_runtime_assets,
        "vorlap_python_runtime_fingerprint": python_runtime_asset_fingerprint,
        "steps": {
            "requested": requested_steps,
            "completed": completed_steps,
        },
        "thread_settings": {
            "threads": threads,
            "use_opencl": use_opencl,
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        },
        "numpy_blas": numpy_blas_metadata(),
        "loader_environment": selected_environment(),
        "case_metadata": case_metadata,
    }
    if response_schema_v2:
        metadata["runner"] = (
            None
            if runner_path is None or not runner_path.is_file()
            else file_sha256(runner_path)
        )
    if dependency_manifest_required:
        metadata["qblade_runtime_dependencies"] = qblade_dependencies
        metadata["qblade_runtime_dependency_fingerprint"] = (
            qblade_dependency_fingerprint
        )
    return metadata


def find_qblade_library(qblade_root: Path) -> Path:
    patterns = ("*QBlade*.so*", "*QBlade*.dll", "*QBlade*.dylib")
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(qblade_root.glob(pattern))
    matches = sorted(path for path in matches if path.is_file())
    if not matches:
        raise FileNotFoundError(f"No QBlade shared library found in {qblade_root}")
    return matches[0]


def qblade_library_dirs(qblade_root: Path) -> list[Path]:
    dirs = [
        qblade_root,
        qblade_root / "Libraries",
        qblade_root / "ControllerFiles",
        qblade_root / "Binaries",
        qblade_root / "Binaries" / "IPCBridge",
    ]
    extra = os.environ.get("QBLADE_EXTRA_LIBRARY_PATHS") or os.environ.get(
        "QBLADE_EXTRA_LIBRARY_PATH", ""
    )
    dirs.extend(Path(part).expanduser() for part in extra.split(os.pathsep) if part)
    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_dirs.append(path.resolve())
    return unique_dirs


def prepend_env_path(env: dict[str, str], key: str, values: list[Path]) -> None:
    existing = [part for part in env.get(key, "").split(os.pathsep) if part]
    rendered = [str(value) for value in values]
    env[key] = os.pathsep.join(
        rendered + [part for part in existing if part not in rendered]
    )


def ensure_qblade_loader_env(qblade_root: Path) -> None:
    if not sys.platform.startswith("linux"):
        return
    library_dirs = qblade_library_dirs(qblade_root)
    current = {
        part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part
    }
    required = {str(path) for path in library_dirs}
    if os.environ.get("VORLAP_QBLADE_LOADER_ENV") == "1" and required.issubset(current):
        return
    env = os.environ.copy()
    prepend_env_path(env, "LD_LIBRARY_PATH", library_dirs)
    env["VORLAP_QBLADE_LOADER_ENV"] = "1"
    original_argv = getattr(sys, "orig_argv", None)
    if (
        isinstance(original_argv, list)
        and len(original_argv) >= 1
        and all(isinstance(value, str) for value in original_argv)
    ):
        reexec_argv = [sys.executable, *original_argv[1:]]
    else:  # pragma: no cover - Python 3.10+ always exposes sys.orig_argv.
        reexec_argv = [sys.executable, *sys.argv]
    os.execvpe(sys.executable, reexec_argv, env)


def load_qblade_api(qblade_root: Path):
    sil_path = qblade_root / "SIL_Interface"
    if not sil_path.is_dir():
        raise FileNotFoundError(f"Missing QBlade SIL_Interface directory: {sil_path}")
    sys.path.insert(0, str(sil_path))
    from QBladeLibrary import QBladeLibrary  # type: ignore

    library_path = find_qblade_library(qblade_root)
    return QBladeLibrary(str(library_path)), library_path


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish JSON without sharing a temporary name across writers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    tmp_path = Path(raw_tmp_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def checkpoint_directory(args: Any, output_dir: Path) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir).expanduser().resolve()
    return output_dir / "checkpoints"


def _checkpoint_project_path(
    metadata: dict[str, object], metadata_path: Path
) -> Path | None:
    raw_project = str(metadata.get("qblade_project", "") or "").strip()
    if not raw_project:
        return None
    project = Path(raw_project).expanduser()
    if not project.is_absolute():
        project = (metadata_path.parent / project).resolve()
    return project


def _checkpoint_project_validation_error(
    metadata: dict[str, object],
    metadata_path: Path,
) -> tuple[Path | None, str | None]:
    project = _checkpoint_project_path(metadata, metadata_path)
    if project is None:
        return None, "metadata does not name a QBlade project"
    if not project.is_file():
        return project, f"QBlade project is missing: {project}"
    stat = project.stat()
    if stat.st_size <= 0:
        return project, f"QBlade project is empty: {project}"

    expected_sha256 = str(metadata.get("qblade_project_sha256", "") or "").strip()
    try:
        expected_size = int(metadata.get("qblade_project_size_bytes", 0) or 0)
    except (TypeError, ValueError):
        expected_size = 0
    if not expected_sha256 or expected_size <= 0:
        return project, "metadata does not bind the QBlade project hash and size"
    if stat.st_size != expected_size:
        return (
            project,
            f"QBlade project size changed from {expected_size} to {stat.st_size} bytes",
        )
    identity = file_sha256(project)
    observed_sha256 = "" if identity is None else str(identity.get("sha256", "") or "")
    if observed_sha256 != expected_sha256:
        return project, "QBlade project SHA-256 no longer matches checkpoint metadata"
    return project, None


def quarantine_incompatible_checkpoint_pairs(
    checkpoint_dir: Path,
    *,
    expected_fingerprint: str,
    stamp: int,
) -> tuple[Path | None, list[str]]:
    """Move stale step metadata/QPR pairs aside so pruning cannot retain them."""
    quarantine_dir = checkpoint_dir / f"incompatible-{stamp}"
    moved: list[str] = []

    def quarantine(path: Path) -> None:
        if not path.is_file():
            return
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        destination = quarantine_dir / path.name
        os.replace(path, destination)
        moved.append(str(destination))

    for metadata_path in sorted(checkpoint_dir.glob("*_step_*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            fingerprint = str(metadata.get("checkpoint_identity_fingerprint", "") or "")
        except (OSError, json.JSONDecodeError, AttributeError):
            fingerprint = ""
        project, project_error = _checkpoint_project_validation_error(
            metadata, metadata_path
        )
        if fingerprint == expected_fingerprint and project_error is None:
            continue
        quarantine(metadata_path)
        if project is not None and project.parent.resolve() == checkpoint_dir.resolve():
            quarantine(project)
        else:
            quarantine(metadata_path.with_suffix(".qpr"))

    # A project without live step metadata is interrupted or stale and must not
    # participate in the next retention/pruning pass.
    for project_path in sorted(checkpoint_dir.glob("*_step_*.qpr")):
        if not project_path.with_suffix(".json").is_file():
            quarantine(project_path)

    return (quarantine_dir if moved else None), moved


def _invalidate_automatic_checkpoint(
    args: Any,
    path: Path,
    checkpoint_dir: Path,
    *,
    message: str,
    expected_fingerprint: str,
    observed_fingerprint: str,
) -> None:
    stamp = time.time_ns()
    quarantined_dir, quarantined_files = quarantine_incompatible_checkpoint_pairs(
        checkpoint_dir,
        expected_fingerprint=expected_fingerprint,
        stamp=stamp,
    )
    invalidated = path.with_name(f"{path.stem}.incompatible-{stamp}{path.suffix}")
    os.replace(path, invalidated)
    setattr(
        args,
        "_checkpoint_rejection",
        {
            "reason": message,
            "invalidated_metadata": str(invalidated),
            "observed_fingerprint": observed_fingerprint or None,
            "expected_fingerprint": expected_fingerprint,
            "quarantined_checkpoint_dir": (
                None if quarantined_dir is None else str(quarantined_dir)
            ),
            "quarantined_checkpoint_files": quarantined_files,
        },
    )
    print(
        f"WARNING: {message} Quarantined pointer as {invalidated}"
        + (
            f" and {len(quarantined_files)} stale checkpoint file(s) under {quarantined_dir}."
            if quarantined_dir is not None
            else "."
        ),
        file=sys.stderr,
        flush=True,
    )


def load_resume_checkpoint(
    args: Any,
    checkpoint_dir: Path,
    *,
    expected_identity: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if args.resume_checkpoint:
        path = Path(args.resume_checkpoint).expanduser().resolve()
    elif args.resume_from_checkpoint:
        path = checkpoint_dir / "latest_checkpoint.json"
    else:
        return None
    if not path.is_file():
        if args.resume_checkpoint:
            raise FileNotFoundError(f"Requested resume checkpoint not found: {path}")
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {path}")
    if expected_identity is not None:
        expected_fingerprint = str(expected_identity.get("fingerprint", ""))
        observed_fingerprint = str(metadata.get("checkpoint_identity_fingerprint", ""))
        if not observed_fingerprint or observed_fingerprint != expected_fingerprint:
            message = (
                f"Checkpoint {path} is incompatible with the current run configuration "
                f"(observed identity {observed_fingerprint or 'missing'}, expected {expected_fingerprint})."
            )
            if args.resume_checkpoint:
                raise ValueError(message)
            _invalidate_automatic_checkpoint(
                args,
                path,
                checkpoint_dir,
                message=message,
                expected_fingerprint=expected_fingerprint,
                observed_fingerprint=observed_fingerprint,
            )
            return None
    else:
        expected_fingerprint = ""
        observed_fingerprint = str(
            metadata.get("checkpoint_identity_fingerprint", "") or ""
        )

    project, project_error = _checkpoint_project_validation_error(metadata, path)
    if project_error is not None:
        message = f"Checkpoint {path} is stale or incomplete: {project_error}."
        if args.resume_checkpoint:
            raise ValueError(message)
        _invalidate_automatic_checkpoint(
            args,
            path,
            checkpoint_dir,
            message=message,
            expected_fingerprint=expected_fingerprint,
            observed_fingerprint=observed_fingerprint,
        )
        return None
    metadata["_metadata_path"] = str(path)
    assert project is not None
    metadata["qblade_project"] = str(project)
    return metadata


def _checkpoint_step_from_metadata(path: Path) -> int:
    match = re.search(r"_step_(\d+)\.json$", path.name)
    if match:
        return int(match.group(1))
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        return int(metadata.get("completed_steps", 0) or 0)
    except Exception:
        return 0


def prune_restart_checkpoints(checkpoint_dir: Path, keep: int = 2) -> list[str]:
    """Delete older complete checkpoint pairs, retaining the newest `keep`."""
    keep = max(0, int(keep))
    metadata_paths = sorted(
        checkpoint_dir.glob("*_step_*.json"),
        key=lambda path: (
            _checkpoint_step_from_metadata(path),
            path.stat().st_mtime,
            path.name,
        ),
    )
    stale = metadata_paths[:-keep] if keep else metadata_paths
    deleted: list[str] = []
    for metadata_path in stale:
        project_path = metadata_path.with_suffix(".qpr")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate = Path(str(metadata.get("qblade_project", ""))).expanduser()
            if not candidate.is_absolute():
                candidate = (metadata_path.parent / candidate).resolve()
            if candidate.name == project_path.name:
                project_path = candidate
        except Exception:
            pass
        for path in (metadata_path, project_path):
            try:
                if path.exists():
                    path.unlink()
                    deleted.append(str(path))
            except FileNotFoundError:
                pass
    return deleted


MODAL_BURST_RESUME_POSITION_SCHEMA = "vorlap-qblade-modal-resume-position-v1"
_MODAL_BURST_RESUME_POSITION_KEYS = frozenset(
    {
        "schema",
        "completed_steps",
        "expected_post_advance_physical_sample_index",
        "qblade_time_post_advance_s",
        "qblade_post_time_grid_audit",
        "latest_wrapper_step_index",
        "latest_qblade_time_pre_advance_s",
        "latest_azimuth_pre_advance_deg",
        "expected_time_increment_from_latest_witness_s",
        "observed_time_increment_from_latest_witness_s",
        "time_increment_tolerance_s",
        "qblade_azimuth_post_advance_deg",
        "expected_azimuth_post_advance_deg",
        "azimuth_periodic_residual_deg",
        "azimuth_tolerance_deg",
        "passed",
    }
)


def modal_burst_resume_position_evidence(
    qblade: Any,
    modal_burst: Any,
    completed_steps: int,
    timing_witnesses: list[dict[str, object]],
) -> dict[str, object]:
    """Bind a stored QPR to the QBlade position after its last completed advance."""

    from .qblade_modal_decay import (
        AZIMUTH_CHANNEL,
        TIME_CHANNEL,
        _validate_qblade_pre_time,
    )

    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
        raise ValueError("Modal-burst completed_steps must be an integer.")
    if completed_steps < 1:
        raise ValueError(
            "Modal-burst resume-position evidence requires a completed step."
        )
    if not timing_witnesses:
        raise ValueError(
            "Modal-burst resume-position evidence requires timing witnesses."
        )
    latest = max(
        timing_witnesses,
        key=lambda row: int(row.get("wrapper_step_index", -1)),
    )
    latest_index = latest.get("wrapper_step_index")
    if latest_index != completed_steps - 1:
        raise ValueError(
            "Modal-burst latest timing witness must be wrapper completed_steps-1 before "
            "checkpoint publication."
        )
    latest_time = float(latest.get("qblade_time_pre_advance_s"))
    latest_azimuth = float(latest.get("azimuth_deg"))
    if not math.isfinite(latest_time) or not math.isfinite(latest_azimuth):
        raise ValueError(
            "Modal-burst latest timing witness has non-finite position data."
        )

    observed_time = float(
        qblade.getCustomData_at_num(TIME_CHANNEL.encode("utf-8"), 0.0, 0)
    )
    observed_azimuth = float(
        qblade.getCustomData_at_num(AZIMUTH_CHANNEL.encode("utf-8"), 0.0, 0)
    )
    if not math.isfinite(observed_azimuth):
        raise RuntimeError("QBlade returned a non-finite post-advance azimuth.")
    expected_physical_index = max(0, completed_steps - 1)
    time_audit = _validate_qblade_pre_time(
        observed_time,
        timestep_s=float(modal_burst.plan.timestep_s),
        expected_physical_sample_index=expected_physical_index,
    )
    expected_increment = (
        0.0 if completed_steps == 1 else float(modal_burst.plan.timestep_s)
    )
    observed_increment = observed_time - latest_time
    latest_audit = latest.get("qblade_pre_time_grid_audit")
    latest_limit = (
        float(latest_audit.get("float32_residual_limit_s", 0.0))
        if isinstance(latest_audit, dict)
        else 0.0
    )
    increment_tolerance = max(
        1.0e-12,
        latest_limit + float(time_audit["float32_residual_limit_s"]),
    )
    if not math.isclose(
        observed_increment,
        expected_increment,
        rel_tol=0.0,
        abs_tol=increment_tolerance,
    ):
        raise RuntimeError(
            "QBlade post-advance time is inconsistent with the latest modal timing witness: "
            f"expected increment={expected_increment:.17g}, "
            f"observed={observed_increment:.17g}, tolerance={increment_tolerance:.9g}."
        )

    expected_azimuth = (
        latest_azimuth
        + float(modal_burst.plan.rotor_speed_rpm) * 6.0 * observed_increment
    )
    try:
        import numpy as np

        with np.errstate(over="ignore", invalid="ignore"):
            azimuth_ulp = abs(float(np.spacing(np.float32(expected_azimuth))))
    except Exception:  # pragma: no cover - numpy is a required VorLap dependency.
        azimuth_ulp = 0.0
    azimuth_tolerance = max(
        1.0e-7,
        4.0 * azimuth_ulp,
        abs(float(modal_burst.plan.rotor_speed_rpm)) * 6.0 * increment_tolerance,
    )
    azimuth_residual = math.remainder(observed_azimuth - expected_azimuth, 360.0)
    if abs(azimuth_residual) > azimuth_tolerance:
        raise RuntimeError(
            "QBlade post-advance azimuth is inconsistent with the latest modal timing "
            f"witness: residual={azimuth_residual:.9g} deg, "
            f"tolerance={azimuth_tolerance:.9g} deg."
        )
    evidence: dict[str, object] = {
        "schema": MODAL_BURST_RESUME_POSITION_SCHEMA,
        "completed_steps": completed_steps,
        "expected_post_advance_physical_sample_index": expected_physical_index,
        "qblade_time_post_advance_s": observed_time,
        "qblade_post_time_grid_audit": time_audit,
        "latest_wrapper_step_index": latest_index,
        "latest_qblade_time_pre_advance_s": latest_time,
        "latest_azimuth_pre_advance_deg": latest_azimuth,
        "expected_time_increment_from_latest_witness_s": expected_increment,
        "observed_time_increment_from_latest_witness_s": observed_increment,
        "time_increment_tolerance_s": increment_tolerance,
        "qblade_azimuth_post_advance_deg": observed_azimuth,
        "expected_azimuth_post_advance_deg": expected_azimuth,
        "azimuth_periodic_residual_deg": azimuth_residual,
        "azimuth_tolerance_deg": azimuth_tolerance,
        "passed": True,
    }
    if set(evidence) != set(_MODAL_BURST_RESUME_POSITION_KEYS):
        raise RuntimeError("Internal modal resume-position evidence schema drifted.")
    return evidence


def validate_modal_burst_resume_position_record(
    record: object,
    *,
    completed_steps: int,
    timing_witnesses: list[dict[str, object]],
) -> dict[str, object]:
    """Validate the immutable portion of one v4 resume-position record."""

    if not isinstance(record, dict) or set(record) != set(
        _MODAL_BURST_RESUME_POSITION_KEYS
    ):
        raise ValueError(
            "Modal-burst resume position must contain the exact locked schema keys."
        )
    if record.get("schema") != MODAL_BURST_RESUME_POSITION_SCHEMA:
        raise ValueError("Modal-burst resume-position schema is invalid.")
    if (
        record.get("passed") is not True
        or record.get("completed_steps") != completed_steps
    ):
        raise ValueError(
            "Modal-burst resume-position completed-step binding is invalid."
        )
    latest = max(
        timing_witnesses,
        key=lambda row: int(row.get("wrapper_step_index", -1)),
    )
    exact = {
        "latest_wrapper_step_index": latest.get("wrapper_step_index"),
        "latest_qblade_time_pre_advance_s": latest.get("qblade_time_pre_advance_s"),
        "latest_azimuth_pre_advance_deg": latest.get("azimuth_deg"),
        "expected_post_advance_physical_sample_index": max(0, completed_steps - 1),
    }
    mismatches = [key for key, value in exact.items() if record.get(key) != value]
    if mismatches:
        raise ValueError(
            "Modal-burst resume position does not match the latest timing witness for: "
            + ", ".join(mismatches)
        )
    audit = record.get("qblade_post_time_grid_audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise ValueError("Modal-burst resume-position time-grid audit is invalid.")
    audit_exact = {
        "expected_physical_sample_index": max(0, completed_steps - 1),
        "expected_grid_index": max(0, completed_steps - 1),
        "observed_time_s": record.get("qblade_time_post_advance_s"),
    }
    audit_mismatches = [
        key for key, value in audit_exact.items() if audit.get(key) != value
    ]
    if audit_mismatches:
        raise ValueError(
            "Modal-burst resume-position time-grid audit is inconsistent for: "
            + ", ".join(audit_mismatches)
        )
    expected_increment = (
        0.0
        if completed_steps == 1
        else (
            float(record["qblade_time_post_advance_s"])
            - float(latest["qblade_time_pre_advance_s"])
        )
    )
    if (
        float(record["observed_time_increment_from_latest_witness_s"])
        != expected_increment
    ):
        raise ValueError(
            "Modal-burst resume-position time increment is inconsistent with its latest "
            "timing witness."
        )
    for key in (
        "qblade_time_post_advance_s",
        "qblade_azimuth_post_advance_deg",
        "time_increment_tolerance_s",
        "azimuth_tolerance_deg",
    ):
        try:
            value = float(record[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Modal-burst resume-position field {key} is not finite."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"Modal-burst resume-position field {key} is not finite.")
    return dict(record)


def validate_loaded_modal_burst_resume_position(
    qblade: Any,
    modal_burst: Any,
    completed_steps: int,
    timing_witnesses: list[dict[str, object]],
    stored_record: dict[str, object],
) -> dict[str, object]:
    """Reject a loaded QPR whose live time/azimuth is not its checkpoint position."""

    stored = validate_modal_burst_resume_position_record(
        stored_record,
        completed_steps=completed_steps,
        timing_witnesses=timing_witnesses,
    )
    current = modal_burst_resume_position_evidence(
        qblade,
        modal_burst,
        completed_steps,
        timing_witnesses,
    )
    time_tolerance = max(
        float(stored["time_increment_tolerance_s"]),
        float(current["time_increment_tolerance_s"]),
        float(stored["qblade_post_time_grid_audit"]["float32_residual_limit_s"]),
        float(current["qblade_post_time_grid_audit"]["float32_residual_limit_s"]),
    )
    if not math.isclose(
        float(current["qblade_time_post_advance_s"]),
        float(stored["qblade_time_post_advance_s"]),
        rel_tol=0.0,
        abs_tol=time_tolerance,
    ):
        raise RuntimeError(
            "Loaded modal-burst QBlade project time does not match its checkpoint "
            "resume position."
        )
    azimuth_tolerance = max(
        float(stored["azimuth_tolerance_deg"]),
        float(current["azimuth_tolerance_deg"]),
    )
    azimuth_delta = math.remainder(
        float(current["qblade_azimuth_post_advance_deg"])
        - float(stored["qblade_azimuth_post_advance_deg"]),
        360.0,
    )
    if abs(azimuth_delta) > azimuth_tolerance:
        raise RuntimeError(
            "Loaded modal-burst QBlade project azimuth does not match its checkpoint "
            "resume position."
        )
    return current


def write_restart_checkpoint(
    *,
    qblade,
    checkpoint_dir: Path,
    sim_path: Path,
    qblade_root: Path,
    output_dir: Path,
    result_name: str,
    coupling_mode: str,
    requested_steps: int,
    completed_steps: int,
    threads: int,
    use_opencl: bool,
    checkpoint_identity: dict[str, object],
    resume_checkpoint: dict[str, object] | None,
    sil_vorlap: SilVorLapCoupler | None,
    reason: str,
    sil_modal_burst: Any | None = None,
    sil_modal_burst_plan_path: Path | None = None,
    sil_modal_burst_timing_witnesses: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sim_path.stem}_step_{completed_steps:07d}"
    tmp_project = checkpoint_dir / f"{stem}.tmp.qpr"
    project = checkpoint_dir / f"{stem}.qpr"
    if tmp_project.exists():
        tmp_project.unlink()
    qblade.storeProject(str(tmp_project).encode("utf-8"))
    if not tmp_project.is_file() or tmp_project.stat().st_size <= 0:
        tmp_project.unlink(missing_ok=True)
        raise RuntimeError(f"QBlade did not write checkpoint project: {tmp_project}")
    modal_native_evidence: dict[str, object] | None = None
    modal_bridge_runtime_audit: dict[str, object] | None = None
    modal_dependency_audit: dict[str, object] | None = None
    modal_runtime_identity_audit: dict[str, object] | None = None
    modal_resume_position: dict[str, object] | None = None
    if sil_modal_burst is not None:
        try:
            if sil_modal_burst_plan_path is None:
                raise ValueError(
                    "Modal-burst checkpoint publication requires the immutable plan path."
                )
            modal_bridge_runtime_audit = require_modal_native_bridge_runtime_contract(
                sim_path,
                qblade_root,
            )
            modal_dependency_audit = require_modal_qblade_runtime_dependency_contract(
                qblade_root,
                checkpoint_identity.get("qblade_runtime_dependency_fingerprint"),
            )
            modal_native_evidence = require_zero_native_callback_evidence(
                native_callback_evidence(sim_path, qblade_root=qblade_root),
                context=f"modal-burst checkpoint at step {completed_steps}",
                reject_existing_log=True,
            )
            modal_runtime_identity_audit = require_modal_runtime_identity_contract(
                sim_path=sim_path,
                qblade_root=qblade_root,
                plan_path=sil_modal_burst_plan_path,
                expected_checkpoint_identity=checkpoint_identity,
            )
            modal_resume_position = modal_burst_resume_position_evidence(
                qblade,
                sil_modal_burst,
                completed_steps,
                list(sil_modal_burst_timing_witnesses or []),
            )
        except Exception:
            tmp_project.unlink(missing_ok=True)
            raise
    os.replace(tmp_project, project)
    project_identity = file_sha256(project)
    if project_identity is None:
        raise RuntimeError(
            f"Could not hash checkpoint project after publication: {project}"
        )

    metadata: dict[str, object] = {
        "version": 4 if sil_modal_burst is not None else 3,
        "created_unix": time.time(),
        "reason": reason,
        "sim": str(sim_path),
        "qblade_root": str(qblade_root),
        "output_dir": str(output_dir),
        "result_name": result_name,
        "coupling_mode": coupling_mode,
        "requested_steps": int(requested_steps),
        "completed_steps": int(completed_steps),
        "qblade_project": str(project),
        "qblade_project_sha256": project_identity["sha256"],
        "qblade_project_size_bytes": project_identity["size_bytes"],
        "threads": int(threads),
        "use_opencl": bool(use_opencl),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_identity_fingerprint": checkpoint_identity.get("fingerprint"),
        "qblade_resume_mode": "loadProject",
        "qblade_resume_note": (
            "QBlade CE SIL exposes loadProject/storeProject but no lower-level documented "
            "integrator-state restart API in the local bindings; resume is therefore "
            "validated by continuing from this stored project."
        ),
        "previous_checkpoint": (
            None
            if resume_checkpoint is None
            else resume_checkpoint.get("_metadata_path")
        ),
    }
    if sil_modal_burst is not None:
        metadata["sil_modal_burst_state"] = sil_modal_burst.checkpoint_state()
        metadata["sil_modal_burst_timing_witnesses"] = list(
            sil_modal_burst_timing_witnesses or []
        )
        metadata["sil_modal_burst_timing_witnesses_fingerprint"] = (
            canonical_fingerprint(metadata["sil_modal_burst_timing_witnesses"])
        )
        metadata["zero_native_callback_evidence"] = modal_native_evidence
        metadata["zero_native_callback_evidence_fingerprint"] = canonical_fingerprint(
            modal_native_evidence
        )
        metadata["native_bridge_runtime_audit"] = modal_bridge_runtime_audit
        metadata["native_bridge_runtime_audit_fingerprint"] = canonical_fingerprint(
            modal_bridge_runtime_audit
        )
        metadata["qblade_runtime_dependency_audit"] = modal_dependency_audit
        metadata["qblade_runtime_dependency_audit_fingerprint"] = canonical_fingerprint(
            modal_dependency_audit
        )
        metadata["modal_runtime_identity_audit"] = modal_runtime_identity_audit
        metadata["modal_runtime_identity_audit_fingerprint"] = canonical_fingerprint(
            modal_runtime_identity_audit
        )
        metadata["sil_modal_burst_resume_position"] = modal_resume_position
        metadata["sil_modal_burst_resume_position_fingerprint"] = canonical_fingerprint(
            modal_resume_position
        )
    else:
        metadata["sil_vorlap_state"] = (
            None if sil_vorlap is None else sil_vorlap.checkpoint_state()
        )
    metadata_path = checkpoint_dir / f"{stem}.json"
    atomic_write_json(metadata_path, metadata)
    latest = dict(metadata)
    latest["_metadata_path"] = str(metadata_path)
    atomic_write_json(checkpoint_dir / "latest_checkpoint.json", latest)
    prune_restart_checkpoints(checkpoint_dir, keep=2)
    return latest


__all__ = [
    "atomic_write_json",
    "build_reproducibility_metadata",
    "build_checkpoint_identity",
    "build_modal_native_bridge_runtime_identity",
    "canonical_fingerprint",
    "checkpoint_directory",
    "clear_native_bridge_log",
    "ensure_qblade_loader_env",
    "file_sha256",
    "find_bridge_library",
    "find_qblade_library",
    "git_metadata",
    "load_json_file",
    "load_qblade_api",
    "load_resume_checkpoint",
    "load_strict_modal_external_config",
    "load_vorlap_config_for_sim",
    "modal_burst_resume_position_evidence",
    "native_bridge_log_path",
    "native_callback_configuration",
    "native_callback_evidence",
    "numpy_blas_metadata",
    "package_versions",
    "preflight_qblade_wind_input",
    "require_zero_native_callback_evidence",
    "require_modal_native_bridge_runtime_contract",
    "require_modal_qblade_runtime_dependency_contract",
    "prepend_env_path",
    "prune_restart_checkpoints",
    "qblade_case_settings",
    "qblade_runtime_asset_fingerprint",
    "qblade_runtime_asset_manifest",
    "vorlap_python_runtime_manifest",
    "qblade_library_dirs",
    "qblade_runtime_dependency_manifest",
    "require_modal_runtime_identity_contract",
    "selected_environment",
    "validate_loaded_modal_burst_resume_position",
    "validate_modal_burst_resume_position_record",
    "vorlap_package_metadata",
    "write_restart_checkpoint",
]
