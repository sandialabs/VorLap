"""Reviewed continuation gate for the sealed H-VAWT complete-DLC campaign.

The original 102-case campaign intentionally authorized only a reusable
20-case turbulent raw/complete checkpoint.  Its older full-run gate depended
on a uniform-flow stationarity pass that the retained v11 evidence did not
achieve.  This module adds a separate, hash-bound continuation envelope after
the turbulent checkpoint has been reviewed.  It does not modify the sealed
campaign manifest or the scientific execution implementation recorded there.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator, Mapping

from vorlap import hvawt_complete_dlc_campaign as sealed
from vorlap.qblade_run_support import atomic_write_json


AUTHORIZATION_NAME = "complementary_execution_authorization.json"
AUTHORIZATION_SCHEMA = "vorlap-hvawt-complementary-execution-authorization-v1"
AUTHORIZATION_STATUS = "authorized_after_reviewed_turbulent_checkpoint"
CHECKPOINT_REVIEW_SCHEMA = "vorlap-hvawt-turbulent-checkpoint-paper-receipt-v1"
CHECKPOINT_REVIEW_STATUS = "passed_scoped_checkpoint_promotion"
CHECKPOINT_SUMMARY_SCHEMA = "vorlap-hvawt-turbulent-checkpoint-paper-results-v1"
CHECKPOINT_SUMMARY_STATUS = "complete_single_seed_turbulent_raw_complete_screen"
CHECKPOINT_CASE_COUNT = 20
COMPLEMENTARY_CASE_COUNT = 82
FULL_CASE_COUNT = 102
VERIFIED_TERMINAL_ARTIFACT_COUNT = 70
EXPECTED_REVIEW_RECEIPT_SHA256 = (
    "f90e5407fac7277709be6d36418e9a9fc30d9fc4e1b0dcdd7cf94d7976e0f34c"
)
COMPLEMENTARY_CASE_INDICES = tuple(
    index
    for index in range(FULL_CASE_COUNT)
    if index not in sealed.CHECKPOINT_CASE_INDICES
)
DEFAULT_REVIEW_RECEIPT = Path(
    "paper_drafts/NAWEA_QBladeBridge/generated/"
    "hvawt_turbulent_checkpoint_receipt.json"
)
CONTINUATION_SOURCE_PATHS = {
    "continuation_model": Path("run_headless/hvawt_complementary_dlc_support.py"),
    "continuation_cli": Path("run_headless/hvawt_complementary_dlc_campaign.py"),
    "continuation_authorization_submission": Path(
        "run_headless/submit_hvawt_complementary_authorization.sj"
    ),
    "continuation_u50_submission": Path(
        "run_headless/submit_hvawt_complementary_u50_smoke.sj"
    ),
    "continuation_full_submission": Path(
        "run_headless/submit_hvawt_complementary_dlc_flight.sj"
    ),
    "continuation_chain_submission": Path(
        "run_headless/submit_hvawt_complementary_chain.sh"
    ),
    "checkpoint_paper_promoter": Path(
        "paper_drafts/NAWEA_QBladeBridge/"
        "prepare_hvawt_turbulent_checkpoint_results.py"
    ),
}
EXPECTED_REVIEW_ARTIFACTS = {
    "figures/hvawt_turbulent_checkpoint_results.pdf",
    "generated/hvawt_turbulent_checkpoint_macros.tex",
    "generated/hvawt_turbulent_checkpoint_summary.json",
    "generated/hvawt_turbulent_checkpoint_table.tex",
    "plot_data/hvawt_turbulent_checkpoint_comparisons.csv",
    "plot_data/hvawt_turbulent_checkpoint_convergence.csv",
    "plot_data/hvawt_turbulent_checkpoint_load_windows.csv",
    "plot_data/hvawt_turbulent_checkpoint_rainflow_validation.csv",
    "plot_data/hvawt_turbulent_checkpoint_response.csv",
    "plot_data/hvawt_turbulent_checkpoint_response_windows.csv",
    "plot_data/hvawt_turbulent_checkpoint_signed_channels.csv",
    "plot_data/hvawt_turbulent_historical_del_context.csv",
}


class HVAWTComplementaryDLCError(RuntimeError):
    """Raised when the reviewed continuation contract is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise HVAWTComplementaryDLCError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HVAWTComplementaryDLCError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise HVAWTComplementaryDLCError(f"{label} must be a JSON object: {path}")
    return value


def _validate_fingerprint(payload: Mapping[str, Any], *, label: str) -> None:
    recorded = payload.get("fingerprint")
    content = dict(payload)
    content.pop("fingerprint", None)
    if recorded != _canonical_fingerprint(content):
        raise HVAWTComplementaryDLCError(f"{label} fingerprint is invalid.")


def _identity(path: Path, *, base: Path) -> dict[str, object]:
    resolved = path.resolve()
    base_resolved = base.resolve()
    try:
        relative = resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise HVAWTComplementaryDLCError(
            f"Authorization source lies outside its declared root: {resolved}"
        ) from exc
    return {
        "path": relative.as_posix(),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_identity(
    base: Path, record: object, *, label: str, expected_name: str | None = None
) -> Path:
    if not isinstance(record, Mapping):
        raise HVAWTComplementaryDLCError(f"Missing identity for {label}.")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise HVAWTComplementaryDLCError(f"Invalid path identity for {label}.")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise HVAWTComplementaryDLCError(f"{label} identity must be relative.")
    resolved_base = base.resolve()
    resolved = (resolved_base / relative).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise HVAWTComplementaryDLCError(
            f"{label} identity escapes its declared root."
        ) from exc
    if expected_name is not None and resolved.name != expected_name:
        raise HVAWTComplementaryDLCError(
            f"{label} identity does not name {expected_name}."
        )
    if not resolved.is_file():
        raise HVAWTComplementaryDLCError(f"Missing {label}: {resolved}")
    if int(record.get("size_bytes", -1)) != resolved.stat().st_size:
        raise HVAWTComplementaryDLCError(f"Byte count changed for {label}.")
    if str(record.get("sha256", "")) != _sha256(resolved):
        raise HVAWTComplementaryDLCError(f"SHA-256 changed for {label}.")
    return resolved


def _verify_checkpoint_review(
    repository: Path, campaign_root: Path, evidence_path: Path
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    review = _load_json(evidence_path, label="turbulent checkpoint paper receipt")
    _validate_fingerprint(review, label="Turbulent checkpoint paper receipt")
    if (
        review.get("schema") != CHECKPOINT_REVIEW_SCHEMA
        or review.get("status") != CHECKPOINT_REVIEW_STATUS
        or review.get("verified_terminal_artifact_count")
        != VERIFIED_TERMINAL_ARTIFACT_COUNT
    ):
        raise HVAWTComplementaryDLCError(
            "Turbulent checkpoint paper receipt is not the reviewed promotion."
        )

    analysis_path = _verify_identity(
        campaign_root,
        review.get("analysis"),
        label="checkpoint analysis",
        expected_name="turbulent_20case_analysis.json",
    )
    manifest_path = _verify_identity(
        campaign_root,
        review.get("campaign"),
        label="campaign manifest",
        expected_name="campaign_manifest.json",
    )
    checkpoint_authorization_path = _verify_identity(
        campaign_root,
        review.get("authorization"),
        label="checkpoint execution authorization",
        expected_name=sealed.CHECKPOINT_AUTHORIZATION_NAME,
    )
    manifest = _load_json(manifest_path, label="campaign manifest")
    analysis = _load_json(analysis_path, label="checkpoint analysis")
    checkpoint_authorization = _load_json(
        checkpoint_authorization_path, label="checkpoint execution authorization"
    )
    _validate_fingerprint(analysis, label="Checkpoint analysis")
    _validate_fingerprint(
        checkpoint_authorization, label="Checkpoint execution authorization"
    )
    if (
        analysis.get("schema") != sealed.CHECKPOINT_ANALYSIS_SCHEMA
        or analysis.get("status")
        != "complete_turbulent_raw_complete_execution_checkpoint"
        or analysis.get("campaign_fingerprint") != manifest.get("fingerprint")
        or analysis.get("complete_checkpoint_case_count") != CHECKPOINT_CASE_COUNT
        or analysis.get("expected_checkpoint_case_count") != CHECKPOINT_CASE_COUNT
        or analysis.get("all_checkpoint_cases_complete") is not True
        or analysis.get("all_pairs_complete") is not True
        or tuple(analysis.get("case_indices", ())) != sealed.CHECKPOINT_CASE_INDICES
        or checkpoint_authorization.get("fingerprint")
        != analysis.get("authorization_fingerprint")
        or checkpoint_authorization.get("schema")
        != sealed.CHECKPOINT_AUTHORIZATION_SCHEMA
        or checkpoint_authorization.get("status")
        != "authorized_for_turbulent_raw_complete_checkpoint_attempt"
        or checkpoint_authorization.get("campaign_fingerprint")
        != manifest.get("fingerprint")
        or checkpoint_authorization.get("authorized_case_count")
        != CHECKPOINT_CASE_COUNT
        or tuple(checkpoint_authorization.get("authorized_case_indices", ()))
        != sealed.CHECKPOINT_CASE_INDICES
        or tuple(checkpoint_authorization.get("scenarios", ()))
        != sealed.CHECKPOINT_SCENARIOS
    ):
        raise HVAWTComplementaryDLCError(
            "Turbulent checkpoint analysis is not the exact complete 20-case result."
        )

    paper_root = evidence_path.resolve().parents[1]
    artifacts = review.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != EXPECTED_REVIEW_ARTIFACTS:
        raise HVAWTComplementaryDLCError(
            "Turbulent checkpoint paper receipt artifact identities are incomplete."
        )
    verified_artifacts: dict[str, Path] = {}
    for role, record in artifacts.items():
        if not isinstance(role, str):
            raise HVAWTComplementaryDLCError("Invalid paper artifact role.")
        verified_artifacts[role] = _verify_identity(
            paper_root, record, label=f"checkpoint paper artifact {role}"
        )
    summary_role = "generated/hvawt_turbulent_checkpoint_summary.json"
    if summary_role not in verified_artifacts:
        raise HVAWTComplementaryDLCError("Checkpoint review summary is not bound.")
    summary_path = verified_artifacts[summary_role]
    summary = _load_json(summary_path, label="turbulent checkpoint paper summary")
    _validate_fingerprint(summary, label="Turbulent checkpoint paper summary")
    if (
        summary.get("schema") != CHECKPOINT_SUMMARY_SCHEMA
        or summary.get("status") != CHECKPOINT_SUMMARY_STATUS
        or summary.get("case_count") != CHECKPOINT_CASE_COUNT
        or summary.get("pair_count") != CHECKPOINT_CASE_COUNT // 2
        or summary.get("campaign_fingerprint") != manifest.get("fingerprint")
        or summary.get("analysis_fingerprint") != analysis.get("fingerprint")
        or summary.get("authorization_fingerprint")
        != checkpoint_authorization.get("fingerprint")
        or review.get("summary_fingerprint") != summary.get("fingerprint")
    ):
        raise HVAWTComplementaryDLCError(
            "Turbulent checkpoint paper summary does not match the sealed campaign."
        )
    return review, summary_path, summary


def _continuation_source_identities(repository: Path) -> dict[str, dict[str, object]]:
    identities: dict[str, dict[str, object]] = {}
    for role, relative in CONTINUATION_SOURCE_PATHS.items():
        source = repository / relative
        if not source.is_file():
            raise HVAWTComplementaryDLCError(
                f"Missing continuation source {role}: {source}"
            )
        identities[role] = _identity(source, base=repository)
    return identities


def verify_sealed_orchestration_sources(
    campaign: Path,
    *,
    repository: Path | None = None,
) -> dict[str, object]:
    """Fail before submission if the checkout differs from campaign generation."""

    repo = (
        Path(__file__).resolve().parents[1]
        if repository is None
        else Path(repository).expanduser().resolve()
    )
    manifest_path = sealed._manifest_path(Path(campaign))
    manifest = _load_json(manifest_path, label="campaign manifest")
    inputs = manifest.get("inputs")
    orchestration = (
        inputs.get("orchestration_sources") if isinstance(inputs, Mapping) else None
    )
    if not isinstance(orchestration, Mapping):
        raise HVAWTComplementaryDLCError(
            "Sealed campaign orchestration identities are missing."
        )

    verified: dict[str, dict[str, object]] = {}
    for role, relative in sealed.ORCHESTRATION_SOURCE_RELATIVE.items():
        recorded = orchestration.get(role)
        source = repo / relative
        if not source.is_file():
            raise HVAWTComplementaryDLCError(
                f"Missing sealed orchestration source {role}: {source}"
            )
        observed = sealed._identity(source)
        if not isinstance(recorded, Mapping) or recorded != observed:
            raise HVAWTComplementaryDLCError(
                f"Sealed orchestration source {role!r} differs from campaign "
                "generation; refuse to submit the continuation chain."
            )
        verified[role] = dict(observed)

    return {
        "status": "sealed_orchestration_sources_match_campaign_generation",
        "campaign_fingerprint": manifest.get("fingerprint"),
        "verified_source_count": len(verified),
        "orchestration_sources": verified,
    }


def _require_clean_tracked_authorization_sources(repository: Path) -> None:
    """Require authorization inputs to equal their checked-out HEAD blobs."""

    relative_paths = [
        DEFAULT_REVIEW_RECEIPT.as_posix(),
        *(path.as_posix() for path in CONTINUATION_SOURCE_PATHS.values()),
    ]
    commands = (
        [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--error-unmatch",
            "--",
            *relative_paths,
        ],
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--quiet",
            "--no-ext-diff",
            "HEAD",
            "--",
            *relative_paths,
        ],
    )
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            raise HVAWTComplementaryDLCError(
                "Git is required to verify tracked continuation sources."
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.strip()
            raise HVAWTComplementaryDLCError(
                "Checkpoint receipt and continuation sources must be tracked, "
                "committed, and byte-identical to HEAD."
                + (f" Git reported: {detail}" if detail else "")
            )


def _verify_checkpoint_case_identity(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    case_index: int,
) -> dict[str, object]:
    """Compare one live terminal case with its immutable analysis receipt."""

    cases = state["manifest"].get("cases")
    if not isinstance(cases, list) or case_index < 0 or case_index >= len(cases):
        raise IndexError("Complete DLC case index is out of range.")
    if case_index not in sealed.CHECKPOINT_CASE_INDICES:
        raise HVAWTComplementaryDLCError(
            f"Case {case_index} is not a retained checkpoint case."
        )
    records = analysis.get("cases")
    matches = (
        [
            row
            for row in records
            if isinstance(row, Mapping) and row.get("case_index") == case_index
        ]
        if isinstance(records, list)
        else []
    )
    if len(matches) != 1:
        raise HVAWTComplementaryDLCError(
            f"Checkpoint case {case_index} lacks one exact snapshotted receipt."
        )
    expected = matches[0]
    case = cases[case_index]
    expected_fields = {
        "case_index": case_index,
        "case_id": case.get("case_id"),
        "row_id": case.get("row_id"),
        "scenario": case.get("scenario"),
        "wind_speed_mps": case.get("wind_speed_mps"),
        "seed": case.get("seed"),
    }
    if any(expected.get(name) != value for name, value in expected_fields.items()):
        raise HVAWTComplementaryDLCError(
            f"Checkpoint case {case_index} metadata differs from the sealed manifest."
        )

    classification = sealed._classify_case_state(state, case_index=case_index)
    if classification.get("status") != "complete":
        raise HVAWTComplementaryDLCError(
            f"Checkpoint case {case_index} is no longer complete; refuse to rerun it."
        )
    if (
        classification.get("case_id") != expected.get("case_id")
        or classification.get("receipt") != expected.get("receipt")
    ):
        raise HVAWTComplementaryDLCError(
            f"Checkpoint case {case_index} terminal identity changed; refuse to rerun it."
        )
    return classification


def _verify_live_checkpoint_cases(
    root: Path,
    manifest: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> list[dict[str, object]]:
    state = {"campaign_root": root, "manifest": manifest}
    return [
        _verify_checkpoint_case_identity(state, analysis, case_index=case_index)
        for case_index in sealed.CHECKPOINT_CASE_INDICES
    ]


def bind_complementary_authorization(
    campaign: Path,
    *,
    evidence_path: Path | None = None,
    repository: Path | None = None,
) -> dict[str, object]:
    """Bind the reviewed checkpoint before any of the remaining 82 cases run."""

    repo = (
        Path(__file__).resolve().parents[1]
        if repository is None
        else Path(repository).expanduser().resolve()
    )
    state = sealed.validate_campaign(
        campaign,
        rehash_external_winds=False,
        rehash_case_assets=True,
        deep_case_validation=True,
    )
    root = state["campaign_root"]
    manifest = state["manifest"]
    sealed._load_runtime(root)
    sealed._load_checkpoint_authorization(root, manifest)
    legacy = root / sealed.EXECUTION_AUTHORIZATION_NAME
    if legacy.exists():
        raise HVAWTComplementaryDLCError(
            "Legacy full-run authorization exists; refuse competing authorization paths."
        )
    expected_review_path = (repo / DEFAULT_REVIEW_RECEIPT).resolve()
    review_path = (
        repo / DEFAULT_REVIEW_RECEIPT
        if evidence_path is None
        else Path(evidence_path).expanduser().resolve()
    ).resolve()
    if review_path != expected_review_path:
        raise HVAWTComplementaryDLCError(
            "Checkpoint review must be the tracked repository receipt: "
            f"{expected_review_path}"
        )
    _require_clean_tracked_authorization_sources(repo)
    if _sha256(review_path) != EXPECTED_REVIEW_RECEIPT_SHA256:
        raise HVAWTComplementaryDLCError(
            "Tracked checkpoint review receipt differs from the reviewed SHA-256."
        )
    review, summary_path, summary = _verify_checkpoint_review(
        repo, root, review_path
    )
    checkpoint_analysis = _load_json(
        root / "turbulent_20case_analysis.json", label="checkpoint analysis"
    )
    _verify_live_checkpoint_cases(root, manifest, checkpoint_analysis)
    continuation_sources = _continuation_source_identities(repo)

    receipt_path = root / AUTHORIZATION_NAME
    snapshot_root = root / "complementary_execution_authorization"
    lock_path = root / ".complementary_execution_authorization.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if receipt_path.exists():
            recorded = _load_complementary_authorization(root, manifest, repository=repo)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return recorded
        if snapshot_root.exists():
            raise HVAWTComplementaryDLCError(
                "Complementary authorization snapshots exist without their receipt."
            )
        snapshot_root.mkdir()
        snapshot_sources = {
            "checkpoint_analysis": root / "turbulent_20case_analysis.json",
            "checkpoint_execution_authorization": root
            / sealed.CHECKPOINT_AUTHORIZATION_NAME,
            "checkpoint_review_receipt": review_path,
            "checkpoint_review_summary": summary_path,
        }
        snapshots: dict[str, dict[str, object]] = {}
        for role, source in snapshot_sources.items():
            destination = snapshot_root / source.name
            shutil.copy2(source, destination)
            snapshots[role] = _identity(destination, base=root)
        receipt: dict[str, object] = {
            "schema": AUTHORIZATION_SCHEMA,
            "status": AUTHORIZATION_STATUS,
            "campaign_fingerprint": manifest["fingerprint"],
            "checkpoint_analysis_fingerprint": checkpoint_analysis["fingerprint"],
            "checkpoint_review_fingerprint": review["fingerprint"],
            "checkpoint_summary_fingerprint": summary["fingerprint"],
            "checkpoint_case_count": CHECKPOINT_CASE_COUNT,
            "complementary_case_count": COMPLEMENTARY_CASE_COUNT,
            "full_case_count": FULL_CASE_COUNT,
            "reused_case_indices": list(sealed.CHECKPOINT_CASE_INDICES),
            "authorized_case_indices": list(COMPLEMENTARY_CASE_INDICES),
            "scenarios": list(sealed.SCENARIOS),
            "snapshots": snapshots,
            "continuation_sources": continuation_sources,
            "claim_boundary": (
                "Execution authorization for the 82 cases complementary to the "
                "reviewed turbulent raw/complete checkpoint. It is not scientific "
                "acceptance of the future results, physical validation, fatigue "
                "life, design certification, or permission to alter sealed physics."
            ),
        }
        receipt["fingerprint"] = _canonical_fingerprint(receipt)
        atomic_write_json(receipt_path, receipt)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return receipt


def _load_complementary_authorization(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    repository: Path | None = None,
) -> dict[str, Any]:
    repo = (
        Path(__file__).resolve().parents[1]
        if repository is None
        else Path(repository).expanduser().resolve()
    )
    if (root / sealed.EXECUTION_AUTHORIZATION_NAME).exists():
        raise HVAWTComplementaryDLCError(
            "Legacy full-run authorization exists beside the complementary receipt."
        )
    receipt = _load_json(
        root / AUTHORIZATION_NAME, label="complementary execution authorization"
    )
    _validate_fingerprint(receipt, label="Complementary execution authorization")
    if (
        receipt.get("schema") != AUTHORIZATION_SCHEMA
        or receipt.get("status") != AUTHORIZATION_STATUS
        or receipt.get("campaign_fingerprint") != manifest.get("fingerprint")
        or receipt.get("checkpoint_case_count") != CHECKPOINT_CASE_COUNT
        or receipt.get("complementary_case_count") != COMPLEMENTARY_CASE_COUNT
        or receipt.get("full_case_count") != FULL_CASE_COUNT
        or tuple(receipt.get("reused_case_indices", ()))
        != sealed.CHECKPOINT_CASE_INDICES
        or tuple(receipt.get("authorized_case_indices", ()))
        != COMPLEMENTARY_CASE_INDICES
        or tuple(receipt.get("scenarios", ())) != sealed.SCENARIOS
    ):
        raise HVAWTComplementaryDLCError(
            "Complementary execution authorization receipt is invalid."
        )
    snapshots = receipt.get("snapshots")
    expected_snapshots = {
        "checkpoint_analysis",
        "checkpoint_execution_authorization",
        "checkpoint_review_receipt",
        "checkpoint_review_summary",
    }
    if not isinstance(snapshots, Mapping) or set(snapshots) != expected_snapshots:
        raise HVAWTComplementaryDLCError(
            "Complementary authorization snapshots are incomplete."
        )
    for role, record in snapshots.items():
        _verify_identity(root, record, label=f"complementary snapshot {role}")
    snapshot_base = root / "complementary_execution_authorization"
    analysis = _load_json(
        snapshot_base / "turbulent_20case_analysis.json",
        label="snapshotted checkpoint analysis",
    )
    checkpoint_authorization = _load_json(
        snapshot_base / sealed.CHECKPOINT_AUTHORIZATION_NAME,
        label="snapshotted checkpoint execution authorization",
    )
    review = _load_json(
        snapshot_base / "hvawt_turbulent_checkpoint_receipt.json",
        label="snapshotted checkpoint review receipt",
    )
    summary = _load_json(
        snapshot_base / "hvawt_turbulent_checkpoint_summary.json",
        label="snapshotted checkpoint review summary",
    )
    for payload, label in (
        (analysis, "Snapshotted checkpoint analysis"),
        (checkpoint_authorization, "Snapshotted checkpoint authorization"),
        (review, "Snapshotted checkpoint review receipt"),
        (summary, "Snapshotted checkpoint review summary"),
    ):
        _validate_fingerprint(payload, label=label)
    if (
        analysis.get("schema") != sealed.CHECKPOINT_ANALYSIS_SCHEMA
        or analysis.get("status")
        != "complete_turbulent_raw_complete_execution_checkpoint"
        or analysis.get("campaign_fingerprint") != manifest.get("fingerprint")
        or analysis.get("complete_checkpoint_case_count") != CHECKPOINT_CASE_COUNT
        or analysis.get("all_checkpoint_cases_complete") is not True
        or analysis.get("all_pairs_complete") is not True
        or analysis.get("fingerprint")
        != receipt.get("checkpoint_analysis_fingerprint")
        or checkpoint_authorization.get("fingerprint")
        != analysis.get("authorization_fingerprint")
        or checkpoint_authorization.get("schema")
        != sealed.CHECKPOINT_AUTHORIZATION_SCHEMA
        or checkpoint_authorization.get("status")
        != "authorized_for_turbulent_raw_complete_checkpoint_attempt"
        or checkpoint_authorization.get("campaign_fingerprint")
        != manifest.get("fingerprint")
        or checkpoint_authorization.get("authorized_case_count")
        != CHECKPOINT_CASE_COUNT
        or tuple(checkpoint_authorization.get("authorized_case_indices", ()))
        != sealed.CHECKPOINT_CASE_INDICES
        or tuple(checkpoint_authorization.get("scenarios", ()))
        != sealed.CHECKPOINT_SCENARIOS
        or review.get("schema") != CHECKPOINT_REVIEW_SCHEMA
        or review.get("status") != CHECKPOINT_REVIEW_STATUS
        or review.get("fingerprint")
        != receipt.get("checkpoint_review_fingerprint")
        or review.get("analysis", {}).get("sha256")
        != snapshots["checkpoint_analysis"].get("sha256")
        or review.get("authorization", {}).get("sha256")
        != snapshots["checkpoint_execution_authorization"].get("sha256")
        or summary.get("schema") != CHECKPOINT_SUMMARY_SCHEMA
        or summary.get("status") != CHECKPOINT_SUMMARY_STATUS
        or summary.get("fingerprint")
        != receipt.get("checkpoint_summary_fingerprint")
        or review.get("summary_fingerprint") != summary.get("fingerprint")
        or summary.get("campaign_fingerprint") != manifest.get("fingerprint")
        or summary.get("analysis_fingerprint") != analysis.get("fingerprint")
        or summary.get("authorization_fingerprint")
        != checkpoint_authorization.get("fingerprint")
    ):
        raise HVAWTComplementaryDLCError(
            "Complementary authorization snapshots are not the reviewed checkpoint."
        )
    sources = receipt.get("continuation_sources")
    if not isinstance(sources, Mapping) or set(sources) != set(
        CONTINUATION_SOURCE_PATHS
    ):
        raise HVAWTComplementaryDLCError(
            "Complementary continuation-source identities are incomplete."
        )
    for role, relative in CONTINUATION_SOURCE_PATHS.items():
        source = _verify_identity(
            repo, sources[role], label=f"continuation source {role}"
        )
        if source != (repo / relative).resolve():
            raise HVAWTComplementaryDLCError(
                f"Continuation source {role} resolves to an unexpected path."
            )
    return receipt


@contextmanager
def _reviewed_authorization_loader() -> Iterator[None]:
    original = sealed._load_execution_authorization

    def load_reviewed(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
        return _load_complementary_authorization(root, manifest)

    sealed._load_execution_authorization = load_reviewed
    try:
        yield
    finally:
        sealed._load_execution_authorization = original


def execute_u50_smoke(*args: Any, **kwargs: Any) -> dict[str, object]:
    """Run the sealed four-step U50 smoke under the reviewed continuation."""

    with _reviewed_authorization_loader():
        return sealed.execute_u50_smoke(*args, **kwargs)


def _validate_checkpoint_case_reuse(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
) -> dict[str, object]:
    """Verify and reuse an accepted checkpoint case without mutating it."""

    state = sealed.validate_campaign(
        campaign,
        rehash_external_winds=False,
        rehash_case_assets=False,
        deep_case_validation=False,
    )
    root = state["campaign_root"]
    manifest = state["manifest"]

    _load_complementary_authorization(root, manifest)
    sealed._load_u50_smoke_receipt(root, manifest)
    runtime = sealed._load_runtime(root)
    if runtime != sealed.build_runtime_identity(qblade_root):
        raise HVAWTComplementaryDLCError(
            "Current runtime differs from the bound Flight runtime."
        )

    analysis = _load_json(
        root
        / "complementary_execution_authorization"
        / "turbulent_20case_analysis.json",
        label="snapshotted checkpoint analysis",
    )
    return _verify_checkpoint_case_identity(state, analysis, case_index=case_index)


def execute_case(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
    python_executable: str,
    threads: int,
    checkpoint_every: int = sealed.FLIGHT_RESTART_SAVE_INTERVAL_STEPS,
    progress_every: int = 1000,
) -> dict[str, object]:
    """Run a complementary case or read-only verify an accepted checkpoint case."""

    if case_index in sealed.CHECKPOINT_CASE_INDICES:
        return _validate_checkpoint_case_reuse(
            campaign,
            case_index=case_index,
            qblade_root=qblade_root,
        )
    with _reviewed_authorization_loader():
        return sealed.execute_case(
            campaign,
            case_index=case_index,
            qblade_root=qblade_root,
            python_executable=python_executable,
            threads=threads,
            checkpoint_every=checkpoint_every,
            progress_every=progress_every,
            authorization_scope="full",
        )


__all__ = [
    "AUTHORIZATION_NAME",
    "AUTHORIZATION_SCHEMA",
    "HVAWTComplementaryDLCError",
    "bind_complementary_authorization",
    "execute_case",
    "execute_u50_smoke",
    "verify_sealed_orchestration_sources",
]
