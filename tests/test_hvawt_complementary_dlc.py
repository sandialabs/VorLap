import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from run_headless import hvawt_complementary_dlc_support as continuation


AUTHORIZATION_BOUND_PROMOTER = Path(
    "paper_drafts/NAWEA_QBladeBridge/prepare_hvawt_turbulent_checkpoint_results.py"
)
AUTHORIZATION_BOUND_PROMOTER_BYTES = 67733
AUTHORIZATION_BOUND_PROMOTER_SHA256 = (
    "8fe417416524b1ca7a67fe78b6f94abbfc15015e9f9b8ab36ec5889e02e694ae"
)


def _fingerprinted(payload):
    result = dict(payload)
    result["fingerprint"] = continuation._canonical_fingerprint(result)
    return result


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.paper
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / AUTHORIZATION_BOUND_PROMOTER).is_file(),
    reason="Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
)
def test_checkpoint_promoter_retains_complementary_authorization_identity():
    repository = Path(__file__).resolve().parents[1]
    relative = continuation.CONTINUATION_SOURCE_PATHS["checkpoint_paper_promoter"]
    assert relative == AUTHORIZATION_BOUND_PROMOTER

    payload = (repository / relative).read_bytes()
    assert len(payload) == AUTHORIZATION_BOUND_PROMOTER_BYTES
    assert hashlib.sha256(payload).hexdigest() == AUTHORIZATION_BOUND_PROMOTER_SHA256

    versioned_promoter = Path(
        "paper_drafts/NAWEA_QBladeBridge/analysis/"
        "prepare_hvawt_turbulent_checkpoint_results_v2.py"
    )
    assert (repository / versioned_promoter).is_file()


def _reviewed_checkpoint(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    root = tmp_path / "campaign"
    paper_root = repository / "paper_drafts/NAWEA_QBladeBridge"
    root.mkdir(parents=True)
    paper_root.mkdir(parents=True)

    source_paths = {
        "continuation_model": Path("vorlap/continuation.py"),
        "continuation_cli": Path("run_headless/continuation.py"),
        "continuation_authorization_submission": Path("run_headless/authorize.sj"),
        "continuation_u50_submission": Path("run_headless/u50.sj"),
        "continuation_full_submission": Path("run_headless/full.sj"),
        "continuation_chain_submission": Path("run_headless/submit_chain.sh"),
        "checkpoint_paper_promoter": Path("paper/promoter.py"),
    }
    monkeypatch.setattr(continuation, "CONTINUATION_SOURCE_PATHS", source_paths)
    for index, relative in enumerate(source_paths.values()):
        source = repository / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"source-{index}\n", encoding="utf-8")

    manifest = {"fingerprint": "a" * 64}
    manifest_path = root / "campaign_manifest.json"
    _write_json(manifest_path, manifest)
    checkpoint_authorization = _fingerprinted(
        {
            "schema": continuation.sealed.CHECKPOINT_AUTHORIZATION_SCHEMA,
            "status": "authorized_for_turbulent_raw_complete_checkpoint_attempt",
            "campaign_fingerprint": manifest["fingerprint"],
            "authorized_case_count": continuation.CHECKPOINT_CASE_COUNT,
            "authorized_case_indices": list(
                continuation.sealed.CHECKPOINT_CASE_INDICES
            ),
            "scenarios": list(continuation.sealed.CHECKPOINT_SCENARIOS),
        }
    )
    authorization_path = root / continuation.sealed.CHECKPOINT_AUTHORIZATION_NAME
    _write_json(authorization_path, checkpoint_authorization)
    analysis = _fingerprinted(
        {
            "schema": continuation.sealed.CHECKPOINT_ANALYSIS_SCHEMA,
            "status": "complete_turbulent_raw_complete_execution_checkpoint",
            "campaign_fingerprint": manifest["fingerprint"],
            "authorization_fingerprint": checkpoint_authorization["fingerprint"],
            "complete_checkpoint_case_count": continuation.CHECKPOINT_CASE_COUNT,
            "expected_checkpoint_case_count": continuation.CHECKPOINT_CASE_COUNT,
            "all_checkpoint_cases_complete": True,
            "all_pairs_complete": True,
            "case_indices": list(continuation.sealed.CHECKPOINT_CASE_INDICES),
        }
    )
    analysis_path = root / "turbulent_20case_analysis.json"
    _write_json(analysis_path, analysis)

    summary = _fingerprinted(
        {
            "schema": continuation.CHECKPOINT_SUMMARY_SCHEMA,
            "status": continuation.CHECKPOINT_SUMMARY_STATUS,
            "case_count": continuation.CHECKPOINT_CASE_COUNT,
            "pair_count": continuation.CHECKPOINT_CASE_COUNT // 2,
            "campaign_fingerprint": manifest["fingerprint"],
            "analysis_fingerprint": analysis["fingerprint"],
            "authorization_fingerprint": checkpoint_authorization["fingerprint"],
        }
    )
    artifacts = {}
    for role in continuation.EXPECTED_REVIEW_ARTIFACTS:
        path = paper_root / role
        path.parent.mkdir(parents=True, exist_ok=True)
        if role == "generated/hvawt_turbulent_checkpoint_summary.json":
            _write_json(path, summary)
        else:
            path.write_text(f"artifact:{role}\n", encoding="utf-8")
        artifacts[role] = continuation._identity(path, base=paper_root)
    review = _fingerprinted(
        {
            "schema": continuation.CHECKPOINT_REVIEW_SCHEMA,
            "status": continuation.CHECKPOINT_REVIEW_STATUS,
            "verified_terminal_artifact_count": (
                continuation.VERIFIED_TERMINAL_ARTIFACT_COUNT
            ),
            "analysis": continuation._identity(analysis_path, base=root),
            "campaign": continuation._identity(manifest_path, base=root),
            "authorization": continuation._identity(authorization_path, base=root),
            "artifacts": artifacts,
            "summary_fingerprint": summary["fingerprint"],
        }
    )
    review_path = paper_root / "generated/hvawt_turbulent_checkpoint_receipt.json"
    _write_json(review_path, review)
    monkeypatch.setattr(
        continuation,
        "EXPECTED_REVIEW_RECEIPT_SHA256",
        continuation._sha256(review_path),
    )

    monkeypatch.setattr(
        continuation.sealed,
        "validate_campaign",
        lambda *_args, **_kwargs: {"campaign_root": root, "manifest": manifest},
    )
    monkeypatch.setattr(continuation.sealed, "_load_runtime", lambda *_args: {})
    monkeypatch.setattr(
        continuation.sealed,
        "_load_checkpoint_authorization",
        lambda *_args: checkpoint_authorization,
    )
    monkeypatch.setattr(
        continuation,
        "_verify_live_checkpoint_cases",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        continuation,
        "_require_clean_tracked_authorization_sources",
        lambda *_args: None,
    )
    return repository, root, manifest, review_path


def test_reviewed_checkpoint_binds_exact_complementary_authorization(
    tmp_path, monkeypatch
):
    repository, root, manifest, review_path = _reviewed_checkpoint(
        tmp_path, monkeypatch
    )

    receipt = continuation.bind_complementary_authorization(
        root, evidence_path=review_path, repository=repository
    )

    assert receipt["status"] == continuation.AUTHORIZATION_STATUS
    assert receipt["checkpoint_case_count"] == 20
    assert receipt["complementary_case_count"] == 82
    assert receipt["full_case_count"] == 102
    assert tuple(receipt["reused_case_indices"]) == (
        continuation.sealed.CHECKPOINT_CASE_INDICES
    )
    assert tuple(receipt["authorized_case_indices"]) == (
        continuation.COMPLEMENTARY_CASE_INDICES
    )
    assert continuation._load_complementary_authorization(
        root, manifest, repository=repository
    ) == receipt


def test_complementary_authorization_rejects_changed_launcher(tmp_path, monkeypatch):
    repository, root, manifest, review_path = _reviewed_checkpoint(
        tmp_path, monkeypatch
    )
    continuation.bind_complementary_authorization(
        root, evidence_path=review_path, repository=repository
    )
    changed = repository / continuation.CONTINUATION_SOURCE_PATHS[
        "continuation_full_submission"
    ]
    changed.write_text("changed\n", encoding="utf-8")

    with pytest.raises(
        continuation.HVAWTComplementaryDLCError,
        match="Byte count|SHA-256",
    ):
        continuation._load_complementary_authorization(
            root, manifest, repository=repository
        )


def test_complementary_authorization_refuses_legacy_full_receipt(
    tmp_path, monkeypatch
):
    repository, root, _manifest, review_path = _reviewed_checkpoint(
        tmp_path, monkeypatch
    )
    (root / continuation.sealed.EXECUTION_AUTHORIZATION_NAME).write_text(
        "{}\n", encoding="utf-8"
    )

    with pytest.raises(
        continuation.HVAWTComplementaryDLCError, match="Legacy full-run"
    ):
        continuation.bind_complementary_authorization(
            root, evidence_path=review_path, repository=repository
        )


def test_complementary_authorization_requires_tracked_review_path(
    tmp_path, monkeypatch
):
    repository, root, _manifest, _review_path = _reviewed_checkpoint(
        tmp_path, monkeypatch
    )

    with pytest.raises(
        continuation.HVAWTComplementaryDLCError,
        match="tracked repository receipt",
    ):
        continuation.bind_complementary_authorization(
            root,
            evidence_path=root / "untracked_review.json",
            repository=repository,
        )


def test_complementary_authorization_requires_reviewed_receipt_hash(
    tmp_path, monkeypatch
):
    repository, root, _manifest, review_path = _reviewed_checkpoint(
        tmp_path, monkeypatch
    )
    review_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        continuation.HVAWTComplementaryDLCError,
        match="reviewed SHA-256",
    ):
        continuation.bind_complementary_authorization(
            root,
            evidence_path=review_path,
            repository=repository,
        )


def test_authorization_sources_must_match_git_head(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    review = Path("paper/review.json")
    launcher = Path("run/launcher.sh")
    for relative in (review, launcher):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"tracked:{relative}\n", encoding="utf-8")
    monkeypatch.setattr(continuation, "DEFAULT_REVIEW_RECEIPT", review)
    monkeypatch.setattr(
        continuation,
        "CONTINUATION_SOURCE_PATHS",
        {"launcher": launcher},
    )
    subprocess.run(["git", "init", "-q", repository], check=True)
    subprocess.run(["git", "-C", repository, "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    continuation._require_clean_tracked_authorization_sources(repository)
    (repository / launcher).write_text("changed\n", encoding="utf-8")

    with pytest.raises(
        continuation.HVAWTComplementaryDLCError,
        match="byte-identical to HEAD",
    ):
        continuation._require_clean_tracked_authorization_sources(repository)


def test_submission_preflight_rejects_sealed_orchestration_drift(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    campaign_root = tmp_path / "campaign"
    source_relative = Path("run_headless/sealed_full_submission.sj")
    source = repository / source_relative
    source.parent.mkdir(parents=True)
    source.write_text("#SBATCH --array=0-101%34\n", encoding="utf-8")
    manifest = {
        "fingerprint": "a" * 64,
        "inputs": {
            "orchestration_sources": {
                "full_submission": continuation.sealed._identity(source),
            }
        },
    }
    _write_json(campaign_root / "campaign_manifest.json", manifest)
    monkeypatch.setattr(
        continuation.sealed,
        "ORCHESTRATION_SOURCE_RELATIVE",
        {"full_submission": source_relative},
    )

    receipt = continuation.verify_sealed_orchestration_sources(
        campaign_root,
        repository=repository,
    )
    assert receipt["status"] == (
        "sealed_orchestration_sources_match_campaign_generation"
    )
    assert receipt["verified_source_count"] == 1

    source.write_text("#SBATCH --array=0-101\n", encoding="utf-8")
    with pytest.raises(
        continuation.HVAWTComplementaryDLCError,
        match="differs from campaign generation; refuse to submit",
    ):
        continuation.verify_sealed_orchestration_sources(
            campaign_root,
            repository=repository,
        )


def test_checkpoint_identity_is_exact_and_read_only(monkeypatch, tmp_path):
    expected_receipt = {"result": {"sha256": "b" * 64}}
    case = {
        "case_id": "checkpoint-case",
        "row_id": "checkpoint-row",
        "scenario": "raw_qblade",
        "wind_speed_mps": 3.0,
        "seed": 4433456,
    }
    state = {
        "campaign_root": tmp_path,
        "manifest": {"cases": [case]},
    }
    analysis = {
        "cases": [
            {
                "case_index": 0,
                **case,
                "status": "complete",
                "receipt": expected_receipt,
            }
        ]
    }
    monkeypatch.setattr(
        continuation.sealed,
        "_classify_case_state",
        lambda *_args, **_kwargs: {
            "case_id": "checkpoint-case",
            "status": "complete",
            "receipt": expected_receipt,
        },
    )

    observed = continuation._verify_checkpoint_case_identity(
        state, analysis, case_index=0
    )

    assert observed["status"] == "complete"
    assert observed["receipt"] == expected_receipt


def test_checkpoint_identity_change_refuses_rerun(monkeypatch, tmp_path):
    case = {
        "case_id": "checkpoint-case",
        "row_id": "checkpoint-row",
        "scenario": "raw_qblade",
        "wind_speed_mps": 3.0,
        "seed": 4433456,
    }
    state = {
        "campaign_root": tmp_path,
        "manifest": {"cases": [case]},
    }
    analysis = {
        "cases": [
            {
                "case_index": 0,
                **case,
                "status": "complete",
                "receipt": {"result": {"sha256": "b" * 64}},
            }
        ]
    }
    monkeypatch.setattr(
        continuation.sealed,
        "_classify_case_state",
        lambda *_args, **_kwargs: {
            "case_id": "checkpoint-case",
            "status": "blocked",
            "reason": "hash changed",
        },
    )
    monkeypatch.setattr(
        continuation.sealed,
        "execute_case",
        lambda *_args, **_kwargs: pytest.fail("checkpoint case must not rerun"),
    )

    with pytest.raises(
        continuation.HVAWTComplementaryDLCError,
        match="no longer complete; refuse to rerun",
    ):
        continuation._verify_checkpoint_case_identity(
            state, analysis, case_index=0
        )


def test_checkpoint_execution_wrapper_never_calls_sealed_executor(monkeypatch):
    expected = {"case_id": "checkpoint", "status": "complete"}
    monkeypatch.setattr(
        continuation,
        "_validate_checkpoint_case_reuse",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        continuation.sealed,
        "execute_case",
        lambda *_args, **_kwargs: pytest.fail("checkpoint case must not rerun"),
    )

    observed = continuation.execute_case(
        Path("campaign"),
        case_index=0,
        qblade_root=Path("qblade"),
        python_executable="python",
        threads=1,
    )

    assert observed == expected


def test_complementary_submission_array_has_exact_82_indices():
    submitter = (
        Path(__file__).resolve().parents[1]
        / "run_headless/submit_hvawt_complementary_dlc_flight.sj"
    )
    line = next(
        row for row in submitter.read_text(encoding="utf-8").splitlines()
        if row.startswith("#SBATCH --array=")
    )
    specification = line.split("=", 1)[1].split("%", 1)[0]
    indices = []
    for part in specification.split(","):
        if "-" in part:
            first, last = (int(value) for value in part.split("-", 1))
            indices.extend(range(first, last + 1))
        else:
            indices.append(int(part))

    assert tuple(indices) == continuation.COMPLEMENTARY_CASE_INDICES
    assert len(indices) == 82
    assert not set(indices).intersection(continuation.sealed.CHECKPOINT_CASE_INDICES)
    assert "%" not in line
    chain = (
        Path(__file__).resolve().parents[1]
        / "run_headless/submit_hvawt_complementary_chain.sh"
    ).read_text(encoding="utf-8")
    assert 'COMPLEMENTARY_ARRAY="1,3-8,10,' in chain
    assert "COMPLEMENTARY_ARRAY:-" not in chain
    array_line = next(
        row for row in chain.splitlines()
        if row.startswith("COMPLEMENTARY_ARRAY=")
    )
    assert "%" not in array_line
    assert (
        '"${VORLAP_ROOT}/run_headless/hvawt_complementary_dlc_campaign.py" '
        "preflight-sources"
        in chain
    )
    assert chain.index("preflight-sources") < chain.index("authorization_submission=")


def test_execution_wrappers_restore_sealed_loader(monkeypatch):
    original = continuation.sealed._load_execution_authorization
    observed = []

    def reviewed(*_args, **_kwargs):
        return {"status": "reviewed"}

    def fake_case(*_args, **kwargs):
        observed.append(continuation.sealed._load_execution_authorization(None, None))
        assert kwargs["authorization_scope"] == "full"
        return {"status": "complete"}

    def fake_smoke(*_args, **_kwargs):
        observed.append(continuation.sealed._load_execution_authorization(None, None))
        return {"status": "smoke"}

    monkeypatch.setattr(continuation, "_load_complementary_authorization", reviewed)
    monkeypatch.setattr(continuation.sealed, "execute_case", fake_case)
    monkeypatch.setattr(continuation.sealed, "execute_u50_smoke", fake_smoke)

    assert continuation.execute_case(
        Path("campaign"),
        case_index=1,
        qblade_root=Path("qblade"),
        python_executable="python",
        threads=1,
    )["status"] == "complete"
    assert continuation.sealed._load_execution_authorization is original
    assert continuation.execute_u50_smoke("campaign")["status"] == "smoke"
    assert continuation.sealed._load_execution_authorization is original
    assert observed == [{"status": "reviewed"}, {"status": "reviewed"}]
