from __future__ import annotations

from collections import namedtuple
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import warnings
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "decuyper2026_cylinder_wake"
SCRIPT = EXAMPLE_ROOT / "acquire_reference.py"
SPEC = importlib.util.spec_from_file_location("decuyper2026_acquisition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
acquisition = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acquisition
SPEC.loader.exec_module(acquisition)


def _load_json(name: str) -> dict:
    return json.loads((EXAMPLE_ROOT / name).read_text(encoding="utf-8"))


def _set_nested(payload: dict, keys: tuple[str, ...], value: object) -> None:
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def _write_zip(path: Path, members: list[tuple[object, bytes]]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            for member, payload in members:
                archive.writestr(member, payload)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]):
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def test_manifest_pins_exact_public_identity_and_noncommercial_license() -> None:
    manifest = acquisition.load_manifest()

    assert manifest["record"]["version"] == 1
    assert manifest["record"]["doi"] == (
        "10.4121/77985fa6-c546-4450-9d84-14b85f2d7e86.v1"
    )
    assert manifest["asset"] == {
        "file_id": "0516b70d-47b7-4008-b2ab-a8f77e2507d3",
        "url": (
            "https://data.4tu.nl/file/77985fa6-c546-4450-9d84-14b85f2d7e86/"
            "0516b70d-47b7-4008-b2ab-a8f77e2507d3"
        ),
        "filename": "Oscillating_cylinder_dataset.zip",
        "bytes": 6_261_659_558,
        "md5": "bed51060b5ce4ea0c36d00e41c8a8b99",
        "sha256": None,
        "sha256_policy": (
            "Compute locally only after the pinned byte count and repository MD5 "
            "pass; record the observed value in the acquisition receipt."
        ),
    }
    assert manifest["license"]["spdx"] == "CC-BY-NC-4.0"
    assert manifest["license"]["commercial_use_permitted"] is False
    assert manifest["license"]["repository_mit_license_applies_to_data"] is False
    assert manifest["acquisition"]["minimum_free_bytes"] == 20 * 1024**3
    assert manifest["acquisition"]["receipt_filename"] == "acquisition_receipt.json"
    assert manifest["acquisition"]["automatic_extraction"] is False


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("record", "doi"), "10.4121/not-the-pinned-version"),
        (("record", "version"), 2),
        (("asset", "url"), "https://example.invalid/archive.zip"),
        (("asset", "bytes"), 1),
        (("asset", "md5"), "0" * 32),
        (("asset", "sha256"), "0" * 64),
        (("license", "spdx"), "CC-BY-4.0"),
        (("acquisition", "minimum_free_bytes"), 20 * 1024**3 - 1),
        (("acquisition", "receipt_filename"), "untrusted-receipt.json"),
    ],
)
def test_manifest_identity_mutations_fail_closed(
    keys: tuple[str, ...], value: object
) -> None:
    manifest = _load_json("acquisition_manifest.json")
    _set_nested(manifest, keys, value)

    with pytest.raises(acquisition.AcquisitionError, match="pin mismatch"):
        acquisition.validate_manifest(manifest)


def test_protocol_freezes_semantic_13_6_16_split_without_raw_ids() -> None:
    protocol = _load_json("benchmark_protocol.json")
    source = protocol["source"]
    inventory = protocol["case_inventory"]
    splits = protocol["split_lock"]

    assert source["reference_type"] == (
        "Re=100 CFD simulation dataset; not a physical experiment"
    )
    assert source["reference_citation"] == "Decuyper, De Troyer & Runacres (2024)"
    assert source["benchmark_report_url"] == (
        "https://www.nonlinearbenchmark.org/benchmarks/unsteady-fluid-mechanics"
    )
    assert source["reported_metadata"] == {
        "sampling_frequency_hz": 40.0,
        "y_units": "m",
        "p_on_cylinder_quantity": "kinematic pressure p_k=p/rho",
        "p_on_cylinder_units": "m^2/s^2",
        "initial_flow": "shared developed stationary initial flow",
    }
    assert inventory["expected_total"] == 35
    assert inventory["raw_case_ids"] is None
    assert inventory["raw_case_indices"] is None
    assert inventory["mapping_status"] == "pending_verified_inventory"
    assert splits["training"]["expected_count"] == 13
    assert splits["development"]["expected_count"] == 6
    assert splits["final_holdout"]["expected_count"] == 16
    assert splits["training"]["selectors"] == [
        {"family": "stationary", "selection": "all", "expected_count": 1},
        {"family": "swept_sine", "selection": "all", "expected_count": 12},
    ]
    assert splits["development"]["selectors"][0]["realization_ordinal"] == 1
    assert splits["final_holdout"]["selectors"][0]["realization_ordinal"] == 2
    assert splits["final_holdout"]["selectors"][1]["family"] == "monosine"

    serialized_selectors = json.dumps(splits)
    assert "case_01" not in serialized_selectors
    assert "raw_case_ids" not in serialized_selectors


def test_protocol_locks_force_sign_before_fit_and_holdout_before_selection() -> None:
    protocol = _load_json("benchmark_protocol.json")
    sign_gate = protocol["force_sign_and_pressure_gate"]
    access = protocol["access_policy"]

    assert sign_gate["status"] == "unresolved_fail_closed"
    assert sign_gate["eligible_data"] == "training split only"
    assert "p_on_cylinder" in sign_gate["required_inputs"]
    assert "cylinder_points" in sign_gate["required_inputs"]
    assert sign_gate["pressure_resultants"][
        "source_kinematic_pressure_p_over_rho"
    ] == (
        "If and only if metadata confirms p_on_cylinder is kinematic pressure "
        "p_kin=p/rho, use C_pforce = -2 integral((p_kin-p_ref)n ds)/(U_inf^2 D), "
        "with p_ref in the same kinematic-pressure units."
    )
    assert all(
        token in sign_gate["pressure_metadata_gate"]
        for token in ("p/rho", "units", "p_ref", "U_inf", "D", "stop")
    )
    assert sign_gate["failure_action"].startswith("Stop before ROM fitting")
    assert access["training_fit_reads"] == ["training"]
    assert access["model_selection_reads"] == ["training", "development"]
    assert any(
        "final holdout for sign selection" in rule
        for rule in access["prohibited_before_final_unlock"]
    )
    assert protocol["minimum_rom"]["equations"] == [
        "d2q_ds2 + epsilon*(q^2 - 1)*dq_ds + q = b*u",
        "cy_hat = c_q*q + c_v*u",
    ]
    rom = protocol["minimum_rom"]
    assert rom["fit_parameters"] == [
        "epsilon",
        "b",
        "c_q",
        "c_v",
        "q0",
        "qdot0",
    ]
    assert rom["parameter_bounds"] == {
        "epsilon": [0.001, 5.0],
        "b": [-50.0, 50.0],
        "c_q": [0.001, 3.0],
        "c_v": [-5.0, 5.0],
        "q0": [-4.0, 4.0],
        "qdot0": [-4.0, 4.0],
    }
    assert "identical shared state" in rom["initial_state"]
    assert "training data only" in rom["initial_state"]
    assert "Record-specific phases or initial states are prohibited" in rom["initial_state"]
    assert rom["initial_state_definition"] == {
        "q0": "q evaluated at s=0",
        "qdot0": (
            "dq/ds evaluated at s=0; the dot denotes differentiation with respect "
            "to dimensionless s, not physical time t"
        ),
    }
    assert "stop and issue a new protocol" in rom["common_initialization_gate"]
    scaling = protocol["preprocessing"]["nondimensionalization"]
    assert scaling["time"] == "s = omega0 * t"
    assert scaling["displacement"] == "eta = y / D"
    assert scaling["transverse_velocity"] == (
        "u = d(eta)/ds = dy/dt / (omega0 * D)"
    )
    anchor = scaling["stationary_frequency_anchor"]
    assert anchor["source"] == "stationary training cy only"
    assert anchor["nominal_frequency_hz"] == 3.0
    assert anchor["relative_tolerance"] == 0.02
    assert anchor["angular_frequency"] == "omega0 = 2*pi*f0_hz"
    assert "not an optimized ROM parameter" in anchor["fit_boundary"]
    assert scaling["required_metadata"] == [
        "time units",
        "y units",
        "positive transverse coordinate and y sign convention",
        "cylinder diameter D",
        "free-stream speed U_inf",
        "pressure representation and units",
        "pressure reference p_ref",
        "cy and cx coefficient normalization and sign convention",
    ]
    assert all(
        token in scaling["metadata_gate"]
        for token in (
            "time units",
            "y units",
            "transverse coordinate sign",
            "D",
            "U_inf",
            "pressure units/reference",
            "coefficient convention",
            "stop",
        )
    )
    assert "source-reported y units of m" in scaling["metadata_gate"]
    assert "kinematic-pressure units of m^2/s^2" in scaling["metadata_gate"]
    assert "fs=40 Hz" in scaling["metadata_gate"]
    assert "shared developed stationary initial flow" in rom[
        "common_initialization_gate"
    ]
    claim = protocol["evidence_boundary"]["claim"]
    assert "code-to-code" in claim
    assert "CFD" in claim
    assert "not experimental validation" in claim
    not_established = protocol["evidence_boundary"]["not_established"]
    assert "experimental validation" in not_established
    assert (
        "QBlade structural-response or QBlade-coupled aeroelastic code-to-code verification"
        in not_established
    )
    assert "QBlade structural or code-to-code verification" not in not_established


def test_dry_run_has_no_write_and_space_gate_is_inclusive(tmp_path: Path) -> None:
    usage = namedtuple("usage", "total used free")
    raw_dir = tmp_path / "never-created" / "raw"

    result = acquisition.dry_run(
        raw_dir=raw_dir,
        disk_usage=lambda _path: usage(30 * 1024**3, 10 * 1024**3, 20 * 1024**3),
    )

    assert result["network_performed"] is False
    assert result["filesystem_writes"] is False
    assert result["available_free_bytes"] == 20 * 1024**3
    assert not raw_dir.exists()

    with pytest.raises(acquisition.AcquisitionError, match="free-space gate failed"):
        acquisition.dry_run(
            raw_dir=raw_dir,
            disk_usage=lambda _path: usage(
                30 * 1024**3, 10 * 1024**3, 20 * 1024**3 - 1
            ),
        )
    assert not raw_dir.exists()


def test_acquire_checks_space_before_network_or_writes(tmp_path: Path) -> None:
    usage = namedtuple("usage", "total used free")
    raw_dir = tmp_path / "raw"
    opener_called = False

    def forbidden_opener(*_args, **_kwargs):
        nonlocal opener_called
        opener_called = True
        raise AssertionError("network opener must not run")

    with pytest.raises(acquisition.AcquisitionError, match="free-space gate failed"):
        acquisition.acquire(
            raw_dir=raw_dir,
            opener=forbidden_opener,
            disk_usage=lambda _path: usage(100, 1, acquisition.MINIMUM_FREE_BYTES - 1),
        )

    assert opener_called is False
    assert not raw_dir.exists()


def test_acquire_reports_raw_directory_permission_failure_cleanly(
    tmp_path: Path,
) -> None:
    usage = namedtuple("usage", "total used free")
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("blocks mkdir\n", encoding="utf-8")
    raw_dir = blocking_file / "raw"

    def forbidden_opener(*_args, **_kwargs):
        raise AssertionError("network must not run when the raw directory is unusable")

    with pytest.raises(
        acquisition.AcquisitionError,
        match="cannot create raw directory",
    ):
        acquisition.acquire(
            raw_dir=raw_dir,
            opener=forbidden_opener,
            disk_usage=lambda _path: usage(
                100 * 1024**3,
                1,
                acquisition.MINIMUM_FREE_BYTES,
            ),
        )

    assert not raw_dir.exists()


def test_acquire_reuses_verified_final_without_space_or_network_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    final = raw_dir / acquisition.PINNED_FILENAME
    final.write_bytes(b"synthetic stand-in")
    opener_called = False
    disk_called = False

    def forbidden_opener(*_args, **_kwargs):
        nonlocal opener_called
        opener_called = True
        raise AssertionError("network must not run for a verified final")

    def forbidden_disk(_path):
        nonlocal disk_called
        disk_called = True
        raise AssertionError("space gate is irrelevant when no acquisition is needed")

    monkeypatch.setattr(
        acquisition,
        "verify_archive",
        lambda path, manifest_path: {
            "archive": str(path),
            "integrity": {
                "bytes": acquisition.PINNED_BYTES,
                "md5": acquisition.PINNED_MD5,
                "sha256": "a" * 64,
            },
            "zip_audit": {"principal_mat_member": acquisition.PRINCIPAL_MAT_BASENAME},
            "license": acquisition.PINNED_LICENSE,
        },
    )

    first_result = acquisition.acquire(
        raw_dir=raw_dir, opener=forbidden_opener, disk_usage=forbidden_disk
    )
    receipt = raw_dir / acquisition.RECEIPT_FILENAME
    first_receipt_bytes = receipt.read_bytes()
    receipt.write_text("stale receipt\n", encoding="utf-8")
    second_result = acquisition.acquire(
        raw_dir=raw_dir, opener=forbidden_opener, disk_usage=forbidden_disk
    )

    assert first_result["reused"] is True
    assert first_result["downloaded_bytes"] == 0
    assert first_result["receipt"] == str(receipt)
    assert second_result["receipt"] == str(receipt)
    assert receipt.read_bytes() == first_receipt_bytes
    receipt_payload = json.loads(first_receipt_bytes)
    assert receipt_payload["schema"] == acquisition.RECEIPT_SCHEMA
    assert receipt_payload["archive"] == {
        "filename": acquisition.PINNED_FILENAME,
        "bytes": acquisition.PINNED_BYTES,
        "md5": acquisition.PINNED_MD5,
        "sha256": "a" * 64,
    }
    assert opener_called is False
    assert disk_called is False


def test_integrity_gate_orders_size_then_md5_then_sha256(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"synthetic reference")
    calls: list[str] = []

    def recording_hash(path: Path, algorithm: str) -> str:
        calls.append(algorithm)
        return hashlib.new(algorithm, path.read_bytes()).hexdigest()

    with pytest.raises(acquisition.AcquisitionError, match="size mismatch"):
        acquisition.verify_file_integrity(candidate, candidate.stat().st_size + 1, "0" * 32, recording_hash)
    assert calls == []

    with pytest.raises(acquisition.AcquisitionError, match="MD5 mismatch"):
        acquisition.verify_file_integrity(candidate, candidate.stat().st_size, "0" * 32, recording_hash)
    assert calls == ["md5"]

    calls.clear()
    expected_md5 = hashlib.md5(candidate.read_bytes()).hexdigest()
    result = acquisition.verify_file_integrity(
        candidate, candidate.stat().st_size, expected_md5, recording_hash
    )
    assert calls == ["md5", "sha256"]
    assert result["hash_order"] == ["size", "md5", "sha256"]
    assert result["sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()


def test_stream_download_fresh_and_exact_resume_are_supported(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.part"
    fresh_payload = b"abcdefgh"
    fresh_requests = []

    def fresh_opener(request, *, timeout):
        fresh_requests.append((request, timeout))
        return FakeResponse(
            fresh_payload,
            status=200,
            headers={"Content-Length": str(len(fresh_payload))},
        )

    fresh_result = acquisition.stream_download(
        "https://example.invalid/data", fresh, len(fresh_payload), opener=fresh_opener
    )
    assert fresh.read_bytes() == fresh_payload
    assert fresh_result == {"resumed_from_bytes": 0, "downloaded_bytes": 8}
    assert fresh_requests[0][0].get_header("Range") is None

    resumed = tmp_path / "resumed.part"
    resumed.write_bytes(b"abc")
    resume_requests = []

    def resume_opener(request, *, timeout):
        resume_requests.append((request, timeout))
        return FakeResponse(
            b"defgh",
            status=206,
            headers={"Content-Range": "bytes 3-7/8", "Content-Length": "5"},
        )

    resumed_result = acquisition.stream_download(
        "https://example.invalid/data", resumed, 8, opener=resume_opener
    )
    assert resumed.read_bytes() == b"abcdefgh"
    assert resumed_result == {"resumed_from_bytes": 3, "downloaded_bytes": 5}
    assert resume_requests[0][0].get_header("Range") == "bytes=3-"


@pytest.mark.parametrize(
    ("status", "headers", "payload", "message"),
    [
        (200, {"Content-Length": "5"}, b"defgh", "requires HTTP 206"),
        (
            206,
            {"Content-Range": "bytes 2-7/8", "Content-Length": "5"},
            b"defgh",
            "does not match",
        ),
        (
            206,
            {"Content-Range": "bytes 3-7/9", "Content-Length": "5"},
            b"defgh",
            "does not match",
        ),
        (
            206,
            {"Content-Range": "bytes 3-7/8", "Content-Length": "6"},
            b"defghi",
            "Content-Length",
        ),
    ],
)
def test_resume_rejects_unsafe_http_responses_without_appending(
    tmp_path: Path,
    status: int,
    headers: dict[str, str],
    payload: bytes,
    message: str,
) -> None:
    partial = tmp_path / "archive.part"
    partial.write_bytes(b"abc")

    def opener(_request, *, timeout):
        return FakeResponse(payload, status=status, headers=headers)

    with pytest.raises(acquisition.AcquisitionError, match=message):
        acquisition.stream_download(
            "https://example.invalid/data", partial, 8, opener=opener
        )
    assert partial.read_bytes() == b"abc"


def test_truncated_and_oversize_streams_leave_a_partial_not_a_final(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.part"

    def truncated_opener(_request, *, timeout):
        return FakeResponse(b"abc", status=200, headers={})

    with pytest.raises(acquisition.AcquisitionError, match="download ended"):
        acquisition.stream_download(
            "https://example.invalid/data", truncated, 8, opener=truncated_opener
        )
    assert truncated.read_bytes() == b"abc"
    assert not (tmp_path / "truncated").exists()

    oversize = tmp_path / "oversize.part"

    def oversize_opener(_request, *, timeout):
        return FakeResponse(b"123456789", status=200, headers={})

    with pytest.raises(acquisition.AcquisitionError, match="exceeds"):
        acquisition.stream_download(
            "https://example.invalid/data", oversize, 8, opener=oversize_opener
        )
    assert oversize.read_bytes() == b""


def test_complete_partial_needs_no_network(tmp_path: Path) -> None:
    partial = tmp_path / "complete.part"
    partial.write_bytes(b"complete")

    def forbidden_opener(*_args, **_kwargs):
        raise AssertionError("a complete partial must be verified without network")

    result = acquisition.stream_download(
        "https://example.invalid/data", partial, 8, opener=forbidden_opener
    )
    assert result == {"resumed_from_bytes": 8, "downloaded_bytes": 0}


def test_existing_zero_byte_partial_is_a_valid_fresh_download_target(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "empty.part"
    partial.touch()

    def opener(_request, *, timeout):
        return FakeResponse(b"abc", status=200, headers={"Content-Length": "3"})

    result = acquisition.stream_download(
        "https://example.invalid/data", partial, 3, opener=opener
    )
    assert partial.read_bytes() == b"abc"
    assert result == {"resumed_from_bytes": 0, "downloaded_bytes": 3}


def test_zip_audit_accepts_nested_unique_principal_without_extracting(tmp_path: Path) -> None:
    archive = tmp_path / "valid.zip"
    _write_zip(
        archive,
        [
            ("metadata/readme.txt", b"metadata"),
            (
                "dataset/oscillating_cylinder_benchmark_dataset_v4.mat",
                b"synthetic mat payload",
            ),
        ],
    )

    result = acquisition.audit_zip_members(archive)

    assert result["member_count"] == 2
    assert result["principal_mat_member"].startswith("dataset/")
    assert result["extracted"] is False
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "/absolute/oscillating_cylinder_benchmark_dataset_v4.mat",
        "../oscillating_cylinder_benchmark_dataset_v4.mat",
        "nested/../../oscillating_cylinder_benchmark_dataset_v4.mat",
        "C:\\unsafe\\oscillating_cylinder_benchmark_dataset_v4.mat",
        "..\\oscillating_cylinder_benchmark_dataset_v4.mat",
    ],
)
def test_zip_audit_rejects_absolute_and_traversing_paths(
    tmp_path: Path, unsafe_name: str
) -> None:
    archive = tmp_path / "unsafe.zip"
    _write_zip(archive, [(unsafe_name, b"payload")])

    with pytest.raises(acquisition.AcquisitionError, match="unsafe|absolute"):
        acquisition.audit_zip_members(archive)


def test_zip_audit_rejects_symlink_and_special_members(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("oscillating_cylinder_benchmark_dataset_v4.mat")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    _write_zip(archive, [(link, b"target")])

    with pytest.raises(acquisition.AcquisitionError, match="symlink or special"):
        acquisition.audit_zip_members(archive)


def test_zip_audit_rejects_missing_duplicate_and_normalized_collision(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.zip"
    _write_zip(missing, [("another.mat", b"payload")])
    with pytest.raises(acquisition.AcquisitionError, match="exactly one"):
        acquisition.audit_zip_members(missing)

    duplicate_principal = tmp_path / "duplicate-principal.zip"
    principal = "oscillating_cylinder_benchmark_dataset_v4.mat"
    _write_zip(
        duplicate_principal,
        [("one/" + principal, b"one"), ("two/" + principal, b"two")],
    )
    with pytest.raises(acquisition.AcquisitionError, match="exactly one"):
        acquisition.audit_zip_members(duplicate_principal)

    collision = tmp_path / "collision.zip"
    _write_zip(
        collision,
        [("nested//" + principal, b"one"), ("nested/" + principal, b"two")],
    )
    with pytest.raises(acquisition.AcquisitionError, match="duplicate normalized"):
        acquisition.audit_zip_members(collision)


def test_zip_audit_rejects_malformed_archive(tmp_path: Path) -> None:
    archive = tmp_path / "not-a-zip.zip"
    archive.write_bytes(b"not a ZIP central directory")

    with pytest.raises(acquisition.AcquisitionError, match="valid ZIP"):
        acquisition.audit_zip_members(archive)


def test_data_products_are_ignored_and_license_boundary_is_explicit() -> None:
    ignored = (EXAMPLE_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")
    notice = (EXAMPLE_ROOT / "DATA_LICENSE_NOTICE.md").read_text(encoding="utf-8")

    assert ignored == ["raw/", "processed/"]
    for text in (readme, notice):
        assert "CC BY-NC 4.0" in text
        assert "noncommercial" in text.lower()
        assert "MIT" in text
        assert "does not" in text.lower()
    assert "does not redistribute" in notice
    assert "C_pforce = -2 integral((p_kin-p_ref)n ds)/(U_inf^2 D)" in readme
    assert "`qdot0` means" in readme
    assert "`dq/ds` evaluated at `s=0`" in readme
    assert "CFD simulation dataset, not a physical" in readme
    assert "Decuyper, De Troyer, and Runacres" in readme
    assert "not experimental validation" in readme
