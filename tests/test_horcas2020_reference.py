from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "examples" / "horcas2020_iea10_tip_geometry"
SPEC = importlib.util.spec_from_file_location("horcas2020_reference", ROOT / "validate.py")
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)


def _copy(tmp_path: Path) -> Path:
    root = tmp_path / "reference"
    root.mkdir(parents=True)
    for name in ("targets.csv", "provenance.json", "validation_protocol.json"):
        shutil.copy2(ROOT / name, root / name)
    return root


def test_complete_multi_geometry_matrix_and_holdout_are_frozen() -> None:
    result = reference.load_reference()
    assert len(result["rows"]) == 55
    assert set(result["by_geometry"]) == {"BASE", "EXT", "SWEEP", "DIHE", "COMB"}
    assert all(row["role"] == "locked_holdout" for row in result["by_geometry"]["COMB"])
    assert sum(row["role"] == "training" for row in result["rows"]) == 44
    assert result["protocol"]["matrix_requirements"]["holdout_case_count"] == 11
    assert result["protocol"]["acceptance"]["maximum_curve_nrmse_by_target_peak"] == 0.30


def test_peak_conversion_structural_rows_and_source_identity() -> None:
    result = reference.load_reference()
    assert result["targets_sha256"] == reference.TARGET_SHA
    assert result["provenance"]["publication"]["source"]["sha256"] == reference.SOURCE_SHA
    for geometry, rows in result["by_geometry"].items():
        expected = reference.GEOMETRIES[geometry]
        assert {float(row["first_edgewise_frequency_hz"]) for row in rows} == {expected[3]}
        for row in rows:
            assert float(row["peak_edge_amplitude_m"]) == pytest.approx(
                0.5 * float(row["peak_to_peak_edge_amplitude_m"]), abs=5.1e-10
            )


def test_mutated_target_and_claim_promotion_fail_closed(tmp_path: Path) -> None:
    copied = _copy(tmp_path)
    target = copied / "targets.csv"
    target.write_text(target.read_text(encoding="utf-8").replace("0.841887431", "0.941887431", 1), encoding="utf-8")
    with pytest.raises(reference.ValidationError, match="Target CSV SHA-256 mismatch"):
        reference.load_reference(copied)

    copied = _copy(tmp_path / "second")
    provenance_path = copied / "provenance.json"
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload["scope"]["supports_del_or_fatigue_validation"] = True
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(reference.ValidationError, match="Provenance SHA-256 mismatch"):
        reference.load_reference(copied)


def test_optional_source_pdf_hash_gate(tmp_path: Path) -> None:
    fake = tmp_path / "source.pdf"
    fake.write_bytes(b"not the source")
    with pytest.raises(reference.ValidationError, match="source PDF SHA-256 mismatch"):
        reference.load_reference(source_pdf=fake)
