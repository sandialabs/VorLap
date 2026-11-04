from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "examples" / "fu2026_iea10"
SCRIPT = REFERENCE_ROOT / "validate.py"
SPEC = importlib.util.spec_from_file_location("fu2026_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)

EXPECTED_SOURCE_PDF_SHA256 = (
    "304bc4e37021f4dd64e17d0a6e3a4b0ed06eddf232a10878031269b9014485e7"
)
EXPECTED_TARGETS_SHA256 = (
    "64e958175ade5423bf51c15ed68756a8458199fc06601b60205f2bf14ff39ff1"
)


def _copy_reference(tmp_path: Path) -> Path:
    copied = tmp_path / "fu2026_iea10"
    copied.mkdir(parents=True)
    shutil.copy2(REFERENCE_ROOT / "targets.csv", copied / "targets.csv")
    shutil.copy2(REFERENCE_ROOT / "provenance.json", copied / "provenance.json")
    return copied


def _set_nested(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    target: object = payload
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = value


def test_fu_reference_is_hash_bound_and_freezes_the_complete_matrix() -> None:
    result = reference.load_reference()
    provenance = result["provenance"]
    rows = result["rows"]

    assert result["targets_sha256"] == EXPECTED_TARGETS_SHA256
    assert reference.EXPECTED_TARGETS_SHA256 == EXPECTED_TARGETS_SHA256
    assert reference.EXPECTED_SOURCE_PDF_SHA256 == EXPECTED_SOURCE_PDF_SHA256
    assert provenance["publication"]["vector_source"]["sha256"] == (
        EXPECTED_SOURCE_PDF_SHA256
    )
    assert len(rows) == 20
    assert {row["config"] for row in rows} == set(reference.EXPECTED_CONFIGS)
    for config in reference.EXPECTED_CONFIGS:
        assert {
            int(row["U_mps"]) for row in rows if row["config"] == config
        } == set(reference.EXPECTED_SPEEDS_MPS)
    assert sum(row["role"] == "debug" for row in rows) == 1
    assert sum(row["role"] == "locked_holdout" for row in rows) == 19
    assert sum(row["freq_scoreable"] == "true" for row in rows) == 12


def test_formulas_table4_cross_check_and_scope_boundary_are_frozen() -> None:
    result = reference.load_reference()
    provenance = result["provenance"]
    rows = result["rows_by_id"]
    constants = provenance["digitization"]["constants"]
    mean_chord_m = constants["mean_chord_m"]
    natural_frequency_hz = constants["first_edgewise_frequency_hz"]

    assert provenance["digitization"]["formulas"] == {
        "reduced_velocity": (
            "Ur = U_mps * cos(gamma_deg*pi/180) / "
            "(first_edgewise_frequency_hz * mean_chord_m)"
        ),
        "edge_amplitude_m": "Ay_m = Ay_over_cbar * mean_chord_m",
        "flap_amplitude_m": "Ax_m = Ax_over_cbar * mean_chord_m",
        "response_frequency_hz": (
            "fy_hz = fy_over_fn2 * first_edgewise_frequency_hz"
        ),
    }
    for row in rows.values():
        expected_ur = (
            float(row["U_mps"])
            * math.cos(math.radians(float(row["gamma_deg"])))
            / (natural_frequency_hz * mean_chord_m)
        )
        assert float(row["Ur"]) == pytest.approx(expected_ur, abs=5.0e-10)
        assert float(row["Ay_m"]) == pytest.approx(
            float(row["Ay_over_cbar"]) * mean_chord_m,
            abs=5.1e-7,
        )
        assert float(row["Ax_m"]) == pytest.approx(
            float(row["Ax_over_cbar"]) * mean_chord_m,
            abs=5.1e-7,
        )
        assert float(row["fy_hz"]) == pytest.approx(
            float(row["fy_over_fn2"]) * natural_frequency_hz,
            abs=5.1e-7,
        )

    p80 = rows["P80A310_U18"]
    p95 = rows["P95A310_U18"]
    assert float(p80["Ay_m"]) == pytest.approx(1.19, abs=float(p80["A_m_unc"]))
    assert float(p80["Ax_m"]) == pytest.approx(0.20, abs=float(p80["A_m_unc"]))
    assert float(p95["Ay_m"]) == pytest.approx(1.23, abs=float(p95["A_m_unc"]))
    assert float(p95["Ax_m"]) == pytest.approx(0.21, abs=float(p95["A_m_unc"]))

    boundary = provenance["recreation_boundary"]
    assert boundary["exact_fu_recreations"] == 0
    assert boundary["as_configured_qblade_input_matches"] == 1
    assert boundary["documented_kinematic_analogue_targets"] == 20
    assert provenance["scope"]["supports_del_or_fatigue_claim"] is False
    assert provenance["scope"]["supports_time_history_comparison"] is False
    assert provenance["scope"]["supports_spanwise_load_comparison"] is False


def test_loader_fails_closed_if_source_hash_is_unavailable(tmp_path: Path) -> None:
    copied = _copy_reference(tmp_path)
    provenance_path = copied / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    del provenance["publication"]["vector_source"]["sha256"]
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(reference.ValidationError, match="SHA-256 is unavailable"):
        reference.load_reference(copied)


def test_loader_rejects_target_drift_and_any_del_promotion(tmp_path: Path) -> None:
    copied = _copy_reference(tmp_path)
    targets_path = copied / "targets.csv"
    targets_path.write_text(
        targets_path.read_text(encoding="utf-8").replace(
            "0.490363,1.824150",
            "0.490363,1.824151",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(reference.ValidationError, match="Target CSV SHA-256 mismatch"):
        reference.load_reference(copied)

    copied = _copy_reference(tmp_path / "second")
    provenance_path = copied / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["scope"]["supports_del_or_fatigue_claim"] = True
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(reference.ValidationError, match="must not support a DEL"):
        reference.load_reference(copied)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        pytest.param(
            ("publication", "doi"),
            "10.invalid/fu2026",
            "Publication DOI changed",
            id="doi",
        ),
        pytest.param(
            ("publication", "doi_url"),
            "https://doi.org/10.invalid/fu2026",
            "Publication DOI URL changed",
            id="doi-url",
        ),
        pytest.param(
            ("publication", "license"),
            "unknown",
            "Publication license changed",
            id="publication-license",
        ),
        pytest.param(
            ("publication", "license_url"),
            "https://example.invalid/license",
            "Publication license URL changed",
            id="publication-license-url",
        ),
        pytest.param(
            ("publication", "vector_source", "url"),
            "https://example.invalid/bladejfs.pdf",
            "Vector-source URL changed",
            id="vector-url",
        ),
        pytest.param(
            ("publication", "vector_source", "pdf_page_1_based"),
            16,
            "Vector-source page or figure changed",
            id="vector-pdf-page",
        ),
        pytest.param(
            ("publication", "vector_source", "article_page"),
            15,
            "Vector-source page or figure changed",
            id="vector-article-page",
        ),
        pytest.param(
            ("publication", "vector_source", "figure"),
            20,
            "Vector-source page or figure changed",
            id="vector-figure",
        ),
        pytest.param(
            ("digitization", "method"),
            "raster point picking",
            "Digitization method changed",
            id="digitization-method",
        ),
        pytest.param(
            ("digitization", "source_status"),
            "author-provided raw values",
            "Digitization source status changed",
            id="source-status",
        ),
        pytest.param(
            ("digitization", "formulas", "reduced_velocity"),
            "Ur = U/(fn*c)",
            "Recorded formula strings changed",
            id="formula-string",
        ),
        pytest.param(
            ("public_iea_model", "repository"),
            "https://example.invalid/iea10",
            "Public IEA model repository changed",
            id="iea-repository",
        ),
        pytest.param(
            ("public_iea_model", "license"),
            "unknown",
            "Public IEA model license changed",
            id="iea-license",
        ),
        pytest.param(
            ("public_iea_model", "files", "yaml/IEA-10-198-RWT.yaml"),
            "0" * 64,
            "Public IEA model file hashes changed",
            id="iea-yaml-hash",
        ),
        pytest.param(
            (
                "public_iea_model",
                "files",
                "openfast/IEA-10.0-198-RWT_BeamDyn.dat",
            ),
            "0" * 64,
            "Public IEA model file hashes changed",
            id="iea-beamdyn-hash",
        ),
        pytest.param(
            (
                "public_iea_model",
                "files",
                "openfast/IEA-10.0-198-RWT_BeamDyn_blade.dat",
            ),
            "0" * 64,
            "Public IEA model file hashes changed",
            id="iea-beamdyn-blade-hash",
        ),
        pytest.param(
            ("preregistration", "debug_prohibited_uses"),
            ["force-amplitude tuning"],
            "Debug prohibited-tuning semantics changed",
            id="prohibited-tuning",
        ),
        pytest.param(
            ("preregistration", "freeze_rule"),
            "Tuning remains allowed after debug.",
            "Holdout freeze rule changed",
            id="freeze-rule",
        ),
    ],
)
def test_loader_rejects_critical_provenance_drift(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: object,
    message: str,
) -> None:
    copied = _copy_reference(tmp_path)
    provenance_path = copied / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    _set_nested(provenance, path, replacement)
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(reference.ValidationError, match=message):
        reference.load_reference(copied)
