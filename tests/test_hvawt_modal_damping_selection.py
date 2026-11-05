from __future__ import annotations

import json
from pathlib import Path

import pytest

import vorlap.hvawt_modal_decay_campaign as decay_campaign
from vorlap.hvawt_modal_damping_selection import (
    _validate_archived_parser_source,
    _validate_c20_substructure,
    derive_damping_selection,
    derive_documented_damping_selection,
    select_from_files,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
IDENTIFICATION_PATH = REPO_ROOT / "run_headless" / "config" / "hvawt_response_mode_identification.json"


def _identification() -> dict[str, object]:
    return json.loads(IDENTIFICATION_PATH.read_text(encoding="utf-8"))


def _analysis(source: str, *, residual: float = 0.0005) -> dict[str, object]:
    projected = {
        "station_vector_dominant": 0.016405402542907024,
        "file_scalar_dominant": 0.04809758189033014,
        "both_file_scalar_and_station_vector": 0.056191393722676466,
        "nonadditive_scalar_station_interaction": 0.03,
        "rayleigh_beta_effect_not_resolved": 0.0005,
    }[source]
    substructure = 0.008311590710560679
    station = {
        "station_vector_dominant": projected - substructure,
        "both_file_scalar_and_station_vector": 0.00809381183234635,
    }.get(source, 0.0)
    scalar = {
        "file_scalar_dominant": projected - substructure,
        "both_file_scalar_and_station_vector": 0.03978599117976946,
    }.get(source, 0.0)
    c20 = residual + substructure + scalar + station
    ratios = {
        "O": c20,
        "S": 0.2,
        "C20": c20,
        "C20_station_zero": residual + substructure + scalar,
        "C20_scalar_zero": residual + substructure + station,
        "all_beta_zero": residual,
    }
    def fit(zeta: float) -> decay_campaign.ModalDecayFit:
        return decay_campaign.ModalDecayFit(
            classification="interpretable_underdamped",
            frequency_hz=4.572593676110874,
            damping_ratio=zeta,
            normalized_rmse=0.01,
            decay_rates_per_s=(1.0,),
            peak_translation_m=0.001,
            burst_end_translation_m=0.0005,
            available_free_cycles=20.0,
            sample_count=1000,
        )
    classification = decay_campaign.classify_rayleigh_discriminator(
        {name: fit(value) for name, value in ratios.items()}
    )
    if source in {
        "nonadditive_scalar_station_interaction",
        "rayleigh_beta_effect_not_resolved",
    }:
        classification["damping_source"] = source
        classification["selection_allowed"] = False
    return {
        "schema": decay_campaign.ANALYSIS_SCHEMA,
        "status": "complete_interpretable",
        "classification": classification,
    }


@pytest.mark.parametrize(
    ("source", "expected_mode6", "expected_mode12", "method"),
    (
        (
            "station_vector_dominant",
            0.10377259458814346,
            0.016405402542907024,
            "c20_substructure_plus_station_vectors",
        ),
        (
            "file_scalar_dominant",
            0.11424928131000144,
            0.04809758189033014,
            "legacy_converter_header_scalar",
        ),
    ),
)
def test_selects_frozen_c20_projection_for_both_modes(
    source, expected_mode6, expected_mode12, method
):
    result = derive_damping_selection(_analysis(source), _identification())

    assert result["status"] == "selected"
    assert result["projection_method"] == method
    assert result["modes"][0]["rayleigh_energy_projected_damping_ratio"] == pytest.approx(
        expected_mode6
    )
    assert result["modes"][1]["rayleigh_energy_projected_damping_ratio"] == pytest.approx(
        expected_mode12
    )
    assert result["modes"][0]["qblade_effective_decay_damping_ratio"] is None
    assert result["modes"][1]["qblade_physical_rayleigh_increment_damping_ratio"] == pytest.approx(
        expected_mode12
    )


def test_combined_scalar_station_projection_counts_common_components_once():
    identification = _identification()
    result = derive_damping_selection(
        _analysis("both_file_scalar_and_station_vector"),
        identification,
    )

    assert result["projection_method"] == "c20_additive_station_and_file_scalar"
    assert result["modes"][0]["rayleigh_energy_projected_damping_ratio"] == pytest.approx(
        0.1156576933979216
    )
    assert result["modes"][1]["rayleigh_energy_projected_damping_ratio"] == pytest.approx(
        0.056191393722676466
    )


@pytest.mark.parametrize(
    "analysis",
    (
        _analysis("nonadditive_scalar_station_interaction"),
        _analysis("rayleigh_beta_effect_not_resolved"),
        _analysis("station_vector_dominant", residual=0.00101),
    ),
)
def test_rejects_unresolved_semantics_or_excess_numerical_damping(analysis):
    with pytest.raises(ValueError):
        derive_damping_selection(analysis, _identification())


def test_file_entrypoint_binds_input_bytes_and_writes_deterministically(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(
        json.dumps(_analysis("station_vector_dominant"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "selection.json"

    first = select_from_files(
        analysis_path=analysis_path,
        identification_path=IDENTIFICATION_PATH,
        output_path=output,
    )
    first_bytes = output.read_bytes()
    second = select_from_files(
        analysis_path=analysis_path,
        identification_path=IDENTIFICATION_PATH,
        output_path=output,
    )

    assert first == second
    assert output.read_bytes() == first_bytes
    assert first["inputs"]["rayleigh_analysis"]["sha256"]
    assert first["inputs"]["mode_identification"]["sha256"]


def test_documented_input_selection_freezes_corrected_c20_projection():
    result = derive_documented_damping_selection(_identification())

    assert result["status"] == "selected"
    assert result["selection_basis"] == (
        "qblade_parser_semantics_and_bounded_engineering_damping_assumption"
    )
    assert result["selected_structural_encoding"] == "C20"
    assert result["parser_semantics"] == "fixed_20th_numeric_token_is_damping"
    assert result["damping_source"] == (
        "distributed_station_beta_with_normalized_scalar_fallback"
    )
    assert result["modes"][0]["rayleigh_energy_projected_damping_ratio"] == pytest.approx(
        0.007448479057063753
    )
    assert result["modes"][1]["rayleigh_energy_projected_damping_ratio"] == pytest.approx(
        0.009503978382703563
    )
    assert all(row["qblade_effective_decay_damping_ratio"] is None for row in result["modes"])
    assert result["active_binary_verification"]["required_for_engineering_selection"] is False
    assert result["engineering_scalar_beta_s"] == pytest.approx(0.0011)
    assert result["engineering_scalar_components"] == ["substructure", "torque_tube", "tower"]


def test_documented_parser_and_c20_layout_are_exact(tmp_path):
    source = """
      subElements = FindNumericDataTable("SUBELEMENTS",subStructureStream,20,error,false);
      DAMP = subElements.at(m).at(19);
      if (subElements.at(m).size() > 20) DRAG = subElements.at(m).at(20);
      msection->SetBeamRaleyghDamping(DAMP);
    """
    _validate_archived_parser_source(source)
    with pytest.raises(ValueError, match="parser assignments"):
        _validate_archived_parser_source(source.replace("at(19)", "at(20)"))

    c20 = tmp_path / "substructure.str"
    c20.write_text(
        "SUBELEMENTS\n"
        "ID RHOA EA EIx EIy GJ GA PITCH KSX KSY RGX RGY XCM YCM XE YE XS YS DIA_[m] DAMP[-]\n"
        "1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1.000 0.02\n"
        "SUBMEMBERS\n",
        encoding="utf-8",
    )
    _validate_c20_substructure(c20)
    c20.write_text(c20.read_text().replace("0.02", "1.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="DAMP=0.02"):
        _validate_c20_substructure(c20)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("frequency_hz", 999.0),
        ("generalized_modal_mass_kg", -1.0),
        ("generalized_modal_stiffness_n_per_m", -1.0),
    ),
)
def test_rejects_mutated_mode_identity(field, value):
    identification = _identification()
    mode = next(row for row in identification["modes"] if row.get("tracked_mode") == 6)
    mode[field] = value
    with pytest.raises(ValueError):
        derive_damping_selection(_analysis("station_vector_dominant"), identification)


def test_rejects_inconsistent_classification_and_mode12_projection():
    analysis = _analysis("station_vector_dominant")
    analysis["classification"]["effective_c20_damping_ratio"] += 0.01
    with pytest.raises(ValueError):
        derive_damping_selection(analysis, _identification())

    analysis = _analysis("file_scalar_dominant")
    analysis["classification"]["damping_source"] = "both_file_scalar_and_station_vector"
    with pytest.raises(ValueError):
        derive_damping_selection(analysis, _identification())

    analysis = _analysis("station_vector_dominant")
    ratios = analysis["classification"]["variant_damping_ratios"]
    ratios["C20"] += 0.02
    analysis["classification"]["effective_c20_damping_ratio"] = ratios["C20"]
    with pytest.raises(ValueError):
        derive_damping_selection(analysis, _identification())
