#!/usr/bin/env python3
"""Load and validate the provenance-bound Fu et al. (2026) Figure 21 targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROVENANCE = ROOT / "provenance.json"

EXPECTED_SCHEMA = "fu2026-iea10-fig21-reference-v1"
EXPECTED_DOI = "10.1016/j.jfluidstructs.2026.104537"
EXPECTED_DOI_URL = "https://doi.org/10.1016/j.jfluidstructs.2026.104537"
EXPECTED_PUBLICATION_LICENSE = "CC BY 4.0"
EXPECTED_PUBLICATION_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
EXPECTED_VECTOR_SOURCE_URL = "https://xuepengfu.github.io/assets/pdf/bladejfs.pdf"
EXPECTED_VECTOR_SOURCE_LOCATION = (17, 16, 21)
EXPECTED_SOURCE_PDF_SHA256 = (
    "304bc4e37021f4dd64e17d0a6e3a4b0ed06eddf232a10878031269b9014485e7"
)
EXPECTED_TARGETS_SHA256 = (
    "64e958175ade5423bf51c15ed68756a8458199fc06601b60205f2bf14ff39ff1"
)
EXPECTED_IEA_COMMIT = "350cf06f5db9cec12bcd73c2b174a5d9642958d8"
EXPECTED_IEA_REPOSITORY = "https://github.com/IEAWindSystems/IEA-10.0-198-RWT"
EXPECTED_IEA_LICENSE = "Apache-2.0"
EXPECTED_IEA_FILE_HASHES = {
    "yaml/IEA-10-198-RWT.yaml": (
        "298ffdcd700e558aa69bee2a78e15c6f43f4e0ff5253a619aa314ffa9bb3b7f5"
    ),
    "openfast/IEA-10.0-198-RWT_BeamDyn.dat": (
        "3d1e6de05d2b7918de0627c74278556dcc304d0542c3fffd156018fa242714f9"
    ),
    "openfast/IEA-10.0-198-RWT_BeamDyn_blade.dat": (
        "0b5372174194eb974e06eedc8faab3386d2a2a2825ea442bdf6037ec56b604c4"
    ),
}
EXPECTED_DIGITIZATION_METHOD = (
    "Circle centers were recovered from the PDF vector paths and calibrated "
    "against each panel's vector axis ticks; no raster point picking was used."
)
EXPECTED_SOURCE_STATUS = (
    "vector-digitized estimates, not author-provided raw solver values"
)
EXPECTED_FORMULAS = {
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
EXPECTED_SPEEDS_MPS = (6, 10, 14, 18, 22)
EXPECTED_CONFIGS = {
    "P80A310": (80, 90, 310, 100, 50),
    "P90A310": (90, 90, 310, 90, 50),
    "P95A310": (95, 90, 310, 85, 50),
    "P95A320": (95, 90, 320, 85, 40),
}
EXPECTED_DEBUG_CASE = "P80A310_U18"
EXPECTED_DEBUG_PROHIBITED_USES = (
    "aerodynamic coefficient tuning",
    "force-amplitude tuning",
    "response-frequency tuning",
    "structural-property tuning to the Fu target",
)
EXPECTED_FREEZE_RULE = (
    "No response-model coefficient, structural property, force scale, or scoring "
    "rule may change after the debug audit if the 19 cases are to remain holdouts."
)
EXPECTED_TABLE4 = {
    "P80A310_U18": (1.19, 0.20),
    "P95A310_U18": (1.23, 0.21),
}
EXPECTED_COLUMNS = (
    "case_id",
    "config",
    "alpha_deg",
    "beta_deg",
    "gamma_deg",
    "dtu_pitch_deg",
    "dtu_inclination_deg",
    "U_mps",
    "Ur",
    "Ay_over_cbar",
    "Ay_m",
    "Ax_over_cbar",
    "Ax_m",
    "fy_over_fn2",
    "fy_hz",
    "Ur_unc",
    "Ac_unc",
    "A_m_unc",
    "fr_unc",
    "f_hz_unc",
    "role",
    "freq_scoreable",
)


class ValidationError(ValueError):
    """Raised when the reference package does not match its preregistration."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"Non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Could not load strict JSON from {path}: {exc}") from exc
    _require(isinstance(payload, dict), "Provenance root must be a JSON object.")
    return payload


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValidationError(f"Missing required {context} field: {key}")
    return mapping[key]


def _finite_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid numeric field {key!r} in {row.get('case_id')}") from exc
    _require(math.isfinite(value), f"Non-finite {key} in {row.get('case_id')}")
    return value


def _exact_int(row: dict[str, str], key: str) -> int:
    value = _finite_float(row, key)
    integer = int(value)
    _require(value == integer, f"{key} must be an exact integer in {row.get('case_id')}")
    return integer


def _load_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(
                tuple(reader.fieldnames or ()) == EXPECTED_COLUMNS,
                "Target CSV columns or ordering changed.",
            )
            rows = list(reader)
    except OSError as exc:
        raise ValidationError(f"Could not load target CSV from {path}: {exc}") from exc
    _require(all(None not in row for row in rows), "Target CSV has extra unnamed fields.")
    _require(
        all(all(value != "" for value in row.values()) for row in rows),
        "Target CSV contains an empty field.",
    )
    return rows


def _expected_case_id(config: str, speed_mps: int) -> str:
    return f"{config}_U{speed_mps:02d}"


def load_reference(root: Path = ROOT, source_pdf: Path | None = None) -> dict[str, Any]:
    """Return validated provenance and rows, or fail closed on any drift."""
    root = Path(root)
    provenance_path = root / "provenance.json"
    provenance = _load_json(provenance_path)
    _require(provenance.get("schema") == EXPECTED_SCHEMA, "Unexpected schema.")

    publication = _required(provenance, "publication", "provenance")
    _require(isinstance(publication, dict), "publication must be an object.")
    _require(publication.get("doi") == EXPECTED_DOI, "Publication DOI changed.")
    _require(
        publication.get("doi_url") == EXPECTED_DOI_URL,
        "Publication DOI URL changed.",
    )
    _require(
        publication.get("license") == EXPECTED_PUBLICATION_LICENSE,
        "Publication license changed.",
    )
    _require(
        publication.get("license_url") == EXPECTED_PUBLICATION_LICENSE_URL,
        "Publication license URL changed.",
    )
    vector_source = _required(publication, "vector_source", "publication")
    _require(isinstance(vector_source, dict), "vector_source must be an object.")
    _require(
        vector_source.get("url") == EXPECTED_VECTOR_SOURCE_URL,
        "Vector-source URL changed.",
    )
    source_location = (
        vector_source.get("pdf_page_1_based"),
        vector_source.get("article_page"),
        vector_source.get("figure"),
    )
    _require(
        source_location == EXPECTED_VECTOR_SOURCE_LOCATION,
        "Vector-source page or figure changed.",
    )
    source_hash = vector_source.get("sha256")
    _require(
        isinstance(source_hash, str) and len(source_hash) == 64,
        "Source PDF SHA-256 is unavailable; refusing unbound reference data.",
    )
    _require(
        source_hash == EXPECTED_SOURCE_PDF_SHA256,
        "Source PDF SHA-256 changed from the preregistered value.",
    )
    if source_pdf is not None:
        _require(source_pdf.is_file(), f"Source PDF is unavailable: {source_pdf}")
        _require(
            sha256_file(source_pdf) == EXPECTED_SOURCE_PDF_SHA256,
            "Provided source PDF does not match the preregistered SHA-256.",
        )

    public_model = _required(provenance, "public_iea_model", "provenance")
    _require(
        public_model.get("repository") == EXPECTED_IEA_REPOSITORY,
        "Public IEA model repository changed.",
    )
    _require(
        public_model.get("commit") == EXPECTED_IEA_COMMIT,
        "Public IEA model commit changed.",
    )
    _require(
        public_model.get("license") == EXPECTED_IEA_LICENSE,
        "Public IEA model license changed.",
    )
    _require(
        public_model.get("files") == EXPECTED_IEA_FILE_HASHES,
        "Public IEA model file hashes changed.",
    )

    artifacts = _required(provenance, "artifacts", "provenance")
    target_meta = _required(artifacts, "targets_csv", "artifacts")
    _require(target_meta.get("path") == "targets.csv", "Unexpected target CSV path.")
    _require(target_meta.get("rows") == 20, "Preregistered row count must remain 20.")
    _require(
        target_meta.get("sha256") == EXPECTED_TARGETS_SHA256,
        "Manifest target CSV SHA-256 changed.",
    )
    target_path = root / "targets.csv"
    target_hash = sha256_file(target_path)
    _require(target_hash == EXPECTED_TARGETS_SHA256, "Target CSV SHA-256 mismatch.")
    rows = _load_rows(target_path)
    _require(len(rows) == 20, "Target CSV must contain exactly 20 rows.")

    digitization = _required(provenance, "digitization", "provenance")
    _require(
        digitization.get("method") == EXPECTED_DIGITIZATION_METHOD,
        "Digitization method changed.",
    )
    _require(
        digitization.get("source_status") == EXPECTED_SOURCE_STATUS,
        "Digitization source status changed.",
    )
    _require(
        digitization.get("formulas") == EXPECTED_FORMULAS,
        "Recorded formula strings changed.",
    )
    constants = _required(digitization, "constants", "digitization")
    mean_chord_m = float(_required(constants, "mean_chord_m", "constants"))
    natural_frequency_hz = float(
        _required(constants, "first_edgewise_frequency_hz", "constants")
    )
    _require(mean_chord_m == 3.72, "Mean chord changed from the published value.")
    _require(
        natural_frequency_hz == 0.6739,
        "First-edgewise frequency changed from the published value.",
    )
    uncertainties = _required(
        digitization, "common_digitization_uncertainty", "digitization"
    )
    expected_uncertainties = (0.01, 0.002, 0.00744, 0.02, 0.013478)
    threshold = float(
        _required(
            _required(digitization, "frequency_scoring_rule", "digitization"),
            "minimum_edge_amplitude_over_mean_chord",
            "frequency_scoring_rule",
        )
    )
    _require(threshold == 0.01, "Frequency scoring threshold changed.")

    expected_ids = {
        _expected_case_id(config, speed)
        for config in EXPECTED_CONFIGS
        for speed in EXPECTED_SPEEDS_MPS
    }
    seen_ids: set[str] = set()
    rows_by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row["case_id"]
        config = row["config"]
        _require(config in EXPECTED_CONFIGS, f"Unexpected configuration in {case_id}.")
        speed = _exact_int(row, "U_mps")
        _require(speed in EXPECTED_SPEEDS_MPS, f"Unexpected wind speed in {case_id}.")
        _require(
            case_id == _expected_case_id(config, speed),
            f"Case identifier/configuration mismatch for {case_id}.",
        )
        _require(case_id not in seen_ids, f"Duplicate target case: {case_id}")
        seen_ids.add(case_id)
        rows_by_id[case_id] = row

        observed_config = tuple(
            _exact_int(row, key)
            for key in (
                "alpha_deg",
                "beta_deg",
                "gamma_deg",
                "dtu_pitch_deg",
                "dtu_inclination_deg",
            )
        )
        _require(
            observed_config == EXPECTED_CONFIGS[config],
            f"Published/DTU angle mapping changed for {case_id}.",
        )
        gamma_deg = observed_config[2]
        expected_ur = (
            speed
            * math.cos(math.radians(gamma_deg))
            / (natural_frequency_hz * mean_chord_m)
        )
        ur = _finite_float(row, "Ur")
        ay_over_cbar = _finite_float(row, "Ay_over_cbar")
        ay_m = _finite_float(row, "Ay_m")
        ax_over_cbar = _finite_float(row, "Ax_over_cbar")
        ax_m = _finite_float(row, "Ax_m")
        frequency_ratio = _finite_float(row, "fy_over_fn2")
        frequency_hz = _finite_float(row, "fy_hz")
        _require(abs(ur - expected_ur) <= 5.0e-10, f"Ur formula failed for {case_id}.")
        _require(
            abs(ay_m - ay_over_cbar * mean_chord_m) <= 5.1e-7,
            f"Edge-amplitude conversion failed for {case_id}.",
        )
        _require(
            abs(ax_m - ax_over_cbar * mean_chord_m) <= 5.1e-7,
            f"Flap-amplitude conversion failed for {case_id}.",
        )
        _require(
            abs(frequency_hz - frequency_ratio * natural_frequency_hz) <= 5.1e-7,
            f"Frequency conversion failed for {case_id}.",
        )
        _require(
            ay_over_cbar >= 0.0 and ax_over_cbar >= 0.0 and frequency_ratio > 0.0,
            f"Response quantities must be nonnegative and frequency positive in {case_id}.",
        )
        observed_uncertainties = tuple(
            _finite_float(row, key)
            for key in ("Ur_unc", "Ac_unc", "A_m_unc", "fr_unc", "f_hz_unc")
        )
        _require(
            observed_uncertainties == expected_uncertainties,
            f"Digitization uncertainty changed for {case_id}.",
        )
        expected_scoreable = ay_over_cbar >= threshold
        _require(
            row["freq_scoreable"] in {"true", "false"},
            f"Invalid freq_scoreable flag for {case_id}.",
        )
        _require(
            (row["freq_scoreable"] == "true") == expected_scoreable,
            f"Frequency scoring rule failed for {case_id}.",
        )
        expected_role = "debug" if case_id == EXPECTED_DEBUG_CASE else "locked_holdout"
        _require(row["role"] == expected_role, f"Preregistered role changed for {case_id}.")

    _require(seen_ids == expected_ids, "The complete 4-by-5 case matrix changed.")
    manifest_uncertainties = (
        float(uncertainties.get("reduced_velocity")),
        float(uncertainties.get("amplitude_over_mean_chord")),
        float(uncertainties.get("amplitude_m")),
        float(uncertainties.get("response_frequency_over_fn2")),
        float(uncertainties.get("response_frequency_hz")),
    )
    _require(
        manifest_uncertainties == expected_uncertainties,
        "Manifest digitization uncertainties changed.",
    )

    preregistration = _required(provenance, "preregistration", "provenance")
    _require(
        preregistration.get("debug_case_id") == EXPECTED_DEBUG_CASE,
        "Debug case changed.",
    )
    _require(
        tuple(preregistration.get("debug_prohibited_uses", ()))
        == EXPECTED_DEBUG_PROHIBITED_USES,
        "Debug prohibited-tuning semantics changed.",
    )
    _require(
        preregistration.get("freeze_rule") == EXPECTED_FREEZE_RULE,
        "Holdout freeze rule changed.",
    )
    holdout_ids = preregistration.get("locked_holdout_case_ids")
    _require(isinstance(holdout_ids, list), "Locked holdouts must be a list.")
    _require(len(holdout_ids) == 19, "There must be exactly 19 locked holdouts.")
    _require(
        set(holdout_ids) == expected_ids - {EXPECTED_DEBUG_CASE},
        "Locked holdout membership changed.",
    )

    boundaries = _required(provenance, "recreation_boundary", "provenance")
    _require(boundaries.get("exact_fu_recreations") == 0, "Exact boundary changed.")
    _require(
        boundaries.get("as_configured_qblade_input_matches") == 1,
        "As-configured QBlade boundary changed.",
    )
    _require(
        boundaries.get("documented_kinematic_analogue_targets") == 20,
        "Analogue boundary changed.",
    )

    cross_checks = _required(provenance, "table4_cross_checks", "provenance")
    _require(len(cross_checks) == 2, "Table 4 cross-check count changed.")
    for check in cross_checks:
        case_id = check.get("case_id")
        _require(case_id in EXPECTED_TABLE4, f"Unexpected Table 4 case: {case_id}")
        expected_edge, expected_flap = EXPECTED_TABLE4[case_id]
        _require(
            check.get("published_edge_amplitude_m") == expected_edge
            and check.get("published_flap_amplitude_m") == expected_flap,
            f"Published Table 4 values changed for {case_id}.",
        )
        row = rows_by_id[case_id]
        amplitude_uncertainty_m = _finite_float(row, "A_m_unc")
        _require(
            abs(_finite_float(row, "Ay_m") - expected_edge) <= amplitude_uncertainty_m,
            f"Figure 21 edge value does not cross-check Table 4 for {case_id}.",
        )
        _require(
            abs(_finite_float(row, "Ax_m") - expected_flap) <= amplitude_uncertainty_m,
            f"Figure 21 flap value does not cross-check Table 4 for {case_id}.",
        )

    scope = _required(provenance, "scope", "provenance")
    _require(
        scope.get("supports_del_or_fatigue_claim") is False,
        "This reference package must not support a DEL or fatigue claim.",
    )
    _require(
        scope.get("supports_time_history_comparison") is False
        and scope.get("supports_spanwise_load_comparison") is False,
        "Figure 21 cannot support time-history or spanwise-load comparisons.",
    )
    no_del_reason = scope.get("no_del_reason")
    _require(
        isinstance(no_del_reason, str) and "time histories" in no_del_reason,
        "A concrete no-DEL reason is required.",
    )

    return {
        "provenance": provenance,
        "rows": rows,
        "rows_by_id": rows_by_id,
        "targets_sha256": target_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Reference-package directory (default: directory containing this script).",
    )
    parser.add_argument(
        "--source-pdf",
        type=Path,
        help="Optional downloaded source PDF to verify against the preregistered SHA-256.",
    )
    args = parser.parse_args()
    result = load_reference(args.root, args.source_pdf)
    print(
        f"validated {len(result['rows'])} Fu et al. Figure 21 targets; "
        f"targets_sha256={result['targets_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
