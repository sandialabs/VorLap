#!/usr/bin/env python3
"""Build the seven-profile Bidadi et al. FFA-W3 VorLap spectral bundle.

The input is the full set of force histories used for the 360-degree,
Re=10^7 FFA-W3 study.  The raw histories stay outside Git; this builder freezes
their relative paths and SHA-256 identities and writes portable HDF5 products.
The Reynolds-axis replication is an explicit engineering assumption, not
Reynolds-resolved CFD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from process_naluout_to_h5 import main as convert_one

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("211", "241", "270", "301", "330", "360", "500")
EXPECTED_AOA_DEG = (
    -180,
    -175,
    -170,
    -165,
    -160,
    -152,
    -144,
    -136,
    -128,
    -120,
    -110,
    -100,
    -90,
    -80,
    -70,
    -60,
    -50,
    -40,
    -33,
    -30,
    -27,
    -24,
    -21,
    -18,
    -15,
    -12,
    -9,
    -6,
    -3,
    0,
    3,
    6,
    9,
    12,
    15,
    18,
    21,
    24,
    27,
    30,
    33,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    128,
    136,
    144,
    152,
    160,
    165,
    170,
    175,
    180,
)
REPLICATED_RE_GRID = (5.0e4, 1.0e7, 2.0e7)
REPLICATION_JUSTIFICATION = (
    "The published source resolves one Re=1e7 plane. For the IEA-10-MW "
    "engineering verification, those fixed-body spectra are held Reynolds-"
    "independent over Re=5e4..2e7; this supplies coverage but no Reynolds "
    "trend and is not Reynolds-resolved validation."
)
SCHEMA = "vorlap-bidadi2024-ffa-w3-spectral-bundle-v1"
SOURCE_ARCHIVE_DOI = "10.5281/zenodo.12706820"
SOURCE_ARCHIVE_FILENAME = "ffa_data_files_ftt_160.zip"
SOURCE_ARCHIVE_MD5 = "fe58ef6d357de9402d1d3ac1a84676d6"
FINAL_WINDOW_TIMESTEP_COUNT = 8000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _aoa_from_path(path: Path, profile: str) -> int:
    prefix = f"ffa_w3_{profile}_"
    if not path.name.startswith(prefix) or path.suffix.lower() != ".dat":
        raise ValueError(f"Unexpected source filename: {path.name}")
    return int(path.stem[len(prefix) :])


def source_inventory(source_root: Path) -> list[dict[str, object]]:
    root = Path(source_root).resolve()
    rows: list[dict[str, object]] = []
    for profile in PROFILES:
        directory = root / f"ffa_w3_{profile}" / "data_files"
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing FFA source directory: {directory}")
        files = sorted(
            directory.glob("*.dat"), key=lambda path: _aoa_from_path(path, profile)
        )
        angles = tuple(_aoa_from_path(path, profile) for path in files)
        if angles != EXPECTED_AOA_DEG:
            raise ValueError(
                f"ffa_w3_{profile} AOA grid differs from the frozen 59-angle grid."
            )
        for path, angle in zip(files, angles):
            header = path.open("r", encoding="utf-8").readline().split()
            if header[:10] != [
                "Time",
                "Fpx",
                "Fpy",
                "Fpz",
                "Fvx",
                "Fvy",
                "Fvz",
                "Mtx",
                "Mty",
                "Mtz",
            ]:
                raise ValueError(f"Unexpected force-history header: {path}")
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "profile": f"FFA-W3-{profile}",
                    "thickness_ratio": int(profile) / 1000.0,
                    "aoa_deg": angle,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def build(source_root: Path, output_dir: Path) -> Path:
    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir).resolve()
    inventory = source_inventory(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    products: list[dict[str, object]] = []
    for profile in PROFILES:
        name = f"ffa_w3_{profile}"
        airfoil_id = f"FFA_W3_{profile}"
        output = output_dir / f"{name}.h5"
        summary = convert_one(
            [
                "--input-dir",
                str(source_root / name),
                "--source-path-root",
                str(source_root),
                "--output",
                str(output),
                "--airfoil-name",
                airfoil_id,
                "--thickness",
                str(int(profile) / 1000.0),
                "--re",
                "1e7",
                "--replicate-re-grid",
                "5e4 1e7 2e7",
                "--replicate-re-justification",
                REPLICATION_JUSTIFICATION,
                "--n-freq",
                "200",
                "--low-freq-skip",
                "6",
                # The primary processing retains the final 8,000 samples.
                # Selecting from the end also treats the 17 shorter archived
                # histories consistently.
                "--initial-timestep-skip",
                "0",
                "--final-timestep-count",
                str(FINAL_WINDOW_TIMESTEP_COUNT),
                "--periodic-aoa",
                "--no-plots",
            ]
        )
        products.append(
            {
                "path": output.name,
                "sha256": sha256_file(output),
                "bytes": output.stat().st_size,
                "airfoil_name": airfoil_id,
                "source_data_sha256": summary["source_data_sha256"],
                "source_files": summary["source_files"],
            }
        )
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "source": {
            "study": "Bidadi et al. (2024), Energies 17, 4328",
            "doi": "10.3390/en17174328",
            "dataset_doi": SOURCE_ARCHIVE_DOI,
            "archive_filename": SOURCE_ARCHIVE_FILENAME,
            "archive_md5": SOURCE_ARCHIVE_MD5,
            "raw_history_count": len(inventory),
            "relative_file_inventory_sha256": semantic_sha256(inventory),
            "files": inventory,
        },
        "processing": {
            "generator": "scripts/process_naluout_to_h5.py",
            "force_normalization": "0.5*rho*U^2*chord*span",
            "rho_kg_m3": 1.2,
            "dynamic_viscosity_pa_s": 9.0e-6,
            "source_reynolds": 1.0e7,
            "source_record_flow_through_times": 160,
            "retained_final_samples": FINAL_WINDOW_TIMESTEP_COUNT,
            "initial_timestep_skip": 0,
            "low_frequency_bins_skipped": 6,
            "frequency_entries_including_dc": 200,
            "aoa_periodic": True,
            "reynolds_model": "replicated_reynolds_independent",
            "output_reynolds": list(REPLICATED_RE_GRID),
            "reynolds_replication_justification": REPLICATION_JUSTIFICATION,
        },
        "products": products,
        "claim_boundary": {
            "static_fixed_body_spectrum": True,
            "force_motion_phase_identified": False,
            "reynolds_dependence_identified": False,
            "use": "engineering spectral forcing and lock-in frequency support",
        },
    }
    manifest["bundle_fingerprint"] = semantic_sha256(manifest)
    path = output_dir / "bundle_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "airfoils" / "bidadi2024_ffa_w3_spectral",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> Path:
    args = parse_args(argv)
    path = build(args.source_root, args.output_dir)
    print(path)
    return path


if __name__ == "__main__":
    main()
