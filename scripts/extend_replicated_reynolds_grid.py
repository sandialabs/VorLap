#!/usr/bin/env python3
"""Extend an approved replicated-Re HDF5 envelope without recomputing spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import h5py
import numpy as np


EXTENSION_GENERATOR = "scripts/extend_replicated_reynolds_grid.py"
SPECTRAL_SUFFIXES = ("_Amp", "_Pha", "_ST")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_increasing_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.replace(",", " ").split())
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("Reynolds grid must contain positive finite values.")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("Reynolds grid must be strictly increasing.")
    return values


def _json_attr(h5: h5py.File, name: str) -> dict[str, object]:
    raw = h5.attrs.get(name)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    parsed = json.loads(str(raw))
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    return parsed


def _float_tuple(values: object, *, label: str) -> tuple[float, ...]:
    try:
        return tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric sequence.") from exc


def _validate_parent(
    parent: Path,
    expected_parent_sha256: str,
) -> tuple[str, tuple[float, ...], dict[str, object], tuple[str, ...]]:
    actual_sha256 = sha256_file(parent)
    if actual_sha256 != expected_parent_sha256:
        raise ValueError(
            f"Parent HDF5 SHA-256 mismatch: expected {expected_parent_sha256}, "
            f"observed {actual_sha256}."
        )

    with h5py.File(parent, "r") as h5:
        if str(h5.attrs.get("reynolds_model", "")) != "replicated_reynolds_independent":
            raise ValueError("Parent HDF5 is not declared replicated_reynolds_independent.")
        if "Re" not in h5:
            raise ValueError("Parent HDF5 has no Re dataset.")
        parent_grid = tuple(float(value) for value in np.asarray(h5["Re"][:]).reshape(-1))
        processing = _json_attr(h5, "processing_config_json")
        if processing.get("reynolds_model") != "replicated_reynolds_independent":
            raise ValueError("Parent processing provenance is not Reynolds-independent.")
        if _float_tuple(processing.get("output_reynolds"), label="output_reynolds") != parent_grid:
            raise ValueError("Parent Re dataset disagrees with processing output_reynolds.")
        if _float_tuple(processing.get("replicate_re_grid"), label="replicate_re_grid") != parent_grid:
            raise ValueError("Parent Re dataset disagrees with processing replicate_re_grid.")
        if _float_tuple(processing.get("source_reynolds"), label="source_reynolds") != (500_000.0,):
            raise ValueError("Approved envelope extension requires a single Re=500,000 source.")

        spectral_names = tuple(sorted(name for name in h5 if name.endswith(SPECTRAL_SUFFIXES)))
        if not spectral_names:
            raise ValueError("Parent HDF5 contains no recognized spectral datasets.")
        for name in spectral_names:
            values = h5[name][...]
            if values.ndim < 1 or values.shape[0] != len(parent_grid):
                raise ValueError(f"Spectral dataset {name} does not align with the parent Re grid.")
            if any(not np.array_equal(values[0], values[index]) for index in range(1, len(parent_grid))):
                raise ValueError(f"Spectral dataset {name} is not exactly replicated across Re.")

    return actual_sha256, parent_grid, processing, spectral_names


def extend_replicated_reynolds_grid(
    *,
    parent: Path,
    output: Path,
    expected_parent_sha256: str,
    reynolds: Sequence[float],
    justification: str,
    overwrite: bool = False,
) -> dict[str, object]:
    parent = parent.expanduser().resolve()
    output = output.expanduser().resolve()
    expected_parent_sha256 = expected_parent_sha256.strip().lower()
    if len(expected_parent_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_parent_sha256
    ):
        raise ValueError("Expected parent SHA-256 must be a 64-character lowercase hex digest.")
    if not parent.is_file():
        raise FileNotFoundError(f"Parent HDF5 does not exist: {parent}")
    if not justification.strip():
        raise ValueError("A nonempty Reynolds-envelope justification is required.")
    new_grid = tuple(float(value) for value in reynolds)
    if not new_grid or any(not np.isfinite(value) or value <= 0.0 for value in new_grid):
        raise ValueError("Reynolds grid must contain positive finite values.")
    if any(right <= left for left, right in zip(new_grid, new_grid[1:])):
        raise ValueError("Reynolds grid must be strictly increasing.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output}")

    parent_sha256, parent_grid, processing, spectral_names = _validate_parent(
        parent,
        expected_parent_sha256,
    )
    if len(new_grid) != len(parent_grid):
        raise ValueError(
            "Envelope-only extension requires the new Re grid to retain the parent grid length."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(parent, temporary)
        with h5py.File(temporary, "r+") as h5:
            h5["Re"][...] = np.asarray(new_grid, dtype=h5["Re"].dtype)
            updated_processing = dict(processing)
            updated_processing.update(
                {
                    "output_reynolds": list(new_grid),
                    "replicate_re_grid": list(new_grid),
                    "replicate_re_justification": justification.strip(),
                    "replicate_re_parent_sha256": parent_sha256,
                    "reynolds_envelope_extension_exact": True,
                    "reynolds_envelope_extension_generator": EXTENSION_GENERATOR,
                    "reynolds_envelope_parent_grid": list(parent_grid),
                }
            )
            h5.attrs["processing_config_json"] = json.dumps(updated_processing, sort_keys=True)
            h5.attrs["reynolds_replication_justification"] = justification.strip()
            h5.attrs["reynolds_replication_parent_sha256"] = parent_sha256
            h5.attrs["reynolds_envelope_extension_exact"] = True
            h5.attrs["reynolds_envelope_extension_generator"] = EXTENSION_GENERATOR
            h5.attrs["reynolds_envelope_parent_grid_json"] = json.dumps(list(parent_grid))

        with h5py.File(parent, "r") as source, h5py.File(temporary, "r") as revised:
            if tuple(float(value) for value in revised["Re"][:]) != new_grid:
                raise ValueError("Revised HDF5 did not retain the requested Re grid exactly.")
            changed_non_re = [
                name
                for name in source
                if name != "Re" and not np.array_equal(source[name][...], revised[name][...])
            ]
            if changed_non_re:
                raise ValueError(
                    "Envelope-only extension changed non-Re datasets: " + ", ".join(changed_non_re)
                )
            if set(source) != set(revised):
                raise ValueError("Envelope-only extension changed the HDF5 dataset inventory.")

        revised_sha256 = sha256_file(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "output": str(output),
        "output_sha256": revised_sha256,
        "parent": str(parent),
        "parent_sha256": parent_sha256,
        "parent_reynolds": list(parent_grid),
        "reynolds": list(new_grid),
        "spectral_datasets_verified": list(spectral_names),
        "non_re_datasets_preserved_exactly": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--reynolds", type=positive_increasing_grid, required=True)
    parser.add_argument("--justification", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    report = extend_replicated_reynolds_grid(
        parent=args.parent,
        output=args.output,
        expected_parent_sha256=args.expected_parent_sha256,
        reynolds=args.reynolds,
        justification=args.justification,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
