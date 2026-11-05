#!/usr/bin/env python3
"""Fail-closed acquisition of the pinned 4TU cylinder-wake archive.

The module intentionally uses only the Python standard library.  It audits the
ZIP central directory but never extracts or decompresses dataset members.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from typing import Any, Callable, Dict, Optional
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "acquisition_manifest.json"
DEFAULT_RAW_DIR = ROOT / "raw"

SCHEMA = "vorlap-reference-acquisition-v1"
PINNED_DOI = "10.4121/77985fa6-c546-4450-9d84-14b85f2d7e86.v1"
PINNED_FILE_ID = "0516b70d-47b7-4008-b2ab-a8f77e2507d3"
PINNED_URL = (
    "https://data.4tu.nl/file/77985fa6-c546-4450-9d84-14b85f2d7e86/"
    "0516b70d-47b7-4008-b2ab-a8f77e2507d3"
)
PINNED_FILENAME = "Oscillating_cylinder_dataset.zip"
PINNED_BYTES = 6_261_659_558
PINNED_MD5 = "bed51060b5ce4ea0c36d00e41c8a8b99"
PINNED_LICENSE = "CC-BY-NC-4.0"
PRINCIPAL_MAT_BASENAME = "oscillating_cylinder_benchmark_dataset_v4.mat"
MINIMUM_FREE_BYTES = 20 * 1024**3
RECEIPT_FILENAME = "acquisition_receipt.json"
RECEIPT_SCHEMA = "vorlap-reference-acquisition-receipt-v1"
DEFAULT_CHUNK_BYTES = 8 * 1024**2
CONTENT_RANGE_RE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class AcquisitionError(RuntimeError):
    """Raised when an acquisition or provenance gate fails."""


def _nested(payload: Dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise AcquisitionError("manifest is missing " + ".".join(keys))
        value = value[key]
    return value


def _require_exact(payload: Dict[str, Any], keys: tuple, expected: Any) -> None:
    observed = _nested(payload, *keys)
    if observed != expected:
        location = ".".join(keys)
        raise AcquisitionError(
            "manifest pin mismatch for {}: expected {!r}, observed {!r}".format(
                location, expected, observed
            )
        )


def validate_manifest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate every identity field that is allowed to drive acquisition."""

    pins = {
        ("schema",): SCHEMA,
        ("record", "version"): 1,
        ("record", "doi"): PINNED_DOI,
        ("asset", "file_id"): PINNED_FILE_ID,
        ("asset", "url"): PINNED_URL,
        ("asset", "filename"): PINNED_FILENAME,
        ("asset", "bytes"): PINNED_BYTES,
        ("asset", "md5"): PINNED_MD5,
        ("asset", "sha256"): None,
        ("license", "spdx"): PINNED_LICENSE,
        ("license", "commercial_use_permitted"): False,
        ("license", "repository_mit_license_applies_to_data"): False,
        ("acquisition", "minimum_free_bytes"): MINIMUM_FREE_BYTES,
        ("acquisition", "partial_suffix"): ".part",
        ("acquisition", "receipt_filename"): RECEIPT_FILENAME,
        ("acquisition", "automatic_extraction"): False,
        ("zip_audit", "principal_mat_basename"): PRINCIPAL_MAT_BASENAME,
    }
    if not isinstance(payload, dict):
        raise AcquisitionError("manifest root must be a JSON object")
    for keys, expected in pins.items():
        _require_exact(payload, keys, expected)
    md5_value = _nested(payload, "asset", "md5")
    if not re.fullmatch(r"[0-9a-f]{32}", md5_value):
        raise AcquisitionError("manifest MD5 must be 32 lowercase hexadecimal digits")
    return payload


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError("cannot read manifest {}: {}".format(path, exc)) from exc
    return validate_manifest(payload)


def _existing_anchor(path: Path) -> Path:
    candidate = Path(path).resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise AcquisitionError("no existing filesystem anchor for {}".format(path))
        candidate = parent
    return candidate


def require_free_space(
    path: Path,
    minimum_bytes: int = MINIMUM_FREE_BYTES,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
) -> int:
    anchor = _existing_anchor(Path(path))
    free_bytes = int(disk_usage(str(anchor)).free)
    if free_bytes < minimum_bytes:
        raise AcquisitionError(
            "free-space gate failed at {}: {} bytes available; {} required".format(
                anchor, free_bytes, minimum_bytes
            )
        )
    return free_bytes


def archive_paths(manifest: Dict[str, Any], raw_dir: Path) -> Dict[str, Path]:
    filename = _nested(manifest, "asset", "filename")
    final_path = Path(raw_dir) / filename
    return {
        "final": final_path,
        "partial": Path(str(final_path) + _nested(manifest, "acquisition", "partial_suffix")),
        "receipt": Path(raw_dir) / _nested(
            manifest, "acquisition", "receipt_filename"
        ),
    }


def dry_run(
    manifest_path: Path = DEFAULT_MANIFEST,
    raw_dir: Path = DEFAULT_RAW_DIR,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
) -> Dict[str, Any]:
    """Return an acquisition plan without network access or filesystem writes."""

    manifest = load_manifest(manifest_path)
    paths = archive_paths(manifest, raw_dir)
    free_bytes = require_free_space(raw_dir, disk_usage=disk_usage)
    partial_bytes = paths["partial"].stat().st_size if paths["partial"].exists() else 0
    if partial_bytes > PINNED_BYTES:
        raise AcquisitionError("partial archive is larger than the pinned archive")
    return {
        "command": "dry-run",
        "network_performed": False,
        "filesystem_writes": False,
        "doi": PINNED_DOI,
        "url": PINNED_URL,
        "archive": str(paths["final"]),
        "partial": str(paths["partial"]),
        "expected_bytes": PINNED_BYTES,
        "partial_bytes": partial_bytes,
        "remaining_bytes": PINNED_BYTES - partial_bytes,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "available_free_bytes": free_bytes,
        "final_exists": paths["final"].exists(),
        "license": PINNED_LICENSE,
        "automatic_extraction": False,
    }


def _hash_file(path: Path, algorithm: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_integrity(
    path: Path,
    expected_bytes: int,
    expected_md5: str,
    hash_file: Callable[[Path, str], str] = _hash_file,
) -> Dict[str, Any]:
    """Check exact size and official MD5 before computing a local SHA-256."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise AcquisitionError("archive must be an existing regular, non-symlink file")
    observed_bytes = candidate.stat().st_size
    if observed_bytes != expected_bytes:
        raise AcquisitionError(
            "archive size mismatch: expected {}, observed {}".format(
                expected_bytes, observed_bytes
            )
        )
    observed_md5 = hash_file(candidate, "md5")
    if observed_md5.lower() != expected_md5.lower():
        raise AcquisitionError(
            "archive MD5 mismatch: expected {}, observed {}".format(
                expected_md5, observed_md5
            )
        )
    observed_sha256 = hash_file(candidate, "sha256")
    return {
        "bytes": observed_bytes,
        "md5": observed_md5.lower(),
        "sha256": observed_sha256.lower(),
        "hash_order": ["size", "md5", "sha256"],
    }


def _safe_member_path(info: zipfile.ZipInfo) -> str:
    original = getattr(info, "orig_filename", info.filename)
    if "\x00" in original or "\x00" in info.filename:
        raise AcquisitionError("ZIP member contains a NUL byte")
    normalized_slashes = original.replace("\\", "/")
    if normalized_slashes.startswith("//") or WINDOWS_DRIVE_RE.match(normalized_slashes):
        raise AcquisitionError("ZIP member uses an absolute Windows path: {!r}".format(original))
    path = PurePosixPath(normalized_slashes)
    if path.is_absolute():
        raise AcquisitionError("ZIP member uses an absolute path: {!r}".format(original))
    if not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise AcquisitionError("ZIP member has an unsafe path: {!r}".format(original))
    canonical = "/".join(path.parts)
    if info.is_dir():
        canonical += "/"
    return canonical


def _reject_special_member(info: zipfile.ZipInfo) -> None:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if info.create_system == 3 and file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise AcquisitionError("ZIP member is a symlink or special file: {!r}".format(info.filename))


def audit_zip_members(
    path: Path,
    principal_basename: str = PRINCIPAL_MAT_BASENAME,
) -> Dict[str, Any]:
    """Audit ZIP metadata without extracting or decompressing member payloads."""

    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if not infos:
                raise AcquisitionError("ZIP archive has no members")
            seen = set()
            principal_members = []
            compressed_bytes = 0
            uncompressed_bytes = 0
            for info in infos:
                canonical = _safe_member_path(info)
                _reject_special_member(info)
                if canonical in seen:
                    raise AcquisitionError(
                        "ZIP archive has a duplicate normalized member: {!r}".format(canonical)
                    )
                seen.add(canonical)
                if not info.is_dir() and PurePosixPath(canonical).name == principal_basename:
                    principal_members.append(canonical)
                compressed_bytes += int(info.compress_size)
                uncompressed_bytes += int(info.file_size)
    except zipfile.BadZipFile as exc:
        raise AcquisitionError("archive is not a valid ZIP central directory") from exc
    if len(principal_members) != 1:
        raise AcquisitionError(
            "ZIP archive must contain exactly one {!r} basename; found {}".format(
                principal_basename, len(principal_members)
            )
        )
    return {
        "member_count": len(infos),
        "compressed_member_bytes": compressed_bytes,
        "uncompressed_member_bytes": uncompressed_bytes,
        "principal_mat_member": principal_members[0],
        "extracted": False,
    }


def verify_archive(
    archive: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    integrity = verify_file_integrity(
        archive,
        int(_nested(manifest, "asset", "bytes")),
        str(_nested(manifest, "asset", "md5")),
    )
    zip_audit = audit_zip_members(
        archive, str(_nested(manifest, "zip_audit", "principal_mat_basename"))
    )
    return {
        "command": "verify",
        "archive": str(Path(archive)),
        "network_performed": False,
        "integrity": integrity,
        "zip_audit": zip_audit,
        "license": _nested(manifest, "license", "spdx"),
    }


def _response_status(response: Any) -> Optional[int]:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    return int(status) if status is not None else None


def _validate_response_headers(response: Any, offset: int, expected_bytes: int) -> None:
    status = _response_status(response)
    content_range = response.headers.get("Content-Range")
    content_length = response.headers.get("Content-Length")
    if offset:
        if status != 206:
            raise AcquisitionError(
                "resume requires HTTP 206; server returned {!r}".format(status)
            )
        match = CONTENT_RANGE_RE.fullmatch(content_range or "")
        if match is None:
            raise AcquisitionError("resume response lacks a valid Content-Range")
        start, end, total = (int(value) for value in match.groups())
        if start != offset or total != expected_bytes or end != expected_bytes - 1:
            raise AcquisitionError(
                "resume Content-Range does not match the partial file and pinned size"
            )
    elif status not in (200, 206):
        raise AcquisitionError("fresh download requires HTTP 200 or 206")
    elif status == 206:
        match = CONTENT_RANGE_RE.fullmatch(content_range or "")
        if match is None:
            raise AcquisitionError("partial response lacks a valid Content-Range")
        start, end, total = (int(value) for value in match.groups())
        if start != 0 or total != expected_bytes or end != expected_bytes - 1:
            raise AcquisitionError("fresh partial response has the wrong range")
    if content_length is not None:
        try:
            observed_length = int(content_length)
        except ValueError as exc:
            raise AcquisitionError("HTTP Content-Length is not an integer") from exc
        if observed_length != expected_bytes - offset:
            raise AcquisitionError(
                "HTTP Content-Length does not match the expected remaining bytes"
            )


def stream_download(
    url: str,
    partial_path: Path,
    expected_bytes: int,
    opener: Callable[..., Any] = urllib.request.urlopen,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Download or resume into a persistent partial file."""

    partial = Path(partial_path)
    if partial.is_symlink():
        raise AcquisitionError("partial archive may not be a symlink")
    if partial.exists() and not partial.is_file():
        raise AcquisitionError("partial archive path is not a regular file")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_bytes:
        raise AcquisitionError("partial archive is larger than the pinned archive")
    if offset == expected_bytes:
        return {"resumed_from_bytes": offset, "downloaded_bytes": 0}

    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "VorLap-reference-acquisition/1",
    }
    if offset:
        headers["Range"] = "bytes={}-".format(offset)
    request = urllib.request.Request(url, headers=headers, method="GET")
    mode = "ab" if partial.exists() else "xb"
    written = 0
    try:
        with opener(request, timeout=timeout_s) as response:
            _validate_response_headers(response, offset, expected_bytes)
            with partial.open(mode) as stream:
                while True:
                    chunk = response.read(chunk_bytes)
                    if not chunk:
                        break
                    if offset + written + len(chunk) > expected_bytes:
                        raise AcquisitionError("download response exceeds the pinned byte count")
                    stream.write(chunk)
                    written += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    except AcquisitionError:
        raise
    except Exception as exc:
        raise AcquisitionError(
            "download interrupted after {} newly written bytes; partial retained: {}".format(
                written, partial
            )
        ) from exc
    final_partial_bytes = partial.stat().st_size if partial.exists() else 0
    if final_partial_bytes != expected_bytes:
        raise AcquisitionError(
            "download ended at {} bytes; expected {}; partial retained".format(
                final_partial_bytes, expected_bytes
            )
        )
    return {"resumed_from_bytes": offset, "downloaded_bytes": written}


def _write_receipt(path: Path, payload: Dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise AcquisitionError("receipt temporary path is not a regular file")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    except OSError as exc:
        raise AcquisitionError("cannot atomically write receipt {}: {}".format(path, exc)) from exc


def _receipt_payload(
    archive: Path,
    verification: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a deterministic receipt from verified content, not transfer history."""

    integrity = verification["integrity"]
    return {
        "schema": RECEIPT_SCHEMA,
        "source": {
            "doi": _nested(manifest, "record", "doi"),
            "file_id": _nested(manifest, "asset", "file_id"),
            "url": _nested(manifest, "asset", "url"),
        },
        "archive": {
            "filename": Path(archive).name,
            "bytes": integrity["bytes"],
            "md5": integrity["md5"],
            "sha256": integrity["sha256"],
        },
        "zip_audit": verification["zip_audit"],
        "license": _nested(manifest, "license", "spdx"),
        "automatic_extraction": False,
    }


def acquire(
    manifest_path: Path = DEFAULT_MANIFEST,
    raw_dir: Path = DEFAULT_RAW_DIR,
    opener: Callable[..., Any] = urllib.request.urlopen,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    paths = archive_paths(manifest, raw_dir)

    if paths["final"].exists():
        result = verify_archive(paths["final"], manifest_path)
        _write_receipt(
            paths["receipt"],
            _receipt_payload(paths["final"], result, manifest),
        )
        result.update(
            {
                "command": "acquire",
                "downloaded_bytes": 0,
                "reused": True,
                "receipt": str(paths["receipt"]),
            }
        )
        return result

    require_free_space(raw_dir, disk_usage=disk_usage)
    try:
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AcquisitionError(
            "cannot create raw directory {}: {}".format(raw_dir, exc)
        ) from exc
    download = stream_download(
        str(_nested(manifest, "asset", "url")),
        paths["partial"],
        int(_nested(manifest, "asset", "bytes")),
        opener=opener,
    )
    verification = verify_archive(paths["partial"], manifest_path)
    if paths["final"].exists():
        raise AcquisitionError("refusing to overwrite an archive created during acquisition")
    os.replace(str(paths["partial"]), str(paths["final"]))
    result = {
        "command": "acquire",
        "archive": str(paths["final"]),
        "network_performed": bool(download["downloaded_bytes"]),
        "downloaded_bytes": download["downloaded_bytes"],
        "resumed_from_bytes": download["resumed_from_bytes"],
        "reused": False,
        "integrity": verification["integrity"],
        "zip_audit": verification["zip_audit"],
        "license": _nested(manifest, "license", "spdx"),
    }
    _write_receipt(
        paths["receipt"],
        _receipt_payload(paths["final"], verification, manifest),
    )
    result["receipt"] = str(paths["receipt"])
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("dry-run", "verify", "acquire"), help="explicit action"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="existing archive for verify only; defaults to the pinned raw path",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "dry-run":
            if args.archive is not None:
                raise AcquisitionError("--archive is valid only with verify")
            result = dry_run(args.manifest, args.raw_dir)
        elif args.command == "verify":
            manifest = load_manifest(args.manifest)
            default_archive = archive_paths(manifest, args.raw_dir)["final"]
            result = verify_archive(args.archive or default_archive, args.manifest)
        else:
            if args.archive is not None:
                raise AcquisitionError("--archive is valid only with verify")
            result = acquire(args.manifest, args.raw_dir)
    except AcquisitionError as exc:
        print("acquisition error: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
