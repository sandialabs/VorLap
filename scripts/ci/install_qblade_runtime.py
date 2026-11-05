"""Install an approved archive; never fetch an unverified or floating runtime."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile


def unpack(archive: Path, destination: Path):
    """Extract only bounded regular files/directories, without archive links."""
    total = 0
    seen = set()

    def target(name, size):
        nonlocal total
        parts = PurePosixPath(name).parts
        if not parts or name.startswith("/") or ".." in parts or "\\" in name:
            raise ValueError("Unsafe runtime archive path")
        canonical_name = PurePosixPath(*parts).as_posix()
        if canonical_name in seen:
            raise ValueError("Duplicate runtime archive path")
        seen.add(canonical_name)
        total += size
        if size < 0 or total > 2_000_000_000:
            raise ValueError("Runtime archive exceeds the 2 GB unpacked limit")
        return destination.joinpath(*parts)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("Runtime archive contains a symbolic link")
                path = target(member.filename, member.file_size)
                if member.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as source, path.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    path.chmod(0o755 if mode & 0o111 else 0o644)
    else:
        with tarfile.open(archive) as t:
            for member in t:
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Runtime archive contains a link or special file")
                path = target(member.name, member.size)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(member) as source, path.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    path.chmod(0o755 if member.mode & 0o111 else 0o644)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--archive", type=Path, help="An already downloaded approved archive")
    args = parser.parse_args()
    digest = os.environ.get("QBLADE_CI_ARCHIVE_SHA256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        parser.error("Set QBLADE_CI_ARCHIVE_SHA256 to the approved archive's exact SHA-256.")
    if args.destination.exists():
        parser.error("Refusing to replace an existing runtime directory.")
    with tempfile.TemporaryDirectory() as temp:
        archive = args.archive
        if archive is None:
            url = os.environ.get("QBLADE_CI_ARCHIVE_URL", "")
            if not url.startswith("https://"):
                parser.error("Set QBLADE_CI_ARCHIVE_URL to an approved noninteractive HTTPS archive URL.")
            archive = Path(temp) / "runtime.archive"
            try:
                with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
                    if not response.url.startswith("https://"):
                        raise ValueError("Runtime download redirected away from HTTPS")
                    total = 0
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > 256_000_000:
                            raise ValueError("Runtime download exceeds 256 MB")
                        output.write(block)
            except Exception:
                # URLs may carry access tokens. Never include them in public logs.
                raise RuntimeError("Could not download the approved QBlade archive") from None
        if archive.stat().st_size > 256_000_000:
            raise ValueError("Runtime archive exceeds 256 MB")
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise ValueError("QBlade archive SHA-256 does not match the approved value")
        staging = Path(temp) / "unpacked"
        staging.mkdir()
        unpack(archive, staging)
        roots = list(staging.rglob("libQBladeCE_2.0.9.7.so.1.0.0"))
        if len(roots) != 1:
            raise ValueError("Archive must contain exactly one qualified QBlade installation")
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(roots[0].parent), args.destination)
        # Always build the bridge from this checkout, never use an archived one.
        (args.destination / "ControllerFiles/libvorlap_qblade_bridge.so").unlink(missing_ok=True)
    print("Installed approved QBlade archive; verify components and build the current bridge next.")


if __name__ == "__main__":
    main()
