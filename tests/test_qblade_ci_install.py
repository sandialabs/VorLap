"""The CI installer must preserve bytes and reject archive path/link attacks."""
import io
import tarfile
import zipfile

import pytest

from scripts.ci.install_qblade_runtime import unpack


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_runtime_archive_preserves_exact_payload(tmp_path, kind):
    archive = tmp_path / f"runtime.{kind}"
    payload = b"\x7fELF\x00qualified vendor bytes\xff"
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("runtime/Libraries/solver.so", payload)
    else:
        with tarfile.open(archive, "w") as t:
            item = tarfile.TarInfo("runtime/Libraries/solver.so")
            item.size = len(payload)
            t.addfile(item, io.BytesIO(payload))
    destination = tmp_path / "extracted"
    unpack(archive, destination)
    assert (destination / "runtime/Libraries/solver.so").read_bytes() == payload


@pytest.mark.parametrize("name", ["../escaped", "/absolute", "runtime/../../escaped", "runtime\\escaped"])
def test_runtime_zip_cannot_escape_staging_directory(tmp_path, name):
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, b"not executable")
    with pytest.raises(ValueError, match="Unsafe"):
        unpack(archive, tmp_path / "staging")
    assert not (tmp_path / "escaped").exists()


def test_runtime_tar_cannot_install_symbolic_links(tmp_path):
    archive = tmp_path / "runtime.tar"
    with tarfile.open(archive, "w") as t:
        link = tarfile.TarInfo("runtime/linked")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        t.addfile(link)
    with pytest.raises(ValueError, match="link or special file"):
        unpack(archive, tmp_path / "staging")


def test_runtime_zip_cannot_overwrite_through_alias_paths(tmp_path):
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("runtime/solver.so", b"qualified")
        z.writestr("runtime/./solver.so", b"changed")
    with pytest.raises(ValueError, match="Duplicate"):
        unpack(archive, tmp_path / "staging")
