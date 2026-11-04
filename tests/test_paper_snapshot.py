"""Artifact integrity checks run without the separate manuscript or QBlade."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts.verify_paper_snapshot import verify_snapshot


@pytest.fixture
def snapshot(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    data = b"compact source\n"
    (paper / "source.txt").write_bytes(data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{
        "path": "source.txt", "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "mode": 0o100644,
    }]}))
    return paper, manifest


def test_snapshot_rejects_same_size_content_change(snapshot):
    paper, manifest = snapshot
    assert len(verify_snapshot(paper, manifest)["files"]) == 1
    (paper / "source.txt").write_bytes(b"changed source\n")
    with pytest.raises(ValueError, match="changed: source.txt"):
        verify_snapshot(paper, manifest)


def test_snapshot_rejects_path_escape(snapshot):
    paper, manifest = snapshot
    data = json.loads(manifest.read_text())
    data["files"][0]["path"] = "../outside.txt"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid snapshot path"):
        verify_snapshot(paper, manifest)


def test_snapshot_rejects_symlink_source(snapshot, tmp_path):
    paper, manifest = snapshot
    source = paper / "source.txt"
    outside = tmp_path / "outside.txt"
    source.rename(outside)
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes its root"):
        verify_snapshot(paper, manifest)


def test_snapshot_archive_is_deterministic_and_preserves_payload(snapshot, tmp_path):
    paper, manifest = snapshot
    script = Path(__file__).resolve().parents[1] / "scripts/verify_paper_snapshot.py"
    outputs = [tmp_path / "first.zip", tmp_path / "second.zip"]
    for output in outputs:
        subprocess.run([
            sys.executable, str(script), "--paper-dir", str(paper),
            "--manifest", str(manifest), "--archive", str(output),
        ], check=True, capture_output=True, text=True)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        assert output.with_suffix(".zip.sha256").read_text() == f"{digest}  {output.name}\n"
        with zipfile.ZipFile(output) as archive:
            assert archive.namelist() == ["source.txt"]
            assert archive.read("source.txt") == (paper / "source.txt").read_bytes()
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
