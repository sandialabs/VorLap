"""Verify the upstream solver and freshly built bridge against the qualified stack."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vorlap.qblade_run_support import qblade_runtime_dependency_manifest

MANIFEST = Path(__file__).resolve().parents[1] / "reproducibility/qblade-ci-runtime.json"


def verify_runtime(root: Path, *, without_bridge: bool = False) -> dict:
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (platform.system(), platform.machine()) != (expected["platform"], expected["machine"]):
        raise RuntimeError("The qualified QBlade tests require Linux x86_64 (Docker is supported).")
    identities = {}
    for name, digest in expected["files"].items():
        if without_bridge and name.startswith("ControllerFiles/"):
            continue
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular runtime file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Runtime identity mismatch for {name}: expected {digest}, observed {actual}")
        identities[name] = actual
    dependencies, fingerprint = qblade_runtime_dependency_manifest(root)
    if fingerprint != expected["dependency_fingerprint"]:
        raise ValueError(f"QBlade dependency fingerprint mismatch: {fingerprint}")
    return {"version": expected["version"], "files": identities,
            "dependency_fingerprint": fingerprint, "dependencies": dependencies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--without-bridge", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify_runtime(args.root, without_bridge=args.without_bridge)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {result['version']}: {len(result['files'])} components and {len(result['dependencies'])} dependencies")


if __name__ == "__main__":
    main()
