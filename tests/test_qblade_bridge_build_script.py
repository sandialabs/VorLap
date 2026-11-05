from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_linux_bridge_build_script_removes_path_mismatched_cmake_cache(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeFiles").mkdir()
    (build_dir / "CMakeFiles" / "stale").write_text("old generator state\n", encoding="utf-8")
    (build_dir / "CMakeCache.txt").write_text(
        "CMAKE_HOME_DIRECTORY:INTERNAL=/old/checkout/qblade_external_bridge\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"-c\" ]]; then\n"
        "  printf 'Python executable: fake\\nPython version: 3.test\\nNumPy version: fake\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_cmake = fake_bin / "cmake"
    fake_cmake.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_CMAKE_LOG\"\n"
        "if [[ \"${1:-}\" == \"-S\" ]]; then\n"
        "  build_dir=''\n"
        "  while [[ $# -gt 0 ]]; do\n"
        "    if [[ \"$1\" == \"-B\" ]]; then\n"
        "      shift\n"
        "      build_dir=\"$1\"\n"
        "    fi\n"
        "    shift || true\n"
        "  done\n"
        "  mkdir -p \"$build_dir\"\n"
        "  printf 'CMAKE_HOME_DIRECTORY:INTERNAL=%s/qblade_external_bridge\\n' \"$REPO_ROOT\" > \"$build_dir/CMakeCache.txt\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"--build\" ]]; then\n"
        "  build_dir=\"$2\"\n"
        "  mkdir -p \"$build_dir\"\n"
        "  printf 'fake shared object\\n' > \"$build_dir/libvorlap_qblade_bridge.so\"\n"
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    fake_cmake.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
            "PYTHON_EXE": str(fake_python),
            "FAKE_CMAKE_LOG": str(tmp_path / "cmake.log"),
            "REPO_ROOT": str(REPO),
        }
    )

    completed = subprocess.run(
        ["bash", str(REPO / "scripts" / "build_qblade_external_linux.sh"), str(build_dir)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "Removing stale CMake cache from different source path" in completed.stdout
    assert not (build_dir / "CMakeFiles" / "stale").exists()
    assert (build_dir / "libvorlap_qblade_bridge.so").is_file()
    assert (tmp_path / "cmake.log").read_text(encoding="utf-8").count(str(build_dir)) >= 2
