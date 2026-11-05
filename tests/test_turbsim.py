from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import pytest

from vorlap.turbsim import (
    expected_output_flags,
    generate_turbsim_wind_file,
    quote_turbsim_string,
    render_turbsim_input,
    turbsim_metadata_path,
    write_turbsim_input,
)


def minimal_template() -> str:
    return "\n".join(
        [
            "False        WrADFF          - binary AeroDyn full-field output",
            "True         WrBLFF          - binary BLADED full-field output",
            "0.01         TimeStep        - time step",
            '"default"    TurbModel       - turbulence model',
            "",
        ]
    )


def test_render_turbsim_input_preserves_keyword_comments() -> None:
    rendered = render_turbsim_input(
        minimal_template(),
        {
            "WrADFF": True,
            "WrBLFF": False,
            "TimeStep": 0.0075,
            "TurbModel": quote_turbsim_string("IECKAI"),
        },
    )

    assert "True" in rendered
    assert "WrADFF" in rendered
    assert "- binary AeroDyn full-field output" in rendered
    assert "False" in rendered
    assert "WrBLFF" in rendered
    assert "0.0075" in rendered
    assert '"IECKAI"' in rendered


def test_expected_output_flags_match_qblade_wind_format() -> None:
    assert expected_output_flags("wind.bts") == {"WrADFF": True, "WrBLFF": False}
    assert expected_output_flags("wind.wnd") == {"WrADFF": False, "WrBLFF": True}


def test_generate_turbsim_wind_file_skips_existing_without_executable(tmp_path: Path) -> None:
    template = tmp_path / "template.inp"
    template.write_text(minimal_template(), encoding="utf-8")
    wind = tmp_path / "winds" / "U05p00ms" / "dlc6p4_U05p00ms.bts"
    wind.parent.mkdir(parents=True)
    wind.write_bytes(b"existing")
    settings = {"TimeStep": 0.0075, "TurbModel": quote_turbsim_string("IECKAI")}
    write_turbsim_input(
        wind.with_suffix(".inp"),
        template_path=template,
        settings={**settings, **expected_output_flags(wind)},
    )

    run = generate_turbsim_wind_file(
        wind_file=wind,
        settings=settings,
        template_path=template,
        adopt_existing=True,
    )

    assert run.generated is False
    assert run.skipped_existing is True
    assert run.command == []
    assert run.input_file.is_file()
    assert run.settings["WrADFF"] is True
    assert run.settings["WrBLFF"] is False
    assert run.provenance_status == "adopted_existing_exact_input"
    assert turbsim_metadata_path(wind).is_file()


def test_generate_turbsim_refuses_untracked_existing_output(tmp_path: Path) -> None:
    template = tmp_path / "template.inp"
    template.write_text(minimal_template(), encoding="utf-8")
    wind = tmp_path / "wind.bts"
    wind.write_bytes(b"untracked")

    with pytest.raises(ValueError, match="Refusing to reuse untracked"):
        generate_turbsim_wind_file(
            wind_file=wind,
            settings={"TimeStep": 0.0075, "TurbModel": quote_turbsim_string("IECKAI")},
            template_path=template,
        )


def test_generate_turbsim_wind_file_uses_executable_and_verifies_output(tmp_path: Path) -> None:
    template = tmp_path / "template.inp"
    template.write_text(minimal_template(), encoding="utf-8")
    executable = tmp_path / "turbsim"
    executable.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "out=\"${1%.inp}.bts\"",
                "printf 'generated wind' > \"$out\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    wind = tmp_path / "winds" / "U05p00ms" / "dlc6p4_U05p00ms.bts"

    run = generate_turbsim_wind_file(
        wind_file=wind,
        settings={"TimeStep": 0.0075, "TurbModel": quote_turbsim_string("IECKAI")},
        executable=executable,
        template_path=template,
    )

    assert run.generated is True
    assert run.skipped_existing is False
    assert run.returncode == 0
    assert run.command == [str(executable.resolve()), run.input_file.name]
    assert wind.read_bytes() == b"generated wind"
    assert run.metadata_file.is_file()

    reused = generate_turbsim_wind_file(
        wind_file=wind,
        settings={"TimeStep": 0.0075, "TurbModel": quote_turbsim_string("IECKAI")},
        template_path=template,
    )
    assert reused.skipped_existing is True
    assert reused.provenance_status == "generated"

    with pytest.raises(ValueError, match="different settings") as mismatch:
        generate_turbsim_wind_file(
            wind_file=wind,
            settings={"TimeStep": 0.01, "TurbModel": quote_turbsim_string("IECKAI")},
            template_path=template,
        )
    assert "settings.TimeStep: existing=0.0075, requested=0.01" in str(mismatch.value)

    wind.write_bytes(b"tampered wind")
    with pytest.raises(ValueError, match="does not match"):
        generate_turbsim_wind_file(
            wind_file=wind,
            settings={"TimeStep": 0.0075, "TurbModel": quote_turbsim_string("IECKAI")},
            template_path=template,
        )


def test_parallel_turbsim_requests_generate_one_atomic_output(tmp_path: Path) -> None:
    template = tmp_path / "template.inp"
    template.write_text(minimal_template(), encoding="utf-8")
    executable = tmp_path / "turbsim"
    executable.write_text(
        "#!/bin/sh\n"
        "sleep 0.1\n"
        "out=\"${1%.inp}.bts\"\n"
        "printf 'generated once' > \"$out\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    wind = tmp_path / "winds" / "shared.bts"
    settings = {"TimeStep": 0.0075, "TurbModel": quote_turbsim_string("IECKAI")}

    def generate():
        return generate_turbsim_wind_file(
            wind_file=wind,
            settings=settings,
            executable=executable,
            template_path=template,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        runs = list(executor.map(lambda _index: generate(), range(2)))

    assert sum(run.generated for run in runs) == 1
    assert sum(run.skipped_existing for run in runs) == 1
    assert wind.read_bytes() == b"generated once"


def test_turbsim_empty_temporary_output_does_not_replace_existing_wind(tmp_path: Path) -> None:
    template = tmp_path / "template.inp"
    template.write_text(minimal_template(), encoding="utf-8")
    executable = tmp_path / "turbsim"
    executable.write_text(
        "#!/bin/sh\n"
        "out=\"${1%.inp}.bts\"\n"
        ": > \"$out\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    wind = tmp_path / "winds" / "shared.bts"
    wind.parent.mkdir(parents=True)
    wind.write_bytes(b"previous complete wind")

    with pytest.raises(FileNotFoundError, match="nonempty temporary wind file"):
        generate_turbsim_wind_file(
            wind_file=wind,
            settings={"TimeStep": 0.0075, "TurbModel": quote_turbsim_string("IECKAI")},
            executable=executable,
            template_path=template,
            force=True,
        )

    assert wind.read_bytes() == b"previous complete wind"
    assert not turbsim_metadata_path(wind).exists()


def test_installer_directly_installs_nonexecutable_built_target(tmp_path: Path) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_git.chmod(0o755)
    fake_cmake = fake_bin / "cmake"
    fake_cmake.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-S" ]]; then
  build=""
  while (($#)); do
    if [[ "$1" == "-B" ]]; then
      build="$2"
      break
    fi
    shift
  done
  mkdir -p "${build}/modules/turbsim"
  printf '#!/bin/sh\\nprintf \"fake TurbSim version 5.0.0\\\\n\"\\n' > "${build}/modules/turbsim/turbsim"
  chmod 644 "${build}/modules/turbsim/turbsim"
  exit 0
fi
if [[ "${1:-}" == "--build" ]]; then
  exit 0
fi
if [[ "${1:-}" == "--install" ]]; then
  touch "${CMAKE_GLOBAL_INSTALL_MARKER:?}"
  exit 99
fi
exit 2
""",
        encoding="utf-8",
    )
    fake_cmake.chmod(0o755)

    openfast_root = tmp_path / "openfast"
    global_install_marker = tmp_path / "global-install-ran"
    source = openfast_root / "source"
    (source / ".git").mkdir(parents=True)
    installed = openfast_root / "install" / "bin" / "turbsim"
    installed.parent.mkdir(parents=True)
    installed.write_text(
        "#!/bin/sh\necho 'broken setup' >&2\nexit 127\n",
        encoding="utf-8",
    )
    installed.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts" / "install_openfast_turbsim.sh"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "OPENFAST_ROOT": str(openfast_root),
            "OPENFAST_SOURCE_DIR": str(source),
            "OPENFAST_BUILD_DIR": str(openfast_root / "build"),
            "OPENFAST_INSTALL_DIR": str(openfast_root / "install"),
            "TURBSIM_EXE": str(installed),
            "OPENFAST_BUILD_JOBS": "1",
            "CMAKE_GLOBAL_INSTALL_MARKER": str(global_install_marker),
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert installed.is_file()
    assert os.access(installed, os.X_OK)
    assert not global_install_marker.exists()
    assert "Existing TurbSim executable failed its startup check" in completed.stdout
    assert "Attempting an automatic rebuild and replacement" in completed.stdout
    assert f"Installing the built TurbSim target directly into {installed}" in completed.stdout
    assert "Direct TurbSim installation succeeded:" in completed.stdout
    assert "fake TurbSim version 5.0.0" in subprocess.run(
        [str(installed), "-h"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
