from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from vorlap.qblade_runtime import (
    VorLapQBladeRuntime,
    build_external_library_table_spec,
    build_qblade_external_config,
    write_qblade_external_config,
)


HARNESS_SOURCE = r"""
#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using update_init_fn = void (*)(const char*);
using update_fn = void (*)(float*);
using update_message_fn = void (*)(char*);

template <typename T>
T load_symbol(void* handle, const char* name) {
    dlerror();
    void* raw = dlsym(handle, name);
    const char* error = dlerror();
    if (error || !raw) {
        std::cerr << "dlsym failed for " << name << ": " << (error ? error : "null symbol") << "\n";
        std::exit(3);
    }
    return reinterpret_cast<T>(raw);
}

void print_force(const char* label, const std::vector<float>& swap, int offset) {
    std::cout << label << "="
              << std::setprecision(9)
              << swap[offset] << ","
              << swap[offset + 1] << ","
              << swap[offset + 2] << "\n";
}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cerr << "usage: harness LIB CONFIG SWAP_SIZE VELOCITY_OFFSET [FAIL_CONFIG]\n";
        return 2;
    }
    const char* library_path = argv[1];
    const char* config_path = argv[2];
    const int swap_size = std::stoi(argv[3]);
    const int velocity_offset = std::stoi(argv[4]);
    const char* fail_config_path = argc == 6 ? argv[5] : nullptr;
    const int force_offset = velocity_offset + 3;

    void* handle = dlopen(library_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "dlopen failed: " << dlerror() << "\n";
        return 2;
    }

    auto update_init = load_symbol<update_init_fn>(handle, "update_init");
    auto update = load_symbol<update_fn>(handle, "update");
    auto update_message = load_symbol<update_message_fn>(handle, "update_message");

    char message[1024];
    std::memset(message, 0, sizeof(message));
    update_init(config_path);
    update_message(message);
    std::cout << "init_message=" << message << "\n";

    std::vector<float> swap(static_cast<size_t>(swap_size), 0.0f);
    swap[0] = 0.125f;
    swap[1] = 0.25f;
    swap[2] = 30.0f;
    swap[velocity_offset + 0] = 0.2f;
    swap[velocity_offset + 1] = -0.4f;
    swap[velocity_offset + 2] = 0.1f;

    update(swap.data());
    std::memset(message, 0, sizeof(message));
    update_message(message);
    std::cout << "update_message=" << message << "\n";
    print_force("force", swap, force_offset);

    update(nullptr);
    std::memset(message, 0, sizeof(message));
    update_message(message);
    std::cout << "null_swap_message=" << message << "\n";

    swap[force_offset + 0] = 123.0f;
    swap[force_offset + 1] = 123.0f;
    swap[force_offset + 2] = 123.0f;
    swap[velocity_offset + 0] = std::numeric_limits<float>::quiet_NaN();
    update(swap.data());
    std::memset(message, 0, sizeof(message));
    update_message(message);
    std::cout << "failure_message=" << message << "\n";
    print_force("failure_force", swap, force_offset);

    swap[force_offset + 0] = 222.0f;
    swap[force_offset + 1] = 222.0f;
    swap[force_offset + 2] = 222.0f;
    swap[0] = 1.0e12f;
    swap[velocity_offset + 0] = 0.2f;
    swap[velocity_offset + 1] = -0.4f;
    swap[velocity_offset + 2] = 0.1f;
    update(swap.data());
    std::memset(message, 0, sizeof(message));
    update_message(message);
    std::cout << "sanity_message=" << message << "\n";
    print_force("sanity_force", swap, force_offset);

    if (fail_config_path) {
        std::vector<float> fail_swap(static_cast<size_t>(swap_size), 0.0f);
        fail_swap[0] = 0.125f;
        fail_swap[1] = 0.25f;
        fail_swap[2] = 30.0f;
        fail_swap[velocity_offset + 0] = 0.2f;
        fail_swap[velocity_offset + 1] = -0.4f;
        fail_swap[velocity_offset + 2] = 0.1f;
        fail_swap[force_offset + 0] = 321.0f;
        fail_swap[force_offset + 1] = 321.0f;
        fail_swap[force_offset + 2] = 321.0f;

        update_init(fail_config_path);
        std::memset(message, 0, sizeof(message));
        update_message(message);
        std::cout << "injected_init_message=" << message << "\n";

        update(fail_swap.data());
        std::memset(message, 0, sizeof(message));
        update_message(message);
        std::cout << "injected_failure_message=" << message << "\n";
        print_force("injected_failure_force", fail_swap, force_offset);
    }

    dlclose(handle);
    return 0;
}
"""


def _build_bridge_library(repo: Path, tmp_path: Path) -> Path:
    if sys.platform == "win32":
        pytest.skip("The checked ABI harness uses the POSIX bridge build script.")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for the bridge ABI harness.")

    build_dir = tmp_path / "qblade_external_bridge_build"
    env = os.environ.copy()
    env["PYTHON_EXE"] = sys.executable
    completed = subprocess.run(
        ["bash", str(repo / "scripts" / "build_qblade_external_linux.sh"), str(build_dir)],
        check=False,
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    libraries = sorted(
        path
        for path in build_dir.rglob("*vorlap_qblade_bridge*")
        if path.is_file() and path.suffix in {".so", ".dylib", ".dll"}
    )
    assert libraries, completed.stdout + completed.stderr
    return libraries[0]


def _write_minimal_airfoil_fft(path: Path) -> None:
    re = np.array([0.1, 1.0e7], dtype=float)
    aoa = np.array([-180.0, 180.0], dtype=float)
    shape = (re.size, aoa.size, 1)
    zeros = np.zeros(shape, dtype=float)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("Airfoilname", data=b"NACA0012")
        h5.create_dataset("Re", data=re)
        h5.create_dataset("AOA", data=aoa)
        h5.create_dataset("Thickness", data=np.array([0.12]))
        h5.create_dataset("CL_ST", data=zeros)
        h5.create_dataset("CD_ST", data=zeros)
        h5.create_dataset("CM_ST", data=zeros)
        h5.create_dataset("CF_ST", data=zeros)
        h5.create_dataset("CL_Amp", data=np.full(shape, 1.25, dtype=float))
        h5.create_dataset("CD_Amp", data=np.full(shape, 0.75, dtype=float))
        h5.create_dataset("CM_Amp", data=zeros)
        h5.create_dataset("CF_Amp", data=zeros)
        h5.create_dataset("CL_Pha", data=zeros)
        h5.create_dataset("CD_Pha", data=zeros)
        h5.create_dataset("CM_Pha", data=zeros)
        h5.create_dataset("CF_Pha", data=zeros)


def _write_minimal_qblade_case(tmp_path: Path) -> tuple[Path, Path]:
    case_dir = tmp_path / "case"
    airfoil_dir = case_dir / "airfoils"
    airfoil_dir.mkdir(parents=True)
    _write_minimal_airfoil_fft(airfoil_dir / "NACA0012.h5")

    sim_path = case_dir / "case.sim"
    sim_path.write_text(
        "\n".join(
            [
                "turbine.trb TURBFILE",
                "1 DENSITYAIR",
                "1 VISCOSITYAIR",
                "0 HORANGLE",
                "0 VERTANGLE",
                "3 MEANINF",
                "0.25 TIMESTEP",
                "2 NUMTIMESTEPS",
                "0 INITIAL_AZIMUTH",
                "0 GLOBPOS_X",
                "0 GLOBPOS_Y",
                "0 GLOBPOS_Z",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (case_dir / "turbine.trb").write_text(
        "\n".join(
            [
                "blade.bld BLADEFILE",
                "1 NUMBLADES",
                "0 TOWERHEIGHT",
                "0 TOWERTOPRAD",
                "0 TOWERBOTRAD",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (case_dir / "blade.bld").write_text(
        "\n".join(
            [
                "1 NUMBLADES",
                "HEIGHT_[m] CHORD_[m] RADIUS_[m] TWIST_[deg] P_AXIS_[-] CIRCANGLE_[deg] POLAR_FILE",
                "0.0 1.0 1.0 0.0 0.25 0.0 NACA0012.plr",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return sim_path, airfoil_dir


def _compile_harness(tmp_path: Path) -> Path:
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the bridge ABI harness.")
    source = tmp_path / "bridge_abi_harness.cpp"
    exe = tmp_path / "bridge_abi_harness"
    source.write_text(HARNESS_SOURCE, encoding="utf-8")
    command = [compiler, "-std=c++17", str(source), "-o", str(exe)]
    if sys.platform.startswith("linux"):
        command.append("-ldl")
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return exe


def _prepend_env_path(env: dict[str, str], key: str, values: list[Path]) -> None:
    existing = [part for part in env.get(key, "").split(os.pathsep) if part]
    rendered = [str(value) for value in values if value]
    env[key] = os.pathsep.join(rendered + [part for part in existing if part not in rendered])


def _parse_harness_output(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _parse_force_triplet(text: str) -> np.ndarray:
    return np.asarray([float(part) for part in text.split(",")], dtype=float)


def test_cpp_bridge_abi_matches_python_runtime_and_clears_failed_update(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    library_path = _build_bridge_library(repo, tmp_path)
    harness = _compile_harness(tmp_path)
    sim_path, airfoil_dir = _write_minimal_qblade_case(tmp_path)
    node_ids = ["BLD_1_0.000000"]

    config = build_qblade_external_config(
        sim_path=os.path.relpath(sim_path, tmp_path / "Control"),
        airfoil_dir=os.path.relpath(airfoil_dir, tmp_path / "Control"),
        node_ids=node_ids,
        n_freq_depth=1,
        python_path_entries=[str(repo)],
        debug=True,
        log_file="vorlap_qblade_runtime.log",
    )
    config_path = Path(write_qblade_external_config(str(tmp_path / "Control" / "vorlap_qblade_external.json"), config))
    failing_config = dict(config)
    failing_config["debug"] = True
    failing_config["debug_failure_inject_update_exception"] = True
    failing_config["log_file"] = "vorlap_qblade_runtime_failure.log"
    failing_config_path = Path(
        write_qblade_external_config(str(tmp_path / "Control" / "vorlap_qblade_external_failure.json"), failing_config)
    )
    spec = build_external_library_table_spec(node_ids)

    expected_runtime = VorLapQBladeRuntime.from_qblade_config(str(config_path))
    expected_swap = np.zeros(spec.swap_size, dtype=np.float32)
    expected_swap[spec.swap_layout["time"].offset] = np.float32(0.125)
    expected_swap[spec.swap_layout["timestep"].offset] = np.float32(0.25)
    expected_swap[spec.swap_layout["azimuth_deg"].offset] = np.float32(30.0)
    expected_swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 3] = np.array(
        [0.2, -0.4, 0.1], dtype=np.float32
    )
    expected_runtime.update(expected_swap)
    expected_force_block = expected_swap[
        spec.swap_layout["force"].offset : spec.swap_layout["force"].offset + 3
    ].copy()
    expected_message = expected_runtime.update_message()

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    # The standalone C++ executable embeds libpython, not the venv launcher.
    # Give it the same standard library and installed dependencies explicitly.
    # Source imports remain bound through the runtime config's python_path.
    import site
    env["PYTHONHOME"] = sys.base_prefix
    _prepend_env_path(env, "PYTHONPATH", [Path(p) for p in site.getsitepackages()])
    python_lib_dirs = [Path(sys.base_prefix) / "lib", Path(sys.prefix) / "lib"]
    if sys.platform == "darwin":
        # A hard DYLD_LIBRARY_PATH makes conda's libiconv override the macOS
        # system dependency used by libunistring. Fallback still locates the
        # embedded Python library without replacing system libraries.
        _prepend_env_path(env, "DYLD_FALLBACK_LIBRARY_PATH", python_lib_dirs)
    else:
        _prepend_env_path(env, "LD_LIBRARY_PATH", python_lib_dirs)

    completed = subprocess.run(
        [
            str(harness),
            str(library_path),
            str(config_path),
            str(spec.swap_size),
            str(spec.swap_layout["velocity"].offset),
            str(failing_config_path),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = _parse_harness_output(completed.stdout)

    np.testing.assert_allclose(
        _parse_force_triplet(output["force"]),
        expected_force_block,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert output["init_message"] == "VorLap QBlade runtime initialized."
    assert output["update_message"] == expected_message
    assert 0 < len(output["update_message"]) < 1023
    assert "null avrSwap pointer" in output["null_swap_message"]
    np.testing.assert_allclose(_parse_force_triplet(output["failure_force"]), 0.0)
    assert "non-finite structural velocity" in output["failure_message"]
    np.testing.assert_allclose(_parse_force_triplet(output["sanity_force"]), 0.0)
    assert "implausible QBlade swap time" in output["sanity_message"]
    assert output["injected_init_message"] == "VorLap QBlade runtime initialized."
    np.testing.assert_allclose(_parse_force_triplet(output["injected_failure_force"]), 0.0)
    assert "debug failure injection requested" in output["injected_failure_message"]
    assert "force output cleared" in output["injected_failure_message"]

    bridge_log = config_path.parent / "vorlap_qblade_bridge_cpp.log"
    assert bridge_log.is_file()
    log_text = bridge_log.read_text(encoding="utf-8")
    assert "update_init entered" in log_text
    assert "python_env_before_import=" in log_text
    assert "python_path_entries_added" in log_text
    assert str(repo) in log_text
    assert "python_env_after_import=" in log_text
    assert "numpy_version" in log_text
    assert "vorlap_qblade_runtime_file" in log_text
    assert "update entered; count=1" in log_text
    assert "VorLap QBlade runtime initialized." in log_text
