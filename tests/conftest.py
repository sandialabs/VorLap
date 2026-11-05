import numpy as np
from pathlib import Path
import json
import os
import platform
import pytest

from vorlap.structs import AirfoilFFT, Component, VIV_Params


def pytest_addoption(parser):
    parser.addoption(
        "--require-paper", action="store_true",
        help="Fail if the separately versioned SIL manuscript checkout is absent.",
    )
    parser.addoption(
        "--require-qblade", action="store_true",
        help="Fail if the qualified QBlade and native bridge installation is absent.",
    )
    parser.addoption(
        "--qblade-inventory", type=Path,
        help="With --require-qblade, require every test node in this JSON inventory (CI completeness gate).",
    )


def pytest_configure(config):
    if config.getoption("--require-qblade"):
        from scripts.verify_qblade_runtime import verify_runtime

        try:
            verify_runtime(Path(__file__).resolve().parents[1] / "run_headless/QBladeCE_2.0.9.7")
        except (OSError, ValueError, RuntimeError) as exc:
            raise pytest.UsageError(f"Qualified QBlade test installation failed verification: {exc}") from exc
        config._qblade_reports = {}
    if config.getoption("--require-paper"):
        paper = Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge"
        required = [
            paper / "NAWEA2026_QBladeBridge.tex",
            paper / "analysis/verify_reproducibility_package.py",
            paper / "analysis/reproducibility/run_all.sh",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise pytest.UsageError("SIL manuscript checkout is incomplete: " + ", ".join(missing))


def _missing_qblade_files():
    root = Path(__file__).resolve().parents[1] / "run_headless/QBladeCE_2.0.9.7"
    return [path for path in (
        root / "libQBladeCE_2.0.9.7.so.1.0.0",
        root / "ControllerFiles/libvorlap_qblade_bridge.so",
        root / "SIL_Interface/QBladeLibrary.py",
    ) if not path.is_file()]


def pytest_collection_modifyitems(items):
    if _missing_qblade_files() or (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        skip = pytest.mark.skip(
            reason="Qualified Linux x86_64 QBlade/bridge installation is unavailable; see docs/reproducibility.md",
        )
        for item in items:
            if item.get_closest_marker("requires_qblade"):
                item.add_marker(skip)


def pytest_collection_finish(session):
    if session.config.getoption("--require-qblade"):
        session.config._qblade_selected = {
            item.nodeid for item in session.items if item.get_closest_marker("requires_qblade")
        }
        if not session.config._qblade_selected:
            raise pytest.UsageError("--require-qblade selected no QBlade tests.")
        inventory = session.config.getoption("--qblade-inventory")
        if inventory:
            required = set(json.loads(inventory.read_text(encoding="utf-8"))["required_nodeids"])
            missing = sorted(required - session.config._qblade_selected)
            if missing:
                raise pytest.UsageError("Required QBlade coverage is missing: " + ", ".join(missing))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    if item.config.getoption("--require-qblade") and item.get_closest_marker("requires_qblade"):
        report = result.get_result()
        item.config._qblade_reports.setdefault(item.nodeid, {})[report.when] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    if not session.config.getoption("--require-qblade"):
        return
    selected = getattr(session.config, "_qblade_selected", set())
    reports = getattr(session.config, "_qblade_reports", {})
    incomplete = sorted(node for node in selected if reports.get(node) != {
        "setup": "passed", "call": "passed", "teardown": "passed",
    })
    evidence = {"selected": sorted(selected), "reports": reports, "incomplete": incomplete}
    output = os.environ.get("VORLAP_QBLADE_TEST_RECEIPT")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    if incomplete or not selected:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminal:
            terminal.write_sep("!", "Required QBlade tests did not all execute and pass")
            for node in incomplete:
                terminal.write_line(node)


def make_constant_airfoil_fft(
    name: str = "default",
    cl_dc: float = 1.0,
    cd_dc: float = 2.0,
    cf_dc: float = 0.0,
    cl_h1_amp: float = 0.0,
    cd_h1_amp: float = 0.0,
    cf_h1_amp: float = 0.1,
    st_h1: float = 0.5,
    n_freq: int = 2,
    re_grid=None,
    reynolds_model: str = "",
) -> AirfoilFFT:
    """Create a small synthetic airfoil FFT dataset for deterministic tests."""
    re_grid = np.array([0.1, 1.0e7], dtype=float) if re_grid is None else np.asarray(re_grid, dtype=float)
    aoa_grid = np.array([-180.0, 180.0], dtype=float)
    shape = (re_grid.size, aoa_grid.size, n_freq)

    def _zeros() -> np.ndarray:
        return np.zeros(shape, dtype=float)

    cl_st = _zeros()
    cd_st = _zeros()
    cm_st = _zeros()
    cf_st = _zeros()
    if n_freq > 1:
        cl_st[:, :, 1] = st_h1
        cd_st[:, :, 1] = st_h1
        cm_st[:, :, 1] = st_h1
        cf_st[:, :, 1] = st_h1

    cl_amp = _zeros()
    cd_amp = _zeros()
    cm_amp = _zeros()
    cf_amp = _zeros()
    cl_amp[:, :, 0] = cl_dc
    cd_amp[:, :, 0] = cd_dc
    cf_amp[:, :, 0] = cf_dc
    if n_freq > 1:
        cl_amp[:, :, 1] = cl_h1_amp
        cd_amp[:, :, 1] = cd_h1_amp
        cf_amp[:, :, 1] = cf_h1_amp

    cl_pha = _zeros()
    cd_pha = _zeros()
    cm_pha = _zeros()
    cf_pha = _zeros()

    return AirfoilFFT(
        name=name,
        Re=re_grid,
        AOA=aoa_grid,
        Thickness=0.12,
        CL_ST=cl_st,
        CD_ST=cd_st,
        CM_ST=cm_st,
        CF_ST=cf_st,
        CL_Amp=cl_amp,
        CD_Amp=cd_amp,
        CM_Amp=cm_amp,
        CF_Amp=cf_amp,
        CL_Pha=cl_pha,
        CD_Pha=cd_pha,
        CM_Pha=cm_pha,
        CF_Pha=cf_pha,
        reynolds_model=reynolds_model,
    )


def make_linear_airfoil_fft() -> AirfoilFFT:
    """Create an FFT grid where each value is linear in Re and AOA for interpolation tests."""
    re_grid = np.array([1.0, 3.0], dtype=float)
    aoa_grid = np.array([0.0, 10.0], dtype=float)
    n_freq = 2
    rr, aa = np.meshgrid(re_grid, aoa_grid, indexing="ij")
    base = rr + 2.0 * aa
    harmonic = 0.1 * rr + 0.05 * aa
    amp_data = np.stack([base, harmonic], axis=2)
    zeros = np.zeros_like(amp_data)

    cl_st = np.stack([np.zeros_like(base), harmonic], axis=2)
    cd_st = np.stack([np.zeros_like(base), harmonic + 0.1], axis=2)
    cm_st = np.stack([np.zeros_like(base), harmonic + 0.2], axis=2)
    cf_st = np.stack([np.zeros_like(base), harmonic + 0.3], axis=2)
    return AirfoilFFT(
        name="linear",
        Re=re_grid,
        AOA=aoa_grid,
        Thickness=0.12,
        CL_ST=cl_st,
        CD_ST=cd_st,
        CM_ST=cm_st,
        CF_ST=cf_st,
        CL_Amp=amp_data + 4.0,
        CD_Amp=amp_data + 5.0,
        CM_Amp=amp_data + 6.0,
        CF_Amp=amp_data + 7.0,
        CL_Pha=zeros + 0.01,
        CD_Pha=zeros + 0.02,
        CM_Pha=zeros + 0.03,
        CF_Pha=zeros + 0.04,
    )


def make_component(n_nodes: int = 2, span: float = 2.0, airfoil_id: str = "default") -> Component:
    """Create a straight component aligned with the z-axis."""
    z = np.linspace(0.0, span, n_nodes, dtype=float)
    shape_xyz = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
    chord = np.ones(n_nodes, dtype=float)
    twist = np.zeros(n_nodes, dtype=float)
    thickness = np.ones(n_nodes, dtype=float) * 0.12
    offset = np.zeros(n_nodes, dtype=float)
    airfoil_ids = [airfoil_id] * n_nodes
    chord_vector = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=float), (n_nodes, 1))
    normal_vector = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=float), (n_nodes, 1))

    return Component(
        id="blade",
        translation=np.zeros(3, dtype=float),
        rotation=np.zeros(3, dtype=float),
        pitch=np.array([0.0], dtype=float),
        shape_xyz=shape_xyz,
        shape_xyz_global=shape_xyz.copy(),
        chord=chord,
        twist=twist,
        thickness=thickness,
        offset=offset,
        airfoil_ids=airfoil_ids,
        chord_vector=chord_vector,
        normal_vector=normal_vector,
    )


def make_viv_params() -> VIV_Params:
    """Create deterministic VIV parameters for unit tests."""
    return VIV_Params(
        fluid_density=1.0,
        fluid_dynamicviscosity=1.0,
        rotation_axis=np.array([0.0, 0.0, 1.0], dtype=float),
        rotation_axis_offset=np.array([0.0, 0.0, 0.0], dtype=float),
        inflow_vec=np.array([1.0, 1.0, 0.0], dtype=float),
        azimuths=np.array([0.0], dtype=float),
        inflow_speeds=np.array([2.0], dtype=float),
        output_time=np.array([0.0, 0.25], dtype=float),
        output_azimuth_vinf=(0.0, 2.0),
        n_harmonic=2,
        amplitude_coeff_cutoff=0.05,
        n_freq_depth=2,
    )
