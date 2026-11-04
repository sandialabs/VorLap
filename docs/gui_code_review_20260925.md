# GUI and code review, 25 September 2026

The offline GUI workflow and affected numerical/bridge interfaces pass the
targeted checks below. This is software verification. It does not establish a
new coupled QBlade result or experimental validation.

## Corrections

- Offline runs now snapshot Tk inputs before starting the numerical worker.
  Component frames are copied so worker calculations cannot mutate displayed
  geometry. Both GUI workers queue messages for the Tk thread, including state
  notifications; closing the bridge tab requests cancellation.
- The Run control prevents overlapping offline analyses and is restored after
  success or failure. A failed rerun cannot expose the preceding result as its
  output. Replacing geometry also invalidates preceding results.
- Imported modal frequencies retain 12 significant digits instead of rounding
  to 0.01 Hz. Offline analysis rejects empty, nonfinite, or nonpositive modal
  frequencies before loading inputs. Rigid-body modes are omitted from this
  frequency-overlap comparison.
- Completed geometry plots use their recorded calculation parameters even
  while the next setup is being edited. Both GUI tabs resolve the same QBlade
  example; installations without that example start with packaged component
  CSVs.
- Reference selection has no implicit fallback to another model. Explicit
  custom model selection remains supported. Structural `SUBFILE` removal now
  requires `VORLAP_STABLE_PATCH_REMOVE_SUBFILE=1`; directory names cannot trigger
  a hidden support-model change. Synthetic test models use generic names.
- GUI smoke tests probe Tk availability separately. Application errors fail
  the tests; `VORLAP_REQUIRE_GUI_TESTS=1` also fails if Tk itself is unavailable.

## Verification

The following command passed **285 tests**, including real Tk windows on macOS
with Python 3.11.5/Tk 8.6. Window-server access required execution outside the
filesystem sandbox.

```sh
MPLCONFIGDIR=/private/tmp/vorlap-matplotlib VORLAP_REQUIRE_GUI_TESTS=1 \
.venv/bin/python -m pytest \
  tests/test_gui_import_state.py tests/test_gui_setup_params.py \
  tests/test_offline_analysis.py tests/test_qblade_bridge_tab.py \
  tests/test_qblade_bridge_service.py tests/test_qblade_case_prep.py \
  tests/test_qblade_runtime.py tests/test_computations.py \
  tests/test_interpolation.py tests/test_structs.py \
  tests/test_model_based_negative_damping.py -q \
  --basetemp=/private/tmp/vorlap-gui-review-20260925
```

The real GUI test loads a synthetic straight component through CSV, uses the
packaged NACA0018 spectral database, runs a small wind/azimuth sweep and force
history, checks finite nonzero loads, writes CSV/JSON output, and renders
geometry, overlap, force, and moment plots. It verifies main-thread state
notifications, frequency precision, duplicate-run protection, plotting during
subsequent edits, and recovery from a missing-inflow-file error. The saved
moment plot was inspected: axes and units are readable, and zero shaft moment
is consistent with this on-axis synthetic component.

The generic reference-selection and support-patch audit tests also pass:

```sh
MPLCONFIGDIR=/private/tmp/vorlap-matplotlib .venv/bin/python -m pytest \
  tests/test_run_headless_campaign.py -q \
  -k 'reference_selection or stable_support'
```

Result: **4 passed, 161 deselected**. These include explicit custom-model
selection, preservation of the support file unless removal is requested, and
failure of a campaign audit when its simulation cannot identify the turbine.
After integrating the campaign/paper-helper and Slurm fixes, the complete
`tests/test_run_headless_campaign.py` file passed **165 tests** (78.09 s).

Test logs and the GUI CSV/JSON/PNG artifacts are retained outside Git under
the author's archived GUI review record. The full repository
test and paper-cleanup results are recorded separately by the integration
review.

## Installed-package verification

Packaging now includes the seven tracked Bidadi spectral HDF5 files and their
bundle manifest, plus the generic bridge build script and CMake/C++ sources.
Those resources were previously absent from wheels. The packaging coverage
test now checks JSON metadata as well as CSV/HDF5 inputs; packaging and build
script tests pass **4/4**.

A fresh tracked-plus-intended source snapshot was built with
`setuptools.build_meta.build_wheel` and `build_sdist`. Archive inspection found
no retired reference-component data, reference parameter/frequency files,
operating-controller directory, or test-component directory. Rebuilding the
extracted sdist produced identical wheel member names and contents.

The wheel was installed without dependencies into a separate virtual
environment. Existing numerical dependencies were supplied through a site
packages path; assertions verified that all imported VorLap, scripts, and
run-headless modules came from the installed wheel, not the development
checkout. Both QBlade CLI `--help` commands passed. Real Tk initialization
selected packaged CSV geometry, loaded 10 components and five modal
frequencies, loaded the airfoil databases, and rendered the geometry plot.
The existing legacy FFT repair warnings remain explicit in the log.

The installed build script also compiled a native bridge using only its
packaged CMake/C++ sources. `ctypes` loaded the resulting macOS library and
resolved `update_init`, `update`, and `update_message`. This host needed the
following environment because its default Xcode selection was unusable and
the Command Line Tools compiler lacked a default libc++ header search path:

```sh
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
export CXX=/Library/Developer/CommandLineTools/usr/bin/clang++
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1"
```

No system settings or license agreements were changed. Build logs, package
inspection/hashes, installed CLI output, and the rendered installed-GUI plot
are in the author's archived packaging review record.

## Remaining evidence limits

No fresh QBlade coupled simulation or Flight job was run in this review. The
repo-local QBlade bundle targets Linux, whereas this GUI run used macOS.
Service and runtime tests verify configuration, load calculations, and
orchestration; they do not prove that a native QBlade run completed. The
negative-damping research prototype remains a separate opt-in API, with its
existing numerical tests; it is not silently activated in the GUI. Historical
design exposure and the retained H-VAWT model boundary require the separate
repository audit.
