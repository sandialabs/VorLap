# Contributing

Use a feature branch and keep changes focused. Explain the affected physical
or user-facing behavior and provide the commands and results used to check it.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs,gui]"
python -m pytest
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Tkinter is supplied by
the Python installation or operating system, rather than pip. On headless
Linux, run GUI checks with `VORLAP_REQUIRE_GUI_TESTS=1 xvfb-run -a python -m
pytest tests/test_gui_import_state.py tests/test_gui_setup_params.py
tests/test_qblade_bridge_tab.py`.

The core suite uses synthetic cases and retained public benchmark inputs.
Tests of the separate manuscript skip when its checkout is absent. Run
`python -m pytest --require-paper` when checking the integrated paper workflow;
see [reproducibility](docs/reproducibility.md). Archived campaign checks require
the explicitly documented external inputs. Runtime-bound tests skip without
the separately installed QBlade/bridge; `python -m pytest --require-qblade`
requires it. Run campaign provenance checks from an installed Git checkout.
A skip is not a passing simulation.

For numerical changes, check units, signs, reference frames, independent
physical limits, and relevant convergence behavior. Explain changes to frozen
results before changing their expected values. Keep historical receipts and
source identities intact; they describe the original calculation.

Keep runtime installations, credentials, raw campaign results, build products,
and operating-system files out of Git. Retain source inputs needed by examples
and document their provenance. See [architecture](docs/developer_architecture.md),
[coordinate frames](docs/coordinate_frames.md), and
[coupling timing](docs/qblade_timing.md).

Use GitHub issues for reproducible defects and pull requests for changes.
Report the Python version, platform, command, and a minimal input. Squash
feature work when merging a release; do not rewrite shared history without
coordinating with repository maintainers.
