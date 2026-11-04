# Examples

These examples are source files in `examples/` and are included directly here to avoid duplicated documentation code.

## Synthetic Analytical Example

```python
--8<-- "examples/basic_synthetic_case.py"
```

## Single-Blade Verification Case

```python
--8<-- "examples/single_blade_verification.py"
```

## Time-Varying Inflow Case

```python
--8<-- "examples/time_varying_inflow_case.py"
```

## QBlade Fast-Path Loading Export

```python
--8<-- "examples/qblade_fastpath_loading.py"
```

## QBlade Bridge Case-Prep Dry Run

This example validates and plans the external-library edits for the bundled
reference QBlade model without launching QBlade or writing files.

```python
--8<-- "examples/qblade_bridge_prepare_dry_run.py"
```

## QBlade Python SIL Smoke Run

This example is for machines with a valid QBlade root. It prepares a generated
working copy in the output directory, runs a short Python SIL bridge case, and
prints the summary/artifact paths. It is not native callback proof.

```python
--8<-- "examples/qblade_sil_smoke_run.py"
```

## Headless QBlade Parked Sweep

The headless parked-load workflow is in `run_headless/`. It prepares copies of
`examples/ReferenceHVAWTQBladeModel`, builds the QBlade controller bridge with
`PYTHON_EXE`, and can submit one parked/idling wind-speed case per Slurm job.

```bash
export PYTHON_EXE=/path/to/python
sbatch run_headless/submit_vorlap_parked.sj
scripts/plot_qblade_parked_sweep.py --run-root run_headless/runs/vorlap_parked
```

## Released NACA0018/Kynema setup

The source tree includes `examples/naca0018_kynema/`: the raw static-airfoil histories, Pointwise recipe, Kynema templates and safe preparation commands. The actual paper runtime HDF5 is retained in `examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/VorLapAirfoils/`. Read the examples' provenance and modeling limits before selecting a source.
