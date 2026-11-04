# NACA0018 source database and Kynema setup

Mesh and setup scripts were developed by Alan Hsieh and Kevin R. Moore
(starting July 2025).

This example releases the static-airfoil force histories, processing scripts,
Pointwise mesh recipe and Kynema/Nalu run templates used in developing the
SIL paper's source database. No CFD solver or licensed mesh generator is
included. The current templates are the authors' working revision; their
low-Re frequency-resolution guard postdates the original CFD runs. Preparing
these templates does not reproduce the archived histories bit for bit.

The paper's actual runtime database is
[`NACA0018.h5`](../ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/VorLapAirfoils/NACA0018.h5),
SHA-256 `64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369`.
It contains replicated Reynolds planes from one static-airfoil source;
replication extends lookup coverage and supplies no new Reynolds-dependent
CFD evidence. The smaller database in `data/airfoils/` is a legacy source.
Do not substitute it when recreating the paper.

`raw_force_histories/NACA18.zip` retains the original angle-indexed histories.
The canonical converters remain in `scripts/process_naluout_to_h5.py` and
`scripts/extend_replicated_reynolds_grid.py`. The former documents startup
discard, spectral normalization, periodic angles and replicated-Re options;
the latter verifies the parent identity before extending the envelope.
Use the paper's recorded conversion receipt and environment for exact replay.
Do not infer a new CFD validation claim from lookup replication.

Generate meshes using `mesh_gen_scripts/IEA_Airfoil_Mesh.glf`, the supplied
airfoil coordinates and `mesh_gen_scripts/create_grids.py`. This historical
driver requires Pointwise; configure its executable, airfoil, angle list and
point count before execution. It does not run as part of VorLap installation.

Prepare one case without submitting a job:

```bash
python examples/naca0018_kynema/nalu_runs/submission_scripts/gen_submit_nalu.py \
  --mesh /absolute/path/NACA0018_84.exo --output /absolute/path/new_cases \
  --airfoil NACA0018 --aoa 84 --speed 3.75
```

This writes a new case and prints the Slurm command. Existing case directories
are refused. Inspect the YAML, configure your qualified Kynema/Nalu and MPI
environment and site resources, then use `--submit` only for a new case you
intend to submit. `KYNEMA_EXE` defaults to `naluX`. No HPC job is submitted by
the tests or release preparation. A real Exodus mesh and the original solver
build are required for CFD reproduction; a preparation smoke check does not
validate a CFD solution.

`source_manifest.json` records every imported file's original hash and source
commit. Original working files and their patch are preserved with the private
paper inputs. Public edits remove site accounts, personal paths and automatic
directory deletion/submission; the numerical preparation formulas are retained.
The spreadsheet and scaling script explain those formulas. Original post-processing plots remain in the private paper source snapshot;
the supported portable HDF5 conversion path is the top-level VorLap converter.
