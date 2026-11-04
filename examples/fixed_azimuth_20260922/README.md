# U11.25 fixed-azimuth matched three-model follow-up

Run **11.25 m/s, rotor azimuth 147°, prescribed 0 RPM**, seed **4433456**, with
QBlade, Spectral VorLap, and Hybrid VorLap as three independent Slurm array
tasks. Each runs **660 s** at **0.0075 s**; compare the final **600 s** after
discarding the first 60 s. The full 660 s is exported so azimuth and RPM can be
checked from initialization through the final sample.

The angle is selected from the existing idling simulation, using the median
over ten revolutions of the high-frequency flapwise-moment RMS at Blade 1,
z/H = 0.23. See `../fixed_azimuth_selection_20260922/evidence/selection.json`.
The three-bladed rotor at 147° has the same set of blade orientations as 27°
after blade permutation. This is a high-excitation candidate for the new
fixed-rotor experiment, not proof of the maximum over all fixed azimuths.
Set the **actual QBlade rotor azimuth to 147°**; do not add the 40.0075° startup
phase offset of the old 1-RPM record.

## Preserved physics and explicit approximation

The preparer checks physical-input hashes against the archived three-path
campaign. It retains the turbine, calibrated NACA0018 spectrum, damping,
`parked-reduced-v2` wake settings, 10 s structural startup, 5 s additional
damping, and the exact turbulent-wind bytes
`dlc6p4_U11p25ms_s4433456.bts` (SHA-256
`dd1608a5bf2dc5415858fe89e97bcecc332eb515b3497dd7b66fb04d735d74b2`).
`PRESCRIBETYPE=1` and `RPMPRESCRIBED=0` hold the rotor for the entire simulation;
there is no coast-down or startup rotation command.

Hybrid VorLap retains the reviewed **1-RPM modal approximation**, while QBlade
itself solves the structure at actual zero RPM. The archived independent
zero-RPM eigensolutions give frequency differences of 0.000112% and 0.001778%
and action-shape MAC values of 0.999956 and 0.999743 for modes 6 and 12.
Preparation requires frequency differences below 0.1% and MAC above 0.999.
The case-specific modal RPM tolerance is widened to 1.001 RPM to admit this
documented transfer; the submitted case and output audit independently require
**actual RPM within 0.001 of zero and azimuth within 0.01° of 147°**. This does
not create a newly identified zero-RPM response basis, change production model
files, or authorize arbitrary rotor speeds. The evidence is saved in
`inputs/modal_basis_transfer.json` and the campaign manifest.

Each case starts fresh and refuses to overwrite an existing run. Bounded
checkpoints are retained for diagnosis; automatic continuation is disabled,
as in the reviewed three-path campaign. A successful process exit additionally
requires all 88,000 integration steps, finite loads, complete rotor-state
histories, and constant measured rotor angle/RPM.

## Update through Git and submit on Flight

Use the established Flight checkout and the same one-command launcher pattern
as the recent DLC runs. From your Flight shell:

```bash
cd /pscratch/kevmoor/vorlap
git switch response-aware-viv-fatigue
git pull --ff-only origin response-aware-viv-fatigue
bash run_headless/submit_fixed_azimuth_chain.sh
```

The launcher uses the existing `.venv/bin/python`, QBlade installation,
`anaconda3/2025.12-2`, and source campaign
`/pscratch/kevmoor/vorlap_runs/hvawt_complete_dlc_three_path_v3`.
It creates a fresh campaign named
`/pscratch/kevmoor/vorlap_runs/u11p25_a147_fixed_<UTC timestamp>`, rebuilds and
verifies the three inputs, then submits one exclusive 112-CPU node per model.
The prepared cases retain the exact original turbulent wind file. No archive
transfer, package installation, controller build, or change of Python
installation is needed.

The launcher prints the array job ID, campaign directory, source commit, and
log directory, and records them in `submission.txt`. Logs are
`logs/simulation-<job>_<index>.out` and `.err`; array indices are
0 = QBlade, 1 = Spectral VorLap, 2 = Hybrid VorLap.
Preparation and validation receipts are saved beside the campaign manifest.
The QBlade library hash is checked before submission and again on the node.
The job checks the Git commit and relevant tracked runtime files before
starting; keep this checkout at the submitted revision while jobs are queued
or running. Unrelated local paper edits are not staged or altered.

The default reservation is the previously used `flight-cldera`. If it has
expired, select the ordinary batch queue explicitly:

```bash
FIXED_AZIMUTH_RESERVATION="" bash run_headless/submit_fixed_azimuth_chain.sh
```

Optional absolute-path overrides are `PYTHON_EXE`, `QBLADE_ROOT`,
`SOURCE_CAMPAIGN`, and `CAMPAIGN_ROOT`. An existing campaign directory is
rejected; each invocation starts fresh. The launcher derives `VORLAP_ROOT`
from its own Git checkout, avoiding a stale shell variable from an old run.

Check each `results/<model>/fixed_azimuth_audit.json` for `passed: true`, then
download the new results and preparation receipts:

```bash
rsync -av --progress kevmoor@flight-login7:/pscratch/kevmoor/vorlap_runs/u11p25_a147_fixed_RUN_TAG/ \
  /Users/kevmoor/Documents/coderepos/VorLap_Bridge/VorLap/run_headless/runs/u11p25_a147_fixed_RUN_TAG/
```

## Local verification

```bash
.venv/bin/python -m pytest -q examples/fixed_azimuth_20260922/test_campaign.py
bash -n run_headless/submit_fixed_azimuth_chain.sh
bash -n examples/fixed_azimuth_20260922/submit_flight.sj
bash examples/fixed_azimuth_20260922/smoke_docker.sh
```

The Docker test uses a separate, explicitly marked **steady-inflow smoke**
campaign with identical structural startup and checks short native runs for all
three models. It is not turbulence or fatigue evidence, and the driver refuses
to run that smoke campaign as a full production simulation.

Recorded verification is in `verification.json`: all three native models
completed 160 steps after the original structural startup, all 27 requested
channels were finite and complete, and all measured azimuths were exactly
147° (maximum absolute RPM below 5e-16). Ten focused tests passed. Before
switching delivery to Git, an extracted committed-source bundle
prepared all three cases, initialized both coupled adapters, and retained its
input hashes after initialization. The scientific driver and native inputs
are unchanged by the Git/launcher update. The full 660 s
turbulent cases have not been submitted or run by this preparation.

The Git launcher also passed eleven isolated orchestration checks, including
prepare/validate/submission ordering, reservation selection, fresh-root
rejection, validation failure, runtime identity, and queued-job source drift.
These checks use a fake scheduler and do not submit Flight jobs.
