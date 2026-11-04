# U11.25 slow-azimuth matched three-model follow-up

Run **11.25 m/s**, seed **4433456**, at prescribed **1/30 RPM =
0.0333333333333 RPM** for QBlade, Spectral VorLap, and Hybrid VorLap.
The angular speed is **0.2°/s**, giving **120° in the retained 600 s**.
Each case runs **660 s at 0.0075 s** (88,000 integration steps), retaining
60–660 s for comparison with the earlier parked and 1-RPM cases.

The three-bladed rotor repeats its geometry after 120° with the blade labels
permuted. Accordingly, the export includes **all three blades**, at the same
four stations z/H = 0.23, 0.50, 0.77, and 0.94: three local forces and three
local moments per station. This gives 72 load channels plus time, RPM, and
rotor azimuth, for **75 requested channels**. A single blade only sweeps a
120° sector; the three blade records together cover all relative orientations.
Turbulence, wake history, structural motion, and response growth still depend
on time, so this sweep does not establish a stationary worst-angle equilibrium.

## Rotation and phase convention

`PRESCRIBETYPE=1`, `RPMPRESCRIBED=1/30`, `INITIAL_AZIMUTH=147`, and
`RELAXSTEPS=0` impose the new rotation. The 147° input continues the
fixed-angle case family. QBlade's original structural initialization can
advance the rotor before the first recorded sample; **147° is an input, not
an imposed measured time-zero angle**. No old 1-RPM startup phase is added.
The output audit records the actual time-zero, 60-s, and final azimuths.

The full 660 s represents a nominal 132° sweep. Exports contain samples
0–659.9925 s, so the expected first-to-last angle difference is **131.9985°**.
Within the retained 60–659.9925 s samples, the difference is **119.9985°**;
the nominal 600-s retained interval is 120°. These endpoints follow the
existing simulation/export convention and do not indicate a missing step.

Acceptance requires measured RPM within **1e-5 RPM** of 1/30 and every
unwrapped angle within **0.01°** of the 0.2°/s trajectory anchored to the
first measured angle. It also requires every requested load channel to be
finite and complete, all 88,000 structural steps, and one SIL update per step
for the two coupled models.

## Preserved physics and explicit modal approximation

The preparer rebuilds clean source inputs from the reviewed archive, verifies
their physical hashes against the previous matched campaign, and never copies
old simulation state. Turbine geometry, NACA0018 spectrum, damping, the
`parked-reduced-v2` wake settings, 10-s structural startup, 5-s additional
damping, and the original turbulent wind bytes remain the same. The wind is
`dlc6p4_U11p25ms_s4433456.bts`, SHA-256
`dd1608a5bf2dc5415858fe89e97bcecc332eb515b3497dd7b66fb04d735d74b2`.

Hybrid VorLap retains the reviewed **1-RPM response basis**, while the full
QBlade structural solver uses actual prescribed **1/30 RPM**. The archived
0/1-RPM eigensolutions differ in frequency by 0.000112% and 0.001778%, with
action-shape MAC values 0.999956 and 0.999743 for modes 6 and 12. This close
endpoint agreement motivates the slow-speed approximation; no eigensolution
was recomputed at 1/30 RPM. Preparation requires frequency differences below
0.1% and MAC above 0.999. The case-specific basis-speed tolerance is
`abs(1 - 1/30) + 1e-5 RPM`; the independent measured-rotation audit enforces
the much tighter actual rotor-speed requirement above. The policy and evidence
are saved in `inputs/modal_basis_transfer.json` and the manifest.

## Pull and submit on Flight

Use the existing checkout, Python environment, QBlade binary, Anaconda module,
and one-command launcher from the recent successful runs:

```bash
cd /pscratch/kevmoor/vorlap
git switch response-aware-viv-fatigue
git pull --ff-only origin response-aware-viv-fatigue
bash run_headless/submit_slow_azimuth_chain.sh
```

The launcher uses `.venv/bin/python`, `run_headless/QBladeCE_2.0.9.7`,
`anaconda3/2025.12-2`, and source campaign
`/pscratch/kevmoor/vorlap_runs/hvawt_complete_dlc_three_path_v3`. It creates
`/pscratch/kevmoor/vorlap_runs/u11p25_rpm0p033333_<UTC timestamp>`, prepares
and validates all three inputs, then submits one exclusive 112-CPU node per
model. The array has three tasks that may run concurrently; its indices are
0 = QBlade, 1 = Spectral VorLap, and 2 = Hybrid VorLap.

The launcher prints and saves the array job ID, campaign directory, source
commit, and log directory in `submission.txt`. Logs are
`logs/simulation-<job>_<index>.out` and `.err`. Existing output directories
are rejected. The QBlade hash and Git/runtime identity are checked before
submission and again on the compute node; keep the checkout at the submitted
revision while jobs are queued or running. Checkpoints are diagnostic only;
there is no automatic continuation or reuse of old results.

The default reservation remains `flight-cldera`. If that reservation has
expired, explicitly use the ordinary queue:

```bash
SLOW_AZIMUTH_RESERVATION="" bash run_headless/submit_slow_azimuth_chain.sh
```

Optional absolute-path overrides are `PYTHON_EXE`, `QBLADE_ROOT`,
`SOURCE_CAMPAIGN`, and `CAMPAIGN_ROOT`. The launcher derives `VORLAP_ROOT`
from its own checkout to avoid a stale environment variable. If the shell
still exports a previous campaign directory, run `unset CAMPAIGN_ROOT`
before launching to select a new timestamped directory automatically.

## Minimal download after completion

Replace `RUN_TAG` with the timestamp printed by the launcher, then run on the
Mac. These filters retain full-resolution loads and coupling histories plus
small metadata, without copying checkpoints, case inputs, or wind files.

```bash
campaign=u11p25_rpm0p033333_RUN_TAG
local_root=/Users/kevmoor/Documents/coderepos/VorLap_Bridge/VorLap/run_headless/runs
mkdir -p "${local_root}/${campaign}"
rsync -avz --partial --progress \
  --include='/campaign_manifest.json' \
  --include='/source_git_commit.txt' \
  --include='/results/' --include='/results/*/' \
  --include='/results/*/loads*.txt' --include='/results/*/loads*.csv' \
  --include='/results/*/*.json' --exclude='*' \
  "kevmoor@flight-login7:/pscratch/kevmoor/vorlap_runs/${campaign}/" \
  "${local_root}/${campaign}/"
```

Check `results/<model>/slow_azimuth_audit.json` for `passed: true` before
using production results. Compare models over the same physical time window
and use actual recorded azimuths when interpreting load peaks.

## Local verification

```bash
.venv/bin/python -m pytest -q examples/slow_azimuth_20260924/test_campaign.py
bash -n run_headless/submit_slow_azimuth_chain.sh
bash -n examples/slow_azimuth_20260924/submit_flight.sj
bash examples/slow_azimuth_20260924/smoke_docker.sh
```

The Docker smoke uses the existing AMD64 QBlade image and a separate campaign
with uniform 11.25-m/s inflow. It retains the original structural startup and
checks 160 native steps for all three models, including measured rotation and
all 75 requested load/state channels. It is a runtime/export check, not
turbulent-response or fatigue evidence. A smoke campaign cannot be run as a
full production campaign. Recorded verification is saved in `verification.json`.

Sixteen focused tests and eleven isolated Git/Slurm launcher checks passed.
The three native Docker checks were started, then intentionally stopped before
completion at the user’s request. Native runtime/export acceptance for this
new slow-speed campaign is therefore **not claimed**. No Flight jobs were
submitted during preparation.
