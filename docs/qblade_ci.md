# QBlade integration checks

The QBlade job runs the qualified QBlade CE 2.0.9.7 Linux x86_64 solver and a
native bridge built from the commit under test. It requires all 74 original
runtime-dependent contract cases and 19 integration/execution-guard cases.
`reproducibility/qblade-test-inventory.json` prevents deleting a test or removing
its marker from silently reducing required coverage. Newly marked tests also run.

The original contract assertions and qualified byte identities are retained.
Many original cases intentionally use an API double to isolate configuration,
restart, provenance and rejection behavior. They are unit/contract evidence.
`tests/test_qblade_integration.py` additionally executes the actual upstream
solver; its successful runs cannot be replaced by those doubles.

## Physical assertions

The fixture is an independent generic cantilever: length 1 m, mass 1 kg/m,
bending stiffness 120 Nm², chord 0.1 m, zero gravity, zero rotor speed and disabled
QBlade aerodynamic loads. A constructed spectrum prescribes known section-normal
and chordwise coefficients. At density 1.225 kg/m³ and wind 4 m/s, its sectional
load scale is 0.98 N/m. No exact paper-case inputs are required.

| Check | Independent reference | Failure exposed |
| --- | --- | --- |
| Static tip displacement | qL⁴/(8EI), with negligible shear correction | Missing, reversed or incorrectly scaled applied load |
| Root bending moment | qL²/2 | Load integration or application errors |
| Initial azimuth 0° and 90° | Explicit global XYZ geometry and load directions | Double rotation, sign or coordinate-frame error |
| Harmonic response at 2 Hz | Literal QBlade LOADINGFILE, bypassing VorLap force/frame/length assembly | Wrong frequency, phase, application timing or force path |
| Mesh and timestep refinement | Independent complex cantilever boundary-value solution | A discretization-dependent accidental match |
| Fault controls | Actual solver runs with exchanged forces dropped, reversed, doubled, axes swapped or delayed | Accepting callback/status/provenance receipts without physical consumption |
| Required execution controls | Deliberately skipped, empty and missing-inventory child pytest runs | A green job without its required tests executing |

The 20-element static displacement differs from continuum theory by about 0.084%;
the 0.2% relative bound plus a 2 µm absolute position bound accommodates nodal
quadrature and QBlade's text-export precision. Root reaction has a separate 0.2%
bound. Harmonic bridge/reference complex error must be below 0.5%, which admits
the small two-way structural-velocity feedback and rejects a single 2 ms force
delay at 2 Hz (about 2.51% complex error). Refinement halves the timestep and
element length in separate comparisons. Time refinement must reduce complex
continuum error by at least 30%, with final error below 1%; static mesh refinement
must reduce displacement error by at least 50%, with final error below 0.04%.
These are verification bounds for this fixture, not universal accuracy
claims for other models. No finite suite guarantees detection of every defect.

## Qualified environment and runtime

`scripts/ci/Dockerfile.qblade` specifies Ubuntu 22.04, Python 3.10.12, GCC 11.4,
CMake 3.22.1 and the numerical versions in
`reproducibility/requirements-qblade-ci.txt`. Compiler and Python development
package versions are pinned. Updating an unavailable pin requires explicit
requalification; the job does not silently fall back to a different toolchain.

The solver, SIL wrapper, Chrono library, native bridge and complete vendor
Libraries/wrapper fingerprint are recorded in
`reproducibility/qblade-ci-runtime.json`. The installer checks the archive SHA-256,
rejects unsafe paths and links, and removes any archived VorLap bridge. The CI
entry point builds the bridge afresh and verifies its original qualification
hash. It does not cache or reuse a prebuilt bridge.

Obtain QBlade separately and retain its notices. The official download currently
requires an interactive CAPTCHA. The existing installed 2.0.9.7 runtime can be
packaged for authorized private CI access; vendor binaries are not committed to
this public repository or included in its test artifacts.

To run the same checks with an installed runtime on Linux:

```bash
docker build --platform linux/amd64 -f scripts/ci/Dockerfile.qblade -t vorlap-qblade-ci .
docker run --rm --platform linux/amd64 -v "$PWD:/work/VorLap" \
  vorlap-qblade-ci bash scripts/ci/run_qblade_checks.sh
```

Start from a clean checkout. The entry point refuses a reused native build
directory. To repeat locally, remove only the generated `build/qblade-ci-native`
and `build/qblade-ci-evidence` directories first. A corporate proxy may require
an additional build secret: `--secret id=local_ca,src=/path/to/ca.crt`; the
certificate is mounted temporarily and is not copied into the image.

## Evidence and scope

`build/qblade-ci-evidence/` contains runtime identities, source commit, package
and system versions, JUnit results, every selected QBlade node and its complete
setup/call/teardown outcomes, and the generic physical-case inputs, results and
source/input hashes. Fault cases are explicitly labeled. Required mode fails on
skips, xfails, missing selections and incomplete execution.

Paper-only historical integrations remain separate: they require the matching
manuscript checkout and any exact case inputs from the authors. Installing
QBlade does not establish access to those inputs or qualify replacement inputs
as the historical case. Collection skips for absent manuscript modules are not
counted as passed QBlade cases; the required execution receipt identifies every
selected QBlade case.

The hosted job reads `QBLADE_CI_RUNTIME_REPOSITORY` and an immutable 40-character
`QBLADE_CI_RUNTIME_REF`, using `QBLADE_CI_DEPLOY_KEY` with read-only access to that
private runtime store. Its public source checkout does not retain credentials.
Only the verification-evidence directory is uploaded; the private checkout and
vendor binaries are excluded. The Docker build context includes only the pinned
dependency list, even when local paper files or archives exist beside the code.

Fork pull requests do not receive secrets. Their QBlade job fails with an access
message rather than skipping tests or claiming success. A maintainer must review
and run the change from a trusted repository branch before accepting it. The
workflow does not use `pull_request_target` to execute untrusted code with secrets.
