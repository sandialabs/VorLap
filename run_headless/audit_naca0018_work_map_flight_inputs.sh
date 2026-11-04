#!/bin/bash
# Read-only Flight audit for the NACA0018 prescribed-motion work-map inputs.

set -uo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
NALU_MESH="${NALU_MESH:-/gpfs/kevmoor/airfoil_vortexshedding_mesh_run_post_nalu/mesh_gen_scripts/grids/NACA0018-90.exo}"
NALU_EXECUTABLE="${NALU_EXECUTABLE:-}"
EXPECTED_NALU_COMMIT="ffa9de729df2a11b5241fdeb7628e7fab9f48f9b"
KNOWN_FLIGHT_INTEL_BIN="/projects/global/toss4/compilers/intel/intel_2021/oneapi/compiler/2021.3.0/linux/bin/intel64"
KNOWN_FLIGHT_OPENMPI_ROOT="/opt/toss/openmpi/4.1/intel"

failures=0
record_failure() {
  printf 'FAIL: %s\n' "$*" >&2
  failures=$((failures + 1))
}

printf 'VORLAP_ROOT=%s\n' "${VORLAP_ROOT}"
printf 'SYSTEM=%s %s\n' "$(uname -s 2>/dev/null || true)" "$(uname -m 2>/dev/null || true)"
if [[ -d "${VORLAP_ROOT}/.git" ]]; then
  printf 'VORLAP_HEAD=%s\n' "$(git -C "${VORLAP_ROOT}" rev-parse HEAD 2>/dev/null || printf unavailable)"
else
  record_failure "VorLap checkout unavailable at ${VORLAP_ROOT}"
fi

python_exe=""
for candidate in "${PYTHON_EXE:-}" "${VORLAP_ROOT}/.venv/bin/python" python3; do
  [[ -n "${candidate}" ]] || continue
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]] || continue
    resolved="$(readlink -f "${candidate}" 2>/dev/null || true)"
  else
    resolved="$(command -v "${candidate}" 2>/dev/null || true)"
  fi
  [[ -n "${resolved}" ]] || continue
  if PYTHONPATH="${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${resolved}" -c \
    'import h5py, scipy, yaml; from examples.ehrmann2014_naca0018 import nalu_prescribed_motion' \
    >/dev/null 2>&1; then
    python_exe="${resolved}"
    break
  fi
done
if [[ -n "${python_exe}" ]]; then
  printf 'PYTHON_EXE=%s\n' "${python_exe}"
else
  record_failure "No Python runtime can import the Nalu campaign generator and its dependencies"
fi

if [[ -z "${NALU_EXECUTABLE}" ]]; then
  NALU_EXECUTABLE="$(command -v naluX 2>/dev/null || true)"
fi
if [[ -z "${NALU_EXECUTABLE}" ]] && command -v spack >/dev/null 2>&1; then
  nalu_prefix="$(spack location -i nalu-wind@2.2.1 2>/dev/null || true)"
  if [[ -n "${nalu_prefix}" && -x "${nalu_prefix}/bin/naluX" ]]; then
    NALU_EXECUTABLE="${nalu_prefix}/bin/naluX"
  fi
fi
if [[ -n "${NALU_EXECUTABLE}" ]]; then
  NALU_EXECUTABLE="$(readlink -f "${NALU_EXECUTABLE}" 2>/dev/null || true)"
fi
if [[ -n "${NALU_EXECUTABLE}" && -x "${NALU_EXECUTABLE}" ]]; then
  printf 'NALU_EXECUTABLE=%s\n' "${NALU_EXECUTABLE}"
  printf 'NALU_EXECUTABLE_SHA256=%s\n' "$(sha256sum "${NALU_EXECUTABLE}" | awk '{print $1}')"
  printf 'EXPECTED_NALU_GIT_COMMIT=%s\n' "${EXPECTED_NALU_COMMIT}"
else
  record_failure "Official Nalu-Wind v2.2.1 naluX was not resolved from NALU_EXECUTABLE, PATH, or Spack"
fi

if command -v spack >/dev/null 2>&1; then
  printf 'SPACK=%s\n' "$(command -v spack)"
  spack --version 2>/dev/null | sed 's/^/SPACK_VERSION=/'
else
  printf 'SPACK=unavailable\n'
fi

# Emit enough compiler/MPI evidence to prepare a pinned native build without
# guessing a module or PMI/PMIx provider.  These are diagnostic-only checks;
# an already-qualified naluX remains the preferred path above.
printf 'KNOWN_FLIGHT_INTEL_BIN=%s\n' "${KNOWN_FLIGHT_INTEL_BIN}"
printf 'KNOWN_FLIGHT_OPENMPI_ROOT=%s\n' "${KNOWN_FLIGHT_OPENMPI_ROOT}"
for command_name in icc icpc gcc g++ mpicc mpicxx mpirun ompi_info; do
  command_path="$(command -v "${command_name}" 2>/dev/null || true)"
  if [[ -n "${command_path}" ]]; then
    printf '%s=%s\n' "$(printf '%s' "${command_name}" | tr '[:lower:]' '[:upper:]')" "$(readlink -f "${command_path}" 2>/dev/null || printf '%s' "${command_path}")"
  else
    printf '%s=unavailable\n' "$(printf '%s' "${command_name}" | tr '[:lower:]' '[:upper:]')"
  fi
done
if command -v mpicc >/dev/null 2>&1; then
  printf 'MPICC_SHOW=%s\n' "$(mpicc -show 2>/dev/null || true)"
fi
if command -v ompi_info >/dev/null 2>&1; then
  ompi_info --version 2>/dev/null | sed 's/^/OMPI_VERSION=/'
  ompi_info --parsable 2>/dev/null \
    | grep -E 'mca:(plm:slurm|pmix:)' \
    | sed 's/^/OMPI_COMPONENT=/' \
    || true
fi
if command -v srun >/dev/null 2>&1; then
  srun --mpi=list 2>&1 | sed 's/^/SLURM_MPI=/' || true
else
  printf 'SRUN=unavailable\n'
fi
if type module >/dev/null 2>&1; then
  module -t list 2>&1 | sed 's/^/LOADED_MODULE=/' || true
fi
if command -v apptainer >/dev/null 2>&1; then
  apptainer --version 2>/dev/null | sed 's/^/APPTAINER_VERSION=/'
elif command -v singularity >/dev/null 2>&1; then
  singularity --version 2>/dev/null | sed 's/^/SINGULARITY_VERSION=/'
else
  printf 'APPTAINER=unavailable\n'
fi

printf 'NALU_MESH=%s\n' "${NALU_MESH}"
if [[ -f "${NALU_MESH}" && ! -L "${NALU_MESH}" ]]; then
  printf 'NALU_MESH_SHA256=%s\n' "$(sha256sum "${NALU_MESH}" | awk '{print $1}')"
  printf 'NALU_MESH_BYTES=%s\n' "$(stat -c %s "${NALU_MESH}")"
else
  record_failure "Exact regular non-symlink Exodus mesh unavailable at ${NALU_MESH}"
fi

if command -v scontrol >/dev/null 2>&1; then
  if scontrol show reservation flight-cldera >/dev/null 2>&1; then
    printf 'FLIGHT_CLDERA_RESERVATION=available\n'
  else
    printf 'FLIGHT_CLDERA_RESERVATION=unavailable\n'
  fi
fi

if (( failures > 0 )); then
  printf 'AUDIT_STATUS=blocked\n'
  exit 2
fi
printf 'AUDIT_STATUS=ready_for_scaling_preflight\n'
