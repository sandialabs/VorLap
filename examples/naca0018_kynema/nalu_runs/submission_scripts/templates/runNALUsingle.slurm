#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --time=30:59:59
set -euo pipefail
# Supply account, partition and task count using your site's sbatch options.
# Load your qualified Kynema/Nalu environment before submission.
case_dir="${1:?case directory required}"
shopt -s nullglob
inputs=("${case_dir}"/*.yaml)
[[ ${#inputs[@]} == 1 ]] || { echo "Expected one case YAML" >&2; exit 2; }
srun "${KYNEMA_EXE:-naluX}" -i "${inputs[0]}" -o "${case_dir}/log.out"
