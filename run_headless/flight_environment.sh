#!/usr/bin/env bash
# Source the same Anaconda module used by the previous accepted Flight workflow.
if ! type module >/dev/null 2>&1; then
  for vorlap_module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash /opt/cray/pe/modules/default/init/bash; do
    if [[ -r "${vorlap_module_init}" ]]; then
      source "${vorlap_module_init}"
      break
    fi
  done
fi
type module >/dev/null 2>&1 || { echo 'Flight module command is unavailable' >&2; return 2; }
module load anaconda3/2025.12-2
