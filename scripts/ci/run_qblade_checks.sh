#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
QBLADE_CI_ROOT="$PWD/run_headless/QBladeCE_2.0.9.7"
QBLADE_CI_EVIDENCE="$PWD/build/qblade-ci-evidence"
mkdir -p "$QBLADE_CI_EVIDENCE"
python3 scripts/verify_qblade_runtime.py "$QBLADE_CI_ROOT" --without-bridge
# A fresh build directory and no prebuilt bridge: only current source is tested.
if [[ -e build/qblade-ci-native ]]; then
    echo 'Refusing to reuse an existing CI native build directory.' >&2
    exit 1
fi
PYTHON_EXE=/usr/bin/python3 bash scripts/build_qblade_external_linux.sh \
    build/qblade-ci-native "$QBLADE_CI_ROOT/ControllerFiles"
python3 scripts/verify_qblade_runtime.py "$QBLADE_CI_ROOT" \
    --receipt "$QBLADE_CI_EVIDENCE/runtime.json"
python3 -m pip freeze > "$QBLADE_CI_EVIDENCE/python-environment.txt"
dpkg-query -W > "$QBLADE_CI_EVIDENCE/system-environment.txt"
git rev-parse HEAD > "$QBLADE_CI_EVIDENCE/commit.txt"
git diff --exit-code
export VORLAP_QBLADE_TEST_RECEIPT="$QBLADE_CI_EVIDENCE/test-execution.json"
export VORLAP_QBLADE_PHYSICS_EVIDENCE="$QBLADE_CI_EVIDENCE/physics"
python3 -m pytest --require-qblade -m requires_qblade \
    --qblade-inventory=reproducibility/qblade-test-inventory.json \
    --junitxml="$QBLADE_CI_EVIDENCE/tests.xml" -ra
