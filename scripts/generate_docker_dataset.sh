#!/usr/bin/env bash
# Generate the schema-v6 6,656 / 832 Docker-observation dataset.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/env.txt" ]]; then
  export CRASHDIAG_ENV_FILE="${REPO_ROOT}/env.txt"
elif [[ -f "${REPO_ROOT}/.env" ]]; then
  export CRASHDIAG_ENV_FILE="${REPO_ROOT}/.env"
fi
export CRASHDIAG_SANDBOX_BACKEND=docker
python3 -m pip install -e ".[artifacts]"
python3 -m training.generate_dataset \
  --train-samples-per-fault 128 \
  --eval-samples-per-fault 16 \
  --seed 42 \
  --sandbox-backend docker \
  "$@"
