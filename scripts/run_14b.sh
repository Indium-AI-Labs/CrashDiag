#!/usr/bin/env bash
# Matched 14B Docker baseline + GRPO. Does not source env.txt.
#
#   bash scripts/run_14b.sh            # baseline gate, then GRPO
#   bash scripts/run_14b.sh baseline
#   bash scripts/run_14b.sh grpo
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/env.txt"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}" >&2
  exit 1
fi
export CRASHDIAG_ENV_FILE="${ENV_FILE}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
cd "${REPO_ROOT}"
python3 -m pip install -e ".[train]"
python3 -u -m training.run_14b "${1:-all}"
