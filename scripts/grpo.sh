#!/usr/bin/env bash
# CrashDiag direct GRPO + evaluation runner.
#
# Start (from anywhere inside the checkout):
#   bash scripts/grpo.sh
#
# Attach to the persistent job:
#   screen -r grpo
#
# Other useful commands:
#   screen -ls                 # list sessions
#   screen -r grpo             # reconnect
#   Ctrl-A, D                  # detach without stopping
#   tail -f grpo.log           # follow the saved log without attaching
#
# Put env.txt at the repository root. It must contain HF_TOKEN,
# CRASHDIAG_DATASET_RUN_ID, CRASHDIAG_SANDBOX_URL, and either
# CRASHDIAG_SANDBOX_TOKEN or CRASHDIAG_API_TOKEN.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/env.txt"
LOG_FILE="${REPO_ROOT}/grpo.log"
SESSION_NAME="${CRASHDIAG_SCREEN_NAME:-grpo}"

if [[ "${1:-}" != "--inside-screen" ]]; then
  if ! command -v screen >/dev/null 2>&1; then
    echo "screen is required. Install it with: sudo apt-get update && sudo apt-get install -y screen" >&2
    exit 1
  fi
  if screen -list | grep -q "\.${SESSION_NAME}[[:space:]]"; then
    echo "screen session '${SESSION_NAME}' is already running. Attach with: screen -r ${SESSION_NAME}" >&2
    exit 1
  fi
  screen -dmS "${SESSION_NAME}" bash -lc \
    "exec env CRASHDIAG_IN_SCREEN=1 bash $(printf '%q' "${BASH_SOURCE[0]}") --inside-screen"
  echo "Started screen session '${SESSION_NAME}'. Attach with: screen -r ${SESSION_NAME}"
  exit 0
fi

cd "${REPO_ROOT}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}" >&2
  exit 1
fi

# Keep the environment in the caller's shell and let grpo_final.py load the
# same root env.txt. Do not print the token or any secret values.
export CRASHDIAG_ENV_FILE="${ENV_FILE}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "repo_root=${REPO_ROOT}"
echo "env_file=${ENV_FILE}"
echo "Installing training packages..."
python3 -m pip uninstall -y torchao >/dev/null 2>&1 || true
python3 -m pip install -e ".[train]"

echo "Starting GRPO and final evaluation..."
python3 -u "${REPO_ROOT}/grpo/grpo_final.py" 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
echo "GRPO process exited with status ${status}"
exit "${status}"
