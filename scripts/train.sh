#!/usr/bin/env bash
set -euo pipefail

# End-to-end cloud run. Python commands load HF_TOKEN and non-secret settings
# from the repository-root .env; the shell never sources or prints that file.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TRAIN_SAMPLES_PER_FAULT="${TRAIN_SAMPLES_PER_FAULT:-128}"
EVAL_SAMPLES_PER_FAULT="${EVAL_SAMPLES_PER_FAULT:-16}"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || printf 'nogit')"
RANDOM_SUFFIX="$(python -c 'import secrets; print(secrets.token_hex(3))')"
export CRASHDIAG_RUN_ID="${CRASHDIAG_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${GIT_SHA}-${RANDOM_SUFFIX}}"
export CRASHDIAG_ARTIFACT_UPLOAD_POLICY="${CRASHDIAG_ARTIFACT_UPLOAD_POLICY:-required}"

ARTIFACT_LOCAL_ROOT="${CRASHDIAG_ARTIFACT_LOCAL_ROOT:-artifacts}"
LOG_DIR="${ARTIFACT_LOCAL_ROOT}/${CRASHDIAG_RUN_ID}/logs"
mkdir -p "${LOG_DIR}"

sync_partial_logs() {
  python -m training.artifacts upload \
    --stage logs \
    --path "${LOG_DIR}" \
    --partial
}

run_logged() {
  local stage="$1"
  shift
  "$@" 2>&1 | tee "${LOG_DIR}/${stage}.log"
  sync_partial_logs
}

finish_run() {
  local status=$?
  local upload_status=0
  trap - EXIT
  set +e

  python -m training.artifacts upload \
    --stage logs \
    --path "${LOG_DIR}"
  upload_status=$?
  if [[ ${status} -eq 0 && ${upload_status} -ne 0 ]]; then
    status=${upload_status}
  fi

  if [[ ${status} -eq 0 ]]; then
    python -m training.artifacts complete \
      --stages datasets sft grpo evaluation logs
    status=$?
  fi
  exit "${status}"
}

trap finish_run EXIT

run_logged preflight python -m training.artifacts preflight

run_logged dataset python -m training.generate_dataset \
  --train-samples-per-fault "${TRAIN_SAMPLES_PER_FAULT}" \
  --eval-samples-per-fault "${EVAL_SAMPLES_PER_FAULT}"

run_logged sft accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --module training.sft \
  --model "${BASE_MODEL}" \
  --output-dir outputs/sft \
  --save-strategy steps \
  --save-steps 50

GRPO_SANDBOX_ARGS=()
if [[ -n "${CRASHDIAG_SANDBOX_URL:-}" ]]; then
  GRPO_SANDBOX_ARGS+=(--sandbox-url "${CRASHDIAG_SANDBOX_URL}")
fi

run_logged grpo accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --module training.grpo \
  --model outputs/sft \
  --output-dir outputs/grpo \
  "${GRPO_SANDBOX_ARGS[@]}"

run_logged evaluation python -m training.evaluate \
  --model outputs/grpo \
  --episodes-per-fault 10 \
  --output outputs/evaluation.json
