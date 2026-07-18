#!/usr/bin/env bash
set -euo pipefail

# Override these from the environment for a larger model or multi-GPU host.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TRAIN_SAMPLES_PER_FAULT="${TRAIN_SAMPLES_PER_FAULT:-128}"
EVAL_SAMPLES_PER_FAULT="${EVAL_SAMPLES_PER_FAULT:-16}"

python -m training.generate_dataset \
  --train-samples-per-fault "${TRAIN_SAMPLES_PER_FAULT}" \
  --eval-samples-per-fault "${EVAL_SAMPLES_PER_FAULT}"

accelerate launch --num_processes "${NUM_PROCESSES}" --module training.sft \
  --model "${BASE_MODEL}" \
  --output-dir outputs/sft

GRPO_SANDBOX_ARGS=()
if [[ -n "${CRASHDIAG_SANDBOX_URL:-}" ]]; then
  GRPO_SANDBOX_ARGS+=(--sandbox-url "${CRASHDIAG_SANDBOX_URL}")
fi

accelerate launch --num_processes "${NUM_PROCESSES}" --module training.grpo \
  --model outputs/sft \
  --output-dir outputs/grpo \
  "${GRPO_SANDBOX_ARGS[@]}"

python -m training.evaluate \
  --model outputs/grpo \
  --episodes-per-fault 10 \
  --output outputs/evaluation.json
