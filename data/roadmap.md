# CrashDiag clean-start, environment-only roadmap

## Goal

Start with an empty private bucket and keep CrashDiag as a deterministic repair
environment. Evaluation is fixed-policy inference: one prompt, one JSON action,
and mechanical sandbox execution. Do not use `training.evaluate`, `BlueAgent`,
planners, tool loops, or LLM judges.

## Order of work

1. **Bootstrap the environment.** Deploy the sandbox and verify its health,
   schema support, and authentication before any model result is recorded.
2. **Generate a fresh standard dataset.** Run `training.generate_dataset` on a
   trusted machine. Save the printed `RUN_ID` and `SOURCE_COMMIT`; its signed
   `datasets` stage contains `grpo_eval.jsonl` with 96 answer-free rows.
3. **Evaluate base Qwen first.** Open `notebooks/eval_base_qwen_hard.ipynb` in
   Kaggle, set `CRASHDIAG_DATASET_RUN_ID`, `CRASHDIAG_DATASET_SOURCE_COMMIT`,
   and a new `CRASHDIAG_BASE_QWEN_RUN_ID`, then run all cells. It evaluates
   `Qwen/Qwen2.5-1.5B-Instruct` with `training.evaluate_jsonl` and uploads a
   separate signed `base-qwen-evaluation` stage.
4. **Train SFT.** Run `notebooks/sft.ipynb` using the same fresh dataset run.
   It produces the parent SFT adapter without completing the whole dataset run.
5. **Evaluate SFT.** In a new evaluation-only run, download and verify the SFT
   adapter, then evaluate it on the exact same signed `grpo_eval.jsonl` with
   the same evaluator commit, precision, and `--max-new-tokens 96` as base
   Qwen.
6. **Generate hard data and train GRPO.** The hard-data generator requires the
   signed SFT parent, so this cannot happen before step 4. Generate a fresh
   hard-data run, run calibration and smoke checks, then train GRPO in a new
   training run.
7. **Evaluate GRPO and compare.** Evaluate base Qwen, SFT, and GRPO on both the
   signed hard split and the 96-row regression split. Use separate stages for
   every model/split and publish one signed comparison report.

## Required fairness controls

Every model in one comparison must use the same signed dataset manifest,
evaluator commit, sandbox deployment, `--max-new-tokens 96`, and deterministic
temperature-zero generation. Record model name, adapter SHA-256 when present,
data manifest SHA-256, per-row mechanical results, strict-JSON rate, and
backend-error rate.

## Decision gates

- Stop and repair the environment if backend-error rate is nonzero.
- Do not compare a stage whose signed dataset or model manifest differs.
- Treat base-Qwen, SFT, and GRPO as separate fixed policies; only their
  mechanical resolution rates and per-fault deltas determine comparison.
- Preserve each completed experiment run. Start a new run ID for any retrain
  rather than overwriting a benchmark result.
