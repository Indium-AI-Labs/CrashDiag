# CrashDiag environment-only model evaluation roadmap

## Objective

Keep CrashDiag as a deterministic repair environment and evaluate fixed model
policies against it. There is no planner, tool-using agent, autonomous loop,
or LLM judge in this roadmap. Each evaluator call renders a signed prompt,
asks one model for one JSON action at temperature zero, and mechanically
executes that action in the sandbox.

## Verified starting state

The private HF bucket contains these signed inputs and results:

| Item | Location or result |
| --- | --- |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Parent SFT adapter | `runs/20260719T113724Z-dataset-b26381b116bc/sft` |
| GRPO adapter | `runs/20260720T164228Z-grpo-hard-7aa31d7f3710/grpo-hard` |
| Signed hard split | `runs/20260720T164228Z-grpo-hard-7aa31d7f3710/datasets/grpo_hard_eval.jsonl` (192 rows) |
| Signed schema-v1 regression split | parent SFT `datasets/grpo_eval.jsonl` (96 rows) |
| Existing SFT hard result | 158/192 (82.29%) |
| Existing GRPO hard result | 175/192 (91.15%) |
| Existing GRPO regression result | 96/96 (100%) |

The existing parent-SFT/GRPO comparison is useful history, but the base-model
comparison should use one pinned evaluator revision and identical generation
settings for every model in the new roster.

## Phase 1: establish the base-Qwen hard baseline

1. Open `notebooks/eval_base_qwen_hard.ipynb` in a fresh Kaggle GPU session.
2. Enable GPU and Internet, then attach `HF_TOKEN` and
   `CRASHDIAG_SANDBOX_TOKEN` as Kaggle secrets.
3. Run every cell in order. The notebook pins the hard data source commit,
   verifies all 192 rows, loads the base Qwen model, and uploads a signed result
   to the separate `base-qwen-hard-7aa31d7f3710` run.
4. Record the printed `success_rate`, strict-JSON rate, backend-error rate, and
   per-fault report. Stop if the backend-error rate is nonzero; that is an
   environment failure, not a model score.

Do not run the full-GRPO cell in `notebooks/grpo_hard.ipynb` while establishing
this baseline: that cell deliberately overwrites the completed GRPO-derived
stages under the hard training run.

## Phase 2: evaluate the trained models under the same contract

Create one new evaluation-only roster run after the base hard result exists.
It should have independent stages for every model and split:

| Stage | Model | Dataset |
| --- | --- | --- |
| `base-qwen-hard` | `Qwen/Qwen2.5-1.5B-Instruct` | 192-row hard split |
| `parent-sft-hard` | signed parent SFT adapter | 192-row hard split |
| `grpo-hard` | signed GRPO adapter | 192-row hard split |
| `base-qwen-regression` | base Qwen | 96-row schema-v1 regression split |
| `parent-sft-regression` | signed parent SFT adapter | 96-row schema-v1 regression split |
| `grpo-regression` | signed GRPO adapter | 96-row schema-v1 regression split |

Every stage must use `training.evaluate_jsonl` with the same pinned evaluator
commit, `--max-new-tokens 96`, automatic GPU precision, temperature zero, and
the same live sandbox URL. Download adapters only from their signed bucket
stages and verify their manifests before loading them.

## Phase 3: publish the comparison, not a new training result

Generate one signed comparison artifact that includes:

- overall resolved episodes and success rate for each model/split;
- strict JSON rate and sandbox backend-error rate;
- per-fault success rates and absolute deltas from base Qwen;
- model identifier, adapter SHA-256 where applicable, evaluator commit, data
  manifest SHA-256, and generation settings.

Only compare results whose data manifest, evaluator commit, model version, and
generation settings match. Keep the raw per-row mechanical results in the
artifact so aggregate claims can be reproduced.

## Decision gates

1. If base-Qwen has nonzero backend errors, repair the environment integration
   and repeat that stage; do not compare it with trained models.
2. If SFT or GRPO has a data or manifest mismatch, discard that stage and
   download the signed artifact again.
3. If GRPO does not outperform the base and SFT baselines on the hard split,
   inspect the per-fault deltas before deciding whether to retrain.
4. Retrain only after this roster evaluation is complete. Use a fresh training
   run ID for a new experiment; preserve the currently complete hard run as an
   audit reference.
