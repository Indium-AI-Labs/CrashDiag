# CrashDiag v6 notebook workflow

The maintained notebooks use one model, `Qwen/Qwen2.5-3B-Instruct`, and train
directly with GRPO. No SFT checkpoint or `CRASHDIAG_SFT_RUN_ID` is required.

## Notebooks

`notebooks/qwen2.5_3b/` contains:

- `eval_base.ipynb` — held-out base-model replay evaluation.
- `grpo.ipynb` — direct-from-base GRPO followed by standalone evaluation.
- `eval_grpo.ipynb` — evaluate an existing GRPO artifact.
- `run_all.ipynb` — generate data, evaluate the base model, train with GRPO,
  evaluate the adapter, and display/upload reports.

All committed notebooks have empty execution counts and outputs so credentials,
machine paths, and logs are not retained in Git.

## Required environment

The notebooks read `env.txt` from their launch directory unless
`CRASHDIAG_ENV_FILE` overrides it. On Kaggle they can load the same keys from
`kaggle_secrets.UserSecretsClient`.

Required:

- `HF_TOKEN`
- `CRASHDIAG_DATASET_RUN_ID` (except when `run_all.ipynb` generates a new stage)
- `CRASHDIAG_SANDBOX_URL`
- `CRASHDIAG_SANDBOX_TOKEN` or `CRASHDIAG_API_TOKEN`

Optional:

- `CRASHDIAG_GRPO_RUN_ID`
- `CRASHDIAG_GRPO_EVAL_RUN_ID`
- `CRASHDIAG_BASE_EVAL_RUN_ID`
- `CRASHDIAG_HF_BUCKET_ID`
- `CRASHDIAG_SOURCE_COMMIT`
- `CRASHDIAG_REPO_URL`
- `CRASHDIAG_WORKDIR`
- `CRASHDIAG_GRPO_MAX_STEPS`

Use [`.env.example.grpo`](../.env.example.grpo) as the starting template.

## Stage handoff

1. Dataset generation creates and mechanically validates the answer-free GRPO
   train/eval JSONL files, then uploads a `datasets` stage.
2. Base evaluation replays every held-out row and uploads `base-eval` reports.
3. Direct GRPO downloads the dataset stage, starts from Qwen2.5-3B-Instruct,
   and uploads the final LoRA adapter and trainer reports in a `grpo` stage.
4. Standalone adapter evaluation downloads the final adapter, replays the same
   held-out rows, and uploads a `grpo-eval` stage.

## Final-run configuration

The released run used BF16 on one NVIDIA L4, batch size 4, gradient
accumulation 2, four generations per prompt, learning rate `5e-6`, 5% warmup,
and 832 optimizer steps. Training and standalone evaluation both use a 96-token
completion limit and strict-JSON-only training reward.

## Evaluation behavior

- Base and adapter evaluation use the same rows and `--no-few-shot`. The
  retained historical base result predates this alignment and used the generic
  format demonstration plus a 64-token limit; the released adapter used no
  demonstration and a 96-token limit.
- `training.evaluate_jsonl` records exact resolution, mean mechanical reward,
  strict JSON, backend errors, per-fault results, and per-profile results.
- Generated workflows are executed; prose plausibility is never graded.
