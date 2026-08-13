# CrashDiag v5 Notebook Workflow

v5 uses a single model: `Qwen/Qwen2.5-3B-Instruct`. The multi-model Qwen2.5 sweep
notebooks (`qwen2.5_14b`, `7b`, `1.5b`, `0.5b` and the `*_all.ipynb` notebooks) are
removed.

## Notebooks

Only `notebooks/qwen2.5_3b/` remains:

- `sft.ipynb` — QLoRA SFT.
- `eval_sft.ipynb` — held-out SFT evaluation.
- `grpo.ipynb` — GRPO smoke then full, followed by eval.
- `eval_grpo.ipynb` — held-out GRPO evaluation.

Regenerate them after any template change:

```powershell
python scripts/generate_notebooks.py
```

## Env variables

Use `.env.example.sft` and `.env.example.grpo` as the copy-paste templates.

Required (SFT):

- `HF_TOKEN`
- `CRASHDIAG_DATASET_RUN_ID`

Required (GRPO / eval, in addition to the above):

- `CRASHDIAG_SFT_RUN_ID`
- `CRASHDIAG_SANDBOX_URL`
- `CRASHDIAG_SANDBOX_TOKEN`

Optional:

- `CRASHDIAG_CURRICULUM` (default `v5`)
- `CRASHDIAG_SOURCE_COMMIT`
- `CRASHDIAG_REPO_URL`
- `CRASHDIAG_ENV_FILE`
- `CRASHDIAG_WORKDIR`
- `CRASHDIAG_ALL_RUN_ID`

## Stage handoff

1. Dataset generation uploads one `datasets` stage. Its run ID becomes
   `CRASHDIAG_DATASET_RUN_ID`.
2. `sft.ipynb` uploads an `sft` stage; its run ID becomes `CRASHDIAG_SFT_RUN_ID`.
3. `grpo.ipynb` downloads `datasets` + `sft`, runs a `grpo-smoke` stage then the full
   `grpo` stage, then evaluates.
4. `eval_sft.ipynb` / `eval_grpo.ipynb` upload `sft-eval` / `grpo-eval` stages.

## Kaggle secrets

Eval notebooks (`eval_sft.ipynb`, `eval_grpo.ipynb`) load secrets via
`kaggle_secrets.UserSecretsClient` when available and do not hard-fail when the local
`env.txt` / `.env` is absent, so the same notebook runs unchanged on Kaggle.

## Evaluation behavior

- Base-model and eval notebooks request the v5 workflow contract and evaluate with
  `--no-few-shot` for post-training checkpoints (those should have internalized the
  contract).
- `evaluate_jsonl.py` replays each row exactly and reports success rate,
  strict-JSON rate, backend-error rate, per-task and per-profile success, and
  sub-fault progress.
