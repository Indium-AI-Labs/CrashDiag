# CrashDiag v5 Dataset Generation

The v5 generator produces 52-task, multi-action datasets. The target train set is
~260k rows (5,000 per task); generation uses chunked streaming writes and
process-level parallelism rather than building the full row list in memory.

## CLI

```powershell
python -m training.generate_dataset `
  --train-samples-per-fault 5000 `
  --eval-samples-per-fault 25 `
  --seed 42
```

Defaults:

| flag | default |
|---|---|
| `--train-samples-per-fault` | `5000` |
| `--eval-samples-per-fault` | `25` |
| `--seed` | `42` |

Outputs land under `data/`:

- `sft_train.jsonl`
- `sft_eval.jsonl`
- `grpo_train.jsonl`
- `grpo_eval.jsonl`
- `grpo_summary.json`

## Upload

By default the generator requires a `HF_TOKEN` and uploads all dataset files into one
`datasets` stage of the private `devaanshpa/CrashDiag` bucket with a unique run ID.
Use `--artifact-upload-policy disabled` for local-only generation.

## Mechanical validation

Every row is built by:

1. Constructing a fresh `MockSandbox`.
2. Injecting the workflow's sub-faults.
3. Capturing the observation *before* acting.
4. Executing the expert workflow.
5. Requiring all sub-faults resolved and `health_check().healthy == true`.

The same prompt/seed identity produces the SFT row (with `completion`) and the GRPO
row (answer-free) independently, so an SFT target can never leak into the online-RL
dataset.

## Parallelism model

- The round-robin loop remains stratified: contiguous groups of 52 rows cover every
  task once.
- Rows are written with `write_jsonl` in chunks, and scenario construction is fanned
  out across `ProcessPoolExecutor` workers for the large run.
- Each `sample_seed` is a deterministic, signed int64 derived from base seed, task
  name, and variation index, so any row can be replayed exactly during GRPO reward
  and evaluation.

## Throughput notes

- A 260,000-row train set is multi-GB on disk. Use streaming chunked writes and
  `sync_bucket`/`upload_directory` semantics for upload rather than holding all files
  in memory.
- Train/eval seeds are disjoint by construction; no random split is performed.
