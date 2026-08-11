# Fresh Qwen3-14B workflow

This repository now starts from an empty private artifact bucket and trains one
model only: `Qwen/Qwen3-14B`. Use two Kaggle T4 GPUs with NF4 4-bit QLoRA.
All run IDs are fresh; never reuse a completed run ID.

## 1. Deploy the disposable sandbox

On the host that exposes the sandbox, update to this revision and deploy it:

```powershell
git pull --ff-only origin main
docker compose -f compose.yaml -f compose.public.yaml up --detach --build
curl.exe --fail https://sandbox.devaanshpathak.com/healthz
```

Keep these environment values only in the ignored `.env` or in Kaggle Secrets:
`HF_TOKEN`, `CRASHDIAG_SANDBOX_URL`, and `CRASHDIAG_API_TOKEN`. The token is a
long-lived secret for this disposable sandbox deployment; it is not committed
or passed on a notebook command line.

## 2. Generate the fresh standard data

On a trusted machine with `.env` loaded, install just the artifact tools and
generate the standard 18-fault dataset. This uploads one fresh `datasets`
stage to the private `devaanshpa/CrashDiag` bucket.

```powershell
python -m pip install -e ".[artifacts]"
python -m training.generate_dataset --train-samples-per-fault 64 --eval-samples-per-fault 8 --seed 42
```

Record the printed `RUN_ID` as `CRASHDIAG_DATASET_RUN_ID`. It contains 1,152
training rows and 144 held-out evaluation rows. The 18 tasks are the original
six faults plus missing-secret, feature-flag, Redis, queue, object-storage,
cache, TLS, permissions, migration, database-pool, DNS, and rate-limit faults.

## 3. Base-model evaluation

In Kaggle, create a two-T4 notebook session with Internet enabled. Add
`HF_TOKEN`, `CRASHDIAG_SANDBOX_URL`, and `CRASHDIAG_API_TOKEN` as Secrets and
set `CRASHDIAG_DATASET_RUN_ID` to the recorded dataset run. Run
`notebooks/qwen3_14b/eval_base.ipynb`.

It creates an IST timestamped base-evaluation run ID unless
`CRASHDIAG_BASE_QWEN3_14B_RUN_ID` is set, evaluates all 144 held-out rows with
Qwen3 thinking disabled, and uploads its report to the bucket.

## 4. QLoRA SFT and SFT evaluation

Run `notebooks/qwen3_14b/sft.ipynb` in a fresh two-T4 Kaggle session with the
same dataset run. It uses NF4 QLoRA, completion-only SFT loss, two epochs, and
uploads an IST timestamped `sft` stage. Copy its printed `SFT_RUN_ID`.

Set `CRASHDIAG_SFT_RUN_ID` to that ID and run
`notebooks/qwen3_14b/eval_sft.ipynb`. The notebook downloads the signed SFT
adapter, evaluates the same 144 rows, and uploads a separate `sft-eval` stage.

## 5. GRPO and final evaluation

With both `CRASHDIAG_DATASET_RUN_ID` and `CRASHDIAG_SFT_RUN_ID` set, run
`notebooks/qwen3_14b/grpo.ipynb` in a fresh two-T4 session. It starts from the
SFT adapter, uses two-process `accelerate` plus NF4 QLoRA, runs a 24-step smoke
stage, then a separate 96-step GRPO stage using four generations.

For the harder redacted/noisy curriculum, generate a new hard dataset after SFT
has completed:

```powershell
python -m training.generate_grpo_hard --parent-sft-run-id <SFT_RUN_ID> --train-samples-per-fault 24 --eval-samples-per-fault 8 --seed 42
```

That run contains 432 hard training rows and 144 hard evaluation rows across
the same 18 fault families. Evaluate the GRPO adapter with `evaluate_jsonl`
using `--load-in-4bit` against both the normal and hard held-out datasets; each
evaluation is uploaded under its own fresh run ID.
