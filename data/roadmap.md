# Fresh Qwen3-14B workflow

This repository now starts from an empty private artifact bucket and trains one
model only: `Qwen/Qwen3-14B`. Use one 40 GB A100 with BF16 NF4 4-bit QLoRA.
All run IDs are fresh; never reuse a completed run ID.

## 1. Deploy the disposable sandbox

On the host that exposes the sandbox, update to this revision and deploy it:

```powershell
git pull --ff-only origin main
docker compose -f compose.yaml -f compose.public.yaml up --detach --build
curl.exe --fail https://sandbox.devaanshpathak.com/healthz
```

Keep these environment values only in the ignored `.env`:
`HF_TOKEN`, `CRASHDIAG_SANDBOX_URL`, and `CRASHDIAG_API_TOKEN`. The token is a
long-lived secret for this disposable sandbox deployment; it is not committed
or passed on a notebook command line.

## 2. Generate the fresh hardened-v1 data

On a trusted machine with `.env` loaded, install just the artifact tools and
generate the hardened-v1, 18-fault dataset. This uploads one fresh `datasets`
stage to the private `devaanshpa/CrashDiag` bucket. Each prompt contains
redacted telemetry, shifted deployment baselines, repaired decoy incidents,
and a recent ineffective remediation; the hidden repair is still mechanically
replayed from the v1 row identity.

```powershell
python -m pip install -e ".[artifacts]"
python -m training.generate_dataset --train-samples-per-fault 64 --eval-samples-per-fault 8 --seed 42
```

Record the printed `RUN_ID` as `CRASHDIAG_DATASET_RUN_ID`. It contains 1,152
training rows and 144 held-out evaluation rows. The 18 tasks are the original
six faults plus missing-secret, feature-flag, Redis, queue, object-storage,
cache, TLS, permissions, migration, database-pool, DNS, and rate-limit faults.

## 3. Base-model evaluation

On Lightning.ai, create a single-A100 40 GB session with Internet enabled.
Place a normal `.env` in the directory from which Jupyter starts, containing
`HF_TOKEN`, `CRASHDIAG_SANDBOX_URL`, `CRASHDIAG_API_TOKEN`, and
`CRASHDIAG_DATASET_RUN_ID`. Run
`notebooks/qwen3_14b/eval_base.ipynb`.

It creates an IST timestamped base-evaluation run ID unless
`CRASHDIAG_BASE_QWEN3_14B_RUN_ID` is set, evaluates all 144 held-out rows with
Qwen3 thinking disabled, and uploads its report to the bucket.

## 4. QLoRA SFT and SFT evaluation

Run `notebooks/qwen3_14b/sft.ipynb` in a single-A100 Lightning session with the
same dataset run. It uses BF16 NF4 QLoRA, rank 16, a 2048-token ceiling,
completion-only SFT loss, two epochs, and
uploads an IST timestamped `sft` stage. Copy its printed `SFT_RUN_ID`.

Set `CRASHDIAG_SFT_RUN_ID` to that ID and run
`notebooks/qwen3_14b/eval_sft.ipynb`. The notebook downloads the signed SFT
adapter, evaluates the same 144 rows, and uploads a separate `sft-eval` stage.

## 5. GRPO and final evaluation

With both `CRASHDIAG_DATASET_RUN_ID` and `CRASHDIAG_SFT_RUN_ID` set, run
`notebooks/qwen3_14b/grpo.ipynb` in a fresh single-A100 session. It starts from the
SFT adapter, uses BF16 NF4 QLoRA, runs a 24-step smoke
stage, then a separate 96-step GRPO stage using two generations. Copy the
printed `GRPO_RUN_ID` after the final `grpo` stage has uploaded successfully.

Set `CRASHDIAG_GRPO_RUN_ID` to the completed GRPO training run ID and run
`notebooks/qwen3_14b/eval_grpo.ipynb`. It downloads the signed final `grpo`
adapter and the hardened-v1 held-out dataset, evaluates all rows with live
cumulative progress, displays the generated SVG graphs, and uploads the report
under a fresh `grpo-eval` run ID. Set `CRASHDIAG_GRPO_EVAL_RUN_ID` only when an
explicit new evaluation ID is desired; never reuse a completed ID.
