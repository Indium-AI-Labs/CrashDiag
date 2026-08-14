# CrashDiag

CrashDiag is a mechanically verified environment for training an infrastructure
repair policy. A policy selects an ordered JSON workflow; a disposable sandbox
executes the actions in order and determines reward from real state and health
checks. No LLM grader is used.

The fresh workflow trains a single model, `Qwen/Qwen2.5-3B-Instruct`, with BF16
NF4 4-bit QLoRA. Qwen thinking is disabled in training and evaluation so the
policy produces only the required action JSON.

## Included

- 52 injectable, mechanically repairable tasks, each resolved by an ordered
  multi-action workflow.
- 27 repair actions (plus the `wait_and_observe` fallback).
- Mock and HTTP sandbox backends, Docker deployment, and public Compose overlay.
- One consolidated v5 SFT/GRPO curriculum with redacted telemetry, shifted
  baselines, repaired decoy incidents, and mechanically replayable partial-credit
  rewards.
- QLoRA SFT, GRPO, exact held-out evaluation, reports, and private Hugging Face
  Storage Bucket artifact persistence.
- Four notebooks for the single model at `notebooks/qwen2.5_3b/`.

## Documentation

See the [`docs/`](docs/README.md) directory for the task catalog, action space,
curriculum contract, dataset generation, and notebook workflow.

## Setup

Keep secrets in the ignored `.env`, never in code:

```text
HF_TOKEN=...
CRASHDIAG_SANDBOX_URL=https://sandbox.example.com
CRASHDIAG_API_TOKEN=...
```

Deploy the disposable sandbox:

```powershell
docker compose -f compose.yaml -f compose.public.yaml up --detach --build
curl.exe --fail https://sandbox.example.com/healthz
```

Generate fresh v5 data (5,000 train and 25 eval variations per task):

```powershell
python -m pip install -e ".[artifacts]"
python -m training.generate_dataset --train-samples-per-fault 5000 --eval-samples-per-fault 25 --seed 42
```

## Validation

```powershell
python smoke_test.py
python -m unittest discover -s tests -v
```
