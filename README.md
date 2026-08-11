# CrashDiag

CrashDiag is a mechanically verified environment for training an infrastructure
repair policy. A policy selects one bounded JSON action; a disposable sandbox
executes it and determines reward from real state and health checks. No LLM
grader is used.

The fresh workflow trains a single model, `Qwen/Qwen3-14B`, with NF4 4-bit
QLoRA on two Kaggle T4 GPUs. Qwen3 thinking is disabled in training and
evaluation so the policy produces only the required action JSON.

## Included

- 18 injectable, mechanically repairable fault families.
- Mock and HTTP sandbox backends, Docker deployment, and public Compose overlay.
- Standard SFT/GRPO data generator plus redacted/noisy hard GRPO curriculum.
- QLoRA SFT, GRPO, exact held-out evaluation, reports, and private Hugging Face
  Storage Bucket artifact persistence.
- Four notebooks only: base evaluation, SFT, SFT evaluation, and GRPO at
  `notebooks/qwen3_14b/`.

## Setup

Keep secrets in the ignored `.env` or Kaggle Secrets, never in code:

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

Generate fresh standard data (64 train and 8 eval variations per fault):

```powershell
python -m pip install -e ".[artifacts]"
python -m training.generate_dataset --train-samples-per-fault 64 --eval-samples-per-fault 8 --seed 42
```

See [data/roadmap.md](data/roadmap.md) for the complete deployment, base-eval,
SFT, GRPO, hard-curriculum, and final-evaluation sequence.

## Validation

```powershell
python smoke_test.py
python -m unittest discover -s tests -v
```
