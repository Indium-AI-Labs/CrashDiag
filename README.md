# CrashDiag

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/Indium-AI-Labs/CrashDiag/blob/main/pyproject.toml)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](LICENSE)
[![CI](https://github.com/Indium-AI-Labs/CrashDiag/actions/workflows/ci.yml/badge.svg)](https://github.com/Indium-AI-Labs/CrashDiag/actions/workflows/ci.yml)
[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97-model-CrashDiag--Qwen2.5--3B--GRPO-yellow)](https://huggingface.co/Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO)

CrashDiag is a mechanically verified environment for training and evaluating
infrastructure-repair policies. A policy receives incomplete operational
telemetry, emits a bounded ordered JSON workflow, and executes that workflow in
a disposable sandbox. Reward comes from resulting state and health checks—not
from an LLM judge.

The environment includes 52 fault families, 27 repair actions plus a
`wait_and_observe` fallback, deterministic dataset generation, direct GRPO,
standalone replay evaluation, artifact persistence, reports, notebooks, tests,
and Docker deployment.

## Benchmark integrity notice

The previously released adapter is
[`Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO`](https://huggingface.co/Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO).
It starts directly from `Qwen/Qwen2.5-3B-Instruct`; no SFT checkpoint is used.

**Do not use the historical v5 scores as valid benchmark results.** The v5 noisy
profiles could execute `restart_app` as a supposed decoy even when restarting
was a repair action for an active sub-fault inside a composite workflow. This
pre-resolved part of 12 workflows before policy inference. Schema v6 fixes the
generator, asserts that zero active sub-faults are resolved before inference,
and rejects stale v5 datasets and sandbox deployments. Corrected base and GRPO
results are pending a clean matched rerun.

See [the v5-to-v6 migration note](docs/migration-v5-to-v6.md) for the failure
mode, safeguards, and clean-rerun procedure.

## How it works

```text
fault injection → observation → policy → ordered actions → sandbox state transition → mechanical verifier
```

- Observations contain redacted, noisy, or shifted-noisy operational telemetry.
- Policies return one strict JSON object with an `actions` array of at most eight
  allowlisted operations.
- The mock or HTTP sandbox applies actions in order; model-provided values cannot
  override deployment history or declared configuration.
- The verifier checks actual post-action state and service health.
- Exact success and subfault-level partial reward are recorded independently.

## Repository map

| Path | Purpose |
|---|---|
| [`crashdiag/`](crashdiag/) | fault registry, policy contract, sandboxes, orchestrator, and verifier |
| [`training/`](training/) | dataset generation, direct GRPO, evaluation, reporting, and the end-to-end pipeline |
| [`notebooks/qwen2.5_3b/`](notebooks/qwen2.5_3b/) | base eval, GRPO, adapter eval, and complete-run notebooks |
| [`scripts/`](scripts/) | persistent training and dataset lifecycle entry points |
| [`deploy/`](deploy/) | public deployment configuration |
| [`docs/`](docs/README.md) | task/action catalogs, curriculum contract, data, and workflow documentation |
| [`tests/`](tests/) | unit, integration, notebook, reward, reporting, and artifact tests |

## Quick start

Install the core environment and run a local mechanical smoke test:

```bash
python -m pip install -e .
python smoke_test.py
```

Run the full test suite:

```bash
python -m unittest discover -s tests -v
```

### Start the HTTP sandbox

Keep credentials in ignored `.env` or `env.txt` files. Never commit them.

```bash
docker compose -f compose.yaml -f compose.public.yaml up --detach --build
curl --fail http://127.0.0.1:8765/healthz
```

For a public deployment, configure `CRASHDIAG_API_TOKEN` and the external URL as
shown in [`.env.example`](.env.example).

### Generate the schema-v6 dataset shape

```bash
python -m pip install -e ".[artifacts]"
python -m training.generate_dataset \
  --train-samples-per-fault 128 \
  --eval-samples-per-fault 16 \
  --seed 42 \
  --artifact-upload-policy disabled
```

This produces 6,656 train and 832 held-out rows. The generator defaults can
produce a larger 260,000/1,300-row corpus; see
[dataset generation](docs/dataset-generation.md).

### Train and evaluate

The persistent runner installs the training extras, downloads the configured
dataset artifact, runs direct GRPO, performs the final standalone evaluation,
and uploads both stages:

```bash
cp .env.example.grpo env.txt
# Fill HF_TOKEN, CRASHDIAG_DATASET_RUN_ID, CRASHDIAG_SANDBOX_URL,
# and CRASHDIAG_SANDBOX_TOKEN in env.txt.
bash scripts/grpo.sh
screen -r grpo
```

Do not run the script with `sudo`; the screen session belongs to the invoking
user. The final experiment used one NVIDIA L4 (24 GB), BF16, four generations
per prompt, an effective batch of eight, 832 optimizer steps, and approximately
8.2 hours of training.

The same workflow is available in
[`notebooks/qwen2.5_3b/grpo.ipynb`](notebooks/qwen2.5_3b/grpo.ipynb), while
[`run_all.ipynb`](notebooks/qwen2.5_3b/run_all.ipynb) includes dataset generation
and the base evaluation.

## Use the released adapter

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_id = "Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
).eval()
```

Use the exact v6 system prompt and validate generated workflows before execution.
The complete inference example, training configuration, raw evaluations, and
limitations are in the [model card](https://huggingface.co/Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO).

## Reproducibility

- Dataset seed: `42`
- Dataset rows: 6,656 train / 832 held out
- Scenario schema and curriculum: v6
- Corrected training/evaluation run IDs: pending clean rerun
- Historical v5 adapter: retained for audit only; its benchmark scores are invalidated

## Scope and limitations

CrashDiag is a research environment, not an autonomous production operator.
Its action space and verifier are deliberately bounded, results cover one model
and one training seed, and mechanical rewards remain only as complete as their
specification. Keep execution sandboxed and require independent safeguards
before adapting the system to real infrastructure.

## License

CrashDiag is licensed under the [Creative Commons Attribution 4.0 International
License (CC BY 4.0)](LICENSE).

Copyright © 2026 Indium AI Labs. When sharing or adapting the project, credit
“CrashDiag contributors, Indium AI Labs,” link to this repository and the
license, and indicate whether changes were made.
