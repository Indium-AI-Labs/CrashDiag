# CrashDiag 3B clean-start runbook

Use `Qwen/Qwen2.5-3B-Instruct` for base, SFT, and GRPO. Evaluation is fixed-policy inference with mechanical sandbox execution; no planner, tool agent, or LLM judge is used.

## 1. Deploy the public sandbox

On the VPS, put `CRASHDIAG_SANDBOX_TOKEN` and `CRASHDIAG_SANDBOX_DOMAIN` in the protected `.env`, then run:

```bash
cd ~/CrashDiag
docker compose -f compose.yaml -f compose.public.yaml up --detach --build
curl --fail https://sandbox.devaanshpathak.com/healthz
```

Store that same sandbox token and a separate `HF_TOKEN` as Kaggle secrets.

## 2. Generate the fresh standard dataset

On a trusted machine with `HF_TOKEN` and bucket settings in `.env`:

```powershell
python -m pip install -e ".[artifacts]"
python -m training.generate_dataset `
  --train-samples-per-fault 128 `
  --eval-samples-per-fault 16 `
  --seed 42
```

Copy the printed `RUN_ID` and `SOURCE_COMMIT`. The signed dataset stage contains 768 training and 96 evaluation rows.

## 3. Evaluate base Qwen 3B

Open `notebooks/eval_base_qwen_hard.ipynb` in a fresh Kaggle GPU session. Set:

```text
CRASHDIAG_DATASET_RUN_ID=<dataset RUN_ID>
CRASHDIAG_DATASET_SOURCE_COMMIT=<dataset SOURCE_COMMIT>
CRASHDIAG_BASE_QWEN_RUN_ID=<new unique base-eval ID>
```

Run all cells. It uploads a signed `base-qwen-evaluation` result for the exact 96 rows.

## 4. Train and evaluate SFT

Run `notebooks/sft.ipynb` with the same dataset `RUN_ID` and `SOURCE_COMMIT`; it now uses Qwen 2.5 3B. Download its signed adapter and evaluate it with `training.evaluate_jsonl` on the same `grpo_eval.jsonl`, temperature zero and `--max-new-tokens 96`, into a new evaluation-only run.

## 5. Train and evaluate GRPO

Generate hard data after signed SFT exists, pass calibration and the smoke gate, then run `notebooks/grpo_hard.ipynb`. Evaluate final GRPO on both its hard split and the original 96-row regression split with the same evaluator commit, sandbox, data manifests, and generation settings.

## 6. Compare

Publish one signed comparison artifact with raw per-row outcomes, overall/per-fault success, strict-JSON and backend-error rates, adapter SHA-256 values, evaluator commit, and data manifest SHA-256. Do not compare mismatched manifests, deployments, or settings.
