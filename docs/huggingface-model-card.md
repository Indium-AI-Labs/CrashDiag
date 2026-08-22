---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
pipeline_tag: text-generation
license: cc-by-4.0
language:
- en
tags:
- peft
- lora
- grpo
- reinforcement-learning
- infrastructure
- agents
- tool-use
---

# CrashDiag Qwen2.5-3B GRPO

This repository contains the final LoRA adapter for the CrashDiag infrastructure
repair policy. It was trained directly from
[`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
with Group Relative Policy Optimization (GRPO); no supervised fine-tuning
checkpoint was used.

CrashDiag policies receive incomplete operational telemetry and emit one strict
JSON object containing an ordered repair workflow. A disposable sandbox executes
the workflow and computes reward from resulting state and health checks. No LLM
grader is used.

Project source: <https://github.com/Indium-AI-Labs/CrashDiag>

## Evaluation

The final standalone evaluation replays the same 832 held-out v5 episodes for
the base model and the adapter. It contains 52 fault families with 16 disjoint
variations per family across redacted, noisy, and shifted-noisy profiles. Both
runs use the same held-out rows. The retained base control used the evaluator's
generic format demonstration and a 64-token generation limit; the adapter used
no demonstration and the training-aligned 96-token limit. These differences are
disclosed confounders, so this is a retained-run comparison rather than a
controlled ablation.

| Policy | Exact resolution | Mean verified reward | Strict JSON | Backend errors |
|---|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | 1.92% (16/832) | 11.88% | 63.58% | 0.00% |
| CrashDiag GRPO adapter | **27.40% (228/832)** | **41.91%** | **94.59%** | 0.00% |

Exact resolution requires every injected subfault to be repaired. Mean verified
reward is partial credit, `resolved_subfaults / total_subfaults`. In the raw
mechanical-evaluation JSON, the historical `success_rate` field stores this mean
verified reward; `resolved_episodes / total_episodes` is the exact-resolution
rate.

![Per-fault evaluation](https://huggingface.co/Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO/resolve/main/evaluation/grpo_success_by_fault.svg)

The repository includes the complete standalone outputs for both policies under
[`evaluation/`](https://huggingface.co/Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO/tree/main/evaluation)
and trainer diagnostics under
[`training/`](https://huggingface.co/Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO/tree/main/training).

## Training

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Method | direct GRPO with mechanically executed rewards |
| Dataset | 6,656 train / 832 held-out v5 episodes |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer steps | 832 |
| Effective batch | 8 prompts (batch 4, accumulation 2) |
| Generations per prompt | 4 |
| Learning rate | `5e-6`, constant with 5% warmup |
| Context limits | 1,024 prompt tokens / 96 completion tokens |
| Sampling | temperature 1.0, top-p 0.95 |
| Precision and compute | BF16 on one NVIDIA L4 (24 GB) |
| Training runtime | approximately 8.2 hours |

The reward function rejects non-strict output during training and never calls a
model. The separately reported strict-JSON rate remains useful because final
evaluation records formatting independently from mechanical partial reward.

![Training reward](https://huggingface.co/Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO/resolve/main/training/reward.svg)

## Loading the adapter

Install the project and model dependencies:

```bash
git clone https://github.com/Indium-AI-Labs/CrashDiag.git
cd CrashDiag
python -m pip install -e ".[train]"
```

Then load the PEFT adapter together with its base model:

```python
import json
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from training.hard_scenarios import HARD_SYSTEM_PROMPT

model_id = "Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
).eval()

observation = {
    "observation": {
        "incident_window": {"gateway": "degraded", "http_family": "5xx"},
        "telemetry": {"signals": ["sensor-12:red", "sensor-20:red"]},
    }
}
messages = [
    {"role": "system", "content": HARD_SYSTEM_PROMPT},
    {"role": "user", "content": json.dumps(observation, separators=(",", ":"))},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=96,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True))
```

Expected output is one JSON object of the form:

```json
{"actions":[{"action":"restart_app","parameters":{}}]}
```

Run outputs through `crashdiag.agents.parse_workflow` before executing them. The
environment caps workflows at eight actions and restricts actions to its declared
allowlist.

## Repository contents

- `adapter_model.safetensors` and `adapter_config.json`: final LoRA adapter.
- tokenizer and chat-template files: generation-compatible Qwen tokenizer state.
- `evaluation/`: full 832-episode GRPO/base outputs, summaries, metrics, and charts.
- `training/`: trainer results, metric history, summaries, and diagnostic charts.

Intermediate checkpoints, optimizer state, credentials, and private storage
manifests are intentionally excluded.

## Limitations and intended use

- The policy is evaluated in the CrashDiag sandbox, not on unrestricted production
  infrastructure.
- Results come from one training seed and one base model.
- The retained base and adapter evaluations differ in format demonstration and
  generation limit; run a matched-token control before treating the delta as a
  causal estimate of GRPO's effect.
- The 52 fault families and action allowlist are finite; performance does not imply
  general incident-response competence.
- Mechanical reward is only as complete as the verifier specification and can
  still be specification-gamed.
- Use this adapter for research and sandboxed evaluation. Do not grant it direct,
  unsupervised access to production systems.

## Provenance

- Training run: `20260820T101800IST-qwen2.5_3b-grpo`
- Standalone adapter evaluation: `20260820T101800IST-qwen2.5_3b-grpo-eval`
- Base-model evaluation: `20260818T092323IST-qwen2.5_3b-base-eval`
- Dataset generation seed: 42
