# CrashDiag

CrashDiag trains a language model to diagnose and repair infrastructure faults.
The policy chooses one bounded JSON action; it never decides whether that action
worked. Success and reward come from current sandbox state and an application
health check, never from an LLM grader or a prose rubric.

## What is implemented

- Six injectable faults with state-based resolution checks.
- A dependency-free `MockSandbox` with coupled process, environment, database,
  dependency, disk, proxy, and HTTP-health state.
- JSON-safe trajectories and generic single/batch episode orchestration.
- Sparse `1.0`/`0.0` reward; optional mechanical shaping is off by default.
- Defensive OpenAI-compatible/vLLM agent output parsing.
- Deterministic, mechanically validated SFT and answer-free GRPO datasets.
- LoRA SFT using TRL `SFTTrainer` with completion-only loss.
- GRPO using TRL `GRPOTrainer` and an executable mechanical reward function.
- Local-weight, PEFT-adapter, or vLLM-endpoint evaluation.
- An isolated HTTP sandbox service, stdlib client, hardened Dockerfile, and
  Compose deployment.

## Mechanical verification guarantee

For each rollout CrashDiag rebuilds the exact seeded, fault-injected scenario
represented in the prompt, compares the reconstructed observation with that
prompt, executes the model's parsed action, observes the result, and calls
`fault.is_resolved(sandbox)`. Built-in faults require both the fault-specific
state and overall mechanical health to pass. Missing seeds, prompt mismatches,
invalid JSON, unknown actions, action errors, transport failures, and unresolved
state score `0.0`.

The GRPO dataset has no completion, answer, or expert-action field. The reward
function does not import or call a reward model.

## Dependency-free checks

Python 3.10 or newer is required. The core loop and sandbox server have no
third-party runtime dependencies.

```bash
python smoke_test.py
python -m unittest discover -s tests -v
```

Smoke-test output:

```text
CrashDiag smoke test
fault=bad_env_var
resolved=True
reward=1.0
steps=1
PASS: BadEnvVar resolved mechanically
```

## Training setup

For GPU training, use a fresh Python 3.11 or 3.12 environment and install the
PyTorch build appropriate for the server's CUDA/ROCm platform. Then install the
project's training extra:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
accelerate config
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`; the
remaining Python commands are unchanged.

The scripts target the current conversational dataset APIs in
[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) and
[TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer). Heavy ML
libraries are imported only when a training or local-evaluation run starts, so
dataset generation and reward tests remain lightweight.

### 1. Generate datasets

```bash
python -m training.generate_dataset \
  --train-samples-per-fault 128 \
  --eval-samples-per-fault 16 \
  --seed 42
```

This writes:

- `data/sft_train.jsonl`: 768 prompt/completion examples;
- `data/sft_eval.jsonl`: 96 prompt/completion examples;
- `data/grpo_train.jsonl`: 768 answer-free prompts;
- `data/grpo_eval.jsonl`: 96 answer-free prompts.

The included files were generated with those defaults. Each SFT target was
executed against a fresh sandbox before it was written. Regeneration with the
same arguments is byte-deterministic.

### 2. Supervised fine-tuning

```bash
accelerate launch --module training.sft \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset data/sft_train.jsonl \
  --eval-dataset data/sft_eval.jsonl \
  --output-dir outputs/sft
```

Defaults use LoRA rank 16, alpha 32, all linear layers, completion-only loss,
automatic BF16/FP16 selection, and gradient checkpointing. Run
`python -m training.sft --help` for batch size, accumulation, precision,
packing, checkpoint-resume, and model overrides.

### 3. Mechanically rewarded GRPO

Fast local sandbox rollouts:

```bash
accelerate launch --module training.grpo \
  --model outputs/sft \
  --train-file data/grpo_train.jsonl \
  --eval-file data/grpo_eval.jsonl \
  --output-dir outputs/grpo
```

Every generated completion gets a fresh sandbox. The dataset's `fault_name`
and `sample_seed` reconstruct the exact observation, the candidate action is
executed, and `CrashDiagVerifier` returns the sparse reward. The effective
generation batch (per-device batch x process count x accumulation) and global
evaluation batch must be divisible by `--num-generations`.

For faster generation, install vLLM according to its
[platform-specific installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/index.html)
and add:

```bash
--use-vllm --vllm-mode colocate
```

TRL's vLLM integration supports colocated and server modes; see the
[vLLM TRL guide](https://docs.vllm.ai/en/latest/training/trl/).

### 4. Evaluate the trained policy

Local weights or adapter:

```bash
python -m training.evaluate \
  --model outputs/grpo \
  --episodes-per-fault 10 \
  --output outputs/evaluation.json
```

OpenAI-compatible/vLLM endpoint:

```bash
python -m training.evaluate \
  --model crashdiag-policy \
  --base-url http://127.0.0.1:8000/v1 \
  --episodes-per-fault 10
```

Both modes execute one parsed sandbox action per episode and report mechanical
overall/per-fault success rates plus complete trajectories. Repetitions use
distinct deterministic held-out scenario seeds instead of repeating one prompt.

### One-command Linux run

After installing the training dependencies and running `accelerate config`:

```bash
bash scripts/train.sh
```

Override `BASE_MODEL`, `NUM_PROCESSES`, `TRAIN_SAMPLES_PER_FAULT`, or
`EVAL_SAMPLES_PER_FAULT` through environment variables.

## Run the sandbox service with Docker Compose

The service gives concurrent training workers isolated sessions and the same
strict action/mutation allowlists as the local backend.

```bash
export CRASHDIAG_SANDBOX_TOKEN="$(openssl rand -hex 32)"
docker compose up --detach --build
docker compose ps
curl http://127.0.0.1:8765/healthz
```

Compose deliberately refuses to start when `CRASHDIAG_SANDBOX_TOKEN` is unset
or empty. You can export it as above or copy `.env.example` to `.env` and replace
the empty token value. The standalone localhost CLI keeps optional authentication for
dependency-free local development.

Put `CRASHDIAG_SANDBOX_TOKEN=<generated value>` in a permission-restricted
`.env` file or server secret store so restarts retain the same token; `.env` is
ignored by Git. On PowerShell, a dependency-free token can be generated with:

```powershell
$env:CRASHDIAG_SANDBOX_TOKEN = [Convert]::ToHexString(
  [Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
).ToLowerInvariant()
```

Use it for GRPO:

```bash
export CRASHDIAG_SANDBOX_URL=http://127.0.0.1:8765

accelerate launch --module training.grpo \
  --model outputs/sft \
  --sandbox-url "$CRASHDIAG_SANDBOX_URL" \
  --sandbox-token "$CRASHDIAG_SANDBOX_TOKEN" \
  --output-dir outputs/grpo
```

Compose binds to `127.0.0.1` by default. For a sandbox on another server,
prefer an SSH tunnel or private network:

```bash
ssh -L 8765:127.0.0.1:8765 user@sandbox-server
```

Do not expose the plain HTTP API directly to the public internet. If remote
access is required, keep the bearer token enabled and put TLS plus network
access controls in front of it. Session count and idle lifetime are controlled
with `CRASHDIAG_MAX_SESSIONS` and `CRASHDIAG_SESSION_TTL_SECONDS`. Per-session
state growth, concurrent request workers, and slow connections are bounded by
`CRASHDIAG_MAX_OPERATIONS_PER_SESSION`, `CRASHDIAG_MAX_WORKERS`, and
`CRASHDIAG_REQUEST_TIMEOUT_SECONDS`.

Stop the service with `docker compose down`.

The image runs as UID/GID `10001`, drops all capabilities, has a read-only root
filesystem, sets `no-new-privileges`, and does not mount the Docker socket.

## Faults and actions

| Fault | Difficulty | Mechanical failure | Recovery action |
| --- | --- | --- | --- |
| `oom_kill` | medium | process stopped with `OOMKilled` | `restart_app` |
| `bad_env_var` | easy | invalid `APP_ENV` | `rollback_env_var` |
| `broken_db_connection` | medium | invalid `DATABASE_URL` | `rollback_env_var` |
| `dependency_mismatch` | hard | installed/required versions differ | `fix_dependency` |
| `disk_full` | medium | usage exceeds health threshold | `clear_disk` |
| `port_proxy_misconfig` | easy | proxy and app ports differ | `fix_port_config` |

`wait_and_observe` is the conservative fallback and changes no failure state.

## Repository layout

- `crashdiag/`: core environment, agents, verifier, and sandbox backends.
- `training/generate_dataset.py`: deterministic dataset construction.
- `training/sft.py`: LoRA supervised fine-tuning.
- `training/grpo.py`: state-executing GRPO reward and trainer.
- `training/evaluate.py`: local/endpoint mechanical evaluation.
- `Dockerfile`, `compose.yaml`: remote safe sandbox service.
- `scripts/train.sh`: end-to-end dataset → SFT → GRPO → evaluation runner.
- `tests/`: dependency-free core, data, reward, evaluator, and HTTP integration
  tests.

## Honest status and boundaries

Working and verified in this repository:

- all six local and HTTP-backed state transitions;
- deterministic generation of the four included datasets;
- exact-seed local and remote GRPO rewards;
- package installation and all command-line entry points;
- Docker build, non-root execution, Compose health, authentication, session
  isolation, capacity, TTL, and allowlist rejection;
- no LLM grading anywhere in the reward path.

Not run in this pass:

- a full SFT or GRPO optimization job, because model weights and the GPU
  training stack are not installed in this local environment;
- a live vLLM inference/training process.

Still stubbed/future work:

- `CoolifySandbox` stores configuration but all deployment-version-dependent
  methods raise `NotImplementedError`; no Coolify endpoints were guessed.
- The Docker service intentionally hosts isolated `MockSandbox` state. It does
  **not** force a real OOM, fill physical storage, alter host packages, or edit
  a host proxy. A separately isolated real-container/Coolify backend is still
  required before claiming real-infrastructure fault injection.

Mock assumptions are port `8080`, a `90%` disk threshold, representative
environment variables, and pinned dependency versions. The broken database
connection is modeled as `DATABASE_URL` corruption and uses the bounded
`rollback_env_var` action.
