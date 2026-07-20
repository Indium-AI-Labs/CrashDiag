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
- A schema-v2 hard GRPO-only curriculum with redacted known-good values,
  genuine stale incident history, harmless failed remediation, and randomized
  hidden configuration while preserving the six one-action fault families.
- LoRA SFT using TRL `SFTTrainer` with completion-only loss.
- GRPO using TRL `GRPOTrainer` and an executable mechanical reward function.
- Local-weight, PEFT-adapter, or vLLM-endpoint evaluation.
- An isolated HTTP sandbox service, stdlib client, hardened Dockerfile, and
  Compose deployment.
- Private Hugging Face Storage Bucket persistence for datasets, every retained
  checkpoint, final adapters, tokenizer/state/metrics, evaluation reports, and
  pipeline logs.
- Dependency-free SVG, strict-JSON, and Markdown reports for SFT, GRPO, and
  held-out mechanical evaluation, displayed directly in the Kaggle notebooks.
- Signed cross-run SFT adapter handoff, reward-variance calibration, a mandatory
  nonzero-update smoke gate, exact 192-row hard evaluation, schema-v1 regression
  evaluation, and fail-closed promotion gates.

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

### Recommended now: hard-only GRPO from the completed SFT adapter

The original SFT run remains immutable. Generate a new GRPO-only run on a
trusted CPU machine; the generator downloads the parent SFT stage manifest,
verifies its success marker and adapter config, and records the signed adapter
SHA without copying any target completion into the new data:

```bash
git pull --ff-only origin main
python -m pip install -e ".[artifacts]"
python -m training.generate_grpo_hard \
  --parent-sft-run-id 20260719T113724Z-dataset-b26381b116bc \
  --train-samples-per-fault 128 \
  --eval-samples-per-fault 32 \
  --seed 42
```

The defaults write and upload 768 `grpo_hard_train.jsonl` rows and 192
`grpo_hard_eval.jsonl` rows to a fresh private-bucket run. Copy the printed
`GRPO_RUN_ID` and full dataset `SOURCE_COMMIT` into
[`notebooks/grpo_hard.ipynb`](notebooks/grpo_hard.ipynb), and set
`TRAINER_COMMIT` to the published notebook/trainer revision. Keeping these two
SHAs separate lets a performance-only trainer fix consume an already signed,
unchanged dataset without weakening dataset provenance.

Generated `data/` files are intentionally ignored by Git. The signed private
HF bucket stage—not the repository—is the dataset system of record.

Before running the notebook, update the Vultr checkout because schema-v2 uses
new setup-only sandbox mutations and advertises supported scenario versions:

```bash
cd ~/CrashDiag
git pull --ff-only origin main
docker compose -f compose.yaml -f compose.vultr.yaml up --detach --build
curl --fail https://sandbox.devaanshpathak.com/healthz
```

The health response must contain `"scenario_schema_versions":[1,2]` and
`"hard_scenario_batch":true`. The latter prepares all deterministic setup state
inside one authenticated request instead of dozens of HTTPS mutations. The
hard notebook then performs, in order:

1. signed download of the hard data, the exact parent SFT adapter, and the
   original schema-v1 evaluation file;
2. 8-generation calibration in the immutable `calibration-contract-v2` stage
   at temperatures `1.5`, `1.6`, then `1.7`, with `top_p=0.9`, `top_k=50`,
   eight concurrent isolated reward workers, and visible per-group progress.
   It requires positive mechanical rewards for every fault family as well as
   useful mixed groups; a rerun downloads and verifies the completed stage
   instead of attempting to overwrite it;
3. a 36-step smoke job that must show positive reward standard deviation,
   positive gradient norm, mixed success, zero backend errors, finite metrics,
   and an adapter SHA different from the parent;
4. a fresh full GRPO job starting from the original SFT adapter;
5. deterministic evaluation of all 192 hard rows and all 96 original held-out
   rows; and
6. promotion only at at least 70% hard success, at least 50% for every fault,
   at least 95% schema-v1 regression success, and zero backend errors.

Every stage uploads JSON reports and SVG graphs to the same private hard-run
prefix. If calibration, smoke, evaluation, or promotion fails, the notebook
raises and never writes the run-level `_SUCCESS.json`.

### Original SFT and schema-v1 workflow

The supported workflow has one CPU data phase followed by two fresh Kaggle GPU
sessions:

1. On a trusted machine, put `HF_TOKEN` in the repository's ignored `.env`,
   install `.[artifacts]`, and run `python -m training.generate_dataset`. The
   generator mechanically validates all examples, creates a unique `RUN_ID`,
   and automatically uploads the four JSONL files, manifest, and dataset
   `_SUCCESS.json` to the private `devaanshpa/CrashDiag` bucket. Save the
   printed `RUN_ID` and `SOURCE_COMMIT`.
2. Open [`notebooks/sft.ipynb`](notebooks/sft.ipynb) in a fresh Kaggle session,
   paste those two values, enable Internet and a GPU, and attach only the
   `HF_TOKEN` Kaggle Secret. The notebook checks out the exact source revision,
   downloads and hash-verifies the completed dataset stage, trains SFT
   exclusively from those downloaded files, and displays the generated loss,
   learning-rate, and gradient charts after their signed bucket upload.
3. Open [`notebooks/grpo.ipynb`](notebooks/grpo.ipynb) in another fresh Kaggle
   session, enable Internet and a GPU, and attach `HF_TOKEN` plus
   `CRASHDIAG_SANDBOX_TOKEN`. Paste the same `RUN_ID` and `SOURCE_COMMIT`, start
   in smoke mode, and proceed to the full run only after checking the displayed
   reward/loss/policy diagnostics and backend-error logs. Full mode also
   displays mechanically verified success by fault after evaluation.

Neither notebook relies on another kernel or `/kaggle/working` files. Their
contract is the private bucket plus the exact `RUN_ID` and `SOURCE_COMMIT`.
SFT permits an incomplete overall run only because generation deliberately
leaves the pipeline open; it still requires the dataset stage's signed
manifest and success marker. GRPO independently downloads the same run and
requires both completed dataset and SFT stages. Neither notebook falls back to
the checked-in `data/` files or an untrained base model. GRPO then probes the
authenticated sandbox at `https://sandbox.devaanshpathak.com` before loading
model weights.

The notebooks are the operational entry points, while `training/*.py` remains
the reusable, tested implementation backend used by both notebooks and the
optional command-line workflows below. This keeps trainer, artifact, reward,
and evaluation behavior in one implementation instead of duplicating it in
notebook-only code.

### Local or direct-CLI environment

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

Kaggle images can include an old optional `torchao` build that current PEFT
rejects while creating an otherwise ordinary LoRA adapter. CrashDiag does not
use TorchAO, so both notebooks detect and uninstall it before installing the
training extra. If an older copy of a notebook has already failed with
`Found an incompatible version of torchao`, run the following in that Kaggle
session, restart the kernel, and then run the notebook from the beginning:

```bash
python -m pip uninstall -y torchao
```

There is no need to regenerate or re-upload a completed dataset for this
environment-only repair.

TRL 1.8 also defaults SFT to its newer `chunked_nll` implementation. That path
patches the model's LM-head forward method and currently fails when PEFT exposes
Qwen's forward call as `functools.partial`. CrashDiag explicitly selects the
documented standard `loss_type="nll"` instead. It keeps completion-only masking
and the same negative-log-likelihood objective while avoiding that optional
memory optimization. The SFT notebook applies the same override before loading
the checked-out training backend, so an existing completed dataset run remains
usable and does not need to be generated again.

The training backend targets the current conversational dataset APIs in
[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) and
[TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer). Heavy ML
libraries are imported only when a training or local-evaluation run starts, so
dataset generation and reward tests remain lightweight.

### Private artifact bucket

Training artifacts are stored in the private Hugging Face Storage Bucket
`devaanshpa/CrashDiag`. Copy `.env.example` to the gitignored `.env` and set a
fine-grained write token; never pass the token as a CLI flag:

```dotenv
HF_TOKEN=hf_...
CRASHDIAG_HF_BUCKET_ID=devaanshpa/CrashDiag
CRASHDIAG_ARTIFACT_UPLOAD_POLICY=required
# CRASHDIAG_RUN_ID=20260719T120000Z-experiment  # optional resume override
```

Environment variables override `.env`. Dataset generation defaults to the
`devaanshpa/CrashDiag` bucket, required upload, and a unique run ID, so its
normal command needs only `HF_TOKEN` in `.env`. The SFT notebook reads only the
`HF_TOKEN` Kaggle Secret; the GRPO notebook independently reads `HF_TOKEN` and
`CRASHDIAG_SANDBOX_TOKEN`. Neither notebook displays a secret or persists it to
`/kaggle/working`. For direct CLI automation, `scripts/train.sh` generates one
unique `CRASHDIAG_RUN_ID` when it is not already set. Direct phase commands
other than dataset generation require a run ID whenever bucket upload is
enabled. Pass `--artifact-upload-policy disabled` only for an intentional
local-only dataset build.

The notebooks perform their corresponding preflights automatically. For an
optional direct CLI run, verify authenticated write access before using GPU
time:

```bash
python -m training.artifacts preflight
```

The client checks that the bucket is private before every write and refuses an
existing public bucket. Set `CRASHDIAG_CREATE_ARTIFACT_BUCKET=true` only if the
first preflight should create a missing private bucket. Stage payloads are
uploaded before SHA-256 manifests and `_SUCCESS.json`; checkpoint callbacks
incrementally sync from rank zero after every Trainer save. Final reports are
generated before stage finalization, so every displayed report is covered by
the same signed manifest and success marker as its model or evaluation stage.

Download a completed run for later promotion to model and dataset repos:

```bash
python -m training.artifacts download --destination downloaded-run
```

The downloader requires a run-level success marker, an empty destination, and
validates every manifest hash. Hugging Face does not currently provide a
server-side bucket-to-repository promotion, so model/dataset repo publishing is
a later download-and-upload step. See the official
[Storage Bucket guide](https://huggingface.co/docs/huggingface_hub/en/guides/buckets).
For checkpoint recovery only, `--allow-incomplete` permits downloading a
partial prefix; it does not relax the default used for later promotion.
Completed stages are manifest-verified. Individual partial checkpoints do not
yet carry their own success manifest, so notebook auto-resume is off by default
and partial recovery should be inspected before opting in.

### Training reports

SFT and GRPO read the numeric history already written by Transformers/TRL and
produce only charts for metrics that are actually present. A typical run stores:

```text
runs/$RUN_ID/sft/reports/
  loss.svg
  learning_rate.svg
  gradient_norm.svg
  metrics_history.json
  metrics_summary.json
  report.md

runs/$RUN_ID/grpo-smoke/reports/   # or grpo/reports/
  loss.svg
  reward.svg
  policy_diagnostics.svg
  metrics_history.json
  metrics_summary.json
  report.md

runs/$RUN_ID/evaluation/
  evaluation.json
  mechanical_success_by_fault.svg
  mechanical_evaluation_metrics.json
  mechanical_evaluation_summary.json
  mechanical_evaluation_report.md
```

The exact training-chart inventory is dynamic: absent TRL metrics are omitted
rather than fabricated. These plots are diagnostics. Reward and evaluation
success remain executable checks against sandbox state; no charting code and no
LLM grades whether a fault is resolved.

### Optional direct CLI and automation

The following commands expose the same tested backend used by the notebooks.
They are useful for local development, CI, custom launchers, or non-Kaggle GPU
hosts; they are not required to use the recommended Kaggle workflow.

#### Generate datasets

Install only the lightweight upload dependencies on the trusted CPU machine:

```bash
python -m pip install -e ".[artifacts]"
```

With `HF_TOKEN` in the repository-root `.env`, run:

```bash
python -m training.generate_dataset \
  --train-samples-per-fault 128 \
  --eval-samples-per-fault 16 \
  --seed 42
```

Upload is automatic and required. The command generates a collision-resistant
run ID, checks that `devaanshpa/CrashDiag` is private and writable before
creating data, uploads payloads before the SHA-256 manifest and success marker,
and prints:

```text
RUN_ID=<copy-this-into-sft-and-grpo>
SOURCE_COMMIT=<copy-this-full-git-sha>
artifacts: hf://buckets/devaanshpa/CrashDiag/runs/<RUN_ID>/datasets
```

If upload fails, the command fails; it does not report successful generation.
To retry the same interrupted prefix, pass the printed ID with
`--run-id <RUN_ID>`. Only use `--artifact-upload-policy disabled` when you
explicitly want local files without a bucket upload.

This writes:

- `data/sft_train.jsonl`: 768 prompt/completion examples;
- `data/sft_eval.jsonl`: 96 prompt/completion examples;
- `data/grpo_train.jsonl`: 768 answer-free prompts;
- `data/grpo_eval.jsonl`: 96 answer-free prompts.

The entire `data/` directory is generated, ignored by Git, and uploaded to the
private bucket with a signed manifest. Regenerate or download it before using
the direct CLI examples below.

Each SFT target is executed against a fresh sandbox before it is written.
Regeneration with the same arguments is byte-deterministic; artifact run IDs
remain unique so unrelated uploads never share a mutable prefix.

#### Supervised fine-tuning

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
packing, step/epoch saves, checkpoint-resume, and model overrides. Bucket
uploads at each save make preemption recovery practical on ephemeral GPU hosts.

#### Mechanically rewarded GRPO

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

For the hard schema-v2 curriculum, use the new run's files and the temperature
selected by `training.calibrate_grpo`. The exact evaluator accepts either
schema version and replays every serialized prompt:

```bash
python -m training.calibrate_grpo \
  --model /path/to/verified-sft \
  --train-file data/grpo_hard_train.jsonl \
  --temperatures 1.5 1.6 1.7 \
  --top-p 0.9 \
  --top-k 50 \
  --artifact-stage calibration-contract-v2

accelerate launch --module training.grpo \
  --model /path/to/verified-sft \
  --train-file data/grpo_hard_train.jsonl \
  --eval-file "" \
  --num-generations 8 \
  --gradient-accumulation-steps 8 \
  --temperature <selected-temperature> \
  --top-p 0.9 \
  --top-k 50 \
  --beta 0.02 \
  --output-dir outputs/grpo-hard

python -m training.evaluate_jsonl \
  --model outputs/grpo-hard \
  --dataset data/grpo_hard_eval.jsonl \
  --output-dir outputs/hard-evaluation
```

For faster generation, install vLLM according to its
[platform-specific installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/index.html)
and add:

```bash
--use-vllm --vllm-mode colocate
```

TRL's vLLM integration supports colocated and server modes; see the
[vLLM TRL guide](https://docs.vllm.ai/en/latest/training/trl/).

#### Evaluate the trained policy

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

#### One-command Linux runner

After installing the training dependencies and running `accelerate config`:

```bash
bash scripts/train.sh
```

Override `BASE_MODEL`, `NUM_PROCESSES`, `TRAIN_SAMPLES_PER_FAULT`, or
`EVAL_SAMPLES_PER_FAULT` through environment variables. This runner defaults to
required artifact persistence, captures each phase's logs, and writes a
run-level success marker only after datasets, SFT, GRPO, evaluation, and logs
have all completed.

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

### Vultr sandbox for Kaggle

Kaggle should reach the long-lived Vultr sandbox over HTTPS, not a public
Docker port. Point a DNS name at the Vultr VPS, allow edge-firewall ingress on
80/443, keep 8765 closed, and add the domain to the Vultr host's `.env`:

```dotenv
CRASHDIAG_SANDBOX_TOKEN=<random-64-hex-value>
CRASHDIAG_SANDBOX_DOMAIN=sandbox.devaanshpathak.com
```

Start the sandbox plus the included Caddy TLS proxy:

```bash
docker compose -f compose.yaml -f compose.vultr.yaml up --detach --build
curl --fail https://sandbox.devaanshpathak.com/healthz
```

Store that sandbox token as the `CRASHDIAG_SANDBOX_TOKEN` Kaggle Secret. The
GRPO notebook is configured for `https://sandbox.devaanshpathak.com`; the SFT
notebook does not need this token or service. Do not copy `HF_TOKEN` to Vultr.
The base sandbox port remains bound to host loopback; Caddy accesses it over the
private Compose network and obtains TLS certificates automatically.

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

- `notebooks/sft.ipynb`: primary independent Kaggle dataset-download and LoRA
  SFT workflow; requires only the `HF_TOKEN` Kaggle Secret.
- `notebooks/grpo.ipynb`: primary independent Kaggle GRPO and mechanical
  evaluation workflow; restores the exact SFT run and uses the authenticated
  Vultr sandbox.
- `notebooks/grpo_hard.ipynb`: hard-only Kaggle pipeline with automatic
  calibration, smoke/full training, two exact evaluations, graphs, private
  uploads, and fail-closed promotion.
- `crashdiag/`: core environment, agents, verifier, and sandbox backends.
- `training/generate_dataset.py`: deterministic dataset construction plus
  automatic private-bucket upload and handoff identifiers.
- `training/generate_grpo_hard.py`: schema-v2 GRPO-only construction and signed
  parent-SFT reference.
- `training/calibrate_grpo.py`, `training/grpo_gates.py`: mechanical variance,
  update, regression, and promotion gates.
- `training/evaluate_jsonl.py`: exact prompt/seed evaluation for schema v1/v2.
- `training/sft.py`: reusable, tested LoRA SFT backend used by the notebook and
  direct CLI.
- `training/grpo.py`: reusable, tested state-executing GRPO reward and trainer
  used by the notebook and direct CLI.
- `training/evaluate.py`: reusable local/endpoint mechanical evaluation
  backend.
- `training/reporting.py`: dependency-free SVG/JSON/Markdown rendering from
  recorded trainer metrics and mechanically computed evaluation results.
- `training/kaggle.py`: optional Kaggle Secrets launcher for automated complete
  or phased CLI jobs.
- `training/artifacts.py`: private bucket preflight, checkpoint sync, manifests,
  completion markers, and verified download.
- `Dockerfile`, `compose.yaml`: remote safe sandbox service.
- `compose.vultr.yaml`, `deploy/vultr/Caddyfile`: HTTPS exposure for Kaggle.
- `scripts/train.sh`: optional automated dataset -> SFT -> GRPO -> evaluation
  runner.
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
- offline-tested private-bucket privacy checks, exact artifact mappings,
  checkpoint callbacks, manifests, markers, and verified downloads;
- a live private-bucket check plus a mechanically validated six-fault dataset
  upload/download/hash-verification integration run;
- all independent notebooks are structurally and offline validated, including
  clean code-cell compilation, secret-safety checks, downloaded-data-only SFT,
  and their bucket, `RUN_ID`, and `SOURCE_COMMIT` handoff contract;
- finite-metric filtering, SVG rendering, strict report JSON, notebook display,
  and signed-manifest inclusion for SFT, GRPO, and mechanical evaluation
  reports;
- no LLM grading anywhere in the reward path.

Previously completed and audited outside this local test pass:

- the parent SFT adapter run
  `20260719T113724Z-dataset-b26381b116bc` and its held-out schema-v1 evaluation;
- the earlier schema-v1 GRPO smoke, which correctly exposed a degenerate
  zero-loss/zero-gradient run and is not treated as a trained GRPO model;
- the hard run's original immutable `calibration` stage, which failed the
  variance gate: temperatures `0.9` and `1.2` had zero mixed groups, while
  `1.5` had only one mixed group out of 36. It remains an audit artifact and
  is not reused as evidence that GRPO is ready;
- its immutable `calibration-wide-v1` retry, which found mixed groups at 1.8
  but dropped to 17.4% strict JSON and 11.5% mean reward; higher temperatures
  collapsed further. This exposed a mismatch between the parent SFT's
  parameterized repair examples and the hidden-value hard action contract.

Not run in this pass:

- the corrected curriculum-v2 calibration, nonzero-update smoke, or full GRPO
  optimization job, because model weights and the GPU training stack are not
  installed in this local environment;
- a live vLLM inference/training process.
- the hard-only notebook end-to-end on a Kaggle GPU; its current validation is
  structural and offline rather than a completed training claim;
- reports from a real Kaggle optimization job have therefore not yet been
  visually inspected or confirmed in the live bucket;
- a live Vultr HTTPS deployment from this development machine.

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
