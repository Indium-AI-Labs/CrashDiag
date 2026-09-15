# Docker victim backend and 14B matched run

The default HTTP sandbox still uses in-memory `MockSandbox`. Training and
evaluation for the 14B run must use the Docker backend so every episode mutates
a disposable Compose stack and health is read from real processes, files, and
`docker inspect`.

## Victim stack

- Compose file: `crashdiag/sandbox_apps/target/compose.yaml`
- Runtime: `crashdiag/sandbox_apps/target/runtime.py` (`VictimRuntime`)
- Host client: `crashdiag/sandbox_apps/docker.py` (`DockerSandbox`)
- Isolation: per-session project `cd<12-hex>`, labels `crashdiag.session` and
  `crashdiag.role=victim`, internal network, 96 MB / 64 pids, tmpfs disk

`CRASHDIAG_SANDBOX_BACKEND=mock|docker` (default `mock`) selects the factory.
`GET /healthz` includes `"backend"` and schema `6`.

## Deploy the Docker sandbox (sandbox host, not the GPU box)

```bash
# Host process (preferred):
export CRASHDIAG_SANDBOX_TOKEN=...
python -m crashdiag.sandbox_server --backend docker --host 0.0.0.0 --port 8765

# Or sibling-Docker API container, from the repo root on the sandbox host:
export CRASHDIAG_SANDBOX_TOKEN=...
docker compose -f compose.docker.yaml up --detach --build
curl --fail http://127.0.0.1:8765/healthz
```

`compose.docker.yaml` passes the host `PWD` as `CRASHDIAG_REPO_ROOT` so the
host engine can build `crashdiag-victim:local`. Do not let that variable point
at a path that exists only inside the API container.

The JSON must contain `"backend": "docker"` and `6` in `scenario_schema_versions`.

Do not mount the Docker socket on the GPU training machine. Keep victim stacks
on a dedicated sandbox host.

## Dataset (6,656 train / 832 eval)

Copy `.env.example.grpo` to `env.txt` (or `.env`) and set `HF_TOKEN` plus
`CRASHDIAG_HF_BUCKET_ID` before uploading. The generator prints `RUN_ID=...`;
store that value as `CRASHDIAG_DATASET_RUN_ID` for training. Never reuse a v5
run id.

```bash
bash scripts/generate_docker_dataset.sh --artifact-upload-policy disabled
# or with upload (new run id, never reuse a v5 id):
bash scripts/generate_docker_dataset.sh
```

This is `python -m training.generate_dataset --train-samples-per-fault 128
--eval-samples-per-fault 16 --seed 42 --sandbox-backend docker`.

## 14B matched run

Model: `Qwen/Qwen2.5-14B-Instruct`. Metric: exact episode resolution.

1. Copy `.env.example.grpo` to `env.txt` and fill tokens, dataset run id, sandbox URL.
2. Set `CRASHDIAG_REQUIRE_SANDBOX_BACKEND=docker`.
3. Baseline:

```bash
bash scripts/run_14b.sh baseline
```

If exact resolution is ≥ 10%, stop. Harden noisy/shifted history and regenerate.
Do not train on an easy split.

4. GRPO (QLoRA NF4, batch 1, grad accum 8, 4 generations, 96-token completions):

```bash
bash scripts/run_14b.sh grpo
```

5. Report exact resolution and mean partial reward separately. The goal is
   **> 90% exact** on the same 832 rows. If the run misses, keep the number.

Hardware: A100 40 GB+ for 14B QLoRA GRPO; Docker sandbox on a separate host.
