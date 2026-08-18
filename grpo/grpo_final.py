"""Run direct GRPO over SSH with a persistent screen session.

Recommended remote workflow:

    cd /path/to/CrashDiag/grpo
    screen -S crashdiag-grpo
    python -u grpo_final.py 2>&1 | tee grpo_final.log
    # Detach without stopping: press Ctrl-A, then D

Reconnect and monitor:

    screen -ls
    screen -r crashdiag-grpo
    tail -f grpo_final.log

If the SSH connection drops, the screen session and training continue.
The script expects env.txt beside this file. It does not run SFT.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_FILE = Path(os.environ.get("CRASHDIAG_ENV_FILE", SCRIPT_DIR / "env.txt"))
if not ENV_FILE.is_absolute():
    ENV_FILE = (SCRIPT_DIR / ENV_FILE).resolve()


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE settings without overwriting shell variables."""

    if not path.is_file():
        raise RuntimeError(f"Missing {path}; copy env.txt beside grpo_final.py")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def ist_run_id(stage: str) -> str:
    stamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%dT%H%M%SIST")
    return f"{stamp}-qwen2.5_3b-{stage}"


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    load_env_file(ENV_FILE)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset_run_id = os.environ.get("CRASHDIAG_DATASET_RUN_ID", "").strip()
    if not dataset_run_id:
        raise RuntimeError("Set CRASHDIAG_DATASET_RUN_ID in grpo/env.txt")
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in grpo/env.txt")
    sandbox_url = os.environ.get("CRASHDIAG_SANDBOX_URL", "").strip()
    sandbox_token = (
        os.environ.get("CRASHDIAG_API_TOKEN", "").strip()
        or os.environ.get("CRASHDIAG_SANDBOX_TOKEN", "").strip()
    )
    if not sandbox_url or not sandbox_token:
        raise RuntimeError(
            "Set CRASHDIAG_SANDBOX_URL and CRASHDIAG_SANDBOX_TOKEN "
            "(or CRASHDIAG_API_TOKEN) in grpo/env.txt"
        )

    bucket_id = os.environ.get("CRASHDIAG_HF_BUCKET_ID", "devaanshpa/CrashDiag")
    grpo_run_id = os.environ.get("CRASHDIAG_GRPO_RUN_ID", "").strip() or ist_run_id("grpo")
    eval_run_id = (
        os.environ.get("CRASHDIAG_GRPO_EVAL_RUN_ID", "").strip()
        or ist_run_id("grpo-eval")
    )
    max_steps = os.environ.get("CRASHDIAG_GRPO_MAX_STEPS", "832")
    # Four samples give GRPO a meaningful within-prompt ranking signal.  The
    # previous two-sample run collapsed to identical 42-token completions.
    num_generations = os.environ.get("CRASHDIAG_GRPO_NUM_GENERATIONS", "4")
    max_completion = os.environ.get("CRASHDIAG_GRPO_MAX_COMPLETION_LENGTH", "96")
    learning_rate = os.environ.get("CRASHDIAG_GRPO_LEARNING_RATE", "5e-6")
    lr_scheduler = os.environ.get(
        "CRASHDIAG_GRPO_LR_SCHEDULER", "constant_with_warmup"
    )
    warmup_ratio = os.environ.get("CRASHDIAG_GRPO_WARMUP_RATIO", "0.05")
    temperature = os.environ.get("CRASHDIAG_GRPO_TEMPERATURE", "1.0")
    top_p = os.environ.get("CRASHDIAG_GRPO_TOP_P", "0.95")
    logging_steps = os.environ.get("CRASHDIAG_GRPO_LOGGING_STEPS", "10")
    save_steps = os.environ.get("CRASHDIAG_GRPO_SAVE_STEPS", "200")
    # The trainer performs a final evaluation after training. Keep periodic
    # evaluation beyond max_steps so it does not interrupt the training loop.
    eval_steps = os.environ.get("CRASHDIAG_GRPO_EVAL_STEPS", "1000000000")

    print(f"repo_root={REPO_ROOT}", flush=True)
    print(f"env_file={ENV_FILE}", flush=True)
    print(f"dataset_run_id={dataset_run_id}", flush=True)
    print(f"grpo_run_id={grpo_run_id}", flush=True)
    print(f"grpo_eval_run_id={eval_run_id}", flush=True)
    print(f"max_steps={max_steps}", flush=True)

    sys.path.insert(0, str(REPO_ROOT))
    from training.artifacts import ArtifactConfig, ArtifactUploader

    dataset_dir = REPO_ROOT / "artifacts" / "datasets"
    ArtifactUploader(
        ArtifactConfig(bucket_id=bucket_id, run_id=dataset_run_id, token=hf_token)
    ).download_stage("datasets", dataset_dir)
    train_file = dataset_dir / "grpo_train.jsonl"
    eval_file = dataset_dir / "grpo_eval.jsonl"
    if not train_file.is_file() or not eval_file.is_file():
        raise RuntimeError(f"Dataset stage is missing {train_file} or {eval_file}")

    eval_command = [
            sys.executable,
            "-u",
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            "1",
            "--num_machines",
            "1",
            "--mixed_precision",
            "bf16",
            "--dynamo_backend",
            "no",
            "-m",
            "training.grpo",
            "--model",
            "Qwen/Qwen2.5-3B-Instruct",
            "--train-file",
            str(train_file),
            "--eval-file",
            str(eval_file),
            "--output-dir",
            str(REPO_ROOT / "outputs" / "grpo"),
            "--no-load-in-4bit",
            "--precision",
            "bf16",
            # Four generations require the global train/eval batch to be
            # divisible by four. Keep the effective batch at eight.
            "--batch-size",
            "4",
            "--gradient-accumulation-steps",
            "2",
            "--num-generations",
            num_generations,
            "--learning-rate",
            learning_rate,
            "--lr-scheduler-type",
            lr_scheduler,
            "--warmup-ratio",
            warmup_ratio,
            "--temperature",
            temperature,
            "--top-p",
            top_p,
            "--logging-steps",
            logging_steps,
            "--save-steps",
            save_steps,
            "--max-prompt-length",
            "1024",
            "--max-completion-length",
            max_completion,
            "--max-steps",
            max_steps,
            "--eval-steps",
            eval_steps,
            "--artifact-bucket",
            bucket_id,
            "--run-id",
            grpo_run_id,
            "--artifact-stage",
            "grpo",
            "--sandbox-url",
            sandbox_url,
            "--sandbox-token",
            sandbox_token,
        ]
    )

    eval_command = [
            sys.executable,
            "-u",
            "-m",
            "training.evaluate_jsonl",
            "--model",
            str(REPO_ROOT / "outputs" / "grpo"),
            "--dataset",
            str(eval_file),
            "--output-dir",
            str(REPO_ROOT / "outputs" / "grpo-eval"),
            "--load-in-4bit",
            "--precision",
            "bf16",
            "--max-new-tokens",
            "64",
            "--sandbox-url",
            sandbox_url,
            "--sandbox-token",
            sandbox_token,
            "--artifact-bucket",
            bucket_id,
            "--run-id",
            eval_run_id,
            "--artifact-stage",
            "grpo-eval",
            "--no-few-shot",
        ]
    if os.environ.get("CRASHDIAG_SKIP_GRPO_EVAL", "0").lower() not in {
        "1", "true", "yes"
    }:
        run(eval_command)
    else:
        print("Skipping final GRPO evaluation (CRASHDIAG_SKIP_GRPO_EVAL=1)", flush=True)
    print(
        f"GRPO complete. Reports: hf://buckets/{bucket_id}/runs/"
        f"{eval_run_id}/grpo-eval/reports",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
