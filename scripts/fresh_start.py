"""Fresh-start a new Qwen3-14B run: clear everything, then regenerate the
v1 (SFT + GRPO) and v3-hard (GRPO) datasets and upload them as ONE fresh
`datasets` stage.

Run with the .env loaded (HF_TOKEN, CRASHDIAG_SANDBOX_URL, CRASHDIAG_API_TOKEN).

Usage:
  python scripts/fresh_start.py
  # prints the new CRASHDIAG_DATASET_RUN_ID to use in the notebooks

The generated files land in data/ with unique names so a single upload
stage can hold all curricula:
  sft_train.jsonl / sft_eval.jsonl        (v1, with completions)
  grpo_train.jsonl / grpo_eval.jsonl      (v1, answer-free)
  grpo_hard_train.jsonl / grpo_hard_eval.jsonl  (v3-hard, answer-free)
"""

from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BUCKET = "devaanshpa/CrashDiag"


def _automatic_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-dataset-{secrets.token_hex(6)}"


def _load_env() -> None:
    from training.artifacts import load_env_file

    load_env_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-samples-per-fault",
        type=int,
        default=64,
        help="v1 training variations per fault (default 64)",
    )
    parser.add_argument(
        "--eval-samples-per-fault",
        type=int,
        default=8,
        help="v1 evaluation variations per fault (default 8)",
    )
    parser.add_argument(
        "--hard-train-samples-per-fault",
        type=int,
        default=24,
        help="v3-hard training variations per fault (default 24)",
    )
    parser.add_argument(
        "--hard-eval-samples-per-fault",
        type=int,
        default=8,
        help="v3-hard evaluation variations per fault (default 8)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None, help="explicit run ID")
    args = parser.parse_args()

    _load_env()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN not set; add it to .env first")

    reset = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "reset_pipeline.py")],
        check=False,
    )
    if reset.returncode != 0:
        raise SystemExit(f"reset_pipeline failed: {reset.returncode}")

    run_id = args.run_id or _automatic_run_id()
    print(f"RUN_ID={run_id}")

    from training.generate_dataset import generate_datasets
    from training.generate_grpo_hard import generate_hard_datasets

    generate_datasets(
        train_samples_per_fault=args.train_samples_per_fault,
        eval_samples_per_fault=args.eval_samples_per_fault,
        seed=args.seed,
    )
    print("wrote v1 SFT + GRPO datasets")

    generate_hard_datasets(
        train_samples_per_fault=args.hard_train_samples_per_fault,
        eval_samples_per_fault=args.hard_eval_samples_per_fault,
        seed=args.seed,
    )
    print("wrote v3-hard GRPO dataset")

    from training.artifacts import (
        ArtifactConfig,
        ArtifactUploader,
        runtime_metadata,
    )

    source_commit = str(runtime_metadata().get("git_commit", "unknown"))
    uploader = ArtifactUploader(
        ArtifactConfig(
            bucket_id=os.environ.get("CRASHDIAG_HF_BUCKET_ID", DEFAULT_BUCKET),
            run_id=run_id,
            token=token,
            policy="required",
        )
    )
    uploader.start_run(
        {
            "entrypoint": "scripts.fresh_start",
            "source_commit": source_commit,
        }
    )
    uploader.start_stage(
        "datasets",
        {
            "source_commit": source_commit,
            "seed": args.seed,
            "train_samples_per_fault": args.train_samples_per_fault,
            "eval_samples_per_fault": args.eval_samples_per_fault,
            "hard_train_samples_per_fault": args.hard_train_samples_per_fault,
            "hard_eval_samples_per_fault": args.hard_eval_samples_per_fault,
            "schema_version": 1,
            "hard_scenario_schema_version": 3,
        },
    )
    uploader.upload_files(
        [
            PROJECT_ROOT / "data" / "sft_train.jsonl",
            PROJECT_ROOT / "data" / "sft_eval.jsonl",
            PROJECT_ROOT / "data" / "grpo_train.jsonl",
            PROJECT_ROOT / "data" / "grpo_eval.jsonl",
            PROJECT_ROOT / "data" / "grpo_hard_train.jsonl",
            PROJECT_ROOT / "data" / "grpo_hard_eval.jsonl",
            PROJECT_ROOT / "data" / "grpo_hard_summary.json",
        ],
        "datasets",
        metadata={
            "source_commit": source_commit,
            "seed": args.seed,
            "mechanically_validated": True,
            "grpo_targets_included": False,
            "curricula": ["v1", "hard-v3"],
        },
    )
    uploader.complete_run({"stages": ["datasets"]})
    print(f"artifacts: {uploader.remote_uri('datasets')}")
    print("\nfresh start complete. Use this RUN_ID as CRASHDIAG_DATASET_RUN_ID:")
    print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
