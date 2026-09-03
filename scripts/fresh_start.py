"""Fresh-start a new v6 run: clear everything, then regenerate the v6
(SFT + GRPO) dataset and upload it as ONE fresh `datasets` stage.

Run with the .env loaded (HF_TOKEN, CRASHDIAG_SANDBOX_URL, CRASHDIAG_API_TOKEN).

Usage:
  python scripts/fresh_start.py --yes
  # prints the new CRASHDIAG_DATASET_RUN_ID to use in the notebooks

The generated files land in data/:
  sft_train.jsonl / sft_eval.jsonl   (v6, with multi-action completions)
  grpo_train.jsonl / grpo_eval.jsonl (v6, answer-free)
  grpo_summary.json                  (v6 summary)
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.hard_scenarios import (
    HARD_CURRICULUM_VERSION,
    HARD_SCENARIO_SCHEMA_VERSION,
)

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
        default=5000,
        help="v6 training variations per workflow (default 5000)",
    )
    parser.add_argument(
        "--eval-samples-per-fault",
        type=int,
        default=25,
        help="v6 evaluation variations per workflow (default 25)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None, help="explicit run ID")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm local cleanup and deletion/recreation of the configured bucket",
    )
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("Refusing destructive fresh start without explicit --yes")

    _load_env()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN not set; add it to .env first")
    bucket_id = os.environ.get("CRASHDIAG_HF_BUCKET_ID", "").strip()
    if not bucket_id:
        raise SystemExit("CRASHDIAG_HF_BUCKET_ID not set; add it to .env first")

    reset = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "reset_pipeline.py"),
            "--bucket-id",
            bucket_id,
            "--yes",
        ],
        check=False,
    )
    if reset.returncode != 0:
        raise SystemExit(f"reset_pipeline failed: {reset.returncode}")

    run_id = args.run_id or _automatic_run_id()
    print(f"RUN_ID={run_id}")

    from training.generate_dataset import generate_datasets

    generate_datasets(
        train_samples_per_fault=args.train_samples_per_fault,
        eval_samples_per_fault=args.eval_samples_per_fault,
        seed=args.seed,
    )
    print("wrote integrity-checked v6 SFT + GRPO datasets")

    from training.artifacts import (
        ArtifactConfig,
        ArtifactUploader,
        runtime_metadata,
    )

    source_commit = str(runtime_metadata().get("git_commit", "unknown"))
    uploader = ArtifactUploader(
        ArtifactConfig(
            bucket_id=bucket_id,
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
            "schema_version": HARD_SCENARIO_SCHEMA_VERSION,
            "curriculum_version": HARD_CURRICULUM_VERSION,
        },
    )
    uploader.upload_files(
        [
            PROJECT_ROOT / "data" / "sft_train.jsonl",
            PROJECT_ROOT / "data" / "sft_eval.jsonl",
            PROJECT_ROOT / "data" / "grpo_train.jsonl",
            PROJECT_ROOT / "data" / "grpo_eval.jsonl",
            PROJECT_ROOT / "data" / "grpo_summary.json",
        ],
        "datasets",
        metadata={
            "source_commit": source_commit,
            "seed": args.seed,
            "mechanically_validated": True,
            "grpo_targets_included": False,
            "curricula": [f"v{HARD_CURRICULUM_VERSION}"],
        },
    )
    uploader.complete_run({"stages": ["datasets"]})
    print(f"artifacts: {uploader.remote_uri('datasets')}")
    print("\nfresh start complete. Use this RUN_ID as CRASHDIAG_DATASET_RUN_ID:")
    print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
