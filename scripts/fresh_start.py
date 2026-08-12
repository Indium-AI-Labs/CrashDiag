"""Fresh-start a new Qwen3-14B run: clear everything, then regenerate the
v1 + v3-hard datasets and upload them as one fresh `datasets` stage.

Run with the .env loaded (HF_TOKEN, CRASHDIAG_SANDBOX_URL, CRASHDIAG_API_TOKEN).

Usage:
  python scripts/fresh_start.py
  # prints the new CRASHDIAG_DATASET_RUN_ID to use in the notebooks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    args = parser.parse_args()

    reset = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "reset_pipeline.py")],
        check=False,
    )
    if reset.returncode != 0:
        raise SystemExit(f"reset_pipeline failed: {reset.returncode}")

    def run(module: str, *extra: str) -> None:
        command = [
            sys.executable,
            "-m",
            module,
            "--artifact-upload-policy",
            "required",
            "--seed",
            str(args.seed),
            *extra,
        ]
        result = subprocess.run(command, check=False, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(f"{module} failed: {result.returncode}")

    # v1 SFT + GRPO dataset (uploaded as the fresh `datasets` stage)
    run(
        "training.generate_dataset",
        "--train-samples-per-fault",
        str(args.train_samples_per_fault),
        "--eval-samples-per-fault",
        str(args.eval_samples_per_fault),
    )
    # v3-hard GRPO curriculum (uploads into the same run's `datasets` stage)
    run(
        "training.generate_grpo_hard",
        "--train-samples-per-fault",
        str(args.hard_train_samples_per_fault),
        "--eval-samples-per-fault",
        str(args.hard_eval_samples_per_fault),
    )

    print("\nfresh start complete. Use the printed RUN_ID as CRASHDIAG_DATASET_RUN_ID.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
