"""Matched Qwen2.5-14B Docker baseline gate and GRPO handoff.

This module does not download weights at import time.  ``scripts/run_14b.sh``
is the persistent wrapper; invoke:

    python -m training.run_14b baseline
    python -m training.run_14b grpo
    python -m training.run_14b all
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .grpo_pipeline import (
    ENV_FILE,
    REPO_ROOT,
    ist_run_id,
    load_env_file,
    require_current_sandbox,
    run,
)


def apply_14b_defaults() -> None:
    """Set 14B QLoRA + Docker-backend defaults without clobbering env.txt."""

    os.environ.setdefault("CRASHDIAG_BASE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    os.environ.setdefault("CRASHDIAG_MODEL_SLUG", "qwen2.5_14b")
    os.environ.setdefault("CRASHDIAG_REQUIRE_SANDBOX_BACKEND", "docker")
    os.environ.setdefault("CRASHDIAG_GRPO_LOAD_IN_4BIT", "1")
    os.environ.setdefault("CRASHDIAG_GRPO_BATCH_SIZE", "1")
    os.environ.setdefault("CRASHDIAG_GRPO_GRAD_ACCUM", "8")
    os.environ.setdefault("CRASHDIAG_GRPO_NUM_GENERATIONS", "4")
    os.environ.setdefault("CRASHDIAG_GRPO_MAX_STEPS", "832")
    os.environ.setdefault("CRASHDIAG_GRPO_MAX_COMPLETION_LENGTH", "96")
    os.environ.setdefault("CRASHDIAG_GRPO_EVAL_MAX_NEW_TOKENS", "96")


def exact_resolution_rate(payload: dict[str, Any]) -> float:
    """Read exact episode resolution from an evaluate_jsonl report."""

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
    if not isinstance(summary, dict):
        raise RuntimeError("evaluation summary is not an object")
    if "exact_resolution_rate" in summary:
        return float(summary["exact_resolution_rate"])
    resolved = summary.get("resolved_episodes")
    total = summary.get("total_episodes")
    if resolved is None or not total:
        raise RuntimeError(f"evaluation summary missing exact resolution fields: {sorted(summary)}")
    return float(resolved) / float(total)


def load_evaluation_summary(output_dir: Path) -> dict[str, Any]:
    candidates = list(output_dir.rglob("*summary*.json")) + list(output_dir.rglob("metrics.json"))
    if not candidates:
        raise RuntimeError(f"no evaluation summary under {output_dir}")
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def download_eval_split() -> Path:
    from .artifacts import ArtifactConfig, ArtifactUploader

    dest = REPO_ROOT / "artifacts" / "datasets"
    ArtifactUploader(
        ArtifactConfig(
            bucket_id=os.environ["CRASHDIAG_HF_BUCKET_ID"],
            run_id=os.environ["CRASHDIAG_DATASET_RUN_ID"],
            token=os.environ["HF_TOKEN"],
        )
    ).download_stage("datasets", dest)
    eval_file = dest / "grpo_eval.jsonl"
    if not eval_file.is_file():
        raise RuntimeError(f"dataset stage is missing {eval_file}")
    return eval_file


def run_baseline() -> float:
    eval_file = download_eval_split()
    run_id = os.environ.get("CRASHDIAG_BASE_EVAL_RUN_ID", "").strip() or ist_run_id("base-eval")
    output_dir = REPO_ROOT / "outputs" / "base-eval"
    sandbox_url = os.environ["CRASHDIAG_SANDBOX_URL"]
    command = [
        sys.executable,
        "-u",
        "-m",
        "training.evaluate_jsonl",
        "--model",
        os.environ["CRASHDIAG_BASE_MODEL"],
        "--dataset",
        str(eval_file),
        "--output-dir",
        str(output_dir),
        "--load-in-4bit",
        "--precision",
        "bf16",
        "--max-new-tokens",
        os.environ.get("CRASHDIAG_GRPO_EVAL_MAX_NEW_TOKENS", "96"),
        "--no-few-shot",
        "--sandbox-url",
        sandbox_url,
        "--artifact-bucket",
        os.environ["CRASHDIAG_HF_BUCKET_ID"],
        "--run-id",
        run_id,
        "--artifact-stage",
        "base-eval",
    ]
    run(command)
    rate = exact_resolution_rate(load_evaluation_summary(output_dir))
    print(f"baseline_exact_rate={rate:.4f}", flush=True)
    if rate >= 0.10:
        raise RuntimeError(
            "baseline exact resolution is >= 10%; harden Docker observations "
            "and regenerate the dataset before GRPO"
        )
    return rate


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    stage = args[0] if args else "all"
    if stage not in {"baseline", "grpo", "all"}:
        raise SystemExit("usage: python -m training.run_14b [baseline|grpo|all]")
    load_env_file(ENV_FILE)
    apply_14b_defaults()
    sandbox_url = os.environ.get("CRASHDIAG_SANDBOX_URL", "").strip()
    if not sandbox_url:
        raise RuntimeError("Set CRASHDIAG_SANDBOX_URL in env.txt")
    require_current_sandbox(sandbox_url)
    if stage in {"baseline", "all"}:
        run_baseline()
    if stage in {"grpo", "all"}:
        from . import grpo_pipeline

        return grpo_pipeline.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
