"""Generate deterministic, mechanically validated CrashDiag training data.

This command has only standard-library dependencies beyond the local
``crashdiag`` package.  Every SFT target is executed against a fresh
``MockSandbox`` and retained only after the selected fault reports resolved and
the sandbox reports healthy.  The GRPO file contains the same prompts and
scenario identifiers but deliberately contains no target completion.

The CLI defaults to a required upload into the private
``devaanshpa/CrashDiag`` Storage Bucket, reading ``HF_TOKEN`` from ``.env`` and
creating a unique run ID.  Local-only generation must be requested explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import secrets
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crashdiag.sandbox_apps.mock import MockSandbox, SandboxBackend

from .artifacts import (
    ArtifactError,
    add_artifact_arguments,
    preload_env,
    runtime_metadata,
    uploader_from_args,
)
from .common import (
    WORKFLOW_NAMES,
    fault_for_name,
    observation_messages,
    workflow_text,
    write_jsonl,
)
from .hard_scenarios import (
    HARD_CURRICULUM_VERSION,
    HARD_SCENARIO_SCHEMA_VERSION,
    HARD_SCENARIO_PROFILES,
    hard_expert_workflow,
    prepare_v6_scenario,
)


SCHEMA_VERSION = HARD_SCENARIO_SCHEMA_VERSION
DEFAULT_TRAIN_SAMPLES_PER_FAULT = 5000
DEFAULT_EVAL_SAMPLES_PER_FAULT = 25
# Eval variations start at a fixed high offset so shrinking the train set does
# not change the held-out rows or invalidate previously reported eval results.
EVAL_START_VARIATION = 1_000_000
DEFAULT_DATASET_BUCKET = "devaanshpa/CrashDiag"
DEFAULT_SFT_TRAIN_OUTPUT = Path("data/sft_train.jsonl")
DEFAULT_SFT_EVAL_OUTPUT = Path("data/sft_eval.jsonl")
DEFAULT_GRPO_TRAIN_OUTPUT = Path("data/grpo_train.jsonl")
DEFAULT_GRPO_EVAL_OUTPUT = Path("data/grpo_eval.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("data/grpo_summary.json")
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _automatic_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-dataset-{secrets.token_hex(6)}"


def _artifact_defaults(args: argparse.Namespace) -> None:
    """Make private upload the default while preserving explicit overrides."""

    if not args.artifact_bucket:
        args.artifact_bucket = (
            os.environ.get("CRASHDIAG_HF_BUCKET_ID", "").strip()
            or os.environ.get("CRASHDIAG_HF_BUCKET", "").strip()
            or DEFAULT_DATASET_BUCKET
        )
    if not args.run_id:
        args.run_id = (
            os.environ.get("CRASHDIAG_RUN_ID", "").strip() or _automatic_run_id()
        )


def sample_seed(base_seed: int, fault_name: str, variation_index: int) -> int:
    """Derive a stable per-scenario seed without Python's randomized hash()."""

    material = f"crashdiag:{base_seed}:{fault_name}:{variation_index}".encode("utf-8")
    # Keep the value inside Arrow/JSON's portable signed int64 range.  The high
    # bit carries no useful entropy for this dataset and can otherwise make
    # ``datasets.load_dataset("json", ...)`` infer an incompatible uint type.
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _prepare_background_state(sandbox: SandboxBackend, rng: random.Random) -> None:
    """Add benign, deterministic operational history before fault injection."""

    sandbox.set_disk_usage(round(rng.uniform(15.0, 80.0), 1))
    for _ in range(rng.randrange(4)):
        sandbox.wait_and_observe()
    for _ in range(rng.randrange(2)):
        sandbox.restart_app()


def _vary_fault(fault: Any, rng: random.Random) -> None:
    """Vary injected values while preserving each fault's mechanical contract."""

    if fault.name == "bad_env_var":
        fault.bad_value = rng.choice(
            ("invalid", "prodution", "development", "PRODUCTION")
        )
    elif fault.name == "broken_db_connection":
        fault.bad_value = rng.choice(
            (
                "postgresql://app:secret@missing-database:5432/app",
                "postgresql://app:secret@database.invalid:5432/app",
                "postgresql://app:secret@database:15432/app",
                "postgresql://app:secret@database:5432/missing_app",
            )
        )
    elif fault.name == "dependency_mismatch":
        fault.bad_version = rng.choice(
            ("0.9.0", "1.3.9", "2.0.0-incompatible", "9.9.9")
        )
    elif fault.name == "disk_full":
        fault.injected_percent = round(rng.uniform(91.0, 100.0), 1)
    elif fault.name == "port_proxy_misconfig":
        fault.wrong_port = rng.choice((80, 3000, 8081, 8888, 65535))


def prepare_scenario(
    fault_name: str,
    scenario_seed: int,
    *,
    sandbox: SandboxBackend | None = None,
) -> tuple[Any, SandboxBackend, random.Random]:
    """Rebuild the exact pre-action workflow state for a dataset prompt.

    GRPO uses this same function with each row's top-level ``sample_seed`` so
    reward is computed against the precise scenario the policy observed, not
    merely another instance of the same task class.
    """

    if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int):
        raise TypeError("scenario_seed must be an integer")
    profile = HARD_SCENARIO_PROFILES[scenario_seed % len(HARD_SCENARIO_PROFILES)]
    return prepare_v6_scenario(
        fault_name,
        scenario_seed,
        profile,
        sandbox=sandbox,
    )


def expert_workflow(fault_name: str) -> dict[str, Any]:
    """Return the ordered multi-action repair proved by the v6 scenario."""

    return hard_expert_workflow(fault_name)


def build_validated_sample(
    fault_name: str,
    *,
    base_seed: int,
    variation_index: int,
    split: str = "train",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build matching SFT/GRPO rows after executing the expert workflow.

    The observation is captured before the expert acts.  A new sandbox is used
    for each call, so validation cannot pass because an earlier scenario left
    behind repaired state.
    """

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if (
        isinstance(variation_index, bool)
        or not isinstance(variation_index, int)
        or variation_index < 0
    ):
        raise ValueError("variation_index must be a non-negative integer")
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")

    current_seed = sample_seed(base_seed, fault_name, variation_index)
    workflow, sandbox_backend, _ = prepare_scenario(fault_name, current_seed)
    if not isinstance(sandbox_backend, MockSandbox):
        raise TypeError("dataset generation requires MockSandbox state access")
    sandbox = sandbox_backend

    observation = sandbox.observe()
    target = expert_workflow(fault_name)
    for action in target["actions"]:
        sandbox.execute_action(action["action"], action["parameters"])

    resolved = workflow.is_resolved(sandbox)
    health_after = sandbox.health_check()
    healthy = isinstance(health_after, Mapping) and health_after.get("healthy") is True
    if not resolved or not healthy:
        raise RuntimeError(
            f"expert workflow failed mechanical validation for {fault_name!r}: "
            f"resolved={resolved}, health={health_after!r}"
        )

    common: dict[str, Any] = {
        "fault_name": workflow.name,
        "difficulty": workflow.difficulty,
        "subfault_count": workflow.subfault_count,
        "sample_seed": current_seed,
        "variation_index": variation_index,
        "scenario_schema_version": SCHEMA_VERSION,
        "curriculum_version": HARD_CURRICULUM_VERSION,
        "scenario_profile": HARD_SCENARIO_PROFILES[
            current_seed % len(HARD_SCENARIO_PROFILES)
        ],
        "prompt": observation_messages(observation, workflow_name=workflow.name),
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "curriculum_version": HARD_CURRICULUM_VERSION,
            "mechanically_validated": True,
            "split": split,
            "variation_index": variation_index,
            "scenario_profile": HARD_SCENARIO_PROFILES[
                current_seed % len(HARD_SCENARIO_PROFILES)
            ],
            "subfault_count": workflow.subfault_count,
        },
    }
    sft = {
        **common,
        "completion": [
            {
                "role": "assistant",
                "content": workflow_text([entry["action"] for entry in target["actions"]]),
            }
        ],
    }
    # Construct independently so a future mutation of the SFT row cannot leak
    # a target completion into the answer-free online-RL dataset.
    grpo = {
        "fault_name": common["fault_name"],
        "difficulty": common["difficulty"],
        "subfault_count": common["subfault_count"],
        "sample_seed": common["sample_seed"],
        "variation_index": common["variation_index"],
        "scenario_schema_version": common["scenario_schema_version"],
        "curriculum_version": common["curriculum_version"],
        "scenario_profile": common["scenario_profile"],
        "prompt": [dict(message) for message in common["prompt"]],
        "metadata": dict(common["metadata"]),
    }
    return sft, grpo


def generate_records(
    *,
    samples_per_fault: int = DEFAULT_TRAIN_SAMPLES_PER_FAULT,
    seed: int = 42,
    start_variation: int = 0,
    split: str = "train",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate equally sized strata for all 52 workflows."""

    if (
        isinstance(samples_per_fault, bool)
        or not isinstance(samples_per_fault, int)
        or samples_per_fault < 1
    ):
        raise ValueError("samples_per_fault must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(start_variation, bool)
        or not isinstance(start_variation, int)
        or start_variation < 0
    ):
        raise ValueError("start_variation must be a non-negative integer")
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")

    sft_rows: list[dict[str, Any]] = []
    grpo_rows: list[dict[str, Any]] = []
    total_rows = samples_per_fault * len(WORKFLOW_NAMES)
    completed = 0
    # Round-robin order keeps every contiguous group of 52 samples stratified.
    for variation_index in range(start_variation, start_variation + samples_per_fault):
        for fault_name in WORKFLOW_NAMES:
            sft, grpo = build_validated_sample(
                fault_name,
                base_seed=seed,
                variation_index=variation_index,
                split=split,
            )
            sft_rows.append(sft)
            grpo_rows.append(grpo)
            completed += 1
            if completed % 5000 == 0 or completed == total_rows:
                print(
                    f"  [{split}] {completed:,}/{total_rows:,} rows "
                    f"({completed / total_rows:.1%})",
                    flush=True,
                )
    return sft_rows, grpo_rows


def generate_datasets(
    sft_train_output: str | Path = DEFAULT_SFT_TRAIN_OUTPUT,
    sft_eval_output: str | Path = DEFAULT_SFT_EVAL_OUTPUT,
    grpo_train_output: str | Path = DEFAULT_GRPO_TRAIN_OUTPUT,
    grpo_eval_output: str | Path = DEFAULT_GRPO_EVAL_OUTPUT,
    summary_output: str | Path = DEFAULT_SUMMARY_OUTPUT,
    *,
    train_samples_per_fault: int = DEFAULT_TRAIN_SAMPLES_PER_FAULT,
    eval_samples_per_fault: int = DEFAULT_EVAL_SAMPLES_PER_FAULT,
    seed: int = 42,
) -> dict[str, int]:
    """Validate and write four stratified datasets plus a summary."""

    paths = {
        "sft_train": Path(sft_train_output),
        "sft_eval": Path(sft_eval_output),
        "grpo_train": Path(grpo_train_output),
        "grpo_eval": Path(grpo_eval_output),
        "summary": Path(summary_output),
    }
    resolved_paths = [path.resolve() for path in paths.values()]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("all SFT, GRPO, and summary output paths must be different files")
    if train_samples_per_fault > EVAL_START_VARIATION:
        raise ValueError(
            "train_samples_per_fault must not exceed the fixed eval offset "
            f"({EVAL_START_VARIATION}); choose a smaller train set"
        )

    sft_train, grpo_train = generate_records(
        samples_per_fault=train_samples_per_fault,
        seed=seed,
        start_variation=0,
        split="train",
    )
    sft_eval, grpo_eval = generate_records(
        samples_per_fault=eval_samples_per_fault,
        seed=seed,
        start_variation=EVAL_START_VARIATION,
        split="eval",
    )
    counts = {
        "sft_train": write_jsonl(paths["sft_train"], sft_train),
        "sft_eval": write_jsonl(paths["sft_eval"], sft_eval),
        "grpo_train": write_jsonl(paths["grpo_train"], grpo_train),
        "grpo_eval": write_jsonl(paths["grpo_eval"], grpo_eval),
    }
    if counts["sft_train"] != counts["grpo_train"]:
        raise RuntimeError("SFT and GRPO train row counts diverged")
    if counts["sft_eval"] != counts["grpo_eval"]:
        raise RuntimeError("SFT and GRPO eval row counts diverged")

    def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in rows).items()))

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "curriculum_version": HARD_CURRICULUM_VERSION,
        "action_contract": "multi_action_workflows",
        "curriculum": f"v{HARD_CURRICULUM_VERSION}",
        "seed": seed,
        "mechanically_validated": True,
        "targets_included": False,
        "workflow_count": len(WORKFLOW_NAMES),
        "train": {
            "rows": counts["sft_train"],
            "samples_per_fault": train_samples_per_fault,
            "workflow_distribution": distribution(sft_train, "fault_name"),
            "profile_distribution": distribution(sft_train, "scenario_profile"),
        },
        "eval": {
            "rows": counts["sft_eval"],
            "samples_per_fault": eval_samples_per_fault,
            "workflow_distribution": distribution(sft_eval, "fault_name"),
            "profile_distribution": distribution(sft_eval, "scenario_profile"),
        },
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"train": counts["sft_train"], "eval": counts["sft_eval"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic CrashDiag SFT and answer-free GRPO JSONL.",
        epilog=(
            "Default behavior requires HF_TOKEN, uploads to the private "
            "devaanshpa/CrashDiag bucket, and creates a unique run ID. Use "
            "--artifact-upload-policy disabled only for a local-only build."
        ),
    )
    parser.add_argument(
        "--sft-train-output", type=Path, default=DEFAULT_SFT_TRAIN_OUTPUT
    )
    parser.add_argument(
        "--sft-eval-output", type=Path, default=DEFAULT_SFT_EVAL_OUTPUT
    )
    parser.add_argument(
        "--grpo-train-output", type=Path, default=DEFAULT_GRPO_TRAIN_OUTPUT
    )
    parser.add_argument(
        "--grpo-eval-output", type=Path, default=DEFAULT_GRPO_EVAL_OUTPUT
    )
    parser.add_argument(
        "--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT
    )
    parser.add_argument(
        "--train-samples-per-fault",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES_PER_FAULT,
        help="training variations for each workflow (default: 5000)",
    )
    parser.add_argument(
        "--eval-samples-per-fault",
        type=int,
        default=DEFAULT_EVAL_SAMPLES_PER_FAULT,
        help="evaluation variations for each workflow (default: 25)",
    )
    parser.add_argument("--seed", type=int, default=42)
    add_artifact_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    preload_env(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    _artifact_defaults(args)
    provenance = runtime_metadata()
    source_commit = str(provenance.get("git_commit", "unknown"))
    print(f"RUN_ID={args.run_id}")
    print(f"SOURCE_COMMIT={source_commit}")
    try:
        uploader = uploader_from_args(args)
        if uploader is not None:
            if _FULL_GIT_SHA.fullmatch(source_commit) is None:
                raise ArtifactError(
                    "automatic dataset upload requires a Git checkout with a "
                    "full source commit so Kaggle can reproduce the generator"
                )
            uploader.start_run(
                {
                    "entrypoint": "training.generate_dataset",
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
                },
            )
        counts = generate_datasets(
            args.sft_train_output,
            args.sft_eval_output,
            args.grpo_train_output,
            args.grpo_eval_output,
            args.summary_output,
            train_samples_per_fault=args.train_samples_per_fault,
            eval_samples_per_fault=args.eval_samples_per_fault,
            seed=args.seed,
        )
        if uploader is not None:
            uploader.upload_files(
                [
                    args.sft_train_output,
                    args.sft_eval_output,
                    args.grpo_train_output,
                    args.grpo_eval_output,
                    args.summary_output,
                ],
                "datasets",
                metadata={
                    "seed": args.seed,
                    "train_rows": counts["train"],
                    "eval_rows": counts["eval"],
                    "mechanically_validated": True,
                    "grpo_targets_included": False,
                    "schema_version": HARD_SCENARIO_SCHEMA_VERSION,
                    "curriculum_version": HARD_CURRICULUM_VERSION,
                },
            )
    except (ArtifactError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"dataset generation failed: {exc}") from exc
    print(
        f"wrote {counts['train']} train + {counts['eval']} eval mechanically "
        "validated SFT samples"
    )
    print(f"  train: {args.sft_train_output}")
    print(f"  eval:  {args.sft_eval_output}")
    print(
        f"wrote {counts['train']} train + {counts['eval']} eval answer-free "
        "GRPO prompts"
    )
    print(f"  train: {args.grpo_train_output}")
    print(f"  eval:  {args.grpo_eval_output}")
    if uploader is not None:
        print(f"artifacts: {uploader.remote_uri('datasets')}")
    else:
        print("artifact upload: disabled explicitly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
