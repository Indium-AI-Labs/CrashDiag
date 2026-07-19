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
import os
import random
import re
import secrets
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
    FAULT_NAMES,
    action_text,
    fault_for_name,
    observation_messages,
    write_jsonl,
)


SCHEMA_VERSION = 1
DEFAULT_DATASET_BUCKET = "devaanshpa/CrashDiag"
DEFAULT_SFT_TRAIN_OUTPUT = Path("data/sft_train.jsonl")
DEFAULT_SFT_EVAL_OUTPUT = Path("data/sft_eval.jsonl")
DEFAULT_GRPO_TRAIN_OUTPUT = Path("data/grpo_train.jsonl")
DEFAULT_GRPO_EVAL_OUTPUT = Path("data/grpo_eval.jsonl")
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
    """Rebuild the exact pre-action state represented by a dataset prompt.

    GRPO uses this same function with each row's top-level ``sample_seed`` so
    reward is computed against the precise scenario the policy observed, not
    merely another instance of the same fault class.
    """

    if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int):
        raise TypeError("scenario_seed must be an integer")
    fault = fault_for_name(fault_name)
    rng = random.Random(scenario_seed)
    target = sandbox if sandbox is not None else MockSandbox()
    _prepare_background_state(target, rng)
    _vary_fault(fault, rng)
    fault.inject(target)
    if fault.is_resolved(target):
        raise RuntimeError(f"fault {fault_name!r} was resolved immediately after injection")
    health = target.health_check()
    if not isinstance(health, Mapping) or health.get("healthy") is not False:
        raise RuntimeError(f"fault {fault_name!r} did not make the sandbox unhealthy")
    return fault, target, rng


def expert_action(fault_name: str, sandbox: MockSandbox, rng: random.Random) -> dict[str, Any]:
    """Return the deterministic one-step expert action for an injected fault."""

    if fault_name == "oom_kill":
        return {"action": "restart_app", "parameters": {}}
    if fault_name == "bad_env_var":
        return {"action": "rollback_env_var", "parameters": {"name": "APP_ENV"}}
    if fault_name == "broken_db_connection":
        return {
            "action": "rollback_env_var",
            "parameters": {"name": "DATABASE_URL"},
        }
    if fault_name == "dependency_mismatch":
        dependency = "web-framework"
        return {
            "action": "fix_dependency",
            "parameters": {
                "name": dependency,
                "version": sandbox.required_dependencies[dependency],
            },
        }
    if fault_name == "disk_full":
        # Keep the target clearly below the mock's 90% health boundary while
        # varying the exact corrective action seen during SFT.
        return {
            "action": "clear_disk",
            "parameters": {"target_percent": rng.choice((25.0, 35.0, 40.0, 50.0, 70.0))},
        }
    if fault_name == "port_proxy_misconfig":
        return {
            "action": "fix_port_config",
            "parameters": {"target_port": sandbox.app_port},
        }
    raise ValueError(f"no expert action for fault {fault_name!r}")


def build_validated_sample(
    fault_name: str,
    *,
    base_seed: int,
    variation_index: int,
    split: str = "train",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build matching SFT/GRPO rows after executing the SFT target.

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
    fault, sandbox_backend, rng = prepare_scenario(fault_name, current_seed)
    if not isinstance(sandbox_backend, MockSandbox):
        raise TypeError("dataset generation requires MockSandbox state access")
    sandbox = sandbox_backend

    observation = sandbox.observe()
    target = expert_action(fault_name, sandbox, rng)
    sandbox.execute_action(target["action"], target["parameters"])

    resolved = fault.is_resolved(sandbox)
    health_after = sandbox.health_check()
    healthy = isinstance(health_after, Mapping) and health_after.get("healthy") is True
    if not resolved or not healthy:
        raise RuntimeError(
            f"expert action failed mechanical validation for {fault_name!r}: "
            f"resolved={resolved}, health={health_after!r}"
        )

    common: dict[str, Any] = {
        "fault_name": fault.name,
        "difficulty": fault.difficulty,
        "sample_seed": current_seed,
        "variation_index": variation_index,
        "prompt": observation_messages(observation),
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "mechanically_validated": True,
            "split": split,
            "variation_index": variation_index,
        },
    }
    sft = {
        **common,
        "completion": [
            {
                "role": "assistant",
                "content": action_text(target["action"], target["parameters"]),
            }
        ],
    }
    # Construct independently so a future mutation of the SFT row cannot leak
    # a target completion into the answer-free online-RL dataset.
    grpo = {
        "fault_name": common["fault_name"],
        "difficulty": common["difficulty"],
        "sample_seed": common["sample_seed"],
        "variation_index": common["variation_index"],
        "prompt": [dict(message) for message in common["prompt"]],
        "metadata": dict(common["metadata"]),
    }
    return sft, grpo


def generate_records(
    *,
    samples_per_fault: int = 128,
    seed: int = 42,
    start_variation: int = 0,
    split: str = "train",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate equally sized strata for all six built-in faults."""

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
    # Round-robin order keeps every contiguous group of six samples stratified.
    for variation_index in range(start_variation, start_variation + samples_per_fault):
        for fault_name in FAULT_NAMES:
            sft, grpo = build_validated_sample(
                fault_name,
                base_seed=seed,
                variation_index=variation_index,
                split=split,
            )
            sft_rows.append(sft)
            grpo_rows.append(grpo)
    return sft_rows, grpo_rows


def generate_datasets(
    sft_train_output: str | Path = DEFAULT_SFT_TRAIN_OUTPUT,
    sft_eval_output: str | Path = DEFAULT_SFT_EVAL_OUTPUT,
    grpo_train_output: str | Path = DEFAULT_GRPO_TRAIN_OUTPUT,
    grpo_eval_output: str | Path = DEFAULT_GRPO_EVAL_OUTPUT,
    *,
    train_samples_per_fault: int = 128,
    eval_samples_per_fault: int = 16,
    seed: int = 42,
) -> dict[str, int]:
    """Validate and write four stratified datasets, returning split row counts."""

    paths = {
        "sft_train": Path(sft_train_output),
        "sft_eval": Path(sft_eval_output),
        "grpo_train": Path(grpo_train_output),
        "grpo_eval": Path(grpo_eval_output),
    }
    resolved_paths = [path.resolve() for path in paths.values()]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("all SFT and GRPO output paths must be different files")

    sft_train, grpo_train = generate_records(
        samples_per_fault=train_samples_per_fault,
        seed=seed,
        start_variation=0,
        split="train",
    )
    sft_eval, grpo_eval = generate_records(
        samples_per_fault=eval_samples_per_fault,
        seed=seed,
        start_variation=train_samples_per_fault,
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
        "--train-samples-per-fault",
        type=int,
        default=128,
        help="training variations for each of the six faults (default: 128)",
    )
    parser.add_argument(
        "--eval-samples-per-fault",
        type=int,
        default=16,
        help="evaluation variations for each of the six faults (default: 16)",
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
                ],
                "datasets",
                metadata={
                    "seed": args.seed,
                    "train_rows": counts["train"],
                    "eval_rows": counts["eval"],
                    "mechanically_validated": True,
                    "grpo_targets_included": False,
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
