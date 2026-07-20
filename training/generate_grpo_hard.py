"""Generate the schema-v2, GRPO-only CrashDiag curriculum.

The output contains no SFT completions or hidden expert labels.  Every row is
mechanically proven solvable against ``MockSandbox`` before it is written.  A
new run also records an immutable, signed handoff to the parent SFT adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactConfig,
    ArtifactError,
    ArtifactUploader,
    add_artifact_arguments,
    preload_env,
    runtime_metadata,
    uploader_from_args,
)
from .common import FAULT_NAMES, write_jsonl
from .generate_dataset import DEFAULT_DATASET_BUCKET
from .hard_scenarios import (
    HARD_CURRICULUM_VERSION,
    HARD_SCENARIO_PROFILES,
    HARD_SCENARIO_SCHEMA_VERSION,
    generate_hard_records,
)


DEFAULT_PARENT_SFT_RUN_ID = "20260719T113724Z-dataset-b26381b116bc"
DEFAULT_TRAIN_OUTPUT = Path("data/grpo_hard_train.jsonl")
DEFAULT_EVAL_OUTPUT = Path("data/grpo_hard_eval.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("data/grpo_hard_summary.json")
DEFAULT_PARENT_OUTPUT = Path("data/parent_sft.json")
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _automatic_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-grpo-hard-{secrets.token_hex(6)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_parent_reference(stage_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Read a completed SFT stage and return its signed adapter identity.

    A selective remote handoff need not download the large adapter weights;
    their byte size and SHA remain covered by the signed stage manifest.  The
    downloaded adapter config is itself checked byte-for-byte.
    """

    root = Path(stage_dir)
    manifest_path = root / "manifest.json"
    success_path = root / "_SUCCESS.json"
    config_path = root / "adapter_config.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        success = json.loads(success_path.read_text(encoding="utf-8"))
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError("parent SFT handoff has invalid or missing files") from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("run_id") != run_id
        or manifest.get("stage") != "sft"
        or not isinstance(manifest.get("files"), list)
    ):
        raise ArtifactError("parent SFT manifest identity mismatch")
    manifest_sha = _sha256(manifest_path)
    if (
        not isinstance(success, Mapping)
        or success.get("status") != "complete"
        or success.get("run_id") != run_id
        or success.get("stage") != "sft"
        or success.get("manifest_sha256") != manifest_sha
    ):
        raise ArtifactError("parent SFT success marker does not sign its manifest")
    entries: dict[str, Mapping[str, Any]] = {}
    for raw in manifest["files"]:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise ArtifactError("parent SFT manifest contains an invalid file entry")
        path = str(raw["path"])
        if path in entries:
            raise ArtifactError(f"parent SFT manifest repeats {path!r}")
        entries[path] = raw
    required = {"adapter_config.json", "adapter_model.safetensors"}
    missing = sorted(required.difference(entries))
    if missing:
        raise ArtifactError("parent SFT manifest is missing: " + ", ".join(missing))
    config_entry = entries["adapter_config.json"]
    if (
        config_path.stat().st_size != config_entry.get("bytes")
        or _sha256(config_path) != config_entry.get("sha256")
    ):
        raise ArtifactError("parent adapter_config.json does not match its manifest")
    weight_entry = entries["adapter_model.safetensors"]
    weight_sha = weight_entry.get("sha256")
    weight_bytes = weight_entry.get("bytes")
    if not isinstance(weight_sha, str) or not isinstance(weight_bytes, int):
        raise ArtifactError("parent adapter weight identity is incomplete")
    base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise ArtifactError("parent adapter config does not name its base model")
    runtime = manifest.get("runtime", {})
    source_commit = runtime.get("git_commit") if isinstance(runtime, Mapping) else None
    return {
        "run_id": run_id,
        "stage": "sft",
        "manifest_sha256": manifest_sha,
        "adapter_path": "adapter_model.safetensors",
        "adapter_sha256": weight_sha,
        "adapter_bytes": weight_bytes,
        "base_model": base_model,
        "source_commit": source_commit or "unknown",
    }


def _parent_uploader(current: ArtifactUploader, parent_run_id: str) -> ArtifactUploader:
    config = current.config
    return ArtifactUploader(
        ArtifactConfig(
            bucket_id=config.bucket_id,
            run_id=parent_run_id,
            prefix=config.prefix,
            policy="required",
            local_root=config.local_root,
            create_bucket=False,
            token=config.secret_token(),
        )
    )


def download_parent_reference(
    uploader: ArtifactUploader,
    parent_run_id: str,
) -> dict[str, Any]:
    """Fetch the smallest signed parent handoff from the private bucket."""

    parent = _parent_uploader(uploader, parent_run_id)
    with tempfile.TemporaryDirectory(prefix="crashdiag-parent-sft-") as directory:
        target = Path(directory)
        parent.download_stage(
            "sft",
            target,
            include_paths=["adapter_config.json"],
        )
        return read_parent_reference(target, parent_run_id)


def generate_hard_datasets(
    train_output: str | Path = DEFAULT_TRAIN_OUTPUT,
    eval_output: str | Path = DEFAULT_EVAL_OUTPUT,
    summary_output: str | Path = DEFAULT_SUMMARY_OUTPUT,
    *,
    train_samples_per_fault: int = 128,
    eval_samples_per_fault: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Write disjoint, balanced hard train/eval rows and their summary."""

    train_path = Path(train_output)
    eval_path = Path(eval_output)
    summary_path = Path(summary_output)
    if len({train_path.resolve(), eval_path.resolve(), summary_path.resolve()}) != 3:
        raise ValueError("hard dataset output paths must be different files")
    train_rows = generate_hard_records(
        samples_per_fault=train_samples_per_fault,
        seed=seed,
        start_variation=0,
        split="train",
    )
    eval_rows = generate_hard_records(
        samples_per_fault=eval_samples_per_fault,
        seed=seed,
        start_variation=train_samples_per_fault,
        split="eval",
    )
    train_seeds = {row["sample_seed"] for row in train_rows}
    eval_seeds = {row["sample_seed"] for row in eval_rows}
    if not train_seeds.isdisjoint(eval_seeds):
        raise RuntimeError("hard GRPO train and eval scenario seeds overlap")
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in rows).items()))

    summary: dict[str, Any] = {
        "schema_version": HARD_SCENARIO_SCHEMA_VERSION,
        "curriculum_version": HARD_CURRICULUM_VERSION,
        "action_contract": "parameter_free_repairs",
        "curriculum": "grpo-hard-only",
        "seed": seed,
        "mechanically_validated": True,
        "targets_included": False,
        "fault_families": list(FAULT_NAMES),
        "profiles": list(HARD_SCENARIO_PROFILES),
        "train": {
            "rows": len(train_rows),
            "samples_per_fault": train_samples_per_fault,
            "fault_distribution": distribution(train_rows, "fault_name"),
            "profile_distribution": distribution(train_rows, "scenario_profile"),
        },
        "eval": {
            "rows": len(eval_rows),
            "samples_per_fault": eval_samples_per_fault,
            "fault_distribution": distribution(eval_rows, "fault_name"),
            "profile_distribution": distribution(eval_rows, "scenario_profile"),
        },
    }
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--parent-output", type=Path, default=DEFAULT_PARENT_OUTPUT)
    parser.add_argument("--train-samples-per-fault", type=int, default=128)
    parser.add_argument("--eval-samples-per-fault", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--parent-sft-run-id",
        default=os.environ.get("CRASHDIAG_PARENT_SFT_RUN_ID", "").strip()
        or DEFAULT_PARENT_SFT_RUN_ID,
    )
    parser.add_argument(
        "--parent-sft-stage-dir",
        type=Path,
        default=None,
        help="local completed SFT stage for explicit offline generation",
    )
    add_artifact_arguments(parser)
    return parser


def _artifact_defaults(args: argparse.Namespace) -> None:
    if not args.artifact_bucket:
        args.artifact_bucket = (
            os.environ.get("CRASHDIAG_HF_BUCKET_ID", "").strip()
            or os.environ.get("CRASHDIAG_HF_BUCKET", "").strip()
            or DEFAULT_DATASET_BUCKET
        )
    if not args.run_id:
        args.run_id = os.environ.get("CRASHDIAG_RUN_ID", "").strip() or _automatic_run_id()


def main(argv: Sequence[str] | None = None) -> int:
    preload_env(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    _artifact_defaults(args)
    provenance = runtime_metadata()
    source_commit = str(provenance.get("git_commit", "unknown"))
    print(f"GRPO_RUN_ID={args.run_id}")
    print(f"SOURCE_COMMIT={source_commit}")
    print(f"PARENT_SFT_RUN_ID={args.parent_sft_run_id}")
    try:
        uploader = uploader_from_args(args)
        if uploader is not None:
            if _FULL_GIT_SHA.fullmatch(source_commit) is None:
                raise ArtifactError("hard dataset upload requires a full Git source commit")
            uploader.start_run(
                {
                    "entrypoint": "training.generate_grpo_hard",
                    "source_commit": source_commit,
                    "parent_sft_run_id": args.parent_sft_run_id,
                }
            )
            parent = download_parent_reference(uploader, args.parent_sft_run_id)
            uploader.start_stage(
                "datasets",
                {
                    "source_commit": source_commit,
                    "schema_version": HARD_SCENARIO_SCHEMA_VERSION,
                    "curriculum_version": HARD_CURRICULUM_VERSION,
                    "parent_sft": parent,
                },
            )
        elif args.parent_sft_stage_dir is not None:
            parent = read_parent_reference(args.parent_sft_stage_dir, args.parent_sft_run_id)
        else:
            raise ArtifactError(
                "local-only generation requires --parent-sft-stage-dir so the parent "
                "adapter remains mechanically traceable"
            )
        summary = generate_hard_datasets(
            args.train_output,
            args.eval_output,
            args.summary_output,
            train_samples_per_fault=args.train_samples_per_fault,
            eval_samples_per_fault=args.eval_samples_per_fault,
            seed=args.seed,
        )
        parent_document = {**parent, "referenced_by_source_commit": source_commit}
        _write_json(args.parent_output, parent_document)
        if uploader is not None:
            uploader.upload_files(
                [args.train_output, args.eval_output, args.summary_output, args.parent_output],
                "datasets",
                metadata={
                    "source_commit": source_commit,
                    "schema_version": HARD_SCENARIO_SCHEMA_VERSION,
                    "curriculum_version": HARD_CURRICULUM_VERSION,
                    "action_contract": summary["action_contract"],
                    "train_rows": summary["train"]["rows"],
                    "eval_rows": summary["eval"]["rows"],
                    "parent_sft": parent,
                    "mechanically_validated": True,
                    "grpo_targets_included": False,
                },
            )
    except (ArtifactError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"hard GRPO dataset generation failed: {exc}") from exc
    print(
        f"wrote {summary['train']['rows']} train + {summary['eval']['rows']} eval "
        "schema-v2 answer-free prompts"
    )
    print(f"  train:   {args.train_output}")
    print(f"  eval:    {args.eval_output}")
    print(f"  summary: {args.summary_output}")
    print(f"  parent:  {args.parent_output}")
    if uploader is not None:
        print(f"artifacts: {uploader.remote_uri('datasets')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
