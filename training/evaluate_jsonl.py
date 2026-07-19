"""Evaluate a local policy against every exact, answer-free JSONL scenario."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, add_artifact_arguments, preload_env, uploader_from_args
from .calibrate_grpo import read_jsonl
from .grpo import configure_reward_backend, mechanical_reward
from .inference import generate_from_messages, load_local_policy
from .reporting import generate_evaluation_report


def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate already-computed mechanical rewards by fault and profile."""

    if not results:
        raise ValueError("evaluation produced no results")

    def grouped(key: str) -> dict[str, Any]:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in results:
            value = item.get(key)
            if value is not None:
                groups[str(value)].append(item)
        return {
            name: {
                "episodes": len(items),
                "resolved": sum(float(item["reward"]) == 1.0 for item in items),
                "success_rate": sum(float(item["reward"]) for item in items) / len(items),
            }
            for name, items in sorted(groups.items())
        }

    rewards = [float(item["reward"]) for item in results]
    return {
        "schema_version": 1,
        "scoring": "mechanical_fault_resolution",
        "summary": {
            "total_episodes": len(results),
            "resolved_episodes": sum(reward == 1.0 for reward in rewards),
            "success_rate": sum(rewards) / len(rewards),
            "strict_json_rate": sum(bool(item["strict_json"]) for item in results) / len(results),
            "backend_error_rate": sum(bool(item["backend_error"]) for item in results) / len(results),
        },
        "per_fault": grouped("fault_name"),
        "per_profile": grouped("scenario_profile"),
        "results": list(results),
    }


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    generate_one: Any,
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    selected = list(rows if max_rows is None else rows[:max_rows])
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        prompt = row.get("prompt")
        if not isinstance(prompt, list):
            raise ValueError(f"row {index} has no conversational prompt")
        completion = str(generate_one(prompt))
        extras: dict[str, list[Any]] = {}

        def log_extra(name: str, values: list[Any]) -> None:
            extras[name] = list(values)

        rewards = mechanical_reward(
            [completion],
            fault_name=[row.get("fault_name")],
            sample_seed=[row.get("sample_seed")],
            prompts=[prompt],
            scenario_schema_version=[row.get("scenario_schema_version", 1)],
            scenario_profile=[row.get("scenario_profile")]
            if row.get("scenario_schema_version", 1) == 2
            else None,
            log_extra=log_extra,
        )
        results.append(
            {
                "row_index": index,
                "fault_name": row.get("fault_name"),
                "scenario_profile": row.get("scenario_profile"),
                "scenario_schema_version": row.get("scenario_schema_version", 1),
                "sample_seed": row.get("sample_seed"),
                "completion": completion,
                "action": extras["crashdiag_action"][0],
                "reward": float(rewards[0]),
                "resolved": bool(extras["crashdiag_resolved"][0]),
                "backend_error": bool(extras["crashdiag_backend_error"][0]),
                "strict_json": bool(extras["crashdiag_strict_json"][0]),
            }
        )
    return summarize_results(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/grpo")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jsonl-evaluation"))
    parser.add_argument("--max-rows", type=int, default=0, help="0 evaluates the complete file")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--sandbox-url", default=os.environ.get("CRASHDIAG_SANDBOX_URL", ""))
    parser.add_argument(
        "--sandbox-token",
        default=os.environ.get("CRASHDIAG_API_TOKEN") or os.environ.get("CRASHDIAG_SANDBOX_TOKEN"),
    )
    parser.add_argument("--sandbox-timeout", type=float, default=15.0)
    parser.add_argument("--artifact-stage", default="evaluation")
    add_artifact_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    preload_env(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_rows < 0 or args.max_new_tokens < 1:
        parser.error("--max-rows cannot be negative and --max-new-tokens must be positive")
    try:
        uploader = uploader_from_args(args)
        if uploader is not None:
            uploader.start_stage(
                args.artifact_stage,
                {
                    "model": args.model,
                    "dataset": args.dataset,
                    "scoring": "mechanical_fault_resolution",
                },
            )
        configure_reward_backend(
            sandbox_url=args.sandbox_url,
            api_token=args.sandbox_token,
            timeout=args.sandbox_timeout,
        )
        rows = read_jsonl(args.dataset)
        model, tokenizer = load_local_policy(
            args.model,
            precision=args.precision,
            trust_remote_code=args.trust_remote_code,
        )

        def generate_one(messages: Sequence[Mapping[str, str]]) -> str:
            return generate_from_messages(
                model,
                tokenizer,
                messages,
                num_return_sequences=1,
                temperature=0.0,
                max_new_tokens=args.max_new_tokens,
            )[0]

        report = evaluate_rows(
            rows,
            generate_one,
            max_rows=args.max_rows or None,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / "mechanical_evaluation.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        generate_evaluation_report(
            report_path,
            args.output_dir / "reports",
            title=f"CrashDiag {args.artifact_stage} exact-dataset success",
        )
        if uploader is not None:
            uploader.upload_directory(
                args.output_dir,
                args.artifact_stage,
                metadata={
                    "episodes": report["summary"]["total_episodes"],
                    "success_rate": report["summary"]["success_rate"],
                    "backend_error_rate": report["summary"]["backend_error_rate"],
                    "scoring": report["scoring"],
                },
            )
    except (ArtifactError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"exact JSONL evaluation failed: {exc}\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
