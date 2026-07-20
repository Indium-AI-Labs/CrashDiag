"""Calibrate schema-v2 GRPO sampling using only mechanical sandbox rewards."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactError,
    add_artifact_arguments,
    preload_env,
    uploader_from_args,
)
from .common import FAULT_NAMES
from .grpo import configure_reward_backend, mechanical_reward
from .hard_scenarios import HARD_SCENARIO_PROFILES, HARD_SCENARIO_SCHEMA_VERSION
from .inference import generate_from_messages, load_local_policy


DEFAULT_TEMPERATURES = (0.9, 1.2, 1.5)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            rows.append(row)
    return rows


def select_calibration_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompts_per_fault_profile: int = 2,
) -> list[dict[str, Any]]:
    """Select a deterministic 6 x 3 stratified calibration slice."""

    if prompts_per_fault_profile < 1:
        raise ValueError("prompts_per_fault_profile must be positive")
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        fault = str(raw.get("fault_name"))
        profile = str(raw.get("scenario_profile"))
        if fault in FAULT_NAMES and profile in HARD_SCENARIO_PROFILES:
            cells[(fault, profile)].append(dict(raw))
    selected: list[dict[str, Any]] = []
    for profile in HARD_SCENARIO_PROFILES:
        for fault in FAULT_NAMES:
            candidates = sorted(
                cells[(fault, profile)],
                key=lambda row: (int(row.get("variation_index", -1)), int(row.get("sample_seed", -1))),
            )
            if len(candidates) < prompts_per_fault_profile:
                raise ValueError(
                    f"calibration needs {prompts_per_fault_profile} rows for "
                    f"{fault}/{profile}, found {len(candidates)}"
                )
            selected.extend(candidates[:prompts_per_fault_profile])
    return selected


def summarize_temperature(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    expected_group_size: int,
    min_strict_json_rate: float = 0.95,
    min_mean_reward: float = 0.15,
    max_mean_reward: float = 0.95,
    min_mixed_group_rate: float = 0.25,
    min_mixed_fault_families: int = 4,
) -> dict[str, Any]:
    """Apply explicit variance gates to recorded programmatic rewards."""

    if not rollouts:
        raise ValueError("calibration produced no rollouts")
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in rollouts:
        groups[int(item["prompt_index"])].append(item)
    incomplete = sorted(index for index, items in groups.items() if len(items) != expected_group_size)
    if incomplete:
        raise ValueError(f"calibration groups have unexpected size: {incomplete}")
    rewards = [float(item["reward"]) for item in rollouts]
    strict_rate = sum(bool(item["strict_json"]) for item in rollouts) / len(rollouts)
    backend_rate = sum(bool(item["backend_error"]) for item in rollouts) / len(rollouts)
    mixed_groups = {
        index
        for index, items in groups.items()
        if 0 < sum(float(item["reward"]) for item in items) < len(items)
    }
    mixed_faults = {
        str(groups[index][0]["fault_name"])
        for index in mixed_groups
    }
    mean_reward = sum(rewards) / len(rewards)
    mixed_group_rate = len(mixed_groups) / len(groups)
    gates = {
        "backend_error_rate_zero": backend_rate == 0.0,
        "strict_json_rate": strict_rate >= min_strict_json_rate,
        "mean_reward_floor": mean_reward >= min_mean_reward,
        "mean_reward_ceiling": mean_reward <= max_mean_reward,
        "mixed_group_rate": mixed_group_rate >= min_mixed_group_rate,
        "mixed_fault_families": len(mixed_faults) >= min_mixed_fault_families,
    }
    per_fault: dict[str, Any] = {}
    for fault in FAULT_NAMES:
        items = [item for item in rollouts if item["fault_name"] == fault]
        per_fault[fault] = {
            "rollouts": len(items),
            "mean_reward": sum(float(item["reward"]) for item in items) / len(items) if items else 0.0,
            "mixed_groups": sum(
                1 for index in mixed_groups if groups[index][0]["fault_name"] == fault
            ),
        }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "rollouts": len(rollouts),
        "prompt_groups": len(groups),
        "mean_reward": mean_reward,
        "reward_counts": dict(sorted(Counter(rewards).items(), key=lambda item: item[0])),
        "strict_json_rate": strict_rate,
        "backend_error_rate": backend_rate,
        "mixed_groups": len(mixed_groups),
        "mixed_group_rate": mixed_group_rate,
        "mixed_fault_families": sorted(mixed_faults),
        "per_fault": per_fault,
    }


def _write_outputs(output_dir: Path, report: Mapping[str, Any], rollouts: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "rollouts.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in rollouts:
            handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    attempts = report.get("attempts", [])
    width, height = 760, 420
    bars = []
    for index, attempt in enumerate(attempts if isinstance(attempts, list) else []):
        rate = float(attempt.get("mean_reward", 0.0))
        x = 100 + index * 190
        bar_height = 250 * rate
        bars.append(
            f'<rect x="{x}" y="{330-bar_height:.1f}" width="90" height="{bar_height:.1f}" fill="#2563eb"/>'
            f'<text x="{x+45}" y="355" text-anchor="middle" font-family="sans-serif">T={attempt.get("temperature")}</text>'
            f'<text x="{x+45}" y="{315-bar_height:.1f}" text-anchor="middle" font-family="sans-serif">{rate:.1%}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="380" y="35" text-anchor="middle" font-family="sans-serif" font-size="22">Calibration mechanical reward</text>'
        '<line x1="70" y1="330" x2="700" y2="330" stroke="black"/>'
        + "".join(bars)
        + "</svg>\n"
    )
    (output_dir / "calibration.svg").write_text(svg, encoding="utf-8")


def calibrate(
    rows: Sequence[Mapping[str, Any]],
    generate_group: Callable[[Sequence[Mapping[str, str]], float, int], Sequence[str]],
    *,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    num_generations: int = 8,
    prompts_per_fault_profile: int = 2,
    reward_workers: int = 8,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Escalate sampling temperature until reward variance gates pass."""

    selected = select_calibration_rows(
        rows,
        prompts_per_fault_profile=prompts_per_fault_profile,
    )
    attempts: list[dict[str, Any]] = []
    all_rollouts: list[dict[str, Any]] = []
    selected_temperature: float | None = None
    if reward_workers < 1:
        raise ValueError("reward_workers must be positive")
    with ThreadPoolExecutor(max_workers=reward_workers) as executor:
        for temperature in temperatures:
            if progress is not None:
                progress(
                    f"calibration temperature={float(temperature):g}: "
                    f"0/{len(selected)} prompt groups"
                )
            temperature_rollouts: list[dict[str, Any]] = []
            for prompt_index, row in enumerate(selected):
                completions = list(
                    generate_group(row["prompt"], float(temperature), num_generations)
                )
                if len(completions) != num_generations:
                    raise RuntimeError("model returned an unexpected calibration group size")

                def score_completion(completion: str) -> dict[str, Any]:
                    extras: dict[str, list[Any]] = {}

                    def log_extra(name: str, values: list[Any]) -> None:
                        extras[name] = list(values)

                    reward = mechanical_reward(
                        [completion],
                        fault_name=[row["fault_name"]],
                        sample_seed=[row["sample_seed"]],
                        prompts=[row["prompt"]],
                        scenario_schema_version=[row["scenario_schema_version"]],
                        scenario_profile=[row["scenario_profile"]],
                        log_extra=log_extra,
                    )[0]
                    return {
                        "reward": float(reward),
                        "action": extras["crashdiag_action"][0],
                        "resolved": bool(extras["crashdiag_resolved"][0]),
                        "backend_error": bool(extras["crashdiag_backend_error"][0]),
                        "strict_json": bool(extras["crashdiag_strict_json"][0]),
                    }

                scored = list(executor.map(score_completion, completions))
                for generation_index, (completion, result) in enumerate(
                    zip(completions, scored, strict=True)
                ):
                    temperature_rollouts.append(
                        {
                            "temperature": float(temperature),
                            "prompt_index": prompt_index,
                            "generation_index": generation_index,
                            "fault_name": row["fault_name"],
                            "scenario_profile": row["scenario_profile"],
                            "sample_seed": row["sample_seed"],
                            "completion": completion,
                            **result,
                        }
                    )
                if progress is not None:
                    progress(
                        f"calibration temperature={float(temperature):g}: "
                        f"{prompt_index + 1}/{len(selected)} prompt groups"
                    )
            summary = summarize_temperature(
                temperature_rollouts,
                expected_group_size=num_generations,
            )
            summary["temperature"] = float(temperature)
            attempts.append(summary)
            all_rollouts.extend(temperature_rollouts)
            if progress is not None:
                progress(
                    f"temperature={float(temperature):g} passed={summary['passed']} "
                    f"reward={summary['mean_reward']:.3f} "
                    f"mixed_groups={summary['mixed_groups']}/{summary['prompt_groups']}"
                )
            if summary["passed"]:
                selected_temperature = float(temperature)
                break
    report = {
        "schema_version": 1,
        "scoring": "mechanical_fault_resolution",
        "scenario_schema_version": HARD_SCENARIO_SCHEMA_VERSION,
        "num_generations": num_generations,
        "prompts_per_fault_profile": prompts_per_fault_profile,
        "reward_workers": reward_workers,
        "selected_temperature": selected_temperature,
        "passed": selected_temperature is not None,
        "attempts": attempts,
    }
    return report, all_rollouts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/sft")
    parser.add_argument("--train-file", default="data/grpo_hard_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grpo-calibration"))
    parser.add_argument("--temperatures", type=float, nargs="+", default=list(DEFAULT_TEMPERATURES))
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--prompts-per-fault-profile", type=int, default=2)
    parser.add_argument("--reward-workers", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--sandbox-url", default=os.environ.get("CRASHDIAG_SANDBOX_URL", ""))
    parser.add_argument(
        "--sandbox-token",
        default=os.environ.get("CRASHDIAG_API_TOKEN") or os.environ.get("CRASHDIAG_SANDBOX_TOKEN"),
    )
    parser.add_argument("--sandbox-timeout", type=float, default=15.0)
    parser.add_argument("--artifact-stage", default="calibration")
    add_artifact_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    preload_env(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.num_generations < 2
        or args.prompts_per_fault_profile < 1
        or args.reward_workers < 1
    ):
        parser.error("calibration requires at least two generations and one prompt per cell")
    if any(not math.isfinite(value) or value <= 0 for value in args.temperatures):
        parser.error("calibration temperatures must be finite and positive")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("--top-p must be finite and in (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k cannot be negative")
    try:
        uploader = uploader_from_args(args)
        if uploader is not None:
            uploader.start_stage(
                args.artifact_stage,
                {
                    "model": args.model,
                    "train_file": args.train_file,
                    "temperatures": args.temperatures,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "mechanical_reward": True,
                },
            )
        configure_reward_backend(
            sandbox_url=args.sandbox_url,
            api_token=args.sandbox_token,
            timeout=args.sandbox_timeout,
        )
        rows = read_jsonl(args.train_file)
        model, tokenizer = load_local_policy(
            args.model,
            precision=args.precision,
            trust_remote_code=args.trust_remote_code,
        )

        def generate_group(messages: Sequence[Mapping[str, str]], temperature: float, count: int) -> Sequence[str]:
            return generate_from_messages(
                model,
                tokenizer,
                messages,
                num_return_sequences=count,
                temperature=temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
            )

        report, rollouts = calibrate(
            rows,
            generate_group,
            temperatures=args.temperatures,
            num_generations=args.num_generations,
            prompts_per_fault_profile=args.prompts_per_fault_profile,
            reward_workers=args.reward_workers,
            progress=print,
        )
        report["sampling"] = {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        }
        _write_outputs(args.output_dir, report, rollouts)
        if uploader is not None:
            uploader.upload_directory(
                args.output_dir,
                args.artifact_stage,
                metadata={
                    "passed": report["passed"],
                    "selected_temperature": report["selected_temperature"],
                    "sampling": report["sampling"],
                    "scoring": report["scoring"],
                },
            )
    except (ArtifactError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"GRPO calibration failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        print(
            "GRPO calibration found no usable mechanical reward variance; "
            "full training aborted",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
