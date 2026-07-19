"""Fail-closed promotion gates for the hard GRPO workflow."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class GateError(RuntimeError):
    """Raised when a training stage cannot be promoted safely."""


def _read(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"invalid gate input: {source}") from exc
    if not isinstance(value, Mapping):
        raise GateError(f"gate input must be a JSON object: {source}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoke_gate(
    trainer_state_path: str | Path,
    output_adapter_path: str | Path,
    parent_reference_path: str | Path,
    *,
    minimum_steps: int = 20,
) -> dict[str, Any]:
    """Require real reward variance, gradients, and a changed adapter."""

    state = _read(trainer_state_path)
    history = state.get("log_history")
    if not isinstance(history, list):
        raise GateError("smoke trainer state has no log_history")
    numeric_records: list[Mapping[str, Any]] = [item for item in history if isinstance(item, Mapping)]
    nonfinite = [
        str(key)
        for item in numeric_records
        for key, value in item.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isfinite(float(value))
    ]
    max_step = max(
        (int(item.get("step", 0)) for item in numeric_records if isinstance(item.get("step", 0), (int, float))),
        default=0,
    )
    reward_std = [
        float(item[key])
        for item in numeric_records
        for key in ("reward_std", "rewards/mechanical_reward/std")
        if isinstance(item.get(key), (int, float))
    ]
    gradients = [
        float(item["grad_norm"])
        for item in numeric_records
        if isinstance(item.get("grad_norm"), (int, float))
    ]
    backend_rates = [
        float(item["crashdiag/backend_error_rate"])
        for item in numeric_records
        if isinstance(item.get("crashdiag/backend_error_rate"), (int, float))
    ]
    success_rates = [
        float(item["crashdiag/success_rate"])
        for item in numeric_records
        if isinstance(item.get("crashdiag/success_rate"), (int, float))
    ]
    parent = _read(parent_reference_path)
    parent_sha = parent.get("adapter_sha256")
    adapter_path = Path(output_adapter_path)
    if not adapter_path.is_file():
        raise GateError(f"smoke adapter is missing: {adapter_path}")
    adapter_sha = _sha256(adapter_path)
    gates = {
        "minimum_steps": max_step >= minimum_steps,
        "finite_metrics": not nonfinite,
        "positive_reward_std": any(value > 0 for value in reward_std),
        "positive_gradient_norm": any(value > 0 for value in gradients),
        "mixed_success_rates": any(0 < value < 1 for value in success_rates),
        "backend_error_rate_zero": bool(backend_rates) and max(backend_rates) == 0.0,
        "adapter_changed_from_sft": isinstance(parent_sha, str) and adapter_sha != parent_sha,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "max_step": max_step,
        "max_reward_std": max(reward_std, default=0.0),
        "max_gradient_norm": max(gradients, default=0.0),
        "backend_error_rate": max(backend_rates, default=None),
        "parent_adapter_sha256": parent_sha,
        "smoke_adapter_sha256": adapter_sha,
        "nonfinite_metrics": nonfinite,
    }


def promotion_gate(
    hard_evaluation_path: str | Path,
    regression_evaluation_path: str | Path,
    *,
    minimum_hard_success: float = 0.70,
    minimum_hard_fault_success: float = 0.50,
    minimum_regression_success: float = 0.95,
) -> dict[str, Any]:
    """Require hard generalization without losing the schema-v1 baseline."""

    hard = _read(hard_evaluation_path)
    regression = _read(regression_evaluation_path)
    hard_summary = hard.get("summary", {})
    regression_summary = regression.get("summary", {})
    hard_faults = hard.get("per_fault", {})
    if not all(isinstance(value, Mapping) for value in (hard_summary, regression_summary, hard_faults)):
        raise GateError("evaluation reports have invalid summaries")
    hard_rate = float(hard_summary.get("success_rate", 0.0))
    regression_rate = float(regression_summary.get("success_rate", 0.0))
    hard_backend = float(hard_summary.get("backend_error_rate", 1.0))
    regression_backend = float(regression_summary.get("backend_error_rate", 1.0))
    per_fault_rates = [
        float(value.get("success_rate", 0.0))
        for value in hard_faults.values()
        if isinstance(value, Mapping)
    ]
    gates = {
        "hard_complete": int(hard_summary.get("total_episodes", 0)) == 192,
        "hard_success": hard_rate >= minimum_hard_success,
        "hard_per_fault": len(per_fault_rates) == 6
        and min(per_fault_rates, default=0.0) >= minimum_hard_fault_success,
        "regression_complete": int(regression_summary.get("total_episodes", 0)) >= 96,
        "regression_success": regression_rate >= minimum_regression_success,
        "backend_error_rate_zero": hard_backend == 0.0 and regression_backend == 0.0,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "hard_success_rate": hard_rate,
        "minimum_hard_fault_success_rate": min(per_fault_rates, default=0.0),
        "regression_success_rate": regression_rate,
        "thresholds": {
            "hard_success": minimum_hard_success,
            "hard_fault_success": minimum_hard_fault_success,
            "regression_success": minimum_regression_success,
        },
    }


def write_gate(path: str | Path, result: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def require_passed(result: Mapping[str, Any], label: str) -> None:
    if result.get("passed") is not True:
        failed = [name for name, passed in result.get("gates", {}).items() if passed is not True]
        raise GateError(f"{label} failed: {', '.join(failed)}")


__all__ = ["GateError", "promotion_gate", "require_passed", "smoke_gate", "write_gate"]
