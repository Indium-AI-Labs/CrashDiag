"""Deterministic, answer-free scenarios for GRPO schema version 2.

The original dataset intentionally exposes a complete state snapshot, which is
useful for SFT but too direct once that policy has saturated.  This module keeps
the same six one-action faults while presenting an operator-style view: raw
signals and genuine event history remain, but derived failure labels and
known-good configuration are withheld.

Every scenario can be reconstructed from ``fault_name``, ``sample_seed``, and
``scenario_profile`` against either :class:`MockSandbox` or :class:`HttpSandbox`.
The hidden expert action is used only to prove one-step solvability during data
generation; it is never serialized into a GRPO row.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from typing import Any

from crashdiag.sandbox_apps.mock import MockSandbox, SandboxBackend

from .common import FAULT_NAMES, fault_for_name


HARD_SCENARIO_SCHEMA_VERSION = 2
HARD_SCENARIO_PROFILES = ("redacted", "noisy", "shifted_noisy")
HARD_SYSTEM_PROMPT = """You diagnose a failing application from incomplete operational telemetry.
Recent logs may include incidents that were already repaired and unsuccessful remediation attempts.
Choose exactly one action from this list:
- restart_app
- rollback_env_var
- fix_dependency
- clear_disk
- fix_port_config
- wait_and_observe

Reply with one JSON object only, using this schema:
{"action": "<action name>", "parameters": {}}
The parameters value must be a JSON object. When a desired configuration value is not observable,
leave optional parameters out instead of guessing. Do not use markdown or prose.
"""

_APP_ENV_VALUES = ("production", "staging", "canary")
_DEPENDENCY_VERSIONS = ("1.4.2", "1.5.0", "1.6.3", "2.1.1")
_APP_PORTS = (3000, 8000, 8080, 8443, 9000)
_DISK_THRESHOLDS = (75.0, 80.0, 85.0, 90.0, 95.0)
_BAD_DEPENDENCY_VERSIONS = ("0.9.0", "1.3.9", "2.0.0-incompatible", "9.9.9")
_BAD_APP_ENV_VALUES = ("invalid", "prodution", "development", "PRODUCTION")


def hard_sample_seed(base_seed: int, fault_name: str, variation_index: int) -> int:
    """Return a stable int64 seed in a namespace disjoint from schema v1."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if fault_name not in FAULT_NAMES:
        raise ValueError(f"unknown fault name: {fault_name!r}")
    if (
        isinstance(variation_index, bool)
        or not isinstance(variation_index, int)
        or variation_index < 0
    ):
        raise ValueError("variation_index must be a non-negative integer")
    material = (
        f"crashdiag:hard-grpo:v{HARD_SCENARIO_SCHEMA_VERSION}:"
        f"{base_seed}:{fault_name}:{variation_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def profile_for_variation(variation_index: int) -> str:
    """Balance profiles deterministically across consecutive variations."""

    if (
        isinstance(variation_index, bool)
        or not isinstance(variation_index, int)
        or variation_index < 0
    ):
        raise ValueError("variation_index must be a non-negative integer")
    return HARD_SCENARIO_PROFILES[variation_index % len(HARD_SCENARIO_PROFILES)]


def _validate_profile(profile: str) -> str:
    if profile not in HARD_SCENARIO_PROFILES:
        raise ValueError(
            f"unknown scenario profile {profile!r}; expected one of "
            + ", ".join(HARD_SCENARIO_PROFILES)
        )
    return profile


def _configure_baseline(
    sandbox: SandboxBackend,
    rng: random.Random,
    profile: str,
) -> None:
    if profile != "shifted_noisy":
        return
    suffix = rng.randrange(10_000, 99_999)
    sandbox.set_expected_env_var("APP_ENV", rng.choice(_APP_ENV_VALUES))
    sandbox.set_expected_env_var(
        "DATABASE_URL",
        f"postgresql://app:secret@db-{suffix}:5432/app_{suffix}",
    )
    sandbox.set_required_dependency_version(
        "web-framework", rng.choice(_DEPENDENCY_VERSIONS)
    )
    sandbox.set_app_port(rng.choice(_APP_PORTS))
    sandbox.set_disk_health_threshold(rng.choice(_DISK_THRESHOLDS))


def _prepare_background(
    sandbox: SandboxBackend,
    rng: random.Random,
) -> None:
    observation = sandbox.observe()
    disk = observation.get("disk", {})
    threshold = float(disk.get("healthy_below_percent", 90.0))
    upper = max(16.0, min(70.0, threshold - 5.0))
    sandbox.set_disk_usage(round(rng.uniform(15.0, upper), 1))
    for _ in range(rng.randrange(4)):
        sandbox.wait_and_observe()
    for _ in range(rng.randrange(2)):
        sandbox.restart_app()


def _vary_fault(
    fault: Any,
    sandbox: SandboxBackend,
    rng: random.Random,
) -> None:
    observation = sandbox.observe()
    if fault.name == "bad_env_var":
        current = observation.get("environment", {}).get("variables", {}).get("APP_ENV")
        choices = [value for value in _BAD_APP_ENV_VALUES if value != current]
        fault.bad_value = rng.choice(choices)
    elif fault.name == "broken_db_connection":
        suffix = rng.randrange(10_000, 99_999)
        fault.bad_value = rng.choice(
            (
                f"postgresql://app:secret@missing-{suffix}:5432/app",
                f"postgresql://app:secret@db-{suffix}.invalid:5432/app",
                f"postgresql://app:secret@db-{suffix}:15432/app",
                f"postgresql://app:secret@db-{suffix}:5432/missing_app",
            )
        )
    elif fault.name == "dependency_mismatch":
        required = (
            observation.get("dependencies", {})
            .get("required", {})
            .get("web-framework")
        )
        choices = [value for value in _BAD_DEPENDENCY_VERSIONS if value != required]
        fault.bad_version = rng.choice(choices)
    elif fault.name == "disk_full":
        threshold = float(
            observation.get("disk", {}).get("healthy_below_percent", 90.0)
        )
        lower = min(100.0, threshold + 1.0)
        fault.injected_percent = round(rng.uniform(lower, 100.0), 1)
    elif fault.name == "port_proxy_misconfig":
        app_port = int(observation.get("network", {}).get("app_port", 8080))
        choices = [port for port in (80, 3000, 8000, 8080, 8081, 8443, 8888, 9000, 65535) if port != app_port]
        fault.wrong_port = rng.choice(choices)


def hard_expert_action(fault_name: str) -> dict[str, Any]:
    """Return the parameter-minimal action used only for mechanical validation."""

    actions = {
        "oom_kill": "restart_app",
        "bad_env_var": "rollback_env_var",
        "broken_db_connection": "rollback_env_var",
        "dependency_mismatch": "fix_dependency",
        "disk_full": "clear_disk",
        "port_proxy_misconfig": "fix_port_config",
    }
    try:
        action = actions[fault_name]
    except KeyError as exc:
        raise ValueError(f"unknown fault name: {fault_name!r}") from exc
    return {"action": action, "parameters": {}}


def _inject_and_repair_decoy(
    sandbox: SandboxBackend,
    fault_name: str,
    rng: random.Random,
) -> None:
    fault = fault_for_name(fault_name)
    _vary_fault(fault, sandbox, rng)
    fault.inject(sandbox)
    if fault.is_resolved(sandbox):
        raise RuntimeError(f"decoy fault {fault_name!r} did not inject")
    action = hard_expert_action(fault_name)
    sandbox.execute_action(action["action"], action["parameters"])
    if not fault.is_resolved(sandbox):
        raise RuntimeError(f"decoy fault {fault_name!r} did not repair")


def _add_real_stale_history(
    sandbox: SandboxBackend,
    active_fault_name: str,
    rng: random.Random,
) -> None:
    candidates = [name for name in FAULT_NAMES if name != active_fault_name]
    for decoy in rng.sample(candidates, 2):
        _inject_and_repair_decoy(sandbox, decoy, rng)


def _add_unsuccessful_remediation(
    sandbox: SandboxBackend,
    active_fault_name: str,
) -> None:
    # This action is deliberately real and mechanically harmless.  It occurs
    # after the active fault so recency alone cannot identify root cause.
    if active_fault_name == "oom_kill":
        sandbox.fix_dependency()
    else:
        sandbox.restart_app()


def prepare_hard_scenario(
    fault_name: str,
    scenario_seed: int,
    scenario_profile: str,
    *,
    sandbox: SandboxBackend | None = None,
) -> tuple[Any, SandboxBackend, random.Random]:
    """Reconstruct one schema-v2 scenario on a local or remote sandbox."""

    if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int):
        raise TypeError("scenario_seed must be an integer")
    profile = _validate_profile(scenario_profile)
    fault = fault_for_name(fault_name)
    rng = random.Random(scenario_seed)
    target = sandbox if sandbox is not None else MockSandbox()
    _configure_baseline(target, rng, profile)
    _prepare_background(target, rng)
    if profile in {"noisy", "shifted_noisy"}:
        _add_real_stale_history(target, fault_name, rng)
    _vary_fault(fault, target, rng)
    fault.inject(target)
    if profile in {"noisy", "shifted_noisy"}:
        _add_unsuccessful_remediation(target, fault_name)
    if fault.is_resolved(target):
        raise RuntimeError(f"fault {fault_name!r} was resolved immediately after injection")
    health = target.health_check()
    if not isinstance(health, Mapping) or health.get("healthy") is not False:
        raise RuntimeError(f"fault {fault_name!r} did not make the sandbox unhealthy")
    return fault, target, rng


def hard_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return raw operational evidence without derived answers or ground truth."""

    process = observation.get("process", {})
    environment = observation.get("environment", {})
    dependencies = observation.get("dependencies", {})
    disk = observation.get("disk", {})
    network = observation.get("network", {})
    recent_logs = (
        [str(item) for item in observation.get("recent_logs", [])]
        if isinstance(observation.get("recent_logs", []), list)
        else []
    )
    if isinstance(network, Mapping):
        app_port = network.get("app_port")
        if isinstance(app_port, int) and not isinstance(app_port, bool):
            recent_logs = [
                item.replace(f"port {app_port}", "port <redacted>")
                for item in recent_logs
            ]
    return {
        "http": {
            "status": observation.get("health_state"),
            "status_code": observation.get("http_status"),
        },
        "process": dict(process) if isinstance(process, Mapping) else {},
        "environment": {
            "variables": dict(environment.get("variables", {}))
            if isinstance(environment, Mapping)
            and isinstance(environment.get("variables", {}), Mapping)
            else {}
        },
        "dependencies": {
            "installed": dict(dependencies.get("installed", {}))
            if isinstance(dependencies, Mapping)
            and isinstance(dependencies.get("installed", {}), Mapping)
            else {}
        },
        "disk": {
            "used_percent": disk.get("used_percent")
            if isinstance(disk, Mapping)
            else None
        },
        "network": {
            "proxy_target_port": network.get("proxy_target_port")
            if isinstance(network, Mapping)
            else None
        },
        "clock_ticks": observation.get("clock_ticks"),
        "recent_logs": recent_logs,
    }


def hard_observation_messages(observation: Mapping[str, Any]) -> list[dict[str, str]]:
    """Render the exact schema-v2 conversational prompt."""

    content = json.dumps(
        {"observation": hard_observation(observation)},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return [
        {"role": "system", "content": HARD_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_hard_grpo_sample(
    fault_name: str,
    *,
    base_seed: int,
    variation_index: int,
    split: str,
) -> dict[str, Any]:
    """Build one answer-free row after proving one-step mechanical solvability."""

    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    profile = profile_for_variation(variation_index)
    scenario_seed = hard_sample_seed(base_seed, fault_name, variation_index)
    fault, target, _ = prepare_hard_scenario(
        fault_name,
        scenario_seed,
        profile,
    )
    prompt = hard_observation_messages(target.observe())
    action = hard_expert_action(fault_name)
    target.execute_action(action["action"], action["parameters"])
    if not fault.is_resolved(target) or target.health_check().get("healthy") is not True:
        raise RuntimeError(f"hard expert action failed for {fault_name!r}")
    return {
        "fault_name": fault.name,
        "difficulty": "hard",
        "sample_seed": scenario_seed,
        "variation_index": variation_index,
        "scenario_schema_version": HARD_SCENARIO_SCHEMA_VERSION,
        "scenario_profile": profile,
        "prompt": prompt,
        "metadata": {
            "schema_version": HARD_SCENARIO_SCHEMA_VERSION,
            "mechanically_validated": True,
            "split": split,
            "variation_index": variation_index,
            "scenario_profile": profile,
        },
    }


def generate_hard_records(
    *,
    samples_per_fault: int,
    seed: int,
    start_variation: int,
    split: str,
) -> list[dict[str, Any]]:
    """Generate balanced schema-v2 records for all six existing faults."""

    if (
        isinstance(samples_per_fault, bool)
        or not isinstance(samples_per_fault, int)
        or samples_per_fault < 1
    ):
        raise ValueError("samples_per_fault must be a positive integer")
    rows: list[dict[str, Any]] = []
    for variation_index in range(start_variation, start_variation + samples_per_fault):
        for fault_name in FAULT_NAMES:
            rows.append(
                build_hard_grpo_sample(
                    fault_name,
                    base_seed=seed,
                    variation_index=variation_index,
                    split=split,
                )
            )
    return rows


__all__ = [
    "HARD_SCENARIO_PROFILES",
    "HARD_SCENARIO_SCHEMA_VERSION",
    "HARD_SYSTEM_PROMPT",
    "build_hard_grpo_sample",
    "generate_hard_records",
    "hard_expert_action",
    "hard_observation",
    "hard_observation_messages",
    "hard_sample_seed",
    "prepare_hard_scenario",
    "profile_for_variation",
]
