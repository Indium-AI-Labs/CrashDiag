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
from crashdiag.faults.workflows import WORKFLOWS, workflow_for_name

from .common import FAULT_NAMES, fault_for_name


HARD_SCENARIO_SCHEMA_VERSION = 5
HARD_CURRICULUM_VERSION = 5
HARD_SCENARIO_PROFILES = ("redacted", "noisy", "shifted_noisy")
HARD_SCENARIO_SCHEMA_VERSION_V3 = 3
HARD_SYSTEM_PROMPT = """You diagnose a failing application from incomplete operational telemetry.
Recent logs may include incidents that were already repaired and unsuccessful remediation attempts.
Choose an ordered list of actions from this list:
- restart_app
- rollback_env_var
- fix_dependency
- clear_disk
- fix_port_config
- clear_cache
- renew_tls_certificate
- restore_file_permissions
- apply_database_migration
- reset_database_pool
- restore_dns_configuration
- restore_rate_limit_configuration
- restart_worker
- redeploy_container
- clear_temp_files
- rotate_logs
- restore_load_balancer_config
- restore_network_config
- sync_replica
- restore_database_config
- flush_dead_letter_queue
- restore_cache_config
- reset_circuit_breaker
- restore_cron_schedule
- rebuild_index
- restore_tls_config
- wait_and_observe

Reply with one JSON object only, using this schema:
{"actions": [{"action": "<action name>", "parameters": {}}]}
For this sandbox, each repair action restores its target from deployment history or declared
configuration. The parameters value must therefore be exactly {} for every action. Never guess
or emit names, versions, ports, or thresholds. Do not use markdown or prose.
"""

_APP_ENV_VALUES = ("production", "staging", "canary")
_DEPENDENCY_VERSIONS = ("1.4.2", "1.5.0", "1.6.3", "2.1.1")
_APP_PORTS = (3000, 8000, 8080, 8443, 9000)
_DISK_THRESHOLDS = (75.0, 80.0, 85.0, 90.0, 95.0)
_BAD_DEPENDENCY_VERSIONS = ("0.9.0", "1.3.9", "2.0.0-incompatible", "9.9.9")
_BAD_APP_ENV_VALUES = ("invalid", "prodution", "development", "PRODUCTION")
_OPAQUE_SIGNATURES = {
    "oom_kill": "sig-7f3a",
    "bad_env_var": "sig-c91e",
    "broken_db_connection": "sig-2bd8",
    "dependency_mismatch": "sig-a64c",
    "disk_full": "sig-54d1",
    "port_proxy_misconfig": "sig-e83b",
    "missing_secret": "sig-19af",
    "feature_flag_misconfiguration": "sig-b276",
    "redis_connection_failure": "sig-6ce4",
    "message_queue_connection_failure": "sig-d508",
    "object_storage_credentials_failure": "sig-38b9",
    "cache_corruption": "sig-f142",
    "tls_certificate_failure": "sig-9a65",
    "file_permission_failure": "sig-47de",
    "schema_migration_pending": "sig-0cb3",
    "database_pool_exhaustion": "sig-8e91",
    "dns_resolution_failure": "sig-31d7",
    "rate_limit_misconfiguration": "sig-65a0",
}

_OPAQUE_SIGNATURES_V4 = {
    "oom_kill": "sig-7f3a",
    "tls_certificate_failure": "sig-7f3a",
    "bad_env_var": "sig-c91e",
    "file_permission_failure": "sig-c91e",
    "broken_db_connection": "sig-2bd8",
    "schema_migration_pending": "sig-2bd8",
    "dependency_mismatch": "sig-a64c",
    "database_pool_exhaustion": "sig-a64c",
    "disk_full": "sig-54d1",
    "dns_resolution_failure": "sig-54d1",
    "port_proxy_misconfig": "sig-e83b",
    "rate_limit_misconfiguration": "sig-e83b",
    "missing_secret": "sig-19af",
    "cache_corruption": "sig-19af",
    "feature_flag_misconfiguration": "sig-b276",
    "redis_connection_failure": "sig-b276",
    "message_queue_connection_failure": "sig-d508",
    "object_storage_credentials_failure": "sig-d508",
}

_SENSOR_POOL = (
    "sensor-01", "sensor-02", "sensor-03", "sensor-04", "sensor-05",
    "sensor-06", "sensor-07", "sensor-08", "sensor-09", "sensor-10",
    "sensor-11", "sensor-12", "sensor-13", "sensor-14", "sensor-15",
    "sensor-16", "sensor-17", "sensor-18", "sensor-19", "sensor-20",
)

_FAULT_SIGNAL_BIAS: dict[str, tuple[str, ...]] = {
    "oom_kill": ("sensor-01", "sensor-04", "sensor-11"),
    "bad_env_var": ("sensor-02", "sensor-05", "sensor-12"),
    "broken_db_connection": ("sensor-02", "sensor-06", "sensor-13"),
    "dependency_mismatch": ("sensor-03", "sensor-07", "sensor-14"),
    "disk_full": ("sensor-03", "sensor-08", "sensor-15"),
    "port_proxy_misconfig": ("sensor-04", "sensor-09", "sensor-16"),
    "missing_secret": ("sensor-05", "sensor-10", "sensor-17"),
    "feature_flag_misconfiguration": ("sensor-06", "sensor-11", "sensor-18"),
    "redis_connection_failure": ("sensor-07", "sensor-12", "sensor-19"),
    "message_queue_connection_failure": ("sensor-08", "sensor-13", "sensor-20"),
    "object_storage_credentials_failure": ("sensor-09", "sensor-14", "sensor-01"),
    "cache_corruption": ("sensor-10", "sensor-15", "sensor-02"),
    "tls_certificate_failure": ("sensor-11", "sensor-16", "sensor-03"),
    "file_permission_failure": ("sensor-12", "sensor-17", "sensor-04"),
    "schema_migration_pending": ("sensor-13", "sensor-18", "sensor-05"),
    "database_pool_exhaustion": ("sensor-14", "sensor-19", "sensor-06"),
    "dns_resolution_failure": ("sensor-15", "sensor-20", "sensor-07"),
    "rate_limit_misconfiguration": ("sensor-16", "sensor-01", "sensor-08"),
}

_ACTION_BY_FAULT = {
    "oom_kill": "restart_app", "bad_env_var": "rollback_env_var",
    "broken_db_connection": "rollback_env_var", "dependency_mismatch": "fix_dependency",
    "disk_full": "clear_disk", "port_proxy_misconfig": "fix_port_config",
    "missing_secret": "rollback_env_var", "feature_flag_misconfiguration": "rollback_env_var",
    "redis_connection_failure": "rollback_env_var", "message_queue_connection_failure": "rollback_env_var",
    "object_storage_credentials_failure": "rollback_env_var", "cache_corruption": "clear_cache",
    "tls_certificate_failure": "renew_tls_certificate", "file_permission_failure": "restore_file_permissions",
    "schema_migration_pending": "apply_database_migration", "database_pool_exhaustion": "reset_database_pool",
    "dns_resolution_failure": "restore_dns_configuration", "rate_limit_misconfiguration": "restore_rate_limit_configuration",
    "stale_secret_rotation": "rollback_env_var", "api_key_expired": "rollback_env_var",
    "env_var_case_mismatch": "rollback_env_var", "worker_down": "restart_worker",
    "container_image_drift": "redeploy_container", "temp_file_pressure": "clear_temp_files",
    "log_volume_exceeded": "rotate_logs", "load_balancer_misroute": "restore_load_balancer_config",
    "network_config_misconfigured": "restore_network_config", "replica_lag": "sync_replica",
    "database_config_misconfigured": "restore_database_config", "dead_letter_backlog": "flush_dead_letter_queue",
    "cache_config_misconfigured": "restore_cache_config", "circuit_breaker_open": "reset_circuit_breaker",
    "cron_skipped": "restore_cron_schedule", "search_index_stale": "rebuild_index",
    "tls_config_misconfigured": "restore_tls_config",
}

_ALL_REPAIR_ACTIONS = tuple(
    action for action in (
        "restart_app", "rollback_env_var", "fix_dependency", "clear_disk", "fix_port_config",
        "clear_cache", "renew_tls_certificate", "restore_file_permissions",
        "apply_database_migration", "reset_database_pool", "restore_dns_configuration",
        "restore_rate_limit_configuration",
    )
)


def _active_fault_name(observation: Mapping[str, Any]) -> str:
    """Infer the active mechanical fault without exposing its raw state."""

    process = observation.get("process", {})
    if isinstance(process, Mapping) and process.get("running") is False:
        return "oom_kill"
    environment = observation.get("environment", {})
    if isinstance(environment, Mapping):
        variables = environment.get("variables", {})
        expected = environment.get("expected", {})
        if isinstance(variables, Mapping) and isinstance(expected, Mapping):
            mismatches = {
                key for key, value in expected.items() if variables.get(key) != value
            }
            env_faults = {
                "APP_ENV": "bad_env_var",
                "DATABASE_URL": "broken_db_connection",
                "API_SIGNING_SECRET": "missing_secret",
                "FEATURE_ASYNC_JOBS": "feature_flag_misconfiguration",
                "REDIS_URL": "redis_connection_failure",
                "QUEUE_URL": "message_queue_connection_failure",
                "OBJECT_STORAGE_TOKEN": "object_storage_credentials_failure",
            }
            for key, fault_name in env_faults.items():
                if key in mismatches:
                    return fault_name
    dependencies = observation.get("dependencies", {})
    if isinstance(dependencies, Mapping):
        installed = dependencies.get("installed", {})
        required = dependencies.get("required", {})
        if isinstance(installed, Mapping) and isinstance(required, Mapping):
            if any(installed.get(key) != value for key, value in required.items()):
                return "dependency_mismatch"
    disk = observation.get("disk", {})
    if isinstance(disk, Mapping):
        if float(disk.get("used_percent", 0.0)) >= float(disk.get("healthy_below_percent", 101.0)):
            return "disk_full"
    network = observation.get("network", {})
    if isinstance(network, Mapping) and network.get("proxy_target_port") != network.get("app_port"):
        return "port_proxy_misconfig"
    services = observation.get("services", {})
    if isinstance(services, Mapping):
        service_faults = {
            "cache": "cache_corruption", "tls": "tls_certificate_failure",
            "permissions": "file_permission_failure", "migration": "schema_migration_pending",
            "db_pool": "database_pool_exhaustion", "dns": "dns_resolution_failure",
            "rate_limit": "rate_limit_misconfiguration", "worker": "worker_down",
            "container": "container_image_drift", "temp": "temp_file_pressure",
            "logs": "log_volume_exceeded", "lb_config": "load_balancer_misroute",
            "network_config": "network_config_misconfigured", "replica": "replica_lag",
            "db_config": "database_config_misconfigured", "dead_letter": "dead_letter_backlog",
            "cache_config": "cache_config_misconfigured", "circuit_breaker": "circuit_breaker_open",
            "cron": "cron_skipped", "search_index": "search_index_stale",
            "tls_config": "tls_config_misconfigured",
        }
        for service, fault_name in service_faults.items():
            if services.get(service) is False:
                return fault_name
    raise ValueError("unable to derive an active fault signature from observation")


def hard_sample_seed(base_seed: int, fault_name: str, variation_index: int) -> int:
    """Return a stable int64 seed in a namespace disjoint from schema v1."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if fault_name not in FAULT_NAMES and fault_name not in WORKFLOWS:
        raise ValueError(f"unknown fault name: {fault_name!r}")
    if (
        isinstance(variation_index, bool)
        or not isinstance(variation_index, int)
        or variation_index < 0
    ):
        raise ValueError("variation_index must be a non-negative integer")
    material = (
        f"crashdiag:hard-grpo:schema{HARD_SCENARIO_SCHEMA_VERSION}:"
        f"curriculum{HARD_CURRICULUM_VERSION}:"
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

    try:
        action = _ACTION_BY_FAULT[fault_name]
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
    """Reconstruct one schema-v3 scenario on a local or remote sandbox."""

    if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int):
        raise TypeError("scenario_seed must be an integer")
    profile = _validate_profile(scenario_profile)
    fault = fault_for_name(fault_name)
    rng = random.Random(scenario_seed)
    target = sandbox if sandbox is not None else MockSandbox()
    native_prepare = getattr(target, "prepare_hard_scenario", None)
    if callable(native_prepare):
        try:
            prepared = native_prepare(fault_name, scenario_seed, profile)
        except NotImplementedError:
            # Older/alternate backends retain the portable sequence below.
            pass
        else:
            if not isinstance(prepared, Mapping):
                raise TypeError("native hard-scenario preparation must return a mapping")
            health = prepared.get("health")
            if not isinstance(health, Mapping) or health.get("healthy") is not False:
                raise RuntimeError(
                    f"native hard scenario {fault_name!r} did not become unhealthy"
                )
            return fault, target, rng
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


def _v4_signals(fault_name: str, internal_fingerprint: str) -> list[str]:
    biased = _FAULT_SIGNAL_BIAS.get(fault_name, ())
    seed = hashlib.sha256(f"crashdiag:v4:signals:{internal_fingerprint}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(seed[:8], "big"))
    chosen: list[str] = []
    pool = list(_SENSOR_POOL)
    rng.shuffle(pool)
    for sensor in biased[:3]:
        if sensor not in chosen:
            chosen.append(sensor)
    for sensor in pool:
        if len(chosen) >= 5:
            break
        if sensor not in chosen:
            chosen.append(sensor)
    chosen = chosen[:5]
    result: list[str] = []
    for idx, sensor in enumerate(chosen):
        is_biased = sensor in biased
        roll = rng.random()
        if is_biased:
            level = "red" if roll < 0.55 else "amber" if roll < 0.85 else "nominal"
        else:
            level = "nominal" if roll < 0.55 else "amber" if roll < 0.85 else "red"
        noise_flip = rng.random() < 0.15
        if noise_flip:
            level = {"red": "nominal", "amber": "red", "nominal": "amber"}[level]
        result.append(f"{sensor}:{level}")
    result.sort()
    return result


def _v4_recent_events(observation: Mapping[str, Any]) -> list[str]:
    history = observation.get("action_history") if isinstance(observation, Mapping) else None
    if not isinstance(history, list):
        recent_logs = observation.get("recent_logs") if isinstance(observation, Mapping) else None
        if isinstance(recent_logs, list) and recent_logs:
            return [str(entry)[:64] for entry in recent_logs[-3:]]
        return []
    events: list[str] = []
    clock = int(observation.get("clock_ticks", 0)) if isinstance(observation, Mapping) else 0
    for entry in history[-5:]:
        if not isinstance(entry, Mapping):
            continue
        action = str(entry.get("action", "unknown"))
        tick = int(entry.get("tick", clock)) if isinstance(entry.get("tick"), int) else clock
        delta = clock - tick
        changed = entry.get("changed")
        status = "ok" if changed is True else "noop" if changed is False else "unknown"
        events.append(f"{action}@tick-{delta}:{status}")
    return events[-5:]


def hard_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return evidence-only opaque telemetry for the v4 curriculum.

    v4 is strictly harder than v3: signature is 2:1 ambiguous, signals are
    fault-correlated with noise, candidate_repairs is removed (13-way), and
    recent_events exposes only action names + recency for decoy disambiguation.
    """

    fault_name = _active_fault_name(observation)
    signature = _OPAQUE_SIGNATURES_V4.get(
        fault_name,
        _OPAQUE_SIGNATURES.get(fault_name, "sig-v4"),
    )
    ticks = int(observation.get("clock_ticks", 0))
    internal_fingerprint = json.dumps(
        observation,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    window = hashlib.sha256(
        f"crashdiag:v4:telemetry:{internal_fingerprint}".encode("utf-8")
    ).hexdigest()[:12]
    signals = _v4_signals(fault_name, internal_fingerprint)
    recent_events = _v4_recent_events(observation)
    telemetry: dict[str, Any] = {
        "signature": signature,
        "signals": signals,
        "sample_clock": ticks,
    }
    if recent_events:
        telemetry["recent_events"] = recent_events
    return {
        "incident_window": {"gateway": "degraded", "http_family": "5xx", "window": window},
        "telemetry": telemetry,
    }


def hard_observation_v3(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Legacy v3 observation for replay compatibility."""

    fault_name = _active_fault_name(observation)
    signature = _OPAQUE_SIGNATURES[fault_name]
    ticks = int(observation.get("clock_ticks", 0))
    internal_fingerprint = json.dumps(
        observation,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    window = hashlib.sha256(
        f"crashdiag:v1:telemetry:{internal_fingerprint}".encode("utf-8")
    ).hexdigest()[:12]
    correct_action = _ACTION_BY_FAULT[fault_name]
    digest = hashlib.sha256(
        f"crashdiag:v1:candidates:{internal_fingerprint}".encode("utf-8")
    ).digest()
    distractors = [action for action in _ALL_REPAIR_ACTIONS if action != correct_action]
    offset = int.from_bytes(digest[:2], "big") % len(distractors)
    distractors = distractors[offset:] + distractors[:offset]
    candidate_repairs = [correct_action, *distractors[:3]]
    rotate = int.from_bytes(digest[2:4], "big") % len(candidate_repairs)
    candidate_repairs = candidate_repairs[rotate:] + candidate_repairs[:rotate]
    return {
        "incident_window": {"gateway": "degraded", "http_family": "5xx", "window": window},
        "telemetry": {
            "signature": signature,
            "signals": ["sensor-04:amber", "sensor-11:amber", "sensor-19:nominal"],
            "sample_clock": ticks,
        },
        "candidate_repairs": candidate_repairs,
    }


def hard_observation_messages(observation: Mapping[str, Any]) -> list[dict[str, str]]:
    """Render the exact schema-v4 conversational prompt."""

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


def hard_observation_messages_v3(observation: Mapping[str, Any]) -> list[dict[str, str]]:
    """Legacy v3 prompt for replay compatibility."""

    content = json.dumps(
        {"observation": hard_observation_v3(observation)},
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
        "curriculum_version": HARD_CURRICULUM_VERSION,
        "scenario_profile": profile,
        "prompt": prompt,
        "metadata": {
            "schema_version": HARD_SCENARIO_SCHEMA_VERSION,
            "curriculum_version": HARD_CURRICULUM_VERSION,
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
    """Generate balanced schema-v4 records for every supported fault."""

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


def _v5_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Render the v5 workflow observation.

    v5 keeps the operator-style opaque telemetry from v4 but removes the legacy
    ``candidate_repairs`` field and any hidden sub-fault labels.  The policy must
    infer the multi-action workflow from signals, signature, and recent events.
    """

    fault_name = _active_fault_name(observation)
    signature = _OPAQUE_SIGNATURES_V4.get(
        fault_name,
        _OPAQUE_SIGNATURES.get(fault_name, "sig-v5"),
    )
    ticks = int(observation.get("clock_ticks", 0))
    internal_fingerprint = json.dumps(
        observation,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    window = hashlib.sha256(
        f"crashdiag:v5:telemetry:{internal_fingerprint}".encode("utf-8")
    ).hexdigest()[:12]
    signals = _v4_signals(fault_name, internal_fingerprint)
    recent_events = _v4_recent_events(observation)
    telemetry: dict[str, Any] = {
        "signature": signature,
        "signals": signals,
        "sample_clock": ticks,
    }
    if recent_events:
        telemetry["recent_events"] = recent_events
    return {
        "incident_window": {"gateway": "degraded", "http_family": "5xx", "window": window},
        "telemetry": telemetry,
    }


def hard_observation_workflow(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the evidence-only opaque telemetry for the v5 curriculum."""

    return _v5_observation(observation)


def hard_observation_workflow_messages(
    observation: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Render the exact schema-v5 conversational prompt."""

    content = json.dumps(
        {"observation": hard_observation_workflow(observation)},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return [
        {"role": "system", "content": HARD_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def hard_expert_workflow(fault_name: str) -> dict[str, Any]:
    """Return the ordered expert action list used only for mechanical validation."""

    workflow = workflow_for_name(fault_name)
    return {
        "actions": [
            {"action": action, "parameters": {}}
            for action in workflow.actions
        ]
    }


def prepare_v5_scenario(
    fault_name: str,
    scenario_seed: int,
    scenario_profile: str,
    *,
    sandbox: SandboxBackend | None = None,
) -> tuple[Any, SandboxBackend, random.Random]:
    """Reconstruct one schema-v5 workflow scenario on a local or remote sandbox."""

    if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int):
        raise TypeError("scenario_seed must be an integer")
    profile = _validate_profile(scenario_profile)
    workflow = workflow_for_name(fault_name)
    rng = random.Random(scenario_seed)
    target = sandbox if sandbox is not None else MockSandbox()

    native_prepare = getattr(target, "prepare_v5_scenario", None)
    if callable(native_prepare):
        try:
            prepared = native_prepare(fault_name, scenario_seed, profile)
        except NotImplementedError:
            pass
        else:
            if not isinstance(prepared, Mapping):
                raise TypeError("native v5-scenario preparation must return a mapping")
            health = prepared.get("health")
            if not isinstance(health, Mapping) or health.get("healthy") is not False:
                raise RuntimeError(
                    f"native v5 scenario {fault_name!r} did not become unhealthy"
                )
            return workflow, target, rng

    _configure_baseline(target, rng, profile)
    _prepare_background(target, rng)
    if profile in {"noisy", "shifted_noisy"}:
        _add_real_stale_history(target, fault_name, rng)
    workflow.inject(target)
    if profile in {"noisy", "shifted_noisy"}:
        _add_unsuccessful_remediation(target, fault_name)
    if workflow.is_resolved(target):
        raise RuntimeError(f"workflow {fault_name!r} was resolved immediately after injection")
    health = target.health_check()
    if not isinstance(health, Mapping) or health.get("healthy") is not False:
        raise RuntimeError(f"workflow {fault_name!r} did not make the sandbox unhealthy")
    return workflow, target, rng


def build_v5_sample(
    fault_name: str,
    *,
    base_seed: int,
    variation_index: int,
    split: str,
) -> dict[str, Any]:
    """Build one answer-free row after proving multi-action mechanical solvability."""

    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    profile = profile_for_variation(variation_index)
    scenario_seed = hard_sample_seed(base_seed, fault_name, variation_index)
    workflow, target, _ = prepare_v5_scenario(
        fault_name,
        scenario_seed,
        profile,
    )
    prompt = hard_observation_workflow_messages(target.observe())
    expert = hard_expert_workflow(fault_name)
    for action in expert["actions"]:
        target.execute_action(action["action"], action["parameters"])
    if not workflow.is_resolved(target) or target.health_check().get("healthy") is not True:
        raise RuntimeError(f"v5 expert workflow failed for {fault_name!r}")
    return {
        "fault_name": workflow.name,
        "difficulty": workflow.difficulty,
        "subfault_count": workflow.subfault_count,
        "sample_seed": scenario_seed,
        "variation_index": variation_index,
        "scenario_schema_version": HARD_SCENARIO_SCHEMA_VERSION,
        "curriculum_version": HARD_CURRICULUM_VERSION,
        "scenario_profile": profile,
        "prompt": prompt,
        "metadata": {
            "schema_version": HARD_SCENARIO_SCHEMA_VERSION,
            "curriculum_version": HARD_CURRICULUM_VERSION,
            "mechanically_validated": True,
            "split": split,
            "variation_index": variation_index,
            "scenario_profile": profile,
            "subfault_count": workflow.subfault_count,
        },
    }


def generate_v5_records(
    *,
    samples_per_fault: int,
    seed: int,
    start_variation: int,
    split: str,
) -> list[dict[str, Any]]:
    """Generate balanced schema-v5 records for every workflow."""

    if (
        isinstance(samples_per_fault, bool)
        or not isinstance(samples_per_fault, int)
        or samples_per_fault < 1
    ):
        raise ValueError("samples_per_fault must be a positive integer")
    rows: list[dict[str, Any]] = []
    for variation_index in range(start_variation, start_variation + samples_per_fault):
        for fault_name in WORKFLOWS:
            rows.append(
                build_v5_sample(
                    fault_name,
                    base_seed=seed,
                    variation_index=variation_index,
                    split=split,
                )
            )
    return rows


__all__ = [
    "HARD_CURRICULUM_VERSION",
    "HARD_SCENARIO_PROFILES",
    "HARD_SCENARIO_SCHEMA_VERSION",
    "HARD_SCENARIO_SCHEMA_VERSION_V3",
    "HARD_SYSTEM_PROMPT",
    "_OPAQUE_SIGNATURES_V4",
    "build_hard_grpo_sample",
    "build_v5_sample",
    "generate_hard_records",
    "generate_v5_records",
    "hard_expert_action",
    "hard_expert_workflow",
    "hard_observation",
    "hard_observation_messages",
    "hard_observation_messages_v3",
    "hard_observation_v3",
    "hard_observation_workflow",
    "hard_observation_workflow_messages",
    "hard_sample_seed",
    "prepare_hard_scenario",
    "prepare_v5_scenario",
    "profile_for_variation",
]
