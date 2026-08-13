"""Dependency-light helpers shared by CrashDiag's training commands."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crashdiag.faults.base import FaultModule
from crashdiag.faults.modules import FAULT_TYPES
from crashdiag.faults.workflows import WORKFLOWS


SYSTEM_PROMPT = """You diagnose a failing application from incomplete operational telemetry.
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
"""The parameter-free multi-action policy contract used for all v5 data and rollouts."""

_FAULT_CLASSES = {fault_type().name: fault_type for fault_type in FAULT_TYPES}
FAULT_NAMES = tuple(_FAULT_CLASSES)
WORKFLOW_NAMES = tuple(WORKFLOWS)
PRECISION_CHOICES = ("auto", "bf16", "fp16", "fp32")


def fault_for_name(name: str) -> FaultModule:
    """Return a fresh fault instance for a stable dataset fault name."""

    if not isinstance(name, str):
        raise TypeError("fault name must be a string")
    try:
        fault_type = _FAULT_CLASSES[name]
    except KeyError as exc:
        choices = ", ".join(FAULT_NAMES)
        raise ValueError(f"unknown fault {name!r}; expected one of: {choices}") from exc
    return fault_type()


def action_text(action: str, parameters: Mapping[str, Any] | None = None) -> str:
    """Serialize one policy action in the exact JSON form used as a target."""

    payload = {
        "action": action,
        "parameters": dict(parameters or {}),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def workflow_text(actions: Sequence[str]) -> str:
    """Serialize an ordered workflow target as ``{"actions": [...]}``."""

    payload = {
        "actions": [
            {"action": action, "parameters": {}}
            for action in actions
        ]
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def completion_text(value: Any) -> str:
    """Extract text from common TRL completion representations.

    A standard GRPO completion is a string.  For a conversational dataset TRL
    may instead provide one assistant message (or a list of content blocks).
    Unknown shapes intentionally become an empty string so reward code can
    treat them as invalid rather than executing an ambiguous value.
    """

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        return completion_text(content) if content is not value else ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text", item.get("content"))
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(text, Sequence) and not isinstance(
                    text, (str, bytes, bytearray)
                ):
                    parts.append(completion_text(text))
            elif isinstance(item, Sequence) and not isinstance(
                item, (str, bytes, bytearray)
            ):
                parts.append(completion_text(item))
        return "".join(parts)
    return ""


def observation_messages(observation: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the redacted operational prompt used by the v5 curriculum."""

    # Keep the prompt projection beside the legacy common helpers while avoiding
    # an import cycle: ``hard_scenarios`` imports ``fault_for_name`` above.
    from .hard_scenarios import hard_observation_workflow

    content = json.dumps(
        {"observation": hard_observation_workflow(observation)},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def resolve_precision(torch_module: Any, choice: str) -> tuple[bool, bool]:
    """Resolve a CLI precision choice into ``(bf16, fp16)`` trainer flags.

    ``torch_module`` is injected by callers so importing this module remains
    dependency-free.  Automatic mode uses BF16 on a capable CUDA device, FP16
    on other CUDA devices, and FP32 on CPU.
    """

    if choice not in PRECISION_CHOICES:
        raise ValueError(
            f"precision must be one of {', '.join(PRECISION_CHOICES)}, got {choice!r}"
        )
    if choice == "bf16":
        return True, False
    if choice == "fp16":
        return False, True
    if choice == "fp32":
        return False, False

    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    if not cuda_available:
        return False, False
    bf16_supported = getattr(cuda, "is_bf16_supported", None)
    if callable(bf16_supported) and bf16_supported():
        return True, False
    return False, True


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> int:
    """Atomically write JSONL rows and return the number written."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            for row in rows:
                line = json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                temporary.write(line)
                temporary.write("\n")
                count += 1
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return count


__all__ = [
    "FAULT_NAMES",
    "PRECISION_CHOICES",
    "SYSTEM_PROMPT",
    "WORKFLOW_NAMES",
    "action_text",
    "completion_text",
    "fault_for_name",
    "observation_messages",
    "resolve_precision",
    "workflow_text",
    "write_jsonl",
]
