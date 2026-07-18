"""Dependency-light helpers shared by CrashDiag's training commands."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crashdiag.agents import DEFAULT_SYSTEM_PROMPT
from crashdiag.faults.base import FaultModule
from crashdiag.faults.modules import FAULT_TYPES


SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
"""The policy contract used for both offline data and online rollouts."""

_FAULT_CLASSES = {fault_type().name: fault_type for fault_type in FAULT_TYPES}
FAULT_NAMES = tuple(_FAULT_CLASSES)
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
    """Build the conversational prompt consumed by BlueAgent-compatible models."""

    content = json.dumps(
        {"observation": observation},
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
    "action_text",
    "completion_text",
    "fault_for_name",
    "observation_messages",
    "resolve_precision",
    "write_jsonl",
]
