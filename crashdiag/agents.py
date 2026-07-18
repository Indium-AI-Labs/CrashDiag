"""Blue-team agents for CrashDiag.

The :class:`BlueAgent` talks to an OpenAI-compatible chat-completions API, as
provided by vLLM.  Importing this module never performs network I/O; a request
is made only when :meth:`BlueAgent.choose_action` is called.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib import request


ACTION_SPACE = (
    "restart_app",
    "rollback_env_var",
    "fix_dependency",
    "clear_disk",
    "fix_port_config",
    "wait_and_observe",
)
"""The complete set of actions a blue agent may emit."""

_ALLOWED_ACTIONS = frozenset(ACTION_SPACE)
DEFAULT_SYSTEM_PROMPT = """You diagnose a failing application from system observations.
Choose exactly one action from this list:
- restart_app
- rollback_env_var
- fix_dependency
- clear_disk
- fix_port_config
- wait_and_observe

Reply with one JSON object only, using this schema:
{"action": "<action name>", "parameters": {}}
The parameters value must be a JSON object. Do not use markdown or prose.
"""


def _safe_wait_action() -> dict[str, Any]:
    """Return a fresh safe fallback so callers cannot mutate global state."""

    return {"action": "wait_and_observe", "parameters": {}}


def _validated_action(value: Any) -> dict[str, Any] | None:
    """Validate a decoded candidate against the deliberately small schema."""

    if not isinstance(value, Mapping):
        return None

    action = value.get("action")
    if not isinstance(action, str) or action not in _ALLOWED_ACTIONS:
        return None

    parameters = value.get("parameters", {})
    if not isinstance(parameters, Mapping):
        return None

    # JSON decoding normally produces dicts, but accepting Mapping makes the
    # parser useful in tests too.  This conversion also guarantees that the
    # returned trajectory entry is JSON-serializable.
    try:
        normalized_parameters = dict(parameters)
        json.dumps(normalized_parameters, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return None

    result: dict[str, Any] = {
        "action": action,
        "parameters": normalized_parameters,
    }
    reason = value.get("reason")
    if isinstance(reason, str):
        result["reason"] = reason
    return result


def parse_action(content: Any) -> dict[str, Any]:
    """Parse a model response, returning a safe wait action on every failure.

    Normal JSON is preferred.  For robustness with chat models, one complete
    Markdown JSON fence is also accepted.  Prose-wrapped or partially malformed
    objects are rejected rather than searching for an executable substring.
    Only a top-level object with an action from :data:`ACTION_SPACE` is accepted.
    """

    if isinstance(content, Mapping):
        return _validated_action(content) or _safe_wait_action()
    if not isinstance(content, str) or not content.strip():
        return _safe_wait_action()

    candidates: list[Any] = []
    stripped = content.strip()

    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
        pass
    else:
        # If the entire response is valid JSON, it must itself match the action
        # schema.  Do not accidentally approve an action nested inside an array
        # or wrapper object by scanning into a schema-invalid JSON document.
        return _validated_action(decoded) or _safe_wait_action()

    # Some otherwise capable models still wrap their entire JSON in a code fence.
    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_match is not None:
        try:
            candidates.append(json.loads(fenced_match.group(1)))
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
            pass

    for candidate in candidates:
        validated = _validated_action(candidate)
        if validated is not None:
            return validated
    return _safe_wait_action()


class BlueAgent:
    """A small vLLM/OpenAI-compatible chat-completions client.

    Parameters are intentionally limited to inference concerns.  ``base_url``
    may be either an API root such as ``http://localhost:8000/v1`` or a full
    ``.../chat/completions`` endpoint.  Transport, response, JSON, and action
    validation errors all result in the safe ``wait_and_observe`` action so a
    malformed model response cannot execute an unintended operation.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("api_key must be a string or None")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature < 0
        ):
            raise ValueError("temperature must be a non-negative finite number")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string")

        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    @property
    def chat_completions_url(self) -> str:
        """Return the configured OpenAI-compatible request URL."""

        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def parse_action(content: Any) -> dict[str, Any]:
        """Expose the defensive parser for callers and offline tests."""

        return parse_action(content)

    def _messages(
        self,
        observation: Mapping[str, Any] | Any,
        history: Sequence[Mapping[str, Any]] | None,
    ) -> list[dict[str, str]]:
        context: dict[str, Any] = {"observation": observation}
        if history:
            context["history"] = list(history)
        user_content = json.dumps(context, sort_keys=True, default=str)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _request_completion(self, messages: list[dict[str, str]]) -> Any:
        """Perform one chat-completions request and return message content."""

        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            allow_nan=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        http_request = request.Request(
            self.chat_completions_url,
            data=body,
            headers=headers,
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if not isinstance(payload, Mapping):
            raise ValueError("chat-completions response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("chat-completions response has no choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ValueError("chat-completions choice must be a JSON object")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("chat-completions choice has no message")
        content = message.get("content")

        # OpenAI-compatible servers normally return a string.  Supporting text
        # content blocks costs little and prevents an otherwise harmless schema
        # variation from crashing an episode.
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
            content = "".join(text_parts)
        return content

    def choose_action(
        self,
        observation: Mapping[str, Any] | Any,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Choose one validated action, falling back safely on any failure."""

        try:
            content = self._request_completion(self._messages(observation, history))
            return parse_action(content)
        except Exception:
            # An RL episode should record a conservative no-op for an unavailable
            # or malformed policy server, never execute an unvalidated action.
            return _safe_wait_action()

    def act(
        self,
        observation: Mapping[str, Any] | Any,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compatibility alias for orchestrators that call ``agent.act``."""

        return self.choose_action(observation, history)


__all__ = ["ACTION_SPACE", "BlueAgent", "parse_action"]
