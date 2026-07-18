"""Episode orchestration shared by CrashDiag-style environments."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .verifier import CrashDiagVerifier


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any, _seen: set[int] | None = None) -> Any:
    """Convert common Python values into strict JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value, _seen)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        try:
            return str(value)
        except Exception:
            return f"<unserializable {type(value).__name__}>"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    seen = _seen if _seen is not None else set()
    track_identity = isinstance(value, (Mapping, list, tuple, set)) or is_dataclass(value)
    if track_identity:
        identity = id(value)
        if identity in seen:
            return "<recursive reference>"
        seen.add(identity)

    try:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item, seen) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item, seen) for item in value]
        if is_dataclass(value) and not isinstance(value, type):
            return {
                item.name: _json_safe(getattr(value, item.name), seen)
                for item in fields(value)
            }
        try:
            return str(value)
        except Exception:
            return f"<unserializable {type(value).__name__}>"
    finally:
        if track_identity:
            seen.discard(id(value))


@dataclass
class Trajectory:
    """A complete, JSON-serializable record of one diagnostic episode."""

    fault_name: str
    difficulty: str = "unknown"
    steps: list[dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    reward: float = 0.0
    initial_observation: Any = field(default_factory=dict)
    final_observation: Any = None
    injection_result: Any = None
    error: str | None = None
    started_at: str = field(default_factory=_utc_now)
    ended_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_log(self) -> list[dict[str, Any]]:
        """Alias emphasizing that ``steps`` is the full episode step log."""

        return self.steps

    def add_step(self, step: Mapping[str, Any]) -> None:
        """Append a JSON-safe snapshot of a step."""

        safe_step = _json_safe(step)
        if not isinstance(safe_step, dict):  # Defensive; Mapping normally guarantees this.
            raise TypeError("a trajectory step must be a mapping")
        self.steps.append(safe_step)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary containing only JSON-compatible data."""

        result = _json_safe(
            {
                "fault_name": self.fault_name,
                "difficulty": self.difficulty,
                "steps": self.steps,
                "resolved": self.resolved,
                "reward": self.reward,
                "initial_observation": self.initial_observation,
                "final_observation": self.final_observation,
                "injection_result": self.injection_result,
                "error": self.error,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "metadata": self.metadata,
            }
        )
        assert isinstance(result, dict)
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize this trajectory as strict JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


class Orchestrator:
    """Inject faults, collect agent actions, apply them, and verify state.

    The canonical integration contract is:

    * ``agent.choose_action(observation, history=None)`` returns a mapping with
      ``action`` and optional ``parameters`` keys.
    * ``sandbox.execute_action(action, parameters)`` applies that action.

    Small compatibility fallbacks support ``agent.act(...)``, callable agents,
    and sandboxes exposing individual action methods.
    """

    DEFAULT_ACTION = "wait_and_observe"

    def __init__(
        self,
        sandbox: Any,
        agent: Any,
        verifier: Any | None = None,
        max_steps: int = 10,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        self.sandbox = sandbox
        self.agent = agent
        self.verifier = verifier or CrashDiagVerifier()
        self.max_steps = max_steps

    def run_episode(
        self,
        fault: Any,
        *,
        max_steps: int | None = None,
        instance: Any | None = None,
    ) -> Trajectory:
        """Run one fault-injection episode and return its complete trajectory."""

        step_limit = self.max_steps if max_steps is None else max_steps
        if isinstance(step_limit, bool) or not isinstance(step_limit, int) or step_limit < 0:
            raise ValueError("max_steps must be a non-negative integer")

        target = self.sandbox if instance is None else instance
        fault_name = str(getattr(fault, "name", fault.__class__.__name__))
        difficulty = str(getattr(fault, "difficulty", "unknown"))
        trajectory = Trajectory(fault_name=fault_name, difficulty=difficulty)

        inject = getattr(fault, "inject", None)
        if not callable(inject):
            trajectory.error = "fault must provide a callable inject(instance)"
            return self._finish(trajectory, fault, target, verification_allowed=False)

        try:
            trajectory.injection_result = _json_safe(inject(target))
        except Exception as exc:  # Preserve a failed episode instead of losing its log.
            trajectory.error = self._format_error("fault injection failed", exc)
            return self._finish(trajectory, fault, target, verification_allowed=False)

        observation, observation_error = self._observe(target)
        trajectory.initial_observation = _json_safe(observation)
        if observation_error:
            trajectory.error = observation_error

        resolved, resolution_error = self._resolution(fault, target)
        if resolution_error:
            trajectory.error = self._join_errors(trajectory.error, resolution_error)

        for step_index in range(step_limit):
            if resolved:
                break

            agent_error: str | None = None
            raw_action: Any = None
            try:
                raw_action = self._ask_agent(observation, trajectory.steps)
                action, parameters, parse_error = self._normalise_action(raw_action)
                agent_error = parse_error
            except Exception as exc:
                action, parameters = self.DEFAULT_ACTION, {}
                agent_error = self._format_error("agent failed", exc)

            action_result: Any = None
            action_error: str | None = None
            try:
                action_result = self._execute_action(target, action, parameters)
            except Exception as exc:
                action_error = self._format_error("action failed", exc)

            observation_after, after_error = self._observe(target)
            resolved, resolution_error = self._resolution(fault, target)

            step: dict[str, Any] = {
                "step": step_index,
                "timestamp": _utc_now(),
                "observation": observation,
                "raw_agent_output": raw_action,
                "action": action,
                "parameters": parameters,
                "action_result": action_result,
                "observation_after": observation_after,
                "resolved": resolved,
            }
            if agent_error:
                step["agent_error"] = agent_error
            if action_error:
                step["action_error"] = action_error
            if after_error:
                step["observation_error"] = after_error
            if resolution_error:
                step["resolution_error"] = resolution_error
            trajectory.add_step(step)

            observation = observation_after

        trajectory.final_observation = _json_safe(observation)
        return self._finish(
            trajectory,
            fault,
            target,
            known_resolved=resolved,
            final_observation_known=True,
        )

    def run_batch(
        self,
        faults: Iterable[Any],
        *,
        episodes_per_fault: int = 1,
        max_steps: int | None = None,
        instance: Any | None = None,
    ) -> list[Trajectory]:
        """Run each supplied fault one or more times, preserving input order.

        The selected sandbox instance is reused. Backends that require strict
        episode isolation should provide a fresh disposable instance per batch
        or reset it before invoking this method.
        """

        if (
            isinstance(episodes_per_fault, bool)
            or not isinstance(episodes_per_fault, int)
            or episodes_per_fault < 1
        ):
            raise ValueError("episodes_per_fault must be a positive integer")

        trajectories: list[Trajectory] = []
        for fault_index, fault in enumerate(faults):
            for repetition in range(episodes_per_fault):
                trajectory = self.run_episode(
                    fault,
                    max_steps=max_steps,
                    instance=instance,
                )
                trajectory.metadata.update(
                    {"fault_index": fault_index, "episode_index": repetition}
                )
                trajectories.append(trajectory)
        return trajectories

    def _finish(
        self,
        trajectory: Trajectory,
        fault: Any,
        instance: Any,
        *,
        known_resolved: bool | None = None,
        verification_allowed: bool = True,
        final_observation_known: bool = False,
    ) -> Trajectory:
        if not final_observation_known:
            observation, observation_error = self._observe(instance)
            trajectory.final_observation = _json_safe(observation)
            if observation_error:
                trajectory.error = self._join_errors(trajectory.error, observation_error)

        if not verification_allowed:
            known_resolved = False
        elif known_resolved is None:
            known_resolved, resolution_error = self._resolution(fault, instance)
            if resolution_error:
                trajectory.error = self._join_errors(trajectory.error, resolution_error)
        trajectory.resolved = bool(known_resolved)

        if verification_allowed:
            try:
                trajectory.reward = float(
                    self._calculate_reward(
                        fault,
                        instance,
                        trajectory,
                        resolved=trajectory.resolved,
                    )
                )
            except Exception as exc:
                trajectory.reward = self._unresolved_reward()
                trajectory.error = self._join_errors(
                    trajectory.error,
                    self._format_error("reward verification failed", exc),
                )
        else:
            trajectory.reward = self._unresolved_reward()
        trajectory.ended_at = _utc_now()
        return trajectory

    def _calculate_reward(
        self,
        fault: Any,
        instance: Any,
        trajectory: Trajectory,
        *,
        resolved: bool,
    ) -> float:
        state_reward = getattr(self.verifier, "reward_for_resolution", None)
        if callable(state_reward):
            return float(state_reward(resolved, instance))
        reward_method = getattr(self.verifier, "reward", None)
        if callable(reward_method):
            return float(reward_method(fault, instance, trajectory))
        verify_method = getattr(self.verifier, "verify", None)
        if callable(verify_method):
            return float(verify_method(fault, instance, trajectory))
        if callable(self.verifier):
            return float(self.verifier(fault, instance, trajectory))
        raise TypeError("verifier must provide reward(...) or verify(...), or be callable")

    def _unresolved_reward(self) -> float:
        config = getattr(self.verifier, "config", None)
        try:
            return float(getattr(config, "unresolved_reward", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _resolution(fault: Any, instance: Any) -> tuple[bool, str | None]:
        check = getattr(fault, "is_resolved", None)
        if not callable(check):
            return False, "fault must provide a callable is_resolved(instance)"
        try:
            result = check(instance)
            if not isinstance(result, bool):
                raise TypeError("fault.is_resolved(instance) must return a boolean")
            return result, None
        except Exception as exc:
            return False, Orchestrator._format_error("resolution check failed", exc)

    @staticmethod
    def _observe(instance: Any) -> tuple[Any, str | None]:
        for method_name in ("observe", "get_observation"):
            method = getattr(instance, method_name, None)
            if callable(method):
                try:
                    return method(), None
                except Exception as exc:
                    return {}, Orchestrator._format_error("observation failed", exc)
        return {}, "sandbox must provide observe()"

    def _ask_agent(self, observation: Any, history: list[dict[str, Any]]) -> Any:
        method = getattr(self.agent, "choose_action", None)
        if not callable(method):
            method = getattr(self.agent, "act", None)
        if not callable(method) and callable(self.agent):
            method = self.agent
        if not callable(method):
            raise TypeError("agent must provide choose_action(...) or act(...), or be callable")

        return self._call_compatible(
            method,
            (
                ((observation,), {"history": history}),
                ((observation, history), {}),
                ((observation,), {}),
            ),
        )

    @classmethod
    def _normalise_action(cls, raw_action: Any) -> tuple[str, dict[str, Any], str | None]:
        if isinstance(raw_action, str) and raw_action.strip():
            return raw_action.strip(), {}, None
        if not isinstance(raw_action, Mapping):
            return cls.DEFAULT_ACTION, {}, "invalid agent output; defaulted to wait_and_observe"

        action = raw_action.get("action")
        parameters = raw_action.get("parameters", {})
        if not isinstance(action, str) or not action.strip() or not isinstance(parameters, Mapping):
            return cls.DEFAULT_ACTION, {}, "invalid agent output; defaulted to wait_and_observe"
        return action.strip(), dict(parameters), None

    def _execute_action(
        self,
        instance: Any,
        action: str,
        parameters: Mapping[str, Any],
    ) -> Any:
        execute = getattr(instance, "execute_action", None)
        if callable(execute):
            return self._call_compatible(
                execute,
                (
                    ((action, parameters), {}),
                    ((action,), {"parameters": parameters}),
                    ((action,), {}),
                ),
            )

        if not action.isidentifier() or action.startswith("_"):
            raise ValueError(f"invalid action name: {action!r}")
        method = getattr(instance, action, None)
        if not callable(method):
            raise ValueError(f"sandbox does not support action {action!r}")
        return self._call_compatible(
            method,
            (
                ((), dict(parameters)),
                ((parameters,), {}),
                ((), {}),
            ),
        )

    @staticmethod
    def _call_compatible(
        function: Callable[..., Any],
        candidates: Iterable[tuple[tuple[Any, ...], dict[str, Any]]],
    ) -> Any:
        """Call the first candidate shape accepted by ``function``'s signature."""

        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            for args, kwargs in candidates:
                try:
                    signature.bind(*args, **kwargs)
                except TypeError:
                    continue
                return function(*args, **kwargs)
            raise TypeError("callable does not accept any supported argument shape")

        # Some extension callables do not expose a signature.  Try in order;
        # unlike the inspected path, a TypeError inside such a callable is
        # indistinguishable from an argument mismatch.
        last_error: TypeError | None = None
        for args, kwargs in candidates:
            try:
                return function(*args, **kwargs)
            except TypeError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    @staticmethod
    def _format_error(context: str, error: Exception) -> str:
        return f"{context}: {error.__class__.__name__}: {error}"

    @staticmethod
    def _join_errors(existing: str | None, new: str | None) -> str | None:
        if not new:
            return existing
        return f"{existing}; {new}" if existing else new


__all__ = ["Orchestrator", "Trajectory"]
