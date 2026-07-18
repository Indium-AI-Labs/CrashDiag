"""Mechanical reward calculation for CrashDiag episodes.

The verifier deliberately knows nothing about language models.  Resolution is
defined exclusively by a fault module's ``is_resolved`` method, which inspects
the real (or mocked) sandbox state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for sparse, mechanically verified rewards.

    Shaping is disabled by default.  When explicitly enabled, an unresolved
    episode can receive ``health_shaping_reward`` only when the sandbox exposes
    a callable ``health_check`` that returns a mechanically healthy result.
    This fallback is still a programmatic system-state check; it never invokes
    an LLM or applies a rubric to the trajectory.
    """

    resolved_reward: float = 1.0
    unresolved_reward: float = 0.0
    enable_shaping: bool = False
    health_shaping_reward: float = 0.1

    def __post_init__(self) -> None:
        for name in ("resolved_reward", "unresolved_reward", "health_shaping_reward"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not isinstance(self.enable_shaping, bool):
            raise ValueError("enable_shaping must be a boolean")

    @property
    def shaping_enabled(self) -> bool:
        """Readable alias for callers that inspect whether shaping is active."""

        return self.enable_shaping


class CrashDiagVerifier:
    """Turn a fault module's mechanical resolution check into an RL reward."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    @staticmethod
    def is_resolved(fault: Any, instance: Any) -> bool:
        """Return resolution based solely on ``fault.is_resolved(instance)``."""

        check = getattr(fault, "is_resolved", None)
        if not callable(check):
            raise TypeError("fault must provide a callable is_resolved(instance)")
        result = check(instance)
        if not isinstance(result, bool):
            raise TypeError("fault.is_resolved(instance) must return a boolean")
        return result

    def verify(
        self,
        fault: Any,
        instance: Any,
        trajectory: Any | None = None,
    ) -> float:
        """Return the reward for current sandbox state.

        ``trajectory`` is accepted so the verifier fits generic orchestration
        interfaces, but is intentionally ignored: prose, model output, and
        action explanations cannot determine success.
        """

        del trajectory
        resolved = self.is_resolved(fault, instance)
        return self.reward_for_resolution(resolved, instance)

    def reward_for_resolution(self, resolved: bool, instance: Any) -> float:
        """Reward one already-computed mechanical resolution result.

        The orchestrator uses this method so the trajectory's terminal
        ``resolved`` flag and reward come from the same state check. Callers
        must pass a boolean obtained from ``fault.is_resolved(instance)``.
        """

        if not isinstance(resolved, bool):
            raise TypeError("resolved must be a boolean mechanical check result")
        if resolved:
            return float(self.config.resolved_reward)

        reward = float(self.config.unresolved_reward)
        if not self.config.enable_shaping:
            return reward

        health_check = getattr(instance, "health_check", None)
        if not callable(health_check):
            return reward
        try:
            health_result = health_check()
        except Exception:
            return reward
        if not self._is_healthy(health_result):
            return reward

        shaped = reward + float(self.config.health_shaping_reward)
        resolved_reward = float(self.config.resolved_reward)

        # Keep partial credit between the configured terminal rewards, even if
        # a caller uses a negative or unusually large shaping value.
        lower = min(reward, resolved_reward)
        upper = max(reward, resolved_reward)
        return max(lower, min(shaped, upper))

    @staticmethod
    def _is_healthy(result: Any) -> bool:
        """Interpret common *programmatic* health-check result shapes."""

        if isinstance(result, bool):
            return result
        if isinstance(result, int):
            return 200 <= result < 400
        if isinstance(result, str):
            return result.strip().lower() in {"healthy", "ok", "running", "up"}
        if isinstance(result, Mapping):
            if "healthy" in result:
                return result["healthy"] is True
            if "status_code" in result:
                try:
                    status_code = int(result["status_code"])
                except (TypeError, ValueError):
                    return False
                return 200 <= status_code < 400
            if "status" in result:
                status = str(result["status"]).strip().lower()
                return status in {"healthy", "ok", "running", "up"}
            return False
        return False

    def reward(
        self,
        fault: Any,
        instance: Any,
        trajectory: Any | None = None,
    ) -> float:
        """Alias for :meth:`verify`, useful to generic RL callers."""

        return self.verify(fault, instance, trajectory)

    def score(
        self,
        fault: Any,
        instance: Any,
        trajectory: Any | None = None,
    ) -> float:
        """Alias for :meth:`verify`."""

        return self.verify(fault, instance, trajectory)


__all__ = ["CrashDiagVerifier", "RewardConfig"]
