"""Sandbox interface and an in-memory implementation for local episodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, ClassVar


class SandboxBackend(ABC):
    """System-state and action contract consumed by CrashDiag.

    Observations and action results must be JSON-serializable.  A production
    backend is responsible for reading actual runtime state after every action;
    accepting an API request is not, by itself, proof that a fix succeeded.
    """

    ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "restart_app",
            "rollback_env_var",
            "fix_dependency",
            "clear_disk",
            "fix_port_config",
            "wait_and_observe",
        }
    )

    @abstractmethod
    def observe(self) -> dict[str, Any]:
        """Return a fresh, JSON-serializable snapshot of system state."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Return a mechanical health result, including ``healthy``."""

    @abstractmethod
    def restart_app(self) -> dict[str, Any]:
        """Restart the application and return observed action data."""

    @abstractmethod
    def rollback_env_var(self, name: str | None = None) -> dict[str, Any]:
        """Restore an environment variable to its previous known-good value."""

    @abstractmethod
    def fix_dependency(
        self,
        name: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Restore one or all required dependency versions."""

    @abstractmethod
    def clear_disk(self, target_percent: float = 40.0) -> dict[str, Any]:
        """Reduce disk utilization and return observed action data."""

    @abstractmethod
    def fix_port_config(self, target_port: int | None = None) -> dict[str, Any]:
        """Point the proxy at the intended application port."""

    @abstractmethod
    def wait_and_observe(self) -> dict[str, Any]:
        """Take no corrective action and return a fresh observation."""

    @abstractmethod
    def trigger_oom_kill(self) -> None:
        """Inject an out-of-memory process termination."""

    @abstractmethod
    def set_env_var(self, name: str, value: str) -> None:
        """Set an environment variable for fault injection."""

    @abstractmethod
    def set_dependency_version(self, name: str, version: str) -> None:
        """Set an installed dependency version for fault injection."""

    @abstractmethod
    def set_disk_usage(self, percent: float) -> None:
        """Set disk utilization for fault injection."""

    @abstractmethod
    def set_proxy_target_port(self, port: int) -> None:
        """Set the reverse proxy's upstream port for fault injection."""

    @abstractmethod
    def set_expected_env_var(self, name: str, value: str) -> None:
        """Configure one known-good environment value for scenario setup."""

    @abstractmethod
    def set_required_dependency_version(self, name: str, version: str) -> None:
        """Configure one known-good dependency version for scenario setup."""

    @abstractmethod
    def set_app_port(self, port: int) -> None:
        """Configure the app and healthy proxy port for scenario setup."""

    @abstractmethod
    def set_disk_health_threshold(self, percent: float) -> None:
        """Configure the disk-health boundary for scenario setup."""

    def execute_action(
        self,
        action: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and dispatch one action from CrashDiag's fixed action space."""

        if action not in self.ACTIONS:
            raise ValueError(f"unsupported sandbox action: {action!r}")
        if parameters is None:
            normalized: dict[str, Any] = {}
        elif isinstance(parameters, Mapping):
            normalized = dict(parameters)
        else:
            raise TypeError("action parameters must be a mapping or None")

        handler = getattr(self, action)
        try:
            return handler(**normalized)
        except TypeError as exc:
            raise ValueError(f"invalid parameters for {action!r}: {normalized!r}") from exc


class MockSandbox(SandboxBackend):
    """A deterministic application with realistic, coupled failure state.

    Health is recomputed from process state, environment, dependency versions,
    disk utilization, and proxy routing on every observation.  Recovery actions
    mutate those underlying values; no independent ``resolved`` flag exists.
    """

    DEFAULT_ENV: ClassVar[dict[str, str]] = {
        "APP_ENV": "production",
        "DATABASE_URL": "postgresql://app:secret@database:5432/app",
    }
    DEFAULT_DEPENDENCIES: ClassVar[dict[str, str]] = {
        "psycopg": "3.1.18",
        "web-framework": "1.4.2",
    }
    DEFAULT_DISK_PERCENT: ClassVar[float] = 35.0
    DISK_HEALTH_THRESHOLD: ClassVar[float] = 90.0
    DEFAULT_APP_PORT: ClassVar[int] = 8080

    def __init__(self) -> None:
        self.expected_env: dict[str, str] = dict(self.DEFAULT_ENV)
        self.env_vars: dict[str, str] = dict(self.DEFAULT_ENV)
        self.required_dependencies: dict[str, str] = dict(self.DEFAULT_DEPENDENCIES)
        self.dependencies: dict[str, str] = dict(self.DEFAULT_DEPENDENCIES)
        self.disk_usage_percent = self.DEFAULT_DISK_PERCENT
        self.disk_health_threshold = self.DISK_HEALTH_THRESHOLD
        self.app_port = self.DEFAULT_APP_PORT
        self.proxy_target_port = self.DEFAULT_APP_PORT
        self.process_running = True
        self.last_exit_reason: str | None = None
        self.restart_count = 0
        self.oom_kill_count = 0
        self.clock_ticks = 0
        self.action_history: list[dict[str, Any]] = []
        self.logs: list[str] = ["application started and passed health check"]
        self._env_history: dict[str, list[str | None]] = {}
        self._env_change_order: list[str] = []

    @property
    def healthy(self) -> bool:
        """Expose the current mechanical health boolean."""

        return bool(self.health_check()["healthy"])

    @property
    def health_state(self) -> str:
        """Expose ``healthy`` or ``unhealthy`` for simple integrations."""

        return str(self.health_check()["status"])

    def _checks(self) -> dict[str, bool]:
        environment_ok = self.env_vars.get("APP_ENV") == self.expected_env["APP_ENV"]
        database_ok = (
            self.env_vars.get("DATABASE_URL") == self.expected_env["DATABASE_URL"]
        )
        dependencies_ok = all(
            self.dependencies.get(name) == required
            for name, required in self.required_dependencies.items()
        )
        disk_ok = self.disk_usage_percent < self.disk_health_threshold
        port_ok = self.proxy_target_port == self.app_port
        return {
            "process": self.process_running,
            "environment": environment_ok,
            "database": database_ok,
            "dependencies": dependencies_ok,
            "disk": disk_ok,
            "port_proxy": port_ok,
        }

    def health_check(self) -> dict[str, Any]:
        """Return an HTTP-like health result derived from current state."""

        checks = self._checks()
        healthy = all(checks.values())
        return {
            "healthy": healthy,
            "status": "healthy" if healthy else "unhealthy",
            "status_code": 200 if healthy else 503,
            "checks": checks,
            "failures": [name for name, passed in checks.items() if not passed],
        }

    def observe(self) -> dict[str, Any]:
        """Return a detached snapshot suitable for a policy and trajectory."""

        health = self.health_check()
        return {
            "health": health,
            "healthy": health["healthy"],
            "health_state": health["status"],
            "http_status": health["status_code"],
            "process": {
                "running": self.process_running,
                "last_exit_reason": self.last_exit_reason,
                "restart_count": self.restart_count,
                "oom_kill_count": self.oom_kill_count,
            },
            "environment": {
                "variables": dict(self.env_vars),
                "expected": dict(self.expected_env),
            },
            "env_vars": dict(self.env_vars),
            "dependencies": {
                "installed": dict(self.dependencies),
                "required": dict(self.required_dependencies),
            },
            "disk": {
                "used_percent": self.disk_usage_percent,
                "healthy_below_percent": self.disk_health_threshold,
            },
            "disk_usage_percent": self.disk_usage_percent,
            "network": {
                "app_port": self.app_port,
                "proxy_target_port": self.proxy_target_port,
            },
            "clock_ticks": self.clock_ticks,
            "recent_logs": list(self.logs[-10:]),
        }

    def _record_action(self, action: str, changed: bool) -> dict[str, Any]:
        record = {"action": action, "changed": bool(changed), "tick": self.clock_ticks}
        self.action_history.append(record)
        return {**record, "observation": self.observe()}

    def restart_app(self) -> dict[str, Any]:
        was_running = self.process_running
        previous_reason = self.last_exit_reason
        self.process_running = True
        self.last_exit_reason = None
        self.restart_count += 1
        self.clock_ticks += 1
        self.logs.append("application process restarted")
        return self._record_action(
            "restart_app", (not was_running) or previous_reason is not None
        )

    def rollback_env_var(self, name: str | None = None) -> dict[str, Any]:
        if name is not None and not isinstance(name, str):
            raise TypeError("environment variable name must be a string or None")

        if name is None:
            while self._env_change_order:
                candidate = self._env_change_order.pop()
                if self._env_history.get(candidate):
                    name = candidate
                    break
            if name is None:
                mismatched = [
                    key
                    for key, value in self.expected_env.items()
                    if self.env_vars.get(key) != value
                ]
                name = mismatched[0] if mismatched else None
        else:
            for index in range(len(self._env_change_order) - 1, -1, -1):
                if self._env_change_order[index] == name:
                    del self._env_change_order[index]
                    break

        changed = False
        if name is not None:
            history = self._env_history.get(name, [])
            previous = history.pop() if history else self.expected_env.get(name)
            current = self.env_vars.get(name)
            if previous is None:
                self.env_vars.pop(name, None)
            else:
                self.env_vars[name] = previous
            changed = current != previous
            self.logs.append(f"environment variable {name} rolled back")

        self.clock_ticks += 1
        return self._record_action("rollback_env_var", changed)

    def fix_dependency(
        self,
        name: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        if name is not None and not isinstance(name, str):
            raise TypeError("dependency name must be a string or None")
        if version is not None and not isinstance(version, str):
            raise TypeError("dependency version must be a string or None")

        changed = False
        if name is None:
            for dependency, required in self.required_dependencies.items():
                if self.dependencies.get(dependency) != required:
                    self.dependencies[dependency] = required
                    changed = True
        else:
            desired = version if version is not None else self.required_dependencies.get(name)
            if desired is None:
                raise ValueError(f"no required version is known for {name!r}")
            if self.dependencies.get(name) != desired:
                self.dependencies[name] = desired
                changed = True

        self.clock_ticks += 1
        self.logs.append("dependency set restored" if changed else "dependency set unchanged")
        return self._record_action("fix_dependency", changed)

    def clear_disk(self, target_percent: float = 40.0) -> dict[str, Any]:
        if isinstance(target_percent, bool) or not isinstance(target_percent, (int, float)):
            raise TypeError("target_percent must be numeric")
        target = float(target_percent)
        if not 0.0 <= target <= 100.0:
            raise ValueError("target_percent must be between 0 and 100")
        previous = self.disk_usage_percent
        self.disk_usage_percent = min(previous, target)
        self.clock_ticks += 1
        self.logs.append(
            f"disk usage reduced from {previous:.1f}% to {self.disk_usage_percent:.1f}%"
        )
        return self._record_action("clear_disk", self.disk_usage_percent != previous)

    def fix_port_config(self, target_port: int | None = None) -> dict[str, Any]:
        if target_port is None:
            target_port = self.app_port
        self._validate_port(target_port)
        previous = self.proxy_target_port
        self.proxy_target_port = target_port
        self.clock_ticks += 1
        self.logs.append(f"proxy upstream changed to port {target_port}")
        return self._record_action("fix_port_config", previous != target_port)

    def wait_and_observe(self) -> dict[str, Any]:
        self.clock_ticks += 1
        return self._record_action("wait_and_observe", False)

    def trigger_oom_kill(self) -> None:
        self.process_running = False
        self.last_exit_reason = "OOMKilled"
        self.oom_kill_count += 1
        self.logs.append("kernel terminated application process: OOMKilled")

    def set_env_var(self, name: str, value: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("environment variable name must be non-empty")
        if not isinstance(value, str):
            raise TypeError("environment variable value must be a string")
        self._env_history.setdefault(name, []).append(self.env_vars.get(name))
        self._env_change_order.append(name)
        self.env_vars[name] = value
        self.logs.append(f"environment variable {name} changed")

    def set_dependency_version(self, name: str, version: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("dependency name must be non-empty")
        if not isinstance(version, str) or not version:
            raise ValueError("dependency version must be non-empty")
        self.dependencies[name] = version
        self.logs.append(f"installed {name} version changed to {version}")

    def set_disk_usage(self, percent: float) -> None:
        if isinstance(percent, bool) or not isinstance(percent, (int, float)):
            raise TypeError("disk percentage must be numeric")
        value = float(percent)
        if not 0.0 <= value <= 100.0:
            raise ValueError("disk percentage must be between 0 and 100")
        self.disk_usage_percent = value
        self.logs.append(f"disk usage reached {value:.1f}%")

    def set_proxy_target_port(self, port: int) -> None:
        self._validate_port(port)
        self.proxy_target_port = port
        self.logs.append(f"proxy upstream changed to port {port}")

    def set_expected_env_var(self, name: str, value: str) -> None:
        """Set a clean environment baseline without adding diagnostic history."""

        if not isinstance(name, str) or not name:
            raise ValueError("environment variable name must be non-empty")
        if not isinstance(value, str):
            raise TypeError("environment variable value must be a string")
        self.expected_env[name] = value
        self.env_vars[name] = value
        self._env_history.pop(name, None)
        self._env_change_order = [item for item in self._env_change_order if item != name]

    def set_required_dependency_version(self, name: str, version: str) -> None:
        """Set installed and required dependency state to the same clean baseline."""

        if not isinstance(name, str) or not name:
            raise ValueError("dependency name must be non-empty")
        if not isinstance(version, str) or not version:
            raise ValueError("dependency version must be non-empty")
        self.required_dependencies[name] = version
        self.dependencies[name] = version

    def set_app_port(self, port: int) -> None:
        """Set the app and proxy to a healthy listening-port baseline."""

        self._validate_port(port)
        self.app_port = port
        self.proxy_target_port = port

    def set_disk_health_threshold(self, percent: float) -> None:
        """Set a finite health boundary while retaining a healthy baseline."""

        if isinstance(percent, bool) or not isinstance(percent, (int, float)):
            raise TypeError("disk health threshold must be numeric")
        threshold = float(percent)
        if not 1.0 <= threshold <= 100.0:
            raise ValueError("disk health threshold must be between 1 and 100")
        self.disk_health_threshold = threshold
        if self.disk_usage_percent >= threshold:
            self.disk_usage_percent = max(0.0, min(40.0, threshold - 5.0))

    @staticmethod
    def _validate_port(port: int) -> None:
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port must be an integer between 1 and 65535")

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of the observation for test/debug callers."""

        return deepcopy(self.observe())


__all__ = ["MockSandbox", "SandboxBackend"]
