"""Built-in CrashDiag fault modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import FaultModule


def _observe(instance: Any) -> Mapping[str, Any]:
    observe = getattr(instance, "observe", None)
    if not callable(observe):
        raise TypeError("fault verification requires instance.observe()")
    observation = observe()
    if not isinstance(observation, Mapping):
        raise TypeError("instance.observe() must return a mapping")
    return observation


def _mechanically_healthy(instance: Any) -> bool:
    """Interpret the backend's health result without any model judgment."""

    health_check = getattr(instance, "health_check", None)
    if not callable(health_check):
        return False
    result = health_check()
    if isinstance(result, bool):
        return result
    if isinstance(result, int) and not isinstance(result, bool):
        return 200 <= result < 400
    if not isinstance(result, Mapping):
        return False
    if "healthy" in result:
        return result["healthy"] is True
    try:
        status_code = int(result["status_code"])
    except (KeyError, TypeError, ValueError):
        return False
    return 200 <= status_code < 400


def _environment(observation: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    environment = observation.get("environment", {})
    if isinstance(environment, Mapping):
        variables = (
            environment["variables"]
            if "variables" in environment
            else observation.get("env_vars", {})
        )
        expected = environment.get("expected", {})
    else:
        variables = observation.get("env_vars", {})
        expected = {}
    if not isinstance(variables, Mapping):
        variables = {}
    if not isinstance(expected, Mapping):
        expected = {}
    return variables, expected


class OOMKill(FaultModule):
    """Terminate the application process as if the kernel OOM killer acted."""

    name = "oom_kill"
    difficulty = "medium"

    def inject(self, instance: Any) -> None:
        instance.trigger_oom_kill()

    def is_resolved(self, instance: Any) -> bool:
        process = _observe(instance).get("process", {})
        return bool(
            isinstance(process, Mapping)
            and process.get("running") is True
            and _mechanically_healthy(instance)
        )


class BadEnvVar(FaultModule):
    """Replace the application's runtime mode with an invalid value."""

    name = "bad_env_var"
    difficulty = "easy"
    variable_name = "APP_ENV"
    expected_value = "production"
    bad_value = "invalid"

    def inject(self, instance: Any) -> None:
        instance.set_env_var(self.variable_name, self.bad_value)

    def is_resolved(self, instance: Any) -> bool:
        variables, expected = _environment(_observe(instance))
        wanted = expected.get(self.variable_name, self.expected_value)
        return (
            variables.get(self.variable_name) == wanted
            and _mechanically_healthy(instance)
        )


class BrokenDBConnection(FaultModule):
    """Point the app at an unreachable database host."""

    name = "broken_db_connection"
    difficulty = "medium"
    variable_name = "DATABASE_URL"
    expected_value = "postgresql://app:secret@database:5432/app"
    bad_value = "postgresql://app:secret@missing-database:5432/app"

    def inject(self, instance: Any) -> None:
        instance.set_env_var(self.variable_name, self.bad_value)

    def is_resolved(self, instance: Any) -> bool:
        variables, expected = _environment(_observe(instance))
        wanted = expected.get(self.variable_name, self.expected_value)
        return (
            variables.get(self.variable_name) == wanted
            and _mechanically_healthy(instance)
        )


class DependencyMismatch(FaultModule):
    """Install a dependency version that disagrees with the app's requirement."""

    name = "dependency_mismatch"
    difficulty = "hard"
    dependency_name = "web-framework"
    bad_version = "2.0.0-incompatible"

    def inject(self, instance: Any) -> None:
        instance.set_dependency_version(self.dependency_name, self.bad_version)

    def is_resolved(self, instance: Any) -> bool:
        dependencies = _observe(instance).get("dependencies", {})
        if not isinstance(dependencies, Mapping):
            return False
        installed = dependencies.get("installed", {})
        required = dependencies.get("required", {})
        if not isinstance(installed, Mapping) or not isinstance(required, Mapping):
            return False
        wanted = required.get(self.dependency_name)
        return bool(
            wanted is not None
            and installed.get(self.dependency_name) == wanted
            and _mechanically_healthy(instance)
        )


class DiskFull(FaultModule):
    """Fill the disposable app's disk beyond its health threshold."""

    name = "disk_full"
    difficulty = "medium"
    injected_percent = 99.0

    def inject(self, instance: Any) -> None:
        instance.set_disk_usage(self.injected_percent)

    def is_resolved(self, instance: Any) -> bool:
        disk = _observe(instance).get("disk", {})
        if not isinstance(disk, Mapping):
            return False
        try:
            used = float(disk["used_percent"])
            threshold = float(disk["healthy_below_percent"])
        except (KeyError, TypeError, ValueError):
            return False
        return used < threshold and _mechanically_healthy(instance)


class PortProxyMisconfig(FaultModule):
    """Route the reverse proxy to a port where the app is not listening."""

    name = "port_proxy_misconfig"
    difficulty = "easy"
    wrong_port = 8081

    def inject(self, instance: Any) -> None:
        instance.set_proxy_target_port(self.wrong_port)

    def is_resolved(self, instance: Any) -> bool:
        network = _observe(instance).get("network", {})
        if not isinstance(network, Mapping):
            return False
        app_port = network.get("app_port")
        proxy_port = network.get("proxy_target_port")
        return bool(
            app_port is not None
            and proxy_port == app_port
            and _mechanically_healthy(instance)
        )


ALL_FAULTS = (
    OOMKill(),
    BadEnvVar(),
    BrokenDBConnection(),
    DependencyMismatch(),
    DiskFull(),
    PortProxyMisconfig(),
)
"""Stateless fault instances ready for :meth:`Orchestrator.run_batch`."""

FAULT_TYPES = (
    OOMKill,
    BadEnvVar,
    BrokenDBConnection,
    DependencyMismatch,
    DiskFull,
    PortProxyMisconfig,
)

__all__ = [
    "ALL_FAULTS",
    "FAULT_TYPES",
    "BadEnvVar",
    "BrokenDBConnection",
    "DependencyMismatch",
    "DiskFull",
    "OOMKill",
    "PortProxyMisconfig",
]
