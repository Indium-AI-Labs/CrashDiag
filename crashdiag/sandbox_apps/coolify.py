"""Conservative Coolify backend skeleton.

Coolify API routes and payloads can vary by release.  This module deliberately
does not invent endpoints: it provides the complete :class:`SandboxBackend`
surface, stores connection configuration, and clearly marks every operation
that still needs validation against the target Coolify deployment.
"""

from __future__ import annotations

from typing import Any, NoReturn

from .mock import SandboxBackend


class CoolifySandbox(SandboxBackend):
    """Configuration and interface-compatible stubs for a Coolify application.

    Constructing this class performs no network I/O.  Before enabling it, each
    operation must be implemented and tested against the exact Coolify version
    in use.  In particular, action success must ultimately be confirmed from
    real process/HTTP/system state rather than inferred from an accepted API
    request.

    Args:
        base_url: Root URL of the target Coolify installation.  No API path is
            appended because that path has not been assumed in this pass.
        api_token: Authentication token for the target installation.
        application_uuid: Coolify identifier for the disposable application.
        health_url: Optional public application URL for a future mechanical
            HTTP health check.  It is stored but not requested in this pass.
    """

    def __init__(
        self,
        base_url: str,
        api_token: str,
        application_uuid: str,
        *,
        health_url: str | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if not api_token:
            raise ValueError("api_token must be non-empty")
        if not application_uuid:
            raise ValueError("application_uuid must be non-empty")

        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.application_uuid = application_uuid
        self.health_url = health_url

    @staticmethod
    def _unimplemented(operation: str) -> NoReturn:
        raise NotImplementedError(
            f"CoolifySandbox.{operation} is a stub: verify the API contract "
            "for the target Coolify version before implementing it"
        )

    def observe(self) -> dict[str, Any]:
        """Collect the standardized sandbox observation.

        A future implementation must retrieve actual runtime, environment,
        dependency, disk, and proxy state.  No Coolify endpoint is assumed.
        """

        self._unimplemented("observe")

    def health_check(self) -> dict[str, Any]:
        """Mechanically check application health from real system state.

        This will typically make a direct HTTP request to ``health_url`` and
        return at least ``{"healthy": bool}``, but response criteria are
        deployment-specific and are intentionally not guessed here.
        """

        self._unimplemented("health_check")

    def restart_app(self) -> dict[str, Any]:
        """Restart the configured application and return observed action data."""

        self._unimplemented("restart_app")

    def rollback_env_var(self, name: str | None = None) -> dict[str, Any]:
        """Restore one environment variable to its known-good value."""

        self._unimplemented("rollback_env_var")

    def fix_dependency(
        self,
        name: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Restore the declared dependency lock and return observed action data."""

        self._unimplemented("fix_dependency")

    def clear_disk(self, target_percent: float = 40.0) -> dict[str, Any]:
        """Reduce disk utilization to ``target_percent`` or lower."""

        self._unimplemented("clear_disk")

    def fix_port_config(self, target_port: int | None = None) -> dict[str, Any]:
        """Restore the proxy target to the application's actual listening port."""

        self._unimplemented("fix_port_config")

    def wait_and_observe(self) -> dict[str, Any]:
        """Take no corrective action, then return a fresh observation."""

        self._unimplemented("wait_and_observe")

    def trigger_oom_kill(self) -> None:
        """Inject memory exhaustion into the disposable application."""

        self._unimplemented("trigger_oom_kill")

    def set_env_var(self, name: str, value: str) -> None:
        """Set an application environment variable for fault injection."""

        self._unimplemented("set_env_var")

    def set_dependency_version(self, name: str, version: str) -> None:
        """Set a dependency version for fault injection and rebuild the app."""

        self._unimplemented("set_dependency_version")

    def set_disk_usage(self, percent: float) -> None:
        """Drive disposable storage utilization to the requested percentage."""

        self._unimplemented("set_disk_usage")

    def set_proxy_target_port(self, port: int) -> None:
        """Point the proxy at ``port`` for fault injection."""

        self._unimplemented("set_proxy_target_port")

    def set_expected_env_var(self, name: str, value: str) -> None:
        """Configure one known-good environment baseline."""

        self._unimplemented("set_expected_env_var")

    def set_required_dependency_version(self, name: str, version: str) -> None:
        """Configure one known-good dependency baseline."""

        self._unimplemented("set_required_dependency_version")

    def set_app_port(self, port: int) -> None:
        """Configure the application and healthy proxy port."""

        self._unimplemented("set_app_port")

    def set_disk_health_threshold(self, percent: float) -> None:
        """Configure the disk-health boundary."""

        self._unimplemented("set_disk_health_threshold")


__all__ = ["CoolifySandbox"]
