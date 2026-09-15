"""Per-session Docker Compose backend for CrashDiag.

Each episode gets an isolated victim stack.  Mechanical health is read from the
in-container runtime (child processes, tmpfs usage, cert/permission files) plus
``docker inspect``.  Model-supplied parameters are never forwarded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar

from crashdiag.sandbox_apps.mock import SandboxBackend


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_COMPOSE = Path(__file__).resolve().parent / "target" / "compose.yaml"
DEFAULT_DOCKER = os.environ.get("CRASHDIAG_DOCKER_BIN", "docker")
VICTIM_IMAGE = "crashdiag-victim:local"


def host_repo_root() -> str:
    """Return the Docker Engine build context, which must be a host path.

    Sibling-Docker deploys talk to the host daemon over a mounted socket, so
    ``/app`` inside the API container is not a valid build context.
    """

    override = os.environ.get("CRASHDIAG_REPO_ROOT", "").strip()
    return override or str(REPO_ROOT)


class DockerUnavailableError(RuntimeError):
    """Raised when the Docker Engine CLI cannot be used."""


def docker_available(docker_bin: str = DEFAULT_DOCKER) -> bool:
    """Return whether ``docker info`` succeeds."""

    if shutil.which(docker_bin) is None:
        return False
    try:
        completed = subprocess.run(
            [docker_bin, "info"],
            check=False,
            capture_output=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _session_token() -> str:
    material = f"{os.getpid()}:{time.time_ns()}:{os.urandom(8).hex()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


class DockerSandbox(SandboxBackend):
    """Launch one labeled Compose project and RPC into its victim control plane."""

    ACTIONS: ClassVar[frozenset[str]] = SandboxBackend.ACTIONS

    def __init__(
        self,
        *,
        session_id: str | None = None,
        docker_bin: str = DEFAULT_DOCKER,
        compose_file: Path | None = None,
        startup_timeout: float = 60.0,
    ) -> None:
        if not docker_available(docker_bin):
            raise DockerUnavailableError(
                "Docker Engine is not available; deploy the sandbox host with "
                "the Docker backend or use CRASHDIAG_SANDBOX_BACKEND=mock"
            )
        self._docker = docker_bin
        self._compose_file = compose_file or TARGET_COMPOSE
        if not self._compose_file.is_file():
            raise FileNotFoundError(f"missing victim compose file: {self._compose_file}")
        self.session_id = session_id or _session_token()
        self.project = f"cd{self.session_id}"
        self._closed = False
        self._up(timeout=startup_timeout)

    def close(self) -> None:
        """Destroy the Compose project and any leftover labeled containers."""

        if self._closed:
            return
        self._closed = True
        self._compose(["down", "-v", "--remove-orphans", "--timeout", "2"], check=False)
        leftover = self._run(
            [
                self._docker,
                "ps",
                "-aq",
                "--filter",
                f"label=crashdiag.session={self.session_id}",
            ],
            check=False,
        )
        ids = leftover.stdout.decode("utf-8", errors="replace").split()
        if ids:
            self._run([self._docker, "rm", "-f", *ids], check=False)

    def __enter__(self) -> "DockerSandbox":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort teardown
        try:
            self.close()
        except Exception:
            return

    def _run(
        self,
        command: list[str],
        *,
        check: bool = True,
        input_bytes: bytes | None = None,
        timeout: float = 30.0,
    ) -> subprocess.CompletedProcess[bytes]:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            input=input_bytes,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        if check and completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"docker command failed ({completed.returncode}): "
                f"{' '.join(command)}\n{stderr}"
            )
        return completed

    def _compose(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        env_prefix = {
            "CRASHDIAG_SESSION_ID": self.session_id,
            "CRASHDIAG_REPO_ROOT": host_repo_root(),
        }
        command = [
            self._docker,
            "compose",
            "-p",
            self.project,
            "-f",
            str(self._compose_file),
            *args,
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=120,
            cwd=str(REPO_ROOT),
            env={**os.environ, **env_prefix},
        )
        if check and completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"docker compose failed ({completed.returncode}): "
                f"{' '.join(command)}\n{stderr}"
            )
        return completed

    def _container_id(self) -> str:
        completed = self._run(
            [
                self._docker,
                "ps",
                "-q",
                "--filter",
                f"label=crashdiag.session={self.session_id}",
                "--filter",
                "label=crashdiag.role=victim",
            ]
        )
        container_id = completed.stdout.decode("utf-8", errors="replace").strip().splitlines()
        if not container_id:
            raise RuntimeError(f"victim container missing for session {self.session_id}")
        return container_id[0]

    def _ensure_victim_image(self) -> None:
        inspect = self._run(
            [self._docker, "image", "inspect", VICTIM_IMAGE],
            check=False,
        )
        if inspect.returncode != 0:
            self._compose(["build"])

    def _up(self, *, timeout: float) -> None:
        self._ensure_victim_image()
        self._compose(["up", "-d", "--remove-orphans"])
        deadline = time.monotonic() + timeout
        last_error = "victim control plane did not start"
        while time.monotonic() < deadline:
            try:
                self.health_check()
                return
            except Exception as exc:  # noqa: BLE001 - retry until timeout
                last_error = str(exc)
                time.sleep(0.4)
        raise RuntimeError(last_error)

    def _rpc(self, op: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("DockerSandbox is closed")
        payload = json.dumps(
            {"op": op, "args": list(args), "kwargs": kwargs},
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        script = (
            "import json,sys,urllib.request;"
            "req=urllib.request.Request('http://127.0.0.1:8080/rpc',data=sys.stdin.buffer.read(),"
            "method='POST',headers={'Content-Type':'application/json'});"
            "sys.stdout.buffer.write(urllib.request.urlopen(req,timeout=8).read())"
        )
        completed = self._run(
            [self._docker, "exec", "-i", self._container_id(), "python", "-c", script],
            input_bytes=payload,
            timeout=20,
        )
        try:
            decoded = json.loads(completed.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("victim RPC returned invalid JSON") from exc
        if not isinstance(decoded, dict) or decoded.get("ok") is not True:
            raise RuntimeError(f"victim RPC failed: {decoded!r}")
        return decoded.get("result")

    def observe(self) -> dict[str, Any]:
        snapshot = self._rpc("observe")
        if not isinstance(snapshot, dict):
            raise RuntimeError("victim observe() must return an object")
        return snapshot

    def health_check(self) -> dict[str, Any]:
        result = self._rpc("health_check")
        if not isinstance(result, dict):
            raise RuntimeError("victim health_check() must return an object")
        return result

    def restart_app(self) -> dict[str, Any]:
        return self._rpc("restart_app")

    def rollback_env_var(self, name: str | None = None) -> dict[str, Any]:
        return self._rpc("rollback_env_var", name=name)

    def fix_dependency(
        self,
        name: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        del version
        return self._rpc("fix_dependency", name=name)

    def clear_disk(self, target_percent: float = 40.0) -> dict[str, Any]:
        return self._rpc("clear_disk", target_percent=target_percent)

    def fix_port_config(self, target_port: int | None = None) -> dict[str, Any]:
        return self._rpc("fix_port_config", target_port=target_port)

    def clear_cache(self) -> dict[str, Any]:
        return self._rpc("clear_cache")

    def renew_tls_certificate(self) -> dict[str, Any]:
        return self._rpc("renew_tls_certificate")

    def restore_file_permissions(self) -> dict[str, Any]:
        return self._rpc("restore_file_permissions")

    def apply_database_migration(self) -> dict[str, Any]:
        return self._rpc("apply_database_migration")

    def reset_database_pool(self) -> dict[str, Any]:
        return self._rpc("reset_database_pool")

    def restore_dns_configuration(self) -> dict[str, Any]:
        return self._rpc("restore_dns_configuration")

    def restore_rate_limit_configuration(self) -> dict[str, Any]:
        return self._rpc("restore_rate_limit_configuration")

    def restart_worker(self) -> dict[str, Any]:
        return self._rpc("restart_worker")

    def redeploy_container(self) -> dict[str, Any]:
        return self._rpc("redeploy_container")

    def clear_temp_files(self) -> dict[str, Any]:
        return self._rpc("clear_temp_files")

    def rotate_logs(self) -> dict[str, Any]:
        return self._rpc("rotate_logs")

    def restore_load_balancer_config(self) -> dict[str, Any]:
        return self._rpc("restore_load_balancer_config")

    def restore_network_config(self) -> dict[str, Any]:
        return self._rpc("restore_network_config")

    def sync_replica(self) -> dict[str, Any]:
        return self._rpc("sync_replica")

    def restore_database_config(self) -> dict[str, Any]:
        return self._rpc("restore_database_config")

    def flush_dead_letter_queue(self) -> dict[str, Any]:
        return self._rpc("flush_dead_letter_queue")

    def restore_cache_config(self) -> dict[str, Any]:
        return self._rpc("restore_cache_config")

    def reset_circuit_breaker(self) -> dict[str, Any]:
        return self._rpc("reset_circuit_breaker")

    def restore_cron_schedule(self) -> dict[str, Any]:
        return self._rpc("restore_cron_schedule")

    def rebuild_index(self) -> dict[str, Any]:
        return self._rpc("rebuild_index")

    def restore_tls_config(self) -> dict[str, Any]:
        return self._rpc("restore_tls_config")

    def wait_and_observe(self) -> dict[str, Any]:
        return self._rpc("wait_and_observe")

    def trigger_oom_kill(self) -> None:
        self._rpc("trigger_oom_kill")

    def set_env_var(self, name: str, value: str) -> None:
        self._rpc("set_env_var", name, value)

    def set_dependency_version(self, name: str, version: str) -> None:
        self._rpc("set_dependency_version", name, version)

    def set_disk_usage(self, percent: float) -> None:
        self._rpc("set_disk_usage", percent)

    def set_proxy_target_port(self, port: int) -> None:
        self._rpc("set_proxy_target_port", port)

    def set_service_state(self, name: str, healthy: bool) -> None:
        self._rpc("set_service_state", name, healthy)

    def set_expected_env_var(self, name: str, value: str) -> None:
        self._rpc("set_expected_env_var", name, value)

    def set_required_dependency_version(self, name: str, version: str) -> None:
        self._rpc("set_required_dependency_version", name, version)

    def set_app_port(self, port: int) -> None:
        self._rpc("set_app_port", port)

    def set_disk_health_threshold(self, percent: float) -> None:
        self._rpc("set_disk_health_threshold", percent)


def sandbox_backend_name() -> str:
    """Return the configured sandbox backend name."""

    return os.environ.get("CRASHDIAG_SANDBOX_BACKEND", "mock").strip().lower() or "mock"


def sandbox_factory_from_env() -> type[SandboxBackend] | Any:
    """Return a zero-argument sandbox constructor from ``CRASHDIAG_SANDBOX_BACKEND``."""

    name = sandbox_backend_name()
    if name in {"", "mock"}:
        from crashdiag.sandbox_apps.mock import MockSandbox

        return MockSandbox
    if name == "docker":
        return DockerSandbox
    raise ValueError(f"unknown CRASHDIAG_SANDBOX_BACKEND {name!r}")


__all__ = [
    "DockerSandbox",
    "DockerUnavailableError",
    "VICTIM_IMAGE",
    "docker_available",
    "host_repo_root",
    "sandbox_backend_name",
    "sandbox_factory_from_env",
]
