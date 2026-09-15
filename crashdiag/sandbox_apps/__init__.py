"""Sandbox backends for CrashDiag."""

from .coolify import CoolifySandbox
from .docker import (
    DockerSandbox,
    DockerUnavailableError,
    docker_available,
    sandbox_backend_name,
    sandbox_factory_from_env,
)
from .http import HTTPSandbox, HttpSandbox, SandboxHTTPError, SandboxTransportError
from .mock import MockSandbox, SandboxBackend

__all__ = [
    "CoolifySandbox",
    "DockerSandbox",
    "DockerUnavailableError",
    "HTTPSandbox",
    "HttpSandbox",
    "MockSandbox",
    "SandboxBackend",
    "SandboxHTTPError",
    "SandboxTransportError",
    "docker_available",
    "sandbox_backend_name",
    "sandbox_factory_from_env",
]
