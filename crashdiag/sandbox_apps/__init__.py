"""Sandbox backends for CrashDiag."""

from .coolify import CoolifySandbox
from .http import HTTPSandbox, HttpSandbox, SandboxHTTPError, SandboxTransportError
from .mock import MockSandbox, SandboxBackend

__all__ = [
    "CoolifySandbox",
    "HTTPSandbox",
    "HttpSandbox",
    "MockSandbox",
    "SandboxBackend",
    "SandboxHTTPError",
    "SandboxTransportError",
]
