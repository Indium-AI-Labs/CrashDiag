"""Sandbox backends for CrashDiag."""

from .coolify import CoolifySandbox
from .mock import MockSandbox, SandboxBackend

__all__ = [
    "CoolifySandbox",
    "MockSandbox",
    "SandboxBackend",
]
