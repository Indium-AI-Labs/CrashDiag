"""Fault contracts and built-in infrastructure failures."""

from .base import FaultModule
from .modules import (
    ALL_FAULTS,
    FAULT_TYPES,
    BadEnvVar,
    BrokenDBConnection,
    DependencyMismatch,
    DiskFull,
    OOMKill,
    PortProxyMisconfig,
)

__all__ = [
    "ALL_FAULTS",
    "FAULT_TYPES",
    "BadEnvVar",
    "BrokenDBConnection",
    "DependencyMismatch",
    "DiskFull",
    "FaultModule",
    "OOMKill",
    "PortProxyMisconfig",
]
