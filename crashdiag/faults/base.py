"""Base contract for mechanically verifiable fault modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FaultModule(ABC):
    """A fault that can be injected and checked against sandbox state.

    Implementations are deliberately small and deterministic.  In particular,
    :meth:`is_resolved` must inspect the supplied instance; it must never ask a
    model to interpret an action log or decide whether a fix looks plausible.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable machine-readable fault name."""

    @property
    @abstractmethod
    def difficulty(self) -> str:
        """Human-readable difficulty tier (``easy``, ``medium``, or ``hard``)."""

    @abstractmethod
    def inject(self, instance: Any) -> None:
        """Mutate ``instance`` so this fault is present."""

    @abstractmethod
    def is_resolved(self, instance: Any) -> bool:
        """Return whether real, current instance state proves the fault fixed."""


__all__ = ["FaultModule"]
