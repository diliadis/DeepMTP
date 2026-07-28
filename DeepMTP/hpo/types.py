"""Shared structural types for hyperparameter optimizers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypedDict

HPOConfiguration = Mapping[str, Any]


class HPOWorkerResult(TypedDict):
    """Stable result returned by an HPO worker."""

    loss: float
    info: dict[str, Any]


class HPOWorker(Protocol):
    """Worker surface consumed by the bundled optimizers."""

    def compute(
        self,
        budget: int | float,
        config: HPOConfiguration,
    ) -> HPOWorkerResult:
        """Evaluate one configuration at the requested budget."""


class ConfigurationSpace(Protocol):
    """Configuration-space surface consumed by the bundled optimizers."""

    def sample_configuration(self) -> HPOConfiguration:
        """Sample one candidate configuration."""
