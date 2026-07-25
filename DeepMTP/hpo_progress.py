"""Progress events shared by hyperparameter optimizers and UI adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

HPOProgressEventKind = Literal[
    "optimizer_started",
    "bracket_started",
    "iteration_started",
    "configuration_started",
    "configuration_completed",
    "optimizer_completed",
]


@dataclass(frozen=True)
class HPOProgressEvent:
    """One optimizer lifecycle event."""

    kind: HPOProgressEventKind
    current: int | None = None
    total: int | None = None
    bracket: int | None = None
    iteration: int | None = None
    config: Any = None
    score: float | None = None


class HPOProgressObserver(Protocol):
    """Receives optimizer lifecycle events."""

    def on_event(self, event: HPOProgressEvent) -> None:
        """Handle an optimizer event."""


class NullHPOProgressObserver:
    """No-op optimizer progress observer."""

    def on_event(self, event: HPOProgressEvent) -> None:
        return None


class ConsoleHPOProgressObserver:
    """Render optimizer events in a terminal."""

    def on_event(self, event: HPOProgressEvent) -> None:
        if event.kind == "bracket_started":
            print(
                f"-- Running bracket {event.current}/{event.total} "
                f"with starting budget {event.bracket}"
            )
        elif event.kind == "configuration_started":
            print(f"---- Evaluating configuration {event.current}/{event.total}...")
        elif event.kind == "configuration_completed":
            print(f"---- Finished evaluating configuration with score: {event.score}")
        elif event.kind == "optimizer_completed":
            print("Hyperparameter optimization completed.")


def build_hpo_progress_observer(verbose: bool) -> HPOProgressObserver:
    if verbose:
        return ConsoleHPOProgressObserver()
    return NullHPOProgressObserver()
