"""Progress events shared by hyperparameter optimizers and UI adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

HPOProgressEventKind = Literal[
    "schedule_calculated",
    "worker_configured",
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
    smax: int | None = None
    eta: int | None = None
    max_budget: int | None = None
    total_budget: int | None = None


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
        if event.kind == "schedule_calculated":
            schedule = event.config if isinstance(event.config, Mapping) else {}
            print(f"smax: {event.smax}")
            print(f"B: {event.total_budget}")
            print()
            for num_configurations, bracket in schedule.items():
                if not isinstance(bracket, Mapping):
                    continue
                iterations = int(bracket["num_iters"])
                budgets = bracket["r_i"]
                configurations = bracket["n_i"]
                print(f"s: {iterations - 1}")
                print(f"     n: {num_configurations}   r: {budgets[0]}")
                print("---------------------------")
                for count, budget in zip(configurations, budgets, strict=True):
                    print(f"     ni: {count}   ri (epochs): {budget}")
                print()
                print("===========================")
        elif event.kind == "worker_configured":
            print(f"HPO worker used this final config: {event.config}")
        elif event.kind == "bracket_started":
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
