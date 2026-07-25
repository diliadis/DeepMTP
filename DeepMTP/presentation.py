"""Run history formatting and user-facing progress events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from prettytable import PrettyTable


@dataclass(frozen=True)
class RunSummaries:
    """Rendered train, validation, and test histories."""

    train: str
    validation: str
    test: str


class RunHistory:
    """Collects experiment results and formats the legacy summary tables."""

    def __init__(
        self,
        metrics: Sequence[str],
        averaging: Sequence[str],
    ):
        self.metrics = tuple(metrics)
        self.averaging = tuple(averaging)
        header = ["mode", "#epoch", "loss"] + [
            f"{metric}_{average}"
            for metric in self.metrics
            for average in self.averaging
        ]
        self._train = PrettyTable(header)
        self._validation = PrettyTable(header + ["early_stopping"])
        self._test = PrettyTable(header)

    def add_training(
        self,
        epoch: int,
        loss: float,
        results: Mapping[str, float],
    ) -> None:
        self._train.add_row(self._row("train", epoch, loss, results))

    def add_validation(
        self,
        epoch: int,
        loss: float,
        results: Mapping[str, float],
        *,
        early_stopping_counter: int,
        early_stopping_patience: int,
    ) -> None:
        row = self._row("val", epoch, loss, results)
        row.append(f"{early_stopping_counter}/{early_stopping_patience}")
        self._validation.add_row(row)

    def add_test(
        self,
        best_epoch: int,
        results: Mapping[str, float],
    ) -> None:
        self._test.add_row(self._row("test", best_epoch, "-", results))

    def render(self) -> RunSummaries:
        return RunSummaries(
            train=self._train.get_string(),
            validation=self._validation.get_string(),
            test=self._test.get_string(),
        )

    def _row(
        self,
        mode: str,
        epoch: int,
        loss: float | str,
        results: Mapping[str, float],
    ) -> list[object]:
        rendered_loss = round(loss, 4) if isinstance(loss, float) else loss
        return [mode, str(epoch), rendered_loss] + [
            round(results[key], 4) if key in results else "-"
            for metric in self.metrics
            for average in self.averaging
            for key in [f"{mode}_{metric}_{average}"]
        ]


ProgressEventKind = Literal[
    "training_started",
    "epoch_started",
    "epoch_completed",
    "validation_started",
    "validation_completed",
    "metrics_started",
    "metrics_completed",
    "early_stopping_triggered",
    "testing_started",
    "testing_completed",
]


@dataclass(frozen=True)
class ProgressEvent:
    """A lifecycle event suitable for console or UI adapters."""

    kind: ProgressEventKind
    epoch: int | None = None
    mode: str | None = None


class ProgressObserver(Protocol):
    """Receives training lifecycle events and final summaries."""

    def on_event(self, event: ProgressEvent) -> None:
        """Handle one lifecycle event."""

    def on_summary(
        self,
        summaries: RunSummaries,
        *,
        include_train: bool,
        include_validation: bool,
    ) -> None:
        """Handle the final rendered run summaries."""


class NullProgressObserver:
    """No-op observer used for non-verbose experiments."""

    def on_event(self, event: ProgressEvent) -> None:
        return None

    def on_summary(
        self,
        summaries: RunSummaries,
        *,
        include_train: bool,
        include_validation: bool,
    ) -> None:
        return None


class ConsoleProgressObserver:
    """Renders progress events using the historical console messages."""

    def on_event(self, event: ProgressEvent) -> None:
        if event.kind == "training_started":
            print("Starting training...")
        elif event.kind == "epoch_started":
            print(f"Epoch:{event.epoch}... ", end="")
        elif event.kind == "epoch_completed":
            print("Done")
        elif event.kind == "validation_started":
            print("  Validating... ", end="")
        elif event.kind == "validation_completed":
            print("Done")
        elif event.kind == "metrics_started":
            print(
                f"Calculating {event.mode or 'prediction'} performance... ",
                end="",
            )
        elif event.kind == "metrics_completed":
            print("Done")
        elif event.kind == "early_stopping_triggered":
            print("Early stopping criterion met. Training stopped!")
        elif event.kind == "testing_started":
            print("Starting testing... ", end="")
        elif event.kind == "testing_completed":
            print("Done")

    def on_summary(
        self,
        summaries: RunSummaries,
        *,
        include_train: bool,
        include_validation: bool,
    ) -> None:
        if include_train:
            print(summaries.train)
        print(20 * "=")
        if include_validation:
            print(summaries.validation)
        print(20 * "=")
        print(summaries.test)


def build_progress_observer(verbose: bool) -> ProgressObserver:
    """Select the default observer for the configured verbosity."""
    if verbose:
        return ConsoleProgressObserver()
    return NullProgressObserver()
