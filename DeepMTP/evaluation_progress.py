"""Structured progress events for metric evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

EvaluationProgressEventKind = Literal[
    "input_inspected",
    "evaluation_started",
    "values_unscaled",
    "metric_calculated",
    "group_results_calculated",
    "aggregate_results_calculated",
    "single_class_group",
    "evaluation_completed",
]
EvaluationAveraging = Literal["micro", "macro", "instance"]
EvaluationGroup = Literal["target", "instance"]


@dataclass(frozen=True)
class EvaluationProgressEvent:
    """A diagnostic or result produced while calculating metrics."""

    kind: EvaluationProgressEventKind
    mode: str | None = None
    nonzero_predictions: int | None = None
    true_values_have_nan: bool = False
    predicted_values_have_nan: bool = False
    values_preview: object | None = None
    announce_unscaled_values: bool = False
    metric_name: str | None = None
    metric_value: float | None = None
    averaging: EvaluationAveraging | None = None
    group: EvaluationGroup | None = None
    group_id: object | None = None
    top_k: int | None = None
    results: object | None = None
    unique_true_values: int | None = None


class EvaluationProgressObserver(Protocol):
    """Receives detailed metric-evaluation events."""

    def on_event(self, event: EvaluationProgressEvent) -> None:
        """Handle one evaluation event."""


class NullEvaluationProgressObserver:
    """No-op observer used when detailed metric output is disabled."""

    def on_event(self, event: EvaluationProgressEvent) -> None:
        return None


class ConsoleEvaluationProgressObserver:
    """Render the historical metric-evaluation console output."""

    def on_event(self, event: EvaluationProgressEvent) -> None:
        if event.kind == "input_inspected":
            print(f"There are {event.nonzero_predictions} non zero predictions")
            if event.true_values_have_nan:
                print("NaN value in true_values_arr")
            if event.predicted_values_have_nan:
                print("NaN value in pred_values_arr")
        elif event.kind == "evaluation_started":
            print(f"========== {event.mode} ==========")
        elif event.kind == "values_unscaled":
            if event.announce_unscaled_values:
                print("These are the unscaled values: ")
            print(f"Unscaled True_values: {event.values_preview}")
        elif event.kind == "metric_calculated":
            print(f"{event.metric_name}_{event.averaging}: {event.metric_value}")
        elif event.kind == "group_results_calculated":
            top_k = f" top_{event.top_k}" if event.top_k is not None else ""
            print(f"Per {event.group}{top_k} results: {event.results}")
        elif event.kind == "aggregate_results_calculated":
            label = "Macro" if event.averaging == "macro" else "Instance"
            print(f"{label} results: {event.results}")
        elif event.kind == "single_class_group":
            print(
                f"Warning: instance{event.group_id} has "
                f"{event.unique_true_values} unique true values"
            )
        elif event.kind == "evaluation_completed":
            print("==================================")
            print("")


def build_evaluation_progress_observer(
    enabled: bool,
) -> EvaluationProgressObserver:
    """Select the default detailed-evaluation observer."""
    if enabled:
        return ConsoleEvaluationProgressObserver()
    return NullEvaluationProgressObserver()
