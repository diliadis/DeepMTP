"""Streamlit renderers for core DeepMTP progress events."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd

from DeepMTP.config import DeepMTPConfig
from DeepMTP.hpo.progress import HPOProgressEvent
from DeepMTP.runtime.evaluation_progress import EvaluationProgressEvent
from DeepMTP.runtime.presentation import ProgressEvent, RunSummaries
from DeepMTP.runtime.reporting import OptionalIntegrationError

__all__ = [
    "StreamlitAPI",
    "StreamlitHPOProgressObserver",
    "StreamlitProgressObserver",
]


class StreamlitAPI(Protocol):
    """Small Streamlit surface used by the progress adapters."""

    def empty(self) -> Any:
        """Create an empty UI slot."""

    def progress(self, value: int) -> Any:
        """Create a progress indicator."""


class StreamlitProgressObserver:
    """Render core training events without owning training logic."""

    def __init__(
        self,
        config: DeepMTPConfig | Mapping[str, Any] | None = None,
        *,
        experiment_dir: str | Path | None = None,
        streamlit_module: StreamlitAPI | None = None,
    ) -> None:
        self._st = streamlit_module or _import_streamlit()
        self._config: DeepMTPConfig | None = None
        self._experiment_dir = None if experiment_dir is None else Path(experiment_dir)
        self._slots: dict[str, Any] = {}
        self._train_rows: list[dict[str, Any]] = []
        self._validation_rows: list[dict[str, Any]] = []
        self._loss_rows: list[dict[str, float]] = []
        if config is not None:
            self.configure(config, experiment_dir=experiment_dir)

    def configure(
        self,
        config: DeepMTPConfig | Mapping[str, Any],
        *,
        experiment_dir: str | Path | None = None,
    ) -> None:
        """Attach the finalized trainer configuration."""
        self._config = DeepMTPConfig.from_mapping(config)
        if experiment_dir is not None:
            self._experiment_dir = Path(experiment_dir)

    def on_event(self, event: ProgressEvent | EvaluationProgressEvent) -> None:
        if isinstance(event, EvaluationProgressEvent):
            return
        if event.kind == "training_started":
            self._start()
        elif event.kind == "epoch_completed":
            self._record_training(event)
        elif event.kind == "validation_completed":
            self._record_validation(event)
        elif event.kind == "early_stopping_triggered":
            self._slots["status"].warning(
                f"Early stopping triggered at epoch {event.epoch}."
            )
        elif event.kind == "testing_started":
            self._slots["status"].info("Evaluating the best model…")
        elif event.kind == "testing_completed":
            self._record_test(event)

    def on_summary(
        self,
        summaries: RunSummaries,
        *,
        include_train: bool,
        include_validation: bool,
    ) -> None:
        if self._experiment_dir is not None:
            self._slots["status"].info(
                f"Experiment artifacts were saved to `{self._experiment_dir}`."
            )
        if self._config is not None and self._config.running_hpo:
            self.clear()

    def clear(self) -> None:
        """Remove transient progress elements during HPO runs."""
        for slot in self._slots.values():
            slot.empty()

    def _start(self) -> None:
        if self._config is None:
            raise RuntimeError(
                "StreamlitProgressObserver must be configured before training."
            )
        self._slots = {
            "chart": self._st.empty(),
            "train_header": self._st.empty(),
            "train_table": self._st.empty(),
            "validation_header": self._st.empty(),
            "validation_table": self._st.empty(),
            "test_header": self._st.empty(),
            "test_table": self._st.empty(),
            "status": self._st.empty(),
        }
        self._slots["train_header"].subheader("Train progress")
        self._slots["validation_header"].subheader("Validation progress")
        self._slots["status"].info("Training started.")

    def _record_training(self, event: ProgressEvent) -> None:
        metrics = dict(event.metrics or {})
        row = self._result_row(
            "train",
            event.epoch,
            metrics,
            loss_key="train_loss",
        )
        self._train_rows.append(row)
        self._slots["train_table"].dataframe(
            pd.DataFrame(self._train_rows),
        )

    def _record_validation(self, event: ProgressEvent) -> None:
        metrics = dict(event.metrics or {})
        row = self._result_row(
            "val",
            event.epoch,
            metrics,
            loss_key="val_loss",
        )
        row["early_stopping"] = (
            f"{event.early_stopping_counter}/{event.early_stopping_patience}"
        )
        self._validation_rows.append(row)
        self._slots["validation_table"].dataframe(
            pd.DataFrame(self._validation_rows),
        )

        train_loss = self._train_rows[-1]["loss"]
        validation_loss = row["loss"]
        self._loss_rows.append({"train": train_loss, "validation": validation_loss})
        self._slots["chart"].line_chart(pd.DataFrame(self._loss_rows))

    def _record_test(self, event: ProgressEvent) -> None:
        self._slots["test_header"].subheader("Performance on the test set")
        row = self._result_row(
            "test",
            event.best_epoch,
            dict(event.metrics or {}),
            loss_key=None,
        )
        self._slots["test_table"].dataframe(
            pd.DataFrame([row]),
        )

    def _result_row(
        self,
        mode: str,
        epoch: int | None,
        results: Mapping[str, float],
        *,
        loss_key: str | None,
    ) -> dict[str, Any]:
        if self._config is None:
            raise RuntimeError("Streamlit progress is not configured.")
        row = {
            "mode": mode,
            "epoch": epoch,
            "loss": (
                round(results[loss_key], 4)
                if loss_key is not None and loss_key in results
                else "-"
            ),
        }
        row.update(
            {
                f"{metric}_{average}": (
                    round(results[key], 4) if key in results else "-"
                )
                for metric in self._config.metrics
                for average in self._config.metrics_average
                for key in [f"{mode}_{metric}_{average}"]
            }
        )
        return row


class StreamlitHPOProgressObserver:
    """Render shared optimizer events in Streamlit."""

    def __init__(self, *, streamlit_module: StreamlitAPI | None = None) -> None:
        self._st = streamlit_module or _import_streamlit()
        self._status: Any = None
        self._overall_progress: Any = None
        self._iteration_progress: Any = None
        self._config: Any = None

    def on_event(self, event: HPOProgressEvent) -> None:
        if event.kind == "optimizer_started":
            self._start()
        elif event.kind == "bracket_started":
            self._status.write(
                f"Running bracket {event.current}/{event.total} "
                f"with starting budget {event.bracket}."
            )
            self._overall_progress.progress(_percentage(event.current, event.total))
        elif event.kind == "iteration_started":
            self._iteration_progress.progress(_percentage(event.current, event.total))
        elif event.kind == "configuration_started":
            self._status.write(
                f"Evaluating configuration {event.current}/{event.total}."
            )
            self._config.json(_config_dictionary(event.config))
        elif event.kind == "configuration_completed":
            self._status.write(
                f"Finished evaluating configuration with score {event.score}."
            )
            if event.bracket is None:
                self._overall_progress.progress(_percentage(event.current, event.total))
        elif event.kind == "optimizer_completed":
            self._overall_progress.progress(100)
            self._iteration_progress.progress(100)
            self._config.empty()
            self._status.success("Hyperparameter optimization completed.")

    def _start(self) -> None:
        self._status = self._st.empty()
        self._overall_progress = self._st.progress(0)
        self._iteration_progress = self._st.progress(0)
        self._config = self._st.empty()


def _percentage(current: int | None, total: int | None) -> int:
    if current is None or total in (None, 0):
        return 0
    return min(100, round(current / total * 100))


def _config_dictionary(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "get_dictionary"):
        return dict(config.get_dictionary())
    try:
        return dict(config)
    except (TypeError, ValueError):
        return {"value": str(config)}


def _import_streamlit() -> StreamlitAPI:
    try:
        return cast(StreamlitAPI, importlib.import_module("streamlit"))
    except ModuleNotFoundError as error:
        raise OptionalIntegrationError(
            "Streamlit support requires the optional 'streamlit' dependency."
        ) from error
