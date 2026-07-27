"""Progress messages shared by data preparation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, cast

DataProgressLevel = Literal["info", "warning", "error"]
DataPrintMode = Literal["basic", "dev"]
DataProgressEventKind = Literal[
    "message",
    "operation_started",
    "operation_completed",
]


@dataclass(frozen=True)
class DataProgressEvent:
    """One user-facing data preparation or validation message."""

    message: str
    kind: DataProgressEventKind = "message"
    subject: str | None = None
    level: DataProgressLevel = "info"
    end: str = "\n"
    value: object | None = None


class DataProgressObserver(Protocol):
    """Receives data preparation and validation messages."""

    def on_event(self, event: DataProgressEvent) -> None:
        """Handle one data progress event."""


class NullDataProgressObserver:
    """No-op observer used when data progress is disabled."""

    def on_event(self, event: DataProgressEvent) -> None:
        return None


class ConsoleDataProgressObserver:
    """Render data progress using the historical basic or developer format."""

    def __init__(self, print_mode: DataPrintMode = "basic") -> None:
        if not isinstance(print_mode, str) or print_mode not in {"basic", "dev"}:
            raise ValueError("print_mode must be either 'basic' or 'dev'.")
        self.print_mode = print_mode

    def on_event(self, event: DataProgressEvent) -> None:
        prefix = (
            f"{event.level}: " if self.print_mode == "dev" and event.message else ""
        )
        print(f"{prefix}{event.message}", end=event.end)


def build_data_progress_observer(
    enabled: bool,
    *,
    print_mode: DataPrintMode = "basic",
) -> DataProgressObserver:
    """Select the default data progress observer."""
    if enabled:
        return ConsoleDataProgressObserver(print_mode)
    return NullDataProgressObserver()


def _resolve_data_progress(
    verbose: bool,
    print_mode: str,
    progress: DataProgressObserver | None,
) -> DataProgressObserver:
    """Resolve an injected observer or construct the default observer."""
    if progress is not None:
        return progress
    return build_data_progress_observer(
        verbose,
        print_mode=cast(DataPrintMode, print_mode),
    )


def _emit_data_operation_started(
    progress: DataProgressObserver,
    *,
    subject: str,
    message: str,
) -> None:
    progress.on_event(
        DataProgressEvent(
            message,
            kind="operation_started",
            subject=subject,
            end="",
        )
    )


def _emit_data_operation_completed(
    progress: DataProgressObserver,
    *,
    subject: str,
    message: str,
    value: object | None = None,
    end: str = "\n",
) -> None:
    progress.on_event(
        DataProgressEvent(
            message,
            kind="operation_completed",
            subject=subject,
            value=value,
            end=end,
        )
    )


def _emit_data_message(
    verbose: bool,
    print_mode: str,
    progress: DataProgressObserver | None,
    *,
    subject: str,
    message: str,
    value: object | None = None,
    level: DataProgressLevel = "info",
    end: str = "\n",
) -> None:
    if not verbose:
        return
    _resolve_data_progress(verbose, print_mode, progress).on_event(
        DataProgressEvent(
            message,
            subject=subject,
            value=value,
            level=level,
            end=end,
        )
    )


def _emit_data_separator(
    verbose: bool,
    print_mode: str,
    progress: DataProgressObserver | None,
) -> None:
    _emit_data_message(
        verbose,
        print_mode,
        progress,
        subject="data_processing",
        message="",
        value="separator",
    )


def _emit_data_operation_started_if_verbose(
    verbose: bool,
    print_mode: str,
    progress: DataProgressObserver | None,
    *,
    subject: str,
    message: str,
) -> None:
    if not verbose:
        return
    _emit_data_operation_started(
        _resolve_data_progress(verbose, print_mode, progress),
        subject=subject,
        message=message,
    )


def _emit_data_operation_completed_if_verbose(
    verbose: bool,
    print_mode: str,
    progress: DataProgressObserver | None,
    *,
    subject: str,
    message: str,
    value: object | None = None,
) -> None:
    if not verbose:
        return
    _emit_data_operation_completed(
        _resolve_data_progress(verbose, print_mode, progress),
        subject=subject,
        message=message,
        value=value,
    )
