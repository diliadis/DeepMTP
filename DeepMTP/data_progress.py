"""Progress messages shared by data preparation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

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
