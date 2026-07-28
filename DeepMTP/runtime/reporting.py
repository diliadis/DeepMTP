"""Optional experiment-reporting integrations."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, TypeAlias

from DeepMTP.config import DeepMTPConfig

ReporterConfigInput: TypeAlias = DeepMTPConfig | Mapping[str, Any]

__all__ = [
    "CompositeReporter",
    "ExperimentReporter",
    "NullReporter",
    "OptionalIntegrationError",
    "ReporterConfigInput",
    "TensorBoardReporter",
    "WandBReporter",
    "build_reporter",
]


@dataclass(frozen=True)
class _ReporterSettings:
    """Typed reporting options resolved at the compatibility boundary."""

    use_tensorboard_logger: bool
    wandb_project_name: str | None
    wandb_project_entity: str | None

    @classmethod
    def from_config(cls, config: ReporterConfigInput) -> _ReporterSettings:
        if isinstance(config, DeepMTPConfig):
            return cls(
                use_tensorboard_logger=config.use_tensorboard_logger,
                wandb_project_name=config.wandb_project_name,
                wandb_project_entity=config.wandb_project_entity,
            )
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or DeepMTPConfig.")
        return cls(
            use_tensorboard_logger=config["use_tensorboard_logger"],
            wandb_project_name=config["wandb_project_name"],
            wandb_project_entity=config["wandb_project_entity"],
        )


class OptionalIntegrationError(ImportError):
    """Raised when a requested optional reporting dependency is unavailable."""


class ExperimentReporter(Protocol):
    """Lifecycle used by the trainer to report experiment information."""

    def start(
        self,
        config: Mapping[str, Any],
        model: Any,
    ) -> None:
        """Initialize the reporter for an experiment."""

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        """Record a group of metrics."""

    def close(self) -> None:
        """Flush resources and finish the reporting run."""


class NullReporter:
    """No-op reporter used when no integrations are configured."""

    def start(self, config: Mapping[str, Any], model: Any) -> None:
        return None

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        return None

    def close(self) -> None:
        return None


class CompositeReporter:
    """Fan reporting events out to multiple integrations."""

    def __init__(self, reporters: Sequence[ExperimentReporter]):
        self._reporters = tuple(reporters)

    def start(self, config: Mapping[str, Any], model: Any) -> None:
        started_reporters = []
        try:
            for reporter in self._reporters:
                reporter.start(config, model)
                started_reporters.append(reporter)
        except Exception:
            for reporter in reversed(started_reporters):
                reporter.close()
            raise

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        for reporter in self._reporters:
            reporter.log_metrics(metrics, step=step)

    def close(self) -> None:
        for reporter in reversed(self._reporters):
            reporter.close()


class TensorBoardReporter:
    """TensorBoard adapter with a lazy optional dependency import."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self._writer: Any = None
        self._last_step = 0

    def start(self, config: Mapping[str, Any], model: Any) -> None:
        tensorboard_module = _import_optional_module(
            "torch.utils.tensorboard",
            integration_name="TensorBoard",
        )
        self._writer = tensorboard_module.SummaryWriter(str(self.log_dir))
        for key, value in config.items():
            self._writer.add_text(key, str(value))

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        if self._writer is None:
            raise RuntimeError("TensorBoardReporter must be started before use.")
        if step is not None:
            self._last_step = step
        for key, value in metrics.items():
            self._writer.add_scalar(
                key.replace("_", "/"),
                value,
                self._last_step,
            )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


class WandBReporter:
    """Weights & Biases adapter with a lazy optional dependency import."""

    def __init__(self, project_name: str, project_entity: str):
        self.project_name = project_name
        self.project_entity = project_entity
        self._run: Any = None

    def start(self, config: Mapping[str, Any], model: Any) -> None:
        wandb_module = _import_optional_module(
            "wandb",
            integration_name="Weights & Biases",
        )
        self._run = wandb_module.init(
            project=self.project_name,
            entity=self.project_entity,
            reinit=True,
        )
        self._run.watch(model)
        self._run.config.update(dict(config))

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        if self._run is None:
            raise RuntimeError("WandBReporter must be started before use.")
        if step is None:
            self._run.log(dict(metrics))
        else:
            self._run.log(dict(metrics), step=step)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def build_reporter(
    config: ReporterConfigInput,
    experiment_dir: str | Path,
) -> ExperimentReporter:
    """Build configured reporters from typed or legacy configuration."""
    settings = _ReporterSettings.from_config(config)

    reporters: list[ExperimentReporter] = []
    if settings.use_tensorboard_logger:
        reporters.append(TensorBoardReporter(experiment_dir))

    if (
        settings.wandb_project_name is not None
        and settings.wandb_project_entity is not None
    ):
        reporters.append(
            WandBReporter(
                settings.wandb_project_name,
                settings.wandb_project_entity,
            )
        )

    if not reporters:
        return NullReporter()
    if len(reporters) == 1:
        return reporters[0]
    return CompositeReporter(reporters)


def _import_optional_module(
    module_name: str,
    *,
    integration_name: str,
) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalIntegrationError(
            f"{integration_name} reporting was requested but its dependency "
            "is not installed."
        ) from error
