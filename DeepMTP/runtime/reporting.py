"""Optional experiment-reporting integrations."""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, TypeAlias

import numpy as np
import pandas as pd
from torch import nn

from DeepMTP.config import DeepMTPConfig

ReporterConfigInput: TypeAlias = DeepMTPConfig | Mapping[str, Any]

__all__ = [
    "CompositeReporter",
    "ExperimentReporter",
    "NullReporter",
    "OptionalIntegrationError",
    "ReporterConfigInput",
    "ReporterModelArtifact",
    "ReporterPredictionTable",
    "TensorBoardReporter",
    "WandBReporter",
    "build_reporter",
    "log_reporter_model_artifact",
    "log_reporter_predictions",
    "log_reporter_summary",
    "mark_reporter_failed",
    "reporter_runtime_metadata",
    "update_reporter_config",
]

_LOWER_IS_BETTER = frozenset({"hamming_loss", "loss", "mae", "mse", "rmse"})
_ARTIFACT_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ReporterModelArtifact:
    """Local files and metadata to publish as one versioned model artifact."""

    checkpoint_path: Path
    files: tuple[Path, ...]
    metadata: Mapping[str, Any]
    name: str | None = None
    aliases: tuple[str, ...] = ("latest", "best")
    registry_name: str | None = None
    registry_collection: str | None = None


@dataclass(frozen=True)
class ReporterPredictionTable:
    """Bounded prediction data and task semantics for rich reporters."""

    split: str
    frame: pd.DataFrame
    problem_mode: str
    classification_mode: str | None
    max_rows: int
    random_seed: int | None


@dataclass(frozen=True)
class _ReporterSettings:
    """Typed reporting options resolved at the compatibility boundary."""

    use_tensorboard_logger: bool
    wandb_project_name: str | None
    wandb_project_entity: str | None
    wandb_mode: str | None
    wandb_run_name: str | None
    wandb_group: str | None
    wandb_job_type: str
    wandb_tags: tuple[str, ...]
    wandb_notes: str | None
    wandb_watch: str | None
    wandb_watch_log_freq: int
    wandb_log_graph: bool
    wandb_log_code: bool
    wandb_input_artifacts: tuple[str, ...]
    metrics: tuple[str, ...]
    metrics_average: tuple[str, ...]
    config_keys: frozenset[str]

    @classmethod
    def from_config(cls, config: ReporterConfigInput) -> _ReporterSettings:
        values = (
            config.to_dict(include_defaults=True)
            if isinstance(config, DeepMTPConfig)
            else dict(config)
        )
        return cls(
            use_tensorboard_logger=bool(values.get("use_tensorboard_logger", False)),
            wandb_project_name=values.get("wandb_project_name"),
            wandb_project_entity=values.get("wandb_project_entity"),
            wandb_mode=values.get("wandb_mode"),
            wandb_run_name=values.get("wandb_run_name"),
            wandb_group=values.get("wandb_group"),
            wandb_job_type=values.get("wandb_job_type", "train"),
            wandb_tags=tuple(values.get("wandb_tags", ())),
            wandb_notes=values.get("wandb_notes"),
            wandb_watch=values.get("wandb_watch", "gradients"),
            wandb_watch_log_freq=values.get("wandb_watch_log_freq", 1000),
            wandb_log_graph=bool(values.get("wandb_log_graph", False)),
            wandb_log_code=bool(values.get("wandb_log_code", False)),
            wandb_input_artifacts=tuple(values.get("wandb_input_artifacts", ())),
            metrics=tuple(values.get("metrics", ())),
            metrics_average=tuple(values.get("metrics_average", ())),
            config_keys=frozenset(values),
        )


class OptionalIntegrationError(ImportError):
    """Raised when a requested optional reporting dependency is unavailable."""


class ExperimentReporter(Protocol):
    """Backward-compatible lifecycle used to report experiment information."""

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

    def update_config(self, config: Mapping[str, Any]) -> None:
        return None

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        return None

    def log_summary(self, values: Mapping[str, Any]) -> None:
        return None

    def log_predictions(self, report: ReporterPredictionTable) -> None:
        return None

    def log_model_artifact(
        self,
        artifact: ReporterModelArtifact,
    ) -> Mapping[str, Any] | None:
        return None

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {}

    def mark_failed(self, error: BaseException) -> None:
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

    def update_config(self, config: Mapping[str, Any]) -> None:
        for reporter in self._reporters:
            update_reporter_config(reporter, config)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        for reporter in self._reporters:
            reporter.log_metrics(metrics, step=step)

    def log_summary(self, values: Mapping[str, Any]) -> None:
        for reporter in self._reporters:
            log_reporter_summary(reporter, values)

    def log_predictions(self, report: ReporterPredictionTable) -> None:
        for reporter in self._reporters:
            log_reporter_predictions(reporter, report)

    def log_model_artifact(
        self,
        artifact: ReporterModelArtifact,
    ) -> Mapping[str, Any] | None:
        artifact_metadata: dict[str, Any] = {}
        for reporter in self._reporters:
            logged = log_reporter_model_artifact(reporter, artifact)
            if logged:
                artifact_metadata.update(logged)
        return artifact_metadata or None

    def runtime_metadata(self) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {}
        for reporter in self._reporters:
            metadata.update(reporter_runtime_metadata(reporter))
        return metadata

    def mark_failed(self, error: BaseException) -> None:
        for reporter in self._reporters:
            mark_reporter_failed(reporter, error)

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
        self.update_config(config)

    def update_config(self, config: Mapping[str, Any]) -> None:
        if self._writer is None:
            raise RuntimeError("TensorBoardReporter must be started before use.")
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

    def log_summary(self, values: Mapping[str, Any]) -> None:
        if self._writer is None:
            raise RuntimeError("TensorBoardReporter must be started before use.")
        for key, value in values.items():
            self._writer.add_text(f"summary/{key}", str(value))

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {"tensorboard": {"log_dir": str(self.log_dir)}}

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


class WandBReporter:
    """Weights & Biases adapter with lazy imports and artifact support."""

    def __init__(
        self,
        project_name: str,
        project_entity: str,
        *,
        experiment_dir: str | Path | None = None,
        mode: str | None = None,
        run_name: str | None = None,
        group: str | None = None,
        job_type: str = "train",
        tags: Sequence[str] = (),
        notes: str | None = None,
        watch: str | None = "gradients",
        watch_log_freq: int = 1000,
        log_graph: bool = False,
        log_code: bool = False,
        input_artifacts: Sequence[str] = (),
    ):
        self.project_name = project_name
        self.project_entity = project_entity
        self.experiment_dir = None if experiment_dir is None else Path(experiment_dir)
        self.mode = mode
        self.run_name = run_name
        self.group = group
        self.job_type = job_type
        self.tags = tuple(tags)
        self.notes = notes
        self.watch = watch
        self.watch_log_freq = watch_log_freq
        self.log_graph = log_graph
        self.log_code = log_code
        self.input_artifacts = tuple(input_artifacts)
        self._run: Any = None
        self._wandb: Any = None
        self._model: Any = None
        self._exit_code = 0
        self._artifact_metadata: dict[str, Any] = {}

    def start(self, config: Mapping[str, Any], model: Any) -> None:
        self._wandb = _import_optional_module(
            "wandb",
            integration_name="Weights & Biases",
        )
        init_arguments: dict[str, Any] = {
            "project": self.project_name,
            "entity": self.project_entity,
            "config": _json_safe_mapping(config),
            "reinit": "create_new",
            "job_type": self.job_type,
            "tags": list(self.tags),
            "save_code": False,
        }
        optional_arguments = {
            "dir": None if self.experiment_dir is None else str(self.experiment_dir),
            "mode": self.mode,
            "name": self.run_name,
            "group": self.group,
            "notes": self.notes,
        }
        init_arguments.update(
            {
                key: value
                for key, value in optional_arguments.items()
                if value is not None
            }
        )
        self._run = self._wandb.init(**init_arguments)
        self._model = model
        settings = _ReporterSettings.from_config(config)
        self._define_metrics(settings)
        self._pin_config(settings.config_keys)
        if self.watch is not None:
            self._run.watch(
                model,
                log=self.watch,
                log_freq=self.watch_log_freq,
                log_graph=self.log_graph,
            )
        if self.log_code:
            self._run.log_code(
                root=str(Path.cwd()),
                include_fn=_include_source_file,
            )
        for artifact_reference in self.input_artifacts:
            self._run.use_artifact(artifact_reference)

        if isinstance(model, nn.Module):
            total_parameters = sum(
                parameter.numel() for parameter in model.parameters()
            )
            trainable_parameters = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            self.log_summary(
                {
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                    "status": "running",
                }
            )

    def update_config(self, config: Mapping[str, Any]) -> None:
        if self._run is None:
            raise RuntimeError("WandBReporter must be started before use.")
        self._run.config.update(
            _json_safe_mapping(config),
            allow_val_change=True,
        )

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        if self._run is None:
            raise RuntimeError("WandBReporter must be started before use.")
        payload: dict[str, Any] = dict(metrics)
        if step is not None:
            payload["epoch"] = step
        self._run.log(payload)

    def log_summary(self, values: Mapping[str, Any]) -> None:
        if self._run is None:
            raise RuntimeError("WandBReporter must be started before use.")
        for key, value in _json_safe_mapping(values).items():
            self._run.summary[key] = value

    def log_predictions(self, report: ReporterPredictionTable) -> None:
        if self._run is None or self._wandb is None:
            raise RuntimeError("WandBReporter must be started before use.")
        frame = _bounded_prediction_frame(report)
        table = self._wandb.Table(
            dataframe=frame,
            log_mode="IMMUTABLE",
        )
        payload: dict[str, Any] = {f"{report.split}_predictions": table}
        if report.problem_mode == "classification":
            payload.update(self._classification_charts(frame, report))
        self._run.log(payload)

    def log_model_artifact(
        self,
        artifact: ReporterModelArtifact,
    ) -> Mapping[str, Any] | None:
        if self._run is None or self._wandb is None:
            raise RuntimeError("WandBReporter must be started before use.")
        artifact_name = artifact.name or _default_artifact_name(
            self.experiment_dir,
            getattr(self._run, "id", None),
        )
        wandb_artifact = self._wandb.Artifact(
            name=artifact_name,
            type="model",
            description="DeepMTP reproducible model checkpoint",
            metadata=_json_safe_mapping(artifact.metadata),
        )
        seen_paths: set[Path] = set()
        seen_names: set[str] = set()
        for path in (artifact.checkpoint_path, *artifact.files):
            normalized_path = path.resolve()
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"W&B model artifact file does not exist: {path}"
                )
            artifact_path = path.name
            if artifact_path in seen_names:
                artifact_path = f"{path.parent.name}-{path.name}"
            seen_names.add(artifact_path)
            wandb_artifact.add_file(str(path), name=artifact_path)
        logged_artifact = self._run.log_artifact(
            wandb_artifact,
            aliases=list(artifact.aliases),
        )
        if (
            artifact.registry_name is not None
            and artifact.registry_collection is not None
        ):
            self._run.link_artifact(
                logged_artifact,
                target_path=(
                    f"wandb-registry-{artifact.registry_name}/"
                    f"{artifact.registry_collection}"
                ),
                aliases=list(artifact.aliases),
            )
        self._artifact_metadata = {
            "name": getattr(logged_artifact, "name", artifact_name),
            "digest": getattr(logged_artifact, "digest", None),
            "aliases": list(artifact.aliases),
        }
        return {"wandb_model_artifact": dict(self._artifact_metadata)}

    def runtime_metadata(self) -> Mapping[str, Any]:
        if self._run is None:
            return {}
        return {
            "wandb": {
                "run_id": getattr(self._run, "id", None),
                "run_name": getattr(self._run, "name", None),
                "url": getattr(self._run, "url", None),
                "project": getattr(self._run, "project", self.project_name),
                "entity": getattr(self._run, "entity", self.project_entity),
                "mode": self.mode,
                **self._artifact_metadata,
            }
        }

    def mark_failed(self, error: BaseException) -> None:
        self._exit_code = 1
        if self._run is not None:
            self.log_summary(
                {
                    "status": "failed",
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                }
            )

    def close(self) -> None:
        if self._run is not None:
            if self._exit_code == 0:
                self.log_summary({"status": "completed"})
            if self.watch is not None and hasattr(self._run, "unwatch"):
                self._run.unwatch(self._model)
            self._run.finish(exit_code=self._exit_code)
            self._run = None
            self._model = None

    def _define_metrics(self, settings: _ReporterSettings) -> None:
        self._run.define_metric("epoch")
        for split in ("train", "val"):
            self._run.define_metric(
                f"{split}_loss",
                step_metric="epoch",
                summary="min",
            )
            for metric in settings.metrics:
                summary = "min" if str(metric).lower() in _LOWER_IS_BETTER else "max"
                for average in settings.metrics_average:
                    self._run.define_metric(
                        f"{split}_{metric}_{average}",
                        step_metric="epoch",
                        summary=summary,
                    )

    def _pin_config(self, config_keys: frozenset[str]) -> None:
        if not hasattr(self._run, "pin_config_keys"):
            return
        keys = [
            key
            for key in (
                "validation_setting",
                "problem_mode",
                "general_architecture_version",
                "instance_branch_architecture",
                "target_branch_architecture",
                "loss",
                "random_seed",
            )
            if key in config_keys
        ]
        self._run.pin_config_keys(keys)

    def _classification_charts(
        self,
        frame: pd.DataFrame,
        report: ReporterPredictionTable,
    ) -> dict[str, Any]:
        true_values = frame["true_values"].to_numpy(dtype=int)
        if report.classification_mode == "multiclass":
            probability_columns = sorted(
                (
                    name
                    for name in frame.columns
                    if name.startswith("probability_class_")
                ),
                key=lambda name: int(name.rsplit("_", maxsplit=1)[1]),
            )
            if not probability_columns:
                return {}
            probabilities = frame.loc[:, probability_columns].to_numpy(dtype=float)
            predicted = frame["predicted_values"].to_numpy(dtype=int)
            class_names = [str(index) for index in range(probabilities.shape[1])]
        else:
            positive_probabilities = frame["predicted_values"].to_numpy(dtype=float)
            probabilities = np.column_stack(
                [1.0 - positive_probabilities, positive_probabilities]
            )
            predicted = (positive_probabilities >= 0.5).astype(int)
            class_names = ["0", "1"]

        charts = {
            f"{report.split}_confusion_matrix": self._wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_values,
                preds=predicted,
                class_names=class_names,
            )
        }
        if set(true_values.tolist()) != set(range(len(class_names))):
            return charts
        charts[f"{report.split}_roc_curve"] = self._wandb.plot.roc_curve(
            y_true=true_values,
            y_probas=probabilities,
            labels=class_names,
        )
        charts[f"{report.split}_pr_curve"] = self._wandb.plot.pr_curve(
            y_true=true_values,
            y_probas=probabilities,
            labels=class_names,
        )
        return charts


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
                experiment_dir=experiment_dir,
                mode=settings.wandb_mode,
                run_name=settings.wandb_run_name,
                group=settings.wandb_group,
                job_type=settings.wandb_job_type,
                tags=settings.wandb_tags,
                notes=settings.wandb_notes,
                watch=settings.wandb_watch,
                watch_log_freq=settings.wandb_watch_log_freq,
                log_graph=settings.wandb_log_graph,
                log_code=settings.wandb_log_code,
                input_artifacts=settings.wandb_input_artifacts,
            )
        )

    if not reporters:
        return NullReporter()
    if len(reporters) == 1:
        return reporters[0]
    return CompositeReporter(reporters)


def update_reporter_config(
    reporter: ExperimentReporter,
    config: Mapping[str, Any],
) -> None:
    method = getattr(reporter, "update_config", None)
    if callable(method):
        method(config)


def log_reporter_summary(
    reporter: ExperimentReporter,
    values: Mapping[str, Any],
) -> None:
    method = getattr(reporter, "log_summary", None)
    if callable(method):
        method(values)


def log_reporter_predictions(
    reporter: ExperimentReporter,
    report: ReporterPredictionTable,
) -> None:
    method = getattr(reporter, "log_predictions", None)
    if callable(method):
        method(report)


def log_reporter_model_artifact(
    reporter: ExperimentReporter,
    artifact: ReporterModelArtifact,
) -> Mapping[str, Any] | None:
    method = getattr(reporter, "log_model_artifact", None)
    if not callable(method):
        return None
    result = method(artifact)
    return result if isinstance(result, Mapping) else None


def reporter_runtime_metadata(
    reporter: ExperimentReporter,
) -> dict[str, Any]:
    method = getattr(reporter, "runtime_metadata", None)
    if not callable(method):
        return {}
    value = method()
    return dict(value) if isinstance(value, Mapping) else {}


def mark_reporter_failed(
    reporter: ExperimentReporter,
    error: BaseException,
) -> None:
    method = getattr(reporter, "mark_failed", None)
    if callable(method):
        method(error)


def _bounded_prediction_frame(report: ReporterPredictionTable) -> pd.DataFrame:
    frame = report.frame.copy(deep=True)
    if len(frame) <= report.max_rows:
        return frame
    sampled = frame.sample(
        n=report.max_rows,
        random_state=0 if report.random_seed is None else report.random_seed,
    )
    return sampled.sort_index()


def _default_artifact_name(
    experiment_dir: Path | None,
    run_id: object,
) -> str:
    candidate = (
        experiment_dir.name
        if experiment_dir is not None
        else f"deepmtp-{run_id or 'model'}"
    )
    normalized = _ARTIFACT_NAME_PATTERN.sub("-", candidate).strip("-.")
    return normalized or f"deepmtp-{run_id or 'model'}"


def _json_safe_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in values.items()}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else str(numeric)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{name}"
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _include_source_file(path: str, root: str | None = None) -> bool:
    relative_path = Path(path)
    if any(part.startswith(".") for part in relative_path.parts):
        return False
    return relative_path.suffix.lower() in {
        ".py",
        ".toml",
        ".md",
        ".rst",
        ".yml",
        ".yaml",
    }


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
            "is not installed. Install DeepMTP[tracking]."
        ) from error
