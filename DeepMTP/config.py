"""Typed configuration and validation for DeepMTP."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal, TypeAlias, cast

ValidationSetting: TypeAlias = Literal["A", "B", "C", "D"]
ProblemMode: TypeAlias = Literal["classification", "regression"]
GeneralArchitecture: TypeAlias = Literal["dot_product", "mlp", "kronecker"]
BranchArchitecture: TypeAlias = Literal[
    "MLP",
    "CONV",
    "CUSTOM",
    "EMBEDDING",
    "TABULAR",
]

CLASSIFICATION_METRICS = frozenset(
    {
        "hamming_loss",
        "auroc",
        "f1_score",
        "aupr",
        "accuracy",
        "recall",
        "precision",
    }
)
REGRESSION_METRICS = frozenset({"RMSE", "MSE", "MAE", "R2"})
METRIC_AVERAGING_SCHEMES = frozenset({"macro", "micro", "instance"})
CONVOLUTIONAL_ARCHITECTURES = frozenset({"resnet", "VGG"})
RESNET_VERSIONS = frozenset({"resnet18", "resnet101"})
RESNET_TRAINABLE_LAYERS = frozenset({"last", "layer4", "layer3", "layer2", "layer1"})
RESNET_DENSE_LAYER_COUNTS = frozenset({1, 2})
_DEVICE_PATTERN = re.compile(r"^(cpu|cuda(?::\d+)?)$")


class ConfigValidationError(ValueError):
    """Raised when a DeepMTP configuration is internally inconsistent."""


class ConfigDeprecationWarning(FutureWarning):
    """Warns that a legacy configuration option has no runtime effect."""


class ConfigNormalizationWarning(UserWarning):
    """Warns that compatibility configuration input was normalized."""


@dataclass
class DeepMTPConfig:
    """Typed representation of a fully materialized DeepMTP configuration.

    ``generate_config`` remains the compatibility entry point and continues to
    return a dictionary. New code can construct this class directly, while old
    dictionaries can be validated with :meth:`from_mapping`.
    """

    validation_setting: ValidationSetting | None = None
    general_architecture_version: GeneralArchitecture | str = "dot_product"
    problem_mode: ProblemMode | None = None

    learning_rate: float = 0.001
    decay: float = 0.0
    batch_norm: bool = False
    dropout_rate: float = 0.0
    dropout_rate_instance_branch: float = 0.0
    dropout_rate_target_branch: float = 0.0
    momentum: float = 0.9
    weighted_loss: bool = False

    compute_mode: str = "cuda:0"
    num_workers: int = 1
    train_batchsize: int = 512
    val_batchsize: int = 512
    num_epochs: int = 100
    random_seed: int | None = 2

    metrics: list[str] = field(
        default_factory=lambda: [
            "hamming_loss",
            "auroc",
            "f1_score",
            "aupr",
            "accuracy",
            "recall",
            "precision",
        ]
    )
    metrics_average: list[str] = field(default_factory=lambda: ["macro", "micro"])
    top_k: int | None = None
    metric_to_optimize_early_stopping: str = "loss"
    metric_to_optimize_best_epoch_selection: str = "loss"

    patience: int = 10
    delta: float = 0.0
    evaluate_train: bool = False
    evaluate_val: bool = False
    eval_every_n_epochs: int = 10
    use_early_stopping: bool = True

    verbose: bool = False
    results_verbose: bool = False
    eval_instance_verbose: bool = False
    eval_target_verbose: bool = False
    return_results_per_target: bool = False

    use_tensorboard_logger: bool = False
    wandb_project_name: str | None = None
    wandb_project_entity: str | None = None
    results_path: str = "./results/"
    experiment_name: str | None = None
    experiment_dir: str | None = None
    save_model: bool = True

    instance_branch_architecture: BranchArchitecture | None = None
    use_instance_features: bool = False
    instance_branch_input_dim: int | None = None
    instance_branch_nodes_reducing_factor: int = 2
    instance_branch_nodes_per_layer: list[int] | int = field(
        default_factory=lambda: [10, 10, 10]
    )
    instance_branch_layers: int | None = None
    instance_branch_conv_architecture: str = "resnet"
    instance_branch_conv_architecture_version: str = "resnet101"
    instance_branch_conv_architecture_dense_layers: int = 1
    instance_branch_conv_architecture_last_layer_trained: str = "last"
    instance_branch_conv_pretrained: bool = True
    instance_train_transforms: Any = None
    instance_inference_transforms: Any = None
    instance_branch_tabular_schema: dict[str, Any] | None = None
    instance_tabular_preprocessing_state: dict[str, Any] | None = None

    target_branch_architecture: BranchArchitecture | None = None
    use_target_features: bool = False
    target_branch_input_dim: int | None = None
    target_branch_nodes_reducing_factor: int = 2
    target_branch_nodes_per_layer: list[int] | int = field(
        default_factory=lambda: [10, 10, 10]
    )
    target_branch_layers: int | None = None
    target_branch_conv_architecture: str = "resnet"
    target_branch_conv_architecture_version: str = "resnet101"
    target_branch_conv_architecture_dense_layers: int = 1
    target_branch_conv_architecture_last_layer_trained: str = "last"
    target_branch_conv_pretrained: bool = True
    target_train_transforms: Any = None
    target_inference_transforms: Any = None
    target_branch_tabular_schema: dict[str, Any] | None = None
    target_tabular_preprocessing_state: dict[str, Any] | None = None

    comb_mlp_nodes_reducing_factor: int = 2
    comb_mlp_nodes_per_layer: list[int] | int = field(
        default_factory=lambda: [10, 10, 10]
    )
    comb_mlp_layers: int | None = None
    embedding_size: int = 100

    load_pretrained_model: bool = False
    pretrained_model_path: str = ""
    running_hpo: bool = False
    hpo_results_path: str = "./"
    additional_info: dict[str, Any] = field(default_factory=dict)

    _extras: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _present_fields: set[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._normalize()
        self.validate()
        self._warn_about_inert_legacy_options()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | DeepMTPConfig) -> DeepMTPConfig:
        """Create a validated typed config without discarding extension keys."""
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError(
                "DeepMTP config must be a mapping or DeepMTPConfig, "
                f"not {type(config).__name__}."
            )

        known_names = {
            config_field.name
            for config_field in fields(cls)
            if not config_field.name.startswith("_")
        }
        known_values = {
            key: _copy_config_value(value)
            for key, value in config.items()
            if key in known_names
        }
        typed_config = cls(**known_values)
        typed_config._present_fields = set(known_values)
        typed_config._extras = {
            key: _copy_config_value(value)
            for key, value in config.items()
            if key not in known_names
        }
        return typed_config

    def to_dict(self, *, include_defaults: bool = False) -> dict[str, Any]:
        """Return a compatibility dictionary, including extension metadata.

        ``include_defaults`` materializes every normalized public field for
        internal runtime consumers. The default preserves the historical
        round-trip behavior for partial legacy mappings.
        """
        public_fields = [
            config_field.name
            for config_field in fields(self)
            if not config_field.name.startswith("_")
        ]
        included_fields = (
            public_fields
            if include_defaults or self._present_fields is None
            else [name for name in public_fields if name in self._present_fields]
        )
        result = {
            name: _copy_config_value(getattr(self, name)) for name in included_fields
        }
        result.update(
            {key: _copy_config_value(value) for key, value in self._extras.items()}
        )
        return result

    def set_runtime_value(self, key: str, value: Any) -> None:
        """Record metadata added while an experiment is running."""
        public_fields = {
            config_field.name
            for config_field in fields(self)
            if not config_field.name.startswith("_")
        }
        if key in public_fields:
            setattr(self, key, value)
            if self._present_fields is not None:
                self._present_fields.add(key)
        else:
            self._extras[key] = value

    def _normalize(self) -> None:
        if isinstance(self.validation_setting, str):
            self.validation_setting = cast(
                ValidationSetting,
                self.validation_setting.upper(),
            )
        if isinstance(self.problem_mode, str):
            self.problem_mode = cast(ProblemMode, self.problem_mode.lower())
        if isinstance(self.general_architecture_version, str):
            self.general_architecture_version = (
                self.general_architecture_version.lower()
            )
        if isinstance(self.instance_branch_architecture, str):
            self.instance_branch_architecture = cast(
                BranchArchitecture,
                self.instance_branch_architecture.upper(),
            )
        if isinstance(self.target_branch_architecture, str):
            self.target_branch_architecture = cast(
                BranchArchitecture,
                self.target_branch_architecture.upper(),
            )

        if self.problem_mode == "classification":
            self.metrics = [metric.lower() for metric in self.metrics]
        elif self.problem_mode == "regression":
            self.metrics = [metric.upper() for metric in self.metrics]
        self.metrics_average = [averaging.lower() for averaging in self.metrics_average]

    def validate(self) -> None:
        """Validate values needed by model construction and training."""
        if self.validation_setting not in {"A", "B", "C", "D"}:
            raise ConfigValidationError(
                "validation_setting must be one of 'A', 'B', 'C', or 'D'; "
                f"received {self.validation_setting!r}."
            )
        if self.problem_mode not in {"classification", "regression"}:
            raise ConfigValidationError(
                "problem_mode must be 'classification' or 'regression'; "
                f"received {self.problem_mode!r}."
            )
        if self.general_architecture_version not in {
            "dot_product",
            "mlp",
            "kronecker",
        }:
            raise ConfigValidationError(
                "general_architecture_version must be 'dot_product', 'mlp', "
                f"or 'kronecker'; received {self.general_architecture_version!r}."
            )

        self._validate_branch("instance")
        self._validate_branch("target")
        self._validate_embedding_novelty()
        self._validate_combined_architecture()

        self._require_positive("learning_rate", self.learning_rate)
        self._require_non_negative("decay", self.decay)
        self._require_probability("dropout_rate", self.dropout_rate)
        self._require_probability(
            "dropout_rate_instance_branch",
            self.dropout_rate_instance_branch,
        )
        self._require_probability(
            "dropout_rate_target_branch",
            self.dropout_rate_target_branch,
        )
        self._require_non_negative_integer("num_workers", self.num_workers)
        self._require_positive_integer("train_batchsize", self.train_batchsize)
        self._require_positive_integer("val_batchsize", self.val_batchsize)
        self._require_positive_integer("num_epochs", self.num_epochs)
        if self.random_seed is not None:
            self._require_non_negative_integer("random_seed", self.random_seed)
        self._require_positive_integer("eval_every_n_epochs", self.eval_every_n_epochs)
        self._require_non_negative_integer("patience", self.patience)
        self._require_positive_integer("embedding_size", self.embedding_size)
        if self.top_k is not None:
            self._require_positive_integer("top_k", self.top_k)

        if not isinstance(self.compute_mode, str) or not _DEVICE_PATTERN.fullmatch(
            self.compute_mode
        ):
            raise ConfigValidationError(
                "compute_mode must be 'cpu', 'cuda', or 'cuda:<index>'; "
                f"received {self.compute_mode!r}."
            )
        if not isinstance(self.results_path, str) or not self.results_path:
            raise ConfigValidationError("results_path must be a non-empty string.")
        if (self.wandb_project_name is None) != (self.wandb_project_entity is None):
            raise ConfigValidationError(
                "wandb_project_name and wandb_project_entity must either both "
                "be provided or both be None."
            )

        valid_metrics = (
            CLASSIFICATION_METRICS
            if self.problem_mode == "classification"
            else REGRESSION_METRICS
        )
        unknown_metrics = set(self.metrics).difference(valid_metrics)
        if unknown_metrics:
            raise ConfigValidationError(
                f"Unsupported {self.problem_mode} metrics: {sorted(unknown_metrics)}."
            )
        unknown_averaging = set(self.metrics_average).difference(
            METRIC_AVERAGING_SCHEMES
        )
        if unknown_averaging:
            raise ConfigValidationError(
                f"Unsupported metric averaging schemes: {sorted(unknown_averaging)}."
            )
        if (
            self.validation_setting == "A"
            and self.metrics
            and self.metrics_average != ["micro"]
        ):
            raise ConfigValidationError(
                "metrics_average must be ['micro'] for validation setting A."
            )
        if self.return_results_per_target and "macro" not in self.metrics_average:
            raise ConfigValidationError(
                "return_results_per_target requires 'macro' in metrics_average."
            )
        if self.top_k is not None and not {"macro", "instance"}.intersection(
            self.metrics_average
        ):
            raise ConfigValidationError(
                "top_k requires 'macro' or 'instance' in metrics_average."
            )

        selectable_metrics = {
            f"{metric}_{averaging}"
            for metric in self.metrics
            for averaging in self.metrics_average
        }
        selectable_metrics.add("loss")
        for field_name in (
            "metric_to_optimize_early_stopping",
            "metric_to_optimize_best_epoch_selection",
        ):
            value = getattr(self, field_name)
            if value not in selectable_metrics:
                raise ConfigValidationError(
                    f"{field_name}={value!r} is unavailable. Select one of "
                    f"{sorted(selectable_metrics)}."
                )

    def _validate_branch(self, branch: Literal["instance", "target"]) -> None:
        architecture = getattr(self, f"{branch}_branch_architecture")
        if architecture not in {"MLP", "CONV", "CUSTOM", "EMBEDDING", "TABULAR"}:
            raise ConfigValidationError(
                f"{branch}_branch_architecture must be 'MLP', 'CONV', "
                f"'CUSTOM', 'EMBEDDING', or 'TABULAR'; received "
                f"{architecture!r}."
            )

        if architecture in {"MLP", "EMBEDDING"}:
            input_dim = getattr(self, f"{branch}_branch_input_dim")
            self._require_positive_integer(f"{branch}_branch_input_dim", input_dim)
        if architecture in {"MLP", "TABULAR"}:
            nodes = getattr(self, f"{branch}_branch_nodes_per_layer")
            layers = getattr(self, f"{branch}_branch_layers")
            self._validate_nodes(f"{branch}_branch_nodes_per_layer", nodes)
            if isinstance(nodes, int):
                self._require_positive_integer(f"{branch}_branch_layers", layers)
        if architecture == "TABULAR":
            self._validate_tabular_branch(branch)
        elif architecture == "CONV":
            conv_architecture = getattr(
                self,
                f"{branch}_branch_conv_architecture",
            )
            if conv_architecture not in CONVOLUTIONAL_ARCHITECTURES:
                raise ConfigValidationError(
                    f"{branch}_branch_conv_architecture must be 'resnet' or "
                    f"'VGG'; received {conv_architecture!r}."
                )
            pretrained = getattr(self, f"{branch}_branch_conv_pretrained")
            if not isinstance(pretrained, bool):
                raise ConfigValidationError(
                    f"{branch}_branch_conv_pretrained must be a boolean; "
                    f"received {pretrained!r}."
                )
            if conv_architecture != "resnet":
                return
            version = getattr(
                self,
                f"{branch}_branch_conv_architecture_version",
            )
            if version not in RESNET_VERSIONS:
                raise ConfigValidationError(
                    f"{branch}_branch_conv_architecture_version must be one of "
                    f"{sorted(RESNET_VERSIONS)}; received {version!r}."
                )
            last_trained_layer = getattr(
                self,
                f"{branch}_branch_conv_architecture_last_layer_trained",
            )
            if last_trained_layer not in RESNET_TRAINABLE_LAYERS:
                raise ConfigValidationError(
                    f"{branch}_branch_conv_architecture_last_layer_trained must "
                    f"be one of {sorted(RESNET_TRAINABLE_LAYERS)}; received "
                    f"{last_trained_layer!r}."
                )
            dense_layers = getattr(
                self, f"{branch}_branch_conv_architecture_dense_layers"
            )
            if (
                isinstance(dense_layers, bool)
                or dense_layers not in RESNET_DENSE_LAYER_COUNTS
            ):
                raise ConfigValidationError(
                    f"{branch}_branch_conv_architecture_dense_layers must be 1 or 2."
                )

    def _validate_tabular_branch(
        self,
        branch: Literal["instance", "target"],
    ) -> None:
        from DeepMTP.data.tabular import (
            TabularPreprocessingState,
            TabularSchema,
        )

        schema_field = f"{branch}_branch_tabular_schema"
        try:
            schema = TabularSchema.from_config(
                getattr(self, schema_field),
                branch=branch,
            )
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(str(exc)) from exc
        setattr(self, schema_field, schema.to_dict())

        state_field = f"{branch}_tabular_preprocessing_state"
        state = getattr(self, state_field)
        if state is None:
            return
        try:
            normalized_state = TabularPreprocessingState.from_config(
                state,
                schema=schema,
                branch=branch,
            )
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(str(exc)) from exc
        setattr(self, state_field, normalized_state.to_dict())

    def _validate_embedding_novelty(self) -> None:
        novelty_settings = {
            "instance": {"B", "D"},
            "target": {"C", "D"},
        }
        for branch, unsupported_settings in novelty_settings.items():
            architecture = getattr(self, f"{branch}_branch_architecture")
            if (
                architecture == "EMBEDDING"
                and self.validation_setting in unsupported_settings
            ):
                entity = "instances" if branch == "instance" else "targets"
                raise ConfigValidationError(
                    f"{branch}_branch_architecture='EMBEDDING' cannot represent "
                    f"novel {entity} in validation setting "
                    f"{self.validation_setting}. Use side features with an MLP, "
                    "TABULAR, CONV, or CUSTOM branch."
                )

    def _validate_combined_architecture(self) -> None:
        if self.general_architecture_version != "mlp":
            return
        self._validate_nodes(
            "comb_mlp_nodes_per_layer",
            self.comb_mlp_nodes_per_layer,
        )
        if isinstance(self.comb_mlp_nodes_per_layer, int):
            self._require_positive_integer(
                "comb_mlp_layers",
                self.comb_mlp_layers,
            )

    def _warn_about_inert_legacy_options(self) -> None:
        inert_options = []
        if self.momentum != 0.9:
            inert_options.append("momentum")
        if self.weighted_loss:
            inert_options.append("weighted_loss")
        if self.use_instance_features:
            inert_options.append("use_instance_features")
        if self.use_target_features:
            inert_options.append("use_target_features")
        if self.load_pretrained_model or self.pretrained_model_path:
            inert_options.extend(["load_pretrained_model", "pretrained_model_path"])
        if (
            self.general_architecture_version == "mlp"
            and self.comb_mlp_nodes_reducing_factor != 2
        ):
            inert_options.append("comb_mlp_nodes_reducing_factor")

        if inert_options:
            warnings.warn(
                "These legacy configuration options currently have no effect "
                "and are scheduled for removal unless implemented: "
                + ", ".join(sorted(set(inert_options))),
                ConfigDeprecationWarning,
                stacklevel=3,
            )

    @staticmethod
    def _validate_nodes(name: str, nodes: list[int] | int) -> None:
        if isinstance(nodes, int) and not isinstance(nodes, bool):
            DeepMTPConfig._require_positive_integer(name, nodes)
            return
        if not isinstance(nodes, list) or not nodes:
            raise ConfigValidationError(
                f"{name} must be a positive integer or a non-empty list of "
                "positive integers."
            )
        for node_count in nodes:
            DeepMTPConfig._require_positive_integer(name, node_count)

    @staticmethod
    def _require_probability(name: str, value: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value < 1
        ):
            raise ConfigValidationError(
                f"{name} must be a number in the interval [0, 1); received {value!r}."
            )

    @staticmethod
    def _require_positive(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigValidationError(
                f"{name} must be greater than zero; received {value!r}."
            )

    @staticmethod
    def _require_non_negative(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigValidationError(
                f"{name} must be non-negative; received {value!r}."
            )

    @staticmethod
    def _require_positive_integer(name: str, value: int | None) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigValidationError(
                f"{name} must be a positive integer; received {value!r}."
            )

    @staticmethod
    def _require_non_negative_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigValidationError(
                f"{name} must be a non-negative integer; received {value!r}."
            )


def _copy_config_value(value: Any) -> Any:
    """Copy mutable containers without copying opaque user integrations."""
    if isinstance(value, dict):
        return {
            _copy_config_value(key): _copy_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_config_value(item) for item in value]
    if isinstance(value, set):
        return {_copy_config_value(item) for item in value}
    if isinstance(value, tuple):
        return tuple(_copy_config_value(item) for item in value)
    return value
