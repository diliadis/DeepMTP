"""Typed configuration and validation for DeepMTP."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

ValidationSetting: TypeAlias = Literal["A", "B", "C", "D"]
ProblemMode: TypeAlias = Literal["classification", "regression"]
ClassificationMode: TypeAlias = Literal["binary", "multiclass"]
MulticlassAverage: TypeAlias = Literal["micro", "macro", "weighted"]
LossName: TypeAlias = Literal[
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "mean_squared_error",
    "mean_absolute_error",
    "huber",
]
GeneralArchitecture: TypeAlias = Literal["dot_product", "mlp", "kronecker"]
BranchArchitecture: TypeAlias = Literal[
    "MLP",
    "SPARSE",
    "CONV",
    "COMPOSITE",
    "CUSTOM",
    "EMBEDDING",
    "GRAPH",
    "SEQUENCE",
    "TABULAR",
]
CompositeFusion: TypeAlias = Literal["attention", "concat", "gated"]
WandBMode: TypeAlias = Literal["online", "offline", "disabled"]
WandBWatch: TypeAlias = Literal["gradients", "parameters", "all"]

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
_WANDB_ARTIFACT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

DEFAULT_LOSS_BY_PROBLEM_MODE: Mapping[str, str] = MappingProxyType(
    {
        "classification": "binary_cross_entropy_with_logits",
        "regression": "mean_squared_error",
    }
)
DEFAULT_LOSS_BY_CLASSIFICATION_MODE: Mapping[str, str] = MappingProxyType(
    {
        "binary": "binary_cross_entropy_with_logits",
        "multiclass": "cross_entropy",
    }
)
SUPPORTED_LOSSES_BY_CLASSIFICATION_MODE: Mapping[str, frozenset[str]] = (
    MappingProxyType(
        {
            "binary": frozenset(
                {
                    "binary_cross_entropy",
                    "binary_cross_entropy_with_logits",
                }
            ),
            "multiclass": frozenset({"cross_entropy"}),
        }
    )
)
SUPPORTED_LOSSES_BY_PROBLEM_MODE: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "classification": frozenset(
            {
                "binary_cross_entropy",
                "binary_cross_entropy_with_logits",
                "cross_entropy",
            }
        ),
        "regression": frozenset(
            {
                "huber",
                "mean_absolute_error",
                "mean_squared_error",
            }
        ),
    }
)


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
    loss: LossName | None = None

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
    wandb_mode: WandBMode | str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "train"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str | None = None
    wandb_watch: WandBWatch | str | None = "gradients"
    wandb_watch_log_freq: int = 1000
    wandb_log_graph: bool = False
    wandb_log_code: bool = False
    wandb_log_model_artifact: bool = False
    wandb_model_artifact_name: str | None = None
    wandb_model_artifact_aliases: list[str] = field(
        default_factory=lambda: ["latest", "best"]
    )
    wandb_input_artifacts: list[str] = field(default_factory=list)
    wandb_registry_name: str | None = None
    wandb_registry_collection: str | None = None
    wandb_log_predictions: bool = False
    wandb_prediction_table_max_rows: int = 1000
    results_path: str = "./results/"
    experiment_name: str | None = None
    experiment_dir: str | None = None
    save_model: bool = True
    data_preparation_state: dict[str, Any] | None = None

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
    instance_branch_composite_components: dict[str, dict[str, Any]] | None = None
    instance_branch_composite_fusion: CompositeFusion | str = "concat"
    instance_branch_composite_attention_dim: int = 64
    instance_branch_composite_attention_num_heads: int = 4
    instance_branch_composite_modality_dropout: float = 0.0
    instance_branch_sequence_encoder: str = "gru"
    instance_branch_sequence_embedding_dim: int = 64
    instance_branch_sequence_output_dim: int = 64
    instance_branch_sequence_conv_kernel_size: int = 3
    instance_branch_sequence_transformer_num_heads: int = 4
    instance_branch_sequence_transformer_feedforward_dim: int = 256
    instance_branch_sequence_num_layers: int = 1
    instance_branch_sequence_padding_idx: int = 0
    instance_branch_graph_edge_dim: int | None = None
    instance_branch_graph_hidden_dim: int = 64
    instance_branch_graph_output_dim: int = 64
    instance_branch_graph_num_layers: int = 2
    instance_branch_graph_pooling: str = "mean"
    instance_branch_graph_use_edge_features: bool = True

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
    target_branch_composite_components: dict[str, dict[str, Any]] | None = None
    target_branch_composite_fusion: CompositeFusion | str = "concat"
    target_branch_composite_attention_dim: int = 64
    target_branch_composite_attention_num_heads: int = 4
    target_branch_composite_modality_dropout: float = 0.0
    target_branch_sequence_encoder: str = "gru"
    target_branch_sequence_embedding_dim: int = 64
    target_branch_sequence_output_dim: int = 64
    target_branch_sequence_conv_kernel_size: int = 3
    target_branch_sequence_transformer_num_heads: int = 4
    target_branch_sequence_transformer_feedforward_dim: int = 256
    target_branch_sequence_num_layers: int = 1
    target_branch_sequence_padding_idx: int = 0
    target_branch_graph_edge_dim: int | None = None
    target_branch_graph_hidden_dim: int = 64
    target_branch_graph_output_dim: int = 64
    target_branch_graph_num_layers: int = 2
    target_branch_graph_pooling: str = "mean"
    target_branch_graph_use_edge_features: bool = True

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
    classification_mode: ClassificationMode | None = None
    num_classes: int | None = None
    multiclass_average: MulticlassAverage | None = None
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = None
    instance_branch_foundation_model: dict[str, Any] | None = None
    target_branch_foundation_model: dict[str, Any] | None = None

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

    @property
    def output_dimension(self) -> int:
        """Number of raw scores produced for each interaction."""
        if self.classification_mode == "multiclass":
            if self.num_classes is None:  # pragma: no cover - validated invariant
                raise RuntimeError("Multiclass configuration has no class count.")
            return self.num_classes
        return 1

    def _normalize(self) -> None:
        if isinstance(self.validation_setting, str):
            self.validation_setting = cast(
                ValidationSetting,
                self.validation_setting.upper(),
            )
        if isinstance(self.problem_mode, str):
            self.problem_mode = cast(ProblemMode, self.problem_mode.lower())
        if isinstance(self.classification_mode, str):
            self.classification_mode = cast(
                ClassificationMode,
                self.classification_mode.lower(),
            )
        if self.problem_mode == "classification" and self.classification_mode is None:
            self.classification_mode = "binary"
        if isinstance(self.multiclass_average, str):
            self.multiclass_average = cast(
                MulticlassAverage,
                self.multiclass_average.lower(),
            )
        if self.classification_mode == "multiclass" and self.multiclass_average is None:
            self.multiclass_average = "macro"
        if isinstance(self.loss, str):
            self.loss = cast(LossName, self.loss.lower())
        if self.loss is None:
            if (
                self.problem_mode == "classification"
                and self.classification_mode is not None
            ):
                default_loss = DEFAULT_LOSS_BY_CLASSIFICATION_MODE.get(
                    self.classification_mode
                )
            elif self.problem_mode is not None:
                default_loss = DEFAULT_LOSS_BY_PROBLEM_MODE.get(self.problem_mode)
            else:
                default_loss = None
            if default_loss is not None:
                self.loss = cast(LossName, default_loss)
        if isinstance(self.general_architecture_version, str):
            self.general_architecture_version = (
                self.general_architecture_version.lower()
            )
        if isinstance(self.wandb_mode, str):
            self.wandb_mode = self.wandb_mode.lower()
        if isinstance(self.wandb_watch, str):
            self.wandb_watch = self.wandb_watch.lower()
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
        for branch in ("instance", "target"):
            composite_fusion = getattr(
                self,
                f"{branch}_branch_composite_fusion",
            )
            if isinstance(composite_fusion, str):
                setattr(
                    self,
                    f"{branch}_branch_composite_fusion",
                    composite_fusion.lower(),
                )
            sequence_encoder = getattr(
                self,
                f"{branch}_branch_sequence_encoder",
            )
            if isinstance(sequence_encoder, str):
                setattr(
                    self,
                    f"{branch}_branch_sequence_encoder",
                    sequence_encoder.lower(),
                )
            graph_pooling = getattr(
                self,
                f"{branch}_branch_graph_pooling",
            )
            if isinstance(graph_pooling, str):
                setattr(
                    self,
                    f"{branch}_branch_graph_pooling",
                    graph_pooling.lower(),
                )
            composite_components = getattr(
                self,
                f"{branch}_branch_composite_components",
            )
            if isinstance(composite_components, Mapping):
                normalized_components: dict[str, Any] = {}
                for name, raw_component in composite_components.items():
                    component = (
                        dict(raw_component)
                        if isinstance(raw_component, Mapping)
                        else raw_component
                    )
                    if isinstance(component, dict):
                        architecture = component.get("architecture")
                        if isinstance(architecture, str):
                            component["architecture"] = architecture.upper()
                        encoder = component.get("encoder")
                        if isinstance(encoder, str):
                            component["encoder"] = encoder.lower()
                    normalized_components[name] = component
                setattr(
                    self,
                    f"{branch}_branch_composite_components",
                    normalized_components,
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
        if self.problem_mode == "classification":
            if self.classification_mode not in {"binary", "multiclass"}:
                raise ConfigValidationError(
                    "classification_mode must be 'binary' or 'multiclass' for "
                    "classification; "
                    f"received {self.classification_mode!r}."
                )
            supported_losses = SUPPORTED_LOSSES_BY_CLASSIFICATION_MODE[
                self.classification_mode
            ]
        else:
            if self.classification_mode is not None:
                raise ConfigValidationError(
                    "classification_mode must be None for regression; "
                    f"received {self.classification_mode!r}."
                )
            supported_losses = SUPPORTED_LOSSES_BY_PROBLEM_MODE["regression"]
        if self.loss not in supported_losses:
            raise ConfigValidationError(
                f"loss={self.loss!r} is incompatible with "
                f"problem_mode={self.problem_mode!r}. Select one of "
                f"{sorted(supported_losses)}."
            )
        if self.classification_mode == "multiclass":
            if (
                isinstance(self.num_classes, bool)
                or not isinstance(self.num_classes, int)
                or self.num_classes < 3
            ):
                raise ConfigValidationError(
                    "num_classes must be an integer greater than or equal to 3 "
                    "for multiclass classification."
                )
            if self.multiclass_average not in {"micro", "macro", "weighted"}:
                raise ConfigValidationError(
                    "multiclass_average must be 'micro', 'macro', or 'weighted'; "
                    f"received {self.multiclass_average!r}."
                )
            if self.top_k is not None:
                raise ConfigValidationError(
                    "top_k is not supported for multiclass classification because "
                    "its existing meaning ranks interactions, not classes."
                )
        else:
            if self.num_classes is not None:
                raise ConfigValidationError(
                    "num_classes is only valid for multiclass classification."
                )
            if self.multiclass_average is not None:
                raise ConfigValidationError(
                    "multiclass_average is only valid for multiclass classification."
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
        self._require_positive_integer(
            "gradient_accumulation_steps",
            self.gradient_accumulation_steps,
        )
        if self.gradient_clip_norm is not None:
            self._require_positive_finite(
                "gradient_clip_norm",
                self.gradient_clip_norm,
            )
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
        self._validate_wandb()
        if self.data_preparation_state is not None:
            try:
                from DeepMTP.data.preprocessing import DataPreparationState

                normalized_preparation_state = DataPreparationState.from_config(
                    self.data_preparation_state
                )
            except (TypeError, ValueError) as error:
                raise ConfigValidationError(str(error)) from error
            self.data_preparation_state = normalized_preparation_state.to_dict()

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
        if architecture not in {
            "MLP",
            "SPARSE",
            "CONV",
            "COMPOSITE",
            "CUSTOM",
            "EMBEDDING",
            "GRAPH",
            "SEQUENCE",
            "TABULAR",
        }:
            raise ConfigValidationError(
                f"{branch}_branch_architecture must be 'MLP', 'SPARSE', "
                "'CONV', 'COMPOSITE', 'CUSTOM', 'EMBEDDING', 'GRAPH', "
                "'SEQUENCE', or 'TABULAR'; received "
                f"{architecture!r}."
            )

        if architecture in {"MLP", "SPARSE", "EMBEDDING", "GRAPH", "SEQUENCE"}:
            input_dim = getattr(self, f"{branch}_branch_input_dim")
            self._require_positive_integer(f"{branch}_branch_input_dim", input_dim)
        if architecture in {"MLP", "SPARSE", "TABULAR"}:
            nodes = getattr(self, f"{branch}_branch_nodes_per_layer")
            layers = getattr(self, f"{branch}_branch_layers")
            self._validate_nodes(f"{branch}_branch_nodes_per_layer", nodes)
            if isinstance(nodes, int):
                self._require_positive_integer(f"{branch}_branch_layers", layers)
        if architecture == "COMPOSITE":
            self._validate_composite_branch(branch)
        elif architecture == "TABULAR":
            self._validate_tabular_branch(branch)
        elif architecture == "SEQUENCE":
            self._validate_sequence_branch(branch)
        elif architecture == "GRAPH":
            self._validate_graph_branch(branch)
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

    def _validate_composite_branch(
        self,
        branch: Literal["instance", "target"],
    ) -> None:
        modality_dropout_field = f"{branch}_branch_composite_modality_dropout"
        self._require_probability(
            modality_dropout_field,
            getattr(self, modality_dropout_field),
        )
        fusion_field = f"{branch}_branch_composite_fusion"
        fusion = getattr(self, fusion_field)
        if fusion not in {"attention", "concat", "gated"}:
            raise ConfigValidationError(
                f"{fusion_field} must be 'attention', 'concat', or 'gated'; "
                f"received {fusion!r}."
            )
        if fusion == "attention":
            attention_dim_field = f"{branch}_branch_composite_attention_dim"
            attention_heads_field = f"{branch}_branch_composite_attention_num_heads"
            attention_dim = getattr(self, attention_dim_field)
            attention_heads = getattr(self, attention_heads_field)
            self._require_positive_integer(attention_dim_field, attention_dim)
            self._require_positive_integer(attention_heads_field, attention_heads)
            if attention_dim % attention_heads != 0:
                raise ConfigValidationError(
                    f"{attention_dim_field}={attention_dim} must be divisible by "
                    f"{attention_heads_field}={attention_heads}."
                )

        field_name = f"{branch}_branch_composite_components"
        raw_components = getattr(self, field_name)
        if not isinstance(raw_components, Mapping) or len(raw_components) < 2:
            raise ConfigValidationError(
                f"{field_name} must contain at least two named components."
            )

        normalized_components: dict[str, dict[str, Any]] = {}
        for name, raw_component in raw_components.items():
            if not isinstance(name, str) or not name:
                raise ConfigValidationError(
                    "Composite component names must be non-empty strings."
                )
            if not isinstance(raw_component, Mapping):
                raise ConfigValidationError(
                    f"{field_name}[{name!r}] must be a mapping."
                )
            component = dict(raw_component)
            architecture = component.get("architecture")
            if architecture not in {
                "MLP",
                "SPARSE",
                "EMBEDDING",
                "GRAPH",
                "SEQUENCE",
            }:
                raise ConfigValidationError(
                    f"{field_name}[{name!r}]['architecture'] must be 'MLP', "
                    "'SPARSE', 'EMBEDDING', 'GRAPH', or 'SEQUENCE'; received "
                    f"{architecture!r}."
                )
            input_field = f"{field_name}[{name!r}]['input_dim']"
            output_field = f"{field_name}[{name!r}]['output_dim']"
            self._require_positive_integer(input_field, component.get("input_dim"))
            self._require_positive_integer(
                output_field,
                component.get("output_dim"),
            )
            component.setdefault("optional", False)
            if not isinstance(component["optional"], bool):
                raise ConfigValidationError(
                    f"{field_name}[{name!r}]['optional'] must be a boolean."
                )
            if architecture == "EMBEDDING" and component["optional"]:
                raise ConfigValidationError(
                    f"{field_name}[{name!r}] cannot mark an EMBEDDING component "
                    "as optional because entity IDs are always available."
                )

            component.setdefault(
                "dropout_rate",
                getattr(self, f"dropout_rate_{branch}_branch"),
            )
            self._require_probability(
                f"{field_name}[{name!r}]['dropout_rate']",
                component["dropout_rate"],
            )
            if architecture in {"MLP", "SPARSE"}:
                component.setdefault("nodes_per_layer", [component["output_dim"]])
                nodes = component["nodes_per_layer"]
                self._validate_nodes(
                    f"{field_name}[{name!r}]['nodes_per_layer']",
                    nodes,
                )
                component.setdefault("num_layers", None)
                if isinstance(nodes, int):
                    self._require_positive_integer(
                        f"{field_name}[{name!r}]['num_layers']",
                        component["num_layers"],
                    )
                component.setdefault("batch_norm", self.batch_norm)
                if not isinstance(component["batch_norm"], bool):
                    raise ConfigValidationError(
                        f"{field_name}[{name!r}]['batch_norm'] must be a boolean."
                    )
                if component["optional"] and component["batch_norm"]:
                    raise ConfigValidationError(
                        f"{field_name}[{name!r}] cannot enable batch_norm when "
                        "optional because a training batch may contain only one "
                        "present row."
                    )
            elif architecture == "SEQUENCE":
                self._validate_composite_sequence_component(
                    field_name,
                    name,
                    component,
                )
            elif architecture == "GRAPH":
                self._validate_composite_graph_component(
                    field_name,
                    name,
                    component,
                )
            normalized_components[name] = component

        if all(
            component["architecture"] == "EMBEDDING"
            for component in normalized_components.values()
        ):
            raise ConfigValidationError(
                f"{field_name} must include at least one side-feature component."
            )
        setattr(self, field_name, normalized_components)

    def _validate_composite_graph_component(
        self,
        field_name: str,
        component_name: str,
        component: dict[str, Any],
    ) -> None:
        prefix = f"{field_name}[{component_name!r}]"
        component.setdefault("edge_dim", None)
        component.setdefault("hidden_dim", component["output_dim"])
        component.setdefault("num_layers", 2)
        component.setdefault("pooling", "mean")
        component.setdefault("use_edge_features", True)
        if isinstance(component["pooling"], str):
            component["pooling"] = component["pooling"].lower()
        self._require_positive_integer(
            f"{prefix}['hidden_dim']",
            component["hidden_dim"],
        )
        self._require_positive_integer(
            f"{prefix}['num_layers']",
            component["num_layers"],
        )
        if component["pooling"] not in {"max", "mean", "sum"}:
            raise ConfigValidationError(
                f"{prefix}['pooling'] must be 'max', 'mean', or 'sum'; "
                f"received {component['pooling']!r}."
            )
        if not isinstance(component["use_edge_features"], bool):
            raise ConfigValidationError(
                f"{prefix}['use_edge_features'] must be a boolean."
            )
        edge_dim = component["edge_dim"]
        if edge_dim is not None:
            self._require_positive_integer(f"{prefix}['edge_dim']", edge_dim)
        if component["use_edge_features"] and edge_dim is None:
            raise ConfigValidationError(
                f"{prefix}['edge_dim'] must be a positive integer when "
                "use_edge_features=True."
            )

    def _validate_composite_sequence_component(
        self,
        field_name: str,
        component_name: str,
        component: dict[str, Any],
    ) -> None:
        prefix = f"{field_name}[{component_name!r}]"
        component.setdefault("encoder", "gru")
        component.setdefault("embedding_dim", 64)
        component.setdefault("num_layers", 1)
        component.setdefault("padding_idx", 0)
        component.setdefault("conv_kernel_size", 3)
        component.setdefault("transformer_num_heads", 4)
        component.setdefault("transformer_feedforward_dim", 256)
        component.setdefault("foundation_model", None)
        if component["encoder"] not in {
            "conv1d",
            "foundation",
            "gru",
            "transformer",
        }:
            raise ConfigValidationError(
                f"{prefix}['encoder'] must be 'conv1d', 'foundation', 'gru', "
                f"or 'transformer'; received {component['encoder']!r}."
            )
        for key in (
            "embedding_dim",
            "num_layers",
        ):
            self._require_positive_integer(
                f"{prefix}[{key!r}]",
                component[key],
            )
        self._require_non_negative_integer(
            f"{prefix}['padding_idx']",
            component["padding_idx"],
        )
        if component["padding_idx"] >= component["input_dim"]:
            raise ConfigValidationError(
                f"{prefix}['padding_idx'] must be smaller than "
                f"input_dim={component['input_dim']}."
            )
        if component["encoder"] == "conv1d":
            self._require_positive_integer(
                f"{prefix}['conv_kernel_size']",
                component["conv_kernel_size"],
            )
            if component["conv_kernel_size"] % 2 == 0:
                raise ConfigValidationError(
                    f"{prefix}['conv_kernel_size'] must be odd."
                )
        if component["encoder"] == "transformer":
            self._require_positive_integer(
                f"{prefix}['transformer_num_heads']",
                component["transformer_num_heads"],
            )
            self._require_positive_integer(
                f"{prefix}['transformer_feedforward_dim']",
                component["transformer_feedforward_dim"],
            )
            if component["embedding_dim"] % component["transformer_num_heads"] != 0:
                raise ConfigValidationError(
                    f"{prefix}['embedding_dim'] must be divisible by "
                    "transformer_num_heads."
                )
        if component["encoder"] == "foundation":
            from DeepMTP.integrations.huggingface import FoundationModelSpec

            try:
                foundation_spec = FoundationModelSpec.from_config(
                    component["foundation_model"]
                )
            except (TypeError, ValueError) as error:
                raise ConfigValidationError(
                    f"{prefix}['foundation_model']: {error}"
                ) from error
            component["foundation_model"] = foundation_spec.to_dict()

    def _validate_sequence_branch(
        self,
        branch: Literal["instance", "target"],
    ) -> None:
        encoder = getattr(self, f"{branch}_branch_sequence_encoder")
        if encoder not in {"conv1d", "foundation", "gru", "transformer"}:
            raise ConfigValidationError(
                f"{branch}_branch_sequence_encoder must be 'conv1d', "
                f"'foundation', 'gru', or 'transformer'; received {encoder!r}."
            )
        for suffix in (
            "embedding_dim",
            "output_dim",
            "num_layers",
        ):
            field_name = f"{branch}_branch_sequence_{suffix}"
            self._require_positive_integer(field_name, getattr(self, field_name))

        padding_field = f"{branch}_branch_sequence_padding_idx"
        padding_idx = getattr(self, padding_field)
        self._require_non_negative_integer(padding_field, padding_idx)
        vocabulary_size = getattr(self, f"{branch}_branch_input_dim")
        if padding_idx >= vocabulary_size:
            raise ConfigValidationError(
                f"{padding_field} must be smaller than "
                f"{branch}_branch_input_dim={vocabulary_size}; received "
                f"{padding_idx}."
            )
        if encoder == "conv1d":
            kernel_field = f"{branch}_branch_sequence_conv_kernel_size"
            kernel_size = getattr(self, kernel_field)
            self._require_positive_integer(kernel_field, kernel_size)
            if kernel_size % 2 == 0:
                raise ConfigValidationError(
                    f"{kernel_field} must be odd so temporal convolutions "
                    f"preserve sequence length; received {kernel_size}."
                )
        if encoder == "transformer":
            heads_field = f"{branch}_branch_sequence_transformer_num_heads"
            feedforward_field = f"{branch}_branch_sequence_transformer_feedforward_dim"
            num_heads = getattr(self, heads_field)
            self._require_positive_integer(heads_field, num_heads)
            self._require_positive_integer(
                feedforward_field,
                getattr(self, feedforward_field),
            )
            embedding_dim = getattr(
                self,
                f"{branch}_branch_sequence_embedding_dim",
            )
            if embedding_dim % num_heads != 0:
                raise ConfigValidationError(
                    f"{branch}_branch_sequence_embedding_dim={embedding_dim} "
                    f"must be divisible by {heads_field}={num_heads}."
                )
        if encoder == "foundation":
            from DeepMTP.integrations.huggingface import FoundationModelSpec

            foundation_field = f"{branch}_branch_foundation_model"
            try:
                foundation_spec = FoundationModelSpec.from_config(
                    getattr(self, foundation_field)
                )
            except (TypeError, ValueError) as error:
                raise ConfigValidationError(f"{foundation_field}: {error}") from error
            setattr(self, foundation_field, foundation_spec.to_dict())

    def _validate_graph_branch(
        self,
        branch: Literal["instance", "target"],
    ) -> None:
        for suffix in ("hidden_dim", "output_dim", "num_layers"):
            field_name = f"{branch}_branch_graph_{suffix}"
            self._require_positive_integer(field_name, getattr(self, field_name))
        pooling_field = f"{branch}_branch_graph_pooling"
        pooling = getattr(self, pooling_field)
        if pooling not in {"max", "mean", "sum"}:
            raise ConfigValidationError(
                f"{pooling_field} must be 'max', 'mean', or 'sum'; "
                f"received {pooling!r}."
            )
        use_edges_field = f"{branch}_branch_graph_use_edge_features"
        use_edge_features = getattr(self, use_edges_field)
        if not isinstance(use_edge_features, bool):
            raise ConfigValidationError(f"{use_edges_field} must be a boolean.")
        edge_dim_field = f"{branch}_branch_graph_edge_dim"
        edge_dim = getattr(self, edge_dim_field)
        if edge_dim is not None:
            self._require_positive_integer(edge_dim_field, edge_dim)
        if use_edge_features and edge_dim is None:
            raise ConfigValidationError(
                f"{edge_dim_field} must be a positive integer when "
                f"{use_edges_field}=True."
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
            composite_components = getattr(
                self,
                f"{branch}_branch_composite_components",
            )
            uses_id_embedding = architecture == "EMBEDDING" or (
                architecture == "COMPOSITE"
                and isinstance(composite_components, Mapping)
                and any(
                    isinstance(component, Mapping)
                    and component.get("architecture") == "EMBEDDING"
                    for component in composite_components.values()
                )
            )
            if uses_id_embedding and self.validation_setting in unsupported_settings:
                entity = "instances" if branch == "instance" else "targets"
                embedding_description = (
                    f"{branch}_branch_architecture='EMBEDDING'"
                    if architecture == "EMBEDDING"
                    else f"{branch} composite ID-embedding components"
                )
                raise ConfigValidationError(
                    f"{embedding_description} cannot represent novel {entity} "
                    "in validation setting "
                    f"{self.validation_setting}. Use side features with an MLP, "
                    "SPARSE, GRAPH, SEQUENCE, TABULAR, CONV, or CUSTOM branch."
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

    def _validate_wandb(self) -> None:
        """Validate optional W&B tracking, artifact, and Registry settings."""
        if self.wandb_mode not in {None, "online", "offline", "disabled"}:
            raise ConfigValidationError(
                "wandb_mode must be 'online', 'offline', 'disabled', or None."
            )
        if self.wandb_watch not in {None, "gradients", "parameters", "all"}:
            raise ConfigValidationError(
                "wandb_watch must be 'gradients', 'parameters', 'all', or None."
            )
        self._require_positive_integer(
            "wandb_watch_log_freq",
            self.wandb_watch_log_freq,
        )
        self._require_positive_integer(
            "wandb_prediction_table_max_rows",
            self.wandb_prediction_table_max_rows,
        )

        for name in (
            "wandb_project_name",
            "wandb_project_entity",
            "wandb_run_name",
            "wandb_group",
            "wandb_job_type",
            "wandb_notes",
            "wandb_registry_name",
            "wandb_registry_collection",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigValidationError(
                    f"{name} must be a non-empty string or None."
                )

        self._validate_string_list("wandb_tags", self.wandb_tags)
        self._validate_string_list(
            "wandb_model_artifact_aliases",
            self.wandb_model_artifact_aliases,
            allow_empty=False,
        )
        self._validate_string_list(
            "wandb_input_artifacts",
            self.wandb_input_artifacts,
        )

        if self.wandb_model_artifact_name is not None and (
            not isinstance(self.wandb_model_artifact_name, str)
            or not _WANDB_ARTIFACT_NAME_PATTERN.fullmatch(
                self.wandb_model_artifact_name
            )
        ):
            raise ConfigValidationError(
                "wandb_model_artifact_name may contain only letters, numbers, "
                "underscores, hyphens, and dots."
            )

        registry_configured = (
            self.wandb_registry_name is not None
            or self.wandb_registry_collection is not None
        )
        if (self.wandb_registry_name is None) != (
            self.wandb_registry_collection is None
        ):
            raise ConfigValidationError(
                "wandb_registry_name and wandb_registry_collection must either "
                "both be provided or both be None."
            )
        expanded_features = (
            self.wandb_log_model_artifact
            or self.wandb_log_predictions
            or self.wandb_log_code
            or bool(self.wandb_input_artifacts)
            or registry_configured
        )
        if expanded_features and self.wandb_project_name is None:
            raise ConfigValidationError(
                "W&B artifact, prediction, code, input-lineage, and Registry "
                "features require wandb_project_name and wandb_project_entity."
            )
        if registry_configured and not self.wandb_log_model_artifact:
            raise ConfigValidationError(
                "W&B Registry publishing requires wandb_log_model_artifact=True."
            )
        if registry_configured and self.wandb_mode in {"offline", "disabled"}:
            raise ConfigValidationError("W&B Registry publishing requires online mode.")
        if self.wandb_input_artifacts and self.wandb_mode in {
            "offline",
            "disabled",
        }:
            raise ConfigValidationError(
                "W&B input artifact lineage requires online mode."
            )

        for name in (
            "wandb_log_graph",
            "wandb_log_code",
            "wandb_log_model_artifact",
            "wandb_log_predictions",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigValidationError(f"{name} must be a boolean.")

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
    def _validate_string_list(
        name: str,
        values: object,
        *,
        allow_empty: bool = True,
    ) -> None:
        if not isinstance(values, list) or (not allow_empty and not values):
            qualifier = "a list" if allow_empty else "a non-empty list"
            raise ConfigValidationError(f"{name} must be {qualifier} of strings.")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ConfigValidationError(f"{name} must contain only non-empty strings.")
        if len(values) != len(set(values)):
            raise ConfigValidationError(f"{name} must not contain duplicates.")

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
    def _require_positive_finite(name: str, value: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ConfigValidationError(
                f"{name} must be a positive finite number; received {value!r}."
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
