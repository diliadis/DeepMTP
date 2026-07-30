"""Two-branch model definitions and construction."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from torch import nn

from DeepMTP._registry import FactoryRegistry
from DeepMTP.config import DeepMTPConfig
from DeepMTP.data.batches import BranchInputKind
from DeepMTP.data.tabular import TabularSchema
from DeepMTP.models.branches import (
    MLP,
    ConvNet,
    IDEmbedding,
    SparseMLP,
    TabularEncoder,
)
from DeepMTP.models.contracts import (
    BranchModelInput,
    encode_branch,
    encoder_input_kind,
    encoder_output_dimension,
)

__all__ = [
    "BranchFactory",
    "FUSION_REGISTRY",
    "FusionModelType",
    "ModelBundle",
    "ModelConfigInput",
    "ModelFactory",
    "TwoBranchDotProductModel",
    "TwoBranchKroneckerModel",
    "TwoBranchMLPModel",
    "isolated_torch_seed",
]

ModelConfigInput: TypeAlias = DeepMTPConfig | Mapping[str, Any]


@dataclass(frozen=True)
class _CombinedMLPSettings:
    """Typed combined-branch options resolved from public config input."""

    nodes_per_layer: list[int] | int
    num_layers: int | None
    dropout_rate: float
    batch_norm: bool

    @classmethod
    def from_config(cls, config: ModelConfigInput) -> _CombinedMLPSettings:
        if isinstance(config, DeepMTPConfig):
            return cls(
                nodes_per_layer=config.comb_mlp_nodes_per_layer,
                num_layers=config.comb_mlp_layers,
                dropout_rate=config.dropout_rate,
                batch_norm=config.batch_norm,
            )
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or DeepMTPConfig.")
        return cls(
            nodes_per_layer=config["comb_mlp_nodes_per_layer"],
            num_layers=config["comb_mlp_layers"],
            dropout_rate=config["dropout_rate"],
            batch_norm=config["batch_norm"],
        )


def _encoder_contract(
    encoder: nn.Module,
    *,
    branch: str,
    explicit_dimension: int | None,
) -> tuple[int, BranchInputKind]:
    return (
        encoder_output_dimension(
            encoder,
            explicit_dimension=explicit_dimension,
            branch=branch,
        ),
        encoder_input_kind(encoder, branch=branch),
    )


@dataclass(frozen=True)
class _TaskOutputSettings:
    """Typed output width resolved at the legacy mapping boundary."""

    output_dimension: int

    @classmethod
    def from_config(cls, config: ModelConfigInput) -> _TaskOutputSettings:
        if isinstance(config, DeepMTPConfig):
            return cls(output_dimension=config.output_dimension)
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or DeepMTPConfig.")
        classification_mode = config.get("classification_mode")
        if classification_mode != "multiclass":
            return cls(output_dimension=1)
        num_classes = config.get("num_classes")
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes < 3
        ):
            raise ValueError(
                "num_classes must be an integer greater than or equal to 3 "
                "for multiclass classification."
            )
        return cls(output_dimension=num_classes)


class TwoBranchDotProductModel(nn.Module):
    """Combine equal-sized branch embeddings with a dot product."""

    def __init__(
        self,
        config: ModelConfigInput,
        instance_branch_model: nn.Module,
        target_branch_model: nn.Module,
        *,
        instance_output_dim: int | None = None,
        target_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.instance_branch_model = instance_branch_model
        self.target_branch_model = target_branch_model
        self.instance_output_dim, self.instance_input_kind = _encoder_contract(
            instance_branch_model,
            explicit_dimension=instance_output_dim,
            branch="instance",
        )
        self.target_output_dim, self.target_input_kind = _encoder_contract(
            target_branch_model,
            explicit_dimension=target_output_dim,
            branch="target",
        )
        if self.instance_output_dim != self.target_output_dim:
            raise ValueError(
                "Dot-product branches must declare the same output_dim; "
                f"received {self.instance_output_dim} and {self.target_output_dim}."
            )
        self.output_dim = _TaskOutputSettings.from_config(config).output_dimension
        if self.output_dim > 1:
            self.class_projection: nn.Linear | None = nn.Linear(
                self.instance_output_dim,
                self.output_dim,
            )
        else:
            self.class_projection = None

    def forward(
        self,
        instance_features: BranchModelInput,
        target_features: BranchModelInput,
    ) -> torch.Tensor:
        instance_embedding = encode_branch(
            self.instance_branch_model,
            instance_features,
            branch="instance",
            explicit_dimension=self.instance_output_dim,
        )
        target_embedding = encode_branch(
            self.target_branch_model,
            target_features,
            branch="target",
            explicit_dimension=self.target_output_dim,
        )
        interaction_embedding = instance_embedding * target_embedding
        if self.class_projection is not None:
            return self.class_projection(interaction_embedding)
        return torch.unsqueeze(interaction_embedding.sum(1), 1)


class TwoBranchMLPModel(nn.Module):
    """Combine branch embeddings using a multilayer perceptron."""

    def __init__(
        self,
        config: ModelConfigInput,
        instance_branch_model: nn.Module,
        target_branch_model: nn.Module,
        *,
        instance_output_dim: int | None = None,
        target_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.instance_branch_model = instance_branch_model
        self.target_branch_model = target_branch_model
        self.instance_output_dim, self.instance_input_kind = _encoder_contract(
            instance_branch_model,
            explicit_dimension=instance_output_dim,
            branch="instance",
        )
        self.target_output_dim, self.target_input_kind = _encoder_contract(
            target_branch_model,
            explicit_dimension=target_output_dim,
            branch="target",
        )
        combined_dimension = self.instance_output_dim + self.target_output_dim
        self.output_dim = _TaskOutputSettings.from_config(config).output_dimension
        settings = _CombinedMLPSettings.from_config(config)
        self.comb_branch = MLP(
            config,
            combined_dimension,
            self.output_dim,
            settings.nodes_per_layer,
            settings.num_layers,
            settings.dropout_rate,
            settings.batch_norm,
            skip_last_act=True,
        )

    def forward(
        self,
        instance_features: BranchModelInput,
        target_features: BranchModelInput,
    ) -> torch.Tensor:
        instance_embedding = encode_branch(
            self.instance_branch_model,
            instance_features,
            branch="instance",
            explicit_dimension=self.instance_output_dim,
        )
        target_embedding = encode_branch(
            self.target_branch_model,
            target_features,
            branch="target",
            explicit_dimension=self.target_output_dim,
        )
        combined_embedding = torch.cat(
            (instance_embedding, target_embedding),
            1,
        )
        return self.comb_branch(combined_embedding)


class TwoBranchKroneckerModel(nn.Module):
    """Combine branch embeddings using batched Kronecker products."""

    def __init__(
        self,
        config: ModelConfigInput,
        instance_branch_model: nn.Module,
        target_branch_model: nn.Module,
        *,
        instance_output_dim: int | None = None,
        target_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.instance_branch_model = instance_branch_model
        self.target_branch_model = target_branch_model
        self.instance_output_dim, self.instance_input_kind = _encoder_contract(
            instance_branch_model,
            explicit_dimension=instance_output_dim,
            branch="instance",
        )
        self.target_output_dim, self.target_input_kind = _encoder_contract(
            target_branch_model,
            explicit_dimension=target_output_dim,
            branch="target",
        )
        combined_dimension = self.instance_output_dim * self.target_output_dim
        self.output_dim = _TaskOutputSettings.from_config(config).output_dimension
        self.comb_branch = nn.Linear(combined_dimension, self.output_dim)

    def forward(
        self,
        instance_features: BranchModelInput,
        target_features: BranchModelInput,
    ) -> torch.Tensor:
        instance_embedding = encode_branch(
            self.instance_branch_model,
            instance_features,
            branch="instance",
            explicit_dimension=self.instance_output_dim,
        )
        target_embedding = encode_branch(
            self.target_branch_model,
            target_features,
            branch="target",
            explicit_dimension=self.target_output_dim,
        )
        combined_embedding = torch.stack(
            [
                torch.kron(instance_vector, target_vector)
                for instance_vector, target_vector in zip(
                    instance_embedding,
                    target_embedding,
                    strict=True,
                )
            ],
            dim=0,
        )
        return self.comb_branch(combined_embedding)


FusionModelType: TypeAlias = (
    type[TwoBranchDotProductModel]
    | type[TwoBranchMLPModel]
    | type[TwoBranchKroneckerModel]
)

FUSION_REGISTRY = FactoryRegistry[FusionModelType](
    "fusion architecture",
    {
        "dot_product": TwoBranchDotProductModel,
        "kronecker": TwoBranchKroneckerModel,
        "mlp": TwoBranchMLPModel,
    },
)


@dataclass(frozen=True)
class ModelBundle:
    """The two branches and their combined model."""

    instance_branch: nn.Module
    target_branch: nn.Module
    model: nn.Module


BranchFactory = Callable[[Mapping[str, Any]], nn.Module]


class ModelFactory:
    """Construct a complete DeepMTP model from validated configuration."""

    def __init__(
        self,
        config: DeepMTPConfig | Mapping[str, Any],
    ):
        self.config = DeepMTPConfig.from_mapping(config)
        self.legacy_config = self.config.to_dict(include_defaults=True)

    def build(
        self,
        *,
        instance_branch_factory: BranchFactory | None = None,
        target_branch_factory: BranchFactory | None = None,
    ) -> ModelBundle:
        """Build branches and their configured combination model."""
        with isolated_torch_seed(self.config.random_seed):
            instance_output_dim, target_output_dim = self._branch_output_dimensions()
            instance_branch = self._build_branch(
                "instance",
                instance_output_dim,
                instance_branch_factory,
            )
            self._ensure_encoder_contract(
                "instance",
                instance_branch,
                configured_output_dim=instance_output_dim,
            )
            target_branch = self._build_branch(
                "target",
                target_output_dim,
                target_branch_factory,
            )
            self._ensure_encoder_contract(
                "target",
                target_branch,
                configured_output_dim=target_output_dim,
            )
            model = self._build_combined_model(
                instance_branch,
                target_branch,
            )
        return ModelBundle(instance_branch, target_branch, model)

    def _branch_output_dimensions(self) -> tuple[int, int]:
        if self.config.general_architecture_version == "dot_product":
            return self.config.embedding_size, self.config.embedding_size
        return (
            self._branch_output_dimension("instance"),
            self._branch_output_dimension("target"),
        )

    def _branch_output_dimension(self, branch: str) -> int:
        if getattr(self.config, f"{branch}_branch_architecture") == "EMBEDDING":
            return self.config.embedding_size
        return self._last_node_count(
            getattr(self.config, f"{branch}_branch_nodes_per_layer")
        )

    @staticmethod
    def _last_node_count(nodes: list[int] | int) -> int:
        return nodes[-1] if isinstance(nodes, list) else nodes

    def _build_branch(
        self,
        branch: str,
        output_dim: int,
        custom_factory: BranchFactory | None,
    ) -> nn.Module:
        if custom_factory is not None:
            model = custom_factory(self.legacy_config)
            if not isinstance(model, nn.Module):
                raise TypeError(f"{branch}_branch_factory must return torch.nn.Module.")
            return model

        architecture = getattr(self.config, f"{branch}_branch_architecture")
        if architecture == "CUSTOM":
            raise ValueError(
                f"{branch}_branch_architecture='CUSTOM' requires a "
                f"{branch}_branch_factory."
            )
        if architecture == "MLP":
            return MLP(
                self.config,
                getattr(self.config, f"{branch}_branch_input_dim"),
                output_dim,
                getattr(self.config, f"{branch}_branch_nodes_per_layer"),
                getattr(self.config, f"{branch}_branch_layers"),
                getattr(self.config, f"dropout_rate_{branch}_branch"),
                self.config.batch_norm,
            )
        if architecture == "SPARSE":
            return SparseMLP(
                self.config,
                getattr(self.config, f"{branch}_branch_input_dim"),
                output_dim,
                getattr(self.config, f"{branch}_branch_nodes_per_layer"),
                getattr(self.config, f"{branch}_branch_layers"),
                getattr(self.config, f"dropout_rate_{branch}_branch"),
                self.config.batch_norm,
            )
        if architecture == "EMBEDDING":
            input_dim = getattr(self.config, f"{branch}_branch_input_dim")
            if input_dim is None:
                raise RuntimeError(
                    f"Validated {branch} embedding configuration has no input size."
                )
            return IDEmbedding(
                num_embeddings=input_dim,
                embedding_dim=output_dim,
            )
        if architecture == "TABULAR":
            schema = TabularSchema.from_config(
                getattr(self.config, f"{branch}_branch_tabular_schema"),
                branch=branch,
            )
            return TabularEncoder(
                self.config,
                schema,
                output_dim,
                getattr(self.config, f"{branch}_branch_nodes_per_layer"),
                getattr(self.config, f"{branch}_branch_layers"),
                getattr(self.config, f"dropout_rate_{branch}_branch"),
                self.config.batch_norm,
            )
        if architecture == "CONV":
            return ConvNet(
                self.config,
                getattr(self.config, f"{branch}_branch_input_dim"),
                output_dim,
                getattr(self.config, f"{branch}_branch_conv_architecture"),
                getattr(
                    self.config,
                    f"{branch}_branch_conv_architecture_version",
                ),
                getattr(
                    self.config,
                    f"{branch}_branch_conv_architecture_last_layer_trained",
                ),
                getattr(
                    self.config,
                    f"{branch}_branch_conv_architecture_dense_layers",
                ),
                pretrained=getattr(
                    self.config,
                    f"{branch}_branch_conv_pretrained",
                ),
            )
        raise ValueError(f"Unsupported {branch} branch architecture: {architecture!r}.")

    def _ensure_encoder_contract(
        self,
        branch: str,
        encoder: nn.Module,
        *,
        configured_output_dim: int,
    ) -> None:
        architecture = getattr(self.config, f"{branch}_branch_architecture")
        expected_input_kinds: dict[str, BranchInputKind] = {
            "MLP": "dense",
            "SPARSE": "sparse",
            "EMBEDDING": "id",
            "CONV": "image",
            "TABULAR": "tabular",
            "CUSTOM": "custom",
        }
        expected_input_kind = expected_input_kinds[architecture]
        declared_input_kind = getattr(encoder, "input_kind", None)
        if declared_input_kind is None:
            encoder.input_kind = expected_input_kind
        elif declared_input_kind != expected_input_kind:
            raise ValueError(
                f"{branch} branch encoder declares input_kind="
                f"{declared_input_kind!r}, but {architecture} data requires "
                f"{expected_input_kind!r}."
            )

        declared_output_dim = getattr(encoder, "output_dim", None)
        if declared_output_dim is None:
            if self.config.general_architecture_version == "dot_product":
                declared_output_dim = configured_output_dim
            else:
                declared_output_dim = _infer_legacy_encoder_output_dimension(
                    encoder,
                    branch=branch,
                )
            encoder.output_dim = declared_output_dim
        resolved_output_dim = encoder_output_dimension(encoder, branch=branch)
        if (
            self.config.general_architecture_version == "dot_product"
            and resolved_output_dim != configured_output_dim
        ):
            raise ValueError(
                f"{branch} branch encoder output_dim={resolved_output_dim} does "
                f"not match embedding_size={configured_output_dim} required by "
                "the dot-product architecture."
            )

    def _build_combined_model(
        self,
        instance_branch: nn.Module,
        target_branch: nn.Module,
    ) -> nn.Module:
        architecture = self.config.general_architecture_version
        model_type = FUSION_REGISTRY.resolve(architecture)
        return model_type(
            self.config,
            instance_branch,
            target_branch,
        )


def _infer_legacy_encoder_output_dimension(
    encoder: nn.Module,
    *,
    branch: str,
) -> int:
    linear_layers = [
        module
        for module in encoder.modules()
        if module is not encoder and isinstance(module, nn.Linear)
    ]
    if isinstance(encoder, nn.Linear):
        linear_layers.append(encoder)
    if not linear_layers:
        raise ValueError(
            f"{branch} custom branch encoder must declare a positive integer "
            "output_dim."
        )
    inferred_dimension = int(linear_layers[-1].out_features)
    warnings.warn(
        f"Inferred output_dim={inferred_dimension} from the {branch} custom "
        "branch's final Linear layer. Declare output_dim explicitly; structural "
        "inference is deprecated.",
        FutureWarning,
        stacklevel=4,
    )
    return inferred_dimension


@contextmanager
def isolated_torch_seed(seed: int | None) -> Iterator[None]:
    """Seed model initialization without changing process-wide RNG state."""
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        yield
