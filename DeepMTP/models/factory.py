"""Two-branch model definitions and construction."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from torch import nn

from DeepMTP.config import DeepMTPConfig
from DeepMTP.models.branches import MLP, ConvNet, IDEmbedding

__all__ = [
    "BranchFactory",
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


class TwoBranchDotProductModel(nn.Module):
    """Combine equal-sized branch embeddings with a dot product."""

    def __init__(
        self,
        config: ModelConfigInput,
        instance_branch_model: nn.Module,
        target_branch_model: nn.Module,
    ) -> None:
        super().__init__()
        self.instance_branch_model = instance_branch_model
        self.target_branch_model = target_branch_model

    def forward(
        self,
        instance_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        instance_embedding = self.instance_branch_model(instance_features)
        target_embedding = self.target_branch_model(target_features)
        return torch.unsqueeze(
            (instance_embedding * target_embedding).sum(1),
            1,
        )


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
        combined_dimension = _branch_output_dimension(
            instance_branch_model,
            explicit_dimension=instance_output_dim,
            branch="instance",
        ) + _branch_output_dimension(
            target_branch_model,
            explicit_dimension=target_output_dim,
            branch="target",
        )
        settings = _CombinedMLPSettings.from_config(config)
        self.comb_branch = MLP(
            config,
            combined_dimension,
            1,
            settings.nodes_per_layer,
            settings.num_layers,
            settings.dropout_rate,
            settings.batch_norm,
            skip_last_act=True,
        )

    def forward(
        self,
        instance_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        instance_embedding = self.instance_branch_model(instance_features)
        target_embedding = self.target_branch_model(target_features)
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
        combined_dimension = _branch_output_dimension(
            instance_branch_model,
            explicit_dimension=instance_output_dim,
            branch="instance",
        ) * _branch_output_dimension(
            target_branch_model,
            explicit_dimension=target_output_dim,
            branch="target",
        )
        self.comb_branch = nn.Linear(combined_dimension, 1)

    def forward(
        self,
        instance_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        instance_embedding = self.instance_branch_model(instance_features)
        target_embedding = self.target_branch_model(target_features)
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
            target_branch = self._build_branch(
                "target",
                target_output_dim,
                target_branch_factory,
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

    def _build_combined_model(
        self,
        instance_branch: nn.Module,
        target_branch: nn.Module,
    ) -> nn.Module:
        architecture = self.config.general_architecture_version
        model_types = {
            "dot_product": TwoBranchDotProductModel,
            "mlp": TwoBranchMLPModel,
            "kronecker": TwoBranchKroneckerModel,
        }
        model_type = model_types[architecture]
        if model_type is TwoBranchDotProductModel:
            return model_type(
                self.config,
                instance_branch,
                target_branch,
            )
        return model_type(
            self.config,
            instance_branch,
            target_branch,
        )


def _branch_output_dimension(
    model: nn.Module,
    *,
    explicit_dimension: int | None,
    branch: str,
) -> int:
    """Resolve a branch output size without assuming a specific module layout."""
    if explicit_dimension is not None:
        return explicit_dimension
    output_dimension = getattr(model, "output_dim", None)
    if (
        isinstance(output_dimension, int)
        and not isinstance(output_dimension, bool)
        and output_dimension > 0
    ):
        return output_dimension
    try:
        legacy_dimension = model[0][-2].out_features
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError(
            f"Could not determine the {branch} branch output dimension. "
            "Set a positive output_dim attribute on custom branch modules."
        ) from error
    if (
        isinstance(legacy_dimension, int)
        and not isinstance(legacy_dimension, bool)
        and legacy_dimension > 0
    ):
        return legacy_dimension
    raise ValueError(f"{branch} branch output dimension must be a positive integer.")


@contextmanager
def isolated_torch_seed(seed: int | None) -> Iterator[None]:
    """Seed model initialization without changing process-wide RNG state."""
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        yield
