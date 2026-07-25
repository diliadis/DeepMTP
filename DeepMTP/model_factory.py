"""Two-branch model definitions and construction."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from DeepMTP.branch_models import MLP, ConvNet
from DeepMTP.config import DeepMTPConfig


class TwoBranchDotProductModel(nn.Module):
    """Combine equal-sized branch embeddings with a dot product."""

    def __init__(self, config, instance_branch_model, target_branch_model):
        super().__init__()
        self.instance_branch_model = instance_branch_model
        self.target_branch_model = target_branch_model

    def forward(self, instance_features, target_features):
        instance_embedding = self.instance_branch_model(instance_features)
        target_embedding = self.target_branch_model(target_features)
        return torch.unsqueeze(
            (instance_embedding * target_embedding).sum(1),
            1,
        )


class TwoBranchMLPModel(nn.Module):
    """Combine branch embeddings using a multilayer perceptron."""

    def __init__(self, config, instance_branch_model, target_branch_model):
        super().__init__()
        self.instance_branch_model = instance_branch_model
        self.target_branch_model = target_branch_model
        combined_dimension = (
            instance_branch_model[0][-2].out_features
            + target_branch_model[0][-2].out_features
        )
        self.comb_branch = MLP(
            config,
            combined_dimension,
            1,
            config["comb_mlp_nodes_per_layer"],
            config["comb_mlp_layers"],
            config["dropout_rate"],
            config["batch_norm"],
            skip_last_act=True,
        )

    def forward(self, instance_features, target_features):
        instance_embedding = self.instance_branch_model(instance_features)
        target_embedding = self.target_branch_model(target_features)
        combined_embedding = torch.cat(
            (instance_embedding, target_embedding),
            1,
        )
        return self.comb_branch(combined_embedding)


class TwoBranchKroneckerModel(nn.Module):
    """Combine branch embeddings using batched Kronecker products."""

    def __init__(self, config, instance_branch_model, target_branch_model):
        super().__init__()
        self.instance_branch_model = instance_branch_model
        self.target_branch_model = target_branch_model
        combined_dimension = (
            instance_branch_model[0][-2].out_features
            * target_branch_model[0][-2].out_features
        )
        self.comb_branch = nn.Linear(combined_dimension, 1)

    def forward(self, instance_features, target_features):
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
        self.legacy_config = self.config.to_dict()

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
            self._last_node_count(self.config.instance_branch_nodes_per_layer),
            self._last_node_count(self.config.target_branch_nodes_per_layer),
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
                self.legacy_config,
                getattr(self.config, f"{branch}_branch_input_dim"),
                output_dim,
                getattr(self.config, f"{branch}_branch_nodes_per_layer"),
                getattr(self.config, f"{branch}_branch_layers"),
                getattr(self.config, f"dropout_rate_{branch}_branch"),
                self.config.batch_norm,
            )
        if architecture == "CONV":
            return ConvNet(
                self.legacy_config,
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
        return model_type(
            self.legacy_config,
            instance_branch,
            target_branch,
        )


@contextmanager
def isolated_torch_seed(seed: int | None) -> Iterator[None]:
    """Seed model initialization without changing process-wide RNG state."""
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        yield
