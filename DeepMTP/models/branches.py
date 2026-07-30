from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Mapping
from types import ModuleType
from typing import Any, ClassVar, TypeAlias, cast

import torch
from torch import nn

from DeepMTP.config import (
    CONVOLUTIONAL_ARCHITECTURES,
    RESNET_DENSE_LAYER_COUNTS,
    RESNET_TRAINABLE_LAYERS,
    RESNET_VERSIONS,
    DeepMTPConfig,
)
from DeepMTP.data.batches import BranchInputKind, SequenceInput, TabularInput
from DeepMTP.data.tabular import TabularSchema

__all__ = [
    "ConvNet",
    "IDEmbedding",
    "MLP",
    "ModelConfigInput",
    "SequenceGRUEncoder",
    "SparseMLP",
    "TabularEncoder",
]

ModelConfigInput: TypeAlias = DeepMTPConfig | Mapping[str, Any]

_RESNET_OUTPUT_FEATURES = {
    "resnet18": 512,
    "resnet101": 2048,
}
_RESNET_TRAINABLE_MODULES = {
    "last": (),
    "layer4": ("avgpool", "layer4"),
    "layer3": ("avgpool", "layer4", "layer3"),
    "layer2": ("avgpool", "layer4", "layer3", "layer2"),
    "layer1": ("avgpool", "layer4", "layer3", "layer2", "layer1"),
}


def _load_torchvision_models() -> ModuleType:
    try:
        return importlib.import_module("torchvision.models")
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Image models require the optional 'image' dependencies. "
            "Install them with `pip install 'DeepMTP[image]'`."
        ) from exc


def _build_torchvision_backbone(
    models: ModuleType,
    *,
    architecture: str,
    version: str,
    pretrained: bool,
) -> nn.Module:
    """Construct a supported torchvision model with the modern weights API."""
    if architecture == "resnet":
        factory_name = version
        weights_name = {
            "resnet18": "ResNet18_Weights",
            "resnet101": "ResNet101_Weights",
        }[version]
    else:
        factory_name = "vgg11_bn"
        weights_name = "VGG11_BN_Weights"

    factory = getattr(models, factory_name)
    weights = getattr(models, weights_name).DEFAULT if pretrained else None
    return cast(nn.Module, factory(weights=weights))


def _configure_resnet_trainable_layers(
    backbone: nn.Module,
    earliest_trainable_layer: str,
) -> None:
    """Freeze a ResNet backbone and enable the requested trailing blocks."""
    backbone.requires_grad_(False)
    for module_name in _RESNET_TRAINABLE_MODULES[earliest_trainable_layer]:
        getattr(backbone, module_name).requires_grad_(True)


class MLP(nn.Sequential):
    """A standard fully connected feed-forward neural network."""

    input_kind: ClassVar[BranchInputKind] = "dense"

    def __init__(
        self,
        config: ModelConfigInput,
        input_dim: int,
        output_dim: int,
        nodes_per_layer: list[int] | int,
        num_layers: int | None,
        dropout_rate: float,
        batch_norm: bool,
        skip_last_act: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 (inclusive) and 1.")
        self.output_dim = output_dim

        if isinstance(nodes_per_layer, list):
            if not nodes_per_layer or any(nodes <= 0 for nodes in nodes_per_layer):
                raise ValueError("nodes_per_layer must contain positive integers.")
            layer_count = len(nodes_per_layer)
            dims = [input_dim] + nodes_per_layer + [output_dim]
        else:
            if nodes_per_layer <= 0:
                raise ValueError("nodes_per_layer must be a positive integer.")
            if num_layers is None or num_layers < 0:
                raise ValueError(
                    "num_layers must be a non-negative integer when "
                    "nodes_per_layer is an integer."
                )
            layer_count = num_layers
            dims = [input_dim] + layer_count * [nodes_per_layer] + [output_dim]
        # self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(num_layers+1)])
        self.predictor = nn.ModuleList()
        for i in range(layer_count + 1):
            self.predictor.append(nn.Linear(dims[i], dims[i + 1]))

            if i != layer_count:
                self.predictor.append(nn.LeakyReLU())
                if batch_norm:
                    # pass
                    self.predictor.append(nn.BatchNorm1d(dims[i + 1]))
                if dropout_rate != 0:
                    self.predictor.append(nn.Dropout(dropout_rate))
            else:
                if not skip_last_act:
                    self.predictor.append(nn.LeakyReLU())

    def forward(self, v: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        # predict
        v = v.float()
        for layer in self.predictor:
            v = layer(v)
        return v


class SparseMLP(nn.Module):
    """Project sparse inputs before applying a dense MLP tail."""

    input_kind: ClassVar[BranchInputKind] = "sparse"

    def __init__(
        self,
        config: ModelConfigInput,
        input_dim: int,
        output_dim: int,
        nodes_per_layer: list[int] | int,
        num_layers: int | None,
        dropout_rate: float,
        batch_norm: bool,
        skip_last_act: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 (inclusive) and 1.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        if isinstance(nodes_per_layer, list):
            if not nodes_per_layer or any(nodes <= 0 for nodes in nodes_per_layer):
                raise ValueError("nodes_per_layer must contain positive integers.")
            layer_count = len(nodes_per_layer)
            dims = [input_dim] + nodes_per_layer + [output_dim]
        else:
            if nodes_per_layer <= 0:
                raise ValueError("nodes_per_layer must be a positive integer.")
            if num_layers is None or num_layers < 0:
                raise ValueError(
                    "num_layers must be a non-negative integer when "
                    "nodes_per_layer is an integer."
                )
            layer_count = num_layers
            dims = [input_dim] + layer_count * [nodes_per_layer] + [output_dim]

        self.input_projection = nn.Linear(dims[0], dims[1])
        self.predictor = nn.ModuleList()
        self._append_post_linear_layers(
            output_width=dims[1],
            is_last=layer_count == 0,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            skip_last_act=skip_last_act,
        )
        for layer_index in range(1, layer_count + 1):
            self.predictor.append(nn.Linear(dims[layer_index], dims[layer_index + 1]))
            self._append_post_linear_layers(
                output_width=dims[layer_index + 1],
                is_last=layer_index == layer_count,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                skip_last_act=skip_last_act,
            )

    def _append_post_linear_layers(
        self,
        *,
        output_width: int,
        is_last: bool,
        dropout_rate: float,
        batch_norm: bool,
        skip_last_act: bool,
    ) -> None:
        if not is_last:
            self.predictor.append(nn.LeakyReLU())
            if batch_norm:
                self.predictor.append(nn.BatchNorm1d(output_width))
            if dropout_rate != 0:
                self.predictor.append(nn.Dropout(dropout_rate))
        elif not skip_last_act:
            self.predictor.append(nn.LeakyReLU())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise TypeError("SparseMLP expects a torch.Tensor.")
        if values.layout not in {torch.sparse_coo, torch.sparse_csr}:
            raise TypeError("SparseMLP expects a PyTorch COO or CSR sparse tensor.")
        if values.ndim != 2:
            raise ValueError("SparseMLP expects a two-dimensional sparse batch.")
        if values.shape[1] != self.input_dim:
            raise ValueError(
                "SparseMLP input width does not match input_dim: "
                f"expected {self.input_dim}, received {values.shape[1]}."
            )

        sparse_values = values.to_sparse_coo().coalesce().float()
        projected = torch.sparse.mm(
            sparse_values,
            self.input_projection.weight.transpose(0, 1),
        )
        if self.input_projection.bias is not None:
            projected = projected + self.input_projection.bias
        for layer in self.predictor:
            projected = layer(projected)
        return projected


class IDEmbedding(nn.Module):
    """Map zero-based entity IDs to trainable dense representations."""

    input_kind: ClassVar[BranchInputKind] = "id"

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        for name, value in {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.num_embeddings = num_embeddings
        self.output_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(entity_ids, torch.Tensor):
            raise TypeError("entity_ids must be a torch.Tensor.")
        if entity_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("entity_ids must use an integer tensor dtype.")
        if entity_ids.ndim != 1:
            raise ValueError("entity_ids must be a one-dimensional batch.")
        return self.embedding(entity_ids)


class SequenceGRUEncoder(nn.Module):
    """Encode padded token sequences with an embedding layer and GRU."""

    input_kind: ClassVar[BranchInputKind] = "sequence"

    def __init__(
        self,
        *,
        vocabulary_size: int,
        embedding_dim: int,
        output_dim: int,
        num_layers: int = 1,
        padding_idx: int = 0,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in {
            "vocabulary_size": vocabulary_size,
            "embedding_dim": embedding_dim,
            "output_dim": output_dim,
            "num_layers": num_layers,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(padding_idx, bool)
            or not isinstance(padding_idx, int)
            or padding_idx < 0
            or padding_idx >= vocabulary_size
        ):
            raise ValueError(
                "padding_idx must be a non-negative integer smaller than "
                "vocabulary_size."
            )
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 (inclusive) and 1.")

        self.vocabulary_size = vocabulary_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=output_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, values: SequenceInput) -> torch.Tensor:
        if not isinstance(values, SequenceInput):
            raise TypeError("SequenceGRUEncoder expects a SequenceInput.")
        if values.padding_idx != self.padding_idx:
            raise ValueError(
                "Sequence input padding_idx does not match the encoder: "
                f"expected {self.padding_idx}, received {values.padding_idx}."
            )
        token_ids = values.token_ids.long()
        if token_ids.numel() and (
            int(token_ids.min()) < 0 or int(token_ids.max()) >= self.vocabulary_size
        ):
            raise ValueError(
                "Sequence token IDs contain an index outside the configured "
                f"vocabulary [0, {self.vocabulary_size})."
            )

        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            values.lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.output_dropout(hidden[-1])


class TabularEncoder(nn.Module):
    """Encode normalized numeric values and embedded categorical columns."""

    input_kind: ClassVar[BranchInputKind] = "tabular"

    def __init__(
        self,
        config: ModelConfigInput,
        schema: TabularSchema,
        output_dim: int,
        nodes_per_layer: list[int] | int,
        num_layers: int | None,
        dropout_rate: float,
        batch_norm: bool,
    ) -> None:
        super().__init__()
        if not isinstance(schema, TabularSchema):
            raise TypeError("schema must be a TabularSchema.")
        if isinstance(output_dim, bool) or not isinstance(output_dim, int):
            raise TypeError("output_dim must be an integer.")
        if output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")

        self.schema = schema
        self.output_dim = output_dim
        self.numeric_width = len(schema.numeric_columns)
        self.categorical_width = len(schema.categorical_columns)
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(
                num_embeddings=column.vocabulary_size,
                embedding_dim=column.embedding_dim,
            )
            for column in schema.categorical_columns
        )
        if schema.feature_gating:
            self.feature_gate = nn.Parameter(torch.zeros(schema.encoded_width))
        else:
            self.register_parameter("feature_gate", None)
        self.projection = MLP(
            config,
            schema.encoded_width,
            output_dim,
            nodes_per_layer,
            num_layers,
            dropout_rate,
            batch_norm,
        )

    def forward(self, values: TabularInput) -> torch.Tensor:
        if not isinstance(values, TabularInput):
            raise TypeError("TabularEncoder expects a TabularInput.")
        if values.numeric.shape[1] != self.numeric_width:
            raise ValueError(
                "Tabular numeric input width does not match the schema: "
                f"expected {self.numeric_width}, received "
                f"{values.numeric.shape[1]}."
            )
        if values.categorical.shape[1] != self.categorical_width:
            raise ValueError(
                "Tabular categorical input width does not match the schema: "
                f"expected {self.categorical_width}, received "
                f"{values.categorical.shape[1]}."
            )

        encoded_parts = []
        if self.numeric_width:
            encoded_parts.append(values.numeric.float())
        for column_index, embedding in enumerate(self.category_embeddings):
            category_ids = values.categorical[:, column_index]
            if category_ids.numel() and (
                int(category_ids.min()) < 0
                or int(category_ids.max()) >= embedding.num_embeddings
            ):
                raise ValueError(
                    f"Categorical input column {column_index} contains an index "
                    "outside its declared vocabulary."
                )
            encoded_parts.append(embedding(category_ids))

        combined = torch.cat(encoded_parts, dim=1)
        if self.feature_gate is not None:
            combined = combined * (2 * torch.sigmoid(self.feature_gate))
        return self.projection(combined)


class ConvNet(nn.Sequential):
    """A convolutional neural network that is based on resnet."""

    input_kind: ClassVar[BranchInputKind] = "image"

    def __init__(
        self,
        config: ModelConfigInput,
        input_dim: int,
        output_dim: int,
        conv_architecture: str,
        conv_architecture_version: str,
        conv_architecture_last_trained_layer: str,
        conv_architecture_dense_layers: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        if conv_architecture not in CONVOLUTIONAL_ARCHITECTURES:
            raise AttributeError(
                "Unknown name for the convolutional architecture passed. "
                f"Currently, the only options available are: {['resnet', 'VGG']}"
            )
        if conv_architecture == "resnet":
            if conv_architecture_version not in RESNET_VERSIONS:
                raise AttributeError(
                    f"{conv_architecture_version} is not a supported resnet version."
                )
            if conv_architecture_last_trained_layer not in RESNET_TRAINABLE_LAYERS:
                raise AttributeError(
                    f"{conv_architecture_last_trained_layer} is not a valid "
                    f"{conv_architecture_version} layer."
                )
            if (
                isinstance(conv_architecture_dense_layers, bool)
                or conv_architecture_dense_layers not in RESNET_DENSE_LAYER_COUNTS
            ):
                raise AttributeError(
                    "Only 1 or 2 dense layers are currently allowed in the "
                    f"{conv_architecture_version} version of resnet."
                )
        if not isinstance(pretrained, bool):
            raise AttributeError("pretrained must be a boolean.")
        self.output_dim = output_dim

        self.predictor = nn.ModuleList()
        models = _load_torchvision_models()

        if conv_architecture == "resnet":
            in_size = _RESNET_OUTPUT_FEATURES[conv_architecture_version]
            self.predictor.append(
                _build_torchvision_backbone(
                    models,
                    architecture=conv_architecture,
                    version=conv_architecture_version,
                    pretrained=pretrained,
                )
            )
            backbone = self.predictor[0]
            _configure_resnet_trainable_layers(
                backbone,
                conv_architecture_last_trained_layer,
            )

            if conv_architecture_dense_layers == 1:
                fc = nn.Sequential(
                    OrderedDict(
                        [
                            ("fc1", nn.Linear(in_size, output_dim)),
                        ]
                    )
                )
            else:
                fc = nn.Sequential(
                    OrderedDict(
                        [
                            ("fc1", nn.Linear(in_size, in_size)),
                            ("relu", nn.LeakyReLU()),
                            ("fc2", nn.Linear(in_size, output_dim)),
                        ]
                    )
                )

            backbone.fc = fc

        elif conv_architecture == "VGG":
            self.predictor.append(
                _build_torchvision_backbone(
                    models,
                    architecture=conv_architecture,
                    version=conv_architecture_version,
                    pretrained=pretrained,
                )
            )
            for param in self.predictor.parameters():
                param.requires_grad = False

            classifier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=output_dim),
                nn.ReLU(inplace=True),
            )
            self.predictor[0].classifier = classifier
        else:
            raise AttributeError(
                "Unknown name for the convolutional architecture passed. Currenty, the only options available are: "
                + str(["resnet", "VGG"])
            )

    def forward(self, v: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        # predict
        v = v.float()
        for layer in self.predictor:
            v = layer(v)
        return v
