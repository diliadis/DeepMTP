from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Mapping
from types import ModuleType
from typing import Any, TypeAlias

import torch
from torch import nn

from DeepMTP.config import (
    CONVOLUTIONAL_ARCHITECTURES,
    RESNET_DENSE_LAYER_COUNTS,
    RESNET_TRAINABLE_LAYERS,
    RESNET_VERSIONS,
    DeepMTPConfig,
)

ModelConfigInput: TypeAlias = DeepMTPConfig | Mapping[str, Any]


def _load_torchvision_models() -> ModuleType:
    try:
        return importlib.import_module("torchvision.models")
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Image models require the optional 'image' dependencies. "
            "Install them with `pip install 'DeepMTP[image]'`."
        ) from exc


class MLP(nn.Sequential):
    """A standard fully connected feed-forward neural network."""

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


class ConvNet(nn.Sequential):
    """A convolutional neural network that is based on resnet."""

    def __init__(
        self,
        config: ModelConfigInput,
        input_dim: int,
        output_dim: int,
        conv_architecture: str,
        conv_architecture_version: str,
        conv_architecture_last_trained_layer: str,
        conv_architecture_dense_layers: int,
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

        self.predictor = nn.ModuleList()
        models = _load_torchvision_models()

        if conv_architecture == "resnet":
            in_size = 512
            if conv_architecture_version == "resnet18":
                self.predictor.append(models.resnet18(pretrained=True))
                in_size = 512
            elif conv_architecture_version == "resnet101":
                self.predictor.append(models.resnet101(pretrained=True))
                in_size = 2048
            else:
                raise AttributeError(
                    "The "
                    + conv_architecture_version
                    + " version of resnet is not currently supported"
                )

            for param in self.predictor.parameters():
                param.requires_grad = False

            if conv_architecture_last_trained_layer == "last":  # pragma: no cover
                pass
            elif conv_architecture_last_trained_layer == "layer4":  # pragma: no cover
                for param in self.predictor[0].avgpool.parameters():
                    param.requires_grad = True
                for param in self.predictor[0].layer4.parameters():
                    param.requires_grad = True
            elif conv_architecture_last_trained_layer == "layer3":  # pragma: no cover
                for param in self.predictor[0].avgpool.parameters():
                    param.requires_grad = True
                for param in self.predictor[0].layer4.parameters():
                    param.requires_grad = True
                for param in self.predictor[0].layer3.parameters():
                    param.requires_grad = True
            elif conv_architecture_last_trained_layer == "layer2":  # pragma: no cover
                for param in self.predictor[0].avgpool.parameters():
                    param.requires_grad = True
                for param in self.predictor[0].layer4.parameters():
                    param.requires_grad = True
                for param in self.predictor[0].layer3.parameters():
                    param.requires_grad = True
                for param in self.predictor[0].layer2.parameters():
                    param.requires_grad = True
            elif conv_architecture_last_trained_layer == "layer1":  # pragma: no cover
                for param in self.predictor[0].avgpool.parameters():
                    param.requires_grad = True
                for param in self.predictor[0].layer4.parameters():
                    param.requires_grad = True
                for param in self.predictor[0].layer3.parameters():
                    param.requires_grad = True
                for param in self.predictor[0].layer2.parameters():
                    param.requires_grad = True
                for param in self.predictor[0].layer1.parameters():
                    param.requires_grad = True
            else:
                raise AttributeError(
                    "Layer: "
                    + str(conv_architecture_last_trained_layer)
                    + " layer is not a valid "
                    + str(conv_architecture_version)
                    + " layer"
                )

            if conv_architecture_dense_layers == 1:
                fc = nn.Sequential(
                    OrderedDict(
                        [
                            ("fc1", nn.Linear(in_size, output_dim)),
                        ]
                    )
                )
            elif conv_architecture_dense_layers == 2:
                fc = nn.Sequential(
                    OrderedDict(
                        [
                            ("fc1", nn.Linear(in_size, in_size)),
                            ("relu", nn.LeakyReLU()),
                            ("fc2", nn.Linear(in_size, output_dim)),
                        ]
                    )
                )
            else:
                raise AttributeError(
                    "Only 1 or 2 dense layers are currently allowed in the "
                    + str(conv_architecture_version)
                    + " version of resnet"
                )

            self.predictor[0].fc = fc

        elif conv_architecture == "VGG":
            self.predictor.append(models.vgg11_bn(pretrained=True))
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
