"""Compatibility facade for :mod:`DeepMTP.models.branches`."""

from DeepMTP.models.branches import (
    MLP,
    ConvNet,
    IDEmbedding,
    ModelConfigInput,
    TabularEncoder,
)

__all__ = [
    "ConvNet",
    "IDEmbedding",
    "MLP",
    "ModelConfigInput",
    "TabularEncoder",
]
