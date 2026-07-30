"""Compatibility facade for :mod:`DeepMTP.models.branches`."""

from DeepMTP.models.branches import (
    MLP,
    ConvNet,
    IDEmbedding,
    ModelConfigInput,
    SparseMLP,
    TabularEncoder,
)

__all__ = [
    "ConvNet",
    "IDEmbedding",
    "MLP",
    "ModelConfigInput",
    "SparseMLP",
    "TabularEncoder",
]
