"""Compatibility facade for :mod:`DeepMTP.models.branches`."""

from DeepMTP.models.branches import (
    MLP,
    ConvNet,
    IDEmbedding,
    ModelConfigInput,
    SequenceConv1DEncoder,
    SequenceGRUEncoder,
    SequenceTransformerEncoder,
    SparseMLP,
    TabularEncoder,
)

__all__ = [
    "ConvNet",
    "IDEmbedding",
    "MLP",
    "ModelConfigInput",
    "SequenceConv1DEncoder",
    "SequenceGRUEncoder",
    "SequenceTransformerEncoder",
    "SparseMLP",
    "TabularEncoder",
]
