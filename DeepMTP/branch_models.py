"""Compatibility facade for :mod:`DeepMTP.models.branches`."""

from DeepMTP.models.branches import (
    MLP,
    CompositeEncoder,
    ConvNet,
    GraphEncoder,
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
    "GraphEncoder",
    "CompositeEncoder",
    "IDEmbedding",
    "MLP",
    "ModelConfigInput",
    "SequenceConv1DEncoder",
    "SequenceGRUEncoder",
    "SequenceTransformerEncoder",
    "SparseMLP",
    "TabularEncoder",
]
