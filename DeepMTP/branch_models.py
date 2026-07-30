"""Compatibility facade for :mod:`DeepMTP.models.branches`."""

from DeepMTP.models.branches import (
    MLP,
    CompositeEncoder,
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
