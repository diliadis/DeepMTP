"""Compatibility facade for :mod:`DeepMTP.models.factory`."""

from DeepMTP.models.factory import (
    BranchFactory,
    ModelBundle,
    ModelConfigInput,
    ModelFactory,
    TwoBranchDotProductModel,
    TwoBranchKroneckerModel,
    TwoBranchMLPModel,
    isolated_torch_seed,
)

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
