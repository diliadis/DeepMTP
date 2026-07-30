"""Branch models, combination architectures, and model construction."""

from DeepMTP.models.branches import MLP, ConvNet, IDEmbedding, SparseMLP, TabularEncoder
from DeepMTP.models.contracts import (
    BranchEncoder,
    BranchModelInput,
    encode_branch,
    encoder_input_kind,
    encoder_output_dimension,
)
from DeepMTP.models.factory import (
    FUSION_REGISTRY,
    BranchFactory,
    FusionModelType,
    ModelBundle,
    ModelFactory,
    TwoBranchDotProductModel,
    TwoBranchKroneckerModel,
    TwoBranchMLPModel,
    isolated_torch_seed,
)

__all__ = [
    "BranchFactory",
    "BranchEncoder",
    "BranchModelInput",
    "ConvNet",
    "FUSION_REGISTRY",
    "FusionModelType",
    "IDEmbedding",
    "MLP",
    "ModelBundle",
    "ModelFactory",
    "SparseMLP",
    "TwoBranchDotProductModel",
    "TwoBranchKroneckerModel",
    "TwoBranchMLPModel",
    "TabularEncoder",
    "encode_branch",
    "encoder_input_kind",
    "encoder_output_dimension",
    "isolated_torch_seed",
]
