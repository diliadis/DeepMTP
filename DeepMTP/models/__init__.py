"""Branch models, combination architectures, and model construction."""

from DeepMTP.models.branches import MLP, ConvNet, IDEmbedding
from DeepMTP.models.contracts import (
    BranchEncoder,
    encode_branch,
    encoder_input_kind,
    encoder_output_dimension,
)
from DeepMTP.models.factory import (
    BranchFactory,
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
    "ConvNet",
    "IDEmbedding",
    "MLP",
    "ModelBundle",
    "ModelFactory",
    "TwoBranchDotProductModel",
    "TwoBranchKroneckerModel",
    "TwoBranchMLPModel",
    "encode_branch",
    "encoder_input_kind",
    "encoder_output_dimension",
    "isolated_torch_seed",
]
