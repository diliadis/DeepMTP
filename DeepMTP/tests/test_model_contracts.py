import pytest
import torch
from torch import nn

from DeepMTP.models import (
    BranchEncoder,
    TwoBranchDotProductModel,
    TwoBranchMLPModel,
    encode_branch,
)


class ContractEncoder(nn.Module):
    input_kind = "dense"
    output_dim = 2

    def forward(self, values):
        return values[:, :2].float()


def test_structural_encoder_contract_and_validated_encoding():
    encoder = ContractEncoder()
    values = torch.ones(3, 4)

    output = encode_branch(encoder, values, branch="instance")

    assert isinstance(encoder, BranchEncoder)
    assert output.shape == (3, 2)


@pytest.mark.parametrize(
    ("returned", "exception", "message"),
    [
        ([1.0], TypeError, "return a torch.Tensor"),
        (torch.ones(2), ValueError, r"\[batch, output_dim\]"),
        (torch.ones(3, 2, 1), ValueError, r"\[batch, output_dim\]"),
        (torch.ones(1, 2), ValueError, "changed the batch size"),
        (torch.ones(2, 3), ValueError, "returned width"),
        (torch.ones(2, 2, dtype=torch.int64), TypeError, "floating-point"),
    ],
)
def test_encode_branch_rejects_invalid_encoder_outputs(
    returned,
    exception,
    message,
):
    class InvalidEncoder(nn.Module):
        input_kind = "dense"
        output_dim = 2

        def forward(self, values):
            return returned

    with pytest.raises(exception, match=message):
        encode_branch(
            InvalidEncoder(),
            torch.ones(2, 4),
            branch="instance",
        )


def test_dot_product_rejects_different_declared_encoder_widths():
    instance_encoder = ContractEncoder()
    target_encoder = ContractEncoder()
    target_encoder.output_dim = 3

    with pytest.raises(ValueError, match="same output_dim"):
        TwoBranchDotProductModel({}, instance_encoder, target_encoder)


def test_combined_mlp_uses_declared_dimensions_without_module_inspection():
    model = TwoBranchMLPModel(
        {
            "comb_mlp_nodes_per_layer": [4],
            "comb_mlp_layers": None,
            "dropout_rate": 0.0,
            "batch_norm": False,
        },
        ContractEncoder(),
        ContractEncoder(),
    )

    output = model(torch.ones(3, 4), torch.ones(3, 5))

    assert output.shape == (3, 1)
