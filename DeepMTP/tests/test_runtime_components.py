import pytest
import torch
from torch import nn

from DeepMTP.runtime import (
    DEFAULT_LOSS_BY_PROBLEM_MODE,
    DEFAULT_LOSS_INPUT_TRANSFORM_BY_PROBLEM_MODE,
    DEFAULT_OUTPUT_HEAD_BY_PROBLEM_MODE,
    LOSS_REGISTRY,
    OUTPUT_HEAD_REGISTRY,
    build_default_loss,
    build_default_loss_input_transform,
    build_default_output_head,
    build_default_task_components,
    build_loss,
    build_output_head,
)


def test_task_component_registries_expose_behavior_preserving_defaults():
    assert LOSS_REGISTRY.names == (
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
        "mean_squared_error",
    )
    assert OUTPUT_HEAD_REGISTRY.names == ("identity", "sigmoid")
    assert dict(DEFAULT_LOSS_BY_PROBLEM_MODE) == {
        "classification": "binary_cross_entropy_with_logits",
        "regression": "mean_squared_error",
    }
    assert dict(DEFAULT_LOSS_INPUT_TRANSFORM_BY_PROBLEM_MODE) == {
        "classification": "identity",
        "regression": "identity",
    }
    assert dict(DEFAULT_OUTPUT_HEAD_BY_PROBLEM_MODE) == {
        "classification": "sigmoid",
        "regression": "identity",
    }


def test_component_builders_return_fresh_modules():
    first_loss = build_loss("binary_cross_entropy")
    second_loss = build_loss("binary_cross_entropy")
    first_head = build_output_head("sigmoid")
    second_head = build_output_head("sigmoid")

    assert isinstance(first_loss, nn.BCELoss)
    assert isinstance(first_head, nn.Sigmoid)
    assert first_loss is not second_loss
    assert first_head is not second_head


@pytest.mark.parametrize(
    ("problem_mode", "loss_type", "loss_input_type", "head_type"),
    [
        ("classification", nn.BCEWithLogitsLoss, nn.Identity, nn.Sigmoid),
        ("regression", nn.MSELoss, nn.Identity, nn.Identity),
    ],
)
def test_default_task_components_follow_problem_mode(
    problem_mode,
    loss_type,
    loss_input_type,
    head_type,
):
    components = build_default_task_components(problem_mode)

    assert isinstance(components.loss, loss_type)
    assert isinstance(components.loss_input_transform, loss_input_type)
    assert isinstance(components.output_head, head_type)


def test_default_task_components_reject_unknown_problem_modes():
    with pytest.raises(ValueError, match="problem_mode"):
        build_default_task_components("ranking")


def test_individual_default_builders_follow_problem_mode():
    assert isinstance(build_default_loss("classification"), nn.BCEWithLogitsLoss)
    assert isinstance(build_default_loss("regression"), nn.MSELoss)
    assert isinstance(
        build_default_loss_input_transform("classification"),
        nn.Identity,
    )
    assert isinstance(
        build_default_loss_input_transform("regression"),
        nn.Identity,
    )
    assert isinstance(build_default_output_head("classification"), nn.Sigmoid)
    assert isinstance(build_default_output_head("regression"), nn.Identity)


def test_output_heads_preserve_existing_prediction_semantics():
    values = torch.tensor([-2.0, 0.0, 2.0])

    classification = build_output_head("sigmoid")(values)
    regression = build_output_head("identity")(values)

    assert classification.tolist() == pytest.approx(torch.sigmoid(values).tolist())
    assert torch.equal(regression, values)


def test_logits_loss_matches_probability_bce_for_moderate_scores():
    logits = torch.tensor([-2.0, 0.0, 2.0])
    targets = torch.tensor([0.0, 1.0, 1.0])

    stable_loss = build_loss("binary_cross_entropy_with_logits")(logits, targets)
    probability_loss = build_loss("binary_cross_entropy")(
        torch.sigmoid(logits),
        targets,
    )

    assert stable_loss.item() == pytest.approx(probability_loss.item())
