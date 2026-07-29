import pytest
import torch
from torch import nn

from DeepMTP.runtime import (
    DEFAULT_LOSS_BY_CLASSIFICATION_MODE,
    DEFAULT_LOSS_BY_PROBLEM_MODE,
    DEFAULT_LOSS_INPUT_TRANSFORM_BY_PROBLEM_MODE,
    DEFAULT_OUTPUT_HEAD_BY_CLASSIFICATION_MODE,
    DEFAULT_OUTPUT_HEAD_BY_PROBLEM_MODE,
    LOSS_INPUT_TRANSFORM_BY_LOSS,
    LOSS_REGISTRY,
    OUTPUT_HEAD_REGISTRY,
    SUPPORTED_LOSSES_BY_CLASSIFICATION_MODE,
    SUPPORTED_LOSSES_BY_PROBLEM_MODE,
    build_default_loss,
    build_default_loss_input_transform,
    build_default_output_head,
    build_default_task_components,
    build_loss,
    build_loss_input_transform,
    build_output_head,
    build_task_components,
)


def test_task_component_registries_expose_behavior_preserving_defaults():
    assert LOSS_REGISTRY.names == (
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
        "cross_entropy",
        "huber",
        "mean_absolute_error",
        "mean_squared_error",
    )
    assert OUTPUT_HEAD_REGISTRY.names == ("identity", "sigmoid", "softmax")
    assert dict(DEFAULT_LOSS_BY_CLASSIFICATION_MODE) == {
        "binary": "binary_cross_entropy_with_logits",
        "multiclass": "cross_entropy",
    }
    assert dict(DEFAULT_LOSS_BY_PROBLEM_MODE) == {
        "classification": "binary_cross_entropy_with_logits",
        "regression": "mean_squared_error",
    }
    assert dict(DEFAULT_LOSS_INPUT_TRANSFORM_BY_PROBLEM_MODE) == {
        "classification": "identity",
        "regression": "identity",
    }
    assert dict(LOSS_INPUT_TRANSFORM_BY_LOSS) == {
        "binary_cross_entropy": "sigmoid",
        "binary_cross_entropy_with_logits": "identity",
        "cross_entropy": "identity",
        "huber": "identity",
        "mean_absolute_error": "identity",
        "mean_squared_error": "identity",
    }
    assert dict(SUPPORTED_LOSSES_BY_PROBLEM_MODE) == {
        "classification": frozenset(
            {
                "binary_cross_entropy",
                "binary_cross_entropy_with_logits",
                "cross_entropy",
            }
        ),
        "regression": frozenset(
            {
                "huber",
                "mean_absolute_error",
                "mean_squared_error",
            }
        ),
    }
    assert dict(DEFAULT_OUTPUT_HEAD_BY_PROBLEM_MODE) == {
        "classification": "sigmoid",
        "regression": "identity",
    }
    assert dict(DEFAULT_OUTPUT_HEAD_BY_CLASSIFICATION_MODE) == {
        "binary": "sigmoid",
        "multiclass": "softmax",
    }
    assert dict(SUPPORTED_LOSSES_BY_CLASSIFICATION_MODE) == {
        "binary": frozenset(
            {
                "binary_cross_entropy",
                "binary_cross_entropy_with_logits",
            }
        ),
        "multiclass": frozenset({"cross_entropy"}),
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
    ("loss_name", "loss_type", "input_transform_type"),
    [
        ("binary_cross_entropy", nn.BCELoss, nn.Sigmoid),
        ("binary_cross_entropy_with_logits", nn.BCEWithLogitsLoss, nn.Identity),
        ("cross_entropy", nn.CrossEntropyLoss, nn.Identity),
        ("huber", nn.HuberLoss, nn.Identity),
        ("mean_absolute_error", nn.L1Loss, nn.Identity),
        ("mean_squared_error", nn.MSELoss, nn.Identity),
    ],
)
def test_registered_losses_select_their_required_input_transform(
    loss_name,
    loss_type,
    input_transform_type,
):
    assert isinstance(build_loss(loss_name), loss_type)
    assert isinstance(build_loss_input_transform(loss_name), input_transform_type)


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


@pytest.mark.parametrize(
    ("problem_mode", "loss_name", "loss_type", "loss_input_type", "head_type"),
    [
        (
            "classification",
            "binary_cross_entropy",
            nn.BCELoss,
            nn.Sigmoid,
            nn.Sigmoid,
        ),
        (
            "classification",
            "binary_cross_entropy_with_logits",
            nn.BCEWithLogitsLoss,
            nn.Identity,
            nn.Sigmoid,
        ),
        ("regression", "mean_squared_error", nn.MSELoss, nn.Identity, nn.Identity),
        ("regression", "mean_absolute_error", nn.L1Loss, nn.Identity, nn.Identity),
        ("regression", "huber", nn.HuberLoss, nn.Identity, nn.Identity),
    ],
)
def test_task_components_follow_explicit_compatible_loss(
    problem_mode,
    loss_name,
    loss_type,
    loss_input_type,
    head_type,
):
    components = build_task_components(problem_mode, loss_name)

    assert isinstance(components.loss, loss_type)
    assert isinstance(components.loss_input_transform, loss_input_type)
    assert isinstance(components.output_head, head_type)


def test_multiclass_task_components_use_logits_loss_and_softmax_predictions():
    components = build_task_components(
        "classification",
        classification_mode="multiclass",
    )
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 1.0]])

    assert isinstance(components.loss, nn.CrossEntropyLoss)
    assert isinstance(components.loss_input_transform, nn.Identity)
    assert isinstance(components.output_head, nn.Softmax)
    assert components.output_head(logits).sum(dim=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )


@pytest.mark.parametrize(
    ("problem_mode", "loss_name"),
    [
        ("classification", "huber"),
        ("regression", "binary_cross_entropy_with_logits"),
        ("regression", ""),
    ],
)
def test_task_components_reject_incompatible_losses(problem_mode, loss_name):
    with pytest.raises(ValueError, match="incompatible"):
        build_task_components(problem_mode, loss_name)


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
    assert isinstance(
        build_default_output_head("classification", "multiclass"),
        nn.Softmax,
    )
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


def test_regression_losses_follow_documented_mean_reductions():
    predictions = torch.tensor([0.0, 2.0, -3.0])
    targets = torch.zeros(3)

    mae = build_loss("mean_absolute_error")(predictions, targets)
    huber = build_loss("huber")(predictions, targets)

    assert mae.item() == pytest.approx(5 / 3)
    assert huber.item() == pytest.approx(4 / 3)
