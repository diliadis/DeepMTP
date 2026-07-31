import math

import pytest
import torch
from torch import nn

from DeepMTP import EpochOutput, TrainingEngine
from DeepMTP.data import DenseBranchBatch, MTPBatch


class TinyTwoInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, instance_features, target_features):
        features = torch.cat((instance_features, target_features), dim=1)
        return self.linear(features)


class ScalarLogitModel(nn.Module):
    def __init__(self, initial_logit):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(float(initial_logit)))

    def forward(self, instance_features, target_features):
        return self.logit.expand(instance_features.shape[0], 1)


class MulticlassLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, instance_features, target_features):
        features = torch.cat((instance_features, target_features), dim=1)
        return self.linear(features)


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, *, lr):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def make_batch(scores=(0.0, 1.0)):
    batch_size = len(scores)
    return {
        "instance_features": torch.arange(
            1,
            batch_size + 1,
            dtype=torch.float32,
        ).reshape(-1, 1),
        "target_features": torch.arange(
            0.5,
            batch_size + 0.5,
            dtype=torch.float32,
        ).reshape(-1, 1),
        "score": torch.tensor(scores),
        "instance_id": torch.arange(10, 10 + batch_size),
        "target_id": torch.arange(20, 20 + batch_size),
    }


def slice_batch(batch, start, stop):
    return {key: value[start:stop] for key, value in batch.items()}


def make_typed_batch(scores=(0.0, 1.0)):
    legacy_batch = make_batch(scores)
    return MTPBatch(
        instance_id=legacy_batch["instance_id"],
        target_id=legacy_batch["target_id"],
        instance_input=DenseBranchBatch(legacy_batch["instance_features"]),
        target_input=DenseBranchBatch(legacy_batch["target_features"]),
        score=legacy_batch["score"],
    )


def test_epoch_output_mean_loss():
    assert EpochOutput(losses=[1.0, 2.0, 3.0]).mean_loss == 2.0
    assert EpochOutput(
        losses=[1.0, 9.0],
        loss_weights=[2, 1],
    ).mean_loss == pytest.approx(11 / 3)
    assert math.isnan(EpochOutput().mean_loss)


@pytest.mark.parametrize("epoch_method", ["train_epoch", "evaluate_epoch"])
def test_epoch_loss_weights_uneven_batches_by_observation(epoch_method):
    model = TinyTwoInputModel()
    with torch.no_grad():
        model.linear.weight.zero_()
    engine = TrainingEngine(
        model=model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
    )

    output = getattr(engine, epoch_method)(
        [
            make_batch(scores=(1.0, 1.0)),
            make_batch(scores=(3.0,)),
        ],
        collect_predictions=False,
    )

    assert output.losses == pytest.approx([1.0, 9.0])
    assert output.loss_weights == [2, 1]
    assert output.mean_loss == pytest.approx(11 / 3)


def test_classification_train_epoch_updates_model_and_collects_probabilities():
    model = TinyTwoInputModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = TrainingEngine(
        model=model,
        criterion=nn.BCELoss(),
        device=torch.device("cpu"),
        problem_mode="classification",
        optimizer=optimizer,
    )
    initial_weights = model.linear.weight.detach().clone()

    output = engine.train_epoch([make_batch()], collect_predictions=True)

    assert not torch.equal(model.linear.weight, initial_weights)
    assert len(output.losses) == 1
    assert output.true_values == [0.0, 1.0]
    assert output.instance_ids == [10, 11]
    assert output.target_ids == [20, 21]
    assert all(0.0 <= prediction <= 1.0 for prediction in output.predicted_values)


def test_training_engine_defaults_preserve_one_step_per_batch():
    model = TinyTwoInputModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    engine = TrainingEngine(
        model=model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
        optimizer=optimizer,
    )

    engine.train_epoch([make_batch(), make_batch()])

    assert engine.gradient_accumulation_steps == 1
    assert engine.gradient_clip_norm is None
    assert optimizer.step_calls == 2


def test_gradient_accumulation_matches_one_combined_uneven_batch():
    combined_model = TinyTwoInputModel()
    accumulated_model = TinyTwoInputModel()
    with torch.no_grad():
        combined_model.linear.weight.copy_(torch.tensor([[0.25, -0.5]]))
        accumulated_model.load_state_dict(combined_model.state_dict())
    combined_optimizer = torch.optim.SGD(combined_model.parameters(), lr=0.05)
    accumulated_optimizer = torch.optim.SGD(
        accumulated_model.parameters(),
        lr=0.05,
    )
    combined_engine = TrainingEngine(
        model=combined_model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
        optimizer=combined_optimizer,
    )
    accumulated_engine = TrainingEngine(
        model=accumulated_model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
        optimizer=accumulated_optimizer,
        gradient_accumulation_steps=2,
    )
    combined_batch = make_batch(scores=(1.0, -2.0, 4.0))

    combined_output = combined_engine.train_epoch([combined_batch])
    accumulated_output = accumulated_engine.train_epoch(
        [
            slice_batch(combined_batch, 0, 2),
            slice_batch(combined_batch, 2, 3),
        ]
    )

    torch.testing.assert_close(
        accumulated_model.linear.weight,
        combined_model.linear.weight,
    )
    assert accumulated_output.mean_loss == pytest.approx(combined_output.mean_loss)


def test_gradient_accumulation_steps_for_final_partial_group():
    model = TinyTwoInputModel()
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    engine = TrainingEngine(
        model=model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
        optimizer=optimizer,
        gradient_accumulation_steps=2,
    )

    engine.train_epoch([make_batch()] * 5)

    assert optimizer.step_calls == 3


def test_incomplete_accumulation_group_uses_its_actual_observations():
    default_model = TinyTwoInputModel()
    accumulated_model = TinyTwoInputModel()
    accumulated_model.load_state_dict(default_model.state_dict())
    default_engine = TrainingEngine(
        model=default_model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
        optimizer=torch.optim.SGD(default_model.parameters(), lr=0.1),
    )
    accumulated_engine = TrainingEngine(
        model=accumulated_model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
        optimizer=torch.optim.SGD(accumulated_model.parameters(), lr=0.1),
        gradient_accumulation_steps=4,
    )
    batch = make_batch(scores=(2.0, -1.0))

    default_engine.train_epoch([batch])
    accumulated_engine.train_epoch([batch])

    torch.testing.assert_close(
        accumulated_model.linear.weight,
        default_model.linear.weight,
    )


def test_gradient_clipping_bounds_accumulated_global_norm():
    model = TinyTwoInputModel()
    with torch.no_grad():
        model.linear.weight.zero_()
    engine = TrainingEngine(
        model=model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
        optimizer=torch.optim.SGD(model.parameters(), lr=1.0),
        gradient_accumulation_steps=2,
        gradient_clip_norm=0.25,
    )
    combined_batch = make_batch(scores=(100.0, -100.0, 50.0))

    engine.train_epoch(
        [
            slice_batch(combined_batch, 0, 2),
            slice_batch(combined_batch, 2, 3),
        ]
    )

    assert torch.linalg.vector_norm(model.linear.weight).item() == pytest.approx(
        0.25,
        rel=1e-5,
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("gradient_accumulation_steps", 0),
        ("gradient_accumulation_steps", True),
        ("gradient_accumulation_steps", 1.5),
        ("gradient_clip_norm", 0),
        ("gradient_clip_norm", -1),
        ("gradient_clip_norm", float("inf")),
        ("gradient_clip_norm", float("nan")),
        ("gradient_clip_norm", True),
    ],
)
def test_training_engine_rejects_invalid_gradient_controls(
    field_name,
    invalid_value,
):
    model = TinyTwoInputModel()
    arguments = {field_name: invalid_value}

    with pytest.raises(ValueError, match=field_name):
        TrainingEngine(
            model=model,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
            problem_mode="regression",
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            **arguments,
        )


def test_bce_with_logits_preserves_gradient_for_extreme_wrong_predictions():
    model = ScalarLogitModel(initial_logit=100.0)
    engine = TrainingEngine(
        model=model,
        criterion=nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        problem_mode="classification",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    output = engine.train_epoch(
        [make_batch(scores=(0.0,))],
        collect_predictions=True,
    )

    assert output.losses == pytest.approx([100.0])
    assert model.logit.item() == pytest.approx(99.9)
    assert output.predicted_values == pytest.approx([1.0])


def test_bce_with_logits_returns_probabilities_during_evaluation():
    model = ScalarLogitModel(initial_logit=-100.0)
    engine = TrainingEngine(
        model=model,
        criterion=nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        problem_mode="classification",
    )

    output = engine.evaluate_epoch([make_batch(scores=(1.0,))])

    assert output.losses == pytest.approx([100.0])
    assert output.predicted_values == pytest.approx([0.0])


def test_direct_bce_criterion_retains_probability_input_compatibility():
    engine = TrainingEngine(
        model=ScalarLogitModel(initial_logit=100.0),
        criterion=nn.BCELoss(),
        device=torch.device("cpu"),
        problem_mode="classification",
    )

    output = engine.evaluate_epoch([make_batch(scores=(0.0,))])

    assert output.losses == pytest.approx([100.0])
    assert output.predicted_values == pytest.approx([1.0])


def test_multiclass_training_uses_class_ids_and_collects_softmax_probabilities():
    model = MulticlassLinearModel()
    with torch.no_grad():
        model.linear.weight.copy_(
            torch.tensor(
                [
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [-2.0, -2.0],
                ]
            )
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine = TrainingEngine(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        problem_mode="classification",
        classification_mode="multiclass",
        num_classes=3,
        optimizer=optimizer,
    )

    output = engine.train_epoch(
        [make_batch(scores=(0.0, 1.0))],
        collect_predictions=True,
    )

    assert len(output.losses) == 1
    assert output.true_values == [0, 1]
    assert output.predicted_values == [0, 0]
    assert len(output.class_probabilities) == 2
    assert [sum(row) for row in output.class_probabilities] == pytest.approx([1.0, 1.0])


def test_multiclass_training_rejects_scalar_model_outputs():
    engine = TrainingEngine(
        model=TinyTwoInputModel(),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        problem_mode="classification",
        classification_mode="multiclass",
        num_classes=3,
    )

    with pytest.raises(ValueError, match=r"\[batch, num_classes\]"):
        engine.evaluate_epoch([make_batch(scores=(0.0, 1.0))])


@pytest.mark.parametrize("scores", [(0.0, 1.5), (0.0, 3.0)])
def test_multiclass_training_rejects_invalid_direct_target_ids(scores):
    engine = TrainingEngine(
        model=MulticlassLinearModel(),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        problem_mode="classification",
        classification_mode="multiclass",
        num_classes=3,
    )

    with pytest.raises(ValueError, match="integer class IDs"):
        engine.evaluate_epoch([make_batch(scores=scores)])


def test_training_engine_accepts_an_explicit_output_head():
    model = TinyTwoInputModel()
    with torch.no_grad():
        model.linear.weight.zero_()
    engine = TrainingEngine(
        model=model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
        output_head=nn.Sigmoid(),
    )

    output = engine.evaluate_epoch([make_batch(scores=(0.5, 0.5))])

    assert output.predicted_values == pytest.approx([0.5, 0.5])
    assert output.mean_loss == pytest.approx(0.0)


def test_training_engine_accepts_typed_batches():
    model = TinyTwoInputModel()
    engine = TrainingEngine(
        model=model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
    )

    output = engine.evaluate_epoch([make_typed_batch(scores=(1.0, 2.0))])

    assert output.true_values == [1.0, 2.0]
    assert output.instance_ids == [10, 11]
    assert output.target_ids == [20, 21]


def test_training_engine_rejects_typed_input_kind_mismatches():
    model = TinyTwoInputModel()
    model.instance_input_kind = "id"
    model.target_input_kind = "dense"
    engine = TrainingEngine(
        model=model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
    )

    with pytest.raises(ValueError, match="expects 'id'.*produced 'dense'"):
        engine.evaluate_epoch([make_typed_batch()])


def test_evaluate_epoch_does_not_update_model_and_restores_training_mode():
    model = TinyTwoInputModel()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[0.5, -0.25]]))
    engine = TrainingEngine(
        model=model,
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
    )
    initial_weights = model.linear.weight.detach().clone()
    model.train()

    output = engine.evaluate_epoch([make_batch(scores=(1.0, 2.0))])

    assert model.training
    assert torch.equal(model.linear.weight, initial_weights)
    assert output.predicted_values == pytest.approx([0.375, 0.625])
    assert output.mean_loss == pytest.approx(1.140625)


def test_evaluate_epoch_can_skip_prediction_collection():
    engine = TrainingEngine(
        model=TinyTwoInputModel(),
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
    )

    output = engine.evaluate_epoch(
        [make_batch()],
        calculate_loss=False,
        collect_predictions=False,
    )

    assert output.losses == []
    assert output.predicted_values == []


def test_train_epoch_requires_optimizer():
    engine = TrainingEngine(
        model=TinyTwoInputModel(),
        criterion=nn.MSELoss(),
        device=torch.device("cpu"),
        problem_mode="regression",
    )

    with pytest.raises(RuntimeError, match="requires an optimizer"):
        engine.train_epoch([make_batch()])


def test_training_engine_rejects_invalid_problem_mode():
    with pytest.raises(ValueError, match="problem_mode"):
        TrainingEngine(
            model=TinyTwoInputModel(),
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
            problem_mode="ranking",
        )
