import math

import pytest
import torch
from torch import nn

from DeepMTP import EpochOutput, TrainingEngine


class TinyTwoInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, instance_features, target_features):
        features = torch.cat((instance_features, target_features), dim=1)
        return self.linear(features)


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
