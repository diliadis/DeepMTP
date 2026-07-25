import pandas as pd
import pytest
import torch

from DeepMTP import DataLoaderFactory, DeepMTPConfig


def make_config(**overrides):
    values = {
        "validation_setting": "B",
        "problem_mode": "classification",
        "compute_mode": "cpu",
        "num_workers": 0,
        "train_batchsize": 2,
        "val_batchsize": 3,
        "metrics": ["accuracy"],
        "metrics_average": ["macro"],
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": 4,
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 3,
        "random_seed": 11,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


def make_data():
    return {
        "y": {
            "data": pd.DataFrame(
                {
                    "instance_id": [0, 1, 2, 3, 0, 1],
                    "target_id": [0, 1, 2, 0, 1, 2],
                    "value": [0, 1, 0, 1, 0, 1],
                }
            )
        },
        "X_instance": None,
        "X_target": None,
    }


def pair_order(dataloader):
    return [
        (int(instance_id), int(target_id))
        for batch in dataloader
        for instance_id, target_id in zip(
            batch["instance_id"],
            batch["target_id"],
        )
    ]


def test_training_loaders_use_configured_batches_and_device_settings():
    data = make_data()
    dataloaders = DataLoaderFactory(
        make_config(),
        torch.device("cpu"),
    ).for_training(data, data, data)

    assert dataloaders.train.batch_size == 2
    assert dataloaders.validation.batch_size == 3
    assert dataloaders.test.batch_size == 3
    assert not dataloaders.train.pin_memory
    assert len(dataloaders.train.dataset) == 6


def test_training_shuffle_is_repeatable_without_global_rng_configuration():
    data = make_data()
    config = make_config(random_seed=29)
    global_state = torch.random.get_rng_state()

    first = DataLoaderFactory(config, torch.device("cpu")).for_training(
        data,
        data,
        data,
    )
    second = DataLoaderFactory(config, torch.device("cpu")).for_training(
        data,
        data,
        data,
    )

    assert pair_order(first.train) == pair_order(second.train)
    assert torch.equal(global_state, torch.random.get_rng_state())


def test_prediction_loader_preserves_input_order():
    data = make_data()
    dataloader = DataLoaderFactory(
        make_config(),
        torch.device("cpu"),
    ).for_prediction(data)

    assert pair_order(dataloader) == list(
        zip(
            data["y"]["data"]["instance_id"],
            data["y"]["data"]["target_id"],
        )
    )


@pytest.mark.parametrize(
    ("validation_data", "test_data", "message"),
    [
        (None, make_data(), "validation_data"),
        (make_data(), None, "test_data"),
    ],
)
def test_training_requires_validation_and_test_data(
    validation_data,
    test_data,
    message,
):
    factory = DataLoaderFactory(make_config(), torch.device("cpu"))

    with pytest.raises(ValueError, match=message):
        factory.for_training(make_data(), validation_data, test_data)
