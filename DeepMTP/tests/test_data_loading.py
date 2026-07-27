import pandas as pd
import pytest
import torch

import DeepMTP.data_loading as data_loading_module
import DeepMTP.utils.data_utils as legacy_data_utils
from DeepMTP import DataLoaderFactory, DeepMTPConfig
from DeepMTP.data_loading import BaseDataset, MTPSampler


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
            strict=True,
        )
    ]


def test_data_components_live_in_data_loading_with_legacy_compatibility():
    assert BaseDataset.__module__ == "DeepMTP.data_loading"
    assert MTPSampler.__module__ == "DeepMTP.data_loading"
    assert legacy_data_utils.BaseDataset is BaseDataset
    assert legacy_data_utils.MTPSampler is MTPSampler


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


@pytest.mark.parametrize(
    ("device", "pin_memory"),
    [("cpu", False), ("cuda:2", True)],
)
def test_factory_normalizes_device_strings(device, pin_memory):
    factory = DataLoaderFactory(make_config(), device)

    dataloader = factory.for_prediction(make_data())

    assert factory.device == torch.device(device)
    assert dataloader.pin_memory is pin_memory


@pytest.mark.parametrize(
    ("device", "exception"),
    [
        (object(), TypeError),
        ("", ValueError),
        ("not-a-device", ValueError),
    ],
)
def test_factory_rejects_invalid_devices(device, exception):
    with pytest.raises(exception, match="device"):
        DataLoaderFactory(make_config(), device)


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
            strict=True,
        )
    )


def test_training_loaders_select_transforms_by_phase():
    instance_train_transform = object()
    instance_inference_transform = object()
    target_train_transform = object()
    target_inference_transform = object()
    config = make_config(
        instance_train_transforms=instance_train_transform,
        instance_inference_transforms=instance_inference_transform,
        target_train_transforms=target_train_transform,
        target_inference_transforms=target_inference_transform,
    )
    data = make_data()

    dataloaders = DataLoaderFactory(config, "cpu").for_training(data, data, data)

    assert dataloaders.train.dataset.instance_transform is instance_train_transform
    assert dataloaders.train.dataset.target_transform is target_train_transform
    for dataloader in (dataloaders.validation, dataloaders.test):
        assert dataloader.dataset.instance_transform is instance_inference_transform
        assert dataloader.dataset.target_transform is target_inference_transform


@pytest.mark.parametrize(
    ("data", "exception", "message"),
    [
        ([], TypeError, "data must be a mapping"),
        ({}, ValueError, "missing required keys"),
        (
            {"y": None, "X_instance": None, "X_target": None},
            TypeError,
            r"data\['y'\]",
        ),
        (
            {**make_data(), "X_instance": []},
            TypeError,
            "X_instance",
        ),
        (
            {**make_data(), "X_target": "features"},
            TypeError,
            "X_target",
        ),
    ],
)
def test_prediction_validates_split_contract(data, exception, message):
    factory = DataLoaderFactory(make_config(), "cpu")

    with pytest.raises(exception, match=message):
        factory.for_prediction(data)


def test_training_validates_every_split_before_constructing_datasets(
    monkeypatch,
):
    constructor_calls = []
    original_dataset = data_loading_module.BaseDataset

    def tracking_dataset(*args, **kwargs):
        constructor_calls.append((args, kwargs))
        return original_dataset(*args, **kwargs)

    monkeypatch.setattr(data_loading_module, "BaseDataset", tracking_dataset)
    invalid_validation_data = {"y": make_data()["y"]}
    factory = DataLoaderFactory(make_config(), "cpu")

    with pytest.raises(ValueError, match="validation_data.*missing required keys"):
        factory.for_training(
            make_data(),
            invalid_validation_data,
            make_data(),
        )

    assert constructor_calls == []


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
