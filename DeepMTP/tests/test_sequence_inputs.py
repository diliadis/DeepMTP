from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from DeepMTP import (
    DataLoaderFactory,
    DeepMTP,
    DeepMTPConfig,
    ModelFactory,
    SequenceConv1DEncoder,
    SequenceGRUEncoder,
)
from DeepMTP.config import ConfigValidationError
from DeepMTP.data import (
    SequenceBranchBatch,
    SequenceInput,
    data_process,
    process_instance_features,
    process_target_features,
)
from DeepMTP.runtime.persistence import ExperimentStore
from DeepMTP.utils.utils import generate_config


def make_config(**overrides):
    values = {
        "validation_setting": "B",
        "problem_mode": "classification",
        "general_architecture_version": "dot_product",
        "compute_mode": "cpu",
        "num_workers": 0,
        "train_batchsize": 2,
        "val_batchsize": 2,
        "metrics": ["accuracy"],
        "metrics_average": ["macro"],
        "instance_branch_architecture": "SEQUENCE",
        "instance_branch_input_dim": 20,
        "instance_branch_sequence_embedding_dim": 6,
        "instance_branch_sequence_output_dim": 7,
        "instance_branch_sequence_conv_kernel_size": 3,
        "instance_branch_sequence_num_layers": 1,
        "instance_branch_sequence_padding_idx": 0,
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 2,
        "target_branch_nodes_per_layer": [4],
        "embedding_size": 4,
        "random_seed": 5,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


def make_prediction_data(sequences):
    return {
        "y": {
            "data": pd.DataFrame(
                {
                    "instance_id": [0, 1, 2, 3],
                    "target_id": [0, 1, 0, 1],
                    "value": [0, 1, 1, 0],
                }
            )
        },
        "X_instance": process_instance_features(
            sequences,
            feature_kind="sequence",
        ),
        "X_target": None,
    }


def test_sequence_features_preserve_variable_length_integer_rows():
    source = pd.DataFrame(
        {
            "id": [10, 30, 50],
            "features": [[1, 4, 8], np.array([2]), torch.tensor([3, 7])],
        }
    )

    processed = process_instance_features(source, feature_kind="sequence")

    assert processed["info"] == "sequence"
    assert processed["num_features"] == 9
    assert processed["data"].index.tolist() == [10, 30, 50]
    assert [row.tolist() for row in processed["data"]["features"]] == [
        [1, 4, 8],
        [2],
        [3, 7],
    ]
    assert all(row.dtype == np.int64 for row in processed["data"]["features"])


@pytest.mark.parametrize(
    ("sequences", "exception", "message"),
    [
        ([[1, 2], []], ValueError, "non-empty"),
        ([[1, 2], [3.0]], TypeError, "contain integers"),
        ([[1, -1]], ValueError, "non-negative"),
    ],
)
def test_sequence_feature_processing_rejects_invalid_rows(
    sequences,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        process_instance_features(sequences, feature_kind="sequence")


def test_data_process_marks_sequences_and_rejects_numeric_scaling():
    scores = (np.arange(10)[:, None] + np.arange(3)[None, :]) % 2
    sequences = [[index % 5 + 1] * (index % 4 + 1) for index in range(10)]

    train, validation, test, info = data_process(
        {
            "train": {
                "y": scores,
                "X_instance": sequences,
                "X_target": None,
            }
        },
        validation_setting="B",
        shuffle=False,
        instance_feature_kind="sequence",
    )

    assert info["instance_branch_input_dim"] == 6
    for split in (train, validation, test):
        assert set(split["X_instance"]) == {"data"}
        assert all(
            row.dtype == np.int64 for row in split["X_instance"]["data"]["features"]
        )

    with pytest.raises(ValueError, match="Token IDs are categorical"):
        data_process(
            {
                "train": {
                    "y": scores,
                    "X_instance": sequences,
                    "X_target": None,
                }
            },
            validation_setting="B",
            instance_feature_kind="sequence",
            scale_instance_features="Standard",
        )


def test_sequence_config_normalizes_and_validates_encoder_options():
    config = make_config(instance_branch_sequence_encoder="GRU")
    conv_config = make_config(
        instance_branch_sequence_encoder="CONV1D",
        instance_branch_sequence_conv_kernel_size=5,
    )

    assert config.instance_branch_architecture == "SEQUENCE"
    assert config.instance_branch_sequence_encoder == "gru"
    assert conv_config.instance_branch_sequence_encoder == "conv1d"
    assert conv_config.instance_branch_sequence_conv_kernel_size == 5

    invalid_values = [
        (
            {"instance_branch_sequence_encoder": "transformer"},
            "must be 'conv1d' or 'gru'",
        ),
        ({"instance_branch_sequence_embedding_dim": 0}, "embedding_dim"),
        ({"instance_branch_sequence_output_dim": 0}, "output_dim"),
        ({"instance_branch_sequence_num_layers": 0}, "num_layers"),
        ({"instance_branch_sequence_padding_idx": -1}, "padding_idx"),
        ({"instance_branch_sequence_padding_idx": 20}, "smaller"),
        (
            {
                "instance_branch_sequence_encoder": "conv1d",
                "instance_branch_sequence_conv_kernel_size": 0,
            },
            "conv_kernel_size",
        ),
        (
            {
                "instance_branch_sequence_encoder": "conv1d",
                "instance_branch_sequence_conv_kernel_size": 2,
            },
            "must be odd",
        ),
    ]
    for overrides, message in invalid_values:
        with pytest.raises(ConfigValidationError, match=message):
            make_config(**overrides)


def test_legacy_gru_config_defaults_the_new_conv_kernel_setting():
    legacy_config = make_config().to_dict(include_defaults=True)
    legacy_config.pop("instance_branch_sequence_conv_kernel_size")

    restored = DeepMTPConfig.from_mapping(legacy_config)

    assert restored.instance_branch_sequence_encoder == "gru"
    assert restored.instance_branch_sequence_conv_kernel_size == 3


def test_sequence_collator_builds_dynamic_tokens_masks_and_lengths():
    collator = DataLoaderFactory(make_config(), "cpu").for_prediction(
        make_prediction_data([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])
    )

    batch = next(iter(collator))

    assert isinstance(batch.instance_input, SequenceBranchBatch)
    assert isinstance(batch.instance_features, SequenceInput)
    assert torch.equal(
        batch.instance_features.token_ids,
        torch.tensor([[1, 2, 3], [4, 0, 0]]),
    )
    assert torch.equal(
        batch.instance_features.attention_mask,
        torch.tensor(
            [[True, True, True], [True, False, False]],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(batch.instance_features.lengths, torch.tensor([3, 1]))


def test_sequence_collator_supports_a_nonzero_padding_id():
    config = make_config(instance_branch_sequence_padding_idx=19)
    loader = DataLoaderFactory(config, "cpu").for_prediction(
        make_prediction_data([[0, 1], [2], [3], [4]])
    )

    sequence_input = next(iter(loader)).instance_features

    assert sequence_input.padding_idx == 19
    assert torch.equal(
        sequence_input.token_ids,
        torch.tensor([[0, 1], [2, 19]]),
    )


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        (
            {"attention_mask": torch.tensor([[True, False], [True, True]])},
            ValueError,
            "left-aligned",
        ),
        (
            {"lengths": torch.tensor([2, 0])},
            ValueError,
            "positive",
        ),
        (
            {"token_ids": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
            TypeError,
            "integer",
        ),
    ],
)
def test_sequence_input_rejects_inconsistent_structured_tensors(
    overrides,
    exception,
    message,
):
    arguments = {
        "token_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.bool),
        "lengths": torch.tensor([2, 2]),
        "padding_idx": 0,
    }
    arguments.update(overrides)

    with pytest.raises(exception, match=message):
        SequenceInput(**arguments)


def test_sequence_gru_ignores_dynamic_padding_and_backpropagates():
    model = SequenceGRUEncoder(
        vocabulary_size=20,
        embedding_dim=5,
        output_dim=4,
        num_layers=2,
        padding_idx=0,
        dropout_rate=0.0,
    )
    model.eval()
    short_batch = next(
        iter(
            DataLoaderFactory(make_config(), "cpu").for_prediction(
                make_prediction_data([[1, 2], [3], [4], [5]])
            )
        )
    ).instance_features
    wide_batch = next(
        iter(
            DataLoaderFactory(make_config(), "cpu").for_prediction(
                make_prediction_data([[1, 2], [3, 4, 5, 6], [7], [8]])
            )
        )
    ).instance_features

    short_output = model(short_batch)
    wide_output = model(wide_batch)
    wide_output.sum().backward()

    assert short_output.shape == (2, 4)
    assert torch.allclose(short_output[0], wide_output[0])
    assert model.embedding.weight.grad is not None
    assert model.gru.weight_ih_l0.grad is not None


def test_sequence_conv1d_ignores_dynamic_padding_and_backpropagates():
    model = SequenceConv1DEncoder(
        vocabulary_size=20,
        embedding_dim=5,
        output_dim=4,
        num_layers=2,
        kernel_size=3,
        padding_idx=0,
        dropout_rate=0.0,
    )
    model.eval()
    short_batch = next(
        iter(
            DataLoaderFactory(make_config(), "cpu").for_prediction(
                make_prediction_data([[1, 2], [3], [4], [5]])
            )
        )
    ).instance_features
    wide_batch = next(
        iter(
            DataLoaderFactory(make_config(), "cpu").for_prediction(
                make_prediction_data([[1, 2], [3, 4, 5, 6], [7], [8]])
            )
        )
    ).instance_features

    short_output = model(short_batch)
    wide_output = model(wide_batch)
    wide_output.sum().backward()

    assert short_output.shape == (2, 4)
    assert torch.allclose(short_output[0], wide_output[0])
    assert model.embedding.weight.grad is not None
    assert model.convolutions[0].weight.grad is not None


@pytest.mark.parametrize("architecture", ["dot_product", "mlp", "kronecker"])
@pytest.mark.parametrize(
    ("encoder_name", "encoder_type"),
    [
        ("gru", SequenceGRUEncoder),
        ("conv1d", SequenceConv1DEncoder),
    ],
)
def test_model_factory_combines_sequence_and_dense_branches(
    architecture,
    encoder_name,
    encoder_type,
):
    config = make_config(
        general_architecture_version=architecture,
        comb_mlp_nodes_per_layer=[5],
        instance_branch_sequence_encoder=encoder_name,
    )
    bundle = ModelFactory(config).build()
    batch = next(
        iter(
            DataLoaderFactory(config, "cpu").for_prediction(
                make_prediction_data([[1, 2], [3], [4, 5], [6]])
            )
        )
    )

    output = bundle.model(batch.instance_features, batch.target_features)

    assert isinstance(bundle.instance_branch, encoder_type)
    assert bundle.model.instance_input_kind == "sequence"
    assert bundle.instance_branch.output_dim == (
        4 if architecture == "dot_product" else 7
    )
    assert output.shape == (2, 1)


@pytest.mark.parametrize(
    ("encoder_name", "encoder_type"),
    [
        ("gru", SequenceGRUEncoder),
        ("conv1d", SequenceConv1DEncoder),
    ],
)
def test_target_branch_accepts_sequence_features(encoder_name, encoder_type):
    config = make_config(
        validation_setting="C",
        instance_branch_architecture="MLP",
        instance_branch_input_dim=2,
        instance_branch_nodes_per_layer=[4],
        target_branch_architecture="SEQUENCE",
        target_branch_input_dim=20,
        target_branch_sequence_embedding_dim=6,
        target_branch_sequence_output_dim=7,
        target_branch_sequence_encoder=encoder_name,
        target_branch_sequence_padding_idx=0,
    )
    data = {
        "y": {
            "data": pd.DataFrame(
                {
                    "instance_id": [0, 1, 0, 1],
                    "target_id": [0, 1, 2, 3],
                    "value": [0, 1, 1, 0],
                }
            )
        },
        "X_instance": None,
        "X_target": process_target_features(
            [[1, 2], [3], [4, 5, 6], [7]],
            feature_kind="sequence",
        ),
    }

    batch = next(iter(DataLoaderFactory(config, "cpu").for_prediction(data)))
    bundle = ModelFactory(config).build()
    output = bundle.model(
        batch.instance_features,
        batch.target_features,
    )

    assert isinstance(batch.target_input, SequenceBranchBatch)
    assert isinstance(bundle.target_branch, encoder_type)
    assert output.shape == (2, 1)


def test_sequence_loader_rejects_padding_and_out_of_vocabulary_tokens():
    with pytest.raises(ValueError, match="cannot contain padding_idx"):
        DataLoaderFactory(make_config(), "cpu").for_prediction(
            make_prediction_data([[1, 2], [0], [3], [4]])
        )

    with pytest.raises(ValueError, match="vocabulary_size=20"):
        DataLoaderFactory(make_config(), "cpu").for_prediction(
            make_prediction_data([[1, 2], [20], [3], [4]])
        )


@pytest.mark.integration
@pytest.mark.filterwarnings(
    "ignore::DeepMTP.config.ConfigNormalizationWarning",
)
@pytest.mark.parametrize(
    ("encoder_name", "encoder_type"),
    [
        ("gru", SequenceGRUEncoder),
        ("conv1d", SequenceConv1DEncoder),
    ],
)
def test_tiny_cpu_sequence_training_prediction_and_checkpoint(
    tmp_path,
    encoder_name,
    encoder_type,
):
    sequences = [
        [index % 7 + 1, (index + 2) % 7 + 1][: index % 2 + 1] for index in range(16)
    ]
    scores = (np.arange(16)[:, None] + np.arange(4)[None, :]) % 2
    train, validation, test, data_info = data_process(
        {
            "train": {
                "y": scores,
                "X_instance": sequences,
                "X_target": None,
            }
        },
        validation_setting="B",
        shuffle=False,
        instance_feature_kind="sequence",
    )
    config = generate_config(
        validation_setting="B",
        problem_mode="classification",
        general_architecture_version="dot_product",
        compute_mode="cpu",
        num_workers=0,
        train_batchsize=16,
        val_batchsize=16,
        num_epochs=1,
        metrics=["accuracy"],
        metrics_average=["macro"],
        evaluate_train=False,
        evaluate_val=True,
        use_early_stopping=False,
        save_model=False,
        results_path=str(tmp_path),
        experiment_name=f"sequence_{encoder_name}",
        instance_branch_architecture="SEQUENCE",
        instance_branch_input_dim=data_info["instance_branch_input_dim"],
        instance_branch_params={
            "instance_branch_sequence_encoder": encoder_name,
            "instance_branch_sequence_embedding_dim": 5,
            "instance_branch_sequence_output_dim": 4,
            "instance_branch_sequence_conv_kernel_size": 3,
            "instance_branch_sequence_num_layers": 1,
            "instance_branch_sequence_padding_idx": 0,
        },
        target_branch_architecture="MLP",
        target_branch_input_dim=data_info["target_branch_input_dim"],
        target_branch_params={
            "target_branch_nodes_per_layer": [4],
            "target_branch_layers": None,
            "target_branch_nodes_reducing_factor": 2,
        },
        embedding_size=4,
    )
    model = DeepMTP(config)

    validation_results = model.train(train, validation, test)
    _, predictions = model.predict(test, return_predictions=True)

    assert isinstance(model.instance_branch_model, encoder_type)
    assert "val_loss" in validation_results
    assert len(predictions) == len(test["y"]["data"])

    model.save_model()
    checkpoint_path = Path(model.experiment_dir) / "model.pt"
    checkpoint = ExperimentStore.load_checkpoint(checkpoint_path)
    assert checkpoint["config"]["instance_branch_sequence_encoder"] == encoder_name
    assert checkpoint["config"]["instance_branch_sequence_conv_kernel_size"] == 3
    restored_config = dict(config)
    restored_config["experiment_name"] = f"sequence_{encoder_name}_restored"
    restored_model = DeepMTP(restored_config, checkpoint_dir=checkpoint_path)
    _, restored_predictions = restored_model.predict(
        test,
        return_predictions=True,
    )

    assert np.allclose(
        predictions["predicted_values"],
        restored_predictions["predicted_values"],
    )
    assert isinstance(restored_model.instance_branch_model, encoder_type)
