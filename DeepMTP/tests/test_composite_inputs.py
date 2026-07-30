from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from DeepMTP import CompositeEncoder, DeepMTP, DeepMTPConfig, ModelFactory
from DeepMTP.config import ConfigValidationError
from DeepMTP.data import (
    CompositeBranchBatch,
    CompositeInput,
    DataLoaderFactory,
    DenseBranchBatch,
    MTPBatchCollator,
    SequenceInput,
    data_process,
    process_instance_features,
    process_target_features,
)
from DeepMTP.runtime.persistence import ExperimentStore


def make_config(**overrides):
    values = {
        "validation_setting": "B",
        "problem_mode": "classification",
        "general_architecture_version": "mlp",
        "compute_mode": "cpu",
        "metrics": ["accuracy"],
        "metrics_average": ["macro"],
        "instance_branch_architecture": "COMPOSITE",
        "instance_branch_composite_components": {
            "descriptors": {
                "architecture": "MLP",
                "input_dim": 2,
                "output_dim": 3,
                "nodes_per_layer": [4],
            },
            "tokens": {
                "architecture": "SEQUENCE",
                "input_dim": 8,
                "output_dim": 2,
                "embedding_dim": 4,
            },
        },
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 2,
        "target_branch_nodes_per_layer": [4],
        "comb_mlp_nodes_per_layer": [4],
        "train_batchsize": 2,
        "val_batchsize": 2,
        "num_workers": 0,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


def make_processed_features():
    return process_instance_features(
        {
            "descriptors": np.array(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
            "tokens": [[1, 2, 3], [4]],
        },
        feature_kind={
            "descriptors": None,
            "tokens": "sequence",
        },
    )


def make_prediction_data():
    return {
        "y": {
            "data": pd.DataFrame(
                {
                    "instance_id": [0, 1],
                    "target_id": [0, 1],
                    "value": [0, 1],
                }
            )
        },
        "X_instance": make_processed_features(),
        "X_target": None,
    }


def test_composite_feature_processing_aligns_named_component_rows():
    processed = make_processed_features()

    assert processed is not None
    assert processed["info"] == "composite"
    assert processed["num_features"] is None
    assert processed["components"] == {
        "descriptors": {
            "num_features": 2,
            "info": "numpy",
        },
        "tokens": {
            "num_features": 5,
            "info": "sequence",
        },
    }
    first_row = processed["data"].loc[0, "features"]
    assert list(first_row) == ["descriptors", "tokens"]
    assert np.array_equal(first_row["descriptors"], np.array([1.0, 2.0]))
    assert np.array_equal(first_row["tokens"], np.array([1, 2, 3]))


def test_composite_feature_processing_rejects_misaligned_entity_ids():
    with pytest.raises(ValueError, match="same instance IDs"):
        process_instance_features(
            {
                "descriptors": pd.DataFrame(
                    {
                        "id": [0, 1],
                        "features": [[1.0], [2.0]],
                    }
                ),
                "tokens": pd.DataFrame(
                    {
                        "id": [0, 2],
                        "features": [[1], [2]],
                    }
                ),
            },
            feature_kind={
                "descriptors": None,
                "tokens": "sequence",
            },
        )


def test_composite_feature_processing_requires_matching_named_sources():
    with pytest.raises(ValueError, match="same names"):
        process_instance_features(
            {"descriptors": np.ones((2, 2))},
            feature_kind={
                "descriptors": None,
                "tokens": "sequence",
            },
        )


def test_composite_collator_preserves_dense_sparse_sequence_and_id_components():
    collator = MTPBatchCollator(
        instance_kind="composite",
        target_kind="dense",
        instance_component_kinds={
            "descriptors": "dense",
            "tokens": "sequence",
            "fingerprints": "sparse",
            "identity": "id",
        },
        instance_component_padding_indices={"tokens": 0},
    )
    items = [
        {
            "instance_id": 0,
            "target_id": 0,
            "instance_features": {
                "descriptors": np.array([1.0, 2.0]),
                "tokens": np.array([1, 2, 3]),
                "fingerprints": sparse.csr_matrix([[0.0, 1.0, 0.0]]),
                "identity": 0,
            },
            "target_features": np.array([1.0]),
            "score": 0,
        },
        {
            "instance_id": 1,
            "target_id": 1,
            "instance_features": {
                "descriptors": np.array([3.0, 4.0]),
                "tokens": np.array([4]),
                "fingerprints": sparse.csr_matrix([[2.0, 0.0, 3.0]]),
                "identity": 1,
            },
            "target_features": np.array([2.0]),
            "score": 1,
        },
    ]

    batch = collator(items)

    assert isinstance(batch.instance_input, CompositeBranchBatch)
    assert isinstance(batch.instance_features, CompositeInput)
    assert isinstance(batch.target_input, DenseBranchBatch)
    assert batch.instance_features["descriptors"].shape == (2, 2)
    assert isinstance(batch.instance_features["tokens"], SequenceInput)
    assert torch.equal(
        batch.instance_features["tokens"].token_ids,
        torch.tensor([[1, 2, 3], [4, 0, 0]]),
    )
    assert batch.instance_features["fingerprints"].layout == torch.sparse_coo
    assert torch.equal(
        batch.instance_features["fingerprints"].to_dense(),
        torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]]),
    )
    assert torch.equal(batch.instance_features["identity"], torch.tensor([0, 1]))


def test_composite_input_requires_aligned_component_batches():
    with pytest.raises(ValueError, match="same batch size"):
        CompositeInput(
            {
                "first": torch.ones(2, 1),
                "second": torch.ones(1, 1),
            }
        )


def test_composite_config_normalizes_components_and_validates_structure():
    config = make_config(
        instance_branch_composite_components={
            "dense": {
                "architecture": "mlp",
                "input_dim": 2,
                "output_dim": 3,
            },
            "tokens": {
                "architecture": "sequence",
                "input_dim": 8,
                "output_dim": 2,
                "encoder": "GRU",
            },
        }
    )

    assert config.instance_branch_composite_components["dense"]["architecture"] == "MLP"
    assert config.instance_branch_composite_components["tokens"]["encoder"] == "gru"

    with pytest.raises(ConfigValidationError, match="must be 'MLP'"):
        make_config(
            instance_branch_composite_components={
                "nested": {
                    "architecture": "COMPOSITE",
                    "input_dim": 2,
                    "output_dim": 3,
                },
                "dense": {
                    "architecture": "MLP",
                    "input_dim": 2,
                    "output_dim": 3,
                },
            }
        )

    with pytest.raises(ConfigValidationError, match="at least two"):
        make_config(
            instance_branch_composite_components={
                "dense": {
                    "architecture": "MLP",
                    "input_dim": 2,
                    "output_dim": 3,
                }
            }
        )


@pytest.mark.parametrize(
    ("sequence_options", "message"),
    [
        (
            {"encoder": "conv1d", "conv_kernel_size": 2},
            "must be odd",
        ),
        (
            {
                "encoder": "transformer",
                "embedding_dim": 5,
                "transformer_num_heads": 2,
            },
            "must be divisible",
        ),
    ],
)
def test_composite_sequence_component_validation(sequence_options, message):
    with pytest.raises(ConfigValidationError, match=message):
        make_config(
            instance_branch_composite_components={
                "descriptors": {
                    "architecture": "MLP",
                    "input_dim": 2,
                    "output_dim": 3,
                },
                "tokens": {
                    "architecture": "SEQUENCE",
                    "input_dim": 8,
                    "output_dim": 2,
                    **sequence_options,
                },
            }
        )


def test_composite_id_component_is_rejected_for_novel_instances():
    with pytest.raises(ConfigValidationError, match="cannot represent novel instances"):
        make_config(
            instance_branch_composite_components={
                "identity": {
                    "architecture": "EMBEDDING",
                    "input_dim": 2,
                    "output_dim": 2,
                },
                "dense": {
                    "architecture": "MLP",
                    "input_dim": 2,
                    "output_dim": 3,
                },
            }
        )


def test_model_factory_combines_component_encoders_and_backpropagates():
    bundle = ModelFactory(make_config()).build()
    batch = next(
        iter(
            DataLoaderFactory(make_config(), "cpu").for_prediction(
                make_prediction_data()
            )
        )
    )

    output = bundle.model(batch.instance_features, batch.target_features)
    output.sum().backward()

    assert isinstance(bundle.instance_branch, CompositeEncoder)
    assert bundle.instance_branch.output_dim == 5
    assert bundle.model.instance_input_kind == "composite"
    assert output.shape == (2, 1)
    assert all(
        parameter.grad is not None
        for parameter in bundle.instance_branch.parameters()
        if parameter.requires_grad
    )


def test_composite_branch_combines_sparse_and_dense_components():
    config = make_config(
        instance_branch_composite_components={
            "fingerprints": {
                "architecture": "SPARSE",
                "input_dim": 3,
                "output_dim": 2,
                "nodes_per_layer": [3],
            },
            "descriptors": {
                "architecture": "MLP",
                "input_dim": 2,
                "output_dim": 3,
                "nodes_per_layer": [4],
            },
        }
    )
    features = process_instance_features(
        {
            "fingerprints": sparse.csr_matrix([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]]),
            "descriptors": np.array([[1.0, 2.0], [3.0, 4.0]]),
        },
        feature_kind={
            "fingerprints": None,
            "descriptors": None,
        },
    )
    data = make_prediction_data()
    data["X_instance"] = features
    bundle = ModelFactory(config).build()
    batch = next(iter(DataLoaderFactory(config, "cpu").for_prediction(data)))

    output = bundle.model(batch.instance_features, batch.target_features)

    assert batch.instance_features["fingerprints"].layout == torch.sparse_coo
    assert output.shape == (2, 1)


def test_composite_branch_combines_known_ids_with_side_features():
    config = make_config(
        validation_setting="A",
        metrics_average=["micro"],
        instance_branch_composite_components={
            "identity": {
                "architecture": "EMBEDDING",
                "input_dim": 2,
                "output_dim": 2,
            },
            "descriptors": {
                "architecture": "MLP",
                "input_dim": 2,
                "output_dim": 3,
                "nodes_per_layer": [4],
            },
        },
    )
    data = make_prediction_data()
    data["X_instance"] = process_instance_features(
        {
            "descriptors": np.array(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            )
        },
        feature_kind={"descriptors": None},
    )
    bundle = ModelFactory(config).build()
    batch = next(iter(DataLoaderFactory(config, "cpu").for_prediction(data)))

    output = bundle.model(batch.instance_features, batch.target_features)

    assert torch.equal(batch.instance_features["identity"], torch.tensor([0, 1]))
    assert output.shape == (2, 1)


def test_composite_target_branch_uses_the_same_data_and_model_contract():
    config = make_config(
        instance_branch_architecture="MLP",
        instance_branch_input_dim=2,
        instance_branch_nodes_per_layer=[4],
        target_branch_architecture="COMPOSITE",
        target_branch_composite_components={
            "descriptors": {
                "architecture": "MLP",
                "input_dim": 2,
                "output_dim": 3,
                "nodes_per_layer": [4],
            },
            "tokens": {
                "architecture": "SEQUENCE",
                "input_dim": 8,
                "output_dim": 2,
                "embedding_dim": 4,
            },
        },
    )
    data = make_prediction_data()
    data["X_instance"] = None
    data["X_target"] = process_target_features(
        {
            "descriptors": np.array(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
            "tokens": [[1, 2, 3], [4]],
        },
        feature_kind={
            "descriptors": None,
            "tokens": "sequence",
        },
    )
    bundle = ModelFactory(config).build()
    batch = next(iter(DataLoaderFactory(config, "cpu").for_prediction(data)))

    output = bundle.model(batch.instance_features, batch.target_features)

    assert isinstance(bundle.target_branch, CompositeEncoder)
    assert isinstance(batch.target_features, CompositeInput)
    assert output.shape == (2, 1)


def test_composite_encoder_rejects_component_name_mismatches():
    encoder = ModelFactory(make_config()).build().instance_branch

    with pytest.raises(ValueError, match="component names"):
        encoder(
            CompositeInput(
                {
                    "descriptors": torch.ones(2, 2),
                    "unexpected": torch.ones(2, 1),
                }
            )
        )


def test_tiny_cpu_composite_training_prediction_and_checkpoint(tmp_path):
    descriptors = np.arange(32, dtype=np.float32).reshape(16, 2)
    sequences = [
        [index % 7 + 1, (index + 2) % 7 + 1][: index % 2 + 1] for index in range(16)
    ]
    scores = (np.arange(16)[:, None] + np.arange(4)[None, :]) % 2
    train, validation, test, data_info = data_process(
        {
            "train": {
                "y": scores,
                "X_instance": {
                    "descriptors": descriptors,
                    "tokens": sequences,
                },
                "X_target": None,
            }
        },
        validation_setting="B",
        shuffle=False,
        instance_feature_kind={
            "descriptors": None,
            "tokens": "sequence",
        },
    )
    component_dims = data_info["instance_branch_component_input_dims"]
    assert component_dims == {
        "descriptors": 2,
        "tokens": 8,
    }
    config = make_config(
        general_architecture_version="dot_product",
        embedding_size=5,
        instance_branch_composite_components={
            "descriptors": {
                "architecture": "MLP",
                "input_dim": component_dims["descriptors"],
                "output_dim": 3,
                "nodes_per_layer": [4],
            },
            "tokens": {
                "architecture": "SEQUENCE",
                "input_dim": component_dims["tokens"],
                "output_dim": 2,
                "embedding_dim": 4,
            },
        },
        target_branch_input_dim=data_info["target_branch_input_dim"],
        num_epochs=1,
        evaluate_val=True,
        use_early_stopping=False,
        save_model=False,
        results_path=str(tmp_path),
        experiment_name="composite",
    )
    model = DeepMTP(config)

    validation_results = model.train(train, validation, test)
    _, predictions = model.predict(test, return_predictions=True)
    model.save_model()

    checkpoint_path = Path(model.experiment_dir) / "model.pt"
    checkpoint = ExperimentStore.load_checkpoint(checkpoint_path)
    saved_components = checkpoint["config"]["instance_branch_composite_components"]
    assert isinstance(model.instance_branch_model, CompositeEncoder)
    assert "val_loss" in validation_results
    assert len(predictions) == len(test["y"]["data"])
    assert saved_components["tokens"]["architecture"] == "SEQUENCE"

    restored_config = config.to_dict(include_defaults=True)
    restored_config["experiment_name"] = "composite-restored"
    restored_model = DeepMTP(restored_config, checkpoint_dir=checkpoint_path)
    _, restored_predictions = restored_model.predict(
        test,
        return_predictions=True,
    )

    assert np.allclose(
        predictions["predicted_values"],
        restored_predictions["predicted_values"],
    )
    assert isinstance(restored_model.instance_branch_model, CompositeEncoder)
