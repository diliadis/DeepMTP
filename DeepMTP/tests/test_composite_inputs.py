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
    MaskedComponentInput,
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


def test_optional_composite_features_align_subset_ids_with_missing_values():
    processed = process_instance_features(
        {
            "descriptors": pd.DataFrame(
                {
                    "id": [0, 1, 2],
                    "features": [[1.0], [2.0], [3.0]],
                }
            ),
            "tokens": pd.DataFrame(
                {
                    "id": [0, 2],
                    "features": [[1, 2], [3]],
                }
            ),
        },
        feature_kind={
            "descriptors": None,
            "tokens": {
                "kind": "sequence",
                "optional": True,
            },
        },
    )

    assert processed is not None
    assert processed["data"].index.tolist() == [0, 1, 2]
    assert processed["data"].at[1, "features"]["tokens"] is None
    assert np.array_equal(
        processed["data"].at[2, "features"]["tokens"],
        np.array([3]),
    )


def test_all_optional_composite_sources_use_the_union_of_declared_ids():
    processed = process_instance_features(
        {
            "descriptors": pd.DataFrame(
                {
                    "id": [0, 1],
                    "features": [[1.0], [2.0]],
                }
            ),
            "tokens": pd.DataFrame(
                {
                    "id": [1, 2],
                    "features": [[1, 2], [3]],
                }
            ),
        },
        feature_kind={
            "descriptors": {"optional": True},
            "tokens": {
                "kind": "sequence",
                "optional": True,
            },
        },
    )

    assert processed is not None
    assert processed["data"].index.tolist() == [0, 1, 2]
    assert processed["data"].at[0, "features"]["tokens"] is None
    assert processed["data"].at[2, "features"]["descriptors"] is None


def test_optional_composite_feature_spec_requires_a_boolean_flag():
    with pytest.raises(ValueError, match="optional must be a boolean"):
        process_instance_features(
            {
                "descriptors": np.ones((2, 1)),
                "tokens": [[1], [2]],
            },
            feature_kind={
                "descriptors": None,
                "tokens": {
                    "kind": "sequence",
                    "optional": "yes",
                },
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


def test_composite_collator_masks_optional_sequence_sub_batches():
    collator = MTPBatchCollator(
        instance_kind="composite",
        target_kind="dense",
        instance_component_kinds={
            "descriptors": "dense",
            "tokens": "sequence",
        },
        instance_component_padding_indices={"tokens": 0},
        instance_component_optional={"tokens": True},
    )
    items = [
        {
            "instance_id": 0,
            "target_id": 0,
            "instance_features": {
                "descriptors": np.array([1.0, 2.0]),
                "tokens": np.array([1, 2]),
            },
            "target_features": np.array([1.0]),
            "score": 0,
        },
        {
            "instance_id": 1,
            "target_id": 1,
            "instance_features": {
                "descriptors": np.array([3.0, 4.0]),
                "tokens": None,
            },
            "target_features": np.array([2.0]),
            "score": 1,
        },
    ]

    batch = collator(items)
    tokens = batch.instance_features["tokens"]

    assert isinstance(tokens, MaskedComponentInput)
    assert torch.equal(tokens.presence_mask, torch.tensor([True, False]))
    assert isinstance(tokens.values, SequenceInput)
    assert tokens.values.batch_size == 1

    all_missing_items = [
        {
            **item,
            "instance_features": {
                **item["instance_features"],
                "tokens": None,
            },
        }
        for item in items
    ]
    all_missing_tokens = collator(all_missing_items).instance_features["tokens"]
    assert isinstance(all_missing_tokens, MaskedComponentInput)
    assert not all_missing_tokens.presence_mask.any()
    assert all_missing_tokens.values is None


@pytest.mark.parametrize(
    ("component_kind", "present_value"),
    [
        ("dense", np.array([5.0, 6.0])),
        ("sparse", sparse.csr_matrix([[0.0, 2.0]])),
    ],
)
def test_composite_collator_masks_optional_dense_and_sparse_sub_batches(
    component_kind,
    present_value,
):
    collator = MTPBatchCollator(
        instance_kind="composite",
        target_kind="dense",
        instance_component_kinds={
            "required": "dense",
            "optional": component_kind,
        },
        instance_component_optional={"optional": True},
    )
    items = [
        {
            "instance_id": 0,
            "target_id": 0,
            "instance_features": {
                "required": np.array([1.0]),
                "optional": present_value,
            },
            "target_features": np.array([1.0]),
            "score": 0,
        },
        {
            "instance_id": 1,
            "target_id": 1,
            "instance_features": {
                "required": np.array([2.0]),
                "optional": None,
            },
            "target_features": np.array([2.0]),
            "score": 1,
        },
    ]

    optional = collator(items).instance_features["optional"]

    assert isinstance(optional, MaskedComponentInput)
    assert torch.equal(optional.presence_mask, torch.tensor([True, False]))
    assert isinstance(optional.values, torch.Tensor)
    assert optional.values.shape == (1, 2)
    if component_kind == "sparse":
        assert optional.values.layout == torch.sparse_coo


def test_composite_input_requires_aligned_component_batches():
    with pytest.raises(ValueError, match="same batch size"):
        CompositeInput(
            {
                "first": torch.ones(2, 1),
                "second": torch.ones(1, 1),
            }
        )


def test_masked_component_input_validates_present_sub_batch_size():
    with pytest.raises(ValueError, match="one row per present"):
        MaskedComponentInput(
            values=torch.ones(2, 1),
            presence_mask=torch.tensor([True, False, False]),
        )
    with pytest.raises(ValueError, match="values are required"):
        MaskedComponentInput(
            values=None,
            presence_mask=torch.tensor([True, False]),
        )
    with pytest.raises(TypeError, match="torch.bool"):
        MaskedComponentInput(
            values=torch.ones(1, 1),
            presence_mask=torch.tensor([1, 0]),
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
    assert config.instance_branch_composite_components["tokens"]["optional"] is False

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
    ("component", "message"),
    [
        (
            {
                "architecture": "SEQUENCE",
                "input_dim": 8,
                "output_dim": 2,
                "optional": "yes",
            },
            "optional.*boolean",
        ),
        (
            {
                "architecture": "EMBEDDING",
                "input_dim": 3,
                "output_dim": 2,
                "optional": True,
            },
            "EMBEDDING.*optional",
        ),
        (
            {
                "architecture": "MLP",
                "input_dim": 2,
                "output_dim": 2,
                "optional": True,
                "batch_norm": True,
            },
            "batch_norm.*optional",
        ),
    ],
)
def test_optional_composite_component_configuration_is_validated(
    component,
    message,
):
    with pytest.raises(ConfigValidationError, match=message):
        make_config(
            instance_branch_composite_components={
                "descriptors": {
                    "architecture": "MLP",
                    "input_dim": 2,
                    "output_dim": 3,
                },
                "optional": component,
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
    assert not bundle.instance_branch.missing_representations
    assert not any(
        "missing_representations" in name
        for name in bundle.instance_branch.state_dict()
    )


def test_composite_encoder_learns_a_missing_sequence_representation():
    config = make_config(
        instance_branch_composite_components={
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
                "optional": True,
            },
        }
    )
    data = make_prediction_data()
    data["X_instance"] = process_instance_features(
        {
            "descriptors": np.array(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
            "tokens": [[1, 2], None],
        },
        feature_kind={
            "descriptors": None,
            "tokens": {
                "kind": "sequence",
                "optional": True,
            },
        },
    )
    bundle = ModelFactory(config).build()
    batch = next(iter(DataLoaderFactory(config, "cpu").for_prediction(data)))

    branch_output = bundle.instance_branch(batch.instance_features)
    branch_output.sum().backward()
    missing = bundle.instance_branch.missing_representations["tokens"]

    assert isinstance(batch.instance_features["tokens"], MaskedComponentInput)
    assert torch.equal(branch_output[1, 3:], missing.detach())
    assert missing.grad is not None
    assert torch.equal(missing.grad, torch.ones_like(missing))


def test_missing_component_values_require_optional_model_configuration():
    data = make_prediction_data()
    data["X_instance"] = process_instance_features(
        {
            "descriptors": np.array(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
            "tokens": [[1, 2], None],
        },
        feature_kind={
            "descriptors": None,
            "tokens": {
                "kind": "sequence",
                "optional": True,
            },
        },
    )

    with pytest.raises(ValueError, match="required composite component 'tokens'"):
        next(iter(DataLoaderFactory(make_config(), "cpu").for_prediction(data)))


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
                "optional": True,
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
            "tokens": [[1, 2, 3], None],
        },
        feature_kind={
            "descriptors": None,
            "tokens": {
                "kind": "sequence",
                "optional": True,
            },
        },
    )
    bundle = ModelFactory(config).build()
    batch = next(iter(DataLoaderFactory(config, "cpu").for_prediction(data)))

    output = bundle.model(batch.instance_features, batch.target_features)

    assert isinstance(bundle.target_branch, CompositeEncoder)
    assert isinstance(batch.target_features, CompositeInput)
    assert isinstance(batch.target_features["tokens"], MaskedComponentInput)
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


def test_tiny_cpu_optional_composite_training_prediction_and_checkpoint(tmp_path):
    descriptors = np.arange(32, dtype=np.float32).reshape(16, 2)
    sequences = [
        (
            None
            if index % 3 == 0
            else [index % 7 + 1, (index + 2) % 7 + 1][: index % 2 + 1]
        )
        for index in range(16)
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
            "tokens": {
                "kind": "sequence",
                "optional": True,
            },
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
                "optional": True,
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
    assert saved_components["tokens"]["optional"] is True
    assert "tokens" in model.instance_branch_model.missing_representations

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
