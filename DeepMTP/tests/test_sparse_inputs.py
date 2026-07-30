import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from DeepMTP import DataLoaderFactory, DeepMTPConfig, ModelFactory
from DeepMTP.data import SparseBranchBatch, data_process
from DeepMTP.data.features import (
    process_instance_features,
    process_target_features,
)


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
        "instance_branch_architecture": "SPARSE",
        "instance_branch_input_dim": 10_000,
        "instance_branch_nodes_per_layer": [4],
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 2,
        "target_branch_nodes_per_layer": [4],
        "embedding_size": 4,
        "random_seed": 5,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


def make_sparse_matrix(width=10_000):
    return sparse.csr_matrix(
        (
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            (
                np.array([0, 1, 2, 3]),
                np.array([1, width - 1, min(7, width - 1), width // 2]),
            ),
        ),
        shape=(4, width),
    )


def make_prediction_data(features):
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
        "X_instance": process_instance_features(features),
        "X_target": None,
    }


@pytest.mark.parametrize(
    ("processor", "expected_info"),
    [
        (process_instance_features, "scipy_sparse"),
        (process_target_features, "scipy_sparse"),
    ],
)
def test_scipy_sparse_matrices_are_normalized_without_densifying(
    processor,
    expected_info,
):
    features = make_sparse_matrix()

    processed = processor(features)

    assert processed["info"] == expected_info
    assert processed["num_features"] == 10_000
    assert processed["data"]["id"].tolist() == [0, 1, 2, 3]
    assert all(sparse.issparse(row) for row in processed["data"]["features"])
    assert all(row.shape == (1, 10_000) for row in processed["data"]["features"])
    assert processed["data"]["features"].map(lambda row: row.nnz).sum() == 4


@pytest.mark.filterwarnings(
    "ignore:Sparse CSR tensor support is in beta state:UserWarning",
)
def test_torch_sparse_matrices_are_normalized_as_sparse_rows():
    dense = torch.tensor(
        [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]],
        dtype=torch.float64,
    )

    processed = process_instance_features(dense.to_sparse_csr())

    assert processed["info"] == "torch_sparse"
    assert processed["num_features"] == 3
    assert all(row.layout == torch.sparse_coo for row in processed["data"]["features"])
    assert all(row.ndim == 1 for row in processed["data"]["features"])


def test_sparse_dataframe_rows_keep_explicit_entity_ids():
    frame = pd.DataFrame(
        {
            "id": [10, 30],
            "features": [
                sparse.csr_matrix([[0.0, 1.0, 0.0]]),
                sparse.csr_matrix([[2.0, 0.0, 3.0]]),
            ],
        }
    )

    processed = process_instance_features(frame)

    assert processed["data"].index.tolist() == [10, 30]
    assert processed["data"]["id"].tolist() == [10, 30]
    assert processed["num_features"] == 3


def test_sparse_feature_validation_rejects_mixed_or_nonfinite_rows():
    mixed = pd.DataFrame(
        {
            "id": [0, 1],
            "features": [
                sparse.csr_matrix([[0.0, 1.0]]),
                np.array([2.0, 0.0]),
            ],
        }
    )
    nonfinite = sparse.csr_matrix([[0.0, np.nan]])

    with pytest.raises(TypeError, match="cannot mix dense and sparse"):
        process_instance_features(mixed)
    with pytest.raises(ValueError, match="must be finite"):
        process_instance_features(nonfinite)


def test_prediction_loader_and_model_keep_high_dimensional_batches_sparse():
    data = make_prediction_data(make_sparse_matrix())
    config = make_config()

    batch = next(iter(DataLoaderFactory(config, "cpu").for_prediction(data)))
    bundle = ModelFactory(config).build()
    output = bundle.model(batch.instance_features, batch.target_features)

    assert isinstance(batch.instance_input, SparseBranchBatch)
    assert batch.instance_features.layout == torch.sparse_coo
    assert batch.instance_features.shape == (2, 10_000)
    assert batch.instance_features._nnz() == 2
    assert output.shape == (2, 1)


def test_sparse_loader_rejects_width_and_architecture_mismatches():
    data = make_prediction_data(make_sparse_matrix(width=12))

    with pytest.raises(ValueError, match="expected 10, received 12"):
        DataLoaderFactory(
            make_config(instance_branch_input_dim=10),
            "cpu",
        ).for_prediction(data)

    with pytest.raises(TypeError, match="architecture='SPARSE'"):
        DataLoaderFactory(
            make_config(
                instance_branch_architecture="MLP",
                instance_branch_input_dim=12,
            ),
            "cpu",
        ).for_prediction(data)

    dense_data = make_prediction_data(np.ones((4, 12), dtype=np.float32))
    with pytest.raises(TypeError, match="Sparse features must be"):
        DataLoaderFactory(
            make_config(instance_branch_input_dim=12),
            "cpu",
        ).for_prediction(dense_data)


def test_legacy_feature_scaling_rejects_sparse_inputs_without_densifying():
    scores = (np.arange(12)[:, None] + np.arange(3)[None, :]) % 2
    features = sparse.eye(12, 256, dtype=np.float32, format="csr")

    with pytest.raises(ValueError, match="without densifying"):
        data_process(
            {"train": {"y": scores, "X_instance": features, "X_target": None}},
            validation_setting="B",
            scale_instance_features="Standard",
        )
