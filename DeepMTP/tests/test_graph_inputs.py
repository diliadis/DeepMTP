from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from DeepMTP import DataLoaderFactory, DeepMTP, DeepMTPConfig, ModelFactory
from DeepMTP.data import (
    CompositeBranchBatch,
    GraphBranchBatch,
    GraphInput,
    data_process,
)
from DeepMTP.data.features import process_instance_features, process_target_features
from DeepMTP.data.graphs import _load_pyg_data, normalize_graph_rows
from DeepMTP.models import GraphEncoder
from DeepMTP.runtime.persistence import ExperimentStore
from DeepMTP.utils.utils import generate_config


def make_graph(index: int, *, edge_features: bool = True) -> Data:
    node_count = index % 3 + 2
    node_features = torch.arange(
        node_count * 3,
        dtype=torch.float32,
    ).reshape(node_count, 3)
    node_features = node_features / 10 + index
    sources = torch.arange(node_count - 1)
    targets = sources + 1
    edge_index = torch.stack(
        [
            torch.cat([sources, targets]),
            torch.cat([targets, sources]),
        ]
    )
    edge_attr = (
        torch.stack(
            [
                torch.ones(edge_index.shape[1]),
                torch.full((edge_index.shape[1],), float(index + 1)),
            ],
            dim=1,
        )
        if edge_features
        else None
    )
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


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
        "instance_branch_architecture": "GRAPH",
        "instance_branch_input_dim": 3,
        "instance_branch_graph_edge_dim": 2,
        "instance_branch_graph_hidden_dim": 8,
        "instance_branch_graph_output_dim": 4,
        "instance_branch_graph_num_layers": 2,
        "instance_branch_graph_pooling": "mean",
        "instance_branch_graph_use_edge_features": True,
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 2,
        "target_branch_nodes_per_layer": [4],
        "embedding_size": 4,
        "random_seed": 5,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


def make_prediction_data(graphs):
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
        "X_instance": process_instance_features(graphs, feature_kind="graph"),
        "X_target": None,
    }


def test_graph_features_are_validated_cloned_and_described():
    graphs = [make_graph(0), make_graph(1)]

    processed = process_instance_features(graphs, feature_kind="graph")

    assert processed["info"] == "graph"
    assert processed["num_features"] == 3
    assert processed["graph_edge_features"] == 2
    assert processed["data"]["id"].tolist() == [0, 1]
    assert processed["data"].iloc[0]["features"] is not graphs[0]
    assert processed["data"].iloc[0]["features"].x.data_ptr() != graphs[0].x.data_ptr()


def test_graph_dataframe_preserves_explicit_entity_ids():
    processed = process_instance_features(
        pd.DataFrame(
            {
                "id": [10, 30],
                "features": [make_graph(0), make_graph(1)],
            }
        ),
        feature_kind="graph",
    )

    assert processed["data"].index.tolist() == [10, 30]
    assert processed["num_features"] == 3
    assert processed["graph_edge_features"] == 2


@pytest.mark.parametrize(
    ("graphs", "message"),
    [
        (
            [
                make_graph(0),
                Data(
                    x=torch.ones(2, 4),
                    edge_index=torch.tensor([[0, 1], [1, 0]]),
                    edge_attr=torch.ones(2, 2),
                ),
            ],
            "same node-feature width",
        ),
        ([make_graph(0), make_graph(1, edge_features=False)], "either omit"),
    ],
)
def test_graph_features_reject_inconsistent_widths(graphs, message):
    with pytest.raises(ValueError, match=message):
        process_instance_features(graphs, feature_kind="graph")


def test_graph_features_reject_invalid_edges_and_nonfinite_values():
    invalid_edges = Data(
        x=torch.ones(2, 3),
        edge_index=torch.tensor([[0, 2], [1, 0]]),
        edge_attr=torch.ones(2, 2),
    )
    nonfinite = make_graph(0)
    nonfinite.x[0, 0] = torch.nan

    with pytest.raises(ValueError, match="invalid"):
        process_instance_features([invalid_edges], feature_kind="graph")
    with pytest.raises(ValueError, match="finite"):
        process_instance_features([nonfinite], feature_kind="graph")


def test_graph_collator_offsets_edges_and_exposes_framework_neutral_input():
    data = make_prediction_data([make_graph(index) for index in range(4)])

    batch = next(iter(DataLoaderFactory(make_config(), "cpu").for_prediction(data)))

    assert isinstance(batch.instance_input, GraphBranchBatch)
    assert isinstance(batch.instance_features, GraphInput)
    assert batch.instance_features.batch_size == 2
    assert batch.instance_features.node_features.shape[1] == 3
    assert batch.instance_features.edge_features.shape[1] == 2
    assert set(batch.instance_features.graph_index.tolist()) == {0, 1}
    first_graph_nodes = make_graph(0).num_nodes
    second_graph_edges = batch.instance_features.edge_index[
        :,
        make_graph(0).num_edges :,
    ]
    assert int(second_graph_edges.min()) >= first_graph_nodes


@pytest.mark.parametrize("pooling", ["max", "mean", "sum"])
def test_graph_encoder_supports_pooling_and_backpropagation(pooling):
    batch = next(
        iter(
            DataLoaderFactory(
                make_config(instance_branch_graph_pooling=pooling),
                "cpu",
            ).for_prediction(
                make_prediction_data([make_graph(index) for index in range(4)])
            )
        )
    )
    bundle = ModelFactory(make_config(instance_branch_graph_pooling=pooling)).build()

    prediction = bundle.model(batch.instance_features, batch.target_features)
    prediction.sum().backward()

    assert isinstance(bundle.instance_branch, GraphEncoder)
    assert prediction.shape == (2, 1)
    assert bundle.instance_branch.node_projection.weight.grad is not None


def test_graph_encoder_can_ignore_absent_edge_features():
    config = make_config(
        instance_branch_graph_edge_dim=None,
        instance_branch_graph_use_edge_features=False,
    )
    batch = next(
        iter(
            DataLoaderFactory(config, "cpu").for_prediction(
                make_prediction_data(
                    [make_graph(index, edge_features=False) for index in range(4)]
                )
            )
        )
    )

    output = (
        ModelFactory(config)
        .build()
        .model(
            batch.instance_features,
            batch.target_features,
        )
    )

    assert batch.instance_features.edge_features is None
    assert output.shape == (2, 1)


def test_target_graph_branch_uses_the_same_typed_contract():
    config = make_config(
        instance_branch_architecture="MLP",
        instance_branch_input_dim=4,
        instance_branch_nodes_per_layer=[4],
        target_branch_architecture="GRAPH",
        target_branch_input_dim=3,
        target_branch_graph_edge_dim=2,
        target_branch_graph_hidden_dim=8,
        target_branch_graph_output_dim=4,
        target_branch_graph_num_layers=2,
        target_branch_graph_pooling="sum",
        target_branch_graph_use_edge_features=True,
    )
    data = {
        "y": {
            "data": pd.DataFrame(
                {
                    "instance_id": [0, 1, 2, 3],
                    "target_id": [0, 1, 0, 1],
                    "value": [0, 1, 1, 0],
                }
            )
        },
        "X_instance": None,
        "X_target": process_target_features(
            [make_graph(0), make_graph(1)],
            feature_kind="graph",
        ),
    }

    batch = next(iter(DataLoaderFactory(config, "cpu").for_prediction(data)))
    bundle = ModelFactory(config).build()
    output = bundle.model(batch.instance_features, batch.target_features)

    assert isinstance(batch.target_input, GraphBranchBatch)
    assert isinstance(bundle.target_branch, GraphEncoder)
    assert output.shape == (2, 1)


def test_graph_config_validation_is_actionable():
    with pytest.raises(ValueError, match="graph_edge_dim"):
        make_config(instance_branch_graph_edge_dim=None)
    with pytest.raises(ValueError, match="graph_pooling"):
        make_config(instance_branch_graph_pooling="median")


def test_data_process_infers_graph_node_and_edge_dimensions():
    scores = (np.arange(8)[:, None] + np.arange(3)[None, :]) % 2

    _, _, _, data_info = data_process(
        {
            "train": {
                "y": scores,
                "X_instance": [make_graph(index) for index in range(8)],
                "X_target": None,
            }
        },
        validation_setting="B",
        shuffle=False,
        instance_feature_kind="graph",
    )

    assert data_info["instance_branch_input_dim"] == 3
    assert data_info["instance_branch_graph_edge_dim"] == 2
    assert data_info["target_branch_graph_edge_dim"] is None


def test_legacy_scaling_rejects_graph_features():
    scores = (np.arange(8)[:, None] + np.arange(3)[None, :]) % 2

    with pytest.raises(ValueError, match="uses graph features"):
        data_process(
            {
                "train": {
                    "y": scores,
                    "X_instance": [make_graph(index) for index in range(8)],
                    "X_target": None,
                }
            },
            validation_setting="B",
            shuffle=False,
            instance_feature_kind="graph",
            scale_instance_features="Standard",
        )


def test_composite_branch_accepts_an_optional_graph_component():
    graphs = [make_graph(index) for index in range(4)]
    graphs[1] = None
    processed = process_instance_features(
        {
            "molecule": graphs,
            "descriptors": np.ones((4, 2), dtype=np.float32),
        },
        feature_kind={
            "molecule": {"kind": "graph", "optional": True},
            "descriptors": None,
        },
    )
    data = {
        "y": {
            "data": pd.DataFrame(
                {
                    "instance_id": [0, 1, 2, 3],
                    "target_id": [0, 1, 0, 1],
                    "value": [0, 1, 1, 0],
                }
            )
        },
        "X_instance": processed,
        "X_target": None,
    }
    config = make_config(
        instance_branch_architecture="COMPOSITE",
        instance_branch_input_dim=None,
        instance_branch_composite_components={
            "molecule": {
                "architecture": "GRAPH",
                "input_dim": 3,
                "edge_dim": 2,
                "hidden_dim": 8,
                "output_dim": 2,
                "num_layers": 2,
                "pooling": "mean",
                "use_edge_features": True,
                "optional": True,
            },
            "descriptors": {
                "architecture": "MLP",
                "input_dim": 2,
                "output_dim": 2,
                "nodes_per_layer": [2],
                "optional": False,
            },
        },
    )

    batch = next(iter(DataLoaderFactory(config, "cpu").for_prediction(data)))
    output = (
        ModelFactory(config)
        .build()
        .model(
            batch.instance_features,
            batch.target_features,
        )
    )

    assert isinstance(batch.instance_input, CompositeBranchBatch)
    assert batch.instance_features["molecule"].presence_mask.tolist() == [True, False]
    assert output.shape == (2, 1)


def test_missing_pyg_dependency_has_installation_guidance(monkeypatch):
    import DeepMTP.data.graphs as graph_helpers

    _load_pyg_data.cache_clear()
    real_import = graph_helpers.importlib.import_module

    def missing_pyg(name):
        if name == "torch_geometric.data":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(graph_helpers.importlib, "import_module", missing_pyg)
    try:
        with pytest.raises(ModuleNotFoundError, match=r"DeepMTP\[graph\]"):
            normalize_graph_rows([make_graph(0)], entity_name="instance")
    finally:
        _load_pyg_data.cache_clear()


@pytest.mark.integration
@pytest.mark.filterwarnings(
    "ignore::DeepMTP.config.ConfigNormalizationWarning",
)
def test_tiny_cpu_graph_training_prediction_and_checkpoint(tmp_path):
    scores = (np.arange(12)[:, None] + np.arange(4)[None, :]) % 2
    train, validation, test, data_info = data_process(
        {
            "train": {
                "y": scores,
                "X_instance": [make_graph(index) for index in range(12)],
                "X_target": None,
            }
        },
        validation_setting="B",
        shuffle=False,
        instance_feature_kind="graph",
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
        experiment_name="graph",
        instance_branch_architecture="GRAPH",
        instance_branch_input_dim=data_info["instance_branch_input_dim"],
        instance_branch_params={
            "instance_branch_graph_edge_dim": data_info[
                "instance_branch_graph_edge_dim"
            ],
            "instance_branch_graph_hidden_dim": 8,
            "instance_branch_graph_output_dim": 4,
            "instance_branch_graph_num_layers": 2,
            "instance_branch_graph_pooling": "mean",
            "instance_branch_graph_use_edge_features": True,
        },
        target_branch_architecture="MLP",
        target_branch_input_dim=data_info["target_branch_input_dim"],
        target_branch_params={"target_branch_nodes_per_layer": [4]},
        embedding_size=4,
    )
    model = DeepMTP(config)

    validation_results = model.train(train, validation, test)
    _, predictions = model.predict(test, return_predictions=True)

    assert isinstance(model.instance_branch_model, GraphEncoder)
    assert "val_loss" in validation_results
    assert len(predictions) == len(test["y"]["data"])

    model.save_model()
    checkpoint_path = Path(model.experiment_dir) / "model.pt"
    checkpoint = ExperimentStore.load_checkpoint(checkpoint_path)
    assert checkpoint["config"]["instance_branch_graph_edge_dim"] == 2

    restored_config = dict(config)
    restored_config["experiment_name"] = "graph_restored"
    restored_model = DeepMTP(restored_config, checkpoint_dir=checkpoint_path)
    _, restored_predictions = restored_model.predict(
        test,
        return_predictions=True,
    )
    assert np.allclose(
        predictions["predicted_values"],
        restored_predictions["predicted_values"],
    )
