from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from torch import nn

from DeepMTP.config import DeepMTPConfig
from DeepMTP.main import DeepMTP
from DeepMTP.models import IDEmbedding, SparseMLP, TabularEncoder
from DeepMTP.runtime.persistence import ExperimentStore
from DeepMTP.utils.data_utils import data_process
from DeepMTP.utils.utils import generate_config


class RecordingReporter:
    def __init__(self):
        self.start_calls = []
        self.metric_calls = []
        self.closed = False

    def start(self, config, model):
        self.start_calls.append((config, model))

    def log_metrics(self, metrics, *, step=None):
        self.metric_calls.append((dict(metrics), step))

    def close(self):
        self.closed = True


class RecordingProgress:
    def __init__(self):
        self.events = []
        self.summary_calls = []

    def on_event(self, event):
        self.events.append(event)

    def on_summary(
        self,
        summaries,
        *,
        include_train,
        include_validation,
    ):
        self.summary_calls.append((summaries, include_train, include_validation))


@pytest.mark.integration
@pytest.mark.filterwarnings(
    "ignore::DeepMTP.config.ConfigNormalizationWarning",
)
def test_tiny_cpu_sparse_training_prediction_and_checkpoint(tmp_path):
    rng = np.random.default_rng(18)
    dense_features = rng.normal(size=(20, 64)).astype(np.float32)
    dense_features[np.abs(dense_features) < 1.5] = 0.0
    instance_features = sparse.csr_matrix(dense_features)
    scores = (np.arange(20)[:, np.newaxis] + np.arange(4)[np.newaxis, :]) % 2
    train, val, test, data_info = data_process(
        {
            "train": {
                "y": scores,
                "X_instance": instance_features,
                "X_target": None,
            }
        },
        validation_setting="B",
        shuffle=False,
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
        gradient_accumulation_steps=2,
        gradient_clip_norm=1.0,
        metrics=["accuracy"],
        metrics_average=["macro"],
        evaluate_train=False,
        evaluate_val=True,
        use_early_stopping=False,
        save_model=False,
        results_path=str(tmp_path),
        experiment_name="sparse",
        instance_branch_architecture="SPARSE",
        instance_branch_input_dim=data_info["instance_branch_input_dim"],
        instance_branch_params={
            "instance_branch_nodes_per_layer": [8, 4],
            "instance_branch_layers": None,
            "instance_branch_nodes_reducing_factor": 2,
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

    validation_results = model.train(train, val, test)
    _, predictions = model.predict(test, return_predictions=True)

    assert isinstance(model.instance_branch_model, SparseMLP)
    assert model.training_engine.gradient_accumulation_steps == 2
    assert model.training_engine.gradient_clip_norm == 1.0
    assert "val_loss" in validation_results
    assert len(predictions) == len(test["y"]["data"])

    model.save_model()
    checkpoint_path = Path(model.experiment_dir) / "model.pt"
    checkpoint = ExperimentStore.load_checkpoint(checkpoint_path)
    restored_config = dict(config)
    restored_config["experiment_name"] = "sparse_restored"
    restored_model = DeepMTP(restored_config, checkpoint_dir=checkpoint_path)
    _, restored_predictions = restored_model.predict(test, return_predictions=True)

    assert checkpoint_path.is_file()
    assert checkpoint["config"]["gradient_accumulation_steps"] == 2
    assert checkpoint["config"]["gradient_clip_norm"] == 1.0
    assert checkpoint["metadata"]["training"]["gradient_accumulation_steps"] == 2
    assert checkpoint["metadata"]["training"]["gradient_clip_norm"] == 1.0
    assert restored_model.training_engine.gradient_accumulation_steps == 2
    assert restored_model.training_engine.gradient_clip_norm == 1.0
    assert np.allclose(
        predictions["predicted_values"],
        restored_predictions["predicted_values"],
    )


@pytest.mark.integration
@pytest.mark.filterwarnings(
    "ignore::DeepMTP.config.ConfigNormalizationWarning",
)
@pytest.mark.parametrize(
    (
        "problem_mode",
        "metrics",
        "validation_setting",
        "loss_name",
        "expected_loss_name",
        "loss_type",
        "loss_input_type",
    ),
    [
        (
            "classification",
            ["accuracy"],
            "B",
            None,
            "binary_cross_entropy_with_logits",
            nn.BCEWithLogitsLoss,
            nn.Identity,
        ),
        (
            "regression",
            ["MSE"],
            "B",
            None,
            "mean_squared_error",
            nn.MSELoss,
            nn.Identity,
        ),
        (
            "classification",
            ["accuracy"],
            "A",
            None,
            "binary_cross_entropy_with_logits",
            nn.BCEWithLogitsLoss,
            nn.Identity,
        ),
        (
            "classification",
            ["accuracy"],
            "B",
            "binary_cross_entropy",
            "binary_cross_entropy",
            nn.BCELoss,
            nn.Sigmoid,
        ),
        (
            "regression",
            ["MSE"],
            "B",
            "mean_absolute_error",
            "mean_absolute_error",
            nn.L1Loss,
            nn.Identity,
        ),
        (
            "regression",
            ["MSE"],
            "B",
            "huber",
            "huber",
            nn.HuberLoss,
            nn.Identity,
        ),
    ],
)
def test_tiny_cpu_training_and_prediction(
    tmp_path,
    problem_mode,
    metrics,
    validation_setting,
    loss_name,
    expected_loss_name,
    loss_type,
    loss_input_type,
):
    rng = np.random.default_rng(42)
    instance_features = rng.normal(size=(24, 3))
    if problem_mode == "classification":
        scores = (np.arange(24)[:, np.newaxis] + np.arange(4)[np.newaxis, :]) % 2
    else:
        scores = rng.normal(size=(24, 4))

    data = {
        "train": {
            "y": scores,
            "X_instance": instance_features,
            "X_target": None,
        }
    }
    train, val, test, data_info = data_process(
        data,
        validation_setting=validation_setting,
    )

    config = generate_config(
        validation_setting=data_info["detected_validation_setting"],
        problem_mode=data_info["detected_problem_mode"],
        loss=loss_name,
        general_architecture_version="dot_product",
        compute_mode="cpu",
        num_workers=0,
        train_batchsize=16,
        val_batchsize=16,
        num_epochs=1,
        metrics=metrics,
        metrics_average=["macro"],
        evaluate_train=False,
        evaluate_val=True,
        use_early_stopping=False,
        save_model=False,
        results_path=str(tmp_path),
        experiment_name=loss_name or problem_mode,
        instance_branch_architecture="MLP",
        instance_branch_input_dim=data_info["instance_branch_input_dim"],
        instance_branch_params={
            "instance_branch_nodes_per_layer": [4],
            "instance_branch_layers": None,
            "instance_branch_nodes_reducing_factor": 2,
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
    if validation_setting == "A":
        assert config["metrics_average"] == ["micro"]
    assert config["loss"] == expected_loss_name

    model_config = (
        DeepMTPConfig.from_mapping(config)
        if problem_mode == "classification"
        else config
    )
    reporter = RecordingReporter() if problem_mode == "classification" else None
    progress = RecordingProgress()
    model = DeepMTP(model_config, reporter=reporter, progress=progress)
    assert isinstance(model.configuration, DeepMTPConfig)
    assert isinstance(model.config, dict)
    assert isinstance(model.criterion, loss_type)
    assert isinstance(model.loss_input_transform, loss_input_type)
    if problem_mode == "classification":
        assert isinstance(model.output_head, nn.Sigmoid)
    else:
        assert isinstance(model.output_head, nn.Identity)
    validation_results = model.train(train, val, test)

    assert "val_loss" in validation_results
    assert (Path(model.experiment_dir) / "config.json").is_file()
    assert [event.kind for event in progress.events] == [
        "device_selected",
        "training_started",
        "epoch_started",
        "epoch_completed",
        "validation_started",
        "metrics_started",
        "metrics_completed",
        "validation_completed",
        "testing_started",
        "metrics_started",
        "metrics_completed",
        "testing_completed",
        "training_completed",
    ]
    event_by_kind = {
        event.kind: event
        for event in progress.events
        if event.kind
        in {"epoch_completed", "validation_completed", "testing_completed"}
    }
    assert "train_loss" in event_by_kind["epoch_completed"].metrics
    assert "val_loss" in event_by_kind["validation_completed"].metrics
    assert event_by_kind["testing_completed"].best_epoch == 0
    assert progress.events[-1].best_epoch == 0
    assert progress.events[-1].metrics == validation_results
    assert len(progress.summary_calls) == 1
    # Prediction output must be collected because it was requested, even when
    # the configured validation schedule would otherwise skip collection.
    model.config["num_epochs"] = 100
    model.config["evaluate_val"] = False
    model.configuration.num_epochs = 100
    model.configuration.evaluate_val = False
    results_only = model.predict(test, return_predictions=False)
    results_with_predictions, predictions = model.predict(test, return_predictions=True)
    assert isinstance(results_only, dict)
    assert results_only == results_with_predictions
    assert len(predictions) == len(test["y"]["data"])

    if problem_mode == "classification":
        assert len(reporter.start_calls) == 1
        assert reporter.closed
        assert [step for _, step in reporter.metric_calls] == [0, 0, None, None]
        reported_metrics = {
            key
            for metrics_for_step, _ in reporter.metric_calls
            for key in metrics_for_step
        }
        assert {
            "train_loss",
            "val_loss",
            "best_val_loss",
            "test_accuracy_micro"
            if validation_setting == "A"
            else "test_accuracy_macro",
        }.issubset(reported_metrics)

        model.save_model()
        checkpoint_path = Path(model.experiment_dir) / "model.pt"
        restored_config = dict(config)
        restored_config["experiment_name"] = "classification_restored"
        restored_model = DeepMTP(
            restored_config,
            checkpoint_dir=checkpoint_path,
        )
        _, restored_predictions = restored_model.predict(test, return_predictions=True)

        assert checkpoint_path.is_file()
        assert np.allclose(
            predictions["predicted_values"],
            restored_predictions["predicted_values"],
        )


@pytest.mark.integration
@pytest.mark.filterwarnings(
    "ignore::DeepMTP.config.ConfigNormalizationWarning",
)
def test_tiny_cpu_training_with_learned_id_embeddings(tmp_path):
    scores = (np.arange(24)[:, np.newaxis] + np.arange(4)[np.newaxis, :]) % 2
    train, val, test, data_info = data_process(
        {"train": {"y": scores, "X_instance": None, "X_target": None}},
        validation_setting="A",
    )
    config = generate_config(
        validation_setting="A",
        problem_mode="classification",
        general_architecture_version="dot_product",
        compute_mode="cpu",
        num_workers=0,
        train_batchsize=16,
        val_batchsize=16,
        num_epochs=1,
        metrics=["accuracy"],
        metrics_average=["micro"],
        evaluate_train=False,
        evaluate_val=True,
        use_early_stopping=False,
        save_model=False,
        results_path=str(tmp_path),
        experiment_name="learned_ids",
        instance_branch_architecture="EMBEDDING",
        instance_branch_input_dim=data_info["instance_branch_input_dim"],
        target_branch_architecture="EMBEDDING",
        target_branch_input_dim=data_info["target_branch_input_dim"],
        embedding_size=4,
    )
    model = DeepMTP(config)

    validation_results = model.train(train, val, test)
    _, predictions = model.predict(test, return_predictions=True)

    assert isinstance(model.instance_branch_model, IDEmbedding)
    assert isinstance(model.target_branch_model, IDEmbedding)
    assert "val_loss" in validation_results
    assert len(predictions) == len(test["y"]["data"])

    model.save_model()
    checkpoint_path = Path(model.experiment_dir) / "model.pt"
    restored_config = dict(config)
    restored_config["experiment_name"] = "learned_ids_restored"
    restored_model = DeepMTP(restored_config, checkpoint_dir=checkpoint_path)
    _, restored_predictions = restored_model.predict(test, return_predictions=True)

    assert checkpoint_path.is_file()
    assert np.allclose(
        predictions["predicted_values"],
        restored_predictions["predicted_values"],
    )


@pytest.mark.integration
@pytest.mark.filterwarnings(
    "ignore::DeepMTP.config.ConfigNormalizationWarning",
)
def test_tiny_cpu_multiclass_training_prediction_and_checkpoint(tmp_path):
    scores = (np.arange(18)[:, np.newaxis] + np.arange(3)[np.newaxis, :]) % 3
    train, val, test, data_info = data_process(
        {"train": {"y": scores, "X_instance": None, "X_target": None}},
        validation_setting="A",
        shuffle=False,
        classification_mode="multiclass",
    )
    config = generate_config(
        validation_setting="A",
        problem_mode="classification",
        classification_mode=data_info["detected_classification_mode"],
        num_classes=data_info["detected_num_classes"],
        general_architecture_version="dot_product",
        compute_mode="cpu",
        num_workers=0,
        train_batchsize=16,
        val_batchsize=16,
        num_epochs=1,
        metrics=["accuracy"],
        metrics_average=["micro"],
        evaluate_val=True,
        use_early_stopping=False,
        save_model=False,
        results_path=str(tmp_path),
        experiment_name="multiclass",
        instance_branch_architecture="EMBEDDING",
        instance_branch_input_dim=data_info["instance_branch_input_dim"],
        target_branch_architecture="EMBEDDING",
        target_branch_input_dim=data_info["target_branch_input_dim"],
        embedding_size=4,
    )
    model = DeepMTP(config)

    validation_results = model.train(train, val, test)
    _, predictions = model.predict(test, return_predictions=True)

    assert isinstance(model.criterion, nn.CrossEntropyLoss)
    assert isinstance(model.output_head, nn.Softmax)
    assert model.deepMTP_model.output_dim == 3
    assert "val_accuracy_micro" in validation_results
    assert {
        "predicted_probability",
        "probability_class_0",
        "probability_class_1",
        "probability_class_2",
    }.issubset(predictions.columns)
    assert np.allclose(
        predictions[
            [
                "probability_class_0",
                "probability_class_1",
                "probability_class_2",
            ]
        ].sum(axis=1),
        1.0,
    )

    model.save_model()
    checkpoint_path = Path(model.experiment_dir) / "model.pt"
    restored = DeepMTP(
        {
            **config,
            "experiment_name": "multiclass_restored",
        },
        checkpoint_dir=checkpoint_path,
    )
    _, restored_predictions = restored.predict(test, return_predictions=True)

    assert checkpoint_path.is_file()
    assert restored.configuration.classification_mode == "multiclass"
    assert restored.configuration.num_classes == 3
    assert np.allclose(
        predictions.filter(like="probability_"),
        restored_predictions.filter(like="probability_"),
    )


@pytest.mark.integration
@pytest.mark.filterwarnings(
    "ignore::DeepMTP.config.ConfigNormalizationWarning",
)
def test_tiny_cpu_training_with_mixed_tabular_features_restores_state(tmp_path):
    scores = (np.arange(12)[:, np.newaxis] + np.arange(3)[np.newaxis, :]) % 2
    instance_features = pd.DataFrame(
        {
            "id": np.arange(12),
            "measurement": [
                1.0,
                2.0,
                np.nan,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ],
            "group": ["A", "B", "A"] * 4,
        }
    )
    train, val, test, data_info = data_process(
        {
            "train": {
                "y": scores,
                "X_instance": instance_features,
                "X_target": None,
            }
        },
        validation_setting="B",
        seed=7,
    )
    config = generate_config(
        validation_setting=data_info["detected_validation_setting"],
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
        experiment_name="mixed_tabular",
        instance_branch_architecture="TABULAR",
        instance_branch_tabular_schema={
            "numeric_columns": ["measurement"],
            "categorical_columns": {
                "group": {
                    "categories": ["A", "B"],
                    "embedding_dim": 2,
                }
            },
            "numeric_normalization": "standard",
            "numeric_missing": "mean",
            "feature_gating": True,
        },
        instance_branch_params={
            "instance_branch_nodes_per_layer": [4],
        },
        target_branch_architecture="EMBEDDING",
        target_branch_input_dim=data_info["target_branch_input_dim"],
        embedding_size=4,
    )
    model = DeepMTP(config)

    model.train(train, val, test)
    _, predictions = model.predict(test, return_predictions=True)
    model.save_model()

    checkpoint_path = Path(model.experiment_dir) / "model.pt"
    checkpoint = ExperimentStore.load_checkpoint(checkpoint_path)
    saved_state = checkpoint["config"]["instance_tabular_preprocessing_state"]
    assert isinstance(model.instance_branch_model, TabularEncoder)
    assert saved_state["numeric_columns"] == ["measurement"]
    assert len(saved_state["numeric_fill_values"]) == 1

    restored_config = dict(config)
    restored_config["experiment_name"] = "mixed_tabular_restored"
    restored_model = DeepMTP(restored_config, checkpoint_dir=checkpoint_path)
    _, restored_predictions = restored_model.predict(test, return_predictions=True)

    assert (
        restored_model.configuration.instance_tabular_preprocessing_state == saved_state
    )
    assert np.allclose(
        predictions["predicted_values"],
        restored_predictions["predicted_values"],
    )
