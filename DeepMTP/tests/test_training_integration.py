from pathlib import Path

import numpy as np
import pytest

from DeepMTP.config import DeepMTPConfig
from DeepMTP.main import DeepMTP
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
@pytest.mark.parametrize(
    ("problem_mode", "metrics"),
    [("classification", ["accuracy"]), ("regression", ["MSE"])],
)
def test_tiny_cpu_training_and_prediction(tmp_path, problem_mode, metrics):
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
    train, val, test, data_info = data_process(data, validation_setting="B")

    config = generate_config(
        validation_setting=data_info["detected_validation_setting"],
        problem_mode=data_info["detected_problem_mode"],
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
        experiment_name=problem_mode,
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
    validation_results = model.train(train, val, test)

    assert "val_loss" in validation_results
    assert (Path(model.experiment_dir) / "config.json").is_file()
    assert [event.kind for event in progress.events] == [
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
            "test_accuracy_macro",
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
