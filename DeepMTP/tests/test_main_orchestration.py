import ast
import inspect
from pathlib import Path

import pytest

from DeepMTP.main import DeepMTP
from DeepMTP.persistence import ExperimentStore


class FailingReporter:
    def __init__(self):
        self.closed = False

    def start(self, config, model):
        raise RuntimeError("reporter failed to start")

    def log_metrics(self, metrics, *, step=None):
        raise AssertionError("metrics must not be logged after a failed start")

    def close(self):
        self.closed = True


class RecordingReporter:
    def __init__(self):
        self.start_calls = 0
        self.closed = False

    def start(self, config, model):
        self.start_calls += 1

    def log_metrics(self, metrics, *, step=None):
        return None

    def close(self):
        self.closed = True


class RecordingProgress:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)

    def on_summary(
        self,
        summaries,
        *,
        include_train,
        include_validation,
    ):
        return None


def make_config(tmp_path, *, experiment_name):
    return {
        "validation_setting": "B",
        "problem_mode": "classification",
        "compute_mode": "cpu",
        "num_workers": 0,
        "metrics": [],
        "metrics_average": ["macro"],
        "results_path": str(tmp_path),
        "experiment_name": experiment_name,
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": 2,
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 2,
    }


def test_trainer_materializes_synced_legacy_config_snapshot(tmp_path):
    trainer = DeepMTP(
        make_config(tmp_path, experiment_name="config_snapshot"),
    )

    assert trainer.configuration.experiment_dir == trainer.experiment_dir
    assert trainer.config["experiment_dir"] == trainer.experiment_dir
    assert trainer.config == trainer.configuration.to_dict(include_defaults=True)


def test_core_trainer_does_not_read_legacy_config_snapshot():
    tree = ast.parse(inspect.getsource(DeepMTP))
    legacy_reads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
        and node.value.attr == "config"
    ]

    assert legacy_reads == []


def test_reporter_is_closed_when_start_fails(tmp_path):
    reporter = FailingReporter()
    trainer = DeepMTP(
        make_config(tmp_path, experiment_name="reporter_failure"),
        reporter=reporter,
    )

    with pytest.raises(RuntimeError, match="failed to start"):
        trainer.train({}, {}, {})

    assert reporter.closed
    assert trainer._training_state == "ready"


def test_training_lifecycle_is_single_run(monkeypatch, tmp_path):
    reporter = RecordingReporter()
    progress = RecordingProgress()
    trainer = DeepMTP(
        make_config(tmp_path, experiment_name="single_run"),
        reporter=reporter,
        progress=progress,
    )
    expected_results = {"val_loss": 0.25}
    monkeypatch.setattr(
        trainer,
        "_train",
        lambda *args, **kwargs: expected_results,
    )

    assert trainer.train({}, {}, {}) == expected_results

    assert trainer._training_state == "completed"
    assert reporter.start_calls == 1
    assert reporter.closed
    assert [event.kind for event in progress.events] == [
        "training_started",
        "training_completed",
    ]
    assert progress.events[-1].metrics == expected_results

    with pytest.raises(RuntimeError, match="supports one training run"):
        trainer.train({}, {}, {})

    assert reporter.start_calls == 1


def test_save_model_requires_completed_training(tmp_path):
    trainer = DeepMTP(
        make_config(tmp_path, experiment_name="untrained_save"),
    )

    with pytest.raises(RuntimeError, match="before training"):
        trainer.save_model()

    assert not (Path(trainer.experiment_dir) / "model.pt").exists()


def test_save_model_does_not_mutate_live_logging_config(tmp_path):
    trainer = DeepMTP(
        make_config(tmp_path, experiment_name="checkpoint_config"),
    )
    trainer.config["use_tensorboard_logger"] = True
    trainer.configuration.use_tensorboard_logger = True
    trainer.early_stopping(
        {"val_loss": 0.25},
        trainer.deepMTP_model,
        0,
        optimizer_state_dict=trainer.optimizer.state_dict(),
    )

    trainer.save_model()

    checkpoint = ExperimentStore.load_checkpoint(
        Path(trainer.experiment_dir) / "model.pt"
    )
    assert trainer.config["use_tensorboard_logger"] is True
    assert trainer.configuration.use_tensorboard_logger is True
    assert checkpoint["config"]["use_tensorboard_logger"] is False


def test_checkpoint_config_must_be_a_mapping(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ExperimentStore,
        "load_checkpoint",
        lambda *args, **kwargs: {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "config": ["not", "a", "mapping"],
        },
    )

    with pytest.raises(ValueError, match="checkpoint config"):
        DeepMTP(
            make_config(tmp_path, experiment_name="bad_checkpoint"),
            checkpoint_dir="ignored.pt",
        )
