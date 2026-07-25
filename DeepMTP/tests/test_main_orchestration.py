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


def test_reporter_is_closed_when_start_fails(tmp_path):
    reporter = FailingReporter()
    trainer = DeepMTP(
        make_config(tmp_path, experiment_name="reporter_failure"),
        reporter=reporter,
    )

    with pytest.raises(RuntimeError, match="failed to start"):
        trainer.train({}, {}, {})

    assert reporter.closed


def test_save_model_requires_completed_training(tmp_path):
    trainer = DeepMTP(
        make_config(tmp_path, experiment_name="untrained_save"),
    )

    with pytest.raises(RuntimeError, match="before training"):
        trainer.save_model()

    assert not (Path(trainer.experiment_dir) / "model.pt").exists()


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
