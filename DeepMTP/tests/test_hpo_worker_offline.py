from pathlib import Path

import pytest

import DeepMTP.hpo.worker as hpo_worker
from DeepMTP.hpo import BaseWorker


class RecordingProgress:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


class FakeTrainer:
    def __init__(self, config, checkpoint_dir):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.experiment_dir = config["results_path"]

    def train(self, train, val, test):
        return {"val_loss": 0.25}


def test_worker_joins_nested_result_path_without_trailing_separator(tmp_path):
    results_path = tmp_path / "nested" / "hpo"

    worker = BaseWorker(
        train=None,
        val=None,
        test=None,
        data_info=None,
        base_config={"hpo_results_path": str(results_path)},
        metric_to_optimize="loss",
    )

    project_path = Path(worker.project_name)
    assert project_path.parent == results_path
    assert project_path.is_dir()


def test_worker_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="Invalid mode"):
        BaseWorker(
            train=None,
            val=None,
            test=None,
            data_info=None,
            base_config={"hpo_results_path": str(tmp_path)},
            metric_to_optimize="loss",
            mode="unknown",
        )


def test_worker_emits_final_configuration_without_direct_output(
    monkeypatch,
    tmp_path,
    capsys,
):
    progress = RecordingProgress()
    worker = BaseWorker(
        train=None,
        val=None,
        test=None,
        data_info=None,
        base_config={
            "hpo_results_path": str(tmp_path),
            "verbose": True,
        },
        metric_to_optimize="loss",
        progress=progress,
    )
    worker.trainer_type = FakeTrainer
    monkeypatch.setattr(
        hpo_worker,
        "generate_config",
        lambda **config: config,
    )

    result = worker.compute(1, {})

    assert result["loss"] == 0.25
    assert [event.kind for event in progress.events] == ["worker_configured"]
    assert progress.events[0].config is result["info"]["config"]
    assert capsys.readouterr().out == ""
