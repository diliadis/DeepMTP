import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import DeepMTP.main as main_module
from DeepMTP import EpochOutput
from DeepMTP.config import ConfigValidationError
from DeepMTP.main import DeepMTP
from DeepMTP.persistence import ExperimentStore
from DeepMTP.utils.model_utils import EarlyStopping


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


def test_invalid_combined_mlp_fails_before_experiment_creation(tmp_path):
    config = make_config(tmp_path, experiment_name="invalid_combined_mlp")
    config.update(
        {
            "general_architecture_version": "mlp",
            "comb_mlp_nodes_per_layer": [],
        }
    )

    with pytest.raises(ConfigValidationError, match="comb_mlp_nodes_per_layer"):
        DeepMTP(config)

    assert not (tmp_path / "invalid_combined_mlp").exists()


def test_invalid_convolutional_config_fails_before_experiment_creation(tmp_path):
    config = make_config(tmp_path, experiment_name="invalid_conv")
    config.update(
        {
            "instance_branch_architecture": "CONV",
            "instance_branch_conv_architecture": "resnet",
            "instance_branch_conv_architecture_version": "resnet50",
        }
    )

    with pytest.raises(
        ConfigValidationError,
        match="instance_branch_conv_architecture_version",
    ):
        DeepMTP(config)

    assert not (tmp_path / "invalid_conv").exists()


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


def test_core_trainer_delegates_output_to_progress_observer():
    tree = ast.parse(inspect.getsource(DeepMTP))
    direct_prints = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

    assert direct_prints == []


def test_training_routes_early_stopping_updates_to_progress_observer(capsys):
    trainer = DeepMTP.__new__(DeepMTP)
    trainer.configuration = SimpleNamespace(
        metrics=(),
        metrics_average=(),
        num_epochs=2,
        evaluate_train=False,
        evaluate_val=True,
        use_early_stopping=True,
        verbose=False,
        save_model=False,
    )
    trainer.config = {}
    trainer.deepMTP_model = torch.nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(trainer.deepMTP_model.parameters(), lr=0.1)
    trainer.early_stopping = EarlyStopping(True, patience=2, verbose=True)
    trainer.progress = RecordingProgress()
    trainer.data_loader_factory = SimpleNamespace(
        for_training=lambda *_: SimpleNamespace(
            train=(),
            validation=(),
            test=(),
        )
    )
    trainer.training_engine = SimpleNamespace(
        train_epoch=lambda *_args, **_kwargs: EpochOutput(
            losses=[0.25],
            loss_weights=[1],
        ),
        replace_model=lambda _model: None,
    )
    trainer.evaluator = SimpleNamespace(
        policy=SimpleNamespace(should_evaluate_training=lambda _epoch: False)
    )
    results = iter(
        [
            {"val_loss": 0.5},
            {"val_loss": 0.6},
            {"test_loss": 0.55},
        ]
    )
    trainer.inference = lambda *_args, **_kwargs: next(results)
    trainer.experiment_store = SimpleNamespace(
        save_summary=lambda *_args: None,
        save_config=lambda _config: None,
    )

    validation_results = trainer._train(
        {},
        {},
        {},
        reporter=RecordingReporter(),
    )

    counter_event = next(
        event
        for event in trainer.progress.events
        if event.kind == "early_stopping_counter_updated"
    )
    event_kinds = [event.kind for event in trainer.progress.events]
    assert validation_results == {"val_loss": 0.5}
    assert counter_event.epoch == 1
    assert counter_event.early_stopping_counter == 1
    assert counter_event.early_stopping_patience == 2
    assert counter_event.best_epoch == 0
    counter_event_index = event_kinds.index("early_stopping_counter_updated")
    assert event_kinds[counter_event_index + 1] == "validation_completed"
    assert capsys.readouterr().out == ""


def test_trainer_builds_reporter_from_typed_configuration(monkeypatch, tmp_path):
    captured_configs = []
    reporter = RecordingReporter()

    def build_recording_reporter(config, experiment_dir):
        captured_configs.append(config)
        return reporter

    monkeypatch.setattr(main_module, "build_reporter", build_recording_reporter)
    trainer = DeepMTP(
        make_config(tmp_path, experiment_name="typed_reporter_config"),
    )
    monkeypatch.setattr(
        trainer,
        "_train",
        lambda *args, **kwargs: {"val_loss": 0.25},
    )

    trainer.train({}, {}, {})

    assert captured_configs == [trainer.configuration]
    assert reporter.closed


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
        "device_selected",
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


def test_checkpoint_lifecycle_uses_injected_progress_without_stdout(
    tmp_path,
    capsys,
):
    source_progress = RecordingProgress()
    source = DeepMTP(
        make_config(tmp_path, experiment_name="checkpoint_source"),
        progress=source_progress,
    )
    source.early_stopping(
        {"val_loss": 0.25},
        source.deepMTP_model,
        0,
        optimizer_state_dict=source.optimizer.state_dict(),
    )
    source_progress.events.clear()

    source.save_model()

    checkpoint_path = Path(source.experiment_dir) / "model.pt"
    assert [event.kind for event in source_progress.events] == [
        "checkpoint_saving_started",
        "checkpoint_saving_completed",
    ]
    assert source_progress.events[-1].message == str(checkpoint_path)

    restored_progress = RecordingProgress()
    restored_config = make_config(tmp_path, experiment_name="checkpoint_restored")
    restored_config["verbose"] = True
    restored = DeepMTP(
        restored_config,
        checkpoint_dir=checkpoint_path,
        progress=restored_progress,
    )

    assert restored.checkpoint_dict is not None
    assert [event.kind for event in restored_progress.events] == [
        "checkpoint_loading_started",
        "checkpoint_loading_completed",
        "device_selected",
        "checkpoint_weights_started",
        "checkpoint_weights_completed",
    ]
    assert restored_progress.events[0].message == str(checkpoint_path)
    assert restored_progress.events[2].message == "cpu"
    assert capsys.readouterr().out == ""


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
