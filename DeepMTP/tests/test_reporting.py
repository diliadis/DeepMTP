from types import SimpleNamespace

import pytest

import DeepMTP.main as main_module
import DeepMTP.reporting as reporting_module
from DeepMTP import DeepMTPConfig
from DeepMTP.reporting import (
    CompositeReporter,
    NullReporter,
    OptionalIntegrationError,
    TensorBoardReporter,
    WandBReporter,
    build_reporter,
)


class RecordingReporter:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def start(self, config, model):
        self.events.append((self.name, "start"))

    def log_metrics(self, metrics, *, step=None):
        self.events.append((self.name, "metrics", dict(metrics), step))

    def close(self):
        self.events.append((self.name, "close"))


def base_reporting_config(**overrides):
    config = {
        "use_tensorboard_logger": False,
        "wandb_project_name": None,
        "wandb_project_entity": None,
    }
    config.update(overrides)
    return config


def typed_reporting_config(**overrides):
    values = {
        "validation_setting": "B",
        "problem_mode": "classification",
        "compute_mode": "cpu",
        "metrics": [],
        "metrics_average": ["macro"],
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": 2,
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 2,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


def test_composite_reporter_fans_out_and_closes_in_reverse_order():
    events = []
    reporter = CompositeReporter(
        [
            RecordingReporter(events, "first"),
            RecordingReporter(events, "second"),
        ]
    )

    reporter.start({}, object())
    reporter.log_metrics({"loss": 1.0}, step=3)
    reporter.close()

    assert events == [
        ("first", "start"),
        ("second", "start"),
        ("first", "metrics", {"loss": 1.0}, 3),
        ("second", "metrics", {"loss": 1.0}, 3),
        ("second", "close"),
        ("first", "close"),
    ]


def test_trainer_closes_reporter_when_training_fails():
    events = []
    reporter = RecordingReporter(events, "reporter")
    trainer = main_module.DeepMTP.__new__(main_module.DeepMTP)
    trainer.reporter = reporter
    trainer.config = {}
    trainer.deepMTP_model = object()
    progress_events = []
    trainer.progress = SimpleNamespace(
        on_event=lambda event: progress_events.append(event)
    )

    def fail_training(*args, **kwargs):
        raise RuntimeError("training failed")

    trainer._train = fail_training

    with pytest.raises(RuntimeError, match="training failed"):
        trainer.train(None, None, None)

    assert events == [("reporter", "start"), ("reporter", "close")]
    assert trainer._training_state == "failed"
    assert [event.kind for event in progress_events] == [
        "training_started",
        "training_failed",
    ]
    assert progress_events[-1].message == "training failed"


def test_disabled_integrations_do_not_import_optional_modules(monkeypatch, tmp_path):
    def fail_import(module_name):
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(reporting_module.importlib, "import_module", fail_import)
    reporter = build_reporter(base_reporting_config(), tmp_path)

    assert isinstance(reporter, NullReporter)
    reporter.start({}, object())
    reporter.log_metrics({"loss": 1.0}, step=0)
    reporter.close()


def test_reporter_builder_accepts_typed_configuration(tmp_path):
    reporter = build_reporter(
        typed_reporting_config(use_tensorboard_logger=True),
        tmp_path,
    )

    assert isinstance(reporter, TensorBoardReporter)


def test_tensorboard_reporter_preserves_legacy_metric_names(monkeypatch, tmp_path):
    writer_events = []

    class FakeWriter:
        def __init__(self, log_dir):
            writer_events.append(("init", log_dir))

        def add_text(self, key, value):
            writer_events.append(("text", key, value))

        def add_scalar(self, key, value, step):
            writer_events.append(("scalar", key, value, step))

        def flush(self):
            writer_events.append(("flush",))

        def close(self):
            writer_events.append(("close",))

    monkeypatch.setattr(
        reporting_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(SummaryWriter=FakeWriter),
    )
    reporter = TensorBoardReporter(tmp_path)

    reporter.start({"learning_rate": 0.1}, object())
    reporter.log_metrics({"val_loss": 2.0}, step=4)
    reporter.log_metrics({"best_val_loss": 1.0})
    reporter.close()

    assert ("text", "learning_rate", "0.1") in writer_events
    assert ("scalar", "val/loss", 2.0, 4) in writer_events
    assert ("scalar", "best/val/loss", 1.0, 4) in writer_events
    assert writer_events[-2:] == [("flush",), ("close",)]


def test_wandb_reporter_uses_run_lifecycle(monkeypatch):
    run_events = []

    class FakeConfig:
        def update(self, config):
            run_events.append(("config", config))

    class FakeRun:
        config = FakeConfig()

        def watch(self, model):
            run_events.append(("watch", model))

        def log(self, metrics, step=None):
            run_events.append(("log", metrics, step))

        def finish(self):
            run_events.append(("finish",))

    def fake_init(**kwargs):
        run_events.append(("init", kwargs))
        return FakeRun()

    monkeypatch.setattr(
        reporting_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(init=fake_init),
    )
    model = object()
    reporter = WandBReporter("project", "entity")

    reporter.start({"learning_rate": 0.1}, model)
    reporter.log_metrics({"val_loss": 2.0}, step=4)
    reporter.log_metrics({"best_val_loss": 1.0})
    reporter.close()

    assert run_events == [
        (
            "init",
            {"project": "project", "entity": "entity", "reinit": True},
        ),
        ("watch", model),
        ("config", {"learning_rate": 0.1}),
        ("log", {"val_loss": 2.0}, 4),
        ("log", {"best_val_loss": 1.0}, None),
        ("finish",),
    ]


def test_requested_missing_integration_has_actionable_error(monkeypatch, tmp_path):
    def missing_import(module_name):
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(
        reporting_module.importlib,
        "import_module",
        missing_import,
    )
    reporter = TensorBoardReporter(tmp_path)

    with pytest.raises(OptionalIntegrationError, match="TensorBoard"):
        reporter.start({}, object())


def test_core_trainer_has_no_eager_reporting_imports():
    assert "wandb" not in main_module.__dict__
    assert "SummaryWriter" not in main_module.__dict__
