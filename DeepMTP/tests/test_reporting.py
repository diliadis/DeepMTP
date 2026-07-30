from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

import pandas as pd
import pytest
import torch

import DeepMTP.main as main_module
import DeepMTP.runtime.reporting as reporting_module
from DeepMTP import DeepMTPConfig
from DeepMTP.runtime.reporting import (
    CompositeReporter,
    NullReporter,
    OptionalIntegrationError,
    ReporterModelArtifact,
    ReporterPredictionTable,
    TensorBoardReporter,
    WandBReporter,
    build_reporter,
    mark_reporter_failed,
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
        def update(self, config, *, allow_val_change=False):
            run_events.append(("config", config, allow_val_change))

    class FakeRun:
        def __init__(self):
            self.config = FakeConfig()
            self.summary = {}
            self.id = "run-id"
            self.name = "display-name"
            self.url = "https://wandb.example/run-id"
            self.project = "project"
            self.entity = "entity"

        def define_metric(self, name, **kwargs):
            run_events.append(("define_metric", name, kwargs))

        def pin_config_keys(self, keys):
            run_events.append(("pin_config_keys", keys))

        def watch(self, model, **kwargs):
            run_events.append(("watch", model, kwargs))

        def unwatch(self, model):
            run_events.append(("unwatch", model))

        def log(self, metrics):
            run_events.append(("log", metrics))

        def finish(self, *, exit_code):
            run_events.append(("finish", exit_code))

    def fake_init(**kwargs):
        run_events.append(("init", kwargs))
        return FakeRun()

    monkeypatch.setattr(
        reporting_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(init=fake_init),
    )
    model = torch.nn.Linear(2, 1)
    reporter = WandBReporter(
        "project",
        "entity",
        mode="offline",
        run_name="named-run",
        group="group",
        job_type="train",
        tags=("nightly",),
        notes="test notes",
        watch="all",
        watch_log_freq=25,
        log_graph=True,
    )

    reporter.start(
        {
            "learning_rate": 0.1,
            "metrics": ["accuracy"],
            "metrics_average": ["macro"],
            "validation_setting": "B",
        },
        model,
    )
    reporter.update_config({"learning_rate": 0.2})
    reporter.log_metrics({"val_loss": 2.0}, step=4)
    reporter.log_metrics({"best_val_loss": 1.0})
    reporter.close()

    init_event = run_events[0]
    assert init_event[0] == "init"
    assert init_event[1] == {
        "project": "project",
        "entity": "entity",
        "config": {
            "learning_rate": 0.1,
            "metrics": ["accuracy"],
            "metrics_average": ["macro"],
            "validation_setting": "B",
        },
        "reinit": "create_new",
        "job_type": "train",
        "tags": ["nightly"],
        "save_code": False,
        "mode": "offline",
        "name": "named-run",
        "group": "group",
        "notes": "test notes",
    }
    assert (
        "define_metric",
        "val_accuracy_macro",
        {"step_metric": "epoch", "summary": "max"},
    ) in run_events
    assert ("pin_config_keys", ["validation_setting"]) in run_events
    assert (
        "watch",
        model,
        {"log": "all", "log_freq": 25, "log_graph": True},
    ) in run_events
    assert ("config", {"learning_rate": 0.2}, True) in run_events
    assert ("log", {"val_loss": 2.0, "epoch": 4}) in run_events
    assert ("log", {"best_val_loss": 1.0}) in run_events
    assert ("unwatch", model) in run_events
    assert run_events[-1] == ("finish", 0)


def test_wandb_reporter_marks_failed_runs(monkeypatch):
    class FakeRun:
        config = SimpleNamespace(update=lambda *args, **kwargs: None)

        def __init__(self):
            self.summary = {}

        def define_metric(self, *args, **kwargs):
            return None

        def log(self, metrics):
            return None

        def finish(self, *, exit_code):
            self.exit_code = exit_code

    run = FakeRun()
    monkeypatch.setattr(
        reporting_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(init=lambda **kwargs: run),
    )
    reporter = WandBReporter("project", "entity", watch=None)

    reporter.start({}, object())
    mark_reporter_failed(reporter, RuntimeError("boom"))
    reporter.close()

    assert run.summary["status"] == "failed"
    assert run.summary["failure_type"] == "RuntimeError"
    assert run.summary["failure_message"] == "boom"
    assert run.exit_code == 1


def test_wandb_reporter_logs_artifact_lineage_registry_and_predictions(
    monkeypatch,
    tmp_path,
):
    artifact_events = []

    class FakeArtifact:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
            self.digest = "digest"
            artifact_events.append(("artifact", kwargs))

        def add_file(self, path, *, name):
            artifact_events.append(("add_file", Path(path), name))

    class FakeRun:
        config = SimpleNamespace(update=lambda *args, **kwargs: None)

        def __init__(self):
            self.summary = {}
            self.id = "run-id"

        def define_metric(self, *args, **kwargs):
            return None

        def use_artifact(self, reference):
            artifact_events.append(("use_artifact", reference))

        def log(self, payload):
            artifact_events.append(("log", payload))

        def log_artifact(self, artifact, *, aliases):
            artifact_events.append(("log_artifact", artifact, aliases))
            return artifact

        def link_artifact(self, artifact, *, target_path, aliases):
            artifact_events.append(("link_artifact", artifact, target_path, aliases))

        def finish(self, *, exit_code):
            artifact_events.append(("finish", exit_code))

    class FakeTable:
        def __init__(self, *, dataframe, log_mode):
            self.dataframe = dataframe
            self.log_mode = log_mode

    fake_plot = SimpleNamespace(
        confusion_matrix=lambda **kwargs: ("confusion", kwargs),
        roc_curve=lambda **kwargs: ("roc", kwargs),
        pr_curve=lambda **kwargs: ("pr", kwargs),
    )
    run = FakeRun()
    monkeypatch.setattr(
        reporting_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(
            init=lambda **kwargs: run,
            Artifact=FakeArtifact,
            Table=FakeTable,
            plot=fake_plot,
        ),
    )
    checkpoint_path = tmp_path / "model.pt"
    config_path = tmp_path / "config.json"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path.write_text("{}")
    reporter = WandBReporter(
        "project",
        "entity",
        experiment_dir=tmp_path,
        watch=None,
        input_artifacts=("entity/project/dataset:latest",),
    )
    reporter.start({}, object())

    artifact_metadata = reporter.log_model_artifact(
        ReporterModelArtifact(
            checkpoint_path=checkpoint_path,
            files=(checkpoint_path, config_path),
            metadata={"format_version": 1},
            name="deepmtp-model",
            aliases=("latest", "production-candidate"),
            registry_name="Models",
            registry_collection="DeepMTP",
        )
    )
    reporter.log_predictions(
        ReporterPredictionTable(
            split="test",
            frame=pd.DataFrame(
                {
                    "instance_id": range(5),
                    "target_id": range(5),
                    "true_values": [0, 1, 0, 1, 1],
                    "predicted_values": [0.1, 0.8, 0.2, 0.9, 0.7],
                }
            ),
            problem_mode="classification",
            classification_mode="binary",
            max_rows=3,
            random_seed=7,
        )
    )
    reporter.log_predictions(
        ReporterPredictionTable(
            split="single_class",
            frame=pd.DataFrame(
                {
                    "instance_id": [0, 1],
                    "target_id": [0, 1],
                    "true_values": [0, 0],
                    "predicted_values": [0.1, 0.2],
                }
            ),
            problem_mode="classification",
            classification_mode="binary",
            max_rows=2,
            random_seed=7,
        )
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        reporter.log_model_artifact(
            ReporterModelArtifact(
                checkpoint_path=tmp_path / "missing-model.pt",
                files=(),
                metadata={},
            )
        )
    reporter.close()

    assert ("use_artifact", "entity/project/dataset:latest") in artifact_events
    assert artifact_metadata == {
        "wandb_model_artifact": {
            "name": "deepmtp-model",
            "digest": "digest",
            "aliases": ["latest", "production-candidate"],
        }
    }
    assert (
        "link_artifact",
        ANY,
        "wandb-registry-Models/DeepMTP",
        ["latest", "production-candidate"],
    ) in artifact_events
    prediction_payload = next(
        event[1]
        for event in artifact_events
        if event[0] == "log" and "test_predictions" in event[1]
    )
    assert len(prediction_payload["test_predictions"].dataframe) == 3
    assert prediction_payload["test_predictions"].log_mode == "IMMUTABLE"
    assert set(prediction_payload) == {
        "test_predictions",
        "test_confusion_matrix",
        "test_roc_curve",
        "test_pr_curve",
    }
    single_class_payload = next(
        event[1]
        for event in artifact_events
        if event[0] == "log" and "single_class_predictions" in event[1]
    )
    assert set(single_class_payload) == {
        "single_class_predictions",
        "single_class_confusion_matrix",
    }


def test_wandb_reporter_real_sdk_offline_smoke(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    for directory_name in ("cache", "config", "data", "wandb"):
        (tmp_path / directory_name).mkdir()
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")
    monkeypatch.setenv("WANDB_SILENT", "true")
    reporter = WandBReporter(
        "deepmtp-offline-smoke",
        "local",
        experiment_dir=tmp_path,
        mode="offline",
        watch=None,
    )

    reporter.start({"metrics": [], "metrics_average": []}, torch.nn.Linear(2, 1))
    reporter.log_metrics({"train_loss": 1.0}, step=0)
    reporter.log_summary({"best_epoch": 0})
    reporter.close()

    assert list((tmp_path / "wandb").glob("offline-run-*"))


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
