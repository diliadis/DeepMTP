from DeepMTP import (
    ConsoleProgressObserver,
    NullProgressObserver,
    ProgressEvent,
    RunHistory,
    RunSummaries,
)


def test_run_history_formats_all_phases_and_missing_metrics():
    history = RunHistory(["accuracy"], ["macro"])
    history.add_training(0, 0.123456, {})
    history.add_validation(
        0,
        0.234567,
        {"val_accuracy_macro": 0.87654},
        early_stopping_counter=1,
        early_stopping_patience=3,
    )
    history.add_test(0, {"test_accuracy_macro": 0.76543})

    summaries = history.render()

    assert "0.1235" in summaries.train
    assert "-" in summaries.train
    assert "0.8765" in summaries.validation
    assert "1/3" in summaries.validation
    assert "0.7654" in summaries.test


def test_console_progress_observer_renders_events_and_summaries(capsys):
    observer = ConsoleProgressObserver()
    observer.on_event(ProgressEvent("training_started"))
    observer.on_event(ProgressEvent("epoch_started", epoch=2))
    observer.on_event(ProgressEvent("epoch_completed", epoch=2))
    observer.on_event(ProgressEvent("early_stopping_triggered", epoch=2))
    observer.on_event(ProgressEvent("training_completed"))
    observer.on_event(ProgressEvent("training_failed", message="bad data"))
    observer.on_summary(
        RunSummaries("train table", "validation table", "test table"),
        include_train=True,
        include_validation=False,
    )

    output = capsys.readouterr().out
    assert "Starting training..." in output
    assert "Epoch:2... Done" in output
    assert "Early stopping criterion met" in output
    assert "Training completed." in output
    assert "Training failed: bad data" in output
    assert "train table" in output
    assert "validation table" not in output
    assert "test table" in output


def test_null_progress_observer_has_no_output(capsys):
    observer = NullProgressObserver()

    observer.on_event(ProgressEvent("training_started"))
    observer.on_summary(
        RunSummaries("train", "validation", "test"),
        include_train=True,
        include_validation=True,
    )

    assert capsys.readouterr().out == ""
