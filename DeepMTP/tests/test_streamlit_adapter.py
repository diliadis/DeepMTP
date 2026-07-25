from DeepMTP import DeepMTPConfig, ProgressEvent, RunSummaries
from DeepMTP.hpo_progress import HPOProgressEvent
from DeepMTP.main import DeepMTP as CoreDeepMTP
from DeepMTP.main_streamlit import DeepMTP as StreamlitDeepMTP
from DeepMTP.random_search import RandomSearch as CoreRandomSearch
from DeepMTP.random_search_streamlit import RandomSearch as StreamlitRandomSearch
from DeepMTP.simple_hyperband import HyperBand as CoreHyperBand
from DeepMTP.simple_hyperband_streamlit import HyperBand as StreamlitHyperBand
from DeepMTP.streamlit_adapter import (
    StreamlitHPOProgressObserver,
    StreamlitProgressObserver,
)


class FakeSlot:
    def __init__(self):
        self.calls = []

    def subheader(self, value):
        self.calls.append(("subheader", value))

    def info(self, value):
        self.calls.append(("info", value))

    def warning(self, value):
        self.calls.append(("warning", value))

    def write(self, value):
        self.calls.append(("write", value))

    def json(self, value):
        self.calls.append(("json", value))

    def success(self, value):
        self.calls.append(("success", value))

    def dataframe(self, value, **kwargs):
        self.calls.append(("dataframe", value.copy(), kwargs))

    def line_chart(self, value):
        self.calls.append(("line_chart", value.copy()))

    def empty(self):
        self.calls.append(("empty",))


class FakeStreamlit:
    def __init__(self):
        self.slots = []
        self.progress_bars = []

    def empty(self):
        slot = FakeSlot()
        self.slots.append(slot)
        return slot

    def progress(self, value):
        progress = FakeProgress(value)
        self.progress_bars.append(progress)
        return progress


class FakeProgress:
    def __init__(self, initial_value):
        self.values = [initial_value]

    def progress(self, value):
        self.values.append(value)


def make_config(**overrides):
    values = {
        "validation_setting": "B",
        "problem_mode": "classification",
        "compute_mode": "cpu",
        "metrics": ["accuracy"],
        "metrics_average": ["macro"],
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": 2,
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 2,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


def test_streamlit_trainer_is_a_thin_core_subclass():
    assert issubclass(StreamlitDeepMTP, CoreDeepMTP)
    assert issubclass(StreamlitRandomSearch, CoreRandomSearch)
    assert issubclass(StreamlitHyperBand, CoreHyperBand)


def test_streamlit_progress_renders_core_lifecycle_events(tmp_path):
    streamlit = FakeStreamlit()
    observer = StreamlitProgressObserver(
        make_config(),
        experiment_dir=tmp_path,
        streamlit_module=streamlit,
    )

    observer.on_event(ProgressEvent("training_started"))
    observer.on_event(
        ProgressEvent(
            "epoch_completed",
            epoch=0,
            metrics={"train_loss": 0.5},
        )
    )
    observer.on_event(
        ProgressEvent(
            "validation_completed",
            epoch=0,
            metrics={
                "val_loss": 0.4,
                "val_accuracy_macro": 0.8,
            },
            early_stopping_counter=1,
            early_stopping_patience=3,
        )
    )
    observer.on_event(ProgressEvent("testing_started"))
    observer.on_event(
        ProgressEvent(
            "testing_completed",
            metrics={"test_accuracy_macro": 0.75},
            best_epoch=0,
        )
    )
    observer.on_summary(
        RunSummaries("train", "validation", "test"),
        include_train=False,
        include_validation=True,
    )

    train_frame = streamlit.slots[2].calls[-1][1]
    validation_frame = streamlit.slots[4].calls[-1][1]
    test_frame = streamlit.slots[6].calls[-1][1]

    assert train_frame.iloc[0]["loss"] == 0.5
    assert validation_frame.iloc[0]["accuracy_macro"] == 0.8
    assert validation_frame.iloc[0]["early_stopping"] == "1/3"
    assert test_frame.iloc[0]["accuracy_macro"] == 0.75
    assert any(call[0] == "line_chart" for call in streamlit.slots[0].calls)
    assert str(tmp_path) in streamlit.slots[7].calls[-1][1]


def test_streamlit_progress_clears_transient_hpo_output():
    streamlit = FakeStreamlit()
    observer = StreamlitProgressObserver(
        make_config(running_hpo=True),
        streamlit_module=streamlit,
    )
    observer.on_event(ProgressEvent("training_started"))

    observer.on_summary(
        RunSummaries("train", "validation", "test"),
        include_train=False,
        include_validation=False,
    )

    assert all(slot.calls[-1] == ("empty",) for slot in streamlit.slots)


def test_streamlit_hpo_progress_renders_shared_optimizer_events():
    streamlit = FakeStreamlit()
    observer = StreamlitHPOProgressObserver(streamlit_module=streamlit)

    observer.on_event(HPOProgressEvent("optimizer_started", total=2))
    observer.on_event(
        HPOProgressEvent(
            "configuration_started",
            current=1,
            total=2,
            config={"learning_rate": 0.01},
        )
    )
    observer.on_event(
        HPOProgressEvent(
            "configuration_completed",
            current=1,
            total=2,
            score=0.4,
        )
    )
    observer.on_event(HPOProgressEvent("optimizer_completed"))

    assert streamlit.progress_bars[0].values == [0, 50, 100]
    assert streamlit.progress_bars[1].values == [0, 100]
    assert streamlit.slots[1].calls[0] == (
        "json",
        {"learning_rate": 0.01},
    )
    assert streamlit.slots[0].calls[-1][0] == "success"
