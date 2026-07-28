import ast
import inspect

import pytest

from DeepMTP import (
    ConsoleHPOProgressObserver,
    HPOProgressEvent,
    NullHPOProgressObserver,
)
from DeepMTP.hpo_worker import BaseWorker
from DeepMTP.random_search import RandomSearch
from DeepMTP.simple_hyperband import HyperBand


class RecordingProgress:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


class FakeConfigSpace:
    def __init__(self):
        self.counter = 0

    def sample_configuration(self):
        self.counter += 1
        return {"score": float(self.counter)}


class FakeWorker:
    def compute(self, budget, config):
        return {
            "loss": config["score"],
            "info": {
                "config": {
                    "experiment_name": "candidate",
                    "budget": budget,
                }
            },
        }


def test_console_hpo_progress_renders_schedule_and_worker_configuration(capsys):
    observer = ConsoleHPOProgressObserver()
    schedule = {
        3: {
            "n_i": [3, 1],
            "r_i": [1, 3],
            "num_iters": 2,
        }
    }

    observer.on_event(
        HPOProgressEvent(
            "schedule_calculated",
            config=schedule,
            smax=1,
            eta=3,
            max_budget=3,
            total_budget=6,
        )
    )
    observer.on_event(
        HPOProgressEvent(
            "worker_configured",
            config={"learning_rate": 0.01},
        )
    )

    output = capsys.readouterr().out
    assert "smax: 1" in output
    assert "B: 6" in output
    assert "n: 3   r: 1" in output
    assert "ni: 1   ri (epochs): 3" in output
    assert "final config: {'learning_rate': 0.01}" in output


def test_null_hpo_progress_is_silent_for_diagnostic_events(capsys):
    observer = NullHPOProgressObserver()

    observer.on_event(
        HPOProgressEvent(
            "schedule_calculated",
            config={},
            smax=0,
            total_budget=1,
        )
    )
    observer.on_event(HPOProgressEvent("worker_configured", config={}))

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("target", [HyperBand, BaseWorker])
def test_hpo_core_delegates_output_to_progress_observers(target):
    tree = ast.parse(inspect.getsource(target))
    direct_prints = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

    assert direct_prints == []


def test_random_search_emits_shared_progress_events():
    progress = RecordingProgress()
    optimizer = RandomSearch(
        FakeWorker(),
        FakeConfigSpace(),
        budget=2,
        max_num_epochs=3,
        progress=progress,
    )

    best = optimizer.run_optimizer()

    assert best.score == 1.0
    assert [event.kind for event in progress.events] == [
        "optimizer_started",
        "configuration_started",
        "configuration_completed",
        "configuration_started",
        "configuration_completed",
        "optimizer_completed",
    ]
    assert optimizer.get_run_summary() == {1: best, 2: optimizer.configs_to_evaluate[1]}


def test_hyperband_emits_shared_progress_events():
    progress = RecordingProgress()
    optimizer = HyperBand(
        FakeWorker(),
        FakeConfigSpace(),
        eta=3,
        max_budget=1,
        progress=progress,
    )

    best = optimizer.run_optimizer()

    assert best.score == 1.0
    assert [event.kind for event in progress.events] == [
        "optimizer_started",
        "bracket_started",
        "iteration_started",
        "configuration_started",
        "configuration_completed",
        "optimizer_completed",
    ]


def test_hyperband_emits_schedule_diagnostics_to_injected_observer(capsys):
    progress = RecordingProgress()
    optimizer = HyperBand(
        FakeWorker(),
        FakeConfigSpace(),
        eta=3,
        max_budget=1,
        verbose=True,
        progress=progress,
    )

    schedule = optimizer.calculate_hyperband_iters(R=3, eta=3, verbose=True)

    assert [event.kind for event in progress.events] == ["schedule_calculated"]
    assert progress.events[0].config is schedule
    assert progress.events[0].smax == 1
    assert progress.events[0].total_budget == 6
    assert capsys.readouterr().out == ""


def test_hyperband_calculation_verbose_flag_selects_console_observer(capsys):
    optimizer = HyperBand(
        FakeWorker(),
        FakeConfigSpace(),
        eta=3,
        max_budget=1,
        verbose=False,
    )

    optimizer.calculate_hyperband_iters(R=3, eta=3, verbose=True)

    output = capsys.readouterr().out
    assert "smax: 1" in output
    assert "ni: 1   ri (epochs): 3" in output


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"budget": 0}, "budget"),
        ({"max_num_epochs": 0}, "max_num_epochs"),
        ({"direction": "sideways"}, "direction"),
    ],
)
def test_random_search_rejects_invalid_parameters(overrides, message):
    with pytest.raises(ValueError, match=message):
        RandomSearch(FakeWorker(), FakeConfigSpace(), **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eta": 1}, "eta"),
        ({"max_budget": 0}, "max_budget"),
        ({"direction": "sideways"}, "direction"),
    ],
)
def test_hyperband_rejects_invalid_parameters(overrides, message):
    with pytest.raises(ValueError, match=message):
        HyperBand(FakeWorker(), FakeConfigSpace(), **overrides)
