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
