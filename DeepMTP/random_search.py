from __future__ import annotations

from DeepMTP.hpo_progress import (
    HPOProgressEvent,
    HPOProgressObserver,
    build_hpo_progress_observer,
)
from DeepMTP.hpo_types import ConfigurationSpace, HPOWorker
from DeepMTP.utils.utils import BaseExperimentInfo


class RandomSearch:
    """Implements the basic Random search HPO method. Nothing fancy, just a for loop over randomly generated configurations."""

    def __init__(
        self,
        base_worker: HPOWorker,
        configspace: ConfigurationSpace,
        budget: int = 1,
        max_num_epochs: int = 100,
        direction: str = "min",
        verbose: bool = False,
        progress: HPOProgressObserver | None = None,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be at least 1.")
        if max_num_epochs < 1:
            raise ValueError("max_num_epochs must be at least 1.")
        if direction not in {"min", "max"}:
            raise ValueError("direction must be either 'min' or 'max'.")

        self.base_worker = base_worker
        self.configspace = configspace
        self.verbose = verbose
        self.direction = direction
        self.budget = budget
        self.max_num_epochs = max_num_epochs
        self.experiment_history: dict[int, BaseExperimentInfo] = {}
        self.configs_to_evaluate: list[BaseExperimentInfo] = []
        self.progress = (
            progress if progress is not None else build_hpo_progress_observer(verbose)
        )

    def get_run_summary(self) -> dict[int, BaseExperimentInfo]:  # pragma: no cover
        return self.experiment_history

    def run_optimizer(self) -> BaseExperimentInfo:  # pragma: no cover
        self.configs_to_evaluate = [
            BaseExperimentInfo(
                config=self.configspace.sample_configuration(),
                budget=self.max_num_epochs,
            )
            for c in range(self.budget)
        ]
        self.progress.on_event(HPOProgressEvent("optimizer_started", total=self.budget))

        for index, experiment in enumerate(self.configs_to_evaluate, start=1):
            self.progress.on_event(
                HPOProgressEvent(
                    "configuration_started",
                    current=index,
                    total=self.budget,
                    config=experiment.config,
                )
            )

            temp_result_dict = self.base_worker.compute(
                self.max_num_epochs, experiment.config
            )

            experiment.score = temp_result_dict["loss"]
            experiment.info = temp_result_dict["info"]

            self.progress.on_event(
                HPOProgressEvent(
                    "configuration_completed",
                    current=index,
                    total=self.budget,
                    config=experiment.config,
                    score=experiment.score,
                )
            )

        self.configs_to_evaluate = sorted(
            self.configs_to_evaluate,
            key=lambda x: x.score,
            reverse=self.direction != "min",
        )
        self.experiment_history = {
            index: experiment
            for index, experiment in enumerate(self.configs_to_evaluate, start=1)
        }

        best_overall_config = self.configs_to_evaluate[0]
        best_overall_config.info["config"]["experiment_name"] = "best_model"

        self.progress.on_event(HPOProgressEvent("optimizer_completed"))
        return best_overall_config
