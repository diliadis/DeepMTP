"""Evaluation policy and metric calculation for DeepMTP experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from DeepMTP.config import DeepMTPConfig
from DeepMTP.training import EpochOutput
from DeepMTP.utils.eval_utils import get_performance_results


@dataclass(frozen=True)
class EvaluationPolicy:
    """Decides when prediction-dependent metrics should be calculated."""

    evaluate_train: bool
    evaluate_val: bool
    eval_every_n_epochs: int
    num_epochs: int
    metric_to_optimize_early_stopping: str

    @classmethod
    def from_config(cls, config: DeepMTPConfig) -> EvaluationPolicy:
        return cls(
            evaluate_train=config.evaluate_train,
            evaluate_val=config.evaluate_val,
            eval_every_n_epochs=config.eval_every_n_epochs,
            num_epochs=config.num_epochs,
            metric_to_optimize_early_stopping=(
                config.metric_to_optimize_early_stopping
            ),
        )

    def should_evaluate(self, mode: str, epoch: int) -> bool:
        """Return whether inference metrics are required for this epoch."""
        return (
            mode == "test"
            or (
                epoch % self.eval_every_n_epochs == 0
                and self.evaluate_val
            )
            or self.metric_to_optimize_early_stopping != "loss"
            or epoch == self.num_epochs - 1
        )

    def should_evaluate_training(self, epoch: int) -> bool:
        """Return whether training metrics are required for this epoch."""
        return self.evaluate_train and (
            epoch % self.eval_every_n_epochs == 0
            or epoch == self.num_epochs - 1
        )


class Evaluator:
    """Calculates configured metrics from a numerical epoch output."""

    def __init__(
        self,
        config: DeepMTPConfig | Mapping[str, Any],
    ):
        self.config = DeepMTPConfig.from_mapping(config)

    @property
    def policy(self) -> EvaluationPolicy:
        """Return policy values reflecting the current typed configuration."""
        return EvaluationPolicy.from_config(self.config)

    def evaluate(
        self,
        mode: str,
        epoch: int,
        output: EpochOutput,
    ) -> dict[str, float]:
        """Calculate configured performance metrics."""
        if not self.config.metrics:
            return {}
        return get_performance_results(
            mode,
            epoch,
            output.instance_ids,
            output.target_ids,
            output.true_values,
            output.predicted_values,
            self.config.validation_setting,
            self.config.problem_mode,
            self.config.metrics,
            self.config.metrics_average,
            verbose=self.config.results_verbose,
            per_target_verbose=self.config.eval_target_verbose,
            per_instance_verbose=self.config.eval_instance_verbose,
            train_true_value=None,
            scaler_per_target=None,
            top_k=self.config.top_k,
        )

    @staticmethod
    def predictions_frame(output: EpochOutput) -> pd.DataFrame:
        """Build the stable tabular prediction result."""
        return pd.DataFrame(
            {
                "instance_id": output.instance_ids,
                "target_id": output.target_ids,
                "true_values": output.true_values,
                "predicted_values": output.predicted_values,
            }
        )
