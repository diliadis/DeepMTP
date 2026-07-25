from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from torch import nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        use_early_stopping: bool,
        patience: int = 7,
        delta: float = 0.0,
        metric_to_track: str = "loss",
        verbose: bool = False,
    ) -> None:
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 0:
            raise ValueError("patience must be a non-negative integer.")
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("delta must be a finite, non-negative number.")

        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.delta = delta
        self.metric_to_track = metric_to_track
        self.verbose = verbose

        self.early_stop_flag = False
        self.counter = 0

        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.best_model: nn.Module | None = None
        self.best_optimizer_state_dict: dict[str, Any] | None = None
        self.best_performance_results: dict[str, float] | None = None

        self.delta_best_score: float | None = None
        self.delta_best_epoch: int | None = None
        self.delta_best_model: nn.Module | None = None
        self.delta_best_optimizer_state_dict: dict[str, Any] | None = None
        self.delta_best_performance_results: dict[str, float] | None = None

        maximize_metrics = (
            "auroc",
            "aupr",
            "recall",
            "f1_score",
            "precision",
            "accuracy",
            "R2",
        )
        minimize_metrics = ("hamming_loss", "RMSE", "MSE", "MAE", "RRMSE", "loss")
        if any(metric in self.metric_to_track for metric in maximize_metrics):
            self.fac = 1
        elif any(metric in self.metric_to_track for metric in minimize_metrics):
            self.fac = -1
        else:
            raise AttributeError(
                "Invalid metric name used for early stopping: "
                + str(self.metric_to_track)
            )

    def __call__(
        self,
        performance_results: Mapping[str, float],
        model: nn.Module | None,
        epoch: int,
        optimizer_state_dict: Mapping[str, Any] | None = None,
    ) -> None:
        metric_key = "val_" + self.metric_to_track
        if metric_key not in performance_results:
            raise KeyError(f"Early-stopping metric {metric_key!r} is unavailable.")
        score = float(performance_results[metric_key]) * self.fac
        if not math.isfinite(score):
            raise ValueError(
                f"Early-stopping metric {metric_key!r} must be finite; "
                f"received {performance_results[metric_key]!r}."
            )
        performance_snapshot = dict(performance_results)
        optimizer_snapshot = (
            dict(optimizer_state_dict) if optimizer_state_dict is not None else None
        )
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model = model
            self.best_performance_results = performance_snapshot
            self.best_optimizer_state_dict = optimizer_snapshot

        elif (score <= self.best_score + self.delta) and self.use_early_stopping:
            self.counter += 1
            if self.verbose:
                print(
                    f"-----------------------------EarlyStopping counter: {self.counter} out of {self.patience}---------------------- best epoch currently {self.best_epoch}"
                )
            if self.counter >= self.patience:
                self.early_stop_flag = True

            if not (score <= self.best_score):
                self.delta_best_score = score
                self.delta_best_epoch = epoch
                self.delta_best_model = model
                self.delta_best_performance_results = performance_snapshot
                self.delta_best_optimizer_state_dict = optimizer_snapshot
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model = model
            self.best_performance_results = performance_snapshot
            self.best_optimizer_state_dict = optimizer_snapshot
            self.counter = 0

            self.delta_best_score = None
            self.delta_best_epoch = None
            self.delta_best_model = None
            self.delta_best_performance_results = None
            self.delta_best_optimizer_state_dict = None

    def get_best_score(self) -> float | None:
        return (
            self.best_score if self.delta_best_score is None else self.delta_best_score
        )

    def get_best_epoch(self) -> int | None:
        return (
            self.best_epoch if self.delta_best_epoch is None else self.delta_best_epoch
        )

    def get_best_model(self) -> nn.Module | None:
        return (
            self.best_model if self.delta_best_model is None else self.delta_best_model
        )

    def get_best_performance_results(self) -> dict[str, float] | None:
        return (
            self.best_performance_results
            if self.delta_best_performance_results is None
            else self.delta_best_performance_results
        )

    def get_best_optimizer_state_dict(self) -> dict[str, Any] | None:
        return (
            self.best_optimizer_state_dict
            if self.delta_best_optimizer_state_dict is None
            else self.delta_best_optimizer_state_dict
        )
