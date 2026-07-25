"""Numerical training and evaluation primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

TensorBatch = Mapping[str, torch.Tensor]


@dataclass
class EpochOutput:
    """Losses and optional predictions collected during one data pass."""

    losses: list[float] = field(default_factory=list)
    true_values: list[Any] = field(default_factory=list)
    predicted_values: list[Any] = field(default_factory=list)
    instance_ids: list[Any] = field(default_factory=list)
    target_ids: list[Any] = field(default_factory=list)
    loss_weights: list[int] = field(default_factory=list)

    @property
    def mean_loss(self) -> float:
        if self.loss_weights and len(self.losses) != len(self.loss_weights):
            raise ValueError("Each recorded loss must have one observation weight.")
        if not self.losses:
            return math.nan
        if self.loss_weights:
            total_weight = sum(self.loss_weights)
            if total_weight <= 0:
                raise ValueError("Recorded loss weights must sum to a positive value.")
            return (
                sum(
                    loss * weight
                    for loss, weight in zip(
                        self.losses,
                        self.loss_weights,
                        strict=True,
                    )
                )
                / total_weight
            )
        return sum(self.losses) / len(self.losses)

    def add_loss(self, loss: float, *, observations: int) -> None:
        """Record a batch-mean loss and the observations it represents."""
        if observations <= 0:
            raise ValueError("observations must be a positive integer.")
        self.losses.append(loss)
        self.loss_weights.append(observations)


class TrainingEngine:
    """Runs forward, backward, and collection logic independently of reporting."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        problem_mode: str,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        if problem_mode not in {"classification", "regression"}:
            raise ValueError(
                "problem_mode must be 'classification' or 'regression'; "
                f"received {problem_mode!r}."
            )
        self.model = model
        self.criterion = criterion
        self.device = device
        self.problem_mode = problem_mode
        self.optimizer = optimizer

    def train_epoch(
        self,
        dataloader: Iterable[TensorBatch],
        *,
        collect_predictions: bool = False,
    ) -> EpochOutput:
        """Optimize the model for one complete pass over ``dataloader``."""
        if self.optimizer is None:
            raise RuntimeError("TrainingEngine requires an optimizer for training.")

        self.model.train()
        output = EpochOutput()
        for batch in dataloader:
            true_values, predictions = self._forward_batch(batch)
            loss = self.criterion(predictions, true_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            output.add_loss(loss.item(), observations=true_values.numel())
            if collect_predictions:
                self._collect_predictions(
                    output,
                    batch,
                    true_values,
                    predictions,
                )
        return output

    def evaluate_epoch(
        self,
        dataloader: Iterable[TensorBatch],
        *,
        calculate_loss: bool = True,
        collect_predictions: bool = True,
    ) -> EpochOutput:
        """Evaluate the model for one pass without constructing gradients."""
        was_training = self.model.training
        self.model.eval()
        output = EpochOutput()
        try:
            with torch.no_grad():
                for batch in dataloader:
                    true_values, predictions = self._forward_batch(batch)
                    if calculate_loss:
                        loss = self.criterion(predictions, true_values)
                        output.add_loss(
                            loss.item(),
                            observations=true_values.numel(),
                        )
                    if collect_predictions:
                        self._collect_predictions(
                            output,
                            batch,
                            true_values,
                            predictions,
                        )
        finally:
            self.model.train(was_training)
        return output

    def replace_model(self, model: nn.Module) -> None:
        """Update the model after best-epoch selection."""
        self.model = model

    def _forward_batch(
        self,
        batch: TensorBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        instance_features = batch["instance_features"].to(self.device)
        target_features = batch["target_features"].to(self.device)
        true_values = batch["score"].float().to(self.device)

        predictions = torch.squeeze(
            self.model(instance_features, target_features),
            1,
        )
        if self.problem_mode == "classification":
            predictions = torch.sigmoid(predictions)
        return true_values, predictions

    @staticmethod
    def _collect_predictions(
        output: EpochOutput,
        batch: TensorBatch,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        output.true_values.extend(true_values.detach().cpu().numpy())
        output.predicted_values.extend(predictions.detach().cpu().numpy())
        output.instance_ids.extend(batch["instance_id"].detach().cpu().numpy())
        output.target_ids.extend(batch["target_id"].detach().cpu().numpy())
