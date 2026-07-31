"""Numerical training and evaluation primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch
from torch import nn

from DeepMTP.data.batches import MTPBatch
from DeepMTP.runtime.components import build_default_output_head

__all__ = ["EpochOutput", "TensorBatch", "TrainingEngine"]

TensorBatch: TypeAlias = MTPBatch | Mapping[str, Any]


@dataclass
class EpochOutput:
    """Losses and optional predictions collected during one data pass."""

    losses: list[float] = field(default_factory=list)
    true_values: list[Any] = field(default_factory=list)
    predicted_values: list[Any] = field(default_factory=list)
    instance_ids: list[Any] = field(default_factory=list)
    target_ids: list[Any] = field(default_factory=list)
    loss_weights: list[int] = field(default_factory=list)
    class_probabilities: list[list[float]] = field(default_factory=list)

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
    """Run numerical training and evaluation independently of reporting.

    ``output_head`` transforms raw model scores for metrics and returned
    predictions. ``loss_input_transform`` independently controls the values
    consumed by ``criterion``. When it is omitted, ``BCEWithLogitsLoss`` sees
    raw scores and other criteria retain the historical behavior of consuming
    the output-head values.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        problem_mode: str,
        optimizer: torch.optim.Optimizer | None = None,
        output_head: nn.Module | None = None,
        loss_input_transform: nn.Module | None = None,
        classification_mode: str | None = None,
        num_classes: int | None = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float | None = None,
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
        self.classification_mode: str | None
        self.num_classes: int | None
        if problem_mode == "classification":
            self.classification_mode = (
                "binary" if classification_mode is None else classification_mode
            )
            if self.classification_mode not in {"binary", "multiclass"}:
                raise ValueError(
                    "classification_mode must be 'binary' or 'multiclass'; "
                    f"received {self.classification_mode!r}."
                )
        else:
            if classification_mode is not None:
                raise ValueError("classification_mode must be None for regression.")
            self.classification_mode = None
        if self.classification_mode == "multiclass":
            if (
                isinstance(num_classes, bool)
                or not isinstance(num_classes, int)
                or num_classes < 3
            ):
                raise ValueError(
                    "num_classes must be an integer greater than or equal to 3 "
                    "for multiclass classification."
                )
            self.num_classes = num_classes
        else:
            if num_classes is not None:
                raise ValueError(
                    "num_classes is only valid for multiclass classification."
                )
            self.num_classes = None
        if (
            isinstance(gradient_accumulation_steps, bool)
            or not isinstance(gradient_accumulation_steps, int)
            or gradient_accumulation_steps <= 0
        ):
            raise ValueError(
                "gradient_accumulation_steps must be a positive integer; "
                f"received {gradient_accumulation_steps!r}."
            )
        if gradient_clip_norm is not None and (
            isinstance(gradient_clip_norm, bool)
            or not isinstance(gradient_clip_norm, (int, float))
            or not math.isfinite(gradient_clip_norm)
            or gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be a positive finite number or None; "
                f"received {gradient_clip_norm!r}."
            )
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.output_head = (
            output_head
            if output_head is not None
            else build_default_output_head(problem_mode, self.classification_mode)
        )
        if not isinstance(self.output_head, nn.Module):
            raise TypeError("output_head must be a torch.nn.Module.")
        self.loss_input_transform = (
            loss_input_transform
            if loss_input_transform is not None
            else (
                nn.Identity()
                if isinstance(
                    criterion,
                    (nn.BCEWithLogitsLoss, nn.CrossEntropyLoss),
                )
                else self.output_head
            )
        )
        if not isinstance(self.loss_input_transform, nn.Module):
            raise TypeError("loss_input_transform must be a torch.nn.Module.")
        self.output_head.to(device)
        self.loss_input_transform.to(device)

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
        accumulated_batches = 0
        accumulated_observations = 0
        for batch in dataloader:
            if accumulated_batches == 0:
                self.optimizer.zero_grad()
            true_values, raw_predictions = self._forward_batch(batch)
            loss_predictions = self.loss_input_transform(raw_predictions)
            loss = self.criterion(loss_predictions, true_values)
            observations = int(true_values.shape[0])

            backward_loss = (
                loss if self.gradient_accumulation_steps == 1 else loss * observations
            )
            backward_loss.backward()
            accumulated_batches += 1
            accumulated_observations += observations
            if accumulated_batches == self.gradient_accumulation_steps:
                self._complete_optimizer_step(
                    accumulated_observations=accumulated_observations,
                )
                accumulated_batches = 0
                accumulated_observations = 0

            output.add_loss(loss.item(), observations=observations)
            if collect_predictions:
                predictions = self.output_head(raw_predictions)
                self._collect_predictions(
                    output,
                    batch,
                    true_values,
                    predictions,
                )
        if accumulated_batches:
            self._complete_optimizer_step(
                accumulated_observations=accumulated_observations,
            )
        return output

    def _complete_optimizer_step(self, *, accumulated_observations: int) -> None:
        if self.optimizer is None:  # pragma: no cover - train_epoch invariant
            raise RuntimeError("TrainingEngine requires an optimizer for training.")
        if self.gradient_accumulation_steps > 1:
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(1.0 / accumulated_observations)
        if self.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        self.optimizer.step()

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
                    true_values, raw_predictions = self._forward_batch(batch)
                    if calculate_loss:
                        loss_predictions = self.loss_input_transform(raw_predictions)
                        loss = self.criterion(loss_predictions, true_values)
                        output.add_loss(
                            loss.item(),
                            observations=int(true_values.shape[0]),
                        )
                    if collect_predictions:
                        predictions = self.output_head(raw_predictions)
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
        if isinstance(batch, MTPBatch):
            self._validate_typed_input_kinds(batch)
            device_batch = batch.to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )
            instance_features = device_batch.instance_input.values
            target_features = device_batch.target_input.values
            score = device_batch.score
        else:
            instance_features = batch["instance_features"].to(self.device)
            target_features = batch["target_features"].to(self.device)
            score = batch["score"].to(self.device)

        if self.classification_mode == "multiclass":
            score_values = score.float()
            if (
                score_values.ndim != 1
                or not torch.isfinite(score_values).all()
                or not torch.equal(score_values, torch.floor(score_values))
                or self.num_classes is None
                or (score_values < 0).any()
                or (score_values >= self.num_classes).any()
            ):
                raise ValueError(
                    "Multiclass targets must have shape [batch] and contain "
                    f"integer class IDs in [0, {self.num_classes})."
                )
            true_values = score.long()
        else:
            true_values = score.float()
        model_output = self.model(instance_features, target_features)
        if self.classification_mode == "multiclass":
            if (
                model_output.ndim != 2
                or self.num_classes is None
                or model_output.shape[1] != self.num_classes
            ):
                raise ValueError(
                    "Multiclass models must return shape [batch, num_classes]; "
                    f"received {tuple(model_output.shape)}."
                )
            raw_predictions = model_output
        else:
            raw_predictions = torch.squeeze(model_output, 1)
        return true_values, raw_predictions

    def _validate_typed_input_kinds(self, batch: MTPBatch) -> None:
        for branch, branch_batch in {
            "instance": batch.instance_input,
            "target": batch.target_input,
        }.items():
            expected_kind = getattr(self.model, f"{branch}_input_kind", None)
            if expected_kind is not None and branch_batch.kind != expected_kind:
                raise ValueError(
                    f"{branch} branch model expects {expected_kind!r} input, "
                    f"but the dataloader produced {branch_batch.kind!r} input."
                )

    def _collect_predictions(
        self,
        output: EpochOutput,
        batch: TensorBatch,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        output.true_values.extend(true_values.detach().cpu().numpy())
        if self.classification_mode == "multiclass":
            probability_rows = predictions.detach().cpu().numpy()
            output.class_probabilities.extend(probability_rows.tolist())
            output.predicted_values.extend(probability_rows.argmax(axis=1).tolist())
        else:
            output.predicted_values.extend(predictions.detach().cpu().numpy())
        output.instance_ids.extend(batch["instance_id"].detach().cpu().numpy())
        output.target_ids.extend(batch["target_id"].detach().cpu().numpy())
