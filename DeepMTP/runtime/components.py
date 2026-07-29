"""Registries for task-level losses and output heads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from torch import nn

from DeepMTP._registry import FactoryRegistry
from DeepMTP.config import (
    DEFAULT_LOSS_BY_PROBLEM_MODE,
    SUPPORTED_LOSSES_BY_PROBLEM_MODE,
)

__all__ = [
    "DEFAULT_LOSS_BY_PROBLEM_MODE",
    "DEFAULT_LOSS_INPUT_TRANSFORM_BY_PROBLEM_MODE",
    "DEFAULT_OUTPUT_HEAD_BY_PROBLEM_MODE",
    "LOSS_INPUT_TRANSFORM_BY_LOSS",
    "LOSS_REGISTRY",
    "OUTPUT_HEAD_REGISTRY",
    "SUPPORTED_LOSSES_BY_PROBLEM_MODE",
    "LossFactory",
    "OutputHeadFactory",
    "TaskComponents",
    "build_default_loss",
    "build_default_loss_input_transform",
    "build_default_output_head",
    "build_default_task_components",
    "build_loss",
    "build_loss_input_transform",
    "build_output_head",
    "build_task_components",
]

LossFactory: TypeAlias = Callable[[], nn.Module]
OutputHeadFactory: TypeAlias = Callable[[], nn.Module]

LOSS_REGISTRY = FactoryRegistry[LossFactory](
    "loss",
    {
        "binary_cross_entropy": lambda: nn.BCELoss(reduction="mean"),
        "binary_cross_entropy_with_logits": lambda: nn.BCEWithLogitsLoss(
            reduction="mean"
        ),
        "huber": lambda: nn.HuberLoss(reduction="mean"),
        "mean_absolute_error": lambda: nn.L1Loss(reduction="mean"),
        "mean_squared_error": lambda: nn.MSELoss(reduction="mean"),
    },
)

OUTPUT_HEAD_REGISTRY = FactoryRegistry[OutputHeadFactory](
    "output head",
    {
        "identity": nn.Identity,
        "sigmoid": nn.Sigmoid,
    },
)

LOSS_INPUT_TRANSFORM_BY_LOSS: Mapping[str, str] = MappingProxyType(
    {
        "binary_cross_entropy": "sigmoid",
        "binary_cross_entropy_with_logits": "identity",
        "huber": "identity",
        "mean_absolute_error": "identity",
        "mean_squared_error": "identity",
    }
)

DEFAULT_LOSS_INPUT_TRANSFORM_BY_PROBLEM_MODE: Mapping[str, str] = MappingProxyType(
    {
        "classification": "identity",
        "regression": "identity",
    }
)

DEFAULT_OUTPUT_HEAD_BY_PROBLEM_MODE: Mapping[str, str] = MappingProxyType(
    {
        "classification": "sigmoid",
        "regression": "identity",
    }
)


@dataclass(frozen=True)
class TaskComponents:
    """Loss-input and user-output components selected for one problem mode."""

    loss: nn.Module
    loss_input_transform: nn.Module
    output_head: nn.Module


def build_loss(name: str) -> nn.Module:
    """Build a registered loss module by name."""
    return _build_module(LOSS_REGISTRY, name)


def build_output_head(name: str) -> nn.Module:
    """Build a registered output-head module by name."""
    return _build_module(OUTPUT_HEAD_REGISTRY, name)


def build_default_task_components(problem_mode: str) -> TaskComponents:
    """Build the behavior-preserving task components for a problem mode."""
    return build_task_components(problem_mode)


def build_task_components(
    problem_mode: str,
    loss_name: str | None = None,
) -> TaskComponents:
    """Build compatible loss and output modules for one problem mode."""
    selected_loss = (
        _default_component_name(DEFAULT_LOSS_BY_PROBLEM_MODE, problem_mode)
        if loss_name is None
        else loss_name
    )
    supported_losses = _supported_loss_names(problem_mode)
    if selected_loss not in supported_losses:
        raise ValueError(
            f"loss={selected_loss!r} is incompatible with "
            f"problem_mode={problem_mode!r}. Select one of "
            f"{sorted(supported_losses)}."
        )
    return TaskComponents(
        loss=build_loss(selected_loss),
        loss_input_transform=build_loss_input_transform(selected_loss),
        output_head=build_default_output_head(problem_mode),
    )


def build_default_loss(problem_mode: str) -> nn.Module:
    """Build the default registered loss for a problem mode."""
    name = _default_component_name(DEFAULT_LOSS_BY_PROBLEM_MODE, problem_mode)
    return build_loss(name)


def build_default_loss_input_transform(problem_mode: str) -> nn.Module:
    """Build the transform applied to raw model scores before the loss."""
    name = _default_component_name(
        DEFAULT_LOSS_INPUT_TRANSFORM_BY_PROBLEM_MODE,
        problem_mode,
    )
    return build_output_head(name)


def build_loss_input_transform(loss_name: str) -> nn.Module:
    """Build the transform applied before a selected registered loss."""
    try:
        transform_name = LOSS_INPUT_TRANSFORM_BY_LOSS[loss_name]
    except KeyError:
        available = ", ".join(
            repr(name) for name in sorted(LOSS_INPUT_TRANSFORM_BY_LOSS)
        )
        raise KeyError(f"Unknown loss {loss_name!r}. Available: {available}.") from None
    return build_output_head(transform_name)


def build_default_output_head(problem_mode: str) -> nn.Module:
    """Build the default registered output head for a problem mode."""
    name = _default_component_name(
        DEFAULT_OUTPUT_HEAD_BY_PROBLEM_MODE,
        problem_mode,
    )
    return build_output_head(name)


def _default_component_name(
    names_by_problem_mode: Mapping[str, str],
    problem_mode: str,
) -> str:
    try:
        return names_by_problem_mode[problem_mode]
    except KeyError:
        raise ValueError(
            "problem_mode must be 'classification' or 'regression'; "
            f"received {problem_mode!r}."
        ) from None


def _supported_loss_names(problem_mode: str) -> frozenset[str]:
    try:
        return SUPPORTED_LOSSES_BY_PROBLEM_MODE[problem_mode]
    except KeyError:
        raise ValueError(
            "problem_mode must be 'classification' or 'regression'; "
            f"received {problem_mode!r}."
        ) from None


def _build_module(
    registry: FactoryRegistry[Callable[[], nn.Module]],
    name: str,
) -> nn.Module:
    factory = registry.resolve(name)
    component = factory()
    if not isinstance(component, nn.Module):
        raise TypeError(
            f"Registered {registry.component_name} factory {name!r} must return "
            "torch.nn.Module."
        )
    return component
