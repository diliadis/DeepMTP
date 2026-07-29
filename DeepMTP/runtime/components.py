"""Registries for task-level losses and output heads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from torch import nn

from DeepMTP._registry import FactoryRegistry

__all__ = [
    "DEFAULT_LOSS_BY_PROBLEM_MODE",
    "DEFAULT_LOSS_INPUT_TRANSFORM_BY_PROBLEM_MODE",
    "DEFAULT_OUTPUT_HEAD_BY_PROBLEM_MODE",
    "LOSS_REGISTRY",
    "OUTPUT_HEAD_REGISTRY",
    "LossFactory",
    "OutputHeadFactory",
    "TaskComponents",
    "build_default_loss",
    "build_default_loss_input_transform",
    "build_default_output_head",
    "build_default_task_components",
    "build_loss",
    "build_output_head",
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

DEFAULT_LOSS_BY_PROBLEM_MODE: Mapping[str, str] = MappingProxyType(
    {
        "classification": "binary_cross_entropy_with_logits",
        "regression": "mean_squared_error",
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
    return TaskComponents(
        loss=build_default_loss(problem_mode),
        loss_input_transform=build_default_loss_input_transform(problem_mode),
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
