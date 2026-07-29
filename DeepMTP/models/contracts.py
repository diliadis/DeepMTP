"""Shared contracts for branch encoders and their standardized outputs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn

from DeepMTP.data.batches import BranchInputKind

__all__ = [
    "BranchEncoder",
    "encode_branch",
    "encoder_input_kind",
    "encoder_output_dimension",
]


@runtime_checkable
class BranchEncoder(Protocol):
    """Structural contract implemented by built-in branch encoders."""

    input_kind: BranchInputKind
    output_dim: int

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Encode one input batch as ``[batch, output_dim]``."""
        ...


def encoder_output_dimension(
    encoder: nn.Module,
    *,
    branch: str,
    explicit_dimension: int | None = None,
) -> int:
    """Return a validated declared encoder output dimension."""
    dimension = (
        explicit_dimension
        if explicit_dimension is not None
        else getattr(encoder, "output_dim", None)
    )
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError(
            f"{branch} branch encoder must declare a positive integer output_dim."
        )
    return dimension


def encoder_input_kind(encoder: nn.Module, *, branch: str) -> BranchInputKind:
    """Return a validated declared encoder input modality."""
    kind = getattr(encoder, "input_kind", None)
    if kind not in {"dense", "id", "image", "custom"}:
        raise ValueError(
            f"{branch} branch encoder must declare input_kind as "
            "'dense', 'id', 'image', or 'custom'."
        )
    return kind


def encode_branch(
    encoder: nn.Module,
    values: torch.Tensor,
    *,
    branch: str,
    explicit_dimension: int | None = None,
) -> torch.Tensor:
    """Run one encoder and validate the common ``[batch, width]`` contract."""
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{branch} branch input must be a torch.Tensor.")
    if values.ndim == 0:
        raise ValueError(f"{branch} branch input must include a batch dimension.")

    encoded = encoder(values)
    if not isinstance(encoded, torch.Tensor):
        raise TypeError(f"{branch} branch encoder must return a torch.Tensor.")
    if encoded.ndim != 2:
        raise ValueError(
            f"{branch} branch encoder must return shape [batch, output_dim]; "
            f"received {tuple(encoded.shape)}."
        )
    if encoded.shape[0] != values.shape[0]:
        raise ValueError(
            f"{branch} branch encoder changed the batch size from "
            f"{values.shape[0]} to {encoded.shape[0]}."
        )
    expected_dimension = encoder_output_dimension(
        encoder,
        branch=branch,
        explicit_dimension=explicit_dimension,
    )
    if encoded.shape[1] != expected_dimension:
        raise ValueError(
            f"{branch} branch encoder declared output_dim={expected_dimension} "
            f"but returned width {encoded.shape[1]}."
        )
    if not torch.is_floating_point(encoded):
        raise TypeError(
            f"{branch} branch encoder output must use a floating-point dtype."
        )
    return encoded
