"""Validation and collation helpers for token-sequence branch inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

__all__ = [
    "collate_sequence_rows",
    "normalize_sequence_rows",
    "sequence_object_array",
]


def _validate_optional_vocabulary_size(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("vocabulary_size must be a positive integer.")
    return value


def _validate_padding_idx(value: int, *, vocabulary_size: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("padding_idx must be a non-negative integer.")
    if vocabulary_size is not None and value >= vocabulary_size:
        raise ValueError(
            "padding_idx must be smaller than vocabulary_size; "
            f"received padding_idx={value} and vocabulary_size={vocabulary_size}."
        )
    return value


def _normalize_sequence_row(
    value: Any,
    *,
    entity_name: str,
    vocabulary_size: int | None,
    padding_idx: int | None,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(f"{entity_name.title()} token sequences must be dense.")
        if value.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(
                f"{entity_name.title()} token sequences must contain integers."
            )
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{entity_name.title()} token sequences must be array-like."
            ) from exc

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"Each {entity_name} token sequence must be a non-empty "
            "one-dimensional vector."
        )
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype,
        np.integer,
    ):
        raise TypeError(f"{entity_name.title()} token sequences must contain integers.")

    normalized = array.astype(np.int64, copy=True)
    if (normalized < 0).any():
        raise ValueError(
            f"{entity_name.title()} token IDs must be non-negative integers."
        )
    if vocabulary_size is not None and int(normalized.max()) >= vocabulary_size:
        raise ValueError(
            f"{entity_name.title()} token IDs must be smaller than vocabulary_size="
            f"{vocabulary_size}; found {int(normalized.max())}."
        )
    if padding_idx is not None and (normalized == padding_idx).any():
        raise ValueError(
            f"{entity_name.title()} token sequences cannot contain padding_idx="
            f"{padding_idx}; padding is added by the batch collator."
        )
    return normalized


def normalize_sequence_rows(
    values: Sequence[Any],
    *,
    entity_name: str,
    vocabulary_size: int | None = None,
    padding_idx: int | None = None,
) -> tuple[list[np.ndarray], int]:
    """Validate token rows and return normalized ``int64`` arrays.

    The returned integer is the minimum vocabulary size implied by the observed
    token IDs. A configured vocabulary may be larger to reserve unseen or
    domain-specific tokens.
    """

    vocabulary_size = _validate_optional_vocabulary_size(vocabulary_size)
    if padding_idx is not None:
        padding_idx = _validate_padding_idx(
            padding_idx,
            vocabulary_size=vocabulary_size,
        )
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{entity_name.title()} token sequences must be a sequence.")
    if not values:
        raise ValueError(
            f"{entity_name.title()} token sequences must contain at least one row."
        )

    normalized = [
        _normalize_sequence_row(
            value,
            entity_name=entity_name,
            vocabulary_size=vocabulary_size,
            padding_idx=padding_idx,
        )
        for value in values
    ]
    maximum_token = max(int(row.max()) for row in normalized)
    inferred_vocabulary_size = maximum_token + 1
    if padding_idx is not None:
        inferred_vocabulary_size = max(inferred_vocabulary_size, padding_idx + 1)
    return normalized, inferred_vocabulary_size


def sequence_object_array(rows: Sequence[np.ndarray]) -> np.ndarray:
    """Store variable-length rows in a one-dimensional object array."""

    values = np.empty(len(rows), dtype=object)
    values[:] = list(rows)
    return values


def collate_sequence_rows(
    values: Sequence[Any],
    *,
    padding_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamically pad token rows and return IDs, mask, and original lengths."""

    padding_idx = _validate_padding_idx(padding_idx)
    rows, _ = normalize_sequence_rows(
        values,
        entity_name="branch",
        padding_idx=padding_idx,
    )
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.int64)
    maximum_length = int(lengths.max())
    token_ids = torch.full(
        (len(rows), maximum_length),
        padding_idx,
        dtype=torch.int64,
    )
    attention_mask = torch.zeros(
        (len(rows), maximum_length),
        dtype=torch.bool,
    )
    for row_index, row in enumerate(rows):
        row_length = len(row)
        token_ids[row_index, :row_length] = torch.from_numpy(row)
        attention_mask[row_index, :row_length] = True
    return token_ids, attention_mask, lengths
