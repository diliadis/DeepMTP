"""Typed model-input batches shared by data loading and training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, TypeAlias, cast

import torch
from torch.utils.data import default_collate

from DeepMTP.data.sequences import collate_sequence_rows
from DeepMTP.data.sparse import (
    collate_sparse_rows,
    is_torch_sparse,
    pin_sparse_tensor,
)

__all__ = [
    "BranchBatch",
    "BranchInputKind",
    "CompositeBranchBatch",
    "CompositeInput",
    "CustomBranchBatch",
    "DenseBranchBatch",
    "IDBranchBatch",
    "ImageBranchBatch",
    "MTPBatch",
    "MTPBatchCollator",
    "SequenceBranchBatch",
    "SequenceInput",
    "SparseBranchBatch",
    "TabularBranchBatch",
    "TabularInput",
    "TensorBranchBatch",
]

BranchInputKind: TypeAlias = Literal[
    "dense",
    "id",
    "image",
    "sequence",
    "sparse",
    "tabular",
    "custom",
    "composite",
]


@dataclass(frozen=True)
class TensorBranchBatch:
    """One branch's tensor payload with an explicit modality contract."""

    values: torch.Tensor
    kind: ClassVar[BranchInputKind] = "custom"

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("Branch batch values must be a torch.Tensor.")
        if self.values.ndim == 0:
            raise ValueError("Branch batch values must include a batch dimension.")
        self._validate_values()

    def _validate_values(self) -> None:
        """Validate modality-specific tensor shape and dtype."""

    @property
    def batch_size(self) -> int:
        """Number of observations represented by this branch batch."""
        return int(self.values.shape[0])

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> TensorBranchBatch:
        """Return the same typed batch with its tensor moved to ``device``."""
        return replace(
            self,
            values=self.values.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> TensorBranchBatch:
        """Return the same typed batch backed by pinned host memory."""
        return replace(self, values=self.values.pin_memory())


class DenseBranchBatch(TensorBranchBatch):
    """A batch of dense feature vectors shaped ``[batch, features]``."""

    kind: ClassVar[BranchInputKind] = "dense"

    def _validate_values(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Dense branch values must have shape [batch, features].")
        if not torch.is_floating_point(self.values):
            raise TypeError("Dense branch values must use a floating-point dtype.")


class IDBranchBatch(TensorBranchBatch):
    """A batch of zero-based integer entity IDs shaped ``[batch]``."""

    kind: ClassVar[BranchInputKind] = "id"

    def _validate_values(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("ID branch values must have shape [batch].")
        if self.values.dtype not in {torch.int32, torch.int64}:
            raise TypeError("ID branch values must use an integer tensor dtype.")


class ImageBranchBatch(TensorBranchBatch):
    """A batch of images shaped ``[batch, channels, height, width]``."""

    kind: ClassVar[BranchInputKind] = "image"

    def _validate_values(self) -> None:
        if self.values.ndim != 4:
            raise ValueError(
                "Image branch values must have shape [batch, channels, height, width]."
            )
        if not torch.is_floating_point(self.values):
            raise TypeError("Image branch values must use a floating-point dtype.")


class SparseBranchBatch(TensorBranchBatch):
    """A sparse feature batch shaped ``[batch, features]``."""

    kind: ClassVar[BranchInputKind] = "sparse"

    def _validate_values(self) -> None:
        if not is_torch_sparse(self.values):
            raise TypeError(
                "Sparse branch values must use PyTorch COO or CSR sparse layout."
            )
        if self.values.ndim != 2:
            raise ValueError("Sparse branch values must have shape [batch, features].")
        if not torch.is_floating_point(self.values):
            raise TypeError("Sparse branch values must use a floating-point dtype.")

    def pin_memory(self) -> SparseBranchBatch:
        """Pin sparse indices and values for asynchronous device transfer."""
        return replace(self, values=pin_sparse_tensor(self.values))


class CustomBranchBatch(TensorBranchBatch):
    """A tensor batch passed unchanged to a user-supplied branch."""

    kind: ClassVar[BranchInputKind] = "custom"


@dataclass(frozen=True)
class SequenceInput:
    """Padded token IDs, attention mask, and original sequence lengths."""

    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    lengths: torch.Tensor
    padding_idx: int = 0

    def __post_init__(self) -> None:
        for name, values in {
            "token_ids": self.token_ids,
            "attention_mask": self.attention_mask,
            "lengths": self.lengths,
        }.items():
            if not isinstance(values, torch.Tensor):
                raise TypeError(f"Sequence {name} must be a torch.Tensor.")
        if self.token_ids.ndim != 2:
            raise ValueError(
                "Sequence token_ids must have shape [batch, sequence_length]."
            )
        if self.attention_mask.shape != self.token_ids.shape:
            raise ValueError(
                "Sequence attention_mask must have the same shape as token_ids."
            )
        if self.lengths.ndim != 1 or self.lengths.shape[0] != self.token_ids.shape[0]:
            raise ValueError("Sequence lengths must have shape [batch].")
        if self.token_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("Sequence token_ids must use an integer tensor dtype.")
        if self.attention_mask.dtype != torch.bool:
            raise TypeError("Sequence attention_mask must use torch.bool.")
        if self.lengths.dtype not in {torch.int32, torch.int64}:
            raise TypeError("Sequence lengths must use an integer tensor dtype.")
        if (
            isinstance(self.padding_idx, bool)
            or not isinstance(self.padding_idx, int)
            or self.padding_idx < 0
        ):
            raise ValueError("Sequence padding_idx must be a non-negative integer.")
        if self.token_ids.shape[1] == 0:
            raise ValueError("Sequence batches must include at least one token column.")
        if self.lengths.numel() and (
            int(self.lengths.min()) <= 0
            or int(self.lengths.max()) > self.token_ids.shape[1]
        ):
            raise ValueError(
                "Sequence lengths must be positive and no greater than the "
                "padded sequence width."
            )

        expected_mask = torch.arange(
            self.token_ids.shape[1],
            device=self.lengths.device,
        ).unsqueeze(0) < self.lengths.unsqueeze(1)
        if not torch.equal(self.attention_mask, expected_mask):
            raise ValueError(
                "Sequence attention_mask must contain one left-aligned valid "
                "position per declared token in lengths."
            )
        if (
            self.token_ids.masked_select(self.attention_mask) == self.padding_idx
        ).any():
            raise ValueError(
                "Sequence padding_idx cannot appear in valid token positions."
            )
        if (
            self.token_ids.masked_select(~self.attention_mask) != self.padding_idx
        ).any():
            raise ValueError(
                "Sequence padded token positions must contain padding_idx."
            )

    @property
    def batch_size(self) -> int:
        """Number of token sequences represented by this input."""
        return int(self.token_ids.shape[0])

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> SequenceInput:
        """Move every sequence tensor to one device."""
        return SequenceInput(
            token_ids=self.token_ids.to(device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(
                device,
                non_blocking=non_blocking,
            ),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            padding_idx=self.padding_idx,
        )

    def pin_memory(self) -> SequenceInput:
        """Pin every sequence tensor for asynchronous device transfer."""
        return SequenceInput(
            token_ids=self.token_ids.pin_memory(),
            attention_mask=self.attention_mask.pin_memory(),
            lengths=self.lengths.pin_memory(),
            padding_idx=self.padding_idx,
        )


@dataclass(frozen=True)
class SequenceBranchBatch:
    """A structured batch consumed by a token-sequence encoder."""

    values: SequenceInput
    kind: ClassVar[BranchInputKind] = "sequence"

    def __post_init__(self) -> None:
        if not isinstance(self.values, SequenceInput):
            raise TypeError("Sequence branch values must be a SequenceInput.")

    @property
    def batch_size(self) -> int:
        """Number of observations represented by this branch batch."""
        return self.values.batch_size

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> SequenceBranchBatch:
        """Move the structured sequence input to one device."""
        return replace(
            self,
            values=self.values.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> SequenceBranchBatch:
        """Pin every tensor in the structured sequence input."""
        return replace(self, values=self.values.pin_memory())


@dataclass(frozen=True)
class TabularInput:
    """Numeric and categorical tensors consumed by a tabular encoder."""

    numeric: torch.Tensor
    categorical: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.numeric, torch.Tensor):
            raise TypeError("Tabular numeric values must be a torch.Tensor.")
        if not isinstance(self.categorical, torch.Tensor):
            raise TypeError("Tabular categorical values must be a torch.Tensor.")
        if self.numeric.ndim != 2:
            raise ValueError(
                "Tabular numeric values must have shape [batch, numeric_features]."
            )
        if self.categorical.ndim != 2:
            raise ValueError(
                "Tabular categorical values must have shape "
                "[batch, categorical_features]."
            )
        if not torch.is_floating_point(self.numeric):
            raise TypeError("Tabular numeric values must use a floating-point dtype.")
        if self.categorical.dtype not in {torch.int32, torch.int64}:
            raise TypeError(
                "Tabular categorical values must use an integer tensor dtype."
            )
        if self.numeric.shape[0] != self.categorical.shape[0]:
            raise ValueError(
                "Tabular numeric and categorical tensors must have the same batch size."
            )

    @property
    def batch_size(self) -> int:
        """Number of observations in the structured input."""
        return int(self.numeric.shape[0])

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> TabularInput:
        """Move both tensors to one device."""
        return TabularInput(
            numeric=self.numeric.to(device, non_blocking=non_blocking),
            categorical=self.categorical.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> TabularInput:
        """Pin both tensors for asynchronous host-to-device transfer."""
        return TabularInput(
            numeric=self.numeric.pin_memory(),
            categorical=self.categorical.pin_memory(),
        )


@dataclass(frozen=True)
class CompositeInput(Mapping[str, Any]):
    """Named model inputs for multiple encoders on one entity axis."""

    components: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.components, Mapping):
            raise TypeError("Composite components must be a mapping.")
        normalized_components = dict(self.components)
        if len(normalized_components) < 2:
            raise ValueError("Composite inputs must contain at least two components.")

        batch_sizes: set[int] = set()
        for name, values in normalized_components.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Composite component names must be non-empty strings.")
            if isinstance(values, torch.Tensor):
                if values.ndim == 0:
                    raise ValueError(
                        f"Composite component {name!r} must include a batch dimension."
                    )
                batch_sizes.add(int(values.shape[0]))
            elif isinstance(values, (SequenceInput, TabularInput)):
                batch_sizes.add(values.batch_size)
            else:
                raise TypeError(
                    f"Composite component {name!r} must be a torch.Tensor, "
                    "SequenceInput, or TabularInput."
                )
        if len(batch_sizes) != 1:
            raise ValueError(
                "Every composite input component must have the same batch size; "
                f"received {sorted(batch_sizes)}."
            )
        object.__setattr__(self, "components", normalized_components)

    @property
    def batch_size(self) -> int:
        """Number of observations shared by every component."""
        first = next(iter(self.components.values()))
        if isinstance(first, torch.Tensor):
            return int(first.shape[0])
        return int(first.batch_size)

    def __getitem__(self, key: str) -> Any:
        return self.components[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> CompositeInput:
        """Move every component to one device."""
        return CompositeInput(
            {
                name: values.to(device, non_blocking=non_blocking)
                for name, values in self.components.items()
            }
        )

    def pin_memory(self) -> CompositeInput:
        """Pin every component for asynchronous host-to-device transfer."""
        return CompositeInput(
            {name: values.pin_memory() for name, values in self.components.items()}
        )


@dataclass(frozen=True)
class TabularBranchBatch:
    """A structured batch of numeric values and categorical indices."""

    values: TabularInput
    kind: ClassVar[BranchInputKind] = "tabular"

    def __post_init__(self) -> None:
        if not isinstance(self.values, TabularInput):
            raise TypeError("Tabular branch values must be a TabularInput.")

    @property
    def batch_size(self) -> int:
        """Number of observations represented by this branch batch."""
        return self.values.batch_size

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> TabularBranchBatch:
        """Move the structured input to one device."""
        return replace(
            self,
            values=self.values.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> TabularBranchBatch:
        """Pin both tensors in the structured input."""
        return replace(self, values=self.values.pin_memory())


@dataclass(frozen=True)
class CompositeBranchBatch:
    """A named collection of inputs consumed by a composite encoder."""

    values: CompositeInput
    kind: ClassVar[BranchInputKind] = "composite"

    def __post_init__(self) -> None:
        if not isinstance(self.values, CompositeInput):
            raise TypeError("Composite branch values must be a CompositeInput.")

    @property
    def batch_size(self) -> int:
        """Number of observations shared by every component."""
        return self.values.batch_size

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> CompositeBranchBatch:
        """Move every component to one device."""
        return replace(
            self,
            values=self.values.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> CompositeBranchBatch:
        """Pin every component in the structured input."""
        return replace(self, values=self.values.pin_memory())


BranchBatch: TypeAlias = (
    CompositeBranchBatch
    | DenseBranchBatch
    | IDBranchBatch
    | ImageBranchBatch
    | SequenceBranchBatch
    | SparseBranchBatch
    | TabularBranchBatch
    | CustomBranchBatch
)


@dataclass(frozen=True)
class MTPBatch(Mapping[str, Any]):
    """A complete interaction batch with mapping compatibility.

    Attribute access exposes the typed branch inputs. Historical mapping keys
    continue to return the underlying tensors so existing dataloader consumers
    remain compatible.
    """

    instance_id: torch.Tensor
    target_id: torch.Tensor
    instance_input: BranchBatch
    target_input: BranchBatch
    score: torch.Tensor

    _KEYS: ClassVar[tuple[str, ...]] = (
        "instance_id",
        "target_id",
        "instance_features",
        "target_features",
        "score",
    )

    def __post_init__(self) -> None:
        for name, values in {
            "instance_id": self.instance_id,
            "target_id": self.target_id,
            "score": self.score,
        }.items():
            if not isinstance(values, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if values.ndim != 1:
                raise ValueError(f"{name} must have shape [batch].")
        for name, values in {
            "instance_id": self.instance_id,
            "target_id": self.target_id,
        }.items():
            if values.dtype not in {torch.int32, torch.int64}:
                raise TypeError(f"{name} must use an integer tensor dtype.")

        sizes = {
            self.instance_id.shape[0],
            self.target_id.shape[0],
            self.instance_input.batch_size,
            self.target_input.batch_size,
            self.score.shape[0],
        }
        if len(sizes) != 1:
            raise ValueError(
                "Every MTP batch component must have the same batch size; "
                f"received {sorted(sizes)}."
            )

    @property
    def batch_size(self) -> int:
        """Number of interactions represented by the batch."""
        return int(self.score.shape[0])

    @property
    def instance_features(
        self,
    ) -> torch.Tensor | CompositeInput | SequenceInput | TabularInput:
        """Compatibility view of the instance branch input."""
        return self.instance_input.values

    @property
    def target_features(
        self,
    ) -> torch.Tensor | CompositeInput | SequenceInput | TabularInput:
        """Compatibility view of the target branch input."""
        return self.target_input.values

    def __getitem__(self, key: str) -> Any:
        values = {
            "instance_id": self.instance_id,
            "target_id": self.target_id,
            "instance_features": self.instance_features,
            "target_features": self.target_features,
            "score": self.score,
        }
        try:
            return values[key]
        except KeyError as error:
            raise KeyError(key) from error

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> MTPBatch:
        """Move every tensor in the batch to one device."""
        return MTPBatch(
            instance_id=self.instance_id.to(device, non_blocking=non_blocking),
            target_id=self.target_id.to(device, non_blocking=non_blocking),
            instance_input=cast(
                BranchBatch,
                self.instance_input.to(device, non_blocking=non_blocking),
            ),
            target_input=cast(
                BranchBatch,
                self.target_input.to(device, non_blocking=non_blocking),
            ),
            score=self.score.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> MTPBatch:
        """Pin every tensor for asynchronous host-to-device transfer."""
        return MTPBatch(
            instance_id=self.instance_id.pin_memory(),
            target_id=self.target_id.pin_memory(),
            instance_input=cast(BranchBatch, self.instance_input.pin_memory()),
            target_input=cast(BranchBatch, self.target_input.pin_memory()),
            score=self.score.pin_memory(),
        )


@dataclass(frozen=True)
class MTPBatchCollator:
    """Collate dataset items into a typed interaction batch."""

    instance_kind: BranchInputKind
    target_kind: BranchInputKind
    instance_sequence_padding_idx: int = 0
    target_sequence_padding_idx: int = 0
    instance_component_kinds: Mapping[str, BranchInputKind] | None = None
    target_component_kinds: Mapping[str, BranchInputKind] | None = None
    instance_component_padding_indices: Mapping[str, int] | None = None
    target_component_padding_indices: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        for name, kind in {
            "instance_kind": self.instance_kind,
            "target_kind": self.target_kind,
        }.items():
            if kind not in {
                "dense",
                "id",
                "image",
                "sequence",
                "sparse",
                "tabular",
                "custom",
                "composite",
            }:
                raise ValueError(
                    f"{name} must be 'dense', 'id', 'image', 'sequence', "
                    "'sparse', 'tabular', 'custom', or 'composite'."
                )
        for name, padding_idx in {
            "instance_sequence_padding_idx": self.instance_sequence_padding_idx,
            "target_sequence_padding_idx": self.target_sequence_padding_idx,
        }.items():
            if (
                isinstance(padding_idx, bool)
                or not isinstance(padding_idx, int)
                or padding_idx < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer.")
        for branch, kind, component_kinds, padding_indices in (
            (
                "instance",
                self.instance_kind,
                self.instance_component_kinds,
                self.instance_component_padding_indices,
            ),
            (
                "target",
                self.target_kind,
                self.target_component_kinds,
                self.target_component_padding_indices,
            ),
        ):
            _validate_composite_collator_settings(
                branch=branch,
                kind=kind,
                component_kinds=component_kinds,
                padding_indices=padding_indices,
            )

    def __call__(self, items: Sequence[Mapping[str, Any]]) -> MTPBatch:
        if not items:
            raise ValueError("Cannot collate an empty collection of dataset items.")
        core_items = []
        instance_values = []
        target_values = []
        for item in items:
            if not isinstance(item, Mapping):
                raise TypeError("Dataset items must be mappings.")
            try:
                core_items.append(
                    {key: item[key] for key in ("instance_id", "target_id", "score")}
                )
                instance_values.append(item["instance_features"])
                target_values.append(item["target_features"])
            except KeyError as error:
                raise ValueError(
                    f"Dataset items are missing required field {error.args[0]!r}."
                ) from error

        collated = default_collate(core_items)
        if not isinstance(collated, Mapping):
            raise TypeError("Dataset items must collate into a mapping.")
        tensors = {
            key: _require_collated_tensor(collated, key)
            for key in ("instance_id", "target_id", "score")
        }
        return MTPBatch(
            instance_id=tensors["instance_id"],
            target_id=tensors["target_id"],
            instance_input=_build_branch_batch(
                self.instance_kind,
                _collate_branch_values(
                    self.instance_kind,
                    instance_values,
                    padding_idx=self.instance_sequence_padding_idx,
                    component_kinds=self.instance_component_kinds,
                    component_padding_indices=(self.instance_component_padding_indices),
                ),
                padding_idx=self.instance_sequence_padding_idx,
                component_kinds=self.instance_component_kinds,
                component_padding_indices=self.instance_component_padding_indices,
            ),
            target_input=_build_branch_batch(
                self.target_kind,
                _collate_branch_values(
                    self.target_kind,
                    target_values,
                    padding_idx=self.target_sequence_padding_idx,
                    component_kinds=self.target_component_kinds,
                    component_padding_indices=self.target_component_padding_indices,
                ),
                padding_idx=self.target_sequence_padding_idx,
                component_kinds=self.target_component_kinds,
                component_padding_indices=self.target_component_padding_indices,
            ),
            score=tensors["score"],
        )


def _require_collated_tensor(
    collated: Mapping[str, Any],
    key: str,
) -> torch.Tensor:
    try:
        value = collated[key]
    except KeyError as error:
        raise ValueError(f"Collated dataset items are missing {key!r}.") from error
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Collated dataset field {key!r} must be a torch.Tensor, "
            f"not {type(value).__name__}."
        )
    return value


def _collate_branch_values(
    kind: BranchInputKind,
    values: Sequence[Any],
    *,
    padding_idx: int,
    component_kinds: Mapping[str, BranchInputKind] | None = None,
    component_padding_indices: Mapping[str, int] | None = None,
) -> Any:
    if kind == "composite":
        if component_kinds is None:  # pragma: no cover - collator invariant
            raise RuntimeError("Composite collation has no component kinds.")
        expected_names = set(component_kinds)
        for row_index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise TypeError(f"Composite feature row {row_index} must be a mapping.")
            if set(value) != expected_names:
                raise ValueError(
                    f"Composite feature row {row_index} must contain exactly "
                    f"{sorted(expected_names)}; received {sorted(value)}."
                )
        return {
            name: _collate_branch_values(
                component_kind,
                [value[name] for value in values],
                padding_idx=(
                    component_padding_indices.get(name, 0)
                    if component_padding_indices is not None
                    else 0
                ),
            )
            for name, component_kind in component_kinds.items()
        }
    if kind == "sparse":
        return collate_sparse_rows(values)
    if kind == "sequence":
        token_ids, attention_mask, lengths = collate_sequence_rows(
            values,
            padding_idx=padding_idx,
        )
        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "lengths": lengths,
        }
    return default_collate(values)


def _build_branch_batch(
    kind: BranchInputKind,
    values: Any,
    *,
    padding_idx: int,
    component_kinds: Mapping[str, BranchInputKind] | None = None,
    component_padding_indices: Mapping[str, int] | None = None,
) -> BranchBatch:
    if kind == "composite":
        if not isinstance(values, Mapping):
            raise TypeError("Collated composite features must be a mapping.")
        if component_kinds is None:  # pragma: no cover - collator invariant
            raise RuntimeError("Composite batching has no component kinds.")
        return CompositeBranchBatch(
            CompositeInput(
                {
                    name: _build_branch_batch(
                        component_kind,
                        values[name],
                        padding_idx=(
                            component_padding_indices.get(name, 0)
                            if component_padding_indices is not None
                            else 0
                        ),
                    ).values
                    for name, component_kind in component_kinds.items()
                }
            )
        )
    if kind == "sequence":
        if not isinstance(values, Mapping):
            raise TypeError(
                "Collated sequence features must contain token IDs, an attention "
                "mask, and lengths."
            )
        return SequenceBranchBatch(
            SequenceInput(
                token_ids=_require_collated_tensor(values, "token_ids"),
                attention_mask=_require_collated_tensor(values, "attention_mask"),
                lengths=_require_collated_tensor(values, "lengths"),
                padding_idx=padding_idx,
            )
        )
    if kind == "tabular":
        if not isinstance(values, Mapping):
            raise TypeError(
                "Collated tabular features must contain numeric and categorical "
                "tensors."
            )
        numeric = _require_collated_tensor(values, "numeric")
        categorical = _require_collated_tensor(values, "categorical")
        if not torch.is_floating_point(numeric):
            numeric = numeric.float()
        return TabularBranchBatch(
            TabularInput(
                numeric=numeric,
                categorical=categorical,
            )
        )
    if kind == "sparse":
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                "Collated sparse branch features must be a PyTorch sparse tensor."
            )
        if not torch.is_floating_point(values):
            values = values.float()
        return SparseBranchBatch(values)
    if not isinstance(values, torch.Tensor):
        raise TypeError(
            f"Collated {kind} branch features must be a torch.Tensor, "
            f"not {type(values).__name__}."
        )
    if kind in {"dense", "image"} and not torch.is_floating_point(values):
        values = values.float()
    if kind == "dense":
        return DenseBranchBatch(values)
    if kind == "id":
        return IDBranchBatch(values)
    if kind == "image":
        return ImageBranchBatch(values)
    return CustomBranchBatch(values)


def _validate_composite_collator_settings(
    *,
    branch: str,
    kind: BranchInputKind,
    component_kinds: Mapping[str, BranchInputKind] | None,
    padding_indices: Mapping[str, int] | None,
) -> None:
    if kind != "composite":
        if component_kinds is not None or padding_indices is not None:
            raise ValueError(
                f"{branch}_component settings require {branch}_kind='composite'."
            )
        return
    if not isinstance(component_kinds, Mapping) or len(component_kinds) < 2:
        raise ValueError(
            f"{branch}_component_kinds must contain at least two components."
        )
    for name, component_kind in component_kinds.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Composite component names must be non-empty strings.")
        if component_kind == "composite":
            raise ValueError("Nested composite components are not supported.")
        if component_kind not in {
            "dense",
            "id",
            "image",
            "sequence",
            "sparse",
            "tabular",
            "custom",
        }:
            raise ValueError(
                f"Unsupported {branch} composite component kind {component_kind!r}."
            )
    if padding_indices is None:
        return
    unknown_names = set(padding_indices).difference(component_kinds)
    if unknown_names:
        raise ValueError(
            f"{branch}_component_padding_indices contains unknown components: "
            f"{sorted(unknown_names)}."
        )
    for name, padding_idx in padding_indices.items():
        if (
            isinstance(padding_idx, bool)
            or not isinstance(padding_idx, int)
            or padding_idx < 0
        ):
            raise ValueError(
                f"Padding index for composite component {name!r} must be a "
                "non-negative integer."
            )
