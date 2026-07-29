"""Typed model-input batches shared by data loading and training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, TypeAlias, cast

import torch
from torch.utils.data import default_collate

__all__ = [
    "BranchBatch",
    "BranchInputKind",
    "CustomBranchBatch",
    "DenseBranchBatch",
    "IDBranchBatch",
    "ImageBranchBatch",
    "MTPBatch",
    "MTPBatchCollator",
    "TensorBranchBatch",
]

BranchInputKind: TypeAlias = Literal["dense", "id", "image", "custom"]


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


class CustomBranchBatch(TensorBranchBatch):
    """A tensor batch passed unchanged to a user-supplied branch."""

    kind: ClassVar[BranchInputKind] = "custom"


BranchBatch: TypeAlias = (
    DenseBranchBatch | IDBranchBatch | ImageBranchBatch | CustomBranchBatch
)


@dataclass(frozen=True)
class MTPBatch(Mapping[str, torch.Tensor]):
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
    def instance_features(self) -> torch.Tensor:
        """Compatibility view of the instance branch tensor."""
        return self.instance_input.values

    @property
    def target_features(self) -> torch.Tensor:
        """Compatibility view of the target branch tensor."""
        return self.target_input.values

    def __getitem__(self, key: str) -> torch.Tensor:
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

    def __post_init__(self) -> None:
        for name, kind in {
            "instance_kind": self.instance_kind,
            "target_kind": self.target_kind,
        }.items():
            if kind not in {"dense", "id", "image", "custom"}:
                raise ValueError(f"{name} must be 'dense', 'id', 'image', or 'custom'.")

    def __call__(self, items: Sequence[Mapping[str, Any]]) -> MTPBatch:
        if not items:
            raise ValueError("Cannot collate an empty collection of dataset items.")
        collated = default_collate(items)
        if not isinstance(collated, Mapping):
            raise TypeError("Dataset items must collate into a mapping.")
        tensors = {
            key: _require_collated_tensor(collated, key) for key in MTPBatch._KEYS
        }
        return MTPBatch(
            instance_id=tensors["instance_id"],
            target_id=tensors["target_id"],
            instance_input=_build_branch_batch(
                self.instance_kind,
                tensors["instance_features"],
            ),
            target_input=_build_branch_batch(
                self.target_kind,
                tensors["target_features"],
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


def _build_branch_batch(
    kind: BranchInputKind,
    values: torch.Tensor,
) -> BranchBatch:
    if kind in {"dense", "image"} and not torch.is_floating_point(values):
        values = values.float()
    if kind == "dense":
        return DenseBranchBatch(values)
    if kind == "id":
        return IDBranchBatch(values)
    if kind == "image":
        return ImageBranchBatch(values)
    return CustomBranchBatch(values)
