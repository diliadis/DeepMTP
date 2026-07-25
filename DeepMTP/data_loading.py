"""Dataset and dataloader construction for DeepMTP experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from DeepMTP.config import DeepMTPConfig
from DeepMTP.utils.data_utils import BaseDataset

_DATA_SPLIT_KEYS = frozenset({"y", "X_instance", "X_target"})


@dataclass(frozen=True)
class TrainingDataLoaders:
    """Dataloaders required by one training run."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader


@dataclass(frozen=True)
class _DatasetComponents:
    """Validated inputs needed to construct one ``BaseDataset``."""

    interactions: Mapping[str, Any]
    instance_features: Mapping[str, Any] | None
    target_features: Mapping[str, Any] | None


def _normalize_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if not isinstance(device, str):
        raise TypeError("device must be a torch.device or device string.")
    try:
        return torch.device(device)
    except RuntimeError as exc:
        raise ValueError(f"Invalid PyTorch device string: {device!r}.") from exc


def _validate_data_split(
    data: object,
    *,
    parameter_name: str,
) -> _DatasetComponents:
    if not isinstance(data, Mapping):
        raise TypeError(f"{parameter_name} must be a mapping.")

    missing_keys = _DATA_SPLIT_KEYS.difference(data)
    if missing_keys:
        raise ValueError(
            f"{parameter_name} is missing required keys: {sorted(missing_keys)}."
        )

    interactions = data["y"]
    if not isinstance(interactions, Mapping):
        raise TypeError(f"{parameter_name}['y'] must be a mapping.")

    features: dict[str, Mapping[str, Any] | None] = {}
    for key in ("X_instance", "X_target"):
        value = data[key]
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f"{parameter_name}[{key!r}] must be a mapping or None.")
        features[key] = value

    return _DatasetComponents(
        interactions=interactions,
        instance_features=features["X_instance"],
        target_features=features["X_target"],
    )


class DataLoaderFactory:
    """Build configured dataloaders without trainer-specific orchestration."""

    def __init__(
        self,
        config: DeepMTPConfig | Mapping[str, Any],
        device: torch.device | str,
    ) -> None:
        self.config = DeepMTPConfig.from_mapping(config)
        self.dataset_config = self.config.to_dict(include_defaults=True)
        self.device = _normalize_device(device)

    def for_training(
        self,
        train_data: Mapping[str, Any],
        validation_data: Mapping[str, Any] | None,
        test_data: Mapping[str, Any] | None,
    ) -> TrainingDataLoaders:
        """Build the train, validation, and test dataloaders."""
        if validation_data is None:
            raise ValueError("validation_data is required for training.")
        if test_data is None:
            raise ValueError("test_data is required for training.")

        train_components = _validate_data_split(
            train_data,
            parameter_name="train_data",
        )
        validation_components = _validate_data_split(
            validation_data,
            parameter_name="validation_data",
        )
        test_components = _validate_data_split(
            test_data,
            parameter_name="test_data",
        )
        return TrainingDataLoaders(
            train=self._build(
                train_components,
                batch_size=self.config.train_batchsize,
                train=True,
            ),
            validation=self._build(
                validation_components,
                batch_size=self.config.val_batchsize,
                train=False,
            ),
            test=self._build(
                test_components,
                batch_size=self.config.val_batchsize,
                train=False,
            ),
        )

    def for_prediction(self, data: Mapping[str, Any]) -> DataLoader:
        """Build a deterministic inference dataloader."""
        return self._build(
            _validate_data_split(data, parameter_name="data"),
            batch_size=self.config.val_batchsize,
            train=False,
        )

    def _build(
        self,
        components: _DatasetComponents,
        *,
        batch_size: int,
        train: bool,
    ) -> DataLoader:
        dataset = BaseDataset(
            self.dataset_config,
            components.interactions,
            components.instance_features,
            components.target_features,
            instance_transform=(
                self.config.instance_train_transforms
                if train
                else self.config.instance_inference_transforms
            ),
            target_transform=(
                self.config.target_train_transforms
                if train
                else self.config.target_inference_transforms
            ),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            generator=self._generator() if train else None,
        )

    def _generator(self) -> torch.Generator | None:
        if self.config.random_seed is None:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.config.random_seed)
        return generator
