"""Dataset and dataloader construction for DeepMTP experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from DeepMTP.config import DeepMTPConfig
from DeepMTP.utils.data_utils import BaseDataset


@dataclass(frozen=True)
class TrainingDataLoaders:
    """Dataloaders required by one training run."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader


class DataLoaderFactory:
    """Build configured dataloaders without trainer-specific orchestration."""

    def __init__(
        self,
        config: DeepMTPConfig | Mapping[str, Any],
        device: torch.device,
    ):
        self.config = DeepMTPConfig.from_mapping(config)
        self.dataset_config = self.config.to_dict()
        self.device = device

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

        return TrainingDataLoaders(
            train=self._build(
                train_data,
                batch_size=self.config.train_batchsize,
                train=True,
            ),
            validation=self._build(
                validation_data,
                batch_size=self.config.val_batchsize,
                train=False,
            ),
            test=self._build(
                test_data,
                batch_size=self.config.val_batchsize,
                train=False,
            ),
        )

    def for_prediction(self, data: Mapping[str, Any]) -> DataLoader:
        """Build a deterministic inference dataloader."""
        return self._build(
            data,
            batch_size=self.config.val_batchsize,
            train=False,
        )

    def _build(
        self,
        data: Mapping[str, Any],
        *,
        batch_size: int,
        train: bool,
    ) -> DataLoader:
        dataset = BaseDataset(
            self.dataset_config,
            data["y"],
            data["X_instance"],
            data["X_target"],
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
