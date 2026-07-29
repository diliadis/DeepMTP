"""Dataset and dataloader construction for DeepMTP experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypedDict, cast

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from DeepMTP.config import BranchArchitecture, DeepMTPConfig

__all__ = [
    "BaseDataset",
    "DataLoaderFactory",
    "DatasetItem",
    "MTPSampler",
    "SamplerMode",
    "TrainingDataLoaders",
]

_DATA_SPLIT_KEYS = frozenset({"y", "X_instance", "X_target"})
SamplerMode: TypeAlias = Literal["micro", "macro", "instance"]


class DatasetItem(TypedDict):
    """One model-ready interaction returned by ``BaseDataset``."""

    instance_id: int
    target_id: int
    instance_features: Any
    target_features: Any
    score: Any


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


@dataclass(frozen=True)
class _DatasetModelSettings:
    """Typed fields needed to turn interactions into model inputs."""

    instance_input_dim: int | None
    target_input_dim: int | None
    instance_architecture: BranchArchitecture | None
    target_architecture: BranchArchitecture | None

    @classmethod
    def from_config(
        cls,
        config: DeepMTPConfig | Mapping[str, Any],
    ) -> _DatasetModelSettings:
        if isinstance(config, DeepMTPConfig):
            return cls(
                instance_input_dim=config.instance_branch_input_dim,
                target_input_dim=config.target_branch_input_dim,
                instance_architecture=config.instance_branch_architecture,
                target_architecture=config.target_branch_architecture,
            )
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or DeepMTPConfig.")
        return cls(
            instance_input_dim=config.get("instance_branch_input_dim"),
            target_input_dim=config.get("target_branch_input_dim"),
            instance_architecture=config.get("instance_branch_architecture"),
            target_architecture=config.get("target_branch_architecture"),
        )


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


class BaseDataset(Dataset):
    """Resolve interaction rows to model-ready entity features."""

    def __init__(
        self,
        config: DeepMTPConfig | Mapping[str, Any],
        data: Mapping[str, Any],
        instance_features: Mapping[str, Any] | None,
        target_features: Mapping[str, Any] | None,
        instance_transform: Callable[[Image.Image], Any] | None = None,
        target_transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        settings = _DatasetModelSettings.from_config(config)
        self.config = (
            config.to_dict(include_defaults=True)
            if isinstance(config, DeepMTPConfig)
            else dict(config)
        )
        self.instance_branch_input_dim = settings.instance_input_dim
        self.target_branch_input_dim = settings.target_input_dim
        self.instance_branch_architecture = settings.instance_architecture
        self.target_branch_architecture = settings.target_architecture
        self.instance_transform = instance_transform
        self.target_transform = target_transform

        triplet_data = data.get("data")
        if not isinstance(triplet_data, pd.DataFrame):
            raise TypeError("data['data'] must be a pandas DataFrame.")
        required_columns = {"instance_id", "target_id", "value"}
        missing_columns = required_columns.difference(triplet_data.columns)
        if missing_columns:
            raise ValueError(
                "Interaction data is missing required columns: "
                f"{sorted(missing_columns)}."
            )
        if triplet_data.empty:
            raise ValueError("Interaction data must contain at least one observation.")
        for id_column, entity_name in (
            ("instance_id", "Instance"),
            ("target_id", "Target"),
        ):
            if not pd.api.types.is_integer_dtype(triplet_data[id_column]):
                raise TypeError(f"{entity_name} IDs must contain integers.")
        if triplet_data["value"].isna().any():
            raise ValueError("Interaction scores cannot be missing.")
        self.triplet_data = triplet_data.copy(deep=True)

        self.instance_features = self._prepare_feature_frame(
            instance_features,
            id_column="instance_id",
            architecture=self.instance_branch_architecture,
            transform=instance_transform,
            entity_name="Instance",
        )
        self.target_features = self._prepare_feature_frame(
            target_features,
            id_column="target_id",
            architecture=self.target_branch_architecture,
            transform=target_transform,
            entity_name="Target",
        )

        if self.instance_features is None:
            self.instance_branch_input_dim = self._validate_input_dimension(
                self.instance_branch_input_dim,
                "instance_branch_input_dim",
            )
            self._validate_entity_ids(
                id_column="instance_id",
                input_dim=self.instance_branch_input_dim,
                entity_name="Instance",
            )
        if self.target_features is None:
            self.target_branch_input_dim = self._validate_input_dimension(
                self.target_branch_input_dim,
                "target_branch_input_dim",
            )
            self._validate_entity_ids(
                id_column="target_id",
                input_dim=self.target_branch_input_dim,
                entity_name="Target",
            )

    @staticmethod
    def _validate_input_dimension(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{name} must be a positive integer when features are not supplied."
            )
        return value

    def _prepare_feature_frame(
        self,
        source: Mapping[str, Any] | None,
        *,
        id_column: Literal["instance_id", "target_id"],
        architecture: object,
        transform: Callable[[Image.Image], Any] | None,
        entity_name: Literal["Instance", "Target"],
    ) -> pd.DataFrame | None:
        if source is None:
            if architecture == "CONV":
                raise ValueError(
                    f"{entity_name} features are required for a CONV branch."
                )
            return None
        if architecture == "EMBEDDING":
            raise ValueError(
                f"{entity_name} side features cannot be supplied to an "
                "EMBEDDING branch. Use an MLP, CONV, or CUSTOM branch when "
                "side features are required."
            )

        frame = source.get("data")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"{entity_name} feature source must contain a pandas DataFrame."
            )
        if "id" not in frame.columns:
            raise ValueError(f"{entity_name} features must contain an 'id' column.")
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError(f"{entity_name} feature IDs must be unique and non-null.")

        value_column = "dir" if architecture == "CONV" else "features"
        if value_column not in frame.columns:
            raise ValueError(
                f"{entity_name} features must contain a {value_column!r} column."
            )
        if architecture == "CONV" and transform is None:
            raise ValueError(
                f"{entity_name.lower()}_transform is required for a CONV branch."
            )

        prepared_frame = frame.copy(deep=True).set_index("id", drop=False)
        interaction_ids = set(self.triplet_data[id_column])
        missing_ids = interaction_ids.difference(prepared_frame.index)
        if missing_ids:
            raise ValueError(
                f"{entity_name} features are missing IDs used by interactions: "
                f"{sorted(missing_ids)!r}."
            )
        return prepared_frame

    def _validate_entity_ids(
        self,
        id_column: Literal["instance_id", "target_id"],
        input_dim: int,
        entity_name: Literal["Instance", "Target"],
    ) -> None:
        ids = self.triplet_data[id_column].to_numpy()
        if not pd.api.types.is_integer_dtype(ids):
            raise TypeError(f"{entity_name} IDs must contain integers.")
        invalid_ids = ids[(ids < 0) | (ids >= input_dim)]
        if invalid_ids.size:
            invalid_id = int(invalid_ids[0])
            raise ValueError(
                f"{entity_name} IDs must be zero-based and smaller than "
                f"{input_dim} when ID-based inputs are used; found "
                f"{invalid_id}. Remap the IDs or provide {entity_name.lower()} "
                "features."
            )

    @staticmethod
    def _load_image_features(
        frame: pd.DataFrame,
        entity_id: int,
        transform: Callable[[Image.Image], Any],
    ) -> Any:
        with Image.open(frame.loc[entity_id, "dir"]) as image:
            return transform(image.convert("RGB"))

    def __len__(self) -> int:
        return len(self.triplet_data)

    def __getitem__(self, idx: int) -> DatasetItem:
        row = self.triplet_data.iloc[idx]
        instance_id = int(row["instance_id"])
        target_id = int(row["target_id"])
        value = row["value"]

        if self.instance_features is not None:
            if self.instance_branch_architecture == "CONV":
                instance_features_vec = self._load_image_features(
                    self.instance_features,
                    instance_id,
                    cast(Callable[[Image.Image], Any], self.instance_transform),
                )
            else:
                instance_features_vec = self.instance_features.loc[
                    instance_id, "features"
                ]
        else:
            if self.instance_branch_architecture == "EMBEDDING":
                instance_features_vec = instance_id
            else:
                instance_features_vec = np.zeros(
                    self.instance_branch_input_dim, dtype=np.float32
                )
                instance_features_vec[instance_id] = 1

        if self.target_features is not None:
            if self.target_branch_architecture == "CONV":
                target_features_vec = self._load_image_features(
                    self.target_features,
                    target_id,
                    cast(Callable[[Image.Image], Any], self.target_transform),
                )
            else:
                target_features_vec = self.target_features.loc[target_id, "features"]
        else:
            if self.target_branch_architecture == "EMBEDDING":
                target_features_vec = target_id
            else:
                target_features_vec = np.zeros(
                    self.target_branch_input_dim, dtype=np.float32
                )
                target_features_vec[target_id] = 1

        return {
            "instance_id": instance_id,
            "target_id": target_id,
            "instance_features": instance_features_vec,
            "target_features": target_features_vec,
            "score": value,
        }


class MTPSampler(Sampler):
    """Undersample classes and provide a fresh order for every iteration.

    ``micro`` balances classes globally, ``macro`` balances them independently
    for each target, and ``instance`` balances them independently for each
    instance. Grouped balancing can omit entities that are absent from the
    retained observations.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        mode: SamplerMode = "micro",
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Build a reproducible class-balancing sampler.

        Args:
            data: Interaction triplets to balance.
            mode: Balance globally, per target, or per instance.
            verbose: Retained for compatibility with the legacy sampler API.
            random_state: Seed for isolated undersampling and iteration order,
                or ``None`` for non-deterministic behavior.

        Raises:
            TypeError: If an input has the wrong type or contains unhashable values.
            ValueError: If an option or the interaction data is invalid.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(mode, str) or mode not in {"micro", "macro", "instance"}:
            raise ValueError(
                "mode must be one of 'micro', 'macro', or 'instance'; "
                f"received {mode!r}."
            )
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, int)
            or random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer or None.")
        required_columns = {"instance_id", "target_id", "value"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(
                f"Sampler data is missing required columns: {sorted(missing_columns)}."
            )
        if data.empty:
            raise ValueError("Sampler data must contain at least one observation.")
        if data["value"].isna().any():
            raise ValueError("Sampler class values cannot be missing.")
        if data[["instance_id", "target_id"]].isna().any().any():
            raise ValueError("Sampler entity IDs cannot be missing.")
        for column in ("instance_id", "target_id", "value"):
            if not data[column].map(pd.api.types.is_hashable).all():
                raise TypeError(f"Sampler {column!r} values must be hashable.")

        self.data = data.copy(deep=True).reset_index(drop=True)
        self.verbose = verbose
        self.random_state = random_state
        self._random_generator = np.random.default_rng(random_state)
        if mode == "micro":
            self.data = self.get_balanced_df(self.data)
        elif mode == "macro":
            self.data = pd.concat(
                self.get_balanced_df(group)
                for _, group in self.data.groupby(
                    "target_id",
                    sort=False,
                    observed=True,
                )
            )
        else:
            self.data = pd.concat(
                self.get_balanced_df(group)
                for _, group in self.data.groupby(
                    "instance_id",
                    sort=False,
                    observed=True,
                )
            )

    def get_balanced_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return one frame undersampled to its smallest class size."""
        class_groups = [
            group
            for _, group in data.groupby(
                "value",
                sort=False,
                observed=True,
            )
        ]
        if not class_groups:
            raise ValueError("Cannot balance data without class values.")

        minority_class_size = min(len(group) for group in class_groups)
        balanced_groups = []
        for group in class_groups:
            if len(group) == minority_class_size:
                balanced_groups.append(group)
                continue
            selected_positions = self._random_generator.choice(
                len(group),
                size=minority_class_size,
                replace=False,
            )
            balanced_groups.append(group.iloc[selected_positions])

        return pd.concat(balanced_groups)

    def __iter__(self) -> Iterator[int]:
        indices = self.data.index.to_numpy(dtype=int, copy=True)
        return iter(self._random_generator.permutation(indices).tolist())

    def __len__(self) -> int:
        return len(self.data)


class DataLoaderFactory:
    """Build configured dataloaders without trainer-specific orchestration."""

    def __init__(
        self,
        config: DeepMTPConfig | Mapping[str, Any],
        device: torch.device | str,
    ) -> None:
        self.config = DeepMTPConfig.from_mapping(config)
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
            self.config,
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
