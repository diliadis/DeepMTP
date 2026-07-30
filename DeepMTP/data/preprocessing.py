"""Serializable preprocessing state shared by data preparation and checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np

DenseScalingMethod: TypeAlias = Literal["MinMax", "Standard"]

DATA_PREPARATION_STATE_KEY = "_deepmtp_data_preparation_state"
_DENSE_SCALING_STATE_VERSION = 1
_DENSE_PREPROCESSING_STATE_VERSION = 1
_DATA_PREPARATION_STATE_VERSION = 1

__all__ = [
    "DATA_PREPARATION_STATE_KEY",
    "DataPreparationState",
    "DensePreprocessingState",
    "DenseScalingMethod",
    "DenseScalingState",
    "transform_dense_features",
]


@dataclass(frozen=True)
class DenseScalingState:
    """Parameters needed to replay one fitted dense-feature transformation."""

    method: DenseScalingMethod
    feature_count: int
    offsets: tuple[float, ...]
    scales: tuple[float, ...]
    data_minimums: tuple[float, ...] = ()
    data_maximums: tuple[float, ...] = ()
    feature_range: tuple[float, float] = (0.0, 1.0)

    @classmethod
    def from_fitted_scaler(
        cls,
        scaler: object,
        *,
        method: DenseScalingMethod,
    ) -> DenseScalingState:
        """Capture explicit numeric state from a fitted scikit-learn scaler."""
        feature_count = getattr(scaler, "n_features_in_", None)
        if (
            isinstance(feature_count, bool)
            or not isinstance(feature_count, (int, np.integer))
            or int(feature_count) <= 0
        ):
            raise ValueError("The fitted dense scaler has no valid feature width.")

        if method == "Standard":
            offsets = getattr(scaler, "mean_", None)
            scales = getattr(scaler, "scale_", None)
            data_minimums: Sequence[float] = ()
            data_maximums: Sequence[float] = ()
            feature_range = (0.0, 1.0)
        else:
            offsets = getattr(scaler, "min_", None)
            scales = getattr(scaler, "scale_", None)
            data_minimums = getattr(scaler, "data_min_", ())
            data_maximums = getattr(scaler, "data_max_", ())
            feature_range = cast(
                tuple[float, float],
                tuple(getattr(scaler, "feature_range", (0.0, 1.0))),
            )

        return cls.from_config(
            {
                "version": _DENSE_SCALING_STATE_VERSION,
                "method": method,
                "feature_count": int(feature_count),
                "offsets": _array_list(offsets, parameter_name="offsets"),
                "scales": _array_list(scales, parameter_name="scales"),
                "data_minimums": _array_list(
                    data_minimums,
                    parameter_name="data_minimums",
                ),
                "data_maximums": _array_list(
                    data_maximums,
                    parameter_name="data_maximums",
                ),
                "feature_range": list(feature_range),
            }
        )

    @classmethod
    def from_config(cls, value: object) -> DenseScalingState:
        """Validate and restore a JSON-compatible scaling-state mapping."""
        if not isinstance(value, Mapping):
            raise TypeError("Dense scaling state must be a mapping.")
        version = value.get("version", _DENSE_SCALING_STATE_VERSION)
        if version != _DENSE_SCALING_STATE_VERSION:
            raise ValueError(f"Unsupported dense scaling state version {version!r}.")

        method = value.get("method")
        if method not in {"MinMax", "Standard"}:
            raise ValueError("Dense scaling method must be 'MinMax' or 'Standard'.")
        feature_count = value.get("feature_count")
        if (
            isinstance(feature_count, bool)
            or not isinstance(feature_count, int)
            or feature_count <= 0
        ):
            raise ValueError("Dense scaling feature_count must be a positive integer.")

        offsets = _finite_float_tuple(
            value.get("offsets"),
            parameter_name="offsets",
            expected_length=feature_count,
        )
        scales = _finite_float_tuple(
            value.get("scales"),
            parameter_name="scales",
            expected_length=feature_count,
        )
        if any(scale <= 0 for scale in scales):
            raise ValueError("Dense scaling scales must contain positive values.")

        raw_minimums = value.get("data_minimums", ())
        raw_maximums = value.get("data_maximums", ())
        expected_minmax_width = feature_count if method == "MinMax" else 0
        data_minimums = _finite_float_tuple(
            raw_minimums,
            parameter_name="data_minimums",
            expected_length=expected_minmax_width,
        )
        data_maximums = _finite_float_tuple(
            raw_maximums,
            parameter_name="data_maximums",
            expected_length=expected_minmax_width,
        )

        feature_range = _finite_float_tuple(
            value.get("feature_range", (0.0, 1.0)),
            parameter_name="feature_range",
            expected_length=2,
        )
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Dense scaling feature_range must have a lower first value."
            )

        return cls(
            method=cast(DenseScalingMethod, method),
            feature_count=feature_count,
            offsets=offsets,
            scales=scales,
            data_minimums=data_minimums,
            data_maximums=data_maximums,
            feature_range=cast(tuple[float, float], feature_range),
        )

    def transform(self, values: object) -> np.ndarray:
        """Transform one vector or matrix using the saved training statistics."""
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("Dense features must be numeric.")
        if array.ndim not in {1, 2}:
            raise ValueError("Dense features must be a one- or two-dimensional array.")
        if array.shape[-1] != self.feature_count:
            raise ValueError(
                "Dense feature width does not match saved preprocessing state: "
                f"expected {self.feature_count}, received {array.shape[-1]}."
            )
        numeric = array.astype(np.float64, copy=False)
        offsets = np.asarray(self.offsets, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        if self.method == "Standard":
            return (numeric - offsets) / scales
        return numeric * scales + offsets

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible checkpoint metadata."""
        return {
            "version": _DENSE_SCALING_STATE_VERSION,
            "method": self.method,
            "feature_count": self.feature_count,
            "offsets": list(self.offsets),
            "scales": list(self.scales),
            "data_minimums": list(self.data_minimums),
            "data_maximums": list(self.data_maximums),
            "feature_range": list(self.feature_range),
        }


@dataclass(frozen=True)
class DensePreprocessingState:
    """Dense scaling state for the instance and target feature axes."""

    instance: DenseScalingState | None = None
    target: DenseScalingState | None = None

    @classmethod
    def from_config(cls, value: object) -> DensePreprocessingState:
        """Validate a combined dense-preprocessing state mapping."""
        if not isinstance(value, Mapping):
            raise TypeError("dense_preprocessing_state must be a mapping.")
        version = value.get("version", _DENSE_PREPROCESSING_STATE_VERSION)
        if version != _DENSE_PREPROCESSING_STATE_VERSION:
            raise ValueError(
                f"Unsupported dense preprocessing state version {version!r}."
            )
        instance = value.get("instance")
        target = value.get("target")
        return cls(
            instance=(
                None if instance is None else DenseScalingState.from_config(instance)
            ),
            target=None if target is None else DenseScalingState.from_config(target),
        )

    def for_branch(self, branch: str) -> DenseScalingState | None:
        """Return the scaler state for one entity axis."""
        if branch == "instance":
            return self.instance
        if branch == "target":
            return self.target
        raise ValueError("branch must be 'instance' or 'target'.")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible checkpoint metadata."""
        return {
            "version": _DENSE_PREPROCESSING_STATE_VERSION,
            "instance": None if self.instance is None else self.instance.to_dict(),
            "target": None if self.target is None else self.target.to_dict(),
        }


@dataclass(frozen=True)
class DataPreparationState:
    """Split provenance and fitted dense preprocessing from ``data_process``."""

    validation_setting: str
    split_method: str
    split_ratio: tuple[tuple[str, float], ...]
    shuffle: bool
    seed: int | None
    dense: DensePreprocessingState

    @classmethod
    def from_config(cls, value: object) -> DataPreparationState:
        """Validate a saved data-preparation state mapping."""
        if not isinstance(value, Mapping):
            raise TypeError("data_preparation_state must be a mapping.")
        version = value.get("version", _DATA_PREPARATION_STATE_VERSION)
        if version != _DATA_PREPARATION_STATE_VERSION:
            raise ValueError(f"Unsupported data preparation state version {version!r}.")
        validation_setting = value.get("validation_setting")
        if validation_setting not in {"A", "B", "C", "D"}:
            raise ValueError(
                "data_preparation_state validation_setting must be A, B, C, or D."
            )
        split_method = value.get("split_method")
        if not isinstance(split_method, str) or not split_method:
            raise ValueError(
                "data_preparation_state split_method must be a non-empty string."
            )
        raw_ratio = value.get("split_ratio")
        if not isinstance(raw_ratio, Mapping):
            raise TypeError("data_preparation_state split_ratio must be a mapping.")
        expected_keys = {"train", "val", "test"}
        if set(raw_ratio) != expected_keys:
            raise ValueError(
                "data_preparation_state split_ratio must contain train, val, and test."
            )
        ratio_items: list[tuple[str, float]] = []
        for name in ("train", "val", "test"):
            raw_fraction = raw_ratio[name]
            if isinstance(raw_fraction, bool) or not isinstance(
                raw_fraction,
                (int, float),
            ):
                raise TypeError("data_preparation_state split ratios must be numeric.")
            fraction = float(raw_fraction)
            if not np.isfinite(fraction) or fraction < 0:
                raise ValueError(
                    "data_preparation_state split ratios must be finite and "
                    "non-negative."
                )
            ratio_items.append((name, fraction))
        if not np.isclose(sum(value for _, value in ratio_items), 1.0):
            raise ValueError("data_preparation_state split ratios must sum to 1.")

        shuffle = value.get("shuffle")
        if not isinstance(shuffle, bool):
            raise TypeError("data_preparation_state shuffle must be a boolean.")
        seed = value.get("seed")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError(
                "data_preparation_state seed must be a non-negative integer or None."
            )
        dense = DensePreprocessingState.from_config(
            value.get(
                "dense",
                {
                    "version": _DENSE_PREPROCESSING_STATE_VERSION,
                    "instance": None,
                    "target": None,
                },
            )
        )
        return cls(
            validation_setting=validation_setting,
            split_method=split_method,
            split_ratio=tuple(ratio_items),
            shuffle=shuffle,
            seed=seed,
            dense=dense,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible checkpoint metadata."""
        return {
            "version": _DATA_PREPARATION_STATE_VERSION,
            "validation_setting": self.validation_setting,
            "split_method": self.split_method,
            "split_ratio": dict(self.split_ratio),
            "shuffle": self.shuffle,
            "seed": self.seed,
            "dense": self.dense.to_dict(),
        }


def transform_dense_features(
    values: object,
    state: Mapping[str, Any] | DenseScalingState,
) -> np.ndarray:
    """Replay a saved dense-feature transformation."""
    restored = (
        state
        if isinstance(state, DenseScalingState)
        else DenseScalingState.from_config(state)
    )
    return restored.transform(values)


def _array_list(values: object, *, parameter_name: str) -> list[float]:
    if values is None:
        raise ValueError(f"The fitted dense scaler is missing {parameter_name}.")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"The fitted dense scaler {parameter_name} must be 1-D.")
    return [float(value) for value in array]


def _finite_float_tuple(
    values: object,
    *,
    parameter_name: str,
    expected_length: int,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{parameter_name} must be a sequence of finite numbers.")
    if len(values) != expected_length:
        raise ValueError(
            f"{parameter_name} must contain {expected_length} values; "
            f"received {len(values)}."
        )
    validated: list[float] = []
    for raw_value in values:
        if isinstance(raw_value, bool) or not isinstance(
            raw_value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(f"{parameter_name} must contain finite numbers.")
        numeric_value = float(raw_value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"{parameter_name} must contain finite numbers.")
        validated.append(numeric_value)
    return tuple(validated)
