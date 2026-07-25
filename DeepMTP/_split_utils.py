"""Internal split validation and partitioning utilities."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np

_SPLIT_NAMES = frozenset({"train", "val", "test"})


def validate_split_ratio(
    ratio: object,
    *,
    parameter_name: str = "ratio",
    require_train: bool = True,
    require_validation: bool = True,
) -> dict[str, float]:
    """Validate a split ratio and return an independent normalized mapping."""
    if not isinstance(ratio, Mapping):
        raise TypeError(f"{parameter_name} must be a mapping.")

    raw_ratio = dict(ratio)
    unknown = set(raw_ratio).difference(_SPLIT_NAMES)
    if unknown:
        unknown_names = sorted(repr(key) for key in unknown)
        raise ValueError(f"{parameter_name} contains unknown keys: {unknown_names}.")

    required = {"test"}
    if require_train:
        required.add("train")
    if require_validation:
        required.add("val")
    missing = required.difference(raw_ratio)
    if "test" in missing and "val" in raw_ratio:
        raise ValueError(
            f"{parameter_name} requires a test ratio when val is provided."
        )
    if missing:
        raise ValueError(
            f"{parameter_name} is missing required keys: {sorted(missing)}."
        )

    validated_values: dict[str, float] = {}
    for split_name in ("train", "val", "test"):
        if split_name not in raw_ratio:
            continue
        value = raw_ratio[split_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"{parameter_name}[{split_name!r}] must be between 0 and 1."
            )
        numeric_value = float(value)
        if not 0 < numeric_value < 1:
            raise ValueError(
                f"{parameter_name}[{split_name!r}] must be between 0 and 1."
            )
        validated_values[split_name] = numeric_value

    validation_ratio = validated_values.get("val", 0.0)
    test_ratio = validated_values["test"]
    derived_train_ratio = 1 - validation_ratio - test_ratio
    if derived_train_ratio <= 0:
        raise ValueError(
            f"{parameter_name} validation and test ratios must sum to less than 1."
        )
    if "train" in validated_values and not np.isclose(
        validated_values["train"],
        derived_train_ratio,
    ):
        raise ValueError(f"{parameter_name} ratios must sum to 1.")

    validated = {
        "train": validated_values.get("train", derived_train_ratio),
        "test": test_ratio,
    }
    if "val" in validated_values:
        validated["val"] = validation_ratio
    return validated


def absolute_split_sizes(
    population_size: int,
    ratio: Mapping[str, float],
    *,
    parameter_name: str,
    population_name: str,
) -> tuple[int, int]:
    """Calculate validation and test counts against the original population."""
    validation_size = int(np.ceil(population_size * ratio.get("val", 0.0)))
    test_size = int(np.ceil(population_size * ratio["test"]))
    if validation_size + test_size >= population_size:
        raise ValueError(
            f"{parameter_name} must leave at least one training {population_name}."
        )
    return validation_size, test_size


def partition_ordered_ids(
    ids: np.ndarray,
    ratio: Mapping[str, float],
    *,
    parameter_name: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Partition ordered IDs while preserving their original order."""
    num_validation, num_test = absolute_split_sizes(
        len(ids),
        ratio,
        parameter_name=parameter_name,
        population_name="ID",
    )
    test_ids = ids[-num_test:]
    remaining_ids = ids[:-num_test]
    if num_validation == 0:
        return remaining_ids, None, test_ids
    return (
        remaining_ids[:-num_validation],
        remaining_ids[-num_validation:],
        test_ids,
    )
