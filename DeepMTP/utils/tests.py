"""Validation helpers for checking generated dataset splits.

The historical module name is retained for compatibility. These functions are
runtime validators, not pytest tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any, TypeAlias

from DeepMTP.data.progress import (
    DataProgressEvent,
    DataProgressObserver,
    build_data_progress_observer,
)

DatasetSplit: TypeAlias = Mapping[str, Any]


class SplitValidationError(ValueError):
    """Raised when generated train, validation, and test splits are inconsistent."""


def _resolve_progress(
    progress: DataProgressObserver | None,
) -> DataProgressObserver:
    return progress if progress is not None else build_data_progress_observer(True)


def _emit_validation_started(
    label: str,
    progress: DataProgressObserver,
) -> None:
    progress.on_event(
        DataProgressEvent(
            f"Checking if {label} splitting results are valid... ",
            kind="operation_started",
            subject=label,
            end="",
        )
    )


def _emit_validation_completed(
    label: str,
    progress: DataProgressObserver,
) -> None:
    progress.on_event(
        DataProgressEvent(
            "Done",
            kind="operation_completed",
            subject=label,
        )
    )


def _feature_ids(split: DatasetSplit, feature_name: str) -> set[Any]:
    return set(split[feature_name]["data"]["id"].unique())


def _interaction_ids(split: DatasetSplit, id_column: str) -> set[Any]:
    return set(split["y"]["data"][id_column].unique())


def _interaction_pairs(split: DatasetSplit) -> set[tuple[Any, Any]]:
    interactions = split["y"]["data"]
    return set(
        interactions[["instance_id", "target_id"]].itertuples(
            index=False,
            name=None,
        )
    )


def _ensure_feature_ids_match(
    split_name: str,
    split: DatasetSplit,
    feature_name: str,
    interaction_column: str,
) -> set[Any]:
    feature_ids = _feature_ids(split, feature_name)
    interaction_ids = _interaction_ids(split, interaction_column)
    if feature_ids != interaction_ids:
        missing_features = interaction_ids.difference(feature_ids)
        unused_features = feature_ids.difference(interaction_ids)
        raise SplitValidationError(
            f"{split_name} {feature_name} IDs do not match interaction "
            f"{interaction_column} IDs; missing features={missing_features!r}, "
            f"unused features={unused_features!r}."
        )
    return feature_ids


def _ensure_pairwise_disjoint(
    named_values: Mapping[str, set[Any]],
    value_description: str,
) -> None:
    for (left_name, left_values), (right_name, right_values) in combinations(
        named_values.items(),
        2,
    ):
        overlap = left_values.intersection(right_values)
        if overlap:
            raise SplitValidationError(
                f"{value_description} overlap between {left_name} and "
                f"{right_name}: {overlap!r}."
            )


def _check_instance_partition(
    label: str,
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
    progress: DataProgressObserver,
) -> None:
    _emit_validation_started(label, progress)
    instance_ids = {
        split_name: _ensure_feature_ids_match(
            split_name,
            split,
            "X_instance",
            "instance_id",
        )
        for split_name, split in (
            ("train", train),
            ("validation", val),
            ("test", test),
        )
    }
    _ensure_pairwise_disjoint(instance_ids, "Instance IDs")
    _emit_validation_completed(label, progress)


def check_mlc_results(
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
    *,
    progress: DataProgressObserver | None = None,
) -> None:
    """Validate a multi-label classification split."""
    _check_instance_partition(
        "MLC",
        train,
        val,
        test,
        _resolve_progress(progress),
    )


def check_mtr_results(
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
    *,
    progress: DataProgressObserver | None = None,
) -> None:
    """Validate a multi-target regression split."""
    _check_instance_partition(
        "MTR",
        train,
        val,
        test,
        _resolve_progress(progress),
    )


def check_dp_results(
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
    *,
    progress: DataProgressObserver | None = None,
) -> None:
    """Validate feature coverage for a dyadic-prediction split."""
    progress_observer = _resolve_progress(progress)
    _emit_validation_started("DP", progress_observer)
    for split_name, split in (
        ("train", train),
        ("validation", val),
        ("test", test),
    ):
        _ensure_feature_ids_match(
            split_name,
            split,
            "X_instance",
            "instance_id",
        )
        _ensure_feature_ids_match(
            split_name,
            split,
            "X_target",
            "target_id",
        )
    _emit_validation_completed("DP", progress_observer)


def check_mc_results(
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
    *,
    progress: DataProgressObserver | None = None,
) -> None:
    """Validate that matrix-completion splits do not share interaction pairs."""
    progress_observer = _resolve_progress(progress)
    _emit_validation_started("MC", progress_observer)
    split_pairs = {
        split_name: _interaction_pairs(split)
        for split_name, split in (
            ("train", train),
            ("validation", val),
            ("test", test),
        )
    }
    _ensure_pairwise_disjoint(split_pairs, "Instance-target pairs")
    _emit_validation_completed("MC", progress_observer)


def check_mtl_results(
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
    *,
    progress: DataProgressObserver | None = None,
) -> None:
    """Validate a multi-task learning split."""
    _check_instance_partition(
        "MTL",
        train,
        val,
        test,
        _resolve_progress(progress),
    )
