"""Dataset splitting, defensive copying, and result validation."""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from DeepMTP._split_utils import absolute_split_sizes as _absolute_split_sizes
from DeepMTP._split_utils import validate_split_ratio
from DeepMTP.data.progress import (
    DataProgressObserver,
    _emit_data_operation_completed_if_verbose,
    _emit_data_operation_started_if_verbose,
)
from DeepMTP.interaction_data import DatasetMapping, ValidationSetting

__all__ = ["split_data"]


def _split_frame_rows(
    frame: pd.DataFrame,
    test_size: int | float,
    *,
    shuffle_rows: bool,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame, held_out_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed if shuffle_rows else None,
        shuffle=shuffle_rows,
    )
    return train_frame.copy(), held_out_frame.copy()


def _interaction_axis_size(
    data: Mapping[str, Any],
    id_column: Literal["instance_id", "target_id"],
) -> int:
    """Count unique IDs across every supplied interaction split."""
    identifiers: set[Any] = set()
    for mode in ("train", "val", "test"):
        source = data[mode]["y"]
        if source is not None:
            identifiers.update(source["data"][id_column].unique())
    return len(identifiers)


def _interaction_row_count(data: Mapping[str, Any]) -> int:
    """Count interaction rows across every supplied split."""
    return sum(
        len(source["data"])
        for mode in ("train", "val", "test")
        if (source := data[mode]["y"]) is not None
    )


def _copy_feature_source(source: Mapping[str, Any]) -> dict[str, Any]:
    copied_source = copy.deepcopy(dict(source))
    source_frame = source.get("data")
    if isinstance(source_frame, pd.DataFrame):
        copied_frame = source_frame.copy(deep=True)
        if "features" in copied_frame:
            copied_frame["features"] = [
                np.asarray(feature_vector).copy()
                for feature_vector in copied_frame["features"]
            ]
        copied_source["data"] = copied_frame
    return copied_source


def _detach_shared_feature_sources(data: MutableMapping[str, Any]) -> None:
    for feature_key in ("X_instance", "X_target"):
        seen_source_ids: set[int] = set()
        seen_frame_ids: set[int] = set()
        for mode in ("train", "val", "test"):
            source = data[mode][feature_key]
            if source is None:
                continue
            source_id = id(source)
            frame_id = id(source.get("data"))
            if source_id in seen_source_ids or frame_id in seen_frame_ids:
                source = _copy_feature_source(source)
                data[mode][feature_key] = source
                source_id = id(source)
                frame_id = id(source.get("data"))
            seen_source_ids.add(source_id)
            seen_frame_ids.add(frame_id)


def _interaction_pairs(frame: pd.DataFrame) -> set[tuple[Any, Any]]:
    return set(
        frame[["instance_id", "target_id"]].itertuples(
            index=False,
            name=None,
        )
    )


def _validate_split_result(
    data: DatasetMapping,
    validation_setting: ValidationSetting,
) -> None:
    split_frames: dict[str, pd.DataFrame] = {}
    for mode in ("train", "val", "test"):
        interaction_source = data[mode]["y"]
        if interaction_source is None:
            raise ValueError(f"Splitting did not produce {mode} interaction data.")
        frame = interaction_source["data"]
        if frame.empty:
            raise ValueError(f"Splitting produced an empty {mode} interaction set.")
        split_frames[mode] = frame

    pair_sets = {
        mode: _interaction_pairs(frame) for mode, frame in split_frames.items()
    }
    for left_mode, right_mode in combinations(pair_sets, 2):
        if pair_sets[left_mode].intersection(pair_sets[right_mode]):
            raise ValueError(f"{left_mode} and {right_mode} interaction sets overlap.")

    disjoint_columns: tuple[str, ...]
    if validation_setting == "B":
        disjoint_columns = ("instance_id",)
    elif validation_setting == "C":
        disjoint_columns = ("target_id",)
    elif validation_setting == "D":
        disjoint_columns = ("instance_id", "target_id")
    else:
        disjoint_columns = ()

    for column in disjoint_columns:
        id_sets = {mode: set(frame[column]) for mode, frame in split_frames.items()}
        for left_mode, right_mode in combinations(id_sets, 2):
            if id_sets[left_mode].intersection(id_sets[right_mode]):
                raise ValueError(
                    f"{column} values overlap between {left_mode} and "
                    f"{right_mode} for validation setting {validation_setting}."
                )


def _emit_split_started(
    verbose: bool,
    print_mode: str,
    progress: DataProgressObserver | None,
    *,
    destination: Literal["test", "val"],
    validation_setting: ValidationSetting,
) -> None:
    subject = "train_test_split" if destination == "test" else "train_validation_split"
    _emit_data_operation_started_if_verbose(
        verbose,
        print_mode,
        progress,
        subject=subject,
        message=(
            f"Splitting train to train-{destination} according to validation "
            f"setting {validation_setting}... "
        ),
    )


def _emit_split_completed(
    verbose: bool,
    print_mode: str,
    progress: DataProgressObserver | None,
    *,
    destination: Literal["test", "val"],
    validation_setting: ValidationSetting,
) -> None:
    subject = "train_test_split" if destination == "test" else "train_validation_split"
    _emit_data_operation_completed_if_verbose(
        verbose,
        print_mode,
        progress,
        subject=subject,
        message="Done",
        value=validation_setting,
    )


def split_data(
    data: MutableMapping[str, Any],
    validation_setting: ValidationSetting,
    split_method: str,
    ratio: object,
    shuffle: bool,
    seed: int | None,
    verbose: bool,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> None:
    """Splits the dataset and offers two main functionalities:
        * split based on the 4 different validation settings (A, B, C, D)
        * if a test set already exists it separates a validation set, otherwise it first creates a test set and then a validation set.

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        validation_setting (str): The validation setting of the current problem.
        split_method (str): The splitting method used. The current implementation only supports the 'random split' using a specific seed but a future goal is to also offer a stratified option.
        ratio (dict): The train, val and test ratios used to split the data.
        seed (int): The seed used to initiate the randomized split
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        progress (DataProgressObserver | None, optional): Receives verbose split lifecycle messages. Defaults to None.

    """
    if validation_setting not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unknown validation setting: {validation_setting!r}.")
    if split_method != "random":
        raise ValueError(
            f"Unsupported split method {split_method!r}; only 'random' is available."
        )
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean.")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an integer or None.")

    ratio = validate_split_ratio(ratio)
    validation_size = test_size = 0
    validation_instance_size = test_instance_size = 0
    validation_target_size = test_target_size = 0
    split_is_missing = data["test"]["y"] is None or data["val"]["y"] is None
    if validation_setting == "A" and split_is_missing:
        validation_size, test_size = _absolute_split_sizes(
            _interaction_row_count(data),
            ratio,
            parameter_name="The requested ratios",
            population_name="interaction",
        )
    elif validation_setting == "B" and split_is_missing:
        validation_size, test_size = _absolute_split_sizes(
            _interaction_axis_size(data, "instance_id"),
            ratio,
            parameter_name="The requested ratios",
            population_name="instance",
        )
    elif validation_setting == "C" and split_is_missing:
        validation_size, test_size = _absolute_split_sizes(
            _interaction_axis_size(data, "target_id"),
            ratio,
            parameter_name="The requested ratios",
            population_name="target",
        )
    elif validation_setting == "D" and split_is_missing:
        validation_instance_size, test_instance_size = _absolute_split_sizes(
            _interaction_axis_size(data, "instance_id"),
            ratio,
            parameter_name="The requested ratios",
            population_name="instance",
        )
        validation_target_size, test_target_size = _absolute_split_sizes(
            _interaction_axis_size(data, "target_id"),
            ratio,
            parameter_name="The requested ratios",
            population_name="target",
        )

    if validation_setting == "B":
        if data["test"]["y"] is None:
            _emit_split_started(
                verbose,
                print_mode,
                progress,
                destination="test",
                validation_setting=validation_setting,
            )
            train_ids, test_ids = train_test_split(
                data["train"]["y"]["data"]["instance_id"].unique(),
                test_size=test_size,
                random_state=seed if shuffle else None,
                shuffle=shuffle,
            )
            train_ids.sort()
            test_ids.sort()
            data["test"]["y"] = {
                "data": data["train"]["y"]["data"].loc[
                    data["train"]["y"]["data"]["instance_id"].isin(test_ids)
                ]
            }
            data["train"]["y"]["data"] = data["train"]["y"]["data"].loc[
                data["train"]["y"]["data"]["instance_id"].isin(train_ids)
            ]
            data["test"]["X_instance"] = {
                "data": data["train"]["X_instance"]["data"].loc[
                    data["train"]["X_instance"]["data"]["id"].isin(test_ids)
                ]
            }
            data["train"]["X_instance"]["data"] = data["train"]["X_instance"][
                "data"
            ].loc[data["train"]["X_instance"]["data"]["id"].isin(train_ids)]
            _emit_split_completed(
                verbose,
                print_mode,
                progress,
                destination="test",
                validation_setting=validation_setting,
            )

        if data["val"]["y"] is None:
            _emit_split_started(
                verbose,
                print_mode,
                progress,
                destination="val",
                validation_setting=validation_setting,
            )
            train_ids, val_ids = train_test_split(
                data["train"]["y"]["data"]["instance_id"].unique(),
                test_size=validation_size,
                random_state=seed if shuffle else None,
                shuffle=shuffle,
            )
            train_ids.sort()
            val_ids.sort()
            data["val"]["y"] = {
                "data": data["train"]["y"]["data"].loc[
                    data["train"]["y"]["data"]["instance_id"].isin(val_ids)
                ]
            }
            data["train"]["y"]["data"] = data["train"]["y"]["data"].loc[
                data["train"]["y"]["data"]["instance_id"].isin(train_ids)
            ]
            instance_data = data["train"]["X_instance"]["data"]
            data["val"]["X_instance"] = {
                "data": instance_data.loc[instance_data["id"].isin(val_ids)]
            }
            if data["test"]["X_instance"] is None:
                test_instance_data = instance_data.loc[
                    instance_data["id"].isin(
                        data["test"]["y"]["data"]["instance_id"].unique()
                    )
                ]
                data["test"]["X_instance"] = {"data": test_instance_data}
            data["train"]["X_instance"]["data"] = instance_data.loc[
                instance_data["id"].isin(train_ids)
            ]
            _emit_split_completed(
                verbose,
                print_mode,
                progress,
                destination="val",
                validation_setting=validation_setting,
            )

        # if data['train']['X_instance'] is not None:
        if data["test"]["X_instance"] is None:
            test_ids = data["test"]["y"]["data"]["instance_id"].unique()
            data["test"]["X_instance"] = {
                "data": data["train"]["X_instance"]["data"].loc[
                    data["train"]["X_instance"]["data"]["id"].isin(test_ids)
                ]
            }
        if data["val"]["X_instance"] is None:
            val_ids = data["val"]["y"]["data"]["instance_id"].unique()
            data["val"]["X_instance"] = {
                "data": data["train"]["X_instance"]["data"].loc[
                    data["train"]["X_instance"]["data"]["id"].isin(val_ids)
                ]
            }
        train_ids = data["train"]["y"]["data"]["instance_id"].unique()
        data["train"]["X_instance"] = {
            "data": data["train"]["X_instance"]["data"].loc[
                data["train"]["X_instance"]["data"]["id"].isin(train_ids)
            ]
        }

        if data["train"]["X_target"] is not None:
            if data["test"]["X_target"] is None:
                data["test"]["X_target"] = data["train"]["X_target"]
            if data["val"]["X_target"] is None:
                data["val"]["X_target"] = data["train"]["X_target"]

    elif validation_setting == "C":
        if data["test"]["y"] is None:
            _emit_split_started(
                verbose,
                print_mode,
                progress,
                destination="test",
                validation_setting=validation_setting,
            )
            train_ids, test_ids = train_test_split(
                data["train"]["y"]["data"]["target_id"].unique(),
                test_size=test_size,
                random_state=seed if shuffle else None,
                shuffle=shuffle,
            )
            train_ids.sort()
            test_ids.sort()
            data["test"]["y"] = {
                "data": data["train"]["y"]["data"].loc[
                    data["train"]["y"]["data"]["target_id"].isin(test_ids)
                ]
            }
            data["train"]["y"]["data"] = data["train"]["y"]["data"].loc[
                data["train"]["y"]["data"]["target_id"].isin(train_ids)
            ]
            data["test"]["X_target"] = {
                "data": data["train"]["X_target"]["data"].loc[
                    data["train"]["X_target"]["data"]["id"].isin(test_ids)
                ]
            }
            data["train"]["X_target"]["data"] = data["train"]["X_target"]["data"].loc[
                data["train"]["X_target"]["data"]["id"].isin(train_ids)
            ]
            _emit_split_completed(
                verbose,
                print_mode,
                progress,
                destination="test",
                validation_setting=validation_setting,
            )

        if data["val"]["y"] is None:
            _emit_split_started(
                verbose,
                print_mode,
                progress,
                destination="val",
                validation_setting=validation_setting,
            )
            train_ids, val_ids = train_test_split(
                data["train"]["y"]["data"]["target_id"].unique(),
                test_size=validation_size,
                random_state=seed if shuffle else None,
                shuffle=shuffle,
            )
            train_ids.sort()
            val_ids.sort()
            data["val"]["y"] = {
                "data": data["train"]["y"]["data"].loc[
                    data["train"]["y"]["data"]["target_id"].isin(val_ids)
                ]
            }
            data["train"]["y"]["data"] = data["train"]["y"]["data"].loc[
                data["train"]["y"]["data"]["target_id"].isin(train_ids)
            ]
            target_data = data["train"]["X_target"]["data"]
            data["val"]["X_target"] = {
                "data": target_data.loc[target_data["id"].isin(val_ids)]
            }
            if data["test"]["X_target"] is None:
                test_target_data = target_data.loc[
                    target_data["id"].isin(
                        data["test"]["y"]["data"]["target_id"].unique()
                    )
                ]
                data["test"]["X_target"] = {"data": test_target_data}
            data["train"]["X_target"]["data"] = target_data.loc[
                target_data["id"].isin(train_ids)
            ]
            _emit_split_completed(
                verbose,
                print_mode,
                progress,
                destination="val",
                validation_setting=validation_setting,
            )

        # if data['train']['X_target'] is not None:
        if data["test"]["X_target"] is None:
            test_ids = data["test"]["y"]["data"]["target_id"].unique()
            data["test"]["X_target"] = {
                "data": data["train"]["X_target"]["data"].loc[
                    data["train"]["X_target"]["data"]["id"].isin(test_ids)
                ]
            }
        if data["val"]["X_target"] is None:
            val_ids = data["val"]["y"]["data"]["target_id"].unique()
            data["val"]["X_target"] = {
                "data": data["train"]["X_target"]["data"].loc[
                    data["train"]["X_target"]["data"]["id"].isin(val_ids)
                ]
            }
        train_ids = data["train"]["y"]["data"]["target_id"].unique()
        data["train"]["X_target"] = {
            "data": data["train"]["X_target"]["data"].loc[
                data["train"]["X_target"]["data"]["id"].isin(train_ids)
            ]
        }

        if data["train"]["X_instance"] is not None:
            if data["test"]["X_instance"] is None:
                data["test"]["X_instance"] = data["train"]["X_instance"]
            if data["val"]["X_instance"] is None:
                data["val"]["X_instance"] = data["train"]["X_instance"]

    elif validation_setting == "A":
        if data["test"]["y"] is None:
            _emit_split_started(
                verbose,
                print_mode,
                progress,
                destination="test",
                validation_setting=validation_setting,
            )
            train_y_data, test_y_data = _split_frame_rows(
                data["train"]["y"]["data"],
                test_size,
                shuffle_rows=shuffle,
                seed=seed,
            )
            data["test"]["y"] = {"data": test_y_data}
            data["train"]["y"]["data"] = train_y_data
            if data["train"]["X_instance"] is not None:
                data["test"]["X_instance"] = data["train"]["X_instance"]
            if data["train"]["X_target"] is not None:
                data["test"]["X_target"] = data["train"]["X_target"]
            _emit_split_completed(
                verbose,
                print_mode,
                progress,
                destination="test",
                validation_setting=validation_setting,
            )

        if data["val"]["y"] is None:
            _emit_split_started(
                verbose,
                print_mode,
                progress,
                destination="val",
                validation_setting=validation_setting,
            )
            train_y_data, val_y_data = _split_frame_rows(
                data["train"]["y"]["data"],
                validation_size,
                shuffle_rows=shuffle,
                seed=seed,
            )
            data["val"]["y"] = {"data": val_y_data}
            data["train"]["y"]["data"] = train_y_data
            if data["train"]["X_instance"] is not None:
                if data["test"]["X_instance"] is None:
                    data["test"]["X_instance"] = data["train"]["X_instance"]
                if data["val"]["X_instance"] is None:
                    data["val"]["X_instance"] = data["train"]["X_instance"]
            if data["train"]["X_target"] is not None:
                if data["test"]["X_target"] is None:
                    data["test"]["X_target"] = data["train"]["X_target"]
                if data["val"]["X_target"] is None:
                    data["val"]["X_target"] = data["train"]["X_target"]
            _emit_split_completed(
                verbose,
                print_mode,
                progress,
                destination="val",
                validation_setting=validation_setting,
            )

    elif validation_setting == "D":
        if data["test"]["y"] is None:
            _emit_split_started(
                verbose,
                print_mode,
                progress,
                destination="test",
                validation_setting=validation_setting,
            )
            train_instance_ids, test_instance_ids = train_test_split(
                data["train"]["y"]["data"]["instance_id"].unique(),
                test_size=test_instance_size,
                random_state=seed if shuffle else None,
                shuffle=shuffle,
            )
            train_instance_ids.sort()
            test_instance_ids.sort()
            train_target_ids, test_target_ids = train_test_split(
                data["train"]["y"]["data"]["target_id"].unique(),
                test_size=test_target_size,
                random_state=seed if shuffle else None,
                shuffle=shuffle,
            )
            train_target_ids.sort()
            test_target_ids.sort()
            data["test"]["y"] = {
                "data": data["train"]["y"]["data"].loc[
                    (data["train"]["y"]["data"]["instance_id"].isin(test_instance_ids))
                    & (data["train"]["y"]["data"]["target_id"].isin(test_target_ids))
                ]
            }
            data["train"]["y"]["data"] = data["train"]["y"]["data"].loc[
                (data["train"]["y"]["data"]["instance_id"].isin(train_instance_ids))
                & (data["train"]["y"]["data"]["target_id"].isin(train_target_ids))
            ]
            data["test"]["X_instance"] = {
                "data": data["train"]["X_instance"]["data"].loc[
                    data["train"]["X_instance"]["data"]["id"].isin(test_instance_ids)
                ]
            }
            data["train"]["X_instance"]["data"] = data["train"]["X_instance"][
                "data"
            ].loc[data["train"]["X_instance"]["data"]["id"].isin(train_instance_ids)]
            data["test"]["X_target"] = {
                "data": data["train"]["X_target"]["data"].loc[
                    data["train"]["X_target"]["data"]["id"].isin(test_target_ids)
                ]
            }
            data["train"]["X_target"]["data"] = data["train"]["X_target"]["data"].loc[
                data["train"]["X_target"]["data"]["id"].isin(train_target_ids)
            ]

            _emit_split_completed(
                verbose,
                print_mode,
                progress,
                destination="test",
                validation_setting=validation_setting,
            )

        if data["val"]["y"] is None:
            _emit_split_started(
                verbose,
                print_mode,
                progress,
                destination="val",
                validation_setting=validation_setting,
            )
            train_instance_ids, val_instance_ids = train_test_split(
                data["train"]["y"]["data"]["instance_id"].unique(),
                test_size=validation_instance_size,
                random_state=seed if shuffle else None,
                shuffle=shuffle,
            )
            train_instance_ids.sort()
            val_instance_ids.sort()
            train_target_ids, val_target_ids = train_test_split(
                data["train"]["y"]["data"]["target_id"].unique(),
                test_size=validation_target_size,
                random_state=seed if shuffle else None,
                shuffle=shuffle,
            )
            train_target_ids.sort()
            val_target_ids.sort()
            data["val"]["y"] = {
                "data": data["train"]["y"]["data"].loc[
                    (data["train"]["y"]["data"]["instance_id"].isin(val_instance_ids))
                    & (data["train"]["y"]["data"]["target_id"].isin(val_target_ids))
                ]
            }
            # data['val']['y'].update({k:v for k,v in data['train']['y'].items() if k!=data})
            data["train"]["y"]["data"] = data["train"]["y"]["data"].loc[
                (data["train"]["y"]["data"]["instance_id"].isin(train_instance_ids))
                & (data["train"]["y"]["data"]["target_id"].isin(train_target_ids))
            ]

            instance_data = data["train"]["X_instance"]["data"]
            data["val"]["X_instance"] = {
                "data": instance_data.loc[instance_data["id"].isin(val_instance_ids)]
            }
            if data["test"]["X_instance"] is None:
                test_instance_data = instance_data.loc[
                    instance_data["id"].isin(
                        data["test"]["y"]["data"]["instance_id"].unique()
                    )
                ]
                data["test"]["X_instance"] = {"data": test_instance_data}
            data["train"]["X_instance"]["data"] = instance_data.loc[
                instance_data["id"].isin(train_instance_ids)
            ]

            target_data = data["train"]["X_target"]["data"]
            data["val"]["X_target"] = {
                "data": target_data.loc[target_data["id"].isin(val_target_ids)]
            }
            if data["test"]["X_target"] is None:
                test_target_data = target_data.loc[
                    target_data["id"].isin(
                        data["test"]["y"]["data"]["target_id"].unique()
                    )
                ]
                data["test"]["X_target"] = {"data": test_target_data}
            data["train"]["X_target"]["data"] = target_data.loc[
                target_data["id"].isin(train_target_ids)
            ]

        if data["test"]["X_instance"] is None or data["val"]["X_instance"] is None:
            if data["test"]["X_instance"] is None:
                test_ids = data["test"]["y"]["data"]["instance_id"].unique()
                data["test"]["X_instance"] = {
                    "data": data["train"]["X_instance"]["data"].loc[
                        data["train"]["X_instance"]["data"]["id"].isin(test_ids)
                    ]
                }
            if data["val"]["X_instance"] is None:
                val_ids = data["val"]["y"]["data"]["instance_id"].unique()
                data["val"]["X_instance"] = {
                    "data": data["train"]["X_instance"]["data"].loc[
                        data["train"]["X_instance"]["data"]["id"].isin(val_ids)
                    ]
                }
            train_ids = data["train"]["y"]["data"]["instance_id"].unique()
            data["train"]["X_instance"] = {
                "data": data["train"]["X_instance"]["data"].loc[
                    data["train"]["X_instance"]["data"]["id"].isin(train_ids)
                ]
            }

        if data["test"]["X_target"] is None or data["val"]["X_target"] is None:
            if data["test"]["X_target"] is None:
                test_ids = data["test"]["y"]["data"]["target_id"].unique()
                data["test"]["X_target"] = {
                    "data": data["train"]["X_target"]["data"].loc[
                        data["train"]["X_target"]["data"]["id"].isin(test_ids)
                    ]
                }
            if data["val"]["X_target"] is None:
                val_ids = data["val"]["y"]["data"]["target_id"].unique()
                data["val"]["X_target"] = {
                    "data": data["train"]["X_target"]["data"].loc[
                        data["train"]["X_target"]["data"]["id"].isin(val_ids)
                    ]
                }
            train_ids = data["train"]["y"]["data"]["target_id"].unique()
            data["train"]["X_target"] = {
                "data": data["train"]["X_target"]["data"].loc[
                    data["train"]["X_target"]["data"]["id"].isin(train_ids)
                ]
            }

        _emit_split_completed(
            verbose,
            print_mode,
            progress,
            destination="val",
            validation_setting=validation_setting,
        )

    _detach_shared_feature_sources(data)
    _validate_split_result(data, validation_setting)
