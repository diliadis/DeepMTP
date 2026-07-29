"""Interaction normalization, validation, and setting inference."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd

from DeepMTP.data.progress import (
    DataProgressEvent,
    DataProgressObserver,
    _emit_data_operation_completed,
    _emit_data_operation_started,
    _resolve_data_progress,
)

__all__ = [
    "DatasetMapping",
    "InteractionInfo",
    "NoveltyInferenceWarning",
    "ValidationSetting",
    "VariableType",
    "check_interaction_files_column_type_format",
    "check_interaction_files_format",
    "check_novel_instances",
    "check_novel_targets",
    "check_target_variable_type",
    "check_variable_type",
    "get_estimated_validation_setting",
    "process_interaction_data",
]

VariableType: TypeAlias = Literal["binary", "real-valued"]
ValidationSetting: TypeAlias = Literal["A", "B", "C", "D"]
DatasetMapping: TypeAlias = Mapping[str, Any]


class NoveltyInferenceWarning(UserWarning):
    """Warning emitted when entity identity was lost in matrix input."""


class InteractionInfo(TypedDict):
    """Normalized interaction data and its detected schema."""

    data: pd.DataFrame
    original_format: Literal["triplets", "numpy"]
    instance_id_type: Literal["int"]
    target_id_type: Literal["int"]
    missing_values: bool


def _validate_interaction_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Interaction data must contain at least one observation.")
    if frame[["instance_id", "target_id"]].isna().any().any():
        raise ValueError("Interaction IDs cannot contain missing values.")
    if not pd.api.types.is_numeric_dtype(frame["value"]):
        raise TypeError("The interaction 'value' column must be numeric.")
    if frame["value"].isna().any():
        raise ValueError(
            "The interaction 'value' column cannot contain missing values."
        )
    if frame.duplicated(subset=["instance_id", "target_id"]).any():
        raise ValueError("Interaction data contains duplicate instance-target pairs.")


def process_interaction_data(
    interaction_data: pd.DataFrame | np.ndarray,
    verbose: bool = False,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> InteractionInfo:
    """A function that processes the interaction data. It is called separately for the train, val and test interaction data.
    There are two main types of interaction data formats that are supported:
    -> numpy format: This is a 2d numpy array that represents the interaction (also called score) matrix usually found in problems settings with fully observed matrices (multi-label classification, multivariate regression)
    -> triplet format: The most flexible format as it can be used to represent every possible problem setting

    Args:
        interaction_data (_type_): a numpy array or a dataframe with the interaction data
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        progress (DataProgressObserver | None, optional): Receives verbose interaction format messages. Defaults to None.

    Returns:
        dict: a dictionary with the interaction data and additional information that could be detected (format, type of instance and target ids, etc.)
    """
    progress_observer = _resolve_data_progress(verbose, print_mode, progress)
    if isinstance(interaction_data, pd.DataFrame):
        if set(interaction_data.columns) != {
            "instance_id",
            "target_id",
            "value",
        }:
            raise ValueError(
                "Interaction file with invalid header detected. Please use the following header <instance_id, target_id, value>"
            )
        interaction_frame = interaction_data.copy()
        original_format: Literal["triplets", "numpy"] = "triplets"
    elif isinstance(interaction_data, np.ndarray):
        if interaction_data.ndim != 2:
            raise ValueError(
                "Interaction arrays must be two-dimensional score matrices."
            )
        if 0 in interaction_data.shape:
            raise ValueError("Interaction arrays cannot have an empty dimension.")
        if verbose:
            progress_observer.on_event(
                DataProgressEvent(
                    "Interaction file: 2d numpy array format detected",
                    subject="interaction_format",
                    value="numpy",
                )
            )

        # converting the 2d numpy array format to the more flexible triplet format
        interaction_frame = pd.DataFrame(interaction_data).reset_index().melt("index")
        interaction_frame.columns = ["instance_id", "target_id", "value"]
        interaction_frame["instance_id"] = interaction_frame["instance_id"].astype(int)
        interaction_frame["target_id"] = interaction_frame["target_id"].astype(int)
        # remove nans
        interaction_frame = interaction_frame.dropna(subset=["value"])
        original_format = "numpy"
    else:
        raise TypeError(
            "Could not recognize the format of the interaction data. A 2d numpy array or a dataframe(<instance_id, target_id, value>) is currently supported"
        )

    _validate_interaction_frame(interaction_frame)
    if not pd.api.types.is_integer_dtype(interaction_frame["instance_id"]):
        raise TypeError("The interaction 'instance_id' column must contain integers.")
    if not pd.api.types.is_integer_dtype(interaction_frame["target_id"]):
        raise TypeError("The interaction 'target_id' column must contain integers.")

    if verbose and original_format == "triplets":
        progress_observer.on_event(
            DataProgressEvent(
                "Interaction file: triplet format detected",
                subject="interaction_format",
                value="triplets",
            )
        )

    possible_pair_count = (
        interaction_frame["instance_id"].nunique()
        * interaction_frame["target_id"].nunique()
    )
    return {
        "data": interaction_frame,
        "original_format": original_format,
        "instance_id_type": "int",
        "target_id_type": "int",
        "missing_values": len(interaction_frame) != possible_pair_count,
    }


def check_interaction_files_format(
    data: DatasetMapping,
    verbose: bool = False,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> None:
    """Checks the format of the interaction data. If any inconsistencies are detected (like different formats between train and test interaction data), an exception is raised

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        progress (DataProgressObserver | None, optional): Receives verbose format-validation messages. Defaults to None.

    """
    progress_observer = _resolve_data_progress(verbose, print_mode, progress)
    distinct_formats = {
        data[mode]["y"]["original_format"]
        for mode in ["train", "test", "val"]
        if data[mode]["y"] is not None
    }
    if len(distinct_formats) != 0:
        if verbose:
            _emit_data_operation_started(
                progress_observer,
                subject="interaction_format_consistency",
                message="Interaction file: checking format consistency... ",
            )
        if len(distinct_formats) == 1:
            if verbose:
                _emit_data_operation_completed(
                    progress_observer,
                    subject="interaction_format_consistency",
                    message="Passed",
                    value=next(iter(distinct_formats)),
                )
        else:
            raise ValueError(
                ("error: " if print_mode == "dev" else "")
                + "Inconsistent file formats across the different interaction files"
            )


def check_interaction_files_column_type_format(
    data: DatasetMapping,
    verbose: bool = False,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> None:
    """Checks the type of the instance and target ids in the interaction data. If any inconsistencies are detected (like different id types between train and test interaction data), an exception is raised

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        progress (DataProgressObserver | None, optional): Receives verbose ID-type validation messages. Defaults to None.

    """
    progress_observer = _resolve_data_progress(verbose, print_mode, progress)
    # check for instance id type concistency across the interaction data sources
    distinct_instance_id_format = {
        data[mode]["y"]["instance_id_type"]
        for mode in ["train", "test", "val"]
        if data[mode]["y"] is not None
    }
    if len(distinct_instance_id_format) != 0:
        if verbose:
            _emit_data_operation_started(
                progress_observer,
                subject="instance_id_format_consistency",
                message=(
                    "Interaction file: checking instance id format consistency... "
                ),
            )
        if len(distinct_instance_id_format) == 1:
            if verbose:
                _emit_data_operation_completed(
                    progress_observer,
                    subject="instance_id_format_consistency",
                    message="Passed",
                    value=next(iter(distinct_instance_id_format)),
                )
        else:
            raise ValueError(
                ("error: " if print_mode == "dev" else "")
                + "Inconsistent instance id column type across the different interaction files"
            )

    # check for target id type concistency across the interaction data sources
    distinct_target_id_format = {
        data[mode]["y"]["target_id_type"]
        for mode in ["train", "test", "val"]
        if data[mode]["y"] is not None
    }
    if len(distinct_target_id_format) != 0:
        if verbose:
            _emit_data_operation_started(
                progress_observer,
                subject="target_id_format_consistency",
                message="Interaction file: checking target id type consistency... ",
            )
        if len(distinct_target_id_format) == 1:
            if verbose:
                _emit_data_operation_completed(
                    progress_observer,
                    subject="target_id_format_consistency",
                    message="Passed",
                    value=next(iter(distinct_target_id_format)),
                )
        else:
            raise ValueError(
                ("error: " if print_mode == "dev" else "")
                + "Inconsistent target id column type across the different interaction files"
            )


def check_variable_type(samples_arr: pd.DataFrame) -> VariableType:
    """Checks the type of the target variable. This implementation is currently very simple. It has to be updated!!

    Args:
        samples_arr (numpy.array): A numpy array with the target variables

    Returns:
        str: The type of the target variable. Possible values are 'real-valued' and 'binary'
    """
    if "value" not in samples_arr:
        raise ValueError("Interaction data must contain a 'value' column.")
    values = samples_arr["value"]
    if values.empty:
        raise ValueError("Cannot detect a target variable type from empty data.")
    if not pd.api.types.is_numeric_dtype(values):
        raise TypeError("Target values must be numeric.")
    if values.isna().any():
        raise ValueError("Target values cannot contain missing values.")
    return "binary" if set(values).issubset({0, 1}) else "real-valued"


def check_target_variable_type(
    data: DatasetMapping,
    verbose: bool = False,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> VariableType:
    """Checks the type of the target variable in the interaction data. If any inconsistencies are detected (like different id types between train and test interaction data), an exception is raised

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        progress (DataProgressObserver | None, optional): Receives verbose target-type validation messages. Defaults to None.

    """
    progress_observer = _resolve_data_progress(verbose, print_mode, progress)
    distinct_target_variable_type = {
        check_variable_type(data[mode]["y"]["data"])
        for mode in ["train", "test", "val"]
        if data[mode]["y"] is not None
    }
    if len(distinct_target_variable_type) != 0:
        if verbose:
            _emit_data_operation_started(
                progress_observer,
                subject="target_variable_type_consistency",
                message=(
                    "Interaction file: checking target variable type consistency... "
                ),
            )
        if len(distinct_target_variable_type) == 1:
            detected_type = next(iter(distinct_target_variable_type))
            if verbose:
                _emit_data_operation_completed(
                    progress_observer,
                    subject="target_variable_type_consistency",
                    message="Passed",
                    value=detected_type,
                )
        else:
            raise ValueError(
                ("error: " if print_mode == "dev" else "")
                + "Inconsistent target variable type across the different interaction files"
            )
    else:
        raise ValueError(
            ("error: " if print_mode == "dev" else "")
            + "Could not detect the type of target variables"
        )
    # return the detected type of target variable
    return next(iter(distinct_target_variable_type))


def _check_novel_entities(
    train: DatasetMapping | None,
    test: DatasetMapping | None,
    *,
    id_column: Literal["instance_id", "target_id"],
    entity_name: Literal["instances", "targets"],
    verbose: bool,
    progress: DataProgressObserver,
) -> bool | None:
    if test is None:
        return None
    if train is None:
        raise ValueError(
            f"Training interactions are required to detect novel {entity_name}."
        )

    train_format = train.get("original_format")
    test_format = test.get("original_format")
    supported_formats = {"triplets", "numpy"}
    if train_format not in supported_formats or test_format not in supported_formats:
        raise ValueError(
            "Interaction sources must declare original_format as 'triplets' or 'numpy'."
        )
    if train_format != test_format:
        raise ValueError(
            "Training and test interaction formats must match for novelty detection."
        )

    train_frame = train.get("data")
    test_frame = test.get("data")
    if not isinstance(train_frame, pd.DataFrame) or not isinstance(
        test_frame, pd.DataFrame
    ):
        raise TypeError("Processed interaction data must be pandas DataFrames.")
    if id_column not in train_frame or id_column not in test_frame:
        raise ValueError(
            f"Processed interaction data must contain an {id_column!r} column."
        )

    if verbose:
        _emit_data_operation_started(
            progress,
            subject=f"novel_{entity_name}",
            message=f"Interaction file: Checking for novel {entity_name}... ",
        )

    if train_format == "numpy":
        warnings.warn(
            f"Cannot determine whether NumPy score matrices contain novel "
            f"{entity_name} because positional axes do not preserve entity "
            "identity; returning None. Use triplet interaction data or provide "
            "validation_setting explicitly.",
            NoveltyInferenceWarning,
            stacklevel=2,
        )
        novelty_detected = None
    else:
        train_ids = set(train_frame[id_column])
        test_ids = set(test_frame[id_column])
        novelty_detected = bool(test_ids.difference(train_ids))

    if verbose:
        _emit_data_operation_completed(
            progress,
            subject=f"novel_{entity_name}",
            message="Done",
            value=novelty_detected,
        )
    return novelty_detected


def check_novel_instances(
    train: DatasetMapping | None,
    test: DatasetMapping | None,
    verbose: bool = False,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> bool | None:
    """Return whether test interactions contain previously unseen instances.

    ``progress`` receives messages when ``verbose`` is enabled.
    """
    return _check_novel_entities(
        train,
        test,
        id_column="instance_id",
        entity_name="instances",
        verbose=verbose,
        progress=_resolve_data_progress(verbose, print_mode, progress),
    )


def check_novel_targets(
    train: DatasetMapping | None,
    test: DatasetMapping | None,
    verbose: bool = False,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> bool | None:
    """Return whether test interactions contain previously unseen targets.

    ``progress`` receives messages when ``verbose`` is enabled.
    """
    return _check_novel_entities(
        train,
        test,
        id_column="target_id",
        entity_name="targets",
        verbose=verbose,
        progress=_resolve_data_progress(verbose, print_mode, progress),
    )


def get_estimated_validation_setting(
    novel_instances_flag: bool | None,
    novel_targets_flag: bool | None,
    verbose: bool,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> ValidationSetting | None:
    """Uses the combination of infrmation about novel instances and novel targets to determine the validation setting that is possible.

    Args:
        novel_instances_flag (bool): A boolean that indicates whether or not the test set contains novel instances.
        novel_targets_flag (bool): A boolean that indicates whether or not the test set contains novel targets.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        progress (DataProgressObserver | None, optional): Receives verbose validation-setting inference messages. Defaults to None.

    Returns:
        str: The validation setting. Possible values are A, B, C, D
    """
    for flag_name, flag in (
        ("novel_instances_flag", novel_instances_flag),
        ("novel_targets_flag", novel_targets_flag),
    ):
        if flag is not None and not isinstance(flag, bool):
            raise TypeError(f"{flag_name} must be a boolean or None.")

    validation_setting_detected: ValidationSetting | None = None
    if novel_instances_flag is not None and novel_targets_flag is not None:
        progress_observer = _resolve_data_progress(verbose, print_mode, progress)
        if verbose:
            _emit_data_operation_started(
                progress_observer,
                subject="validation_setting",
                message="Estimating validation setting... ",
            )
        if novel_instances_flag and not novel_targets_flag:
            validation_setting_detected = "B"
        elif not novel_instances_flag and novel_targets_flag:
            validation_setting_detected = "C"
        elif novel_instances_flag and novel_targets_flag:
            validation_setting_detected = "D"
        else:  # novel_instances_flag and not novel_targets_flag:
            validation_setting_detected = "A"
        if verbose:
            _emit_data_operation_completed(
                progress_observer,
                subject="validation_setting",
                message="Done",
                value=validation_setting_detected,
                end="",
            )
            progress_observer.on_event(
                DataProgressEvent(
                    "-- Detected as setting :" + validation_setting_detected,
                    subject="validation_setting",
                    value=validation_setting_detected,
                )
            )
    return validation_setting_detected
