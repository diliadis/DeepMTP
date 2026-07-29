"""End-to-end data preparation and feature scaling orchestration."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping, MutableMapping
from typing import Any, Literal, Protocol, TypeAlias, TypedDict, cast

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from DeepMTP._split_utils import validate_split_ratio as _validate_split_ratio
from DeepMTP.data.features import (
    cross_input_consistency_check_instances,
    cross_input_consistency_check_targets,
    process_instance_features,
    process_target_features,
)
from DeepMTP.data.interactions import (
    ValidationSetting,
    check_interaction_files_column_type_format,
    check_interaction_files_format,
    check_novel_instances,
    check_novel_targets,
    check_target_variable_type,
    get_estimated_validation_setting,
    process_interaction_data,
)
from DeepMTP.data.progress import (
    DataProgressObserver,
    _emit_data_message,
    _emit_data_operation_completed_if_verbose,
    _emit_data_operation_started_if_verbose,
    _emit_data_separator,
    _resolve_data_progress,
)
from DeepMTP.data.splitting import split_data

__all__ = [
    "DataInfo",
    "ProcessedSplit",
    "ScalingMethod",
    "Transformer",
    "data_process",
    "normalize",
    "validate_split_ratio",
]

ScalingMethod: TypeAlias = Literal["MinMax", "Standard"]
ProcessedSplit: TypeAlias = dict[str, Any]


class DataInfo(TypedDict):
    """Metadata inferred while preparing a multi-target dataset."""

    detected_validation_setting: ValidationSetting
    detected_problem_mode: Literal["classification", "regression"]
    instance_branch_input_dim: int | None
    target_branch_input_dim: int | None


class Transformer(Protocol):
    """Minimal interface implemented by supported feature scalers."""

    def transform(self, values: Any) -> Any:
        """Transform values using a previously fitted scaler."""


def validate_split_ratio(ratio: object) -> dict[str, float]:
    """Validate and copy a train/validation/test split ratio mapping."""
    return _validate_split_ratio(ratio)


def normalize(row: Any, scaler: Transformer) -> Any:  # pragma: no cover
    """Just normalizes a row or features

    Args:
        row (numpy.array): an array of features
        scaler (sklearn.scaler): the scaler that will be used to scale the features

    Returns:
        numpy.array: a scaled array of feature
    """
    return scaler.transform([row])[0]


def _prepare_dataset_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping.")
    if "train" not in data:
        raise AttributeError(
            "The train key-value pair should always be present in the custom data "
            "object"
        )

    prepared_data = copy.deepcopy(dict(data))
    for mode in ("train", "val", "test"):
        source = prepared_data.get(mode, {})
        if not isinstance(source, Mapping):
            raise TypeError(f"data[{mode!r}] must be a mapping.")
        prepared_source = dict(source)
        for source_key in ("y", "X_instance", "X_target"):
            prepared_source.setdefault(source_key, None)
        prepared_data[mode] = prepared_source
    return prepared_data


def _normalize_validation_setting(
    validation_setting: str | None,
) -> ValidationSetting | None:
    if validation_setting is None:
        return None
    if not isinstance(validation_setting, str):
        raise TypeError("validation_setting must be a string or None.")

    normalized_setting = validation_setting.upper()
    if normalized_setting == "A":
        return "A"
    if normalized_setting == "B":
        return "B"
    if normalized_setting == "C":
        return "C"
    if normalized_setting == "D":
        return "D"
    raise ValueError(f"Unknown validation setting: {validation_setting!r}.")


def _normalize_scaling_method(
    scaling_method: str | None,
    parameter_name: str,
) -> ScalingMethod | None:
    if scaling_method is None:
        return None
    if scaling_method == "MinMax":
        return "MinMax"
    if scaling_method == "Standard":
        return "Standard"
    raise ValueError(
        f"{parameter_name} must be 'MinMax', 'Standard', or None; "
        f"received {scaling_method!r}."
    )


def _scale_feature_splits(
    data: MutableMapping[str, Any],
    feature_key: Literal["X_instance", "X_target"],
    entity_name: Literal["instance", "target"],
    scaling_method: ScalingMethod | None,
    *,
    verbose: bool,
    print_mode: str,
    progress: DataProgressObserver | None = None,
) -> None:
    if scaling_method is None:
        return

    train_source = data["train"][feature_key]
    if train_source is None:
        _emit_data_message(
            verbose,
            print_mode,
            progress,
            subject=f"{entity_name}_feature_scaling",
            message=(
                f"-- Requested {scaling_method} scaling for the {entity_name} "
                f"features without supplying any {entity_name} features. This "
                "step will be skipped"
            ),
            value="missing_features",
            level="warning",
        )
        return

    train_frame = train_source["data"]
    if "features" not in train_frame.columns:
        if "dir" not in train_frame.columns:
            raise ValueError(
                f"Legacy {entity_name} feature scaling cannot be applied to an "
                "explicit tabular schema. Configure numeric_normalization in "
                f"the {entity_name} TABULAR branch schema instead."
            )
        _emit_data_message(
            verbose,
            print_mode,
            progress,
            subject=f"{entity_name}_feature_scaling",
            message=(
                f"-- Requested {scaling_method} scaling for the {entity_name} "
                "features while supplying images. This step will be skipped"
            ),
            value="images",
            level="warning",
        )
        return

    _emit_data_operation_started_if_verbose(
        verbose,
        print_mode,
        progress,
        subject=f"{entity_name}_feature_scaling",
        message=f"Scaling {entity_name} features... ",
    )

    scaler = MinMaxScaler() if scaling_method == "MinMax" else StandardScaler()
    scaler.fit(np.stack(train_frame["features"].to_numpy()))
    for mode in ("train", "val", "test"):
        source = data[mode][feature_key]
        if source is None:
            raise ValueError(
                f"{entity_name.capitalize()} features are missing from the {mode} "
                "split after splitting."
            )
        feature_frame = source["data"]
        feature_frame["features"] = feature_frame.apply(
            lambda row: normalize(row["features"], scaler),
            axis=1,
        )

    _emit_data_operation_completed_if_verbose(
        verbose,
        print_mode,
        progress,
        subject=f"{entity_name}_feature_scaling",
        message="Done",
        value=scaling_method,
    )


def data_process(
    data: Mapping[str, Any],
    validation_setting: str | None = None,
    split_method: str = "random",
    ratio: object = None,
    shuffle: bool = True,
    seed: int | None = 42,
    verbose: bool = False,
    print_mode: str = "basic",
    scale_instance_features: str | None = None,
    scale_target_features: str | None = None,
    *,
    progress: DataProgressObserver | None = None,
) -> tuple[ProcessedSplit, ProcessedSplit, ProcessedSplit, DataInfo]:
    """The main function that handles all the preprocessing steps and checks needed to prepare the dataset to be used by the model.

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        validation_setting (str, optional): The validation setting of the current problem. Defaults to None.
        split_method (str, optional): The splitting method used. The current implementation only supports the 'random split' using a specific seed but a future goal is to also offer a stratified option. Defaults to 'random'.
        ratio (dict, optional): The train, val and test ratios used to split the data. Defaults to {'train': 0.7, 'test': 0.2, 'val': 0.1}.
        shuffle(bool, optional): Whether or not the dataset will be shuffled before the split. If is set to False, the seed value is not used.
        seed (int, optional): The seed used to initiate the randomized split. Defaults to 42.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        scale_instance_features (str, optional): The scaler used for the instance features. Possible values are 'MinMax' for the MinMax scaler and 'Standard' for the standard scaler. Defaults to None.
        scale_target_features (str, optional): The scaler used for the target features. Possible values are 'MinMax' for the MinMax scaler and 'Standard' for the standard scaler. Defaults to None.
        progress (DataProgressObserver | None, optional): Receives verbose messages from the complete data-processing pipeline. Defaults to None.

    Returns:
        dict, dict, dict, dict: Four different dictionaries containing:
            * Train processed data
            * Validation processed data
            * Test processed data
            * general information about the datasets
    """
    data = _prepare_dataset_mapping(data)
    ratio = validate_split_ratio(
        {"train": 0.7, "test": 0.2, "val": 0.1} if ratio is None else ratio
    )
    validation_setting = _normalize_validation_setting(validation_setting)
    instance_scaling_method = _normalize_scaling_method(
        scale_instance_features,
        "scale_instance_features",
    )
    target_scaling_method = _normalize_scaling_method(
        scale_target_features,
        "scale_target_features",
    )

    # check if at least one train dataset exists (Can be only the y_train)
    if data["train"]["y"] is None:
        raise AttributeError(
            ("error: " if print_mode == "dev" else "")
            + "Passing y_train is the minimum requirement for creating a dataset"
        )

    if data["test"]["y"] is None and data["val"]["y"] is not None:
        raise ValueError(
            ("error: " if print_mode == "dev" else "")
            + "Supplying a validation set without a test set is not currently supported!!"
        )

    """
    check if the user provides only the training data without a validation setting.
    In this case we cannot automatically infer the validation setting.

    Also instance or target side information without the corresponding score matrices doesn't make sense
    """
    if data["test"]["y"] is None and validation_setting is None:
        raise AttributeError(
            ("error: " if print_mode == "dev" else "")
            + "The validation setting must be specified manually. To automatically infer it you must pass the test set as well!!"
        )
    """
    else:
        if data['train']['X_instance'] is not None and data['test']['X_instance'] is None:
            raise AttributeError(('error: ' if print_mode=='dev' else '')+'Instance features are available for the train set  but not the test set')
        if data['train']['X_target'] is not None and data['test']['X_target'] is None:
            raise AttributeError(('error: ' if print_mode=='dev' else '')+'Target features are available for the train set  but not the test set')
    """

    if data["val"]["y"] is None and (
        data["val"]["X_instance"] is not None or data["val"]["X_target"] is not None
    ):
        warnings.warn(
            ("warning: " if print_mode == "dev" else "")
            + "Warning: You provided side information for the validation set without the interaction matrix. This info won't be used",
            stacklevel=2,
        )
    """
    else:
        if data['train']['X_instance'] is not None and data['val']['X_instance'] is None:
            raise AttributeError(('error: ' if print_mode=='dev' else '')+'Instance features are available for the train set  but not the validation set')
        if data['train']['X_target'] is not None and data['val']['X_target'] is None:
            raise AttributeError(('error: ' if print_mode=='dev' else '')+'Target features are available for the train set  but not the validation set')
    """

    # check if the specified validation setting makes sense given the supplied datasets
    if validation_setting is not None:
        if validation_setting == "B" and data["train"]["X_instance"] is None:
            raise ValueError(
                ("error: " if print_mode == "dev" else "")
                + "Specified validation setting B without supplying instance features"
            )
        elif validation_setting == "C" and data["train"]["X_target"] is None:
            raise ValueError(
                ("error: " if print_mode == "dev" else "")
                + "Specified validation setting C without supplying target features"
            )
        elif validation_setting == "D" and data["train"]["X_instance"] is None:
            raise ValueError(
                ("error: " if print_mode == "dev" else "")
                + "Specified validation setting D without supplying instance features"
            )
        elif validation_setting == "D" and data["train"]["X_target"] is None:
            raise ValueError(
                ("error: " if print_mode == "dev" else "")
                + "Specified validation setting D without supplying target features"
            )

    progress_observer = _resolve_data_progress(verbose, print_mode, progress)

    # process the interaction data. The data dictionary will be augmented with additional inferred info
    data["train"]["y"] = process_interaction_data(
        data["train"]["y"],
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )
    if data["test"]["y"] is not None:
        data["test"]["y"] = process_interaction_data(
            data["test"]["y"],
            verbose=verbose,
            print_mode=print_mode,
            progress=progress_observer,
        )
    if data["val"]["y"] is not None:
        data["val"]["y"] = process_interaction_data(
            data["val"]["y"],
            verbose=verbose,
            print_mode=print_mode,
            progress=progress_observer,
        )

    # check for format consistency across the interaction files
    check_interaction_files_format(
        data,
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )
    # check for format consistency for instance and target ids across the interaction files
    check_interaction_files_column_type_format(
        data,
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )

    _emit_data_separator(verbose, print_mode, progress_observer)
    # get the type of the target variable.
    target_variable_type = check_target_variable_type(
        data,
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )
    _emit_data_message(
        verbose,
        print_mode,
        progress_observer,
        subject="target_variable_type",
        message=(
            "Automatically detected type of target variable type: "
            + target_variable_type
        ),
        value=target_variable_type,
        end="\n\n",
    )

    validation_setting_detected: ValidationSetting | None = None
    if validation_setting is None:
        novel_instances = check_novel_instances(
            data["train"]["y"],
            data["test"]["y"],
            verbose=verbose,
            print_mode=print_mode,
            progress=progress_observer,
        )
        if novel_instances is None:
            novel_instances_message = (
                "-- Could not determine whether novel instances exist"
            )
        elif novel_instances:
            novel_instances_message = "-- Novel instances detected in the test set"
        else:
            novel_instances_message = "-- No novel instances detected in the test set"
        _emit_data_message(
            verbose,
            print_mode,
            progress_observer,
            subject="novel_instances",
            message=novel_instances_message,
            value=novel_instances,
        )

        novel_targets = check_novel_targets(
            data["train"]["y"],
            data["test"]["y"],
            verbose=verbose,
            print_mode=print_mode,
            progress=progress_observer,
        )
        if novel_targets is None:
            novel_targets_message = "-- Could not determine whether novel targets exist"
        elif novel_targets:
            novel_targets_message = "-- Novel targets detected in the test set"
        else:
            novel_targets_message = "-- No novel targets detected in the test set"
        _emit_data_message(
            verbose,
            print_mode,
            progress_observer,
            subject="novel_targets",
            message=novel_targets_message,
            value=novel_targets,
        )

        validation_setting_detected = get_estimated_validation_setting(
            novel_instances,
            novel_targets,
            verbose,
            print_mode=print_mode,
            progress=progress_observer,
        )
        _emit_data_separator(verbose, print_mode, progress_observer)
        if validation_setting_detected is None:
            raise ValueError(
                ("error: " if print_mode == "dev" else "")
                + "Validation setting could not be inferred. Provide "
                "validation_setting explicitly or use triplet interaction data "
                "with explicit entity IDs."
            )
        validation_setting = validation_setting_detected

    # process the instance features
    data["train"]["X_instance"] = process_instance_features(
        data["train"]["X_instance"],
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )
    data["test"]["X_instance"] = process_instance_features(
        data["test"]["X_instance"],
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )
    data["val"]["X_instance"] = process_instance_features(
        data["val"]["X_instance"],
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )
    # process the target features
    data["train"]["X_target"] = process_target_features(
        data["train"]["X_target"],
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )
    data["test"]["X_target"] = process_target_features(
        data["test"]["X_target"],
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )
    data["val"]["X_target"] = process_target_features(
        data["val"]["X_target"],
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )

    _emit_data_separator(verbose, print_mode, progress_observer)
    if data["train"]["X_instance"] is not None:
        instance_branch_input_dim = data["train"]["X_instance"]["num_features"]
    else:
        instance_branch_input_dim = len(
            data["train"]["y"]["data"]["instance_id"].unique()
        )
    if data["train"]["X_target"] is not None:
        target_branch_input_dim = data["train"]["X_target"]["num_features"]
    else:
        target_branch_input_dim = len(data["train"]["y"]["data"]["target_id"].unique())
    data_info: DataInfo = {
        "detected_validation_setting": validation_setting,
        "detected_problem_mode": (
            "classification" if target_variable_type == "binary" else "regression"
        ),
        "instance_branch_input_dim": instance_branch_input_dim,
        "target_branch_input_dim": target_branch_input_dim,
    }
    data["info"] = data_info

    cross_input_consistency_check_instances(
        data,
        validation_setting,
        verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )
    cross_input_consistency_check_targets(
        data,
        validation_setting,
        verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )

    _emit_data_separator(verbose, print_mode, progress_observer)
    split_data(
        data,
        validation_setting=validation_setting,
        split_method=split_method,
        ratio=ratio,
        shuffle=shuffle,
        seed=seed,
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )

    _scale_feature_splits(
        data,
        "X_instance",
        "instance",
        instance_scaling_method,
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )
    _scale_feature_splits(
        data,
        "X_target",
        "target",
        target_scaling_method,
        verbose=verbose,
        print_mode=print_mode,
        progress=progress_observer,
    )

    return (
        cast(ProcessedSplit, data["train"]),
        cast(ProcessedSplit, data["val"]),
        cast(ProcessedSplit, data["test"]),
        data_info,
    )
