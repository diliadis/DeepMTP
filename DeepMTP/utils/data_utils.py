from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from itertools import combinations
from typing import Any, Literal, Protocol, TypeAlias, TypedDict, cast

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample, shuffle
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from DeepMTP._split_utils import (
    absolute_split_sizes as _absolute_split_sizes,
)
from DeepMTP._split_utils import (
    validate_split_ratio as _validate_split_ratio,
)

pd.set_option("mode.chained_assignment", None)

VariableType: TypeAlias = Literal["binary", "real-valued"]
ValidationSetting: TypeAlias = Literal["A", "B", "C", "D"]
ScalingMethod: TypeAlias = Literal["MinMax", "Standard"]
SamplerMode: TypeAlias = Literal["micro", "macro", "instance"]
DatasetMapping: TypeAlias = Mapping[str, Any]
ProcessedSplit: TypeAlias = dict[str, Any]


class DataInfo(TypedDict):
    """Metadata inferred while preparing a multi-target dataset."""

    detected_validation_setting: ValidationSetting
    detected_problem_mode: Literal["classification", "regression"]
    instance_branch_input_dim: int | None
    target_branch_input_dim: int | None


class DatasetItem(TypedDict):
    """One model-ready interaction returned by ``BaseDataset``."""

    instance_id: int
    target_id: int
    instance_features: Any
    target_features: Any
    score: Any


class NoveltyInferenceWarning(UserWarning):
    """Warning emitted when entity identity was lost in matrix input."""


class InteractionInfo(TypedDict):
    """Normalized interaction data and its detected schema."""

    data: pd.DataFrame
    original_format: Literal["triplets", "numpy"]
    instance_id_type: Literal["int"]
    target_id_type: Literal["int"]
    missing_values: bool


class FeatureInfo(TypedDict):
    """Normalized entity features and their detected representation."""

    data: pd.DataFrame
    num_features: int | None
    info: Literal["numpy", "dataframe", "images"]


class Transformer(Protocol):
    """Minimal interface implemented by supported feature scalers."""

    def transform(self, values: Any) -> Any:
        """Transform values using a previously fitted scaler."""


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


def process_interaction_data(
    interaction_data: pd.DataFrame | np.ndarray,
    verbose: bool = False,
    print_mode: str = "basic",
) -> InteractionInfo:
    """A function that processes the interaction data. It is called separately for the train, val and test interaction data.
    There are two main types of interaction data formats that are supported:
    -> numpy format: This is a 2d numpy array that represents the interaction (also called score) matrix usually found in problems settings with fully observed matrices (multi-label classification, multivariate regression)
    -> triplet format: The most flexible format as it can be used to represent every possible problem setting

    Args:
        interaction_data (_type_): a numpy array or a dataframe with the interaction data
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    Returns:
        dict: a dictionary with the interaction data and additional information that could be detected (format, type of instance and target ids, etc.)
    """
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
            print(
                ("info: " if print_mode == "dev" else "")
                + "Interaction file: 2d numpy array format detected"
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
        print(
            ("info: " if print_mode == "dev" else "")
            + "Interaction file: triplet format detected"
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
) -> None:
    """Checks the format of the interaction data. If any inconsistencies are detected (like different formats between train and test interaction data), an exception is raised

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    """
    distinct_formats = {
        data[mode]["y"]["original_format"]
        for mode in ["train", "test", "val"]
        if data[mode]["y"] is not None
    }
    if len(distinct_formats) != 0:
        if verbose:
            print(
                ("info: " if print_mode == "dev" else "")
                + "Interaction file: checking format consistency... ",
                end="",
            )
        if len(distinct_formats) == 1:
            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Passed")
        else:
            raise ValueError(
                ("error: " if print_mode == "dev" else "")
                + "Inconsistent file formats across the different interaction files"
            )


def check_interaction_files_column_type_format(
    data: DatasetMapping,
    verbose: bool = False,
    print_mode: str = "basic",
) -> None:
    """Checks the type of the instance and target ids in the interaction data. If any inconsistencies are detected (like different id types between train and test interaction data), an exception is raised

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    """
    # check for instance id type concistency across the interaction data sources
    distinct_instance_id_format = {
        data[mode]["y"]["instance_id_type"]
        for mode in ["train", "test", "val"]
        if data[mode]["y"] is not None
    }
    if len(distinct_instance_id_format) != 0:
        if verbose:
            print(
                ("info: " if print_mode == "dev" else "")
                + "Interaction file: checking instance id format consistency... ",
                end="",
            )
        if len(distinct_instance_id_format) == 1:
            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Passed")
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
            print(
                ("info: " if print_mode == "dev" else "")
                + "Interaction file: checking target id type consistency... ",
                end="",
            )
        if len(distinct_target_id_format) == 1:
            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Passed")
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
) -> VariableType:
    """Checks the type of the target variable in the interaction data. If any inconsistencies are detected (like different id types between train and test interaction data), an exception is raised

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    """
    distinct_target_variable_type = {
        check_variable_type(data[mode]["y"]["data"])
        for mode in ["train", "test", "val"]
        if data[mode]["y"] is not None
    }
    if len(distinct_target_variable_type) != 0:
        if verbose:
            print(
                ("info: " if print_mode == "dev" else "")
                + "Interaction file: checking target variable type consistency... ",
                end="",
            )
        if len(distinct_target_variable_type) == 1:
            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Passed")
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
    return distinct_target_variable_type.pop()


def _check_novel_entities(
    train: DatasetMapping | None,
    test: DatasetMapping | None,
    *,
    id_column: Literal["instance_id", "target_id"],
    entity_name: Literal["instances", "targets"],
    verbose: bool,
    print_mode: str,
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
        print(
            ("info: " if print_mode == "dev" else "")
            + f"Interaction file: Checking for novel {entity_name}... ",
            end="",
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
        print(("info: " if print_mode == "dev" else "") + "Done")
    return novelty_detected


def check_novel_instances(
    train: DatasetMapping | None,
    test: DatasetMapping | None,
    verbose: bool = False,
    print_mode: str = "basic",
) -> bool | None:
    """Return whether test interactions contain previously unseen instances."""
    return _check_novel_entities(
        train,
        test,
        id_column="instance_id",
        entity_name="instances",
        verbose=verbose,
        print_mode=print_mode,
    )


def check_novel_targets(
    train: DatasetMapping | None,
    test: DatasetMapping | None,
    verbose: bool = False,
    print_mode: str = "basic",
) -> bool | None:
    """Return whether test interactions contain previously unseen targets."""
    return _check_novel_entities(
        train,
        test,
        id_column="target_id",
        entity_name="targets",
        verbose=verbose,
        print_mode=print_mode,
    )


def get_estimated_validation_setting(
    novel_instances_flag: bool | None,
    novel_targets_flag: bool | None,
    verbose: bool,
    print_mode: str = "basic",
) -> ValidationSetting | None:
    """Uses the combination of infrmation about novel instances and novel targets to determine the validation setting that is possible.

    Args:
        novel_instances_flag (bool): A boolean that indicates whether or not the test set contains novel instances.
        novel_targets_flag (bool): A boolean that indicates whether or not the test set contains novel targets.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

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
        if verbose:
            print(
                ("info: " if print_mode == "dev" else "")
                + "Estimating validation setting... ",
                end="",
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
            print(("info: " if print_mode == "dev" else "") + "Done", end="")
    if verbose and validation_setting_detected is not None:
        print(
            ("info: " if print_mode == "dev" else "")
            + "-- Detected as setting :"
            + validation_setting_detected
        )
    return validation_setting_detected


def _feature_vector_size(features: pd.Series, entity_name: str) -> int:
    dimensions: set[int] = set()
    for feature_vector in features:
        vector = np.asarray(feature_vector)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(
                f"Each {entity_name} feature value must be a non-empty "
                "one-dimensional vector."
            )
        if not np.issubdtype(vector.dtype, np.number):
            raise TypeError(f"{entity_name.title()} feature vectors must be numeric.")
        dimensions.add(int(vector.size))
    if len(dimensions) != 1:
        raise ValueError(
            f"All {entity_name} feature vectors must have the same dimension."
        )
    return dimensions.pop()


def _process_entity_features(
    features: pd.DataFrame | np.ndarray | None,
    entity_name: Literal["instance", "target"],
    verbose: bool,
    print_mode: str,
) -> FeatureInfo | None:
    if features is None:
        return None
    if verbose:
        print(
            ("info: " if print_mode == "dev" else "")
            + f"{entity_name.title()} features file: processing features... ",
            end="",
        )

    if isinstance(features, np.ndarray):
        if features.ndim != 2:
            raise ValueError(
                f"{entity_name.title()} feature arrays must be two-dimensional."
            )
        if 0 in features.shape:
            raise ValueError(
                f"{entity_name.title()} feature arrays cannot have an empty dimension."
            )
        if not np.issubdtype(features.dtype, np.number):
            raise TypeError(f"{entity_name.title()} feature arrays must be numeric.")
        feature_frame = pd.DataFrame({"id": np.arange(len(features))})
        feature_frame["features"] = [row.copy() for row in features]
        feature_info: FeatureInfo = {
            "data": feature_frame,
            "num_features": int(features.shape[1]),
            "info": "numpy",
        }
    elif isinstance(features, pd.DataFrame):
        feature_frame = features.copy()
        if feature_frame.empty:
            raise ValueError(
                f"{entity_name.title()} features must contain at least one row."
            )
        if set(feature_frame.columns) not in ({"id", "features"}, {"id", "dir"}):
            raise ValueError(
                f"Columns {list(feature_frame.columns)!r} in the {entity_name} "
                "features dataframe are not recognized."
            )
        if feature_frame["id"].isna().any():
            raise ValueError(f"{entity_name.title()} feature IDs cannot be missing.")
        if feature_frame["id"].duplicated().any():
            raise ValueError(f"{entity_name.title()} feature IDs must be unique.")
        feature_frame = feature_frame.set_index("id", drop=False)
        if "features" in feature_frame:
            feature_dimension = _feature_vector_size(
                feature_frame["features"],
                entity_name,
            )
            feature_frame["features"] = [
                np.asarray(feature_vector).copy()
                for feature_vector in feature_frame["features"]
            ]
            feature_info = {
                "data": feature_frame,
                "num_features": feature_dimension,
                "info": "dataframe",
            }
        else:
            if feature_frame["dir"].isna().any():
                raise ValueError(
                    f"{entity_name.title()} feature directories cannot be missing."
                )
            feature_info = {
                "data": feature_frame,
                "num_features": None,
                "info": "images",
            }
    else:
        raise TypeError(
            f"Could not recognize the {entity_name} feature data. Expected a "
            "two-dimensional NumPy array or a DataFrame with <id, features> "
            "or <id, dir> columns."
        )

    if verbose:
        print(("info: " if print_mode == "dev" else "") + "Done")
    return feature_info


def process_instance_features(
    instance_features: pd.DataFrame | np.ndarray | None,
    verbose: bool = False,
    print_mode: str = "basic",
) -> FeatureInfo | None:
    """Normalize instance features without mutating caller-owned inputs."""
    return _process_entity_features(
        instance_features,
        "instance",
        verbose,
        print_mode,
    )


def process_target_features(
    target_features: pd.DataFrame | np.ndarray | None,
    verbose: bool = False,
    print_mode: str = "basic",
) -> FeatureInfo | None:
    """Normalize target features without mutating caller-owned inputs."""
    return _process_entity_features(
        target_features,
        "target",
        verbose,
        print_mode,
    )


def _cross_input_source_layout(
    data: DatasetMapping,
    feature_key: Literal["X_instance", "X_target"],
) -> tuple[int, int, list[str]]:
    interaction_modes = [
        mode for mode in ("train", "test", "val") if data[mode]["y"] is not None
    ]
    feature_modes = [
        mode for mode in ("train", "test", "val") if data[mode][feature_key] is not None
    ]
    if len(feature_modes) == len(interaction_modes) and set(feature_modes) != set(
        interaction_modes
    ):
        raise ValueError(
            f"{feature_key} files must correspond to the supplied interaction splits."
        )
    if len(feature_modes) == 1 and feature_modes[0] != "train":
        raise ValueError(f"A shared {feature_key} file must be supplied on train.")
    return len(interaction_modes), len(feature_modes), interaction_modes


def cross_input_consistency_check_instances(
    data: DatasetMapping,
    validation_setting: ValidationSetting,
    verbose: bool,
    print_mode: str = "basic",
) -> None:
    """Checks the consistency of instance ids in the interaction data and instance features. The requirements to pass this check change depending on the format of the interaction data and instance features

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        validation_setting (str): The validation setting of the current problem.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    """
    if validation_setting not in {"A", "B", "C", "D"}:
        raise ValueError(
            f"Unknown validation setting {validation_setting!r} for instance checks."
        )
    (
        num_interaction_data_sources,
        num_instance_features_sources,
        valid_modes,
    ) = _cross_input_source_layout(
        data,
        "X_instance",
    )

    if num_instance_features_sources != 0:
        # if the interaction files have a numpy format, the only case currently allowed is the following: There are as many interaction files as there are feature files
        if "numpy" in set(
            [
                data[mode]["y"]["original_format"]
                for mode in ["train", "test", "val"]
                if data[mode]["y"] is not None
            ]
        ):
            if validation_setting in ["B", "D"]:
                if num_interaction_data_sources != num_instance_features_sources:
                    raise ValueError(
                        ("error: " if print_mode == "dev" else "")
                        + "Different number of (numpy) interaction files and instance feature files is not currently supported"
                    )
                else:
                    if verbose:
                        print(
                            ("info: " if print_mode == "dev" else "")
                            + "Cross input consistency for (numpy) interaction data and instance features checks out"
                        )
                    # if validation_setting == 'B':
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = set(
                            data[mode]["y"]["data"]["instance_id"]
                        )
                        unique_entities_in_features_file = set(
                            data[mode]["X_instance"]["data"]["id"].unique()
                        )
                        if (
                            unique_entities_in_interactions_file.symmetric_difference(
                                unique_entities_in_features_file
                            )
                            == set()
                        ):
                            if verbose:
                                print(
                                    ("info: " if print_mode == "dev" else "")
                                    + "-- Same instance ids in the interaction and features files for the "
                                    + mode
                                    + " set"
                                )
                        else:
                            raise ValueError(
                                ("error: " if print_mode == "dev" else "")
                                + "Different instance ids in the interaction and features files for the "
                                + mode
                                + " set."
                            )

            elif validation_setting in ["C", "A"]:
                if num_instance_features_sources != 1:
                    raise ValueError(
                        ("error: " if print_mode == "dev" else "")
                        + "Setting "
                        + validation_setting
                        + " needs only one instance feature file"
                    )
                else:
                    if verbose:
                        print(
                            ("info: " if print_mode == "dev" else "")
                            + "Cross input consistency for (numpy) interaction data and instance features checks out"
                        )
                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = (
                            unique_entities_in_interactions_file.union(
                                set(data[mode]["y"]["data"]["instance_id"])
                            )
                        )
                    unique_entities_in_features_file = set(
                        data["train"]["X_instance"]["data"]["id"].unique()
                    )
                    if (
                        unique_entities_in_interactions_file.symmetric_difference(
                            unique_entities_in_features_file
                        )
                        == set()
                    ):
                        if verbose:
                            print(
                                ("info: " if print_mode == "dev" else "")
                                + "-- Same instance ids in the interaction and features file"
                            )
                    else:
                        raise ValueError(
                            ("error: " if print_mode == "dev" else "")
                            + "Different instance ids in the interaction and features files"
                        )
            else:  # pragma: no cover
                pass
        else:  # dataframe version
            """
            if the interaction files have a triplet format, only two cases are currently allowed
                1.There are as many interaction files as there are feature files
                2.There are multiple interaction files and a single feature file
            """

            if validation_setting in ["B", "D", "A"]:
                if num_instance_features_sources > num_interaction_data_sources:
                    raise ValueError(
                        ("error: " if print_mode == "dev" else "")
                        + "More instance feature files than (triplet) interaction files provided. This is currently not supported"
                    )
                elif num_instance_features_sources == num_interaction_data_sources:
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = set(
                            data[mode]["y"]["data"]["instance_id"]
                        )
                        unique_entities_in_features_file = set(
                            data[mode]["X_instance"]["data"]["id"].unique()
                        )
                        if (
                            unique_entities_in_interactions_file.symmetric_difference(
                                unique_entities_in_features_file
                            )
                            == set()
                        ):
                            if verbose:
                                print(
                                    ("info: " if print_mode == "dev" else "")
                                    + "-- Same instance ids in the interaction and features files for the "
                                    + mode
                                    + " set"
                                )
                        else:
                            raise ValueError(
                                ("error: " if print_mode == "dev" else "")
                                + "Different instance ids in the interaction and features files for the "
                                + mode
                                + " set"
                            )
                else:
                    if (
                        num_interaction_data_sources == 3
                        and num_instance_features_sources == 2
                    ):
                        raise ValueError(
                            ("error: " if print_mode == "dev" else "")
                            + "When three (triplet) interaction files are provided, only one instance features file is allowed"
                        )
                    else:
                        unique_entities_in_interactions_file = set()
                        for mode in valid_modes:
                            unique_entities_in_interactions_file = (
                                unique_entities_in_interactions_file.union(
                                    set(data[mode]["y"]["data"]["instance_id"])
                                )
                            )
                        unique_entities_in_features_file = set(
                            data["train"]["X_instance"]["data"]["id"].unique()
                        )
                        if (
                            unique_entities_in_interactions_file.symmetric_difference(
                                unique_entities_in_features_file
                            )
                            == set()
                        ):
                            if verbose:
                                print(
                                    ("info: " if print_mode == "dev" else "")
                                    + "-- Same instance ids in the interaction and features file"
                                )
                        else:
                            raise ValueError(
                                ("error: " if print_mode == "dev" else "")
                                + "Different instance ids in the interaction and features files"
                            )
            elif validation_setting == "C":
                if num_instance_features_sources != 1:
                    raise ValueError(
                        ("error: " if print_mode == "dev" else "")
                        + "When "
                        + str(len(valid_modes))
                        + " (triplet) interaction files are provided with setting "
                        + validation_setting
                        + ", only one instance features file is allowed"
                    )
                else:
                    if verbose:
                        print(
                            ("info: " if print_mode == "dev" else "")
                            + "-- Cross input consistency for (triplet) interaction data and instance features checks out"
                        )
                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = (
                            unique_entities_in_interactions_file.union(
                                set(data[mode]["y"]["data"]["instance_id"])
                            )
                        )
                    unique_entities_in_features_file = set(
                        data["train"]["X_instance"]["data"]["id"].unique()
                    )
                    if (
                        unique_entities_in_interactions_file.symmetric_difference(
                            unique_entities_in_features_file
                        )
                        == set()
                    ):
                        if verbose:
                            print(
                                ("info: " if print_mode == "dev" else "")
                                + "-- Same instance ids in the interaction and features file"
                            )
                    else:
                        raise ValueError(
                            ("error: " if print_mode == "dev" else "")
                            + "Different instance ids in the interaction and features files"
                        )
            else:  # pragma: no cover
                pass


def cross_input_consistency_check_targets(
    data: DatasetMapping,
    validation_setting: ValidationSetting,
    verbose: bool,
    print_mode: str = "basic",
) -> None:
    """Checks the consistency of target ids in the interaction data and target features. The requirements to pass this check change depending on the format of the interaction data and target features

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        validation_setting (str): The validation setting of the current problem.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    """
    if validation_setting not in {"A", "B", "C", "D"}:
        raise ValueError(
            f"Unknown validation setting {validation_setting!r} for target checks."
        )
    (
        num_interaction_data_sources,
        num_target_features_sources,
        valid_modes,
    ) = _cross_input_source_layout(
        data,
        "X_target",
    )

    if num_target_features_sources != 0:
        # if the interaction files have a numpy format, the only case currently allowed is the following: There are as many interaction files as there are feature files
        if "numpy" in set(
            [
                data[mode]["y"]["original_format"]
                for mode in ["train", "test", "val"]
                if data[mode]["y"] is not None
            ]
        ):
            if validation_setting in ["C", "D"]:
                if num_interaction_data_sources != num_target_features_sources:
                    raise ValueError(
                        ("error: " if print_mode == "dev" else "")
                        + "Different number of (numpy) interaction files and target feature files is not currently supported"
                    )
                else:
                    if verbose:
                        print(
                            ("info: " if print_mode == "dev" else "")
                            + "Cross input consistency for (numpy) interaction data and target features checks out"
                        )
                    # if validation_setting == 'C':
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = set(
                            data[mode]["y"]["data"]["target_id"]
                        )
                        unique_entities_in_features_file = set(
                            data[mode]["X_target"]["data"]["id"].unique()
                        )
                        if (
                            unique_entities_in_interactions_file.symmetric_difference(
                                unique_entities_in_features_file
                            )
                            == set()
                        ):
                            if verbose:
                                print(
                                    ("info: " if print_mode == "dev" else "")
                                    + "-- Same target ids in the interaction and features files for the "
                                    + mode
                                    + " set"
                                )
                        else:
                            raise ValueError(
                                ("error: " if print_mode == "dev" else "")
                                + "Different target ids in the interaction and features files for the "
                                + mode
                                + " set."
                            )
            elif validation_setting in ["B", "A"]:
                if num_target_features_sources != 1:
                    raise ValueError(
                        ("error: " if print_mode == "dev" else "")
                        + "Setting "
                        + validation_setting
                        + " needs only one target feature file"
                    )
                else:
                    if verbose:
                        print(
                            ("info: " if print_mode == "dev" else "")
                            + "Cross input consistency for (numpy) interaction data and target features checks out"
                        )
                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = (
                            unique_entities_in_interactions_file.union(
                                set(data[mode]["y"]["data"]["target_id"])
                            )
                        )
                    unique_entities_in_features_file = set(
                        data["train"]["X_target"]["data"]["id"].unique()
                    )
                    if (
                        unique_entities_in_interactions_file.symmetric_difference(
                            unique_entities_in_features_file
                        )
                        == set()
                    ):
                        if verbose:
                            print(
                                ("info: " if print_mode == "dev" else "")
                                + "-- Same target ids in the interaction and features file"
                            )
                    else:
                        raise ValueError(
                            ("error: " if print_mode == "dev" else "")
                            + "Different target ids in the interaction and features files"
                        )
            else:  # pragma: no cover
                pass
        else:
            """
            if the interaction files have a triplet format, only two cases are currently allowed
                1.There are as many interaction files as there are feature files
                2.There are multiple interaction files and a single feature file
            """

            if validation_setting in ["C", "D", "A"]:
                if num_target_features_sources > num_interaction_data_sources:
                    raise ValueError(
                        ("error: " if print_mode == "dev" else "")
                        + "More target feature files than (triplet) interaction files provided. This is currently not supported"
                    )
                elif num_target_features_sources == num_interaction_data_sources:
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = set(
                            data[mode]["y"]["data"]["target_id"]
                        )
                        unique_entities_in_features_file = set(
                            data[mode]["X_target"]["data"]["id"].unique()
                        )
                        if (
                            unique_entities_in_interactions_file.symmetric_difference(
                                unique_entities_in_features_file
                            )
                            == set()
                        ):
                            if verbose:
                                print(
                                    ("info: " if print_mode == "dev" else "")
                                    + "-- Same target ids in the interaction and features files for the "
                                    + mode
                                    + " set"
                                )
                        else:
                            raise ValueError(
                                ("error: " if print_mode == "dev" else "")
                                + "Different target ids in the interaction and features files for the "
                                + mode
                                + " set"
                            )
                else:
                    if (
                        num_interaction_data_sources == 3
                        and num_target_features_sources == 2
                    ):
                        raise ValueError(
                            ("error: " if print_mode == "dev" else "")
                            + "When three (triplet) interaction files are provided, only one target features file is allowed"
                        )
                    else:
                        unique_entities_in_interactions_file = set()
                        for mode in valid_modes:
                            unique_entities_in_interactions_file = (
                                unique_entities_in_interactions_file.union(
                                    set(data[mode]["y"]["data"]["target_id"])
                                )
                            )
                        unique_entities_in_features_file = set(
                            data["train"]["X_target"]["data"]["id"].unique()
                        )
                        if (
                            unique_entities_in_interactions_file.symmetric_difference(
                                unique_entities_in_features_file
                            )
                            == set()
                        ):
                            if verbose:
                                print(
                                    ("info: " if print_mode == "dev" else "")
                                    + "-- Same target ids in the interaction and features file"
                                )
                        else:
                            raise ValueError(
                                ("error: " if print_mode == "dev" else "")
                                + "Different target ids in the interaction and features files"
                            )
            elif validation_setting == "B":
                if num_target_features_sources != 1:
                    raise ValueError(
                        ("error: " if print_mode == "dev" else "")
                        + "When "
                        + str(len(valid_modes))
                        + " (triplet) interaction files are provided with setting "
                        + validation_setting
                        + ", only one target features file is allowed"
                    )
                else:
                    if verbose:
                        print(
                            ("info: " if print_mode == "dev" else "")
                            + "-- Cross input consistency for (triplet) interaction data and target features checks out"
                        )
                    unique_entities_in_interactions_file = set()
                    for mode in valid_modes:
                        unique_entities_in_interactions_file = (
                            unique_entities_in_interactions_file.union(
                                set(data[mode]["y"]["data"]["target_id"])
                            )
                        )
                    unique_entities_in_features_file = set(
                        data["train"]["X_target"]["data"]["id"].unique()
                    )
                    if (
                        unique_entities_in_interactions_file.symmetric_difference(
                            unique_entities_in_features_file
                        )
                        == set()
                    ):
                        if verbose:
                            print(
                                ("info: " if print_mode == "dev" else "")
                                + "-- Same target ids in the interaction and features file"
                            )
                    else:
                        raise ValueError(
                            ("error: " if print_mode == "dev" else "")
                            + "Different target ids in the interaction and features files"
                        )
            else:  # pragma: no cover
                pass


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


def split_data(
    data: MutableMapping[str, Any],
    validation_setting: ValidationSetting,
    split_method: str,
    ratio: object,
    shuffle: bool,
    seed: int | None,
    verbose: bool,
    print_mode: str = "basic",
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
            if verbose:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "Splitting train to train-test according to validation setting "
                    + validation_setting
                    + "... ",
                    end="",
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
            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Done")

        if data["val"]["y"] is None:
            if verbose:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "Splitting train to train-val according to validation setting "
                    + validation_setting
                    + "... ",
                    end="",
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
            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Done")

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
            if verbose:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "Splitting train to train-test according to validation setting "
                    + validation_setting
                    + "... ",
                    end="",
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
            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Done")

        if data["val"]["y"] is None:
            if verbose:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "Splitting train to train-val according to validation setting "
                    + validation_setting
                    + "... ",
                    end="",
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
            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Done")

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
            if verbose:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "Splitting train to train-test according to validation setting "
                    + validation_setting
                    + "... ",
                    end="",
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
            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Done")

        if data["val"]["y"] is None:
            if verbose:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "Splitting train to train-val according to validation setting "
                    + validation_setting
                    + "... ",
                    end="",
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
            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Done")

    elif validation_setting == "D":
        if data["test"]["y"] is None:
            if verbose:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "Splitting train to train-test according to validation setting "
                    + validation_setting
                    + "... ",
                    end="",
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

            if verbose:
                print(("info: " if print_mode == "dev" else "") + "Done")

        if data["val"]["y"] is None:
            if verbose:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "Splitting train to train-val according to validation setting "
                    + validation_setting
                    + "... ",
                    end="",
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

        if verbose:
            print(("info: " if print_mode == "dev" else "") + "Done")

    _detach_shared_feature_sources(data)
    _validate_split_result(data, validation_setting)


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
) -> None:
    if scaling_method is None:
        return

    train_source = data["train"][feature_key]
    if train_source is None:
        if verbose:
            print(
                ("warning: " if print_mode == "dev" else "")
                + f"-- Requested {scaling_method} scaling for the {entity_name} "
                f"features without supplying any {entity_name} features. This "
                "step will be skipped"
            )
        return

    train_frame = train_source["data"]
    if "features" not in train_frame.columns:
        if verbose:
            print(
                ("warning: " if print_mode == "dev" else "")
                + f"-- Requested {scaling_method} scaling for the {entity_name} "
                "features while supplying images. This step will be skipped"
            )
        return

    if verbose:
        print(
            ("info: " if print_mode == "dev" else "")
            + f"Scaling {entity_name} features... ",
            end="",
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

    if verbose:
        print(("info: " if print_mode == "dev" else "") + "Done")


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

    # process the interaction data. The data dictionary will be augmented with additional inferred info
    data["train"]["y"] = process_interaction_data(
        data["train"]["y"], verbose=verbose, print_mode=print_mode
    )
    if data["test"]["y"] is not None:
        data["test"]["y"] = process_interaction_data(
            data["test"]["y"], verbose=verbose, print_mode=print_mode
        )
    if data["val"]["y"] is not None:
        data["val"]["y"] = process_interaction_data(
            data["val"]["y"], verbose=verbose, print_mode=print_mode
        )

    # check for format consistency across the interaction files
    check_interaction_files_format(data, verbose=verbose, print_mode=print_mode)
    # check for format consistency for instance and target ids across the interaction files
    check_interaction_files_column_type_format(
        data, verbose=verbose, print_mode=print_mode
    )

    if verbose:
        print()
    # get the type of the target variable.
    target_variable_type = check_target_variable_type(
        data, verbose=verbose, print_mode=print_mode
    )
    if verbose:
        print(
            ("info: " if print_mode == "dev" else "")
            + "Automatically detected type of target variable type: "
            + target_variable_type
            + "\n"
        )

    validation_setting_detected: ValidationSetting | None = None
    if validation_setting is None:
        novel_instances = check_novel_instances(
            data["train"]["y"],
            data["test"]["y"],
            verbose=verbose,
            print_mode=print_mode,
        )
        if verbose:  # pragma: no cover
            if novel_instances is None:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "-- Could not determine whether novel instances exist"
                )
            elif novel_instances:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "-- Novel instances detected in the test set"
                )
            else:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "-- No novel instances detected in the test set"
                )

        novel_targets = check_novel_targets(
            data["train"]["y"],
            data["test"]["y"],
            verbose=verbose,
            print_mode=print_mode,
        )
        if verbose:  # pragma: no cover
            if novel_targets is None:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "-- Could not determine whether novel targets exist"
                )
            elif novel_targets:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "-- Novel targets detected in the test set"
                )
            else:
                print(
                    ("info: " if print_mode == "dev" else "")
                    + "-- No novel targets detected in the test set"
                )

        validation_setting_detected = get_estimated_validation_setting(
            novel_instances,
            novel_targets,
            verbose,
            print_mode=print_mode,
        )
        if verbose:
            print()
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
        data["train"]["X_instance"], verbose=verbose, print_mode=print_mode
    )
    data["test"]["X_instance"] = process_instance_features(
        data["test"]["X_instance"], verbose=verbose, print_mode=print_mode
    )
    data["val"]["X_instance"] = process_instance_features(
        data["val"]["X_instance"], verbose=verbose, print_mode=print_mode
    )
    # process the target features
    data["train"]["X_target"] = process_target_features(
        data["train"]["X_target"], verbose=verbose, print_mode=print_mode
    )
    data["test"]["X_target"] = process_target_features(
        data["test"]["X_target"], verbose=verbose, print_mode=print_mode
    )
    data["val"]["X_target"] = process_target_features(
        data["val"]["X_target"], verbose=verbose, print_mode=print_mode
    )

    if verbose:
        print()
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
        data, validation_setting, verbose, print_mode=print_mode
    )
    cross_input_consistency_check_targets(
        data, validation_setting, verbose, print_mode=print_mode
    )

    if verbose:
        print()
    split_data(
        data,
        validation_setting=validation_setting,
        split_method=split_method,
        ratio=ratio,
        shuffle=shuffle,
        seed=seed,
        verbose=verbose,
        print_mode=print_mode,
    )

    _scale_feature_splits(
        data,
        "X_instance",
        "instance",
        instance_scaling_method,
        verbose=verbose,
        print_mode=print_mode,
    )
    _scale_feature_splits(
        data,
        "X_target",
        "target",
        target_scaling_method,
        verbose=verbose,
        print_mode=print_mode,
    )

    return (
        cast(ProcessedSplit, data["train"]),
        cast(ProcessedSplit, data["val"]),
        cast(ProcessedSplit, data["test"]),
        data_info,
    )


class BaseDataset(Dataset):
    """Resolve interaction rows to model-ready entity features."""

    def __init__(
        self,
        config: Mapping[str, Any],
        data: Mapping[str, Any],
        instance_features: Mapping[str, Any] | None,
        target_features: Mapping[str, Any] | None,
        instance_transform: Callable[[Image.Image], Any] | None = None,
        target_transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.config = dict(config)
        self.instance_branch_input_dim = config.get("instance_branch_input_dim")
        self.target_branch_input_dim = config.get("target_branch_input_dim")
        self.instance_branch_architecture = config.get("instance_branch_architecture")
        self.target_branch_architecture = config.get("target_branch_architecture")
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
            self._validate_one_hot_ids(
                id_column="instance_id",
                input_dim=self.instance_branch_input_dim,
                entity_name="Instance",
            )
        if self.target_features is None:
            self.target_branch_input_dim = self._validate_input_dimension(
                self.target_branch_input_dim,
                "target_branch_input_dim",
            )
            self._validate_one_hot_ids(
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

    def _validate_one_hot_ids(
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
                f"{input_dim} when features are not supplied; found "
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
    """A sampler that can be used with pytorch dataloaders. The sampler can perform class balancing (binary and multi-class are currenty supported) at three different levels:
    a) micro: balancing classes at the level of the entire score matrix
    b) macro: balancing classes at the target level (basically balancing every column of the score matrix). WARNING: this can cause instances to be excluded from the final balanced dataset.
    c) instance: balancing classes at the instance level (basically balancing every row of the score matrix). WARNING: this can cause targets to be excluded from the final balanced dataset.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        mode: str = "micro",
        verbose: bool = False,
    ) -> None:
        """
        Args:
            data (DataFrame): The dataset that will be balanced
            mode (str, optional): The mode of balancing that will be used by the sampler
                            Default: micro
            verbose (float, optional): If True, prints status updated while balancing. This can be helpful for "instance" or "target" modes that take time to complete
                            Default: False
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if mode not in {"micro", "macro", "instance"}:
            raise ValueError(
                "mode must be one of 'micro', 'macro', or 'instance'; "
                f"received {mode!r}."
            )
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

        sampler_mode = cast(SamplerMode, mode)
        self.data = data.copy(deep=True).reset_index(drop=True)
        self.verbose = verbose
        if sampler_mode == "micro":
            self.data = self.get_balanced_df(self.data)
        elif sampler_mode == "macro":
            self.data = pd.concat(
                self.get_balanced_df(group)
                for _, group in self.data.groupby("target_id")
            )
        else:
            self.data = pd.concat(
                self.get_balanced_df(group)
                for _, group in self.data.groupby("instance_id")
            )

        # shuffle the dataframe so that you don't have the different classes in order
        self.data = shuffle(self.data, random_state=0)

    def get_balanced_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            data (DataFrame): The dataset that will be balanced
        """
        # isolate the minority class
        class_counts = data["value"].value_counts()
        if class_counts.empty:
            raise ValueError("Cannot balance data without class values.")
        minority_class = class_counts.idxmin()
        minority_class_size = int(class_counts.loc[minority_class])
        # gather remaining classes to be sampled
        majority_classes = list(data["value"].unique())
        majority_classes.remove(minority_class)

        sampled_df_per_majority_class: list[pd.DataFrame] = []
        # subsample from every non-minority class
        for class_val in majority_classes:
            sampled_df_per_majority_class.append(
                resample(
                    data[data["value"] == class_val],
                    replace=False,
                    n_samples=minority_class_size,
                    random_state=42,
                )
            )
        sampled_df_per_majority_class.append(data[data["value"] == minority_class])

        # concatenate all the sampled dataframes from every class
        return pd.concat(sampled_df_per_majority_class)

    def __iter__(self) -> Iterator[int]:
        return iter(self.data.index.tolist())

    def __len__(self) -> int:
        return len(self.data)
