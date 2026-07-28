"""Feature normalization and cross-input consistency validation."""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from DeepMTP.data_progress import (
    DataProgressObserver,
    _emit_data_message,
    _emit_data_operation_completed,
    _emit_data_operation_started,
    _resolve_data_progress,
)
from DeepMTP.interaction_data import DatasetMapping, ValidationSetting

__all__ = [
    "FeatureInfo",
    "cross_input_consistency_check_instances",
    "cross_input_consistency_check_targets",
    "process_instance_features",
    "process_target_features",
]


class FeatureInfo(TypedDict):
    """Normalized entity features and their detected representation."""

    data: pd.DataFrame
    num_features: int | None
    info: Literal["numpy", "dataframe", "images"]


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
    progress: DataProgressObserver | None,
) -> FeatureInfo | None:
    if features is None:
        return None
    progress_observer = _resolve_data_progress(verbose, print_mode, progress)
    if verbose:
        _emit_data_operation_started(
            progress_observer,
            subject=f"{entity_name}_features",
            message=f"{entity_name.title()} features file: processing features... ",
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
        _emit_data_operation_completed(
            progress_observer,
            subject=f"{entity_name}_features",
            message="Done",
            value=feature_info["info"],
        )
    return feature_info


def process_instance_features(
    instance_features: pd.DataFrame | np.ndarray | None,
    verbose: bool = False,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> FeatureInfo | None:
    """Normalize instance features without mutating caller-owned inputs.

    ``progress`` receives messages when ``verbose`` is enabled.
    """
    return _process_entity_features(
        instance_features,
        "instance",
        verbose,
        print_mode,
        progress,
    )


def process_target_features(
    target_features: pd.DataFrame | np.ndarray | None,
    verbose: bool = False,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> FeatureInfo | None:
    """Normalize target features without mutating caller-owned inputs.

    ``progress`` receives messages when ``verbose`` is enabled.
    """
    return _process_entity_features(
        target_features,
        "target",
        verbose,
        print_mode,
        progress,
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
    *,
    progress: DataProgressObserver | None = None,
) -> None:
    """Checks the consistency of instance ids in the interaction data and instance features. The requirements to pass this check change depending on the format of the interaction data and instance features

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        validation_setting (str): The validation setting of the current problem.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        progress (DataProgressObserver | None, optional): Receives verbose instance-consistency messages. Defaults to None.

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
                    _emit_data_message(
                        verbose,
                        print_mode,
                        progress,
                        subject="instance_feature_consistency",
                        message=(
                            "Cross input consistency for (numpy) interaction "
                            "data and instance features checks out"
                        ),
                        value="numpy",
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
                            _emit_data_message(
                                verbose,
                                print_mode,
                                progress,
                                subject="instance_feature_consistency",
                                message=(
                                    "-- Same instance ids in the interaction and "
                                    f"features files for the {mode} set"
                                ),
                                value=mode,
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
                    _emit_data_message(
                        verbose,
                        print_mode,
                        progress,
                        subject="instance_feature_consistency",
                        message=(
                            "Cross input consistency for (numpy) interaction "
                            "data and instance features checks out"
                        ),
                        value="numpy",
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
                        _emit_data_message(
                            verbose,
                            print_mode,
                            progress,
                            subject="instance_feature_consistency",
                            message=(
                                "-- Same instance ids in the interaction and "
                                "features file"
                            ),
                            value="shared",
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
                            _emit_data_message(
                                verbose,
                                print_mode,
                                progress,
                                subject="instance_feature_consistency",
                                message=(
                                    "-- Same instance ids in the interaction and "
                                    f"features files for the {mode} set"
                                ),
                                value=mode,
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
                            _emit_data_message(
                                verbose,
                                print_mode,
                                progress,
                                subject="instance_feature_consistency",
                                message=(
                                    "-- Same instance ids in the interaction and "
                                    "features file"
                                ),
                                value="shared",
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
                    _emit_data_message(
                        verbose,
                        print_mode,
                        progress,
                        subject="instance_feature_consistency",
                        message=(
                            "-- Cross input consistency for (triplet) interaction "
                            "data and instance features checks out"
                        ),
                        value="triplets",
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
                        _emit_data_message(
                            verbose,
                            print_mode,
                            progress,
                            subject="instance_feature_consistency",
                            message=(
                                "-- Same instance ids in the interaction and "
                                "features file"
                            ),
                            value="shared",
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
    *,
    progress: DataProgressObserver | None = None,
) -> None:
    """Checks the consistency of target ids in the interaction data and target features. The requirements to pass this check change depending on the format of the interaction data and target features

    Args:
        data (dict): The dictionary that store all possible data available in a multi-target prediction dataset.
        validation_setting (str): The validation setting of the current problem.
        verbose (bool, optional): Whether or not to print usefull info in the terminal. Defaults to False.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        progress (DataProgressObserver | None, optional): Receives verbose target-consistency messages. Defaults to None.

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
                    _emit_data_message(
                        verbose,
                        print_mode,
                        progress,
                        subject="target_feature_consistency",
                        message=(
                            "Cross input consistency for (numpy) interaction "
                            "data and target features checks out"
                        ),
                        value="numpy",
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
                            _emit_data_message(
                                verbose,
                                print_mode,
                                progress,
                                subject="target_feature_consistency",
                                message=(
                                    "-- Same target ids in the interaction and "
                                    f"features files for the {mode} set"
                                ),
                                value=mode,
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
                    _emit_data_message(
                        verbose,
                        print_mode,
                        progress,
                        subject="target_feature_consistency",
                        message=(
                            "Cross input consistency for (numpy) interaction "
                            "data and target features checks out"
                        ),
                        value="numpy",
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
                        _emit_data_message(
                            verbose,
                            print_mode,
                            progress,
                            subject="target_feature_consistency",
                            message=(
                                "-- Same target ids in the interaction and features file"
                            ),
                            value="shared",
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
                            _emit_data_message(
                                verbose,
                                print_mode,
                                progress,
                                subject="target_feature_consistency",
                                message=(
                                    "-- Same target ids in the interaction and "
                                    f"features files for the {mode} set"
                                ),
                                value=mode,
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
                            _emit_data_message(
                                verbose,
                                print_mode,
                                progress,
                                subject="target_feature_consistency",
                                message=(
                                    "-- Same target ids in the interaction and "
                                    "features file"
                                ),
                                value="shared",
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
                    _emit_data_message(
                        verbose,
                        print_mode,
                        progress,
                        subject="target_feature_consistency",
                        message=(
                            "-- Cross input consistency for (triplet) interaction "
                            "data and target features checks out"
                        ),
                        value="triplets",
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
                        _emit_data_message(
                            verbose,
                            print_mode,
                            progress,
                            subject="target_feature_consistency",
                            message=(
                                "-- Same target ids in the interaction and features file"
                            ),
                            value="shared",
                        )
                    else:
                        raise ValueError(
                            ("error: " if print_mode == "dev" else "")
                            + "Different target ids in the interaction and features files"
                        )
            else:  # pragma: no cover
                pass
