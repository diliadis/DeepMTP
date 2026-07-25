from __future__ import annotations

import json
import os
import pickle
import zipfile
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import arff
import numpy as np
import pandas as pd
import requests
import scipy
import wget
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from skmultilearn.dataset import available_data_sets, load_dataset

from DeepMTP._download_utils import download_and_extract_zip
from DeepMTP._split_utils import (
    partition_ordered_ids as _partition_ids,
)
from DeepMTP._split_utils import (
    validate_split_ratio as _validate_split_ratio,
)

ArrayFormat: TypeAlias = Literal["numpy", "dataframe"]
DatasetVariant: TypeAlias = Literal["undivided", "divided"]
PrintMode: TypeAlias = Literal["basic", "dev"]
DatasetValue: TypeAlias = np.ndarray | pd.DataFrame | None

_MTR_ARCHIVE_URL = "https://github.com/diliadis/MTR/blob/main/mtr-datasets.zip?raw=true"
_MTR_DATASET_TARGETS = {
    "atp1d": 6,
    "atp7d": 6,
    "oes97": 16,
    "oes10": 16,
    "rf1": 8,
    "rf2": 8,
    "scm1d": 16,
    "scm20d": 16,
    "edm": 2,
    "sf1": 3,
    "sf2": 3,
    "jura": 3,
    "wq": 14,
    "enb": 2,
    "slump": 3,
    "andro": 6,
    "osales": 12,
    "scpf": 3,
}
_MTL_DATASET_NAMES = frozenset({"bird", "dog"})
_MTL_ARCHIVE_URL_TEMPLATE = (
    "https://github.com/diliadis/MTL/blob/main/{dataset_name}.zip?raw=true"
)


class DatasetSplit(TypedDict):
    """One raw train, validation, or test split."""

    y: DatasetValue
    X_instance: DatasetValue
    X_target: DatasetValue


class DatasetBundle(TypedDict):
    """Stable outer structure returned by legacy dataset helpers."""

    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


class ProcessedDatasetValue(TypedDict):
    """Processed data container consumed by legacy data utilities."""

    data: pd.DataFrame


class ProcessedDatasetSplit(TypedDict):
    """One processed split with optional interactions and side features."""

    y: ProcessedDatasetValue | None
    X_instance: ProcessedDatasetValue | None
    X_target: ProcessedDatasetValue | None


class ProcessedDatasetBundle(TypedDict):
    """Processed train, validation, and test data."""

    train: ProcessedDatasetSplit
    val: ProcessedDatasetSplit
    test: ProcessedDatasetSplit


def _split_ratio(
    ratio: Mapping[str, float] | None,
) -> dict[str, float]:
    if ratio is None:
        return {"train": 0.7, "val": 0.1, "test": 0.2}
    return _validate_split_ratio(
        ratio,
        parameter_name="split ratio",
        require_train=False,
    )


def _optional_split_ratio(
    ratio: Mapping[str, float] | None,
    *,
    parameter_name: str,
) -> dict[str, float] | None:
    """Validate an optional train/test or train/validation/test split."""
    if ratio is None:
        return None
    return _validate_split_ratio(
        ratio,
        parameter_name=parameter_name,
        require_train=False,
        require_validation=False,
    )


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_optional_non_negative_integer(
    name: str,
    value: int | None,
) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer or None.")


def _generate_static_feature_frame(
    num_entities: int,
    num_features: int | None,
) -> pd.DataFrame | None:
    if num_features in {None, 0}:
        return None

    entity_ids = np.arange(num_entities)
    feature_values = np.repeat(
        entity_ids.astype(float)[:, np.newaxis],
        num_features,
        axis=1,
    )
    return pd.DataFrame(
        {
            "id": entity_ids,
            "features": [row.copy() for row in feature_values],
        }
    )


def _copy_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy(deep=True)
    copied["features"] = [
        np.asarray(feature_vector).copy()
        for feature_vector in copied["features"].to_numpy()
    ]
    return copied


def _select_feature_ids(
    frame: pd.DataFrame,
    ids: np.ndarray,
) -> pd.DataFrame:
    return _copy_feature_frame(frame.loc[frame["id"].isin(ids)])


def _random_generator(
    random_state: int | None,
    *,
    parameter_name: str = "random_state",
) -> np.random.Generator:
    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
        or not 0 <= random_state <= np.iinfo(np.uint32).max
    ):
        raise ValueError(
            f"{parameter_name} must be a non-negative 32-bit integer or None."
        )
    return np.random.default_rng(random_state)


def _prepare_dataset_root(path: str | os.PathLike[str]) -> Path:
    try:
        data_root = Path(path)
    except TypeError as exc:
        raise TypeError("path must be a string or path-like object.") from exc
    if data_root.exists() and not data_root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {data_root}.")
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


def _validate_mtl_scores(scores: object, *, source: Path) -> np.ndarray:
    try:
        score_matrix = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MTL scores in {source} must be numeric.") from exc
    if score_matrix.ndim != 2 or score_matrix.shape[0] < 1 or score_matrix.shape[1] < 1:
        raise ValueError(
            f"MTL scores in {source} must be a two-dimensional matrix with "
            f"at least one row and column; received shape {score_matrix.shape}."
        )
    if np.isinf(score_matrix).any():
        raise ValueError(f"MTL scores in {source} cannot contain infinite values.")
    return score_matrix


def _load_mtl_interaction_matrix(input_path: str | os.PathLike[str]) -> np.ndarray:
    source = Path(input_path)
    with source.open(encoding="utf-8") as input_file:
        try:
            data = json.load(input_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"MTL dataset in {source} is not valid JSON.") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"MTL dataset in {source} must contain a JSON object.")
    try:
        raw_labels = data["true_labels"]
        raw_worker_labels = data["WorkerLabels"]
    except KeyError as exc:
        raise ValueError(
            f"MTL dataset in {source} requires true_labels and WorkerLabels."
        ) from exc

    try:
        labels = np.asarray(raw_labels, dtype=float)
        worker_labels = np.asarray(raw_worker_labels, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MTL labels in {source} must be numeric.") from exc
    if labels.ndim != 1 or labels.size < 1 or not np.isfinite(labels).all():
        raise ValueError(
            f"true_labels in {source} must be a finite one-dimensional array "
            "with at least one entry."
        )
    if worker_labels.ndim == 1:
        if worker_labels.size % labels.size != 0:
            raise ValueError(
                f"WorkerLabels in {source} cannot be reshaped to match "
                f"{labels.size} true labels."
            )
        worker_labels = worker_labels.reshape(labels.size, -1)
    if (
        worker_labels.ndim != 2
        or worker_labels.shape[0] != labels.size
        or worker_labels.shape[1] < 1
        or not np.isfinite(worker_labels).all()
    ):
        raise ValueError(
            f"WorkerLabels in {source} must be a finite two-dimensional matrix "
            f"with {labels.size} rows."
        )

    missing = worker_labels == -1
    score_matrix = (worker_labels == labels[:, np.newaxis]).astype(float)
    score_matrix[missing] = np.nan
    return _validate_mtl_scores(score_matrix, source=source)


def load_process_MLC(
    path: str = "./data",
    dataset_name: str = "bibtex",
    variant: DatasetVariant = "undivided",
    features_type: ArrayFormat = "numpy",
    print_mode: PrintMode = "basic",
) -> DatasetBundle:
    """Load multi-label classfication datasets from the mulan repository.

    Args:
        path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
        dataset_name (str, optional): The name of the multi-label classification dataset. Defaults to 'bibtex'.
        variant (str, optional): The version of the dataset. If value set to 'undivided' the whole dataset is returned as a training set. Defaults to 'undivided'.
        features_type (str, optional): The format of the instance features. There are two possible values, numpy and dataframe. This is intended to test the functionality of the data_process function. Defaults to 'numpy'.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    Raises:
        AttributeError: if the dataset_name doesn't exist in the mulan repository

    Returns:
        dict: A dictionary with all the available data for the multi-label classification dataset.
    """
    if features_type not in ["numpy", "dataframe"]:
        raise AttributeError(
            "Please use one of the valid features_type values: "
            + str(["numpy", "dataframe"])
        )
    if variant not in ["undivided", "divided"]:
        raise AttributeError(
            "Please use one of the valid variant values: "
            + str(["undivided", "divided"])
        )

    # The current version just uses the datasets from the skmultilearn library. In the future I will be hosting everything from the university server
    X_train_instance, X_val_instance, X_test_instance = None, None, None
    X_train_target, X_val_target, X_test_target = None, None, None
    y_train, y_val, y_test = None, None, None

    path += "/multi-label_classification"

    print(("info: " if print_mode == "dev" else "") + "Processing...")
    available_datasets = {x[0] for x in available_data_sets()}

    if not os.path.exists(path):  # pragma: no cover
        os.makedirs(path)

    if dataset_name not in available_datasets:
        raise AttributeError(
            "Please use one of the valid dataset names: "
            + str(list(available_data_sets().keys()))
        )
    else:
        if variant == "undivided":
            X_train_instance, y_train, _, _ = load_dataset(
                dataset_name, variant, data_home=path
            )
        else:
            X_train_instance, y_train, _, _ = load_dataset(dataset_name, "train")
            X_test_instance, y_test, _, _ = load_dataset(dataset_name, "test")

    if X_train_instance is not None and scipy.sparse.issparse(
        X_train_instance
    ):  # pragma: no cover
        X_train_instance = X_train_instance.toarray()
    if X_test_instance is not None and scipy.sparse.issparse(
        X_test_instance
    ):  # pragma: no cover
        X_test_instance = X_test_instance.toarray()

    if features_type == "dataframe":
        temp_df_train = pd.DataFrame(np.arange(len(X_train_instance)), columns=["id"])
        temp_df_train["features"] = [r for r in X_train_instance]
        X_train_instance = temp_df_train

        if X_test_instance is not None:
            temp_df_test = pd.DataFrame(np.arange(len(X_test_instance)), columns=["id"])
            temp_df_test["features"] = [r for r in X_test_instance]
            X_test_instance = temp_df_test

    if y_train is not None and scipy.sparse.issparse(y_train):  # pragma: no cover
        y_train = y_train.toarray()

    if y_test is not None and scipy.sparse.issparse(y_test):  # pragma: no cover
        y_test = y_test.toarray()

    print(("info: " if print_mode == "dev" else "") + "Done")

    # return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
    return {
        "train": {
            "y": y_train,
            "X_instance": X_train_instance,
            "X_target": X_train_target,
        },
        "test": {"y": y_test, "X_instance": X_test_instance, "X_target": X_test_target},
        "val": {"y": y_val, "X_instance": X_val_instance, "X_target": X_val_target},
    }


def process_dummy_MLC(
    num_features: int = 10,
    num_instances: int = 50,
    num_targets: int = 5,
    interaction_matrix_format: ArrayFormat = "numpy",
    features_format: ArrayFormat = "numpy",
    random_state: int | None = 42,
) -> DatasetBundle:  # pragma: no cover
    """Generates a dummy multi-label classification dataset

    Args:
        num_features (int, optional): The number of instance features. Defaults to 10.
        num_instances (int, optional): The number of instances. Defaults to 50.
        num_targets (int, optional): The number of targets. Defaults to 5.
        random_state (int, optional): Seed for an isolated NumPy generator. Use
            None for non-deterministic data. Defaults to 42.

    Returns:
        dict: A dictionary with all the available data for the multi-label classification dataset.
    """
    if interaction_matrix_format not in ["numpy", "dataframe"]:
        raise AttributeError(
            "Please use one of the valid interaction_matrix_format values: "
            + str(["numpy", "dataframe"])
        )
    if features_format not in {"numpy", "dataframe"}:
        raise ValueError("features_format must be either 'numpy' or 'dataframe'.")
    _require_positive_integer("num_features", num_features)
    _require_positive_integer("num_instances", num_instances)
    _require_positive_integer("num_targets", num_targets)
    X_train_instance, X_val_instance, X_test_instance = None, None, None
    X_train_target, X_val_target, X_test_target = None, None, None
    y_train, y_val, y_test = None, None, None

    generator = _random_generator(random_state)
    X_train_instance = generator.random((num_instances, num_features))
    for i in range(num_instances):
        X_train_instance[i, 0] = i
    if features_format == "dataframe":
        temp_instance_features_df = pd.DataFrame(
            np.arange(len(X_train_instance)), columns=["id"]
        )
        temp_instance_features_df["features"] = [r for r in X_train_instance]
        X_train_instance = temp_instance_features_df

    y_train = generator.integers(2, size=(num_instances, num_targets))
    if interaction_matrix_format == "dataframe":
        triplets = [
            (i, j, y_train[i, j])
            for i in range(y_train.shape[0])
            for j in range(y_train.shape[1])
        ]
        y_train = pd.DataFrame(triplets, columns=["instance_id", "target_id", "value"])

    # return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
    return {
        "train": {
            "y": y_train,
            "X_instance": X_train_instance,
            "X_target": X_train_target,
        },
        "test": {"y": y_test, "X_instance": X_test_instance, "X_target": X_test_target},
        "val": {"y": y_val, "X_instance": X_val_instance, "X_target": X_val_target},
    }


def load_process_MTR(
    path: str | os.PathLike[str] = "./data",
    dataset_name: str = "enb",
    features_type: ArrayFormat = "numpy",
    print_mode: PrintMode = "basic",
) -> DatasetBundle:
    """Load multivariate regression datasets from the mulan repository.

    Args:
        path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
        dataset_name (str, optional): The name of the multivariate regression dataset. Defaults to 'enb'.
        features_type (str, optional): The format of the instance features. There are two possible values, numpy and dataframe. This is intended to test the functionality of the data_process function. Defaults to 'numpy'.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    Raises:
        ValueError: If a public option or downloaded dataset is invalid.
        DatasetDownloadError: If the dataset archive cannot be downloaded safely.
        FileNotFoundError: If the archive does not contain the requested dataset.

    Returns:
        dict: A dictionary with all the available data for the multivariate regression dataset.
    """
    if features_type not in {"numpy", "dataframe"}:
        raise ValueError("features_type must be either 'numpy' or 'dataframe'.")
    if print_mode not in {"basic", "dev"}:
        raise ValueError("print_mode must be either 'basic' or 'dev'.")
    if dataset_name not in _MTR_DATASET_TARGETS:
        raise ValueError(f"dataset_name must be one of {sorted(_MTR_DATASET_TARGETS)}.")
    data_root = _prepare_dataset_root(path)

    # The current version downloads all the datasets from a personal github repo
    X_train_instance, X_val_instance, X_test_instance = None, None, None
    X_train_target, X_val_target, X_test_target = None, None, None
    y_train, y_val, y_test = None, None, None

    print(("info: " if print_mode == "dev" else "") + "Processing...")

    """
    +--------+------------+--------------------+----------+
    |  name  | #instances | #instance_features | #targets |
    +--------+------------+--------------------+----------+
    | atp1d  |    337     |        411         |    6     |
    | atp7d  |    296     |        411         |    6     |
    | oes97  |    334     |        263         |    16    |
    | oes10  |    403     |        298         |    16    |
    |  rf1   |    9125    |         64         |    8     |
    |  rf2   |    9125    |        576         |    8     |
    | scm1d  |    9803    |        280         |    16    |
    | scm20d |    8966    |         61         |    16    |
    |  edm   |    154     |         16         |    2     |
    |  sf1   |    323     |         10         |    3     |
    |  sf2   |    1066    |         10         |    3     |
    |  jura  |    359     |         15         |    3     |
    |   wq   |    1060    |         16         |    14    |
    |  enb   |    768     |         8          |    2     |
    | slump  |    103     |         7          |    3     |
    | andro  |     49     |         30         |    6     |
    | osales |    639     |        413         |    12    |
    |  scfp  |    1137    |         23         |    3     |
    +--------+------------+--------------------+----------+
    """

    dataset_path = data_root / "mtr-datasets" / f"{dataset_name}.arff"
    if not dataset_path.is_file():  # pragma: no cover
        print(
            ("info: " if print_mode == "dev" else "")
            + "Downloading and extracting dataset from scratch... ",
            end="",
        )
        download_and_extract_zip(_MTR_ARCHIVE_URL, data_root)
        print(("info: " if print_mode == "dev" else "") + "Done")
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset archive did not contain the expected file: {dataset_path}."
        )

    with dataset_path.open(encoding="utf-8") as dataset_file:
        dataset = arff.load(dataset_file)
    data = np.array(dataset["data"])
    num_targets = _MTR_DATASET_TARGETS[dataset_name]
    if data.ndim != 2 or data.shape[1] <= num_targets:
        raise ValueError(f"Dataset file has an invalid shape: {data.shape}.")

    X_train_instance = np.array(data[:, :-num_targets])
    if features_type == "dataframe":
        temp_df = pd.DataFrame(np.arange(len(X_train_instance)), columns=["id"])
        temp_df["features"] = [r for r in X_train_instance]
        X_train_instance = temp_df

    y_train = np.array(data[:, -num_targets:])

    print(("info: " if print_mode == "dev" else "") + "Done")

    # return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
    return {
        "train": {
            "y": y_train,
            "X_instance": X_train_instance,
            "X_target": X_train_target,
        },
        "test": {"y": y_test, "X_instance": X_test_instance, "X_target": X_test_target},
        "val": {"y": y_val, "X_instance": X_val_instance, "X_target": X_val_target},
    }


def process_dummy_MTR(
    num_features: int = 10,
    num_instances: int = 50,
    num_targets: int = 5,
    interaction_matrix_format: ArrayFormat = "numpy",
    features_format: ArrayFormat = "numpy",
    variant: DatasetVariant = "undivided",
    split_ratio: Mapping[str, float] | None = None,
    random_state: int | None = 42,
) -> DatasetBundle:  # pragma: no cover
    """Generates a dummy multivariate regression dataset

    Args:
        num_features (int, optional): The number of instance features. Defaults to 10.
        num_instances (int, optional): The number of instances. Defaults to 50.
        num_targets (int, optional): The number of targets. Defaults to 5.
        random_state (int, optional): Seed used for both data generation and
            splitting. Use None for non-deterministic data. Defaults to 42.

    Returns:
        dict: A dictionary with all the available data for the multivariate regression dataset.
    """
    split_ratio = _split_ratio(split_ratio)
    if interaction_matrix_format not in ["numpy", "dataframe"]:
        raise AttributeError(
            "Please use one of the valid interaction_matrix_format values: "
            + str(["numpy", "dataframe"])
        )
    if features_format not in {"numpy", "dataframe"}:
        raise ValueError("features_format must be either 'numpy' or 'dataframe'.")
    if variant not in {"undivided", "divided"}:
        raise ValueError("variant must be either 'undivided' or 'divided'.")
    _require_positive_integer("num_features", num_features)
    _require_positive_integer("num_instances", num_instances)
    _require_positive_integer("num_targets", num_targets)

    X_train_instance, X_val_instance, X_test_instance = None, None, None
    X_train_target, X_val_target, X_test_target = None, None, None
    y_train, y_val, y_test = None, None, None

    generator = _random_generator(random_state)
    split_random_state = (
        random_state
        if random_state is not None
        else int(generator.integers(0, np.iinfo(np.uint32).max))
    )
    X_train_instance = generator.random((num_instances, num_features))
    for i in range(num_instances):
        X_train_instance[i, 0] = i

    y_train = generator.integers(10, size=(num_instances, num_targets))

    if features_format == "dataframe":
        temp_instance_features_df = pd.DataFrame(
            np.arange(len(X_train_instance)), columns=["id"]
        )
        temp_instance_features_df["features"] = [r for r in X_train_instance]
        X_train_instance = temp_instance_features_df
    if interaction_matrix_format == "dataframe":
        triplets = [
            (i, j, y_train[i, j])
            for i in range(y_train.shape[0])
            for j in range(y_train.shape[1])
        ]
        y_train = pd.DataFrame(triplets, columns=["instance_id", "target_id", "value"])

    if variant == "divided":
        train_ids, test_ids = train_test_split(
            range(num_instances),
            test_size=split_ratio["test"],
            random_state=split_random_state,
        )
        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=split_ratio["val"] / (1 - split_ratio["test"]),
            random_state=split_random_state,
        )

        if interaction_matrix_format == "dataframe":
            y_test = y_train.loc[y_train["instance_id"].isin(test_ids)]
            y_val = y_train.loc[y_train["instance_id"].isin(val_ids)]
            y_train = y_train.loc[y_train["instance_id"].isin(train_ids)]
        else:
            y_test = y_train[test_ids, :]
            y_val = y_train[val_ids, :]
            y_train = y_train[train_ids, :]

        if features_format == "dataframe":
            X_test_instance = X_train_instance.loc[
                X_train_instance["id"].isin(test_ids)
            ]
            X_val_instance = X_train_instance.loc[X_train_instance["id"].isin(val_ids)]
            X_train_instance = X_train_instance.loc[
                X_train_instance["id"].isin(train_ids)
            ]
        else:
            X_test_instance = X_train_instance[test_ids, :]
            X_val_instance = X_train_instance[val_ids, :]
            X_train_instance = X_train_instance[train_ids, :]

    # return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
    return {
        "train": {
            "y": y_train,
            "X_instance": X_train_instance,
            "X_target": X_train_target,
        },
        "test": {"y": y_test, "X_instance": X_test_instance, "X_target": X_test_target},
        "val": {"y": y_val, "X_instance": X_val_instance, "X_target": X_val_target},
    }


def print_MTR_datasets() -> None:  # pragma: no cover
    """Prints a table with the basic info for every multivariate regression dataset available in the Mulan repository"""
    table = PrettyTable(["name", "#instances", "#instance_features", "#targets"])
    table.add_row(["atp1d", "337", "411", "6"])
    table.add_row(["atp7d", "296", "411", "6"])
    table.add_row(["oes97", "334", "263", "16"])
    table.add_row(["oes10", "403", "298", "16"])
    table.add_row(["rf1", "9125", "64", "8"])
    table.add_row(["rf2", "9125", "576", "8"])
    table.add_row(["scm1d", "9803", "280", "16"])
    table.add_row(["scm20d", "8966", "61", "16"])
    table.add_row(["edm", "154", "16", "2"])
    table.add_row(["sf1", "323", "10", "3"])
    table.add_row(["sf2", "1066", "10", "3"])
    table.add_row(["jura", "359", "15", "3"])
    table.add_row(["wq", "1060", "16", "14"])
    table.add_row(["enb", "768", "8", "2"])
    table.add_row(["slump", "103", "7", "3"])
    table.add_row(["andro", "49", "30", "6"])
    table.add_row(["osales", "639", "413", "12"])
    table.add_row(["scpf", "1137", "23", "3"])
    print(table.get_string())


def load_process_DP(
    path: str = "./data",
    dataset_name: str = "ern",
    variant: DatasetVariant = "undivided",
    random_state: int = 42,
    split_ratio: Mapping[str, float] | None = None,
    split_instance_features: bool = False,
    split_target_features: bool = False,
    validation_setting: str = "B",
    print_mode: PrintMode = "basic",
) -> DatasetBundle:
    """Load dyadic prediction datasets from the following repository: https://people.montefiore.uliege.be.

    Args:
        path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
        dataset_name (str, optional): The name of the dyadic prediction dataset. Defaults to 'ern'.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    Raises:
        AttributeError: if the dataset_name doesn't exist in the repository

    Returns:
        dict: A dictionary with all the available data for the dyadic prediction dataset.
    """
    split_ratio = _split_ratio(split_ratio)
    if variant not in ["undivided", "divided"]:
        raise AttributeError(
            "Please use one of the valid variant values: "
            + str(["undivided", "divided"])
        )

    if validation_setting.upper() not in ["B", "C", "D"]:
        raise AttributeError(
            "Invalid validation setting symbol prodived: "
            + str(validation_setting.upper())
            + ". Choose one of the following: "
            + str(["B", "C", "D"])
        )
    validation_setting = validation_setting.upper()

    # The current version downloads all the datasets from https://people.montefiore.uliege.be
    X_train_instance, X_val_instance, X_test_instance = None, None, None
    X_train_target, X_val_target, X_test_target = None, None, None
    y_train, y_val, y_test = None, None, None

    print(("info: " if print_mode == "dev" else "") + "Processing...")

    available_data_sets = ["ern", "srn", "dpie", "dpii"]

    if dataset_name.startswith("dp"):  # pragma: no cover
        url = (
            "https://people.montefiore.uliege.be/schrynemackers/dpix/"
            + dataset_name
            + "_"
        )
    else:
        url = "https://people.montefiore.uliege.be/schrynemackers/" + dataset_name + "/"

    if dataset_name not in available_data_sets:
        raise AttributeError(
            "Please use one of the valid dataset names: " + str(available_data_sets)
        )

    else:
        if not os.path.exists(
            path + "/dyadic_prediction-datasets/" + dataset_name
        ):  # pragma: no cover
            os.makedirs(path + "/dyadic_prediction-datasets/" + dataset_name)
            print(
                ("info: " if print_mode == "dev" else "")
                + "Downloading dataset from scratch... ",
                end="",
            )
            wget.download(
                url + "Y.txt",
                path + "/dyadic_prediction-datasets/" + dataset_name + "/",
            )
            wget.download(
                url + "X1.txt",
                path + "/dyadic_prediction-datasets/" + dataset_name + "/",
            )
            wget.download(
                url + "X2.txt",
                path + "/dyadic_prediction-datasets/" + dataset_name + "/",
            )
            print(("info: " if print_mode == "dev" else "") + "Done")

        if (dataset_name == "ern") or (dataset_name == "srn"):
            y_train = np.genfromtxt(
                path + "/dyadic_prediction-datasets/" + dataset_name + "/Y.txt",
                delimiter=",",
            )
            X_train_instance = np.genfromtxt(
                path + "/dyadic_prediction-datasets/" + dataset_name + "/X1.txt",
                delimiter=",",
            )
            X_train_target = np.genfromtxt(
                path + "/dyadic_prediction-datasets/" + dataset_name + "/X2.txt",
                delimiter=",",
            )
        else:
            y_train = np.genfromtxt(
                path
                + "/dyadic_prediction-datasets/"
                + dataset_name
                + "/"
                + dataset_name
                + "_Y.txt"
            )
            X_train_instance = np.genfromtxt(
                path
                + "/dyadic_prediction-datasets/"
                + dataset_name
                + "/"
                + dataset_name
                + "_X1.txt"
            )
            X_train_target = np.genfromtxt(
                path
                + "/dyadic_prediction-datasets/"
                + dataset_name
                + "/"
                + dataset_name
                + "_X2.txt"
            )

        if variant != "undivided":
            if validation_setting == "B":
                if split_instance_features:
                    (
                        X_train_instance,
                        X_test_instance,
                        y_train,
                        y_test,
                    ) = train_test_split(
                        X_train_instance,
                        y_train,
                        test_size=split_ratio["test"],
                        random_state=random_state,
                    )
                    (
                        X_train_instance,
                        X_val_instance,
                        y_train,
                        y_val,
                    ) = train_test_split(
                        X_train_instance,
                        y_train,
                        test_size=split_ratio["val"] / (1 - split_ratio["test"]),
                        random_state=random_state,
                    )

                else:
                    (
                        y_train,
                        y_test,
                    ) = train_test_split(
                        y_train,
                        test_size=split_ratio["test"],
                        random_state=random_state,
                    )
                    (
                        y_train,
                        y_val,
                    ) = train_test_split(
                        y_train,
                        test_size=split_ratio["val"] / (1 - split_ratio["test"]),
                        random_state=random_state,
                    )

            elif validation_setting == "C":
                (
                    y_train_target_ids,
                    y_test_target_ids,
                ) = train_test_split(
                    range(y_train.shape[1]),
                    test_size=split_ratio["test"],
                    random_state=random_state,
                )
                (
                    y_train_target_ids,
                    y_val_target_ids,
                ) = train_test_split(
                    y_train_target_ids,
                    test_size=split_ratio["val"] / (1 - split_ratio["test"]),
                    random_state=random_state,
                )
                y_val = y_train[:, y_val_target_ids]
                y_test = y_train[:, y_test_target_ids]
                y_train = y_train[:, y_train_target_ids]

                if split_target_features:
                    X_val_target = X_train_target[y_val_target_ids, :]
                    X_test_target = X_train_target[y_test_target_ids, :]
                    X_train_target = X_train_target[y_train_target_ids, :]

            else:  # validation_setting == 'D'
                (
                    y_train_instance_ids,
                    y_test_instance_ids,
                ) = train_test_split(
                    range(y_train.shape[0]),
                    test_size=split_ratio["test"],
                    random_state=random_state,
                )
                (
                    y_train_instance_ids,
                    y_val_instance_ids,
                ) = train_test_split(
                    y_train_instance_ids,
                    test_size=split_ratio["val"] / (1 - split_ratio["test"]),
                    random_state=random_state,
                )
                (
                    y_train_target_ids,
                    y_test_target_ids,
                ) = train_test_split(
                    range(y_train.shape[1]),
                    test_size=split_ratio["test"],
                    random_state=random_state,
                )
                (
                    y_train_target_ids,
                    y_val_target_ids,
                ) = train_test_split(
                    y_train_target_ids,
                    test_size=split_ratio["val"] / (1 - split_ratio["test"]),
                    random_state=random_state,
                )

                y_val = y_train[y_val_instance_ids, :][:, y_val_target_ids]
                y_test = y_train[y_test_instance_ids, :][:, y_test_target_ids]
                y_train = y_train[y_train_instance_ids, :][:, y_train_target_ids]

                if split_instance_features:
                    X_val_instance = X_train_instance[y_val_instance_ids, :]
                    X_test_instance = X_train_instance[y_test_instance_ids, :]
                    X_train_instance = X_train_instance[y_train_instance_ids, :]

                if split_target_features:
                    X_val_target = X_train_target[y_val_target_ids, :]
                    X_test_target = X_train_target[y_test_target_ids, :]
                    X_train_target = X_train_target[y_train_target_ids, :]

    print(("info: " if print_mode == "dev" else "") + "Done")

    # return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
    return {
        "train": {
            "y": y_train,
            "X_instance": X_train_instance,
            "X_target": X_train_target,
        },
        "test": {"y": y_test, "X_instance": X_test_instance, "X_target": X_test_target},
        "val": {"y": y_val, "X_instance": X_val_instance, "X_target": X_val_target},
    }


def process_dummy_DP(
    num_instance_features: int = 10,
    num_target_features: int = 3,
    num_instances: int = 50,
    num_targets: int = 5,
    interaction_matrix_format: ArrayFormat = "numpy",
    instance_features_format: ArrayFormat = "numpy",
    target_features_format: ArrayFormat = "numpy",
    random_state: int | None = 42,
) -> DatasetBundle:  # pragma: no cover
    """Generates a dummy multivariate regression dataset

    Args:
        num_instance_features (int, optional): The number of instance features. Defaults to 10.
        num_target_features (int, optional): The number of target features. Defaults to 3.
        num_instances (int, optional): The number of instances. Defaults to 50.
        num_targets (int, optional): The number of targets. Defaults to 5.
        random_state (int, optional): Seed for an isolated NumPy generator. Use
            None for non-deterministic data. Defaults to 42.

    Returns:
        dict: A dictionary with all the available data for the dyadic prediction dataset.
    """
    X_train_instance, X_val_instance, X_test_instance = None, None, None
    X_train_target, X_val_target, X_test_target = None, None, None
    y_train, y_val, y_test = None, None, None
    for name, value in {
        "interaction_matrix_format": interaction_matrix_format,
        "instance_features_format": instance_features_format,
        "target_features_format": target_features_format,
    }.items():
        if value not in {"numpy", "dataframe"}:
            raise ValueError(f"{name} must be either 'numpy' or 'dataframe'.")
    _require_positive_integer("num_instance_features", num_instance_features)
    _require_positive_integer("num_target_features", num_target_features)
    _require_positive_integer("num_instances", num_instances)
    _require_positive_integer("num_targets", num_targets)

    generator = _random_generator(random_state)
    X_train_instance = generator.random((num_instances, num_instance_features))
    for i in range(num_instances):
        X_train_instance[i, 0] = i
    if instance_features_format == "dataframe":
        temp_instance_features_df = pd.DataFrame(
            np.arange(len(X_train_instance)), columns=["id"]
        )
        temp_instance_features_df["features"] = [r for r in X_train_instance]
        X_train_instance = temp_instance_features_df

    X_train_target = generator.random((num_targets, num_target_features))
    for i in range(num_targets):
        X_train_target[i, 0] = i
    if target_features_format == "dataframe":
        temp_target_features_df = pd.DataFrame(
            np.arange(len(X_train_target)), columns=["id"]
        )
        temp_target_features_df["features"] = [r for r in X_train_target]
        X_train_target = temp_target_features_df

    y_train = generator.integers(10, size=(num_instances, num_targets))
    if interaction_matrix_format == "dataframe":
        triplets = [
            (i, j, y_train[i, j])
            for i in range(y_train.shape[0])
            for j in range(y_train.shape[1])
        ]
        y_train = pd.DataFrame(triplets, columns=["instance_id", "target_id", "value"])

    # return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
    return {
        "train": {
            "y": y_train,
            "X_instance": X_train_instance,
            "X_target": X_train_target,
        },
        "test": {"y": y_test, "X_instance": X_test_instance, "X_target": X_test_target},
        "val": {"y": y_val, "X_instance": X_val_instance, "X_target": X_val_target},
    }


def load_process_MC(
    path: str = "./data",
    dataset_name: str = "ml-100k",
    print_mode: PrintMode = "basic",
) -> DatasetBundle:
    """Load matrix completion datasets from the Movielens repository.

    Args:
        path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
        dataset_name (str, optional): The name of the matrix completion dataset. Defaults to 'ml-100k'.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    Raises:
        AttributeError: if the dataset_name doesn't exist in the repository

    Returns:
        dict: A dictionary with all the available data for the matrix completion dataset.
    """
    X_train_instance, X_val_instance, X_test_instance = None, None, None
    X_train_target, X_val_target, X_test_target = None, None, None
    y_train, y_val, y_test = None, None, None

    path += "/matrix-completion-datasets"
    urls_checksums_per_dataset = {
        "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        # 'ml-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        # 'ml-10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
    }

    if dataset_name not in urls_checksums_per_dataset:  # pragma: no cover
        raise AttributeError(
            "Please use one of the valid dataset names: "
            + str(list(urls_checksums_per_dataset.keys()))
        )

    url = urls_checksums_per_dataset[dataset_name]

    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, url.split("/")[-1])
    # print('fname: '+str(fname))

    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    # print('base_dir: '+str(base_dir))
    # print('data_dir: '+str(data_dir))
    # print('ext: '+str(ext))

    if not os.path.exists(data_dir):  # pragma: no cover
        print(
            ("info: " if print_mode == "dev" else "")
            + "Downloading dataset "
            + dataset_name
            + " from scratch... ",
            end="",
        )
        r = requests.get(url, stream=True, verify=True)
        with open(fname, "wb") as f:
            f.write(r.content)
        print(("info: " if print_mode == "dev" else "") + "Done")

    fp = zipfile.ZipFile(fname, "r")
    fp.extractall(base_dir)

    names = ["instance_id", "target_id", "value", "timestamp"]
    y_train = pd.read_csv(
        filepath_or_buffer=os.path.join(data_dir, "u.data"),
        sep="\t",
        names=names,
        engine="python",
    )

    y_train.drop(["timestamp"], axis=1, inplace=True)
    unique_instance_ids = y_train["instance_id"].unique()
    unique_target_ids = y_train["target_id"].unique()

    old_to_new_instance_ids_map = {
        old_id: new_id for new_id, old_id in enumerate(unique_instance_ids)
    }
    old_to_new_target_ids_map = {
        old_id: new_id for new_id, old_id in enumerate(unique_target_ids)
    }

    y_train["instance_id"] = y_train["instance_id"].map(old_to_new_instance_ids_map)
    y_train["target_id"] = y_train["target_id"].map(old_to_new_target_ids_map)

    # return {'X_train_instance': X_train_instance, 'X_train_target' :X_train_target, 'y_train' :y_train, 'X_test_instance' :X_test_instance, 'X_test_target' :X_test_target, 'y_test' :y_test, 'X_val_instance' :X_val_instance, 'X_val_target' :X_val_target, 'y_val' :y_val}
    return {
        "train": {
            "y": y_train,
            "X_instance": X_train_instance,
            "X_target": X_train_target,
        },
        "test": {"y": y_test, "X_instance": X_test_instance, "X_target": X_test_target},
        "val": {"y": y_val, "X_instance": X_val_instance, "X_target": X_val_target},
    }


def load_process_MTL(
    path: str | os.PathLike[str] = "./data",
    dataset_name: str = "dog",
    print_mode: PrintMode = "basic",
) -> DatasetBundle:
    """Load multi-task learning datasets from my custom repository.

    Args:
        path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
        dataset_name (str, optional): The name of the multi-task learning dataset. Defaults to 'dog'.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.

    Raises:
        ValueError: If a public option or dataset file is invalid.
        DatasetDownloadError: If the dataset archive cannot be downloaded safely.
        FileNotFoundError: If required score or image files are missing.

    Returns:
        dict: A dictionary with all the available data for the multi-task learning dataset.
    """
    X_train_instance, X_val_instance, X_test_instance = None, None, None
    X_train_target, X_val_target, X_test_target = None, None, None
    y_train, y_val, y_test = None, None, None

    if dataset_name not in _MTL_DATASET_NAMES:
        raise ValueError(f"dataset_name must be one of {sorted(_MTL_DATASET_NAMES)}.")
    if print_mode not in {"basic", "dev"}:
        raise ValueError("print_mode must be either 'basic' or 'dev'.")

    data_root = _prepare_dataset_root(path)
    base_dir = data_root / "multi-task_learning-datasets"
    data_dir = base_dir / dataset_name
    json_path = data_dir / "data.json"
    legacy_scores_path = data_dir / "y_dog.pkl"

    if not json_path.is_file() and not legacy_scores_path.is_file():  # pragma: no cover
        print(
            ("info: " if print_mode == "dev" else "")
            + "Downloading dataset "
            + dataset_name
            + " from scratch... ",
            end="",
        )
        download_and_extract_zip(
            _MTL_ARCHIVE_URL_TEMPLATE.format(dataset_name=dataset_name),
            base_dir,
        )
        print(("info: " if print_mode == "dev" else "") + "Done")

    if json_path.is_file():
        scores_matrix = _load_mtl_interaction_matrix(json_path)
    elif legacy_scores_path.is_file():
        # Compatibility with caches created by older DeepMTP releases.
        with legacy_scores_path.open("rb") as scores_file:
            scores_matrix = _validate_mtl_scores(
                pickle.load(scores_file),
                source=legacy_scores_path,
            )
    else:
        raise FileNotFoundError(
            f"Dataset archive did not contain the expected file: {json_path}."
        )
    if scores_matrix.shape[0] < 2:
        raise ValueError("MTL datasets require at least two instances to split.")

    image_dir = data_dir / "image"
    instance_ids = np.array(
        [str(image_dir / f"{index + 1}.jpg") for index in range(scores_matrix.shape[0])]
    )
    missing_images = [
        image_path for image_path in instance_ids if not Path(image_path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"MTL dataset image is missing: {missing_images[0]}.")

    train_ids, test_ids = train_test_split(
        np.arange(scores_matrix.shape[0]), test_size=0.25, shuffle=True, random_state=42
    )

    train_arr = [
        (i, j, scores_matrix[i, j])
        for i in train_ids
        for j in range(scores_matrix.shape[1])
        if not np.isnan(scores_matrix[i, j])
    ]
    test_arr = [
        (i, j, scores_matrix[i, j])
        for i in test_ids
        for j in range(scores_matrix.shape[1])
        if not np.isnan(scores_matrix[i, j])
    ]

    y_train = pd.DataFrame(train_arr, columns=["instance_id", "target_id", "value"])
    y_test = pd.DataFrame(test_arr, columns=["instance_id", "target_id", "value"])

    X_train_instance = pd.DataFrame(
        [(i, instance_ids[i]) for i in train_ids], columns=["id", "dir"]
    )
    X_test_instance = pd.DataFrame(
        [(i, instance_ids[i]) for i in test_ids], columns=["id", "dir"]
    )

    return {
        "train": {
            "y": y_train,
            "X_instance": X_train_instance,
            "X_target": X_train_target,
        },
        "test": {"y": y_test, "X_instance": X_test_instance, "X_target": X_test_target},
        "val": {"y": y_val, "X_instance": X_val_instance, "X_target": X_val_target},
    }


def generate_interaction_matrix(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> None:
    """Convert annotator labels into a binary correctness matrix."""
    y = _load_mtl_interaction_matrix(input_path)
    with open(output_path, "wb") as output_file:
        pickle.dump(y, output_file)


def generate_dummy_dataset(
    num_instances: int,
    num_targets: int,
    num_instance_features: int,
    num_target_features: int,
    error_mu: float,
    error_sigma: float,
    sklearn_version: bool = False,
    seed: int = 42,
    mode: str = "u+logv",
    split_ratio: Mapping[str, float] | None = None,
) -> DatasetBundle:  # pragma: no cover
    """Generate a reproducible dyadic regression dataset.

    ``sklearn_version`` starts from normally distributed features. For modes
    involving logarithms or fractional powers, the relevant features are
    transformed to a positive domain so every generated score remains real.
    """
    split_ratio = _split_ratio(split_ratio)
    _require_positive_integer("num_instances", num_instances)
    _require_positive_integer("num_targets", num_targets)
    _require_positive_integer("num_instance_features", num_instance_features)
    _require_positive_integer("num_target_features", num_target_features)
    valid_modes = {
        "u+logv",
        "u*logv",
        "u^v",
        "u^v_reversed",
        "max(u,v)",
        "u*v",
    }
    if not isinstance(mode, str) or mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}.")
    if not isinstance(sklearn_version, bool):
        raise TypeError("sklearn_version must be a boolean.")
    for parameter_name, value in {
        "error_mu": error_mu,
        "error_sigma": error_sigma,
    }.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{parameter_name} must be a finite real number.")
        if not np.isfinite(value):
            raise ValueError(f"{parameter_name} must be finite.")
    if error_sigma < 0:
        raise ValueError("error_sigma must be non-negative.")
    error_mu = float(error_mu)
    error_sigma = float(error_sigma)
    if num_instance_features != num_target_features:
        raise ValueError(
            "num_instance_features and num_target_features must match for "
            f"mode {mode!r}."
        )
    X_train_instance, X_val_instance, X_test_instance = None, None, None
    X_train_target, X_val_target, X_test_target = None, None, None
    y_train, y_val, y_test = None, None, None

    # generate the instance and target feature numpy arrays
    generator = _random_generator(seed, parameter_name="seed")
    if sklearn_version:
        # inspired by sklearn
        X_train_instance = generator.standard_normal(
            size=(num_instances, num_instance_features)
        )
        X_train_target = generator.standard_normal(
            size=(num_targets, num_target_features)
        )
        if mode in {"u+logv", "u*logv"}:
            X_train_target = np.exp(X_train_target)
        elif mode in {"u^v", "u^v_reversed"}:
            X_train_instance = np.exp(X_train_instance)
    else:
        X_train_instance = generator.random((num_instances, num_instance_features))
        X_train_target = generator.random((num_targets, num_target_features))
    if mode in {"u+logv", "u*logv"}:
        X_train_target = np.maximum(X_train_target, np.finfo(float).tiny)

    # generate the score matrix using some kind of relationship between the instance and target features
    y_train = np.zeros((num_instances, num_targets))
    if mode == "u+logv":
        for i in range(num_instances):
            for j in range(num_targets):
                y_train[i, j] = np.sum(
                    X_train_instance[i] + np.log(X_train_target[j])
                ) + generator.normal(error_mu, error_sigma)
    elif mode == "u*logv":
        for i in range(num_instances):
            for j in range(num_targets):
                y_train[i, j] = np.sum(
                    X_train_instance[i] * np.log(X_train_target[j])
                ) + generator.normal(error_mu, error_sigma)
    elif mode == "u^v":
        for i in range(num_instances):
            for j in range(num_targets):
                y_train[i, j] = np.sum(
                    X_train_instance[i] ** X_train_target[j]
                ) + generator.normal(error_mu, error_sigma)
    elif mode == "u^v_reversed":
        for i in range(num_instances):
            for j in range(num_targets):
                y_train[i, j] = np.sum(
                    X_train_instance[i] ** X_train_target[j][::-1]
                ) + generator.normal(error_mu, error_sigma)
    elif mode == "max(u,v)":
        for i in range(num_instances):
            for j in range(num_targets):
                y_train[i, j] = np.sum(
                    [
                        *map(
                            max,
                            zip(X_train_instance[i], X_train_target[j], strict=True),
                        )
                    ]
                )
    elif mode == "u*v":
        for i in range(num_instances):
            for j in range(num_targets):
                y_train[i, j] = np.sum(X_train_instance[i] * X_train_target[j])
    if not np.isfinite(y_train).all():
        raise ValueError(
            "Synthetic score generation produced non-finite values; "
            "use less extreme feature or noise parameters."
        )

    (
        X_train_instance,
        X_test_instance,
        y_train,
        y_test,
    ) = train_test_split(
        X_train_instance, y_train, test_size=split_ratio["test"], random_state=seed
    )
    (
        X_train_instance,
        X_val_instance,
        y_train,
        y_val,
    ) = train_test_split(
        X_train_instance,
        y_train,
        test_size=split_ratio["val"] / (1 - split_ratio["test"]),
        random_state=seed,
    )

    return {
        "train": {
            "y": y_train,
            "X_instance": X_train_instance,
            "X_target": X_train_target,
        },
        "test": {"y": y_test, "X_instance": X_test_instance, "X_target": X_test_target},
        "val": {"y": y_val, "X_instance": X_val_instance, "X_target": X_val_target},
    }


def generate_MTP_dataset(
    num_instances: int,
    num_targets: int,
    num_instance_features: int | None = None,
    num_target_features: int | None = None,
    split_instances: Mapping[str, float] | None = None,
    split_targets: Mapping[str, float] | None = None,
    return_static_features_data: bool = False,
) -> ProcessedDatasetBundle:  # pragma: no cover
    """Generate deterministic triplet data with optional axis-level splits.

    Each split mapping requires a test ratio and may include a validation
    ratio. The train ratio is derived when omitted and verified when supplied.
    """
    _require_positive_integer("num_instances", num_instances)
    _require_positive_integer("num_targets", num_targets)
    _require_optional_non_negative_integer(
        "num_instance_features",
        num_instance_features,
    )
    _require_optional_non_negative_integer(
        "num_target_features",
        num_target_features,
    )
    if not isinstance(return_static_features_data, bool):
        raise TypeError("return_static_features_data must be a boolean.")

    X_train_instance = _generate_static_feature_frame(
        num_instances,
        num_instance_features,
    )
    X_val_instance: pd.DataFrame | None = None
    X_test_instance: pd.DataFrame | None = None
    X_train_target = _generate_static_feature_frame(
        num_targets,
        num_target_features,
    )
    X_val_target: pd.DataFrame | None = None
    X_test_target: pd.DataFrame | None = None
    y_val: pd.DataFrame | None = None
    y_test: pd.DataFrame | None = None
    split_instances = _optional_split_ratio(
        split_instances,
        parameter_name="split_instances",
    )
    split_targets = _optional_split_ratio(
        split_targets,
        parameter_name="split_targets",
    )
    train_instance_ids, val_instance_ids, test_instance_ids = None, None, None
    train_target_ids, val_target_ids, test_target_ids = None, None, None

    instance_ids = np.repeat(np.arange(num_instances), num_targets)
    target_ids = np.tile(np.arange(num_targets), num_instances)
    y_train = pd.DataFrame(
        {
            "instance_id": instance_ids,
            "target_id": target_ids,
            "value": instance_ids.astype(float),
        }
    )

    if split_instances is not None:
        train_instance_ids, val_instance_ids, test_instance_ids = _partition_ids(
            y_train["instance_id"].unique(),
            split_instances,
            parameter_name="split_instances",
        )

    if split_targets is not None:
        train_target_ids, val_target_ids, test_target_ids = _partition_ids(
            y_train["target_id"].unique(),
            split_targets,
            parameter_name="split_targets",
        )

    if val_instance_ids is not None:
        if X_train_instance is not None:
            X_val_instance = _select_feature_ids(
                X_train_instance,
                val_instance_ids,
            )
        y_val = y_train.loc[y_train["instance_id"].isin(val_instance_ids)].copy()
    elif X_train_instance is not None and return_static_features_data:
        X_val_instance = _copy_feature_frame(X_train_instance)

    if test_instance_ids is not None:
        if X_train_instance is not None:
            X_test_instance = _select_feature_ids(
                X_train_instance,
                test_instance_ids,
            )
        y_test = y_train.loc[y_train["instance_id"].isin(test_instance_ids)].copy()
    elif X_train_instance is not None and return_static_features_data:
        X_test_instance = _copy_feature_frame(X_train_instance)

    if train_instance_ids is not None:
        if X_train_instance is not None:
            X_train_instance = _select_feature_ids(
                X_train_instance,
                train_instance_ids,
            )
        y_train = y_train.loc[y_train["instance_id"].isin(train_instance_ids)].copy()

    if val_target_ids is not None:
        if X_train_target is not None:
            X_val_target = _select_feature_ids(X_train_target, val_target_ids)
        if y_val is not None:
            y_val = y_val.loc[y_val["target_id"].isin(val_target_ids)].copy()
        else:
            y_val = y_train.loc[y_train["target_id"].isin(val_target_ids)].copy()
    elif X_train_target is not None and return_static_features_data:
        X_val_target = _copy_feature_frame(X_train_target)

    if test_target_ids is not None:
        if X_train_target is not None:
            X_test_target = _select_feature_ids(X_train_target, test_target_ids)
        if y_test is not None:
            y_test = y_test.loc[y_test["target_id"].isin(test_target_ids)].copy()
        else:
            y_test = y_train.loc[y_train["target_id"].isin(test_target_ids)].copy()
    elif X_train_target is not None and return_static_features_data:
        X_test_target = _copy_feature_frame(X_train_target)

    if train_target_ids is not None:
        if X_train_target is not None:
            X_train_target = _select_feature_ids(X_train_target, train_target_ids)
        y_train = y_train.loc[y_train["target_id"].isin(train_target_ids)].copy()

    return {
        "train": {
            "y": {"data": y_train} if y_train is not None else None,
            "X_instance": {"data": X_train_instance}
            if X_train_instance is not None
            else None,
            "X_target": {"data": X_train_target}
            if X_train_target is not None
            else None,
        },
        "val": {
            "y": {"data": y_val} if y_val is not None else None,
            "X_instance": {"data": X_val_instance}
            if X_val_instance is not None
            else None,
            "X_target": {"data": X_val_target} if X_val_target is not None else None,
        },
        "test": {
            "y": {"data": y_test} if y_test is not None else None,
            "X_instance": {"data": X_test_instance}
            if X_test_instance is not None
            else None,
            "X_target": {"data": X_test_target} if X_test_target is not None else None,
        },
    }
