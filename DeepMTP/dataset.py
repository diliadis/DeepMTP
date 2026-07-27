from __future__ import annotations

import json
import os
import pickle
import tempfile
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict, cast

import arff
import numpy as np
import pandas as pd
import scipy
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from skmultilearn.dataset import load_dataset_dump

from DeepMTP._download_utils import (
    DatasetDownloadError,
    download_and_extract_zip,
    download_file,
    extract_zip_file,
)
from DeepMTP._split_utils import (
    absolute_split_sizes as _absolute_split_sizes,
)
from DeepMTP._split_utils import (
    partition_ordered_ids as _partition_ids,
)
from DeepMTP._split_utils import (
    validate_split_ratio as _validate_split_ratio,
)
from DeepMTP.data_progress import (
    DataProgressEvent,
    DataProgressObserver,
    build_data_progress_observer,
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
_MC_DATASET_URLS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
}
_MOVIELENS_RATINGS_COLUMNS = (
    "instance_id",
    "target_id",
    "value",
    "timestamp",
)
_DP_DATASET_SOURCES: dict[str, tuple[str, str | None]] = {
    "ern": (
        "https://people.montefiore.uliege.be/schrynemackers/ern/",
        ",",
    ),
    "srn": (
        "https://people.montefiore.uliege.be/schrynemackers/srn/",
        ",",
    ),
    "dpie": (
        "https://people.montefiore.uliege.be/schrynemackers/dpix/dpie_",
        None,
    ),
    "dpii": (
        "https://people.montefiore.uliege.be/schrynemackers/dpix/dpii_",
        None,
    ),
}
_DP_FILE_STEMS = ("Y", "X1", "X2")
_MLC_DATASET_BASE_URL = "https://scikit.ml/datasets/"
_MLC_DATASET_NAMES = frozenset(
    {
        "Corel5k",
        "bibtex",
        "birds",
        "delicious",
        "emotions",
        "enron",
        "genbase",
        "mediamill",
        "medical",
        "rcv1subset1",
        "rcv1subset2",
        "rcv1subset3",
        "rcv1subset4",
        "rcv1subset5",
        "scene",
        "tmc2007_500",
        "yeast",
    }
)
_MLC_DATASET_VARIANTS = frozenset({"train", "test", "undivided"})
_SYNTHETIC_RELATIONSHIP_MODES = frozenset(
    {
        "u+logv",
        "u*logv",
        "u^v",
        "u^v_reversed",
        "max(u,v)",
        "u*v",
    }
)
_SYNTHETIC_NOISE_MODES = frozenset({"u+logv", "u*logv", "u^v", "u^v_reversed"})


class DatasetSplit(TypedDict):
    """One raw train, validation, or test split."""

    y: DatasetValue
    X_instance: DatasetValue
    X_target: DatasetValue


class DatasetBundle(TypedDict):
    """Stable outer structure returned by legacy dataset helpers."""

    train: DatasetSplit
    test: DatasetSplit
    val: DatasetSplit


def _dataset_split(
    *,
    y: DatasetValue = None,
    X_instance: DatasetValue = None,
    X_target: DatasetValue = None,
) -> DatasetSplit:
    """Build one dataset split with the stable public keys."""
    return {"y": y, "X_instance": X_instance, "X_target": X_target}


def _dataset_bundle(
    *,
    train: DatasetSplit,
    val: DatasetSplit | None = None,
    test: DatasetSplit | None = None,
) -> DatasetBundle:
    """Build a dataset bundle with independent empty holdout splits."""
    return {
        "train": train,
        "test": test if test is not None else _dataset_split(),
        "val": val if val is not None else _dataset_split(),
    }


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


def _processed_dataset_value(
    data: pd.DataFrame | None,
) -> ProcessedDatasetValue | None:
    return {"data": data} if data is not None else None


def _processed_dataset_split(
    *,
    y: pd.DataFrame | None = None,
    X_instance: pd.DataFrame | None = None,
    X_target: pd.DataFrame | None = None,
) -> ProcessedDatasetSplit:
    return {
        "y": _processed_dataset_value(y),
        "X_instance": _processed_dataset_value(X_instance),
        "X_target": _processed_dataset_value(X_target),
    }


def _processed_dataset_bundle(
    *,
    train: ProcessedDatasetSplit,
    val: ProcessedDatasetSplit | None = None,
    test: ProcessedDatasetSplit | None = None,
) -> ProcessedDatasetBundle:
    return {
        "train": train,
        "val": val if val is not None else _processed_dataset_split(),
        "test": test if test is not None else _processed_dataset_split(),
    }


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


def _validate_array_format(name: str, value: object) -> None:
    if not isinstance(value, str) or value not in {"numpy", "dataframe"}:
        raise ValueError(f"{name} must be either 'numpy' or 'dataframe'.")


def _validate_print_mode(print_mode: object) -> None:
    if not isinstance(print_mode, str) or print_mode not in {"basic", "dev"}:
        raise ValueError("print_mode must be either 'basic' or 'dev'.")


def _resolve_dataset_progress(
    print_mode: PrintMode,
    progress: DataProgressObserver | None,
) -> DataProgressObserver:
    if progress is not None:
        return progress
    return build_data_progress_observer(True, print_mode=print_mode)


def _emit_dataset_operation_started(
    progress: DataProgressObserver,
    *,
    subject: str,
    message: str,
    value: object | None = None,
    end: str = "\n",
) -> None:
    progress.on_event(
        DataProgressEvent(
            message,
            kind="operation_started",
            subject=subject,
            value=value,
            end=end,
        )
    )


def _emit_dataset_operation_completed(
    progress: DataProgressObserver,
    *,
    subject: str,
    value: object | None = None,
) -> None:
    progress.on_event(
        DataProgressEvent(
            "Done",
            kind="operation_completed",
            subject=subject,
            value=value,
        )
    )


def _emit_dataset_message(
    progress: DataProgressObserver,
    *,
    subject: str,
    message: str,
    value: object | None = None,
) -> None:
    progress.on_event(
        DataProgressEvent(
            message,
            subject=subject,
            value=value,
        )
    )


def _format_feature_matrix(
    features: np.ndarray,
    features_format: ArrayFormat,
) -> np.ndarray | pd.DataFrame:
    if features_format == "numpy":
        return features
    feature_frame = pd.DataFrame({"id": np.arange(len(features))})
    feature_frame["features"] = [row.copy() for row in features]
    return feature_frame


def _format_interaction_matrix(
    interactions: np.ndarray,
    interaction_matrix_format: ArrayFormat,
) -> np.ndarray | pd.DataFrame:
    if interaction_matrix_format == "numpy":
        return interactions
    instance_ids, target_ids = np.indices(interactions.shape)
    return pd.DataFrame(
        {
            "instance_id": instance_ids.ravel(),
            "target_id": target_ids.ravel(),
            "value": interactions.ravel().copy(),
        }
    )


def _generate_dummy_features(
    generator: np.random.Generator,
    *,
    num_entities: int,
    num_features: int,
    features_format: ArrayFormat,
) -> np.ndarray | pd.DataFrame:
    features = generator.random((num_entities, num_features))
    features[:, 0] = np.arange(num_entities)
    return _format_feature_matrix(features, features_format)


def _generate_synthetic_scores(
    X_instance: np.ndarray,
    X_target: np.ndarray,
    *,
    mode: str,
    generator: np.random.Generator,
    error_mu: float,
    error_sigma: float,
) -> np.ndarray:
    instance_values = X_instance[:, np.newaxis, :]
    target_values = X_target[np.newaxis, :, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if mode == "u+logv":
            scores = np.sum(instance_values + np.log(target_values), axis=2)
        elif mode == "u*logv":
            scores = np.sum(instance_values * np.log(target_values), axis=2)
        elif mode == "u^v":
            scores = np.sum(instance_values**target_values, axis=2)
        elif mode == "u^v_reversed":
            scores = np.sum(
                instance_values ** target_values[:, :, ::-1],
                axis=2,
            )
        elif mode == "max(u,v)":
            scores = np.sum(np.maximum(instance_values, target_values), axis=2)
        else:  # mode == "u*v"
            scores = np.sum(instance_values * target_values, axis=2)

    if mode in _SYNTHETIC_NOISE_MODES:
        scores = scores + generator.normal(
            error_mu,
            error_sigma,
            size=scores.shape,
        )
    if not np.isfinite(scores).all():
        raise ValueError(
            "Synthetic score generation produced non-finite values; "
            "use less extreme feature or noise parameters."
        )
    return scores


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


def _partition_mtp_axis(
    num_entities: int,
    ratio: Mapping[str, float] | None,
    *,
    parameter_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Resolve IDs used by each split for one MTP axis.

    An omitted validation partition reuses the training IDs. This keeps a
    validation split created on the other axis aligned with its static axis.
    """
    all_ids = np.arange(num_entities)
    if ratio is None:
        return all_ids, all_ids, all_ids, False, False

    train_ids, val_ids, test_ids = _partition_ids(
        all_ids,
        ratio,
        parameter_name=parameter_name,
    )
    return (
        train_ids,
        val_ids if val_ids is not None else train_ids,
        test_ids,
        val_ids is not None,
        True,
    )


def _select_mtp_interactions(
    interactions: pd.DataFrame,
    instance_ids: np.ndarray,
    target_ids: np.ndarray,
) -> pd.DataFrame:
    return interactions.loc[
        interactions["instance_id"].isin(instance_ids)
        & interactions["target_id"].isin(target_ids)
    ].copy()


def _select_optional_feature_ids(
    features: pd.DataFrame | None,
    ids: np.ndarray,
    *,
    include: bool,
) -> pd.DataFrame | None:
    if features is None or not include:
        return None
    return _select_feature_ids(features, ids)


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


def _parse_mlc_manifest(
    manifest_path: Path,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Parse the scikit-multilearn checksum manifest."""
    try:
        rows = manifest_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Multi-label dataset manifest is not valid UTF-8: {manifest_path}."
        ) from exc

    manifest: dict[tuple[str, str], tuple[str, str]] = {}
    for line_number, raw_row in enumerate(rows, start=1):
        row = raw_row.strip()
        if not row:
            continue
        parts = row.split(";")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid multi-label manifest row {line_number}: {raw_row!r}."
            )
        checksum, filename = (part.strip() for part in parts)
        if len(checksum) != 32 or any(
            character not in "0123456789abcdefABCDEF" for character in checksum
        ):
            raise ValueError(
                f"Invalid MD5 checksum on multi-label manifest row {line_number}."
            )
        if (
            "/" in filename
            or "\\" in filename
            or Path(filename).name != filename
            or not filename.endswith(".bz2")
        ):
            raise ValueError(
                f"Unsafe multi-label filename on manifest row {line_number}: "
                f"{filename!r}."
            )
        try:
            dataset_name, variant = filename.removesuffix(".bz2").rsplit("-", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid multi-label filename on manifest row {line_number}: "
                f"{filename!r}."
            ) from exc
        if not dataset_name or variant not in _MLC_DATASET_VARIANTS:
            raise ValueError(
                f"Invalid multi-label filename on manifest row {line_number}: "
                f"{filename!r}."
            )
        key = (dataset_name, variant)
        if key in manifest:
            raise ValueError(
                f"Duplicate multi-label manifest entry for {dataset_name}-{variant}."
            )
        manifest[key] = (checksum.lower(), filename)

    if not manifest:
        raise ValueError(f"Multi-label dataset manifest is empty: {manifest_path}.")
    return manifest


def _download_mlc_manifest(
    manifest_path: Path,
    required_paths: list[Path],
) -> None:
    try:
        download_file(f"{_MLC_DATASET_BASE_URL}data.list", manifest_path)
    except DatasetDownloadError as exc:
        raise DatasetDownloadError(
            "The multi-label dataset is not cached and the upstream catalog "
            "could not be retrieved. Place the trusted scikit-multilearn dump "
            f"files at {[str(path) for path in required_paths]}."
        ) from exc


def _ensure_mlc_cache(
    dataset_dir: Path,
    dataset_name: str,
    variants: tuple[str, ...],
) -> tuple[dict[str, Path], bool]:
    dataset_files = {
        variant: dataset_dir / f"{dataset_name}-{variant}.bz2" for variant in variants
    }
    invalid_cache_paths = [
        dataset_file
        for dataset_file in dataset_files.values()
        if dataset_file.exists() and not dataset_file.is_file()
    ]
    if invalid_cache_paths:
        raise IsADirectoryError(
            "Expected multi-label cache paths to be files: "
            f"{[str(path) for path in invalid_cache_paths]}."
        )

    missing_variants = [
        variant
        for variant, dataset_file in dataset_files.items()
        if not dataset_file.is_file()
    ]
    if not missing_variants:
        return dataset_files, False

    dataset_dir.mkdir(parents=True, exist_ok=True)
    required_paths = [dataset_files[variant] for variant in missing_variants]
    manifest_path = dataset_dir / "data.list"
    if manifest_path.exists() and not manifest_path.is_file():
        raise IsADirectoryError(
            f"Expected multi-label manifest path to be a file: {manifest_path}."
        )
    manifest_was_cached = manifest_path.is_file()
    if not manifest_was_cached:
        _download_mlc_manifest(manifest_path, required_paths)

    try:
        manifest = _parse_mlc_manifest(manifest_path)
    except ValueError as exc:
        if not manifest_was_cached:
            raise DatasetDownloadError(
                f"The downloaded multi-label dataset manifest is invalid: "
                f"{manifest_path}."
            ) from exc
        _download_mlc_manifest(manifest_path, required_paths)
        try:
            manifest = _parse_mlc_manifest(manifest_path)
        except ValueError as refresh_exc:
            raise DatasetDownloadError(
                f"The refreshed multi-label dataset manifest is invalid: "
                f"{manifest_path}."
            ) from refresh_exc

    for variant in missing_variants:
        key = (dataset_name, variant)
        if key not in manifest:
            raise DatasetDownloadError(
                f"The multi-label dataset catalog has no entry for "
                f"{dataset_name}-{variant}."
            )
        checksum, filename = manifest[key]
        destination = dataset_files[variant]
        if filename != destination.name:
            raise DatasetDownloadError(
                f"Unexpected cache filename for {dataset_name}-{variant}: {filename!r}."
            )
        try:
            download_file(
                f"{_MLC_DATASET_BASE_URL}{filename}",
                destination,
                expected_md5=checksum,
            )
        except DatasetDownloadError as exc:
            raise DatasetDownloadError(
                f"Failed to download multi-label dataset {dataset_name}-{variant}."
            ) from exc
    return dataset_files, True


def _load_mlc_dump(dataset_path: Path) -> tuple[object, object]:
    try:
        loaded = load_dataset_dump(str(dataset_path))
        X, y, _, _ = loaded
    except (
        AttributeError,
        EOFError,
        KeyError,
        OSError,
        pickle.PickleError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Multi-label dataset cache could not be loaded: {dataset_path}."
        ) from exc
    return X, y


def _coerce_mlc_matrix(
    matrix: object,
    *,
    matrix_name: str,
    source: Path,
) -> np.ndarray:
    try:
        if scipy.sparse.issparse(matrix):
            matrix = cast(Any, matrix).toarray()
        dense_matrix = np.asarray(matrix)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{matrix_name} in {source} could not be converted to a matrix."
        ) from exc
    if dense_matrix.ndim != 2 or dense_matrix.shape[0] < 1 or dense_matrix.shape[1] < 1:
        raise ValueError(
            f"{matrix_name} in {source} must be a non-empty two-dimensional matrix."
        )
    if not np.issubdtype(dense_matrix.dtype, np.number) or np.iscomplexobj(
        dense_matrix
    ):
        raise TypeError(f"{matrix_name} in {source} must contain real numeric values.")
    if not np.isfinite(dense_matrix).all():
        raise ValueError(f"{matrix_name} in {source} must contain finite values.")
    return dense_matrix.copy()


def _load_mlc_variant(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw_features, raw_labels = _load_mlc_dump(dataset_path)
    features = _coerce_mlc_matrix(
        raw_features,
        matrix_name="Instance features",
        source=dataset_path,
    )
    labels = _coerce_mlc_matrix(
        raw_labels,
        matrix_name="Labels",
        source=dataset_path,
    )
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Instance features and labels in {dataset_path} must have the same "
            f"number of rows; received {features.shape[0]} and {labels.shape[0]}."
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Labels in {dataset_path} must be binary values.")
    return features, labels


def load_process_MLC(
    path: str | os.PathLike[str] = "./data",
    dataset_name: str = "bibtex",
    variant: DatasetVariant = "undivided",
    features_type: ArrayFormat = "numpy",
    print_mode: PrintMode = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> DatasetBundle:
    """Load a cached or downloadable scikit-multilearn benchmark dataset.

    Args:
        path: Directory used for the local dataset cache.
        dataset_name: Supported multi-label benchmark name.
        variant: Return the complete dataset or its published train/test split.
        features_type: Return instance features as an array or feature DataFrame.
        print_mode: ``"basic"`` for ordinary messages or ``"dev"`` for
            messages prefixed for the development application.
        progress: Receives dataset loading and download status events.

    Raises:
        ValueError: If an option or cached dataset is invalid.
        DatasetDownloadError: If required cache files cannot be downloaded safely.
        IsADirectoryError: If an expected cache file path is a directory.
    """
    if dataset_name not in _MLC_DATASET_NAMES:
        raise ValueError(f"dataset_name must be one of {sorted(_MLC_DATASET_NAMES)}.")
    if not isinstance(variant, str) or variant not in {"undivided", "divided"}:
        raise ValueError("variant must be either 'undivided' or 'divided'.")
    _validate_array_format("features_type", features_type)
    _validate_print_mode(print_mode)
    progress_observer = _resolve_dataset_progress(print_mode, progress)

    data_root = _prepare_dataset_root(path)
    dataset_dir = data_root / "multi-label_classification"
    variants = ("undivided",) if variant == "undivided" else ("train", "test")
    _emit_dataset_operation_started(
        progress_observer,
        subject="mlc_dataset_loading",
        message="Processing...",
        value=dataset_name,
    )
    dataset_files, downloaded = _ensure_mlc_cache(
        dataset_dir,
        dataset_name,
        variants,
    )
    if downloaded:
        _emit_dataset_message(
            progress_observer,
            subject="mlc_dataset_download",
            message="Downloaded missing dataset files",
            value=dataset_name,
        )

    X_train_array, y_train = _load_mlc_variant(dataset_files[variants[0]])
    X_test_array: np.ndarray | None = None
    y_test: np.ndarray | None = None
    if variant == "divided":
        X_test_array, y_test = _load_mlc_variant(dataset_files["test"])
        if X_train_array.shape[1] != X_test_array.shape[1]:
            raise ValueError(
                "Multi-label train and test instance features must have the same "
                f"column count; received {X_train_array.shape[1]} and "
                f"{X_test_array.shape[1]}."
            )
        if y_train.shape[1] != y_test.shape[1]:
            raise ValueError(
                "Multi-label train and test labels must have the same column "
                f"count; received {y_train.shape[1]} and {y_test.shape[1]}."
            )

    X_train_instance = _format_feature_matrix(X_train_array, features_type)
    X_test_instance = (
        _format_feature_matrix(X_test_array, features_type)
        if X_test_array is not None
        else None
    )
    _emit_dataset_operation_completed(
        progress_observer,
        subject="mlc_dataset_loading",
        value=dataset_name,
    )
    return _dataset_bundle(
        train=_dataset_split(y=y_train, X_instance=X_train_instance),
        test=_dataset_split(y=y_test, X_instance=X_test_instance),
    )


def process_dummy_MLC(
    num_features: int = 10,
    num_instances: int = 50,
    num_targets: int = 5,
    interaction_matrix_format: ArrayFormat = "numpy",
    features_format: ArrayFormat = "numpy",
    random_state: int | None = 42,
) -> DatasetBundle:
    """Generate a reproducible dummy multi-label classification dataset.

    Args:
        num_features: Number of features per instance.
        num_instances: Number of instances.
        num_targets: Number of binary labels.
        interaction_matrix_format: Return labels as a dense matrix or triplets.
        features_format: Return features as a dense matrix or feature DataFrame.
        random_state: Seed for an isolated NumPy generator. Use ``None`` for
            non-deterministic data.

    Returns:
        A dataset bundle containing an undivided training split.

    Raises:
        ValueError: If a format, dimension, or random state is invalid.
    """
    _validate_array_format("interaction_matrix_format", interaction_matrix_format)
    _validate_array_format("features_format", features_format)
    _require_positive_integer("num_features", num_features)
    _require_positive_integer("num_instances", num_instances)
    _require_positive_integer("num_targets", num_targets)

    generator = _random_generator(random_state)
    X_train_instance = _generate_dummy_features(
        generator,
        num_entities=num_instances,
        num_features=num_features,
        features_format=features_format,
    )
    y_train = _format_interaction_matrix(
        generator.integers(2, size=(num_instances, num_targets)),
        interaction_matrix_format,
    )
    return _dataset_bundle(train=_dataset_split(y=y_train, X_instance=X_train_instance))


def load_process_MTR(
    path: str | os.PathLike[str] = "./data",
    dataset_name: str = "enb",
    features_type: ArrayFormat = "numpy",
    print_mode: PrintMode = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> DatasetBundle:
    """Load multivariate regression datasets from the mulan repository.

    Args:
        path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
        dataset_name (str, optional): The name of the multivariate regression dataset. Defaults to 'enb'.
        features_type (str, optional): The format of the instance features. There are two possible values, numpy and dataframe. This is intended to test the functionality of the data_process function. Defaults to 'numpy'.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        progress (DataProgressObserver | None, optional): Receives dataset loading and download status events. Defaults to None.

    Raises:
        ValueError: If a public option or downloaded dataset is invalid.
        DatasetDownloadError: If the dataset archive cannot be downloaded safely.
        FileNotFoundError: If the archive does not contain the requested dataset.

    Returns:
        dict: A dictionary with all the available data for the multivariate regression dataset.
    """
    _validate_array_format("features_type", features_type)
    _validate_print_mode(print_mode)
    if dataset_name not in _MTR_DATASET_TARGETS:
        raise ValueError(f"dataset_name must be one of {sorted(_MTR_DATASET_TARGETS)}.")
    data_root = _prepare_dataset_root(path)
    progress_observer = _resolve_dataset_progress(print_mode, progress)

    _emit_dataset_operation_started(
        progress_observer,
        subject="mtr_dataset_loading",
        message="Processing...",
        value=dataset_name,
    )

    dataset_path = data_root / "mtr-datasets" / f"{dataset_name}.arff"
    if not dataset_path.is_file():
        _emit_dataset_operation_started(
            progress_observer,
            subject="mtr_dataset_download",
            message="Downloading and extracting dataset from scratch... ",
            value=dataset_name,
            end="",
        )
        download_and_extract_zip(_MTR_ARCHIVE_URL, data_root)
        _emit_dataset_operation_completed(
            progress_observer,
            subject="mtr_dataset_download",
            value=dataset_name,
        )
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

    X_train_instance = _format_feature_matrix(
        np.array(data[:, :-num_targets]),
        features_type,
    )
    y_train = np.array(data[:, -num_targets:])

    _emit_dataset_operation_completed(
        progress_observer,
        subject="mtr_dataset_loading",
        value=dataset_name,
    )
    return _dataset_bundle(
        train=_dataset_split(y=y_train, X_instance=X_train_instance),
    )


def process_dummy_MTR(
    num_features: int = 10,
    num_instances: int = 50,
    num_targets: int = 5,
    interaction_matrix_format: ArrayFormat = "numpy",
    features_format: ArrayFormat = "numpy",
    variant: DatasetVariant = "undivided",
    split_ratio: Mapping[str, float] | None = None,
    random_state: int | None = 42,
) -> DatasetBundle:
    """Generate a reproducible dummy multivariate regression dataset.

    Args:
        num_features: Number of features per instance.
        num_instances: Number of instances.
        num_targets: Number of regression targets.
        interaction_matrix_format: Return scores as a dense matrix or triplets.
        features_format: Return features as a dense matrix or feature DataFrame.
        variant: Return an undivided dataset or train/validation/test splits.
        split_ratio: Ratios used for the divided variant.
        random_state: Seed used for both data generation and splitting. Use
            ``None`` for non-deterministic data.

    Returns:
        A dataset bundle containing the generated scores and features.

    Raises:
        TypeError: If ``split_ratio`` is not a mapping.
        ValueError: If an option, dimension, ratio, or random state is invalid.
    """
    split_ratio = _split_ratio(split_ratio)
    _validate_array_format("interaction_matrix_format", interaction_matrix_format)
    _validate_array_format("features_format", features_format)
    if not isinstance(variant, str) or variant not in {"undivided", "divided"}:
        raise ValueError("variant must be either 'undivided' or 'divided'.")
    _require_positive_integer("num_features", num_features)
    _require_positive_integer("num_instances", num_instances)
    _require_positive_integer("num_targets", num_targets)
    if variant == "divided":
        _absolute_split_sizes(
            num_instances,
            split_ratio,
            parameter_name="split ratio",
            population_name="instance",
        )

    generator = _random_generator(random_state)
    split_random_state = (
        random_state
        if random_state is not None
        else int(generator.integers(0, np.iinfo(np.uint32).max))
    )
    X_train_instance = _generate_dummy_features(
        generator,
        num_entities=num_instances,
        num_features=num_features,
        features_format=features_format,
    )
    y_train = _format_interaction_matrix(
        generator.integers(10, size=(num_instances, num_targets)),
        interaction_matrix_format,
    )
    X_val_instance: DatasetValue = None
    X_test_instance: DatasetValue = None
    y_val: DatasetValue = None
    y_test: DatasetValue = None

    if variant == "divided":
        train_ids, test_ids = train_test_split(
            np.arange(num_instances),
            test_size=split_ratio["test"],
            random_state=split_random_state,
        )
        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=split_ratio["val"] / (1 - split_ratio["test"]),
            random_state=split_random_state,
        )

        if isinstance(y_train, pd.DataFrame):
            y_test = y_train.loc[y_train["instance_id"].isin(test_ids)]
            y_val = y_train.loc[y_train["instance_id"].isin(val_ids)]
            y_train = y_train.loc[y_train["instance_id"].isin(train_ids)]
        else:
            y_test = y_train[test_ids, :]
            y_val = y_train[val_ids, :]
            y_train = y_train[train_ids, :]

        if isinstance(X_train_instance, pd.DataFrame):
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

    return _dataset_bundle(
        train=_dataset_split(y=y_train, X_instance=X_train_instance),
        val=_dataset_split(y=y_val, X_instance=X_val_instance),
        test=_dataset_split(y=y_test, X_instance=X_test_instance),
    )


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


def _load_numeric_matrix(
    file_path: Path,
    *,
    delimiter: str | None,
) -> np.ndarray:
    """Load one non-empty, finite, two-dimensional numeric matrix."""
    if file_path.stat().st_size == 0:
        raise ValueError(f"Dataset file must contain a non-empty matrix: {file_path}.")
    try:
        matrix = np.loadtxt(
            file_path,
            delimiter=delimiter,
            dtype=float,
            ndmin=2,
        )
    except ValueError as exc:
        raise ValueError(
            f"Dataset file must contain a numeric matrix: {file_path}."
        ) from exc

    if matrix.size == 0 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"Dataset file must contain a non-empty matrix: {file_path}.")
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"Dataset file must contain only finite numeric values: {file_path}."
        )
    return matrix


def _load_dp_matrices(
    dataset_files: Mapping[str, Path],
    *,
    delimiter: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _load_numeric_matrix(dataset_files["Y"], delimiter=delimiter)
    X_instance = _load_numeric_matrix(dataset_files["X1"], delimiter=delimiter)
    X_target = _load_numeric_matrix(dataset_files["X2"], delimiter=delimiter)

    if X_instance.shape[0] != y.shape[0]:
        raise ValueError(
            "Dyadic prediction instance features must have one row per "
            f"interaction-matrix row; received {X_instance.shape[0]} and {y.shape[0]}."
        )
    if X_target.shape[0] != y.shape[1]:
        raise ValueError(
            "Dyadic prediction target features must have one row per "
            f"interaction-matrix column; received {X_target.shape[0]} and {y.shape[1]}."
        )
    return y, X_instance, X_target


def _split_dp_matrices(
    y_train: np.ndarray,
    X_train_instance: np.ndarray,
    X_train_target: np.ndarray,
    *,
    random_state: int | None,
    split_ratio: Mapping[str, float],
    split_instance_features: bool,
    split_target_features: bool,
    validation_setting: str,
) -> DatasetBundle:
    X_val_instance, X_test_instance = None, None
    X_val_target, X_test_target = None, None
    y_val, y_test = None, None

    if validation_setting in {"B", "D"}:
        _absolute_split_sizes(
            y_train.shape[0],
            split_ratio,
            parameter_name="split ratio",
            population_name="instance",
        )
    if validation_setting in {"C", "D"}:
        _absolute_split_sizes(
            y_train.shape[1],
            split_ratio,
            parameter_name="split ratio",
            population_name="target",
        )

    validation_size = split_ratio["val"] / (1 - split_ratio["test"])
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
                test_size=validation_size,
                random_state=random_state,
            )
        else:
            y_train, y_test = train_test_split(
                y_train,
                test_size=split_ratio["test"],
                random_state=random_state,
            )
            y_train, y_val = train_test_split(
                y_train,
                test_size=validation_size,
                random_state=random_state,
            )
    elif validation_setting == "C":
        y_train_target_ids, y_test_target_ids = train_test_split(
            range(y_train.shape[1]),
            test_size=split_ratio["test"],
            random_state=random_state,
        )
        y_train_target_ids, y_val_target_ids = train_test_split(
            y_train_target_ids,
            test_size=validation_size,
            random_state=random_state,
        )
        y_val = y_train[:, y_val_target_ids]
        y_test = y_train[:, y_test_target_ids]
        y_train = y_train[:, y_train_target_ids]

        if split_target_features:
            X_val_target = X_train_target[y_val_target_ids, :]
            X_test_target = X_train_target[y_test_target_ids, :]
            X_train_target = X_train_target[y_train_target_ids, :]
    else:  # validation_setting == "D"
        y_train_instance_ids, y_test_instance_ids = train_test_split(
            range(y_train.shape[0]),
            test_size=split_ratio["test"],
            random_state=random_state,
        )
        y_train_instance_ids, y_val_instance_ids = train_test_split(
            y_train_instance_ids,
            test_size=validation_size,
            random_state=random_state,
        )
        y_train_target_ids, y_test_target_ids = train_test_split(
            range(y_train.shape[1]),
            test_size=split_ratio["test"],
            random_state=random_state,
        )
        y_train_target_ids, y_val_target_ids = train_test_split(
            y_train_target_ids,
            test_size=validation_size,
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

    return _dataset_bundle(
        train=_dataset_split(
            y=y_train,
            X_instance=X_train_instance,
            X_target=X_train_target,
        ),
        val=_dataset_split(
            y=y_val,
            X_instance=X_val_instance,
            X_target=X_val_target,
        ),
        test=_dataset_split(
            y=y_test,
            X_instance=X_test_instance,
            X_target=X_test_target,
        ),
    )


def load_process_DP(
    path: str | os.PathLike[str] = "./data",
    dataset_name: str = "ern",
    variant: DatasetVariant = "undivided",
    random_state: int | None = 42,
    split_ratio: Mapping[str, float] | None = None,
    split_instance_features: bool = False,
    split_target_features: bool = False,
    validation_setting: str = "B",
    print_mode: PrintMode = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> DatasetBundle:
    """Load a dyadic-prediction dataset and optionally create held-out splits.

    Args:
        path: Directory used for the local dataset cache.
        dataset_name: One of ``"ern"``, ``"srn"``, ``"dpie"``, or ``"dpii"``.
        variant: Return the complete matrices or create train/validation/test splits.
        random_state: Seed used by the split operations, or ``None``.
        split_ratio: Train, validation, and test proportions.
        split_instance_features: Split instance features with interaction rows.
        split_target_features: Split target features with interaction columns.
        validation_setting: Novelty setting ``"B"``, ``"C"``, or ``"D"``.
        print_mode: ``"basic"`` for ordinary messages or ``"dev"`` for
            messages prefixed for the development application.
        progress: Receives dataset loading and download status events.

    Raises:
        ValueError: If an option or dataset matrix is invalid.
        TypeError: If a feature-splitting flag is not boolean.
        DatasetDownloadError: If a required dataset file cannot be downloaded.
        FileNotFoundError: If a download does not create all required files.
    """
    validated_split_ratio = _split_ratio(split_ratio)
    if dataset_name not in _DP_DATASET_SOURCES:
        raise ValueError(f"dataset_name must be one of {sorted(_DP_DATASET_SOURCES)}.")
    if variant not in {"undivided", "divided"}:
        raise ValueError("variant must be either 'undivided' or 'divided'.")
    if not isinstance(validation_setting, str):
        raise ValueError("validation_setting must be one of ['B', 'C', 'D'].")
    validation_setting = validation_setting.upper()
    if validation_setting not in {"B", "C", "D"}:
        raise ValueError("validation_setting must be one of ['B', 'C', 'D'].")
    if not isinstance(split_instance_features, bool):
        raise TypeError("split_instance_features must be a boolean.")
    if not isinstance(split_target_features, bool):
        raise TypeError("split_target_features must be a boolean.")
    _validate_print_mode(print_mode)
    _random_generator(random_state)
    progress_observer = _resolve_dataset_progress(print_mode, progress)

    data_root = _prepare_dataset_root(path)
    dataset_dir = data_root / "dyadic_prediction-datasets" / dataset_name
    url_prefix, delimiter = _DP_DATASET_SOURCES[dataset_name]
    dataset_files = {
        stem: dataset_dir / Path(f"{url_prefix}{stem}.txt").name
        for stem in _DP_FILE_STEMS
    }
    invalid_cache_paths = [
        dataset_file
        for dataset_file in dataset_files.values()
        if dataset_file.exists() and not dataset_file.is_file()
    ]
    if invalid_cache_paths:
        raise IsADirectoryError(
            "Expected dyadic prediction cache paths to be files: "
            f"{[str(cache_path) for cache_path in invalid_cache_paths]}."
        )
    missing_files = [
        (stem, dataset_file)
        for stem, dataset_file in dataset_files.items()
        if not dataset_file.is_file()
    ]

    _emit_dataset_operation_started(
        progress_observer,
        subject="dp_dataset_loading",
        message="Processing...",
        value=dataset_name,
    )
    if missing_files:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _emit_dataset_operation_started(
            progress_observer,
            subject="dp_dataset_download",
            message="Downloading missing dataset files... ",
            value=dataset_name,
            end="",
        )
        for stem, destination in missing_files:
            download_file(f"{url_prefix}{stem}.txt", destination)
        _emit_dataset_operation_completed(
            progress_observer,
            subject="dp_dataset_download",
            value=dataset_name,
        )

    still_missing = [
        str(dataset_file)
        for dataset_file in dataset_files.values()
        if not dataset_file.is_file()
    ]
    if still_missing:
        raise FileNotFoundError(
            "Dyadic prediction download did not create the expected files: "
            f"{still_missing}."
        )

    y_train, X_train_instance, X_train_target = _load_dp_matrices(
        dataset_files,
        delimiter=delimiter,
    )
    if variant == "undivided":
        data = _dataset_bundle(
            train=_dataset_split(
                y=y_train,
                X_instance=X_train_instance,
                X_target=X_train_target,
            ),
        )
    else:
        data = _split_dp_matrices(
            y_train,
            X_train_instance,
            X_train_target,
            random_state=random_state,
            split_ratio=validated_split_ratio,
            split_instance_features=split_instance_features,
            split_target_features=split_target_features,
            validation_setting=validation_setting,
        )

    _emit_dataset_operation_completed(
        progress_observer,
        subject="dp_dataset_loading",
        value=dataset_name,
    )
    return data


def process_dummy_DP(
    num_instance_features: int = 10,
    num_target_features: int = 3,
    num_instances: int = 50,
    num_targets: int = 5,
    interaction_matrix_format: ArrayFormat = "numpy",
    instance_features_format: ArrayFormat = "numpy",
    target_features_format: ArrayFormat = "numpy",
    random_state: int | None = 42,
) -> DatasetBundle:
    """Generate a reproducible dummy dyadic prediction dataset.

    Args:
        num_instance_features: Number of features per instance.
        num_target_features: Number of features per target.
        num_instances: Number of instances.
        num_targets: Number of targets.
        interaction_matrix_format: Return scores as a dense matrix or triplets.
        instance_features_format: Return instance features as an array or DataFrame.
        target_features_format: Return target features as an array or DataFrame.
        random_state: Seed for an isolated NumPy generator. Use ``None`` for
            non-deterministic data.

    Returns:
        A dataset bundle containing an undivided training split.

    Raises:
        ValueError: If a format, dimension, or random state is invalid.
    """
    for name, value in {
        "interaction_matrix_format": interaction_matrix_format,
        "instance_features_format": instance_features_format,
        "target_features_format": target_features_format,
    }.items():
        _validate_array_format(name, value)
    _require_positive_integer("num_instance_features", num_instance_features)
    _require_positive_integer("num_target_features", num_target_features)
    _require_positive_integer("num_instances", num_instances)
    _require_positive_integer("num_targets", num_targets)

    generator = _random_generator(random_state)
    X_train_instance = _generate_dummy_features(
        generator,
        num_entities=num_instances,
        num_features=num_instance_features,
        features_format=instance_features_format,
    )
    X_train_target = _generate_dummy_features(
        generator,
        num_entities=num_targets,
        num_features=num_target_features,
        features_format=target_features_format,
    )
    y_train = _format_interaction_matrix(
        generator.integers(10, size=(num_instances, num_targets)),
        interaction_matrix_format,
    )
    return _dataset_bundle(
        train=_dataset_split(
            y=y_train,
            X_instance=X_train_instance,
            X_target=X_train_target,
        )
    )


def _load_movielens_ratings(ratings_path: Path) -> pd.DataFrame:
    """Read and validate a MovieLens 100K ratings file."""
    try:
        ratings = pd.read_csv(
            ratings_path,
            sep="\t",
            header=None,
            dtype=str,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"MovieLens ratings file is empty: {ratings_path}.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"MovieLens ratings file could not be parsed: {ratings_path}."
        ) from exc

    if ratings.empty:
        raise ValueError(f"MovieLens ratings file is empty: {ratings_path}.")
    if ratings.shape[1] != len(_MOVIELENS_RATINGS_COLUMNS):
        raise ValueError(
            "MovieLens ratings file has missing fields or extra fields; "
            f"expected exactly {len(_MOVIELENS_RATINGS_COLUMNS)} columns."
        )
    ratings.columns = list(_MOVIELENS_RATINGS_COLUMNS)
    if ratings.isna().any(axis=None):
        raise ValueError(f"MovieLens ratings file has missing fields: {ratings_path}.")

    for column in _MOVIELENS_RATINGS_COLUMNS:
        try:
            ratings[column] = pd.to_numeric(ratings[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"MovieLens ratings column {column!r} must be numeric."
            ) from exc

    numeric_values = ratings[list(_MOVIELENS_RATINGS_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("MovieLens ratings must contain only finite numeric values.")

    for column in ("instance_id", "target_id", "timestamp"):
        values = ratings[column].to_numpy(dtype=float)
        if (values <= 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(
                f"MovieLens ratings column {column!r} must contain positive integers."
            )

    if not ratings["value"].between(1, 5, inclusive="both").all():
        raise ValueError("MovieLens rating values must be between 1 and 5.")
    rating_values = ratings["value"].to_numpy(dtype=float)
    if not np.equal(rating_values, np.floor(rating_values)).all():
        raise ValueError("MovieLens rating values must be integers.")
    if ratings.duplicated(subset=["instance_id", "target_id"]).any():
        raise ValueError(
            "MovieLens ratings contain duplicate instance-target interactions."
        )

    ratings["instance_id"] = pd.factorize(ratings["instance_id"], sort=False)[0]
    ratings["target_id"] = pd.factorize(ratings["target_id"], sort=False)[0]
    return ratings.drop(columns="timestamp")


def _download_movielens_archive(
    url: str,
    archive_path: Path,
    dataset_name: str,
    progress: DataProgressObserver,
) -> None:
    _emit_dataset_operation_started(
        progress,
        subject="mc_dataset_download",
        message=f"Downloading dataset {dataset_name} from scratch... ",
        value=dataset_name,
        end="",
    )
    download_file(url, archive_path)
    _emit_dataset_operation_completed(
        progress,
        subject="mc_dataset_download",
        value=dataset_name,
    )


def load_process_MC(
    path: str | os.PathLike[str] = "./data",
    dataset_name: str = "ml-100k",
    print_mode: PrintMode = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> DatasetBundle:
    """Load a matrix-completion dataset from the MovieLens repository.

    Args:
        path: Directory used for the local dataset cache.
        dataset_name: Matrix-completion dataset to load.
        print_mode: ``"basic"`` for ordinary messages or ``"dev"`` for
            messages prefixed for the development application.
        progress: Receives dataset download status events.

    Raises:
        ValueError: If an option or the ratings data is invalid.
        DatasetDownloadError: If the archive cannot be downloaded or extracted.
        FileNotFoundError: If the archive does not contain the expected ratings file.

    Returns:
        The ratings triplets and empty side-feature and held-out splits.
    """
    if dataset_name not in _MC_DATASET_URLS:
        raise ValueError(f"dataset_name must be one of {sorted(_MC_DATASET_URLS)}.")
    _validate_print_mode(print_mode)
    progress_observer = _resolve_dataset_progress(print_mode, progress)

    data_root = _prepare_dataset_root(path)
    base_dir = data_root / "matrix-completion-datasets"
    data_dir = base_dir / dataset_name
    ratings_path = data_dir / "u.data"
    url = _MC_DATASET_URLS[dataset_name]
    archive_path = base_dir / Path(url).name

    if not ratings_path.is_file():
        base_dir.mkdir(parents=True, exist_ok=True)
        cached_archive = archive_path.is_file()
        if not cached_archive:
            _download_movielens_archive(
                url,
                archive_path,
                dataset_name,
                progress_observer,
            )

        try:
            extract_zip_file(archive_path, base_dir)
        except DatasetDownloadError:
            if not cached_archive:
                raise
            _download_movielens_archive(
                url,
                archive_path,
                dataset_name,
                progress_observer,
            )
            extract_zip_file(archive_path, base_dir)

        if not ratings_path.is_file() and cached_archive:
            _download_movielens_archive(
                url,
                archive_path,
                dataset_name,
                progress_observer,
            )
            extract_zip_file(archive_path, base_dir)

    if not ratings_path.is_file():
        raise FileNotFoundError(
            "MovieLens dataset extraction did not create the expected file "
            f"{ratings_path}."
        )

    y_train = _load_movielens_ratings(ratings_path)
    return _dataset_bundle(train=_dataset_split(y=y_train))


def load_process_MTL(
    path: str | os.PathLike[str] = "./data",
    dataset_name: str = "dog",
    print_mode: PrintMode = "basic",
    *,
    progress: DataProgressObserver | None = None,
) -> DatasetBundle:
    """Load multi-task learning datasets from my custom repository.

    Args:
        path (str, optional): The path where the datasets should be stored. If it doesn't exist, download and store it in this directory. Defaults to './data'.
        dataset_name (str, optional): The name of the multi-task learning dataset. Defaults to 'dog'.
        print_mode (str, optional): The mode of printing. Two values are possible. If 'basic', the prints are just regural python prints. If 'dev' then a prefix is used so that the streamlit application can print more usefull messages. Defaults to 'basic'.
        progress (DataProgressObserver | None, optional): Receives dataset download status events. Defaults to None.

    Raises:
        ValueError: If a public option or dataset file is invalid.
        DatasetDownloadError: If the dataset archive cannot be downloaded safely.
        FileNotFoundError: If required score or image files are missing.

    Returns:
        dict: A dictionary with all the available data for the multi-task learning dataset.
    """
    if dataset_name not in _MTL_DATASET_NAMES:
        raise ValueError(f"dataset_name must be one of {sorted(_MTL_DATASET_NAMES)}.")
    _validate_print_mode(print_mode)
    progress_observer = _resolve_dataset_progress(print_mode, progress)

    data_root = _prepare_dataset_root(path)
    base_dir = data_root / "multi-task_learning-datasets"
    data_dir = base_dir / dataset_name
    json_path = data_dir / "data.json"
    legacy_scores_path = data_dir / "y_dog.pkl"

    if not json_path.is_file() and not legacy_scores_path.is_file():
        _emit_dataset_operation_started(
            progress_observer,
            subject="mtl_dataset_download",
            message=f"Downloading dataset {dataset_name} from scratch... ",
            value=dataset_name,
            end="",
        )
        download_and_extract_zip(
            _MTL_ARCHIVE_URL_TEMPLATE.format(dataset_name=dataset_name),
            base_dir,
        )
        _emit_dataset_operation_completed(
            progress_observer,
            subject="mtl_dataset_download",
            value=dataset_name,
        )

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

    return _dataset_bundle(
        train=_dataset_split(y=y_train, X_instance=X_train_instance),
        test=_dataset_split(y=y_test, X_instance=X_test_instance),
    )


def generate_interaction_matrix(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> None:
    """Convert annotator labels into an atomically written correctness matrix."""
    try:
        destination = Path(output_path)
    except TypeError as exc:
        raise TypeError("output_path must be a string or path-like object.") from exc
    if destination.exists() and not destination.is_file():
        raise IsADirectoryError(f"Output path is not a file: {destination}.")
    if destination.parent.exists() and not destination.parent.is_dir():
        raise NotADirectoryError(
            f"Output parent path is not a directory: {destination.parent}."
        )

    scores = _load_mtl_interaction_matrix(input_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            pickle.dump(scores, output_file, protocol=pickle.HIGHEST_PROTOCOL)
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
) -> DatasetBundle:
    """Generate a reproducible dyadic regression dataset.

    ``sklearn_version`` starts from normally distributed features. For modes
    involving logarithms or fractional powers, the relevant features are
    transformed to a positive domain so every generated score remains real.
    The error distribution applies to the four legacy noisy relationship modes.

    Args:
        num_instances: Number of instances.
        num_targets: Number of targets.
        num_instance_features: Number of features per instance.
        num_target_features: Number of features per target.
        error_mu: Mean of the Gaussian error added by noisy modes.
        error_sigma: Non-negative standard deviation of that Gaussian error.
        sklearn_version: Generate normally distributed rather than uniform features.
        seed: Non-negative 32-bit seed used for generation and splitting.
        mode: Mathematical relationship used to generate scores.
        split_ratio: Train, validation, and test ratios.

    Returns:
        A dataset bundle containing aligned train, validation, and test rows.

    Raises:
        TypeError: If a boolean or error parameter has the wrong type.
        ValueError: If a dimension, mode, seed, ratio, or generated value is invalid.
    """
    split_ratio = _split_ratio(split_ratio)
    _require_positive_integer("num_instances", num_instances)
    _require_positive_integer("num_targets", num_targets)
    _require_positive_integer("num_instance_features", num_instance_features)
    _require_positive_integer("num_target_features", num_target_features)
    if not isinstance(mode, str) or mode not in _SYNTHETIC_RELATIONSHIP_MODES:
        raise ValueError(
            f"mode must be one of {sorted(_SYNTHETIC_RELATIONSHIP_MODES)}."
        )
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
    if seed is None:
        raise ValueError("seed must be a non-negative 32-bit integer.")
    num_validation, num_test = _absolute_split_sizes(
        num_instances,
        split_ratio,
        parameter_name="split ratio",
        population_name="instance",
    )

    generator = _random_generator(seed, parameter_name="seed")
    if sklearn_version:
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

    y_train = _generate_synthetic_scores(
        X_train_instance,
        X_train_target,
        mode=mode,
        generator=generator,
        error_mu=error_mu,
        error_sigma=error_sigma,
    )

    (
        X_train_instance,
        X_test_instance,
        y_train,
        y_test,
    ) = train_test_split(
        X_train_instance,
        y_train,
        test_size=num_test,
        random_state=seed,
    )
    (
        X_train_instance,
        X_val_instance,
        y_train,
        y_val,
    ) = train_test_split(
        X_train_instance,
        y_train,
        test_size=num_validation,
        random_state=seed,
    )

    return _dataset_bundle(
        train=_dataset_split(
            y=y_train,
            X_instance=X_train_instance,
            X_target=X_train_target,
        ),
        test=_dataset_split(y=y_test, X_instance=X_test_instance),
        val=_dataset_split(y=y_val, X_instance=X_val_instance),
    )


def generate_MTP_dataset(
    num_instances: int,
    num_targets: int,
    num_instance_features: int | None = None,
    num_target_features: int | None = None,
    split_instances: Mapping[str, float] | None = None,
    split_targets: Mapping[str, float] | None = None,
    return_static_features_data: bool = False,
) -> ProcessedDatasetBundle:
    """Generate deterministic triplet data with optional axis-level splits.

    Each split mapping requires a test ratio and may include a validation
    ratio. The train ratio is derived when omitted and verified when supplied.
    A test-only axis reuses its training IDs in a validation split created on
    the other axis.

    Args:
        num_instances: Number of instances.
        num_targets: Number of targets.
        num_instance_features: Number of generated instance features, or none.
        num_target_features: Number of generated target features, or none.
        split_instances: Optional instance-axis split ratios.
        split_targets: Optional target-axis split ratios.
        return_static_features_data: Repeat non-novel side features in holdouts.

    Returns:
        Processed triplet and feature frames for train, validation, and test.

    Raises:
        TypeError: If a split or boolean flag has the wrong type.
        ValueError: If a dimension or split ratio is invalid.
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

    split_instances = _optional_split_ratio(
        split_instances,
        parameter_name="split_instances",
    )
    split_targets = _optional_split_ratio(
        split_targets,
        parameter_name="split_targets",
    )
    (
        train_instance_ids,
        val_instance_ids,
        test_instance_ids,
        validation_instances_are_novel,
        test_instances_are_novel,
    ) = _partition_mtp_axis(
        num_instances,
        split_instances,
        parameter_name="split_instances",
    )
    (
        train_target_ids,
        val_target_ids,
        test_target_ids,
        validation_targets_are_novel,
        test_targets_are_novel,
    ) = _partition_mtp_axis(
        num_targets,
        split_targets,
        parameter_name="split_targets",
    )

    instance_features = _generate_static_feature_frame(
        num_instances,
        num_instance_features,
    )
    target_features = _generate_static_feature_frame(
        num_targets,
        num_target_features,
    )
    instance_ids = np.repeat(np.arange(num_instances), num_targets)
    target_ids = np.tile(np.arange(num_targets), num_instances)
    all_interactions = pd.DataFrame(
        {
            "instance_id": instance_ids,
            "target_id": target_ids,
            "value": instance_ids.astype(float),
        }
    )

    y_train = _select_mtp_interactions(
        all_interactions,
        train_instance_ids,
        train_target_ids,
    )
    has_validation_split = (
        validation_instances_are_novel or validation_targets_are_novel
    )
    y_val = (
        _select_mtp_interactions(
            all_interactions,
            val_instance_ids,
            val_target_ids,
        )
        if has_validation_split
        else None
    )
    has_test_split = test_instances_are_novel or test_targets_are_novel
    y_test = (
        _select_mtp_interactions(
            all_interactions,
            test_instance_ids,
            test_target_ids,
        )
        if has_test_split
        else None
    )

    X_train_instance = _select_optional_feature_ids(
        instance_features,
        train_instance_ids,
        include=True,
    )
    X_val_instance = _select_optional_feature_ids(
        instance_features,
        val_instance_ids,
        include=validation_instances_are_novel or return_static_features_data,
    )
    X_test_instance = _select_optional_feature_ids(
        instance_features,
        test_instance_ids,
        include=test_instances_are_novel or return_static_features_data,
    )
    X_train_target = _select_optional_feature_ids(
        target_features,
        train_target_ids,
        include=True,
    )
    X_val_target = _select_optional_feature_ids(
        target_features,
        val_target_ids,
        include=validation_targets_are_novel or return_static_features_data,
    )
    X_test_target = _select_optional_feature_ids(
        target_features,
        test_target_ids,
        include=test_targets_are_novel or return_static_features_data,
    )

    return _processed_dataset_bundle(
        train=_processed_dataset_split(
            y=y_train,
            X_instance=X_train_instance,
            X_target=X_train_target,
        ),
        val=_processed_dataset_split(
            y=y_val,
            X_instance=X_val_instance,
            X_target=X_val_target,
        ),
        test=_processed_dataset_split(
            y=y_test,
            X_instance=X_test_instance,
            X_target=X_test_target,
        ),
    )
