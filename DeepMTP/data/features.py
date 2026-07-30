"""Feature normalization and cross-input consistency validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict, cast

import numpy as np
import pandas as pd
import torch

from DeepMTP.data.graphs import graph_object_array, normalize_graph_rows
from DeepMTP.data.interactions import DatasetMapping, ValidationSetting
from DeepMTP.data.progress import (
    DataProgressObserver,
    _emit_data_message,
    _emit_data_operation_completed,
    _emit_data_operation_started,
    _resolve_data_progress,
)
from DeepMTP.data.sequences import (
    normalize_sequence_rows,
    sequence_object_array,
)
from DeepMTP.data.sparse import (
    is_sparse_value,
    normalize_sparse_matrix,
    normalize_sparse_rows,
    sparse_object_array,
)

__all__ = [
    "CompositeFeatureSpec",
    "FeatureKind",
    "FeatureComponentInfo",
    "FeatureInfo",
    "cross_input_consistency_check_instances",
    "cross_input_consistency_check_targets",
    "process_instance_features",
    "process_target_features",
]

AtomicFeatureKind = Literal["graph", "sequence"] | None


class CompositeFeatureSpec(TypedDict, total=False):
    """Data-preparation options for one named composite component."""

    kind: AtomicFeatureKind
    optional: bool


CompositeFeatureKind = AtomicFeatureKind | CompositeFeatureSpec
FeatureKind = AtomicFeatureKind | Mapping[str, CompositeFeatureKind]


class _FeatureComponentInfoRequired(TypedDict):
    """Representation metadata for one component of a composite feature."""

    num_features: int | None
    info: Literal[
        "numpy",
        "dataframe",
        "graph",
        "sequence",
        "scipy_sparse",
        "torch_sparse",
    ]


class FeatureComponentInfo(_FeatureComponentInfoRequired, total=False):
    """Representation metadata for one component of a composite feature."""

    graph_edge_features: int | None


class _FeatureInfoRequired(TypedDict):
    """Normalized entity features and their detected representation."""

    data: pd.DataFrame
    num_features: int | None
    info: Literal[
        "numpy",
        "dataframe",
        "images",
        "graph",
        "sequence",
        "tabular",
        "scipy_sparse",
        "torch_sparse",
        "composite",
    ]
    components: dict[str, FeatureComponentInfo] | None


class FeatureInfo(_FeatureInfoRequired, total=False):
    """Normalized entity features and their detected representation."""

    graph_edge_features: int | None


def _process_graph_features(
    features: object,
    *,
    entity_name: Literal["instance", "target"],
) -> FeatureInfo:
    if isinstance(features, pd.DataFrame):
        feature_frame = features.copy(deep=True)
        if feature_frame.empty:
            raise ValueError(
                f"{entity_name.title()} features must contain at least one row."
            )
        if set(feature_frame.columns) != {"id", "features"}:
            raise ValueError(
                f"{entity_name.title()} graph features must contain exactly "
                "'id' and 'features' columns."
            )
        if feature_frame["id"].isna().any():
            raise ValueError(f"{entity_name.title()} feature IDs cannot be missing.")
        if feature_frame["id"].duplicated().any():
            raise ValueError(f"{entity_name.title()} feature IDs must be unique.")
        feature_frame = feature_frame.set_index("id", drop=False)
        rows = feature_frame["features"].tolist()
    elif isinstance(features, np.ndarray):
        if features.ndim != 1:
            raise ValueError(
                f"{entity_name.title()} graph arrays must be one-dimensional "
                "object arrays."
            )
        rows = list(features)
        feature_frame = pd.DataFrame({"id": np.arange(len(rows))})
    elif isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
        rows = list(features)
        feature_frame = pd.DataFrame({"id": np.arange(len(rows))})
    else:
        raise TypeError(
            f"{entity_name.title()} graph features must be a sequence of "
            "torch_geometric.data.Data objects or a DataFrame with 'id' and "
            "'features' columns."
        )

    normalized_rows, node_dimension, edge_dimension = normalize_graph_rows(
        rows,
        entity_name=entity_name,
    )
    feature_frame["features"] = graph_object_array(normalized_rows)
    return {
        "data": feature_frame,
        "num_features": node_dimension,
        "graph_edge_features": edge_dimension,
        "info": "graph",
        "components": None,
    }


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


def _process_sequence_features(
    features: object,
    *,
    entity_name: Literal["instance", "target"],
) -> FeatureInfo:
    if isinstance(features, pd.DataFrame):
        feature_frame = features.copy(deep=True)
        if feature_frame.empty:
            raise ValueError(
                f"{entity_name.title()} features must contain at least one row."
            )
        if set(feature_frame.columns) != {"id", "features"}:
            raise ValueError(
                f"{entity_name.title()} sequence features must contain exactly "
                "'id' and 'features' columns."
            )
        if feature_frame["id"].isna().any():
            raise ValueError(f"{entity_name.title()} feature IDs cannot be missing.")
        if feature_frame["id"].duplicated().any():
            raise ValueError(f"{entity_name.title()} feature IDs must be unique.")
        feature_frame = feature_frame.set_index("id", drop=False)
        rows = feature_frame["features"].tolist()
    elif isinstance(features, torch.Tensor):
        if features.ndim != 2 or 0 in features.shape:
            raise ValueError(
                f"{entity_name.title()} sequence tensors must have shape "
                "[entities, sequence_length] with no empty dimension."
            )
        rows = list(features)
        feature_frame = pd.DataFrame({"id": np.arange(len(rows))})
    elif isinstance(features, np.ndarray):
        if features.ndim not in {1, 2} or features.size == 0:
            raise ValueError(
                f"{entity_name.title()} sequence arrays must contain one "
                "non-empty row per entity."
            )
        rows = list(features)
        feature_frame = pd.DataFrame({"id": np.arange(len(rows))})
    elif isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
        rows = list(features)
        feature_frame = pd.DataFrame({"id": np.arange(len(rows))})
    else:
        raise TypeError(
            f"{entity_name.title()} sequence features must be a sequence collection, "
            "a two-dimensional integer tensor/array, or a DataFrame with 'id' and "
            "'features' columns."
        )

    normalized_rows, inferred_vocabulary_size = normalize_sequence_rows(
        rows,
        entity_name=entity_name,
    )
    feature_frame["features"] = sequence_object_array(normalized_rows)
    return {
        "data": feature_frame,
        "num_features": inferred_vocabulary_size,
        "info": "sequence",
        "components": None,
    }


def _normalize_composite_feature_specs(
    component_specs: Mapping[str, CompositeFeatureKind],
) -> dict[str, tuple[AtomicFeatureKind, bool]]:
    normalized: dict[str, tuple[AtomicFeatureKind, bool]] = {}
    for name, raw_spec in component_specs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Composite component names must be non-empty strings.")
        if isinstance(raw_spec, Mapping):
            unknown_keys = set(raw_spec).difference({"kind", "optional"})
            if unknown_keys:
                raise ValueError(
                    f"Composite component {name!r} feature specification contains "
                    f"unknown keys: {sorted(str(key) for key in unknown_keys)}."
                )
            feature_kind = raw_spec.get("kind")
            optional = raw_spec.get("optional", False)
        else:
            feature_kind = raw_spec
            optional = False
        if feature_kind not in {None, "graph", "sequence"}:
            raise ValueError(
                f"Composite component {name!r} feature kind must be 'graph', "
                f"'sequence', or None; received {feature_kind!r}."
            )
        if not isinstance(optional, bool):
            raise ValueError(
                f"Composite component {name!r} optional must be a boolean."
            )
        normalized[name] = (feature_kind, optional)
    return normalized


def _is_missing_component_value(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _prepare_optional_component_source(
    source: object,
    *,
    entity_name: Literal["instance", "target"],
    component_name: str,
) -> tuple[object, list[Any] | None]:
    if isinstance(source, pd.DataFrame) and "features" in source.columns:
        frame = source.copy(deep=True)
        if frame.empty:
            raise ValueError(
                f"Composite component {component_name!r} must contain at least "
                f"one {entity_name}."
            )
        if "id" not in frame.columns:
            return source, None
        if frame["id"].isna().any():
            raise ValueError(
                f"Composite component {component_name!r} IDs cannot be missing."
            )
        if frame["id"].duplicated().any():
            raise ValueError(
                f"Composite component {component_name!r} IDs must be unique."
            )
        missing = frame["features"].map(_is_missing_component_value)
        if not missing.any():
            return source, None
        present = frame.loc[~missing].copy(deep=True)
        if present.empty:
            raise ValueError(
                f"Optional composite component {component_name!r} must contain "
                "at least one present value so its input dimension can be inferred."
            )
        return present, frame["id"].tolist()

    rows: list[Any] | None = None
    if (isinstance(source, np.ndarray) and source.ndim == 1) or (
        isinstance(source, Sequence) and not isinstance(source, (str, bytes))
    ):
        rows = list(source)
    if rows is None:
        return source, None

    missing = [_is_missing_component_value(value) for value in rows]
    if not any(missing):
        return source, None
    present_ids = [index for index, is_missing in enumerate(missing) if not is_missing]
    if not present_ids:
        raise ValueError(
            f"Optional composite component {component_name!r} must contain at "
            "least one present value so its input dimension can be inferred."
        )
    present = pd.DataFrame(
        {
            "id": present_ids,
            "features": [rows[index] for index in present_ids],
        }
    )
    return present, list(range(len(rows)))


def _process_composite_features(
    features: object,
    *,
    component_kinds: Mapping[str, CompositeFeatureKind],
    entity_name: Literal["instance", "target"],
) -> FeatureInfo:
    if not isinstance(features, Mapping):
        raise TypeError(
            f"{entity_name.title()} composite features must be a mapping from "
            "component names to feature collections."
        )
    if not component_kinds:
        raise ValueError(
            f"{entity_name.title()} composite features must contain at least "
            "one side-feature component."
        )
    normalized_specs = _normalize_composite_feature_specs(component_kinds)
    if set(features) != set(component_kinds):
        raise ValueError(
            f"{entity_name.title()} composite feature sources and feature-kind "
            "declarations must use the same names: expected "
            f"{sorted(component_kinds)}, received {sorted(features)}."
        )

    processed_components: dict[str, FeatureInfo] = {}
    component_metadata: dict[str, FeatureComponentInfo] = {}
    component_source_ids: dict[str, list[Any]] = {}
    for name, (feature_kind, optional) in normalized_specs.items():
        component_source = features[name]
        declared_ids: list[Any] | None = None
        if optional:
            component_source, declared_ids = _prepare_optional_component_source(
                component_source,
                entity_name=entity_name,
                component_name=name,
            )
        component = _process_entity_features(
            component_source,
            entity_name,
            False,
            "basic",
            None,
            feature_kind,
        )
        if component is None:
            raise ValueError(
                f"Composite component {name!r} must provide feature values."
            )
        if component["info"] not in {
            "numpy",
            "dataframe",
            "graph",
            "sequence",
            "scipy_sparse",
            "torch_sparse",
        }:
            raise ValueError(
                f"Composite component {name!r} uses unsupported representation "
                f"{component['info']!r}. Composite branches currently support "
                "dense, graph, sparse, and token-sequence side features."
            )
        processed_components[name] = component
        component_metadata[name] = {
            "num_features": component["num_features"],
            "info": component["info"],
        }
        if component["info"] == "graph":
            component_metadata[name]["graph_edge_features"] = component.get(
                "graph_edge_features"
            )
        component_source_ids[name] = (
            declared_ids
            if declared_ids is not None
            else component["data"]["id"].tolist()
        )

    indexed_frames = {
        name: component["data"].set_index("id", drop=False)
        for name, component in processed_components.items()
    }
    required_names = [
        name for name, (_, optional) in normalized_specs.items() if not optional
    ]
    if required_names:
        first_name = required_names[0]
        ordered_ids = indexed_frames[first_name]["id"].tolist()
        expected_ids = set(ordered_ids)
        for name in required_names:
            component_ids = set(component_source_ids[name])
            if component_ids == expected_ids:
                continue
            missing_ids = sorted(expected_ids.difference(component_ids))
            unexpected_ids = sorted(component_ids.difference(expected_ids))
            raise ValueError(
                f"Composite component {name!r} must describe the same "
                f"{entity_name} IDs as {first_name!r}; missing {missing_ids}, "
                f"unexpected {unexpected_ids}."
            )
        for name, (_, optional) in normalized_specs.items():
            if not optional:
                continue
            unexpected_ids = sorted(
                set(component_source_ids[name]).difference(expected_ids)
            )
            if unexpected_ids:
                raise ValueError(
                    f"Optional composite component {name!r} contains unexpected "
                    f"{entity_name} IDs not present in required component "
                    f"{first_name!r}: {unexpected_ids}."
                )
    else:
        ordered_ids = []
        seen_ids: set[Any] = set()
        for name in normalized_specs:
            for entity_id in component_source_ids[name]:
                if entity_id not in seen_ids:
                    seen_ids.add(entity_id)
                    ordered_ids.append(entity_id)

    feature_frame = pd.DataFrame({"id": ordered_ids})
    feature_frame["features"] = [
        {
            name: (
                indexed_frames[name].at[entity_id, "features"]
                if entity_id in indexed_frames[name].index
                else None
            )
            for name in normalized_specs
        }
        for entity_id in ordered_ids
    ]
    feature_frame = feature_frame.set_index("id", drop=False)
    return {
        "data": feature_frame,
        "num_features": None,
        "info": "composite",
        "components": component_metadata,
    }


def _process_entity_features(
    features: object,
    entity_name: Literal["instance", "target"],
    verbose: bool,
    print_mode: str,
    progress: DataProgressObserver | None,
    feature_kind: FeatureKind,
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

    if isinstance(feature_kind, Mapping):
        feature_info = _process_composite_features(
            features,
            component_kinds=feature_kind,
            entity_name=entity_name,
        )
    elif feature_kind not in {None, "graph", "sequence"}:
        raise ValueError(
            f"{entity_name}_feature_kind must be 'graph', 'sequence', a component "
            "mapping, or None; "
            f"received {feature_kind!r}."
        )
    elif feature_kind == "graph":
        feature_info = _process_graph_features(
            features,
            entity_name=entity_name,
        )
    elif feature_kind == "sequence":
        feature_info = _process_sequence_features(
            features,
            entity_name=entity_name,
        )
    elif is_sparse_value(features):
        sparse_rows, feature_dimension, backend = normalize_sparse_matrix(
            features,
            entity_name=entity_name,
        )
        feature_frame = pd.DataFrame({"id": np.arange(len(sparse_rows))})
        feature_frame["features"] = sparse_object_array(sparse_rows)
        feature_info = {
            "data": feature_frame,
            "num_features": feature_dimension,
            "info": cast(
                Literal["scipy_sparse", "torch_sparse"],
                f"{backend}_sparse",
            ),
            "components": None,
        }
    elif isinstance(features, np.ndarray):
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
        feature_info = {
            "data": feature_frame,
            "num_features": int(features.shape[1]),
            "info": "numpy",
            "components": None,
        }
    elif isinstance(features, pd.DataFrame):
        feature_frame = features.copy()
        if feature_frame.empty:
            raise ValueError(
                f"{entity_name.title()} features must contain at least one row."
            )
        if "id" not in feature_frame.columns or len(feature_frame.columns) < 2:
            raise ValueError(
                f"Columns {list(feature_frame.columns)!r} in the {entity_name} "
                "features dataframe are not recognized. Expected an 'id' column "
                "and at least one feature column."
            )
        if feature_frame["id"].isna().any():
            raise ValueError(f"{entity_name.title()} feature IDs cannot be missing.")
        if feature_frame["id"].duplicated().any():
            raise ValueError(f"{entity_name.title()} feature IDs must be unique.")
        feature_frame = feature_frame.set_index("id", drop=False)
        if set(feature_frame.columns) == {"id", "features"}:
            feature_values = feature_frame["features"].tolist()
            sparse_flags = [is_sparse_value(value) for value in feature_values]
            if any(sparse_flags):
                if not all(sparse_flags):
                    raise TypeError(
                        f"{entity_name.title()} feature rows cannot mix dense and "
                        "sparse values."
                    )
                sparse_rows, feature_dimension, backend = normalize_sparse_rows(
                    feature_values,
                    entity_name=entity_name,
                )
                feature_frame["features"] = sparse_object_array(sparse_rows)
                feature_info = {
                    "data": feature_frame,
                    "num_features": feature_dimension,
                    "info": cast(
                        Literal["scipy_sparse", "torch_sparse"],
                        f"{backend}_sparse",
                    ),
                    "components": None,
                }
            else:
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
                    "components": None,
                }
        elif set(feature_frame.columns) == {"id", "dir"}:
            if feature_frame["dir"].isna().any():
                raise ValueError(
                    f"{entity_name.title()} feature directories cannot be missing."
                )
            feature_info = {
                "data": feature_frame,
                "num_features": None,
                "info": "images",
                "components": None,
            }
        else:
            feature_info = {
                "data": feature_frame,
                "num_features": len(feature_frame.columns) - 1,
                "info": "tabular",
                "components": None,
            }
    else:
        raise TypeError(
            f"Could not recognize the {entity_name} feature data. Expected a "
            "two-dimensional NumPy array, a SciPy sparse matrix, a PyTorch "
            "COO/CSR sparse tensor, a graph collection declared with "
            "feature_kind='graph', or a DataFrame with an 'id' column and "
            "feature columns."
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
    instance_features: Any,
    verbose: bool = False,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
    feature_kind: FeatureKind = None,
) -> FeatureInfo | None:
    """Normalize instance features without mutating caller-owned inputs.

    Composite component specifications may set ``optional=True`` to align a
    subset of entity IDs and preserve missing rows as ``None``.
    ``progress`` receives messages when ``verbose`` is enabled.
    """
    return _process_entity_features(
        instance_features,
        "instance",
        verbose,
        print_mode,
        progress,
        feature_kind,
    )


def process_target_features(
    target_features: Any,
    verbose: bool = False,
    print_mode: str = "basic",
    *,
    progress: DataProgressObserver | None = None,
    feature_kind: FeatureKind = None,
) -> FeatureInfo | None:
    """Normalize target features without mutating caller-owned inputs.

    Composite component specifications may set ``optional=True`` to align a
    subset of entity IDs and preserve missing rows as ``None``.
    ``progress`` receives messages when ``verbose`` is enabled.
    """
    return _process_entity_features(
        target_features,
        "target",
        verbose,
        print_mode,
        progress,
        feature_kind,
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
