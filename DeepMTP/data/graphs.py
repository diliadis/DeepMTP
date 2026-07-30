"""Optional PyTorch Geometric normalization and batching helpers."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Any, cast

import numpy as np
import torch


@dataclass(frozen=True)
class CollatedGraph:
    """Framework-neutral tensors extracted from one PyG graph batch."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor | None
    graph_index: torch.Tensor
    num_graphs: int


@lru_cache(maxsize=1)
def _load_pyg_data() -> ModuleType | None:
    try:
        return importlib.import_module("torch_geometric.data")
    except (ImportError, ModuleNotFoundError):
        return None


def require_pyg_data() -> ModuleType:
    """Return PyG's data module or explain how to install graph support."""
    data_module = _load_pyg_data()
    if data_module is None:
        raise ModuleNotFoundError(
            "Graph inputs require the optional 'graph' dependencies. "
            "Install them with `pip install 'DeepMTP[graph]'`."
        )
    return data_module


def is_graph_value(value: object) -> bool:
    """Return whether ``value`` is a PyTorch Geometric graph object."""
    data_module = _load_pyg_data()
    return data_module is not None and isinstance(value, data_module.Data)


def graph_object_array(values: Sequence[object]) -> np.ndarray:
    """Store graph objects without invoking third-party array protocols."""
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def normalize_graph_rows(
    rows: Iterable[object],
    *,
    entity_name: str,
) -> tuple[list[Any], int, int | None]:
    """Validate homogeneous PyG graphs and return independent normalized copies."""
    data_module = require_pyg_data()
    graph_rows = list(rows)
    if not graph_rows:
        raise ValueError(
            f"{entity_name.title()} graph features must contain at least one graph."
        )

    normalized: list[Any] = []
    node_dimensions: set[int] = set()
    edge_dimensions: set[int | None] = set()
    for row_index, graph in enumerate(graph_rows):
        if not isinstance(graph, data_module.Data) or isinstance(
            graph,
            data_module.Batch,
        ):
            raise TypeError(
                f"{entity_name.title()} graph row {row_index} must be a "
                "torch_geometric.data.Data object representing one graph."
            )
        try:
            graph.validate(raise_on_error=True)
        except ValueError as exc:
            raise ValueError(
                f"{entity_name.title()} graph row {row_index} is invalid: {exc}"
            ) from exc

        node_features = getattr(graph, "x", None)
        if not isinstance(node_features, torch.Tensor):
            raise TypeError(
                f"{entity_name.title()} graph row {row_index} must define tensor x."
            )
        if node_features.ndim != 2 or 0 in node_features.shape:
            raise ValueError(
                f"{entity_name.title()} graph row {row_index} x must have shape "
                "[nodes, node_features] with no empty dimension."
            )
        _validate_real_finite_tensor(
            node_features,
            context=f"{entity_name.title()} graph row {row_index} node features",
        )

        edge_index = getattr(graph, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(
                f"{entity_name.title()} graph row {row_index} must define tensor "
                "edge_index."
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"{entity_name.title()} graph row {row_index} edge_index must "
                "have shape [2, edges]."
            )
        if edge_index.dtype not in {torch.int32, torch.int64}:
            raise TypeError(
                f"{entity_name.title()} graph row {row_index} edge_index must "
                "use an integer dtype."
            )

        edge_features = getattr(graph, "edge_attr", None)
        edge_dimension: int | None = None
        if edge_features is not None:
            if not isinstance(edge_features, torch.Tensor):
                raise TypeError(
                    f"{entity_name.title()} graph row {row_index} edge_attr must "
                    "be a tensor when supplied."
                )
            if (
                edge_features.ndim != 2
                or edge_features.shape[0] != edge_index.shape[1]
                or edge_features.shape[1] == 0
            ):
                raise ValueError(
                    f"{entity_name.title()} graph row {row_index} edge_attr must "
                    "have shape [edges, edge_features] with a positive feature width."
                )
            _validate_real_finite_tensor(
                edge_features,
                context=f"{entity_name.title()} graph row {row_index} edge features",
            )
            edge_dimension = int(edge_features.shape[1])

        node_dimensions.add(int(node_features.shape[1]))
        edge_dimensions.add(edge_dimension)
        normalized.append(
            data_module.Data(
                x=node_features.detach().clone().float(),
                edge_index=edge_index.detach().clone().long(),
                edge_attr=(
                    None
                    if edge_features is None
                    else edge_features.detach().clone().float()
                ),
            )
        )

    if len(node_dimensions) != 1:
        raise ValueError(
            f"All {entity_name} graphs must use the same node-feature width."
        )
    if len(edge_dimensions) != 1:
        raise ValueError(
            f"All {entity_name} graphs must either omit edge features or use "
            "the same edge-feature width."
        )
    return normalized, node_dimensions.pop(), edge_dimensions.pop()


def copy_graph_value(value: object) -> Any:
    """Clone one validated graph without sharing its tensors."""
    if not is_graph_value(value):
        raise TypeError("Graph feature values must be PyTorch Geometric Data objects.")
    return cast(Any, value).clone()


def collate_graph_rows(rows: Sequence[object]) -> CollatedGraph:
    """Batch graph rows with PyG and expose only DeepMTP's tensor contract."""
    data_module = require_pyg_data()
    normalized, _, _ = normalize_graph_rows(rows, entity_name="branch")
    batch = data_module.Batch.from_data_list(normalized)
    graph_index = getattr(batch, "batch", None)
    if not isinstance(graph_index, torch.Tensor):  # pragma: no cover - PyG invariant
        raise RuntimeError("PyG graph batching did not produce a graph index.")
    return CollatedGraph(
        node_features=batch.x,
        edge_index=batch.edge_index,
        edge_features=batch.edge_attr,
        graph_index=graph_index,
        num_graphs=int(batch.num_graphs),
    )


def _validate_real_finite_tensor(values: torch.Tensor, *, context: str) -> None:
    if values.is_complex() or values.dtype == torch.bool:
        raise TypeError(f"{context} must contain real numeric values.")
    if values.numel() and not torch.isfinite(values.float()).all():
        raise ValueError(f"{context} must contain only finite values.")
