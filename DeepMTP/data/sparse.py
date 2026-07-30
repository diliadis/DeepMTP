"""Sparse feature normalization and batching utilities."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from functools import lru_cache
from types import ModuleType
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import torch

SparseBackend: TypeAlias = Literal["scipy", "torch"]

_SUPPORTED_TORCH_SPARSE_LAYOUTS = frozenset(
    {
        torch.sparse_coo,
        torch.sparse_csr,
    }
)


@lru_cache(maxsize=1)
def _load_scipy_sparse() -> ModuleType | None:
    """Load SciPy's sparse module only when sparse SciPy input is used."""
    try:
        return importlib.import_module("scipy.sparse")
    except ModuleNotFoundError:
        return None


def is_scipy_sparse(value: object) -> bool:
    """Return whether ``value`` is a SciPy sparse matrix."""
    sparse_module = _load_scipy_sparse()
    return sparse_module is not None and bool(sparse_module.issparse(value))


def is_torch_sparse(value: object) -> bool:
    """Return whether ``value`` is a supported PyTorch sparse tensor."""
    return (
        isinstance(value, torch.Tensor)
        and value.layout in _SUPPORTED_TORCH_SPARSE_LAYOUTS
    )


def is_sparse_value(value: object) -> bool:
    """Return whether ``value`` uses a supported sparse representation."""
    return is_torch_sparse(value) or is_scipy_sparse(value)


def sparse_object_array(values: Sequence[object]) -> np.ndarray:
    """Return an object array without invoking sparse tensor array protocols."""
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def sparse_backend(value: object) -> SparseBackend:
    """Return the sparse backend used by ``value``."""
    if is_torch_sparse(value):
        return "torch"
    if is_scipy_sparse(value):
        return "scipy"
    if isinstance(value, torch.Tensor) and value.layout != torch.strided:
        raise TypeError(
            "PyTorch sparse inputs must use COO or CSR layout; "
            f"received {value.layout}."
        )
    raise TypeError(
        "Sparse features must be a SciPy sparse matrix or a PyTorch COO/CSR "
        "sparse tensor."
    )


def _validate_sparse_values(values: object, *, context: str) -> None:
    if isinstance(values, torch.Tensor):
        if values.is_complex():
            raise TypeError(f"{context} values must be real numeric values.")
        if values.dtype == torch.bool:
            raise TypeError(f"{context} values must be numeric, not boolean.")
        if values.numel() and not torch.isfinite(values.float()).all():
            raise ValueError(f"{context} values must be finite.")
        return

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise TypeError(f"{context} values must be real numeric values.")
    if array.dtype == np.bool_:
        raise TypeError(f"{context} values must be numeric, not boolean.")
    if array.size and not np.isfinite(array.astype(float, copy=False)).all():
        raise ValueError(f"{context} values must be finite.")


def normalize_sparse_matrix(
    matrix: object,
    *,
    entity_name: str,
) -> tuple[list[Any], int, SparseBackend]:
    """Validate a 2-D sparse matrix and return independent sparse rows."""
    backend = sparse_backend(matrix)
    shape = tuple(cast(Any, matrix).shape)
    if len(shape) != 2:
        raise ValueError(
            f"{entity_name.title()} sparse feature matrices must be two-dimensional."
        )
    if 0 in shape:
        raise ValueError(
            f"{entity_name.title()} sparse feature matrices cannot have an empty "
            "dimension."
        )

    if backend == "torch":
        tensor = cast(torch.Tensor, matrix)
        _validate_sparse_values(
            tensor.values()
            if tensor.layout == torch.sparse_csr
            else tensor.coalesce().values(),
            context=f"{entity_name.title()} sparse feature",
        )
        normalized = tensor.to_sparse_coo().coalesce()
        rows = [normalized[index].clone().coalesce() for index in range(shape[0])]
    else:
        sparse_module = _require_scipy_sparse()
        normalized = sparse_module.csr_matrix(matrix, copy=True)
        normalized.sum_duplicates()
        _validate_sparse_values(
            normalized.data,
            context=f"{entity_name.title()} sparse feature",
        )
        rows = [normalized.getrow(index) for index in range(shape[0])]
    return rows, int(shape[1]), backend


def normalize_sparse_rows(
    rows: Iterable[object],
    *,
    entity_name: str,
) -> tuple[list[Any], int, SparseBackend]:
    """Validate and copy sparse feature rows with one consistent width/backend."""
    row_values = list(rows)
    if not row_values:
        raise ValueError(
            f"{entity_name.title()} sparse features must contain at least one row."
        )
    backends = {sparse_backend(row) for row in row_values}
    if len(backends) != 1:
        raise TypeError(
            f"{entity_name.title()} sparse feature rows must all use the same "
            "SciPy or PyTorch backend."
        )
    backend = backends.pop()

    normalized_rows: list[Any] = []
    widths: set[int] = set()
    for row in row_values:
        if backend == "torch":
            tensor = cast(torch.Tensor, row)
            if tensor.ndim == 2:
                if tensor.shape[0] != 1:
                    raise ValueError(
                        f"Each {entity_name} sparse feature value must represent "
                        "one row."
                    )
                tensor = tensor[0]
            if tensor.ndim != 1:
                raise ValueError(
                    f"Each {entity_name} sparse feature value must be one-dimensional."
                )
            tensor = tensor.to_sparse_coo().coalesce()
            _validate_sparse_values(
                tensor.values(),
                context=f"{entity_name.title()} sparse feature",
            )
            widths.add(int(tensor.shape[0]))
            normalized_rows.append(tensor.clone())
        else:
            sparse_module = _require_scipy_sparse()
            matrix = sparse_module.csr_matrix(row, copy=True)
            if matrix.ndim != 2 or matrix.shape[0] != 1:
                raise ValueError(
                    f"Each {entity_name} sparse feature value must represent one row."
                )
            matrix.sum_duplicates()
            _validate_sparse_values(
                matrix.data,
                context=f"{entity_name.title()} sparse feature",
            )
            widths.add(int(matrix.shape[1]))
            normalized_rows.append(matrix)

    if 0 in widths:
        raise ValueError(f"{entity_name.title()} sparse feature rows cannot be empty.")
    if len(widths) != 1:
        raise ValueError(
            f"All {entity_name} sparse feature rows must have the same dimension."
        )
    return normalized_rows, widths.pop(), backend


def copy_feature_value(value: object) -> Any:
    """Copy one dense or sparse feature value without densifying it."""
    if is_torch_sparse(value):
        return cast(torch.Tensor, value).clone()
    if is_scipy_sparse(value):
        return cast(Any, value).copy()
    return np.asarray(value).copy()


def collate_sparse_rows(rows: Sequence[object]) -> torch.Tensor:
    """Stack sparse rows into one floating-point PyTorch COO batch."""
    normalized_rows, width, backend = normalize_sparse_rows(
        rows,
        entity_name="branch",
    )
    if backend == "torch":
        return torch.stack(
            [cast(torch.Tensor, row).float() for row in normalized_rows]
        ).coalesce()

    sparse_module = _require_scipy_sparse()
    stacked = sparse_module.vstack(normalized_rows, format="coo", dtype=np.float32)
    indices = torch.from_numpy(
        np.vstack((stacked.row, stacked.col)).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(stacked.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(len(normalized_rows), width),
    ).coalesce()


def pin_sparse_tensor(values: torch.Tensor) -> torch.Tensor:
    """Pin sparse tensor components because ``Tensor.pin_memory`` lacks support."""
    if values.layout == torch.sparse_coo:
        coalesced = values.coalesce()
        return torch.sparse_coo_tensor(
            coalesced.indices().pin_memory(),
            coalesced.values().pin_memory(),
            size=coalesced.shape,
        ).coalesce()
    if values.layout == torch.sparse_csr:
        return torch.sparse_csr_tensor(
            values.crow_indices().pin_memory(),
            values.col_indices().pin_memory(),
            values.values().pin_memory(),
            size=values.shape,
        )
    raise TypeError("Sparse batches must use PyTorch COO or CSR layout before pinning.")


def _require_scipy_sparse() -> ModuleType:
    sparse_module = _load_scipy_sparse()
    if sparse_module is None:
        raise ModuleNotFoundError(
            "SciPy sparse inputs require the optional 'sparse' dependencies. "
            "Install them with `pip install 'DeepMTP[sparse]'`."
        )
    return sparse_module
