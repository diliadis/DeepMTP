"""Schemas and reproducible preprocessing for mixed tabular branch inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd

__all__ = [
    "CategoricalColumn",
    "CategoricalValue",
    "TabularPreprocessingState",
    "TabularPreprocessor",
    "TabularSchema",
]

CategoricalValue: TypeAlias = str | int
NumericNormalization: TypeAlias = Literal["none", "standard", "minmax"]
NumericMissingPolicy: TypeAlias = Literal["error", "mean", "zero"]
CategoricalValuePolicy: TypeAlias = Literal["error", "unknown"]
_PREPROCESSING_STATE_VERSION = 1


@dataclass(frozen=True)
class CategoricalColumn:
    """One categorical feature and its fixed training vocabulary."""

    name: str
    categories: tuple[CategoricalValue, ...]
    embedding_dim: int

    @classmethod
    def from_config(cls, name: object, value: object) -> CategoricalColumn:
        """Validate one categorical-column declaration."""
        if not isinstance(name, str) or not name:
            raise ValueError("Categorical column names must be non-empty strings.")

        if isinstance(value, Mapping):
            categories = value.get("categories")
            embedding_dim = value.get("embedding_dim")
        else:
            categories = value
            embedding_dim = None

        if (
            isinstance(categories, (str, bytes))
            or not isinstance(categories, Sequence)
            or not categories
        ):
            raise ValueError(
                f"Categorical column {name!r} must declare a non-empty "
                "'categories' sequence."
            )

        validated_categories: list[CategoricalValue] = []
        for category in categories:
            if isinstance(category, bool) or not isinstance(category, (str, int)):
                raise TypeError(
                    f"Categories for {name!r} must be strings or integers; "
                    f"received {category!r}."
                )
            if category in validated_categories:
                raise ValueError(
                    f"Categories for {name!r} must be unique; "
                    f"received duplicate {category!r}."
                )
            validated_categories.append(category)

        if embedding_dim is None:
            embedding_dim = min(16, max(1, math.ceil(math.sqrt(len(categories) + 1))))
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim <= 0
        ):
            raise ValueError(
                f"embedding_dim for categorical column {name!r} must be a "
                "positive integer."
            )
        return cls(name, tuple(validated_categories), embedding_dim)

    @property
    def vocabulary_size(self) -> int:
        """Embedding-table size, including index zero for unknown values."""
        return len(self.categories) + 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible public configuration mapping."""
        return {
            "categories": list(self.categories),
            "embedding_dim": self.embedding_dim,
        }


@dataclass(frozen=True)
class TabularSchema:
    """Explicit numeric and categorical layout for one entity branch."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[CategoricalColumn, ...]
    numeric_normalization: NumericNormalization = "standard"
    numeric_missing: NumericMissingPolicy = "mean"
    categorical_missing: CategoricalValuePolicy = "unknown"
    categorical_unknown: CategoricalValuePolicy = "unknown"
    feature_gating: bool = False

    @classmethod
    def from_config(cls, value: object, *, branch: str) -> TabularSchema:
        """Validate a public tabular schema mapping."""
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{branch}_branch_tabular_schema must be a mapping for a "
                "TABULAR branch."
            )

        raw_numeric = value.get("numeric_columns", ())
        if isinstance(raw_numeric, (str, bytes)) or not isinstance(
            raw_numeric, Sequence
        ):
            raise TypeError("numeric_columns must be a sequence of column names.")
        numeric_columns: list[str] = []
        for name in raw_numeric:
            if not isinstance(name, str) or not name:
                raise ValueError("numeric_columns must contain non-empty string names.")
            if name in numeric_columns:
                raise ValueError(f"numeric_columns contains duplicate {name!r}.")
            numeric_columns.append(name)

        raw_categorical = value.get("categorical_columns", {})
        if not isinstance(raw_categorical, Mapping):
            raise TypeError(
                "categorical_columns must map column names to category declarations."
            )
        categorical_columns = tuple(
            CategoricalColumn.from_config(name, declaration)
            for name, declaration in raw_categorical.items()
        )

        categorical_names = [column.name for column in categorical_columns]
        overlap = set(numeric_columns).intersection(categorical_names)
        if overlap:
            raise ValueError(
                "Tabular columns cannot be both numeric and categorical: "
                f"{sorted(overlap)}."
            )
        if not numeric_columns and not categorical_columns:
            raise ValueError(
                "A tabular schema must declare at least one numeric or "
                "categorical column."
            )

        numeric_normalization = value.get("numeric_normalization", "standard")
        if numeric_normalization not in {"none", "standard", "minmax"}:
            raise ValueError(
                "numeric_normalization must be 'none', 'standard', or 'minmax'."
            )
        numeric_missing = value.get("numeric_missing", "mean")
        if numeric_missing not in {"error", "mean", "zero"}:
            raise ValueError("numeric_missing must be 'error', 'mean', or 'zero'.")
        categorical_missing = value.get("categorical_missing", "unknown")
        if categorical_missing not in {"error", "unknown"}:
            raise ValueError("categorical_missing must be 'error' or 'unknown'.")
        categorical_unknown = value.get("categorical_unknown", "unknown")
        if categorical_unknown not in {"error", "unknown"}:
            raise ValueError("categorical_unknown must be 'error' or 'unknown'.")
        feature_gating = value.get("feature_gating", False)
        if not isinstance(feature_gating, bool):
            raise TypeError("feature_gating must be a boolean.")

        return cls(
            numeric_columns=tuple(numeric_columns),
            categorical_columns=categorical_columns,
            numeric_normalization=cast(
                NumericNormalization,
                numeric_normalization,
            ),
            numeric_missing=cast(NumericMissingPolicy, numeric_missing),
            categorical_missing=cast(
                CategoricalValuePolicy,
                categorical_missing,
            ),
            categorical_unknown=cast(
                CategoricalValuePolicy,
                categorical_unknown,
            ),
            feature_gating=feature_gating,
        )

    @property
    def categorical_names(self) -> tuple[str, ...]:
        """Categorical column names in their stable tensor order."""
        return tuple(column.name for column in self.categorical_columns)

    @property
    def encoded_width(self) -> int:
        """Width after concatenating numeric values and category embeddings."""
        return len(self.numeric_columns) + sum(
            column.embedding_dim for column in self.categorical_columns
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible public configuration mapping."""
        return {
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": {
                column.name: column.to_dict() for column in self.categorical_columns
            },
            "numeric_normalization": self.numeric_normalization,
            "numeric_missing": self.numeric_missing,
            "categorical_missing": self.categorical_missing,
            "categorical_unknown": self.categorical_unknown,
            "feature_gating": self.feature_gating,
        }


@dataclass(frozen=True)
class TabularPreprocessingState:
    """Numeric statistics fitted only on training entities."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_fill_values: tuple[float, ...]
    numeric_offsets: tuple[float, ...]
    numeric_scales: tuple[float, ...]

    @classmethod
    def from_config(
        cls,
        value: object,
        *,
        schema: TabularSchema,
        branch: str,
    ) -> TabularPreprocessingState:
        """Restore and validate preprocessing metadata from a checkpoint config."""
        if not isinstance(value, Mapping):
            raise TypeError(f"{branch}_tabular_preprocessing_state must be a mapping.")
        version = value.get("version", _PREPROCESSING_STATE_VERSION)
        if version != _PREPROCESSING_STATE_VERSION:
            raise ValueError(
                f"Unsupported tabular preprocessing state version {version!r}."
            )

        numeric_columns = _string_tuple(value.get("numeric_columns"), "numeric_columns")
        categorical_columns = _string_tuple(
            value.get("categorical_columns"),
            "categorical_columns",
        )
        if numeric_columns != schema.numeric_columns:
            raise ValueError(
                f"{branch} tabular preprocessing numeric columns do not match "
                "the configured schema."
            )
        if categorical_columns != schema.categorical_names:
            raise ValueError(
                f"{branch} tabular preprocessing categorical columns do not "
                "match the configured schema."
            )

        expected_width = len(schema.numeric_columns)
        fill_values = _float_tuple(
            value.get("numeric_fill_values"),
            "numeric_fill_values",
            expected_width,
        )
        offsets = _float_tuple(
            value.get("numeric_offsets"),
            "numeric_offsets",
            expected_width,
        )
        scales = _float_tuple(
            value.get("numeric_scales"),
            "numeric_scales",
            expected_width,
        )
        if any(scale <= 0 for scale in scales):
            raise ValueError("numeric_scales must contain positive values.")
        return cls(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            numeric_fill_values=fill_values,
            numeric_offsets=offsets,
            numeric_scales=scales,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible checkpoint metadata."""
        return {
            "version": _PREPROCESSING_STATE_VERSION,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "numeric_fill_values": list(self.numeric_fill_values),
            "numeric_offsets": list(self.numeric_offsets),
            "numeric_scales": list(self.numeric_scales),
        }


@dataclass(frozen=True)
class TabularPreprocessor:
    """Transform entity feature frames using one schema and fitted state."""

    schema: TabularSchema
    state: TabularPreprocessingState
    branch: str

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        schema: TabularSchema,
        *,
        branch: str,
    ) -> TabularPreprocessor:
        """Fit numeric imputation and normalization on a training frame."""
        _validate_required_columns(frame, schema, branch=branch)
        numeric = _numeric_values(frame, schema, branch=branch)

        if numeric.shape[1] == 0:
            fill_values = offsets = np.empty(0, dtype=np.float64)
            scales = np.empty(0, dtype=np.float64)
        else:
            missing = np.isnan(numeric)
            if schema.numeric_missing == "error" and missing.any():
                raise ValueError(
                    f"{branch} tabular numeric features contain missing values "
                    "while numeric_missing='error'."
                )
            if schema.numeric_missing == "mean":
                with np.errstate(invalid="ignore"):
                    fill_values = np.nanmean(numeric, axis=0)
                if np.isnan(fill_values).any():
                    column_index = int(np.flatnonzero(np.isnan(fill_values))[0])
                    raise ValueError(
                        f"{branch} numeric column "
                        f"{schema.numeric_columns[column_index]!r} contains only "
                        "missing values and cannot use mean imputation."
                    )
            else:
                fill_values = np.zeros(numeric.shape[1], dtype=np.float64)

            imputed = np.where(missing, fill_values, numeric)
            if schema.numeric_normalization == "standard":
                offsets = imputed.mean(axis=0)
                scales = imputed.std(axis=0)
            elif schema.numeric_normalization == "minmax":
                offsets = imputed.min(axis=0)
                scales = imputed.max(axis=0) - offsets
            else:
                offsets = np.zeros(imputed.shape[1], dtype=np.float64)
                scales = np.ones(imputed.shape[1], dtype=np.float64)
            scales = np.where(scales == 0, 1.0, scales)

        state = TabularPreprocessingState(
            numeric_columns=schema.numeric_columns,
            categorical_columns=schema.categorical_names,
            numeric_fill_values=tuple(float(value) for value in fill_values),
            numeric_offsets=tuple(float(value) for value in offsets),
            numeric_scales=tuple(float(value) for value in scales),
        )
        return cls(schema=schema, state=state, branch=branch)

    @classmethod
    def from_state(
        cls,
        schema: TabularSchema,
        state: Mapping[str, Any] | TabularPreprocessingState,
        *,
        branch: str,
    ) -> TabularPreprocessor:
        """Build a preprocessor from saved training metadata."""
        restored_state = (
            state
            if isinstance(state, TabularPreprocessingState)
            else TabularPreprocessingState.from_config(
                state,
                schema=schema,
                branch=branch,
            )
        )
        return cls(schema=schema, state=restored_state, branch=branch)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return an indexed feature frame with structured tensor-ready values."""
        _validate_required_columns(frame, self.schema, branch=self.branch)
        numeric = _numeric_values(frame, self.schema, branch=self.branch)
        if numeric.shape[1]:
            missing = np.isnan(numeric)
            if self.schema.numeric_missing == "error" and missing.any():
                raise ValueError(
                    f"{self.branch} tabular numeric features contain missing "
                    "values while numeric_missing='error'."
                )
            fill_values = np.asarray(
                self.state.numeric_fill_values,
                dtype=np.float64,
            )
            numeric = np.where(missing, fill_values, numeric)
            numeric = (
                numeric - np.asarray(self.state.numeric_offsets, dtype=np.float64)
            ) / np.asarray(self.state.numeric_scales, dtype=np.float64)
        numeric = numeric.astype(np.float32, copy=False)

        categorical = np.empty(
            (len(frame), len(self.schema.categorical_columns)),
            dtype=np.int64,
        )
        for column_index, column in enumerate(self.schema.categorical_columns):
            vocabulary = {
                value: category_index
                for category_index, value in enumerate(column.categories, start=1)
            }
            categorical[:, column_index] = [
                self._encode_category(column.name, raw_value, vocabulary)
                for raw_value in frame[column.name]
            ]

        prepared = frame.loc[:, ["id"]].copy(deep=True)
        prepared["features"] = [
            {
                "numeric": numeric[row_index].copy(),
                "categorical": categorical[row_index].copy(),
            }
            for row_index in range(len(prepared))
        ]
        return prepared.set_index("id", drop=False)

    def _encode_category(
        self,
        column_name: str,
        value: object,
        vocabulary: Mapping[CategoricalValue, int],
    ) -> int:
        if _is_missing_scalar(value):
            if self.schema.categorical_missing == "unknown":
                return 0
            raise ValueError(
                f"{self.branch} categorical column {column_name!r} contains a "
                "missing value while categorical_missing='error'."
            )
        try:
            encoded = vocabulary.get(cast(CategoricalValue, value))
        except TypeError as exc:
            raise TypeError(
                f"{self.branch} categorical column {column_name!r} contains an "
                f"unhashable value {value!r}."
            ) from exc
        if encoded is not None:
            return encoded
        if self.schema.categorical_unknown == "unknown":
            return 0
        raise ValueError(
            f"{self.branch} categorical column {column_name!r} contains unknown "
            f"value {value!r}."
        )


def _validate_required_columns(
    frame: object,
    schema: TabularSchema,
    *,
    branch: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{branch} tabular features must be a pandas DataFrame.")
    required = {"id", *schema.numeric_columns, *schema.categorical_names}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{branch} tabular features are missing schema columns: {sorted(missing)}."
        )


def _numeric_values(
    frame: pd.DataFrame,
    schema: TabularSchema,
    *,
    branch: str,
) -> np.ndarray:
    if not schema.numeric_columns:
        return np.empty((len(frame), 0), dtype=np.float64)
    try:
        numeric = frame.loc[:, list(schema.numeric_columns)].to_numpy(
            dtype=np.float64,
            copy=True,
            na_value=np.nan,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{branch} tabular numeric columns must contain numbers or missing values."
        ) from exc
    if np.isinf(numeric).any():
        raise ValueError(f"{branch} tabular numeric columns cannot contain infinity.")
    return numeric


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(not isinstance(item, str) for item in value)
    ):
        raise TypeError(f"{name} must be a sequence of strings.")
    return tuple(value)


def _float_tuple(value: object, name: str, expected_width: int) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != expected_width
    ):
        raise ValueError(f"{name} must contain {expected_width} values.")
    converted: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain numeric values.")
        converted_item = float(item)
        if not math.isfinite(converted_item):
            raise ValueError(f"{name} values must be finite.")
        converted.append(converted_item)
    return tuple(converted)


def _is_missing_scalar(value: object) -> bool:
    """Return whether a scalar is missing without accepting array-like values."""
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)
