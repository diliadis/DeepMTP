"""Compatibility facade for :mod:`DeepMTP.data.preparation`."""

from DeepMTP.data.preparation import (
    DataInfo,
    ProcessedSplit,
    ScalingMethod,
    Transformer,
    data_process,
    normalize,
    validate_split_ratio,
)

__all__ = [
    "DataInfo",
    "ProcessedSplit",
    "ScalingMethod",
    "Transformer",
    "data_process",
    "normalize",
    "validate_split_ratio",
]
