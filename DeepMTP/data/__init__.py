"""Data loading, preparation, validation, and progress services."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from DeepMTP.data.features import (
    FeatureInfo,
    cross_input_consistency_check_instances,
    cross_input_consistency_check_targets,
    process_instance_features,
    process_target_features,
)
from DeepMTP.data.interactions import (
    DatasetMapping,
    InteractionInfo,
    NoveltyInferenceWarning,
    ValidationSetting,
    VariableType,
    check_interaction_files_column_type_format,
    check_interaction_files_format,
    check_novel_instances,
    check_novel_targets,
    check_target_variable_type,
    check_variable_type,
    get_estimated_validation_setting,
    process_interaction_data,
)
from DeepMTP.data.loading import (
    BaseDataset,
    DataLoaderFactory,
    DatasetItem,
    MTPSampler,
    SamplerMode,
    TrainingDataLoaders,
)
from DeepMTP.data.preparation import (
    DataInfo,
    ProcessedSplit,
    ScalingMethod,
    Transformer,
    data_process,
    normalize,
    validate_split_ratio,
)
from DeepMTP.data.progress import (
    ConsoleDataProgressObserver,
    DataPrintMode,
    DataProgressEvent,
    DataProgressEventKind,
    DataProgressLevel,
    DataProgressObserver,
    NullDataProgressObserver,
    build_data_progress_observer,
)
from DeepMTP.data.splitting import split_data

if TYPE_CHECKING:
    from DeepMTP.data.datasets import (
        ArrayFormat,
        DatasetBundle,
        DatasetSplit,
        DatasetValue,
        DatasetVariant,
        PrintMode,
        ProcessedDatasetBundle,
        ProcessedDatasetSplit,
        ProcessedDatasetValue,
        format_mtr_datasets,
        generate_dummy_dataset,
        generate_interaction_matrix,
        generate_MTP_dataset,
        load_process_DP,
        load_process_MC,
        load_process_MLC,
        load_process_MTL,
        load_process_MTR,
        print_MTR_datasets,
        process_dummy_DP,
        process_dummy_MLC,
        process_dummy_MTR,
    )

_DATASET_EXPORTS = frozenset(
    {
        "ArrayFormat",
        "DatasetBundle",
        "DatasetSplit",
        "DatasetValue",
        "DatasetVariant",
        "PrintMode",
        "ProcessedDatasetBundle",
        "ProcessedDatasetSplit",
        "ProcessedDatasetValue",
        "format_mtr_datasets",
        "generate_MTP_dataset",
        "generate_dummy_dataset",
        "generate_interaction_matrix",
        "load_process_DP",
        "load_process_MC",
        "load_process_MLC",
        "load_process_MTL",
        "load_process_MTR",
        "print_MTR_datasets",
        "process_dummy_DP",
        "process_dummy_MLC",
        "process_dummy_MTR",
    }
)

__all__ = [
    "ArrayFormat",
    "BaseDataset",
    "ConsoleDataProgressObserver",
    "DataInfo",
    "DataLoaderFactory",
    "DataPrintMode",
    "DataProgressEvent",
    "DataProgressEventKind",
    "DataProgressLevel",
    "DataProgressObserver",
    "DatasetBundle",
    "DatasetItem",
    "DatasetMapping",
    "DatasetSplit",
    "DatasetValue",
    "DatasetVariant",
    "FeatureInfo",
    "InteractionInfo",
    "MTPSampler",
    "NoveltyInferenceWarning",
    "NullDataProgressObserver",
    "PrintMode",
    "ProcessedDatasetBundle",
    "ProcessedDatasetSplit",
    "ProcessedDatasetValue",
    "ProcessedSplit",
    "SamplerMode",
    "ScalingMethod",
    "TrainingDataLoaders",
    "Transformer",
    "ValidationSetting",
    "VariableType",
    "build_data_progress_observer",
    "check_interaction_files_column_type_format",
    "check_interaction_files_format",
    "check_novel_instances",
    "check_novel_targets",
    "check_target_variable_type",
    "check_variable_type",
    "cross_input_consistency_check_instances",
    "cross_input_consistency_check_targets",
    "data_process",
    "format_mtr_datasets",
    "generate_MTP_dataset",
    "generate_dummy_dataset",
    "generate_interaction_matrix",
    "get_estimated_validation_setting",
    "load_process_DP",
    "load_process_MC",
    "load_process_MLC",
    "load_process_MTL",
    "load_process_MTR",
    "normalize",
    "print_MTR_datasets",
    "process_dummy_DP",
    "process_dummy_MLC",
    "process_dummy_MTR",
    "process_instance_features",
    "process_interaction_data",
    "process_target_features",
    "split_data",
    "validate_split_ratio",
]


def __getattr__(name: str) -> Any:
    """Load optional dataset helpers only when requested."""
    if name in _DATASET_EXPORTS:
        datasets = import_module("DeepMTP.data.datasets")
        value = getattr(datasets, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return eager and lazy public data attributes."""
    return sorted(set(globals()) | _DATASET_EXPORTS)
