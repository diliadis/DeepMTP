"""Compatibility facade for :mod:`DeepMTP.data.interactions`."""

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

__all__ = [
    "DatasetMapping",
    "InteractionInfo",
    "NoveltyInferenceWarning",
    "ValidationSetting",
    "VariableType",
    "check_interaction_files_column_type_format",
    "check_interaction_files_format",
    "check_novel_instances",
    "check_novel_targets",
    "check_target_variable_type",
    "check_variable_type",
    "get_estimated_validation_setting",
    "process_interaction_data",
]
