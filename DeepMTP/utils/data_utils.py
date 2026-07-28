"""Compatibility façade for the historical data utility module.

New code should import from the focused data-loading, preparation, splitting,
feature, and interaction modules.
"""

from DeepMTP.data.features import FeatureInfo as FeatureInfo
from DeepMTP.data.features import (
    _cross_input_source_layout as _cross_input_source_layout,
)
from DeepMTP.data.features import _feature_vector_size as _feature_vector_size
from DeepMTP.data.features import _process_entity_features as _process_entity_features
from DeepMTP.data.features import (
    cross_input_consistency_check_instances as cross_input_consistency_check_instances,
)
from DeepMTP.data.features import (
    cross_input_consistency_check_targets as cross_input_consistency_check_targets,
)
from DeepMTP.data.features import (
    process_instance_features as process_instance_features,
)
from DeepMTP.data.features import process_target_features as process_target_features
from DeepMTP.data.interactions import DatasetMapping as DatasetMapping
from DeepMTP.data.interactions import InteractionInfo as InteractionInfo
from DeepMTP.data.interactions import (
    NoveltyInferenceWarning as NoveltyInferenceWarning,
)
from DeepMTP.data.interactions import ValidationSetting as ValidationSetting
from DeepMTP.data.interactions import VariableType as VariableType
from DeepMTP.data.interactions import (
    _check_novel_entities as _check_novel_entities,
)
from DeepMTP.data.interactions import (
    check_interaction_files_column_type_format as check_interaction_files_column_type_format,
)
from DeepMTP.data.interactions import (
    check_interaction_files_format as check_interaction_files_format,
)
from DeepMTP.data.interactions import (
    check_novel_instances as check_novel_instances,
)
from DeepMTP.data.interactions import check_novel_targets as check_novel_targets
from DeepMTP.data.interactions import (
    check_target_variable_type as check_target_variable_type,
)
from DeepMTP.data.interactions import check_variable_type as check_variable_type
from DeepMTP.data.interactions import (
    get_estimated_validation_setting as get_estimated_validation_setting,
)
from DeepMTP.data.interactions import (
    process_interaction_data as process_interaction_data,
)
from DeepMTP.data.loading import BaseDataset as BaseDataset
from DeepMTP.data.loading import DatasetItem as DatasetItem
from DeepMTP.data.loading import MTPSampler as MTPSampler
from DeepMTP.data.loading import SamplerMode as SamplerMode
from DeepMTP.data.preparation import DataInfo as DataInfo
from DeepMTP.data.preparation import ProcessedSplit as ProcessedSplit
from DeepMTP.data.preparation import ScalingMethod as ScalingMethod
from DeepMTP.data.preparation import Transformer as Transformer
from DeepMTP.data.preparation import (
    _normalize_scaling_method as _normalize_scaling_method,
)
from DeepMTP.data.preparation import (
    _normalize_validation_setting as _normalize_validation_setting,
)
from DeepMTP.data.preparation import (
    _prepare_dataset_mapping as _prepare_dataset_mapping,
)
from DeepMTP.data.preparation import (
    _scale_feature_splits as _scale_feature_splits,
)
from DeepMTP.data.preparation import data_process as data_process
from DeepMTP.data.preparation import normalize as normalize
from DeepMTP.data.preparation import validate_split_ratio as validate_split_ratio
from DeepMTP.data.progress import DataProgressObserver as DataProgressObserver
from DeepMTP.data.progress import _emit_data_message as _emit_data_message
from DeepMTP.data.progress import (
    _emit_data_operation_completed as _emit_data_operation_completed,
)
from DeepMTP.data.progress import (
    _emit_data_operation_completed_if_verbose as _emit_data_operation_completed_if_verbose,
)
from DeepMTP.data.progress import (
    _emit_data_operation_started as _emit_data_operation_started,
)
from DeepMTP.data.progress import (
    _emit_data_operation_started_if_verbose as _emit_data_operation_started_if_verbose,
)
from DeepMTP.data.progress import _emit_data_separator as _emit_data_separator
from DeepMTP.data.progress import _resolve_data_progress as _resolve_data_progress
from DeepMTP.data.splitting import _copy_feature_source as _copy_feature_source
from DeepMTP.data.splitting import (
    _detach_shared_feature_sources as _detach_shared_feature_sources,
)
from DeepMTP.data.splitting import _emit_split_completed as _emit_split_completed
from DeepMTP.data.splitting import _emit_split_started as _emit_split_started
from DeepMTP.data.splitting import (
    _interaction_axis_size as _interaction_axis_size,
)
from DeepMTP.data.splitting import _interaction_pairs as _interaction_pairs
from DeepMTP.data.splitting import (
    _interaction_row_count as _interaction_row_count,
)
from DeepMTP.data.splitting import _split_frame_rows as _split_frame_rows
from DeepMTP.data.splitting import _validate_split_result as _validate_split_result
from DeepMTP.data.splitting import split_data as split_data
