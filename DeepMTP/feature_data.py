"""Compatibility facade for :mod:`DeepMTP.data.features`."""

from DeepMTP.data.features import (
    FeatureComponentInfo,
    FeatureInfo,
    FeatureKind,
    cross_input_consistency_check_instances,
    cross_input_consistency_check_targets,
    process_instance_features,
    process_target_features,
)

__all__ = [
    "FeatureKind",
    "FeatureComponentInfo",
    "FeatureInfo",
    "cross_input_consistency_check_instances",
    "cross_input_consistency_check_targets",
    "process_instance_features",
    "process_target_features",
]
