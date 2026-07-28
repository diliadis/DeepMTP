import ast
import inspect
import pickle
import subprocess
import sys

import pytest

import DeepMTP as deepmtp_api
import DeepMTP.data as data_api
import DeepMTP.data.features as features_module
import DeepMTP.data.interactions as interactions_module
import DeepMTP.data.loading as loading_module
import DeepMTP.data.preparation as preparation_module
import DeepMTP.data.progress as progress_module
import DeepMTP.data.splitting as splitting_module
import DeepMTP.data_loading as legacy_loading
import DeepMTP.data_preparation as legacy_preparation
import DeepMTP.data_progress as legacy_progress
import DeepMTP.data_splitting as legacy_splitting
import DeepMTP.feature_data as legacy_features
import DeepMTP.interaction_data as legacy_interactions


def test_data_package_exposes_canonical_foundation_services():
    expected_symbols = {
        "BaseDataset": loading_module.BaseDataset,
        "ConsoleDataProgressObserver": progress_module.ConsoleDataProgressObserver,
        "DataInfo": preparation_module.DataInfo,
        "DataLoaderFactory": loading_module.DataLoaderFactory,
        "DataPrintMode": progress_module.DataPrintMode,
        "DataProgressEvent": progress_module.DataProgressEvent,
        "DataProgressEventKind": progress_module.DataProgressEventKind,
        "DataProgressLevel": progress_module.DataProgressLevel,
        "DataProgressObserver": progress_module.DataProgressObserver,
        "DatasetItem": loading_module.DatasetItem,
        "DatasetMapping": interactions_module.DatasetMapping,
        "FeatureInfo": features_module.FeatureInfo,
        "InteractionInfo": interactions_module.InteractionInfo,
        "MTPSampler": loading_module.MTPSampler,
        "NoveltyInferenceWarning": interactions_module.NoveltyInferenceWarning,
        "NullDataProgressObserver": progress_module.NullDataProgressObserver,
        "ProcessedSplit": preparation_module.ProcessedSplit,
        "SamplerMode": loading_module.SamplerMode,
        "ScalingMethod": preparation_module.ScalingMethod,
        "TrainingDataLoaders": loading_module.TrainingDataLoaders,
        "Transformer": preparation_module.Transformer,
        "ValidationSetting": interactions_module.ValidationSetting,
        "VariableType": interactions_module.VariableType,
        "build_data_progress_observer": (progress_module.build_data_progress_observer),
        "check_interaction_files_column_type_format": (
            interactions_module.check_interaction_files_column_type_format
        ),
        "check_interaction_files_format": (
            interactions_module.check_interaction_files_format
        ),
        "check_novel_instances": interactions_module.check_novel_instances,
        "check_novel_targets": interactions_module.check_novel_targets,
        "check_target_variable_type": interactions_module.check_target_variable_type,
        "check_variable_type": interactions_module.check_variable_type,
        "cross_input_consistency_check_instances": (
            features_module.cross_input_consistency_check_instances
        ),
        "cross_input_consistency_check_targets": (
            features_module.cross_input_consistency_check_targets
        ),
        "data_process": preparation_module.data_process,
        "get_estimated_validation_setting": (
            interactions_module.get_estimated_validation_setting
        ),
        "normalize": preparation_module.normalize,
        "process_instance_features": features_module.process_instance_features,
        "process_interaction_data": interactions_module.process_interaction_data,
        "process_target_features": features_module.process_target_features,
        "split_data": splitting_module.split_data,
        "validate_split_ratio": preparation_module.validate_split_ratio,
    }

    assert set(data_api.__all__) == set(expected_symbols)
    for name, canonical in expected_symbols.items():
        assert getattr(data_api, name) is canonical


def test_top_level_data_api_uses_canonical_implementations():
    for name in (
        "ConsoleDataProgressObserver",
        "DataLoaderFactory",
        "DataProgressEvent",
        "DataProgressObserver",
        "NullDataProgressObserver",
        "TrainingDataLoaders",
        "data_process",
    ):
        assert getattr(deepmtp_api, name) is getattr(data_api, name)


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module"),
    [
        (legacy_features, features_module),
        (legacy_interactions, interactions_module),
        (legacy_loading, loading_module),
        (legacy_preparation, preparation_module),
        (legacy_progress, progress_module),
        (legacy_splitting, splitting_module),
    ],
)
def test_legacy_data_modules_reexport_canonical_symbols(
    legacy_module,
    canonical_module,
):
    assert set(legacy_module.__all__) == set(canonical_module.__all__)
    for name in canonical_module.__all__:
        assert getattr(legacy_module, name) is getattr(canonical_module, name)


@pytest.mark.parametrize(
    "legacy_module",
    [
        legacy_features,
        legacy_interactions,
        legacy_loading,
        legacy_preparation,
        legacy_progress,
        legacy_splitting,
    ],
)
def test_legacy_data_modules_are_definition_free_facades(legacy_module):
    tree = ast.parse(inspect.getsource(legacy_module))

    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert definitions == []


@pytest.mark.parametrize(
    ("legacy_module", "name", "canonical"),
    [
        ("DeepMTP.feature_data", "FeatureInfo", features_module.FeatureInfo),
        (
            "DeepMTP.interaction_data",
            "NoveltyInferenceWarning",
            interactions_module.NoveltyInferenceWarning,
        ),
        ("DeepMTP.data_loading", "BaseDataset", loading_module.BaseDataset),
        (
            "DeepMTP.data_preparation",
            "DataInfo",
            preparation_module.DataInfo,
        ),
        (
            "DeepMTP.data_progress",
            "DataProgressEvent",
            progress_module.DataProgressEvent,
        ),
        ("DeepMTP.data_splitting", "split_data", splitting_module.split_data),
    ],
)
def test_historical_data_pickle_globals_resolve(
    legacy_module,
    name,
    canonical,
):
    payload = f"c{legacy_module}\n{name}\n.".encode()

    assert pickle.loads(payload) is canonical


def test_data_classes_report_their_canonical_module_names():
    assert features_module.FeatureInfo.__module__ == "DeepMTP.data.features"
    assert interactions_module.NoveltyInferenceWarning.__module__ == (
        "DeepMTP.data.interactions"
    )
    assert loading_module.BaseDataset.__module__ == "DeepMTP.data.loading"
    assert loading_module.MTPSampler.__module__ == "DeepMTP.data.loading"
    assert preparation_module.DataInfo.__module__ == "DeepMTP.data.preparation"
    assert progress_module.DataProgressEvent.__module__ == "DeepMTP.data.progress"
    assert splitting_module.split_data.__module__ == "DeepMTP.data.splitting"


def test_data_package_does_not_eagerly_import_dataset_helpers():
    script = """
import sys

import DeepMTP.data

if "DeepMTP.dataset" in sys.modules:
    raise AssertionError("Dataset helpers were imported eagerly")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
