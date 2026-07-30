import ast
import inspect
import pickle
import subprocess
import sys

import pytest

import DeepMTP as deepmtp_api
import DeepMTP.data as data_api
import DeepMTP.data.batches as batches_module
import DeepMTP.data.datasets as datasets_module
import DeepMTP.data.features as features_module
import DeepMTP.data.interactions as interactions_module
import DeepMTP.data.loading as loading_module
import DeepMTP.data.preparation as preparation_module
import DeepMTP.data.progress as progress_module
import DeepMTP.data.splitting as splitting_module
import DeepMTP.data.tabular as tabular_module
import DeepMTP.data_loading as legacy_loading
import DeepMTP.data_preparation as legacy_preparation
import DeepMTP.data_progress as legacy_progress
import DeepMTP.data_splitting as legacy_splitting
import DeepMTP.dataset as legacy_datasets
import DeepMTP.feature_data as legacy_features
import DeepMTP.interaction_data as legacy_interactions


def test_data_package_exposes_canonical_services():
    expected_symbols = {
        "ArrayFormat": datasets_module.ArrayFormat,
        "BaseDataset": loading_module.BaseDataset,
        "BranchBatch": batches_module.BranchBatch,
        "BranchInputKind": batches_module.BranchInputKind,
        "CategoricalColumn": tabular_module.CategoricalColumn,
        "ConsoleDataProgressObserver": progress_module.ConsoleDataProgressObserver,
        "CustomBranchBatch": batches_module.CustomBranchBatch,
        "DataInfo": preparation_module.DataInfo,
        "DataLoaderFactory": loading_module.DataLoaderFactory,
        "DataPrintMode": progress_module.DataPrintMode,
        "DataProgressEvent": progress_module.DataProgressEvent,
        "DataProgressEventKind": progress_module.DataProgressEventKind,
        "DataProgressLevel": progress_module.DataProgressLevel,
        "DataProgressObserver": progress_module.DataProgressObserver,
        "DatasetBundle": datasets_module.DatasetBundle,
        "DatasetItem": loading_module.DatasetItem,
        "DatasetMapping": interactions_module.DatasetMapping,
        "DatasetSplit": datasets_module.DatasetSplit,
        "DatasetValue": datasets_module.DatasetValue,
        "DatasetVariant": datasets_module.DatasetVariant,
        "DenseBranchBatch": batches_module.DenseBranchBatch,
        "FeatureKind": features_module.FeatureKind,
        "FeatureInfo": features_module.FeatureInfo,
        "InteractionInfo": interactions_module.InteractionInfo,
        "IDBranchBatch": batches_module.IDBranchBatch,
        "ImageBranchBatch": batches_module.ImageBranchBatch,
        "MTPSampler": loading_module.MTPSampler,
        "MTPBatch": batches_module.MTPBatch,
        "MTPBatchCollator": batches_module.MTPBatchCollator,
        "NoveltyInferenceWarning": interactions_module.NoveltyInferenceWarning,
        "NullDataProgressObserver": progress_module.NullDataProgressObserver,
        "PrintMode": datasets_module.PrintMode,
        "ProcessedDatasetBundle": datasets_module.ProcessedDatasetBundle,
        "ProcessedDatasetSplit": datasets_module.ProcessedDatasetSplit,
        "ProcessedDatasetValue": datasets_module.ProcessedDatasetValue,
        "ProcessedSplit": preparation_module.ProcessedSplit,
        "SamplerMode": loading_module.SamplerMode,
        "ScalingMethod": preparation_module.ScalingMethod,
        "SequenceBranchBatch": batches_module.SequenceBranchBatch,
        "SequenceInput": batches_module.SequenceInput,
        "SparseBranchBatch": batches_module.SparseBranchBatch,
        "TabularBranchBatch": batches_module.TabularBranchBatch,
        "TabularInput": batches_module.TabularInput,
        "TabularPreprocessingState": tabular_module.TabularPreprocessingState,
        "TabularPreprocessor": tabular_module.TabularPreprocessor,
        "TabularSchema": tabular_module.TabularSchema,
        "TrainingDataLoaders": loading_module.TrainingDataLoaders,
        "TensorBranchBatch": batches_module.TensorBranchBatch,
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
        "format_mtr_datasets": datasets_module.format_mtr_datasets,
        "generate_MTP_dataset": datasets_module.generate_MTP_dataset,
        "generate_dummy_dataset": datasets_module.generate_dummy_dataset,
        "generate_interaction_matrix": datasets_module.generate_interaction_matrix,
        "get_estimated_validation_setting": (
            interactions_module.get_estimated_validation_setting
        ),
        "load_process_DP": datasets_module.load_process_DP,
        "load_process_MC": datasets_module.load_process_MC,
        "load_process_MLC": datasets_module.load_process_MLC,
        "load_process_MTL": datasets_module.load_process_MTL,
        "load_process_MTR": datasets_module.load_process_MTR,
        "normalize": preparation_module.normalize,
        "print_MTR_datasets": datasets_module.print_MTR_datasets,
        "process_dummy_DP": datasets_module.process_dummy_DP,
        "process_dummy_MLC": datasets_module.process_dummy_MLC,
        "process_dummy_MTR": datasets_module.process_dummy_MTR,
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
        (legacy_datasets, datasets_module),
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
        legacy_datasets,
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
        ("DeepMTP.dataset", "DatasetBundle", datasets_module.DatasetBundle),
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
    assert datasets_module.DatasetBundle.__module__ == "DeepMTP.data.datasets"
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
if "DeepMTP.data.datasets" in sys.modules:
    raise AssertionError("Canonical dataset helpers were imported eagerly")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_data_package_loads_dataset_helpers_on_demand():
    script = """
import sys

import DeepMTP.data as data

helper = data.generate_MTP_dataset
if "DeepMTP.data.datasets" not in sys.modules:
    raise AssertionError("Dataset helpers were not loaded on demand")
if helper.__module__ != "DeepMTP.data.datasets":
    raise AssertionError(f"Unexpected helper owner: {helper.__module__}")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
