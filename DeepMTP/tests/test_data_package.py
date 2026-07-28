import ast
import inspect
import pickle
import subprocess
import sys

import pytest

import DeepMTP as deepmtp_api
import DeepMTP.data as data_api
import DeepMTP.data.loading as loading_module
import DeepMTP.data.progress as progress_module
import DeepMTP.data_loading as legacy_loading
import DeepMTP.data_progress as legacy_progress


def test_data_package_exposes_canonical_foundation_services():
    expected_symbols = {
        "BaseDataset": loading_module.BaseDataset,
        "ConsoleDataProgressObserver": progress_module.ConsoleDataProgressObserver,
        "DataLoaderFactory": loading_module.DataLoaderFactory,
        "DataPrintMode": progress_module.DataPrintMode,
        "DataProgressEvent": progress_module.DataProgressEvent,
        "DataProgressEventKind": progress_module.DataProgressEventKind,
        "DataProgressLevel": progress_module.DataProgressLevel,
        "DataProgressObserver": progress_module.DataProgressObserver,
        "DatasetItem": loading_module.DatasetItem,
        "MTPSampler": loading_module.MTPSampler,
        "NullDataProgressObserver": progress_module.NullDataProgressObserver,
        "SamplerMode": loading_module.SamplerMode,
        "TrainingDataLoaders": loading_module.TrainingDataLoaders,
        "build_data_progress_observer": (progress_module.build_data_progress_observer),
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
    ):
        assert getattr(deepmtp_api, name) is getattr(data_api, name)


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module"),
    [
        (legacy_loading, loading_module),
        (legacy_progress, progress_module),
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
    [legacy_loading, legacy_progress],
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
        ("DeepMTP.data_loading", "BaseDataset", loading_module.BaseDataset),
        (
            "DeepMTP.data_progress",
            "DataProgressEvent",
            progress_module.DataProgressEvent,
        ),
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
    assert loading_module.BaseDataset.__module__ == "DeepMTP.data.loading"
    assert loading_module.MTPSampler.__module__ == "DeepMTP.data.loading"
    assert progress_module.DataProgressEvent.__module__ == "DeepMTP.data.progress"


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
