import ast
import inspect
import pickle
import subprocess
import sys

import pytest

import DeepMTP.integrations.streamlit as streamlit_api
import DeepMTP.integrations.streamlit.hpo as hpo_module
import DeepMTP.integrations.streamlit.progress as progress_module
import DeepMTP.integrations.streamlit.trainer as trainer_module
import DeepMTP.main_streamlit as legacy_trainer
import DeepMTP.random_search_streamlit as legacy_random_search
import DeepMTP.simple_hyperband_streamlit as legacy_hyperband
import DeepMTP.streamlit_adapter as legacy_progress
from DeepMTP.runtime.reporting import OptionalIntegrationError


def test_streamlit_package_exposes_canonical_implementations():
    expected_symbols = {
        "DeepMTP": trainer_module.DeepMTP,
        "HyperBand": hpo_module.HyperBand,
        "RandomSearch": hpo_module.RandomSearch,
        "StreamlitAPI": progress_module.StreamlitAPI,
        "StreamlitHPOProgressObserver": progress_module.StreamlitHPOProgressObserver,
        "StreamlitProgressObserver": progress_module.StreamlitProgressObserver,
        "initialize_mode": trainer_module.initialize_mode,
    }

    assert set(streamlit_api.__all__) == set(expected_symbols)
    for name, canonical in expected_symbols.items():
        assert getattr(streamlit_api, name) is canonical


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module", "public_names"),
    [
        (
            legacy_progress,
            progress_module,
            (
                "StreamlitAPI",
                "StreamlitHPOProgressObserver",
                "StreamlitProgressObserver",
            ),
        ),
        (
            legacy_trainer,
            trainer_module,
            ("DeepMTP", "initialize_mode"),
        ),
        (
            legacy_random_search,
            hpo_module,
            ("RandomSearch",),
        ),
        (
            legacy_hyperband,
            hpo_module,
            ("HyperBand",),
        ),
    ],
)
def test_legacy_streamlit_modules_reexport_canonical_symbols(
    legacy_module,
    canonical_module,
    public_names,
):
    assert set(legacy_module.__all__) == set(public_names)
    for name in public_names:
        assert getattr(legacy_module, name) is getattr(canonical_module, name)


@pytest.mark.parametrize(
    "legacy_module",
    [
        legacy_progress,
        legacy_trainer,
        legacy_random_search,
        legacy_hyperband,
    ],
)
def test_legacy_streamlit_modules_are_definition_free_facades(legacy_module):
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
        ("DeepMTP.main_streamlit", "DeepMTP", trainer_module.DeepMTP),
        (
            "DeepMTP.random_search_streamlit",
            "RandomSearch",
            hpo_module.RandomSearch,
        ),
        (
            "DeepMTP.simple_hyperband_streamlit",
            "HyperBand",
            hpo_module.HyperBand,
        ),
        (
            "DeepMTP.streamlit_adapter",
            "StreamlitProgressObserver",
            progress_module.StreamlitProgressObserver,
        ),
    ],
)
def test_historical_streamlit_pickle_globals_resolve(
    legacy_module,
    name,
    canonical,
):
    payload = f"c{legacy_module}\n{name}\n.".encode()

    assert pickle.loads(payload) is canonical


def test_streamlit_classes_report_their_canonical_module_names():
    assert trainer_module.DeepMTP.__module__ == (
        "DeepMTP.integrations.streamlit.trainer"
    )
    assert hpo_module.RandomSearch.__module__ == "DeepMTP.integrations.streamlit.hpo"
    assert hpo_module.HyperBand.__module__ == "DeepMTP.integrations.streamlit.hpo"
    assert progress_module.StreamlitProgressObserver.__module__ == (
        "DeepMTP.integrations.streamlit.progress"
    )


def test_integrations_package_does_not_eagerly_import_streamlit_adapters():
    script = """
import sys

import DeepMTP.integrations

if "DeepMTP.integrations.streamlit" in sys.modules:
    raise AssertionError("Streamlit adapters were imported eagerly")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_missing_streamlit_dependency_has_focused_error(monkeypatch):
    def fail_import(name):
        assert name == "streamlit"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(progress_module.importlib, "import_module", fail_import)

    with pytest.raises(OptionalIntegrationError, match="Streamlit support"):
        progress_module.StreamlitProgressObserver()
