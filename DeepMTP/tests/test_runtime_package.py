import ast
import inspect
import pickle
import subprocess
import sys

import pytest

import DeepMTP as deepmtp_api
import DeepMTP.evaluation_progress as legacy_evaluation_progress
import DeepMTP.persistence as legacy_persistence
import DeepMTP.presentation as legacy_presentation
import DeepMTP.reporting as legacy_reporting
import DeepMTP.runtime as runtime_api
import DeepMTP.runtime.evaluation_progress as evaluation_progress_module
import DeepMTP.runtime.persistence as persistence_module
import DeepMTP.runtime.presentation as presentation_module
import DeepMTP.runtime.reporting as reporting_module


def test_runtime_package_exposes_canonical_support_services():
    expected_symbols = {
        "CHECKPOINT_KEYS": persistence_module.CHECKPOINT_KEYS,
        "CompositeReporter": reporting_module.CompositeReporter,
        "ConsoleEvaluationProgressObserver": (
            evaluation_progress_module.ConsoleEvaluationProgressObserver
        ),
        "ConsoleProgressObserver": presentation_module.ConsoleProgressObserver,
        "EvaluationAveraging": evaluation_progress_module.EvaluationAveraging,
        "EvaluationGroup": evaluation_progress_module.EvaluationGroup,
        "EvaluationProgressEvent": (evaluation_progress_module.EvaluationProgressEvent),
        "EvaluationProgressEventKind": (
            evaluation_progress_module.EvaluationProgressEventKind
        ),
        "EvaluationProgressObserver": (
            evaluation_progress_module.EvaluationProgressObserver
        ),
        "ExperimentReporter": reporting_module.ExperimentReporter,
        "ExperimentStore": persistence_module.ExperimentStore,
        "NullEvaluationProgressObserver": (
            evaluation_progress_module.NullEvaluationProgressObserver
        ),
        "NullProgressObserver": presentation_module.NullProgressObserver,
        "NullReporter": reporting_module.NullReporter,
        "OptionalIntegrationError": reporting_module.OptionalIntegrationError,
        "ProgressEvent": presentation_module.ProgressEvent,
        "ProgressEventKind": presentation_module.ProgressEventKind,
        "ProgressObserver": presentation_module.ProgressObserver,
        "ReporterConfigInput": reporting_module.ReporterConfigInput,
        "RunHistory": presentation_module.RunHistory,
        "RunSummaries": presentation_module.RunSummaries,
        "TensorBoardReporter": reporting_module.TensorBoardReporter,
        "WandBReporter": reporting_module.WandBReporter,
        "build_evaluation_progress_observer": (
            evaluation_progress_module.build_evaluation_progress_observer
        ),
        "build_progress_observer": presentation_module.build_progress_observer,
        "build_reporter": reporting_module.build_reporter,
    }

    assert set(runtime_api.__all__) == set(expected_symbols)
    for name, canonical in expected_symbols.items():
        assert getattr(runtime_api, name) is canonical


def test_top_level_runtime_api_uses_canonical_implementations():
    for name in (
        "CompositeReporter",
        "ConsoleEvaluationProgressObserver",
        "ConsoleProgressObserver",
        "EvaluationProgressEvent",
        "ExperimentReporter",
        "ExperimentStore",
        "NullEvaluationProgressObserver",
        "NullProgressObserver",
        "NullReporter",
        "OptionalIntegrationError",
        "ProgressEvent",
        "ProgressObserver",
        "RunHistory",
        "RunSummaries",
        "TensorBoardReporter",
        "WandBReporter",
    ):
        assert getattr(deepmtp_api, name) is getattr(runtime_api, name)


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module"),
    [
        (legacy_evaluation_progress, evaluation_progress_module),
        (legacy_persistence, persistence_module),
        (legacy_presentation, presentation_module),
        (legacy_reporting, reporting_module),
    ],
)
def test_legacy_runtime_modules_reexport_canonical_symbols(
    legacy_module,
    canonical_module,
):
    assert set(legacy_module.__all__) == set(canonical_module.__all__)
    for name in canonical_module.__all__:
        assert getattr(legacy_module, name) is getattr(canonical_module, name)


@pytest.mark.parametrize(
    "legacy_module",
    [
        legacy_evaluation_progress,
        legacy_persistence,
        legacy_presentation,
        legacy_reporting,
    ],
)
def test_legacy_runtime_modules_are_definition_free_facades(legacy_module):
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
        (
            "DeepMTP.evaluation_progress",
            "EvaluationProgressEvent",
            evaluation_progress_module.EvaluationProgressEvent,
        ),
        (
            "DeepMTP.persistence",
            "ExperimentStore",
            persistence_module.ExperimentStore,
        ),
        (
            "DeepMTP.presentation",
            "ProgressEvent",
            presentation_module.ProgressEvent,
        ),
        (
            "DeepMTP.reporting",
            "CompositeReporter",
            reporting_module.CompositeReporter,
        ),
    ],
)
def test_historical_runtime_pickle_globals_resolve(
    legacy_module,
    name,
    canonical,
):
    payload = f"c{legacy_module}\n{name}\n.".encode()

    assert pickle.loads(payload) is canonical


def test_runtime_support_classes_report_their_canonical_module_names():
    assert persistence_module.ExperimentStore.__module__ == (
        "DeepMTP.runtime.persistence"
    )
    assert presentation_module.ProgressEvent.__module__ == (
        "DeepMTP.runtime.presentation"
    )
    assert reporting_module.CompositeReporter.__module__ == (
        "DeepMTP.runtime.reporting"
    )
    assert evaluation_progress_module.EvaluationProgressEvent.__module__ == (
        "DeepMTP.runtime.evaluation_progress"
    )


def test_runtime_package_does_not_eagerly_import_reporting_dependencies():
    script = """
import sys

import DeepMTP.runtime

for module_name in ("wandb", "torch.utils.tensorboard"):
    if module_name in sys.modules:
        raise AssertionError(f"{module_name} was imported eagerly")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
