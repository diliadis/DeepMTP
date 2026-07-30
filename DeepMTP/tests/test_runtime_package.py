import ast
import inspect
import pickle
import subprocess
import sys

import pytest

import DeepMTP as deepmtp_api
import DeepMTP.evaluation as legacy_evaluation
import DeepMTP.evaluation_progress as legacy_evaluation_progress
import DeepMTP.persistence as legacy_persistence
import DeepMTP.presentation as legacy_presentation
import DeepMTP.reporting as legacy_reporting
import DeepMTP.runtime as runtime_api
import DeepMTP.runtime.components as components_module
import DeepMTP.runtime.evaluation as evaluation_module
import DeepMTP.runtime.evaluation_progress as evaluation_progress_module
import DeepMTP.runtime.metadata as metadata_module
import DeepMTP.runtime.persistence as persistence_module
import DeepMTP.runtime.presentation as presentation_module
import DeepMTP.runtime.reporting as reporting_module
import DeepMTP.runtime.training as training_module
import DeepMTP.runtime.wandb as wandb_module
import DeepMTP.training as legacy_training


def test_runtime_package_exposes_canonical_services():
    expected_symbols = {
        "CHECKPOINT_KEYS": persistence_module.CHECKPOINT_KEYS,
        "CHECKPOINT_FORMAT_VERSION": metadata_module.CHECKPOINT_FORMAT_VERSION,
        "CHECKPOINT_METADATA_KEY": persistence_module.CHECKPOINT_METADATA_KEY,
        "CompositeReporter": reporting_module.CompositeReporter,
        "ConsoleEvaluationProgressObserver": (
            evaluation_progress_module.ConsoleEvaluationProgressObserver
        ),
        "ConsoleProgressObserver": presentation_module.ConsoleProgressObserver,
        "DEFAULT_LOSS_BY_PROBLEM_MODE": (
            components_module.DEFAULT_LOSS_BY_PROBLEM_MODE
        ),
        "DEFAULT_LOSS_BY_CLASSIFICATION_MODE": (
            components_module.DEFAULT_LOSS_BY_CLASSIFICATION_MODE
        ),
        "DEFAULT_LOSS_INPUT_TRANSFORM_BY_PROBLEM_MODE": (
            components_module.DEFAULT_LOSS_INPUT_TRANSFORM_BY_PROBLEM_MODE
        ),
        "DEFAULT_OUTPUT_HEAD_BY_PROBLEM_MODE": (
            components_module.DEFAULT_OUTPUT_HEAD_BY_PROBLEM_MODE
        ),
        "DEFAULT_OUTPUT_HEAD_BY_CLASSIFICATION_MODE": (
            components_module.DEFAULT_OUTPUT_HEAD_BY_CLASSIFICATION_MODE
        ),
        "EvaluationAveraging": evaluation_progress_module.EvaluationAveraging,
        "EvaluationGroup": evaluation_progress_module.EvaluationGroup,
        "EvaluationPolicy": evaluation_module.EvaluationPolicy,
        "EvaluationProgressEvent": (evaluation_progress_module.EvaluationProgressEvent),
        "EvaluationProgressEventKind": (
            evaluation_progress_module.EvaluationProgressEventKind
        ),
        "EvaluationProgressObserver": (
            evaluation_progress_module.EvaluationProgressObserver
        ),
        "Evaluator": evaluation_module.Evaluator,
        "EpochOutput": training_module.EpochOutput,
        "ExperimentReporter": reporting_module.ExperimentReporter,
        "ExperimentStore": persistence_module.ExperimentStore,
        "LOSS_INPUT_TRANSFORM_BY_LOSS": (
            components_module.LOSS_INPUT_TRANSFORM_BY_LOSS
        ),
        "LOSS_REGISTRY": components_module.LOSS_REGISTRY,
        "LossFactory": components_module.LossFactory,
        "NullEvaluationProgressObserver": (
            evaluation_progress_module.NullEvaluationProgressObserver
        ),
        "NullProgressObserver": presentation_module.NullProgressObserver,
        "NullReporter": reporting_module.NullReporter,
        "OptionalIntegrationError": reporting_module.OptionalIntegrationError,
        "OUTPUT_HEAD_REGISTRY": components_module.OUTPUT_HEAD_REGISTRY,
        "OutputHeadFactory": components_module.OutputHeadFactory,
        "ProgressEvent": presentation_module.ProgressEvent,
        "ProgressEventKind": presentation_module.ProgressEventKind,
        "ProgressObserver": presentation_module.ProgressObserver,
        "ReporterConfigInput": reporting_module.ReporterConfigInput,
        "ReporterModelArtifact": reporting_module.ReporterModelArtifact,
        "ReporterPredictionTable": reporting_module.ReporterPredictionTable,
        "RunHistory": presentation_module.RunHistory,
        "RunSummaries": presentation_module.RunSummaries,
        "SUPPORTED_LOSSES_BY_PROBLEM_MODE": (
            components_module.SUPPORTED_LOSSES_BY_PROBLEM_MODE
        ),
        "SUPPORTED_LOSSES_BY_CLASSIFICATION_MODE": (
            components_module.SUPPORTED_LOSSES_BY_CLASSIFICATION_MODE
        ),
        "TensorBoardReporter": reporting_module.TensorBoardReporter,
        "TensorBatch": training_module.TensorBatch,
        "TaskComponents": components_module.TaskComponents,
        "TrainingEngine": training_module.TrainingEngine,
        "WandBReporter": reporting_module.WandBReporter,
        "WandBCheckpoint": wandb_module.WandBCheckpoint,
        "build_checkpoint_metadata": metadata_module.build_checkpoint_metadata,
        "build_default_loss": components_module.build_default_loss,
        "build_default_loss_input_transform": (
            components_module.build_default_loss_input_transform
        ),
        "build_default_output_head": components_module.build_default_output_head,
        "build_default_task_components": (
            components_module.build_default_task_components
        ),
        "build_evaluation_progress_observer": (
            evaluation_progress_module.build_evaluation_progress_observer
        ),
        "build_progress_observer": presentation_module.build_progress_observer,
        "build_reporter": reporting_module.build_reporter,
        "build_loss": components_module.build_loss,
        "build_loss_input_transform": (components_module.build_loss_input_transform),
        "build_output_head": components_module.build_output_head,
        "build_task_components": components_module.build_task_components,
        "download_wandb_checkpoint": wandb_module.download_wandb_checkpoint,
        "log_reporter_model_artifact": (reporting_module.log_reporter_model_artifact),
        "log_reporter_predictions": reporting_module.log_reporter_predictions,
        "log_reporter_summary": reporting_module.log_reporter_summary,
        "mark_reporter_failed": reporting_module.mark_reporter_failed,
        "reporter_runtime_metadata": reporting_module.reporter_runtime_metadata,
        "update_reporter_config": reporting_module.update_reporter_config,
        "validate_checkpoint_metadata": metadata_module.validate_checkpoint_metadata,
    }

    assert set(runtime_api.__all__) == set(expected_symbols)
    for name, canonical in expected_symbols.items():
        assert getattr(runtime_api, name) is canonical


def test_top_level_runtime_api_uses_canonical_implementations():
    for name in (
        "CompositeReporter",
        "ConsoleEvaluationProgressObserver",
        "ConsoleProgressObserver",
        "EpochOutput",
        "EvaluationPolicy",
        "EvaluationProgressEvent",
        "Evaluator",
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
        "ReporterModelArtifact",
        "ReporterPredictionTable",
        "TensorBoardReporter",
        "TrainingEngine",
        "WandBReporter",
        "WandBCheckpoint",
    ):
        assert getattr(deepmtp_api, name) is getattr(runtime_api, name)


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module"),
    [
        (legacy_evaluation, evaluation_module),
        (legacy_evaluation_progress, evaluation_progress_module),
        (legacy_persistence, persistence_module),
        (legacy_presentation, presentation_module),
        (legacy_reporting, reporting_module),
        (legacy_training, training_module),
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
        legacy_evaluation,
        legacy_evaluation_progress,
        legacy_persistence,
        legacy_presentation,
        legacy_reporting,
        legacy_training,
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
            "DeepMTP.evaluation",
            "EvaluationPolicy",
            evaluation_module.EvaluationPolicy,
        ),
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
        (
            "DeepMTP.training",
            "EpochOutput",
            training_module.EpochOutput,
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


def test_runtime_classes_report_their_canonical_module_names():
    assert evaluation_module.EvaluationPolicy.__module__ == (
        "DeepMTP.runtime.evaluation"
    )
    assert training_module.EpochOutput.__module__ == "DeepMTP.runtime.training"
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
