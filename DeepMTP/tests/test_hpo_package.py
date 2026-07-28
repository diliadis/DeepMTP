import ast
import inspect
import pickle

import pytest

import DeepMTP.hpo as hpo_api
import DeepMTP.hpo.hyperband as hyperband_module
import DeepMTP.hpo.progress as progress_module
import DeepMTP.hpo.random_search as random_search_module
import DeepMTP.hpo.types as types_module
import DeepMTP.hpo.worker as worker_module
import DeepMTP.hpo_progress as legacy_progress
import DeepMTP.hpo_types as legacy_types
import DeepMTP.hpo_worker as legacy_worker
import DeepMTP.random_search as legacy_random_search
import DeepMTP.simple_hyperband as legacy_hyperband


def test_hpo_package_exposes_canonical_implementations():
    expected_symbols = {
        "BaseWorker": worker_module.BaseWorker,
        "ConfigurationSpace": types_module.ConfigurationSpace,
        "ConsoleHPOProgressObserver": progress_module.ConsoleHPOProgressObserver,
        "HPOConfiguration": types_module.HPOConfiguration,
        "HPOProgressEvent": progress_module.HPOProgressEvent,
        "HPOProgressEventKind": progress_module.HPOProgressEventKind,
        "HPOProgressObserver": progress_module.HPOProgressObserver,
        "HPOWorker": types_module.HPOWorker,
        "HPOWorkerResult": types_module.HPOWorkerResult,
        "HyperBand": hyperband_module.HyperBand,
        "HyperbandBracket": hyperband_module.HyperbandBracket,
        "NullHPOProgressObserver": progress_module.NullHPOProgressObserver,
        "RandomSearch": random_search_module.RandomSearch,
        "build_hpo_progress_observer": progress_module.build_hpo_progress_observer,
    }

    assert set(hpo_api.__all__) == set(expected_symbols)
    for name, canonical in expected_symbols.items():
        assert getattr(hpo_api, name) is canonical


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module", "public_names"),
    [
        (
            legacy_progress,
            progress_module,
            (
                "ConsoleHPOProgressObserver",
                "HPOProgressEvent",
                "HPOProgressEventKind",
                "HPOProgressObserver",
                "NullHPOProgressObserver",
                "build_hpo_progress_observer",
            ),
        ),
        (
            legacy_types,
            types_module,
            (
                "ConfigurationSpace",
                "HPOConfiguration",
                "HPOWorker",
                "HPOWorkerResult",
            ),
        ),
        (legacy_worker, worker_module, ("BaseWorker",)),
        (legacy_random_search, random_search_module, ("RandomSearch",)),
        (
            legacy_hyperband,
            hyperband_module,
            ("HyperBand", "HyperbandBracket"),
        ),
    ],
)
def test_legacy_hpo_modules_reexport_canonical_symbols(
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
        legacy_types,
        legacy_worker,
        legacy_random_search,
        legacy_hyperband,
    ],
)
def test_legacy_hpo_modules_are_definition_free_facades(legacy_module):
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
        ("DeepMTP.hpo_progress", "HPOProgressEvent", progress_module.HPOProgressEvent),
        ("DeepMTP.hpo_worker", "BaseWorker", worker_module.BaseWorker),
        ("DeepMTP.random_search", "RandomSearch", random_search_module.RandomSearch),
        ("DeepMTP.simple_hyperband", "HyperBand", hyperband_module.HyperBand),
    ],
)
def test_historical_pickle_globals_resolve_to_canonical_symbols(
    legacy_module,
    name,
    canonical,
):
    payload = f"c{legacy_module}\n{name}\n.".encode()

    assert pickle.loads(payload) is canonical


def test_hpo_classes_report_their_canonical_module_names():
    assert worker_module.BaseWorker.__module__ == "DeepMTP.hpo.worker"
    assert random_search_module.RandomSearch.__module__ == "DeepMTP.hpo.random_search"
    assert hyperband_module.HyperBand.__module__ == "DeepMTP.hpo.hyperband"
    assert progress_module.HPOProgressEvent.__module__ == "DeepMTP.hpo.progress"
