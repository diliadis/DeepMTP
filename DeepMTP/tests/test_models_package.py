import ast
import inspect
import pickle
import subprocess
import sys

import pytest

import DeepMTP as deepmtp_api
import DeepMTP.branch_models as legacy_branches
import DeepMTP.model_factory as legacy_factory
import DeepMTP.models as models_api
import DeepMTP.models.branches as branches_module
import DeepMTP.models.factory as factory_module


def test_models_package_exposes_canonical_implementations():
    expected_symbols = {
        "BranchFactory": factory_module.BranchFactory,
        "ConvNet": branches_module.ConvNet,
        "MLP": branches_module.MLP,
        "ModelBundle": factory_module.ModelBundle,
        "ModelFactory": factory_module.ModelFactory,
        "TwoBranchDotProductModel": factory_module.TwoBranchDotProductModel,
        "TwoBranchKroneckerModel": factory_module.TwoBranchKroneckerModel,
        "TwoBranchMLPModel": factory_module.TwoBranchMLPModel,
        "isolated_torch_seed": factory_module.isolated_torch_seed,
    }

    assert set(models_api.__all__) == set(expected_symbols)
    for name, canonical in expected_symbols.items():
        assert getattr(models_api, name) is canonical


def test_top_level_model_api_uses_canonical_implementations():
    for name in (
        "ModelBundle",
        "ModelFactory",
        "TwoBranchDotProductModel",
        "TwoBranchKroneckerModel",
        "TwoBranchMLPModel",
    ):
        assert getattr(deepmtp_api, name) is getattr(models_api, name)


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module", "public_names"),
    [
        (
            legacy_branches,
            branches_module,
            ("ConvNet", "MLP", "ModelConfigInput"),
        ),
        (
            legacy_factory,
            factory_module,
            (
                "BranchFactory",
                "ModelBundle",
                "ModelConfigInput",
                "ModelFactory",
                "TwoBranchDotProductModel",
                "TwoBranchKroneckerModel",
                "TwoBranchMLPModel",
                "isolated_torch_seed",
            ),
        ),
    ],
)
def test_legacy_model_modules_reexport_canonical_symbols(
    legacy_module,
    canonical_module,
    public_names,
):
    assert set(legacy_module.__all__) == set(public_names)
    for name in public_names:
        assert getattr(legacy_module, name) is getattr(canonical_module, name)


@pytest.mark.parametrize(
    "legacy_module",
    [legacy_branches, legacy_factory],
)
def test_legacy_model_modules_are_definition_free_facades(legacy_module):
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
        ("DeepMTP.branch_models", "MLP", branches_module.MLP),
        ("DeepMTP.branch_models", "ConvNet", branches_module.ConvNet),
        (
            "DeepMTP.model_factory",
            "TwoBranchDotProductModel",
            factory_module.TwoBranchDotProductModel,
        ),
        ("DeepMTP.model_factory", "ModelFactory", factory_module.ModelFactory),
    ],
)
def test_historical_model_pickle_globals_resolve(
    legacy_module,
    name,
    canonical,
):
    payload = f"c{legacy_module}\n{name}\n.".encode()

    assert pickle.loads(payload) is canonical


def test_model_classes_report_their_canonical_module_names():
    assert branches_module.MLP.__module__ == "DeepMTP.models.branches"
    assert branches_module.ConvNet.__module__ == "DeepMTP.models.branches"
    assert factory_module.ModelFactory.__module__ == "DeepMTP.models.factory"
    assert factory_module.TwoBranchDotProductModel.__module__ == (
        "DeepMTP.models.factory"
    )


def test_models_package_does_not_eagerly_import_torchvision():
    script = """
import sys

import DeepMTP.models

if "torchvision" in sys.modules or "torchvision.models" in sys.modules:
    raise AssertionError("Torchvision was imported eagerly")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
