from types import ModuleType

import pytest

import scripts.smoke_test_optional_extras as optional_smoke


def test_optional_extra_smoke_covers_every_declared_runtime_extra():
    imported_modules = []

    def import_module(name):
        imported_modules.append(name)
        return ModuleType(name)

    imported_extras = optional_smoke.verify_optional_integrations(import_module)

    assert imported_extras == (
        "datasets",
        "foundation",
        "graph",
        "hpo",
        "image",
        "sparse",
        "streamlit",
        "tracking",
    )
    assert imported_modules == [
        module_name
        for module_names in optional_smoke._OPTIONAL_IMPORTS.values()
        for module_name in module_names
    ]


def test_optional_extra_smoke_identifies_the_broken_extra_and_import():
    def import_module(name):
        if name == "arff":
            raise ModuleNotFoundError(name)
        return ModuleType(name)

    with pytest.raises(
        RuntimeError,
        match=r"DeepMTP\[datasets\] could not import 'arff'",
    ):
        optional_smoke.verify_optional_integrations(import_module)


def test_optional_extra_smoke_rejects_source_checkout():
    with pytest.raises(RuntimeError, match="source checkout"):
        optional_smoke.require_installed_wheel()


def test_optional_extra_smoke_cli_reports_result(monkeypatch, capsys):
    monkeypatch.setattr(
        optional_smoke,
        "verify_optional_integrations",
        lambda: ("datasets", "hpo"),
    )

    assert optional_smoke.main([]) == 0
    assert "datasets, hpo" in capsys.readouterr().out
