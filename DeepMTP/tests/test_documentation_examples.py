import ast
import json
import math
import re
import runpy
import sys
import textwrap
from importlib.metadata import version as distribution_version
from pathlib import Path

import pandas as pd
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

_PROJECT_ROOT = Path(__file__).parents[2]
_DEMO_NOTEBOOKS = sorted((_PROJECT_ROOT / "DEMO").glob("*.ipynb"))
_HPO_DEMO_NOTEBOOKS = {
    "hyperband_example.ipynb",
    "random_search_example.ipynb",
}
_COLAB_CORE_REQUIREMENT = (
    "DeepMTP[datasets] @ git+https://github.com/diliadis/DeepMTP.git@main"
)
_COLAB_HPO_REQUIREMENT = (
    "DeepMTP[datasets,hpo] @ git+https://github.com/diliadis/DeepMTP.git@main"
)
_RELEASE_CANDIDATE_WORKLOADS = [
    pytest.param(
        "main_example_MLC.ipynb",
        "B",
        "classification",
        {"val_loss", "val_hamming_loss_macro", "val_auroc_macro"},
        True,
        False,
        id="MLC",
    ),
    pytest.param(
        "main_example_MTR.ipynb",
        "B",
        "regression",
        {"val_loss", "val_RMSE_macro", "val_R2_macro"},
        True,
        False,
        id="MTR",
    ),
    pytest.param(
        "main_example_DP.ipynb",
        "D",
        "classification",
        {
            "val_loss",
            "val_aupr_macro",
            "val_aupr_micro",
            "val_auroc_macro",
            "val_auroc_micro",
        },
        True,
        True,
        id="DP",
    ),
    pytest.param(
        "main_example_MTL.ipynb",
        "B",
        "classification",
        {"val_loss", "val_aupr_macro", "val_auroc_macro"},
        True,
        False,
        id="MTL",
    ),
    pytest.param(
        "main_example_MC.ipynb",
        "A",
        "regression",
        {"val_loss", "val_RMSE_micro", "val_MSE_micro"},
        False,
        False,
        id="MC",
    ),
]
_FORBIDDEN_NOTEBOOK_SNIPPETS = (
    "from DeepMTP.main import DeepMTP",
    "from DeepMTP.utils.data_utils import data_process",
    "from ast import arg",
    "os.chdir(",
    "BaseDataset",
    'default_value="1e-3"',
    "/data/dimitriosi",
    "load_process_",
    "use_tensorboard_logger = True",
    "'use_tensorboard_logger': True",
    "wandb_project_name = 'DeepMTP_v2'",
    "compute_mode = 'cuda",
    "'compute_mode': 'cuda",
    "branch_architecture = 'CONV'",
    "ConfigSpace.hyperparameters",
    "add_hyperparameters",
    "add_condition",
    "from DeepMTP.utils.tests import",
    "from DeepMTP.utils.utils import generate_config",
)


def _marked_python_example(document: Path, marker: str) -> str:
    text = document.read_text(encoding="utf-8")
    marked_section = text.split(f"<!-- {marker}:start -->", maxsplit=1)[1].split(
        f"<!-- {marker}:end -->",
        maxsplit=1,
    )[0]
    return marked_section.split("```python", maxsplit=1)[1].split(
        "```",
        maxsplit=1,
    )[0]


def _notebook_code_cells(notebook_path: Path) -> list[str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _execute_notebook(
    notebook_path: Path,
    *,
    max_code_cells: int | None = None,
) -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": "__main__"}
    code_cells = _notebook_code_cells(notebook_path)
    if max_code_cells is not None:
        code_cells = code_cells[:max_code_cells]
    for cell_number, code in enumerate(code_cells, start=1):
        exec(
            compile(
                code,
                f"{notebook_path.name}:cell-{cell_number}",
                "exec",
            ),
            namespace,
        )
    return namespace


def _marked_rst_python_example(document: Path, marker: str) -> str:
    text = document.read_text(encoding="utf-8")
    marked_section = text.split(f".. {marker}-start", maxsplit=1)[1].split(
        f".. {marker}-end",
        maxsplit=1,
    )[0]
    code_block = marked_section.split(".. code-block:: python", maxsplit=1)[1]
    return textwrap.dedent(code_block).strip()


def _rst_python_examples(document: Path) -> list[str]:
    lines = document.read_text(encoding="utf-8").splitlines()
    examples = []
    line_number = 0
    while line_number < len(lines):
        if lines[line_number].strip() != ".. code-block:: python":
            line_number += 1
            continue

        line_number += 1
        while line_number < len(lines) and (
            not lines[line_number].strip()
            or lines[line_number].lstrip().startswith(":")
        ):
            line_number += 1

        block = []
        while line_number < len(lines) and (
            not lines[line_number].strip() or lines[line_number].startswith((" ", "\t"))
        ):
            block.append(lines[line_number])
            line_number += 1
        examples.append(textwrap.dedent("\n".join(block)).strip())
    return examples


def test_readme_quickstart_runs_offline(monkeypatch, tmp_path):
    example = _marked_python_example(
        _PROJECT_ROOT / "README.md",
        "quickstart-example",
    )
    namespace = {"__name__": "__main__"}
    monkeypatch.chdir(tmp_path)

    exec(compile(example, "README.md", "exec"), namespace)

    assert "val_loss" in namespace["validation_results"]
    assert isinstance(namespace["test_results"], dict)
    assert isinstance(namespace["predictions"], pd.DataFrame)
    assert len(namespace["predictions"]) == len(namespace["test"]["y"]["data"])
    assert (tmp_path / "results" / "quickstart" / "config.json").is_file()


@pytest.mark.parametrize(
    ("marker", "expected_instance_architecture", "expected_target_architecture"),
    [
        ("id-embedding-example", "EMBEDDING", "EMBEDDING"),
        ("tabular-config-example", "TABULAR", "EMBEDDING"),
        ("sequence-config-example", "SEQUENCE", "MLP"),
    ],
)
def test_readme_branch_configuration_examples_are_valid(
    marker,
    expected_instance_architecture,
    expected_target_architecture,
):
    example = _marked_python_example(_PROJECT_ROOT / "README.md", marker)
    namespace = {"__name__": "__main__"}

    exec(compile(example, "README.md", "exec"), namespace)

    config = namespace["config"]
    assert config.instance_branch_architecture == expected_instance_architecture
    assert config.target_branch_architecture == expected_target_architecture
    if marker == "sequence-config-example":
        assert config.instance_branch_sequence_encoder == "transformer"


def test_readme_multiclass_configuration_example_is_valid():
    example = _marked_python_example(
        _PROJECT_ROOT / "README.md",
        "multiclass-config-example",
    )
    namespace = {"__name__": "__main__"}

    exec(compile(example, "README.md", "exec"), namespace)

    data_info = namespace["data_info"]
    config = namespace["config"]
    assert data_info["detected_classification_mode"] == "multiclass"
    assert data_info["detected_num_classes"] == 3
    assert config.classification_mode == "multiclass"
    assert config.num_classes == 3
    assert config.loss == "cross_entropy"


def test_hpo_documentation_example_runs_offline(monkeypatch, tmp_path):
    readme_example = _marked_python_example(
        _PROJECT_ROOT / "README.md",
        "hpo-example",
    ).strip()
    sphinx_example = _marked_rst_python_example(
        _PROJECT_ROOT / "docs" / "source" / "tutorial" / "hpo.rst",
        "hpo-example",
    )
    namespace = {"__name__": "__main__"}
    monkeypatch.chdir(tmp_path)

    assert sphinx_example == readme_example
    exec(compile(readme_example, "README.md", "exec"), namespace)

    assert isinstance(namespace["best_model_results"], dict)
    assert namespace["best_experiment"].info["model_dir"].endswith("model.pt")
    assert Path(namespace["best_experiment"].info["model_dir"]).is_file()


def test_migration_guide_tracks_the_current_public_contract():
    migration_path = _PROJECT_ROOT / "MIGRATION.rst"
    migration = migration_path.read_text(encoding="utf-8")
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    index = (_PROJECT_ROOT / "docs" / "source" / "index.rst").read_text(
        encoding="utf-8"
    )
    sphinx_page = (
        _PROJECT_ROOT / "docs" / "source" / "tutorial" / "migration.rst"
    ).read_text(encoding="utf-8")

    assert "DeepMTP 0.0.22" in migration
    assert "0.0.22 to 0.0.23" in migration
    assert "v0.0.22" in migration
    assert "from DeepMTP import DeepMTP, DeepMTPConfig, data_process" in migration
    assert "DeepMTPConfig.from_mapping(existing_config)" in migration
    assert "checkpoint_dir=checkpoint_path" in migration
    assert "return_predictions=True" in migration
    assert "MIGRATION.rst" in readme
    assert "tutorial/migration" in index
    assert sphinx_page.strip() == ".. include:: ../../../MIGRATION.rst"

    examples = _rst_python_examples(migration_path)
    assert examples
    for example in examples:
        ast.parse(example, filename="MIGRATION.rst")


def test_project_policies_are_complete_and_linked():
    policy_names = (
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "MIGRATION.rst",
        "SECURITY.md",
        "RELEASING.md",
    )
    policies = {
        name: (_PROJECT_ROOT / name).read_text(encoding="utf-8")
        for name in policy_names
    }
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    project_page = (_PROJECT_ROOT / "docs" / "source" / "project.rst").read_text(
        encoding="utf-8"
    )
    index = (_PROJECT_ROOT / "docs" / "source" / "index.rst").read_text(
        encoding="utf-8"
    )
    publish_workflow = (
        _PROJECT_ROOT / ".github" / "workflows" / "publish.yml"
    ).read_text(encoding="utf-8")
    ci_workflow = (_PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    dependency_review_workflow = (
        _PROJECT_ROOT / ".github" / "workflows" / "dependency-review.yml"
    ).read_text(encoding="utf-8")
    conda_environment = (_PROJECT_ROOT / "environment.yml").read_text(encoding="utf-8")
    readthedocs_config = (_PROJECT_ROOT / ".readthedocs.yaml").read_text(
        encoding="utf-8"
    )
    dependabot_config = (_PROJECT_ROOT / ".github" / "dependabot.yml").read_text(
        encoding="utf-8"
    )

    for policy_name in policy_names:
        assert policy_name in readme or policy_name == "MIGRATION.rst"
        assert policy_name in project_page

    assert "## [Unreleased]" in policies["CHANGELOG.md"]
    assert "## [0.0.23] - 2026-07-30" in policies["CHANGELOG.md"]
    assert "## [0.0.22] - 2022-12-08" in policies["CHANGELOG.md"]
    assert "[migration guide](MIGRATION.rst)" in policies["CHANGELOG.md"]
    assert "sphinx-build -W --keep-going" in policies["CONTRIBUTING.md"]
    assert "[SECURITY.md](SECURITY.md)" in policies["CONTRIBUTING.md"]
    assert "dimitrios.iliadis@ugent.be" in policies["SECURITY.md"]
    assert "Load only" in policies["SECURITY.md"]
    assert "`setuptools-scm`" in policies["RELEASING.md"]
    assert "PyPI Trusted Publishing" in policies["RELEASING.md"]
    assert "environment named `pypi`" in policies["RELEASING.md"]
    assert "python -m pytest" in policies["RELEASING.md"]
    assert "python -m pytest -m release_candidate" in policies["RELEASING.md"]
    assert "python -m twine check --strict" in policies["RELEASING.md"]
    assert "vX.Y.Z" in policies["RELEASING.md"]
    assert "scripts/verify_release_artifacts.py vX.Y.Z" in policies["RELEASING.md"]
    assert (
        "scripts/verify_release_artifacts.py --contract-only"
        in policies["RELEASING.md"]
    )
    assert "scripts/smoke_test_installation.py" in policies["RELEASING.md"]
    assert "scripts/smoke_test_optional_extras.py" in policies["RELEASING.md"]
    assert "Project policies <project>" in index
    assert "python -m twine check --strict dist/*" in publish_workflow
    assert "${{ github.event.release.tag_name }}" in publish_workflow
    assert 'scripts/verify_release_artifacts.py "$RELEASE_TAG" dist/*' in (
        publish_workflow
    )
    assert 'pip install "${wheel}[all]"' in publish_workflow
    assert "scripts/smoke_test_installation.py" in publish_workflow
    assert "scripts/smoke_test_optional_extras.py" in publish_workflow
    assert "PYPI_API_TOKEN" not in publish_workflow
    assert 'python: "3.13"' in ci_workflow
    assert 'python: "3.14"' in ci_workflow
    assert "python -m pip install -e . --group test" in ci_workflow
    assert "--constraint constraints/minimum.txt" in ci_workflow
    assert "python -m pip check" in ci_workflow
    assert "python -m pip install -e . --group quality" in ci_workflow
    assert "python -m pip install -e . --group docs" in ci_workflow
    assert 'pip install "${wheel}[all]"' in ci_workflow
    assert "scripts/smoke_test_installation.py" in ci_workflow
    assert "scripts/smoke_test_optional_extras.py" in ci_workflow
    assert "verify_release_artifacts.py --contract-only" in ci_workflow
    assert "--require-wheel-install" in ci_workflow
    assert "working-directory: ${{ runner.temp }}" in ci_workflow
    assert "actions/dependency-review-action@" in dependency_review_workflow
    assert 'fail-on-severity: "moderate"' in dependency_review_workflow
    assert (
        'fail-on-scopes: "runtime, development, unknown"' in dependency_review_workflow
    )
    assert "license-check: false" in dependency_review_workflow
    assert "contents: read" in dependency_review_workflow
    assert "contents: write" not in dependency_review_workflow
    assert "pull-requests: write" not in dependency_review_workflow
    assert "pip>=25.1" in conda_environment
    assert "- -e ." in conda_environment
    assert ".[dev]" not in conda_environment
    assert "-e .[all]" not in conda_environment
    assert "python -m pip install -e . --group docs" in readthedocs_config
    assert "configuration: docs/source/conf.py" in readthedocs_config
    assert re.findall(
        r'package-ecosystem: "([^"]+)"',
        dependabot_config,
    ) == ["pip", "github-actions", "pre-commit"]
    assert dependabot_config.count('interval: "weekly"') == 3
    assert dependabot_config.count("default-days: 7") == 3
    assert 'versioning-strategy: "increase-if-necessary"' in dependabot_config
    assert '"constraints/minimum.txt"' in dependabot_config
    assert dependabot_config.count('applies-to: "version-updates"') == 3
    assert 'applies-to: "security-updates"' not in dependabot_config


def test_publish_workflow_isolates_trusted_publishing_permissions():
    publish_workflow = (
        _PROJECT_ROOT / ".github" / "workflows" / "publish.yml"
    ).read_text(encoding="utf-8")
    build_job, publish_job = publish_workflow.split("\n  pypi-publish:", maxsplit=1)

    assert "permissions: {}" in publish_workflow
    assert "\n  release-build:" in build_job
    assert "contents: read" in build_job
    assert "id-token: write" not in build_job
    assert "actions/upload-artifact@" in build_job
    assert "needs: release-build" in publish_job
    assert "name: pypi" in publish_job
    assert "id-token: write" in publish_job
    assert "contents: read" not in publish_job
    assert "actions/download-artifact@" in publish_job
    assert "digest-mismatch: error" in publish_job
    assert "pypa/gh-action-pypi-publish@" in publish_job
    assert "packages-dir: dist/" in publish_job
    assert "attestations: true" in publish_job
    assert "actions/checkout@" not in publish_job
    assert "actions/setup-python@" not in publish_job
    assert "\n        run:" not in publish_job
    assert "password:" not in publish_workflow
    assert "secrets." not in publish_workflow


def test_remote_workflow_actions_use_immutable_commit_pins():
    action_reference = re.compile(
        r"^\s*uses:\s+"
        r"(?P<action>[^@\s]+)@"
        r"(?P<commit>[0-9a-f]{40})"
        r"\s+#\s+(?P<version>\S+)\s*$"
    )
    workflow_paths = sorted((_PROJECT_ROOT / ".github" / "workflows").glob("*.y*ml"))

    assert workflow_paths
    for workflow_path in workflow_paths:
        uses_lines = [
            line
            for line in workflow_path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("uses:")
        ]
        for line in uses_lines:
            reference = line.split("uses:", maxsplit=1)[1].strip()
            if reference.startswith(("./", "docker://")):
                continue
            assert action_reference.fullmatch(line), (
                f"{workflow_path.name} must pin remote action reference "
                f"{reference!r} to a full commit SHA with a version comment"
            )


def test_dependency_metadata_separates_runtime_features_from_development_tools():
    project_metadata = tomllib.loads(
        (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    runtime_extras = project_metadata["project"]["optional-dependencies"]
    dependency_groups = project_metadata["dependency-groups"]

    assert set(runtime_extras) == {
        "all",
        "datasets",
        "foundation",
        "graph",
        "hpo",
        "image",
        "sparse",
        "streamlit",
        "tracking",
    }
    assert set(dependency_groups) == {"dev", "docs", "quality", "test"}
    assert dependency_groups["dev"] == [
        {"include-group": "test"},
        {"include-group": "quality"},
        {"include-group": "docs"},
        "build>=1.2",
        "twine>=6",
    ]


def test_minimum_constraints_cover_direct_test_dependencies():
    project_metadata = tomllib.loads(
        (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    constraint_lines = (
        (_PROJECT_ROOT / "constraints" / "minimum.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    constraints = {}
    for raw_line in constraint_lines:
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        requirement = line.split(";", maxsplit=1)[0].strip()
        name, version = requirement.split("==", maxsplit=1)
        constraints[name.lower()] = version

    direct_requirements = [
        *project_metadata["project"]["dependencies"],
        *project_metadata["dependency-groups"]["test"],
    ]
    direct_dependency_names = {
        re.split(r"[<>=!~\s\[]", requirement, maxsplit=1)[0].lower()
        for requirement in direct_requirements
    }

    assert set(constraints) == direct_dependency_names
    assert constraints == {
        "configspace": "1.2.2",
        "liac-arff": "2.5.0",
        "numpy": "1.23.5",
        "pandas": "1.5.3",
        "peft": "0.14.0",
        "pillow": "9.5.0",
        "prettytable": "3.3.0",
        "pytest": "8.0.2",
        "requests": "2.28.2",
        "scikit-learn": "1.2.2",
        "scipy": "1.9.3",
        "tomli": "2.0.1",
        "torch": "2.0.1",
        "torch-geometric": "2.5.0",
        "transformers": "4.48.3",
        "wandb": "0.27.2",
    }


def test_sphinx_config_exposes_installed_distribution_version(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))

    configuration = runpy.run_path(str(_PROJECT_ROOT / "docs" / "source" / "conf.py"))
    installed_version = distribution_version("DeepMTP")

    assert configuration["version"] == installed_version
    assert configuration["release"] == installed_version


def test_demo_links_reference_version_controlled_notebooks():
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    sphinx_demos = (
        _PROJECT_ROOT / "docs" / "source" / "tutorial" / "demos.rst"
    ).read_text(encoding="utf-8")

    for document in (readme, sphinx_demos):
        assert "colab.research.google.com/drive/" not in document
        for notebook_path in _DEMO_NOTEBOOKS:
            assert f"DEMO/{notebook_path.name}" in document


def test_colab_link_checker_maps_every_demo_notebook():
    checker = runpy.run_path(str(_PROJECT_ROOT / "scripts" / "check_colab_links.py"))
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    links = checker["extract_colab_links"](readme)
    checker["validate_notebook_mapping"](links)

    assert len(links) == len(_DEMO_NOTEBOOKS)


def test_local_dataset_notebook_runs_offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    namespace = _execute_notebook(_PROJECT_ROOT / "DEMO" / "load_datasets.ipynb")

    assert namespace["data_info_mlc"]["detected_problem_mode"] == "classification"
    assert namespace["data_info_mtr"]["detected_problem_mode"] == "regression"
    assert namespace["data_info_mtl"]["detected_problem_mode"] == "classification"
    assert namespace["data_info_dp"]["detected_validation_setting"] == "D"
    assert namespace["data_info_mc"]["detected_validation_setting"] == "A"


@pytest.mark.release_candidate
@pytest.mark.integration
@pytest.mark.parametrize(
    (
        "notebook_name",
        "validation_setting",
        "problem_mode",
        "expected_validation_metrics",
        "has_instance_features",
        "has_target_features",
    ),
    _RELEASE_CANDIDATE_WORKLOADS,
)
def test_training_notebook_runs_offline(
    notebook_name,
    validation_setting,
    problem_mode,
    expected_validation_metrics,
    has_instance_features,
    has_target_features,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    namespace = _execute_notebook(_PROJECT_ROOT / "DEMO" / notebook_name)

    model = namespace["model"]
    data_info = namespace["data_info"]
    validation_results = namespace["validation_results"]
    predictions = namespace["preds"]
    restored_results = namespace["restored_results"]

    assert model.configuration.compute_mode == "cpu"
    assert model.configuration.num_epochs == 1
    assert model.configuration.use_tensorboard_logger is False
    assert model.configuration.validation_setting == validation_setting
    assert model.configuration.problem_mode == problem_mode
    assert data_info["detected_validation_setting"] == validation_setting
    assert data_info["detected_problem_mode"] == problem_mode
    assert (
        namespace["data"]["train"]["X_instance"] is not None
    ) is has_instance_features
    assert (namespace["data"]["train"]["X_target"] is not None) is has_target_features

    assert expected_validation_metrics.issubset(validation_results)
    assert all(
        math.isfinite(validation_results[metric])
        for metric in expected_validation_metrics
    )
    assert isinstance(predictions, pd.DataFrame)
    assert list(predictions) == [
        "instance_id",
        "target_id",
        "true_values",
        "predicted_values",
    ]
    assert len(predictions) == len(namespace["train"]["y"]["data"])

    checkpoint_path = Path(model.experiment_dir) / "model.pt"
    assert checkpoint_path.is_file()
    assert isinstance(restored_results, dict)
    original_test_results = model.predict(namespace["test"])
    assert original_test_results == pytest.approx(restored_results)


@pytest.mark.parametrize(
    "notebook_name",
    ["hyperband_example.ipynb", "random_search_example.ipynb"],
)
def test_hpo_notebook_search_spaces_use_current_configspace_api(
    notebook_name,
):
    namespace = _execute_notebook(
        _PROJECT_ROOT / "DEMO" / notebook_name,
        max_code_cells=3,
    )

    assert len(namespace["cs"]) == 8


@pytest.mark.parametrize(
    "notebook_name",
    ["hyperband_example.ipynb", "random_search_example.ipynb"],
)
def test_hpo_notebook_runs_offline(
    notebook_name,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    namespace = _execute_notebook(_PROJECT_ROOT / "DEMO" / notebook_name)

    assert isinstance(namespace["best_model_results"], dict)
    assert namespace["best_overall_config"].info["model_dir"].endswith("model.pt")


@pytest.mark.parametrize("notebook_path", _DEMO_NOTEBOOKS, ids=lambda path: path.name)
def test_demo_notebook_code_uses_current_public_api(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = _notebook_code_cells(notebook_path)
    source = "\n\n".join(code_cells)
    document_source = "\n\n".join("".join(cell["source"]) for cell in notebook["cells"])

    code_cell_data = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cell_data)
    assert all(cell["outputs"] == [] for cell in code_cell_data)
    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    assert "vscode" not in notebook["metadata"]
    assert "version" not in notebook["metadata"]["language_info"]
    assert 'src="../images/' not in document_source

    bootstrap_source = "".join(code_cell_data[0]["source"])
    expected_requirement = (
        _COLAB_HPO_REQUIREMENT
        if notebook_path.name in _HPO_DEMO_NOTEBOOKS
        else _COLAB_CORE_REQUIREMENT
    )
    assert 'find_spec("google.colab")' in bootstrap_source
    assert expected_requirement in bootstrap_source
    assert "subprocess.run(" in bootstrap_source
    assert "check=True" in bootstrap_source

    if notebook_path.name.startswith("main_example_"):
        assert "DeepMTPConfig(" in source

    for snippet in _FORBIDDEN_NOTEBOOK_SNIPPETS:
        assert snippet not in document_source

    tree = ast.parse(source, filename=str(notebook_path))
    positional_checkpoint_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DeepMTP"
        and len(node.args) > 1
    ]

    assert positional_checkpoint_calls == []
