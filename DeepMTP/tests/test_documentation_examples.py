import ast
import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parents[2]
_DEMO_NOTEBOOKS = sorted((_PROJECT_ROOT / "DEMO").glob("*.ipynb"))
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


def test_demo_links_reference_version_controlled_notebooks():
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    sphinx_demos = (
        _PROJECT_ROOT / "docs" / "source" / "tutorial" / "demos.rst"
    ).read_text(encoding="utf-8")

    for document in (readme, sphinx_demos):
        assert "colab.research.google.com/drive/" not in document
        for notebook_path in _DEMO_NOTEBOOKS:
            assert f"DEMO/{notebook_path.name}" in document


def test_local_dataset_notebook_runs_offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    namespace = _execute_notebook(_PROJECT_ROOT / "DEMO" / "load_datasets.ipynb")

    assert namespace["data_info_mlc"]["detected_problem_mode"] == "classification"
    assert namespace["data_info_mtr"]["detected_problem_mode"] == "regression"
    assert namespace["data_info_mtl"]["detected_problem_mode"] == "classification"
    assert namespace["data_info_dp"]["detected_validation_setting"] == "D"
    assert namespace["data_info_mc"]["detected_validation_setting"] == "A"


@pytest.mark.parametrize(
    "notebook_name",
    [
        "main_example_DP.ipynb",
        "main_example_MC.ipynb",
        "main_example_MLC.ipynb",
        "main_example_MTL.ipynb",
        "main_example_MTR.ipynb",
    ],
)
def test_training_notebook_runs_offline(
    notebook_name,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    namespace = _execute_notebook(_PROJECT_ROOT / "DEMO" / notebook_name)

    model = namespace["model"]
    assert model.configuration.compute_mode == "cpu"
    assert model.configuration.num_epochs == 1
    assert model.configuration.use_tensorboard_logger is False
    assert "val_loss" in namespace["validation_results"]
    assert isinstance(namespace["preds"], pd.DataFrame)
    assert isinstance(namespace["restored_results"], dict)


@pytest.mark.parametrize(
    "notebook_name",
    ["hyperband_example.ipynb", "random_search_example.ipynb"],
)
def test_hpo_notebook_search_spaces_use_current_configspace_api(
    notebook_name,
):
    namespace = _execute_notebook(
        _PROJECT_ROOT / "DEMO" / notebook_name,
        max_code_cells=2,
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
