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


def _marked_rst_python_example(document: Path, marker: str) -> str:
    text = document.read_text(encoding="utf-8")
    marked_section = text.split(f".. {marker}-start", maxsplit=1)[1].split(
        f".. {marker}-end",
        maxsplit=1,
    )[0]
    code_block = marked_section.split(".. code-block:: python", maxsplit=1)[1]
    return textwrap.dedent(code_block).strip()


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


@pytest.mark.parametrize("notebook_path", _DEMO_NOTEBOOKS, ids=lambda path: path.name)
def test_demo_notebook_code_uses_current_public_api(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = _notebook_code_cells(notebook_path)
    source = "\n\n".join(code_cells)

    code_cell_data = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cell_data)
    assert all(cell["outputs"] == [] for cell in code_cell_data)

    for snippet in _FORBIDDEN_NOTEBOOK_SNIPPETS:
        assert snippet not in source

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
