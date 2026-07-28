from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).parents[2]


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
