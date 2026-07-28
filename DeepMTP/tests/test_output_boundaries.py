import ast
import subprocess
import sys
from pathlib import Path

import DeepMTP

_ALLOWED_PRINT_METHODS = {
    Path("data_progress.py"): {
        ("ConsoleDataProgressObserver", "on_event"),
    },
    Path("evaluation_progress.py"): {
        ("ConsoleEvaluationProgressObserver", "on_event"),
    },
    Path("hpo/progress.py"): {
        ("ConsoleHPOProgressObserver", "on_event"),
    },
    Path("presentation.py"): {
        ("ConsoleProgressObserver", "on_event"),
        ("ConsoleProgressObserver", "on_summary"),
    },
}

_ALLOWED_CONFIG_LOOKUP_METHODS = {
    Path("data_loading.py"): {
        ("_DatasetModelSettings", "from_config"),
    },
    Path("model_factory.py"): {
        ("_CombinedMLPSettings", "from_config"),
    },
    Path("reporting.py"): {
        ("_ReporterSettings", "from_config"),
    },
}


class _DirectPrintVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.class_names: list[str] = []
        self.function_names: list[str] = []
        self.locations: list[tuple[int, str | None, str | None]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_names.append(node.name)
        self.generic_visit(node)
        self.class_names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_names.append(node.name)
        self.generic_visit(node)
        self.function_names.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_names.append(node.name)
        self.generic_visit(node)
        self.function_names.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.locations.append(
                (
                    node.lineno,
                    self.class_names[-1] if self.class_names else None,
                    self.function_names[-1] if self.function_names else None,
                )
            )
        self.generic_visit(node)


class _DirectConfigLookupVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.class_names: list[str] = []
        self.function_names: list[str] = []
        self.locations: list[tuple[int, str | None, str | None]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_names.append(node.name)
        self.generic_visit(node)
        self.class_names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_names.append(node.name)
        self.generic_visit(node)
        self.function_names.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_names.append(node.name)
        self.generic_visit(node)
        self.function_names.pop()

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "config"
        ):
            self._record(node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "config"
        ):
            self._record(node.lineno)
        self.generic_visit(node)

    def _record(self, line_number: int) -> None:
        self.locations.append(
            (
                line_number,
                self.class_names[-1] if self.class_names else None,
                self.function_names[-1] if self.function_names else None,
            )
        )


def test_direct_printing_is_confined_to_console_adapters():
    package_root = Path(DeepMTP.__file__).parent
    violations = []

    for source_path in sorted(package_root.rglob("*.py")):
        relative_path = source_path.relative_to(package_root)
        if relative_path.parts[0] == "tests":
            continue

        visitor = _DirectPrintVisitor()
        visitor.visit(ast.parse(source_path.read_text(encoding="utf-8")))
        allowed_methods = _ALLOWED_PRINT_METHODS.get(relative_path, set())
        for line_number, class_name, function_name in visitor.locations:
            if (class_name, function_name) not in allowed_methods:
                violations.append(
                    f"{relative_path}:{line_number} "
                    f"({class_name or '<module>'}.{function_name or '<module>'})"
                )

    assert violations == [], (
        "Direct printing must stay inside explicit console adapters; use an "
        f"injectable observer instead: {violations}"
    )


def test_direct_config_lookups_are_confined_to_compatibility_adapters():
    package_root = Path(DeepMTP.__file__).parent
    violations = []

    for source_path in sorted(package_root.rglob("*.py")):
        relative_path = source_path.relative_to(package_root)
        if relative_path.parts[0] == "tests":
            continue

        visitor = _DirectConfigLookupVisitor()
        visitor.visit(ast.parse(source_path.read_text(encoding="utf-8")))
        allowed_methods = _ALLOWED_CONFIG_LOOKUP_METHODS.get(relative_path, set())
        for line_number, class_name, function_name in visitor.locations:
            if (class_name, function_name) not in allowed_methods:
                violations.append(
                    f"{relative_path}:{line_number} "
                    f"({class_name or '<module>'}.{function_name or '<module>'})"
                )

    assert violations == [], (
        "Direct config mapping reads must stay inside explicit compatibility "
        f"adapters; use DeepMTPConfig attributes internally: {violations}"
    )


def test_importing_data_utils_preserves_pandas_options():
    script = """
import pandas as pd

pd.set_option("mode.chained_assignment", "raise")
import DeepMTP.utils.data_utils  # noqa: F401

actual = pd.get_option("mode.chained_assignment")
if actual != "raise":
    raise AssertionError(
        "import changed mode.chained_assignment from 'raise' to "
        f"{actual!r}"
    )
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
