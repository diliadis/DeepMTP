import ast
from pathlib import Path

import DeepMTP

_ALLOWED_PRINT_METHODS = {
    Path("data_progress.py"): {
        ("ConsoleDataProgressObserver", "on_event"),
    },
    Path("evaluation_progress.py"): {
        ("ConsoleEvaluationProgressObserver", "on_event"),
    },
    Path("hpo_progress.py"): {
        ("ConsoleHPOProgressObserver", "on_event"),
    },
    Path("presentation.py"): {
        ("ConsoleProgressObserver", "on_event"),
        ("ConsoleProgressObserver", "on_summary"),
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
