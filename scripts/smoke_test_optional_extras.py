"""Smoke-import every optional integration from an installed DeepMTP wheel."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import DeepMTP as deepmtp_package

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PACKAGE_ROOT = _PROJECT_ROOT / "DeepMTP"
_OPTIONAL_IMPORTS = {
    "datasets": ("arff", "requests", "scipy", "DeepMTP.data.datasets"),
    "foundation": ("transformers", "peft", "DeepMTP.integrations.huggingface"),
    "graph": ("torch_geometric", "DeepMTP.data.graphs"),
    "hpo": ("ConfigSpace", "DeepMTP.hpo"),
    "image": ("torchvision",),
    "sparse": ("scipy", "DeepMTP.data.sparse"),
    "streamlit": ("streamlit", "DeepMTP.integrations.streamlit"),
    "tracking": ("tensorboard", "wandb", "DeepMTP.runtime.reporting"),
    "visualization": ("rich", "DeepMTP.architecture"),
}


def require_installed_wheel() -> None:
    """Reject an import resolved from the source checkout."""
    package_file = deepmtp_package.__file__
    if package_file is None:
        raise RuntimeError("DeepMTP does not expose an importable package path.")
    package_path = Path(package_file).resolve()
    if package_path.is_relative_to(_SOURCE_PACKAGE_ROOT):
        raise RuntimeError(
            f"DeepMTP resolved from the source checkout at {package_path}; "
            "the optional-extra smoke test requires an installed wheel."
        )


def verify_optional_integrations(
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> tuple[str, ...]:
    """Import each dependency and DeepMTP surface promised by runtime extras."""
    imported_extras = []
    for extra, module_names in _OPTIONAL_IMPORTS.items():
        for module_name in module_names:
            try:
                import_module(module_name)
            except Exception as exc:
                raise RuntimeError(
                    f"DeepMTP[{extra}] could not import {module_name!r}."
                ) from exc
        imported_extras.append(extra)
    return tuple(imported_extras)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional-integration smoke test from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-wheel-install",
        action="store_true",
        help="Fail if DeepMTP resolves from the source checkout.",
    )
    arguments = parser.parse_args(argv)

    if arguments.require_wheel_install:
        require_installed_wheel()
    imported_extras = verify_optional_integrations()
    print(f"Verified installed DeepMTP optional extras: {', '.join(imported_extras)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
