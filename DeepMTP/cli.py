"""Command-line tools for inspecting DeepMTP experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from DeepMTP.architecture import (
    describe_architecture,
    export_architecture_svg,
    print_architecture,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepmtp")
    commands = parser.add_subparsers(dest="command", required=True)
    architecture = commands.add_parser(
        "architecture",
        help="Visualize the architecture described by a JSON configuration.",
    )
    architecture.add_argument("config", type=Path, help="Path to config.json.")
    architecture.add_argument(
        "--format",
        choices=("auto", "json", "plain", "rich"),
        default="auto",
        help="Terminal rendering format (default: auto).",
    )
    architecture.add_argument(
        "--width",
        type=int,
        default=None,
        help="Rendering width; the current terminal width is used by default.",
    )
    architecture.add_argument(
        "--export-svg",
        type=Path,
        help="Also export the architecture diagram to this SVG path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DeepMTP command-line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        config_data = json.loads(arguments.config.read_text(encoding="utf-8"))
        if not isinstance(config_data, dict):
            raise ValueError("the configuration root must be a JSON object")
        description = describe_architecture(config_data)
        if arguments.format == "json":
            sys.stdout.write(
                json.dumps(description.to_dict(), indent=2, sort_keys=True) + "\n"
            )
        else:
            print_architecture(
                description,
                style=arguments.format,
                width=arguments.width,
            )
        if arguments.export_svg is not None:
            export_architecture_svg(
                description,
                arguments.export_svg,
                width=arguments.width or 140,
            )
    except (ImportError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"deepmtp: error: {error}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
