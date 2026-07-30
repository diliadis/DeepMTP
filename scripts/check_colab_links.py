"""Check that every README Colab demo link resolves successfully."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_COLAB_LINK_PATTERN = re.compile(
    r"https://colab\.research\.google\.com/github/diliadis/DeepMTP/"
    r"blob/main/DEMO/[A-Za-z0-9_]+\.ipynb"
)


def extract_colab_links(readme: str) -> tuple[str, ...]:
    """Return unique DeepMTP Colab links in document order."""
    return tuple(dict.fromkeys(_COLAB_LINK_PATTERN.findall(readme)))


def validate_notebook_mapping(links: tuple[str, ...]) -> None:
    """Require exactly one README link for every version-controlled demo."""
    expected_names = {path.name for path in (_PROJECT_ROOT / "DEMO").glob("*.ipynb")}
    linked_names = {Path(urlparse(link).path).name for link in links}
    if linked_names != expected_names or len(links) != len(expected_names):
        missing = sorted(expected_names - linked_names)
        unexpected = sorted(linked_names - expected_names)
        raise ValueError(
            "README Colab links do not match DEMO notebooks: "
            f"missing={missing}, unexpected={unexpected}."
        )


def check_colab_link(url: str, *, timeout: float = 30.0) -> None:
    """Open one Colab URL and require a successful HTTP response."""
    request = Request(
        url,
        headers={"User-Agent": "DeepMTP-Colab-Link-Check/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}.")
        response.read(1)


def main() -> int:
    """Validate README coverage and the live Colab endpoints."""
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    links = extract_colab_links(readme)
    try:
        validate_notebook_mapping(links)
        for link in links:
            check_colab_link(link)
            print(f"OK {link}")
    except (OSError, RuntimeError, URLError, ValueError) as error:
        print(f"Colab link check failed: {error}", file=sys.stderr)
        return 1
    print(f"Verified {len(links)} Colab demo links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
