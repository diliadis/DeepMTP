"""Verify that built DeepMTP artifacts match an intended release tag."""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

_DISTRIBUTION_NAME = "DeepMTP"
_RELEASE_TAG_PATTERN = re.compile(
    r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?)$"
)
_EXPECTED_REQUIRES_PYTHON = ">=3.10"
_EXPECTED_EXTRAS = frozenset(
    {
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
)
_EXPECTED_PROJECT_URLS = frozenset(
    {
        ("Documentation", "https://deepmtp.readthedocs.io/"),
        ("Issues", "https://github.com/diliadis/DeepMTP/issues"),
        ("Repository", "https://github.com/diliadis/DeepMTP"),
    }
)
_REQUIRED_WHEEL_FILES = frozenset(
    {
        PurePosixPath("DeepMTP/__init__.py"),
        PurePosixPath("DeepMTP/config.py"),
        PurePosixPath("DeepMTP/data/__init__.py"),
        PurePosixPath("DeepMTP/hpo/__init__.py"),
        PurePosixPath("DeepMTP/integrations/huggingface.py"),
        PurePosixPath("DeepMTP/integrations/streamlit/__init__.py"),
        PurePosixPath("DeepMTP/main.py"),
        PurePosixPath("DeepMTP/models/__init__.py"),
        PurePosixPath("DeepMTP/runtime/__init__.py"),
    }
)
_REQUIRED_SDIST_FILES = frozenset(
    {
        PurePosixPath(".github/workflows/ci.yml"),
        PurePosixPath(".github/workflows/publish.yml"),
        PurePosixPath("CHANGELOG.md"),
        PurePosixPath("CONTRIBUTING.md"),
        PurePosixPath("DEMO/main_example_MLC.ipynb"),
        PurePosixPath("LICENSE.md"),
        PurePosixPath("MIGRATION.rst"),
        PurePosixPath("README.md"),
        PurePosixPath("RELEASING.md"),
        PurePosixPath("SECURITY.md"),
        PurePosixPath("docs/source/tutorial/foundation_models.rst"),
        PurePosixPath("docs/source/index.rst"),
        PurePosixPath("pyproject.toml"),
        PurePosixPath("scripts/smoke_test_installation.py"),
        PurePosixPath("scripts/smoke_test_optional_extras.py"),
        PurePosixPath("scripts/verify_release_artifacts.py"),
    }
)


class ReleaseArtifactError(ValueError):
    """Raised when release artifacts do not match the intended release."""


@dataclass(frozen=True)
class ArtifactMetadata:
    """Metadata read directly from one built distribution."""

    path: Path
    kind: str
    name: str
    version: str
    requires_python: str | None
    license_expression: str | None
    extras: frozenset[str]
    project_urls: frozenset[tuple[str, str]]
    files: frozenset[PurePosixPath]


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_project_url(value: str, path: Path) -> tuple[str, str]:
    parts = value.split(",", maxsplit=1)
    if len(parts) != 2:
        raise ReleaseArtifactError(
            f"{path} contains malformed Project-URL metadata: {value!r}."
        )
    return parts[0].strip(), parts[1].strip()


def _parse_metadata(
    raw_metadata: bytes,
    path: Path,
    kind: str,
    files: frozenset[PurePosixPath],
) -> ArtifactMetadata:
    parsed = BytesParser().parsebytes(raw_metadata)
    name = parsed.get("Name")
    version = parsed.get("Version")
    if not name or not version:
        raise ReleaseArtifactError(
            f"{path} has incomplete package metadata; Name and Version are required."
        )
    return ArtifactMetadata(
        path=path,
        kind=kind,
        name=name,
        version=version,
        requires_python=parsed.get("Requires-Python"),
        license_expression=parsed.get("License-Expression"),
        extras=frozenset(parsed.get_all("Provides-Extra", [])),
        project_urls=frozenset(
            _parse_project_url(value, path)
            for value in parsed.get_all("Project-URL", [])
        ),
        files=files,
    )


def _read_wheel_metadata(path: Path) -> ArtifactMetadata:
    try:
        with ZipFile(path) as archive:
            files = frozenset(
                PurePosixPath(name)
                for name in archive.namelist()
                if not name.endswith("/")
            )
            metadata_names = [
                str(name) for name in files if str(name).endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise ReleaseArtifactError(
                    f"{path} must contain exactly one .dist-info/METADATA file."
                )
            return _parse_metadata(
                archive.read(metadata_names[0]),
                path,
                "wheel",
                files,
            )
    except ReleaseArtifactError:
        raise
    except (BadZipFile, OSError, KeyError) as exc:
        raise ReleaseArtifactError(
            f"Could not read wheel metadata from {path}."
        ) from exc


def _read_sdist_metadata(path: Path) -> ArtifactMetadata:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            archive_members = [
                member for member in archive.getmembers() if member.isfile()
            ]
            files = frozenset(
                PurePosixPath(*member_path.parts[1:])
                for member in archive_members
                if len((member_path := PurePosixPath(member.name)).parts) > 1
            )
            metadata_members = [
                member
                for member in archive_members
                if len(PurePosixPath(member.name).parts) == 2
                and PurePosixPath(member.name).name == "PKG-INFO"
            ]
            if len(metadata_members) != 1:
                raise ReleaseArtifactError(
                    f"{path} must contain exactly one sdist-root PKG-INFO file."
                )
            metadata_file = archive.extractfile(metadata_members[0])
            if metadata_file is None:
                raise ReleaseArtifactError(f"Could not read PKG-INFO from {path}.")
            return _parse_metadata(metadata_file.read(), path, "sdist", files)
    except ReleaseArtifactError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseArtifactError(
            f"Could not read source-distribution metadata from {path}."
        ) from exc


def read_artifact_metadata(path: str | Path) -> ArtifactMetadata:
    """Read package name and version directly from a wheel or source archive."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ReleaseArtifactError(f"Release artifact does not exist: {artifact_path}")
    if artifact_path.suffix == ".whl":
        return _read_wheel_metadata(artifact_path)
    if artifact_path.name.endswith(".tar.gz"):
        return _read_sdist_metadata(artifact_path)
    raise ReleaseArtifactError(
        f"Unsupported release artifact {artifact_path}; expected .whl or .tar.gz."
    )


def release_version_from_tag(release_tag: str) -> str:
    """Return the exact artifact version permitted by a release tag."""
    match = _RELEASE_TAG_PATTERN.fullmatch(release_tag)
    if match is None:
        raise ReleaseArtifactError(
            "Release tags must use vX.Y.Z or a PEP 440 prerelease suffix such as "
            "vX.Y.Zrc1."
        )
    return match.group("version")


def verify_distribution_contract(
    artifact_paths: Sequence[str | Path],
) -> str:
    """Verify metadata and contents shared by development and release builds."""
    artifacts = [read_artifact_metadata(path) for path in artifact_paths]
    artifact_kinds = [artifact.kind for artifact in artifacts]
    if artifact_kinds.count("wheel") != 1 or artifact_kinds.count("sdist") != 1:
        raise ReleaseArtifactError(
            "A release must contain exactly one wheel and one source distribution."
        )

    expected_name = _normalize_distribution_name(_DISTRIBUTION_NAME)
    for artifact in artifacts:
        if _normalize_distribution_name(artifact.name) != expected_name:
            raise ReleaseArtifactError(
                f"{artifact.path} contains distribution {artifact.name!r}; "
                f"expected {_DISTRIBUTION_NAME!r}."
            )
        if artifact.requires_python != _EXPECTED_REQUIRES_PYTHON:
            raise ReleaseArtifactError(
                f"{artifact.path} declares Requires-Python "
                f"{artifact.requires_python!r}; expected {_EXPECTED_REQUIRES_PYTHON!r}."
            )
        if artifact.license_expression != "MIT":
            raise ReleaseArtifactError(
                f"{artifact.path} declares License-Expression "
                f"{artifact.license_expression!r}; expected 'MIT'."
            )
        if artifact.extras != _EXPECTED_EXTRAS:
            raise ReleaseArtifactError(
                f"{artifact.path} extras do not match the runtime contract: "
                f"expected={sorted(_EXPECTED_EXTRAS)}, "
                f"actual={sorted(artifact.extras)}."
            )
        if artifact.project_urls != _EXPECTED_PROJECT_URLS:
            raise ReleaseArtifactError(
                f"{artifact.path} project URLs do not match the runtime contract."
            )

    versions = {artifact.version for artifact in artifacts}
    if len(versions) != 1:
        raise ReleaseArtifactError(
            "The wheel and source distribution must declare the same version; "
            f"found {sorted(versions)}."
        )

    wheel = next(artifact for artifact in artifacts if artifact.kind == "wheel")
    missing_wheel_files = _REQUIRED_WHEEL_FILES - wheel.files
    if missing_wheel_files:
        raise ReleaseArtifactError(
            f"{wheel.path} is missing required runtime files: "
            f"{sorted(map(str, missing_wheel_files))}."
        )
    packaged_tests = sorted(
        str(path)
        for path in wheel.files
        if len(path.parts) >= 2 and path.parts[:2] == ("DeepMTP", "tests")
    )
    if packaged_tests:
        raise ReleaseArtifactError(
            f"{wheel.path} must not contain DeepMTP/tests; found {packaged_tests}."
        )

    sdist = next(artifact for artifact in artifacts if artifact.kind == "sdist")
    missing_sdist_files = _REQUIRED_SDIST_FILES - sdist.files
    if missing_sdist_files:
        raise ReleaseArtifactError(
            f"{sdist.path} is missing required project files: "
            f"{sorted(map(str, missing_sdist_files))}."
        )

    return versions.pop()


def verify_release_artifacts(
    release_tag: str,
    artifact_paths: Sequence[str | Path],
) -> str:
    """Verify one wheel and one sdist against the intended release tag."""
    expected_version = release_version_from_tag(release_tag)
    artifact_version = verify_distribution_contract(artifact_paths)
    if ".dev" in artifact_version or "+" in artifact_version:
        raise ReleaseArtifactError(
            f"Artifacts contain non-release version {artifact_version!r}."
        )
    if artifact_version != expected_version:
        raise ReleaseArtifactError(
            f"Artifacts contain version {artifact_version!r}; "
            f"release tag {release_tag!r} requires {expected_version!r}."
        )
    return expected_version


def main(argv: Sequence[str] | None = None) -> int:
    """Run release artifact verification from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="Validate metadata and contents without requiring a release tag.",
    )
    parser.add_argument(
        "values",
        nargs="+",
        help="Release tag followed by artifacts, or artifacts with --contract-only.",
    )
    arguments = parser.parse_args(argv)

    try:
        if arguments.contract_only:
            verified_version = verify_distribution_contract(arguments.values)
        else:
            if len(arguments.values) < 3:
                parser.error("a release tag, wheel, and sdist are required")
            release_tag, *artifact_paths = arguments.values
            verified_version = verify_release_artifacts(
                release_tag,
                artifact_paths,
            )
    except ReleaseArtifactError as exc:
        print(f"Release artifact verification failed: {exc}", file=sys.stderr)
        return 1

    verification_kind = (
        "distribution contract" if arguments.contract_only else "release artifacts"
    )
    print(
        f"Verified DeepMTP {verified_version} {verification_kind}: "
        "one wheel and one source distribution."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
