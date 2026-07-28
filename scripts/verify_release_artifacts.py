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


class ReleaseArtifactError(ValueError):
    """Raised when release artifacts do not match the intended release."""


@dataclass(frozen=True)
class ArtifactMetadata:
    """Metadata read directly from one built distribution."""

    path: Path
    kind: str
    name: str
    version: str


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_metadata(raw_metadata: bytes, path: Path, kind: str) -> ArtifactMetadata:
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
    )


def _read_wheel_metadata(path: Path) -> ArtifactMetadata:
    try:
        with ZipFile(path) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise ReleaseArtifactError(
                    f"{path} must contain exactly one .dist-info/METADATA file."
                )
            return _parse_metadata(
                archive.read(metadata_names[0]),
                path,
                "wheel",
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
            metadata_members = [
                member
                for member in archive.getmembers()
                if member.isfile()
                and len(PurePosixPath(member.name).parts) == 2
                and PurePosixPath(member.name).name == "PKG-INFO"
            ]
            if len(metadata_members) != 1:
                raise ReleaseArtifactError(
                    f"{path} must contain exactly one sdist-root PKG-INFO file."
                )
            metadata_file = archive.extractfile(metadata_members[0])
            if metadata_file is None:
                raise ReleaseArtifactError(f"Could not read PKG-INFO from {path}.")
            return _parse_metadata(metadata_file.read(), path, "sdist")
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


def verify_release_artifacts(
    release_tag: str,
    artifact_paths: Sequence[str | Path],
) -> str:
    """Verify one wheel and one sdist against the intended release tag."""
    expected_version = release_version_from_tag(release_tag)
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
        if ".dev" in artifact.version or "+" in artifact.version:
            raise ReleaseArtifactError(
                f"{artifact.path} contains non-release version {artifact.version!r}."
            )
        if artifact.version != expected_version:
            raise ReleaseArtifactError(
                f"{artifact.path} contains version {artifact.version!r}; "
                f"release tag {release_tag!r} requires {expected_version!r}."
            )
    return expected_version


def main(argv: Sequence[str] | None = None) -> int:
    """Run release artifact verification from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_tag", help="Git tag, for example v0.1.0")
    parser.add_argument("artifacts", nargs="+", help="Built wheel and sdist paths")
    arguments = parser.parse_args(argv)

    try:
        verified_version = verify_release_artifacts(
            arguments.release_tag,
            arguments.artifacts,
        )
    except ReleaseArtifactError as exc:
        print(f"Release artifact verification failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Verified DeepMTP {verified_version}: one wheel and one source distribution."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
