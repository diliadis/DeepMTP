import tarfile
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.verify_release_artifacts import (
    ReleaseArtifactError,
    main,
    release_version_from_tag,
    verify_release_artifacts,
)


def _metadata(name: str, version: str) -> bytes:
    return (f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n").encode()


def _write_wheel(
    directory: Path,
    *,
    name: str = "DeepMTP",
    version: str = "0.1.0",
) -> Path:
    path = directory / f"deepmtp-{version}-py3-none-any.whl"
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            f"deepmtp-{version}.dist-info/METADATA",
            _metadata(name, version),
        )
    return path


def _write_sdist(
    directory: Path,
    *,
    name: str = "DeepMTP",
    version: str = "0.1.0",
) -> Path:
    path = directory / f"deepmtp-{version}.tar.gz"
    metadata = _metadata(name, version)
    metadata_info = tarfile.TarInfo(f"deepmtp-{version}/PKG-INFO")
    metadata_info.size = len(metadata)
    egg_info = tarfile.TarInfo(f"deepmtp-{version}/DeepMTP.egg-info/PKG-INFO")
    egg_info.size = len(metadata)
    with tarfile.open(path, mode="w:gz") as archive:
        archive.addfile(metadata_info, BytesIO(metadata))
        archive.addfile(egg_info, BytesIO(metadata))
    return path


@pytest.mark.parametrize(
    ("release_tag", "version"),
    [
        ("v0.1.0", "0.1.0"),
        ("v1.2.3a1", "1.2.3a1"),
        ("v1.2.3b2", "1.2.3b2"),
        ("v1.2.3rc4", "1.2.3rc4"),
    ],
)
def test_release_artifacts_match_exact_release_tag(
    tmp_path,
    release_tag,
    version,
):
    artifacts = [
        _write_wheel(tmp_path, version=version),
        _write_sdist(tmp_path, version=version),
    ]

    assert verify_release_artifacts(release_tag, artifacts) == version


@pytest.mark.parametrize(
    "release_tag",
    [
        "0.1.0",
        "release-0.1.0",
        "v0.1",
        "v0.1.0.dev1",
        "v0.1.0+local",
    ],
)
def test_release_tag_rejects_non_release_formats(release_tag):
    with pytest.raises(ReleaseArtifactError, match="Release tags must use"):
        release_version_from_tag(release_tag)


@pytest.mark.parametrize("artifact_version", ["0.1.0.dev1", "0.1.0+local"])
def test_release_artifacts_reject_non_release_versions(tmp_path, artifact_version):
    artifacts = [
        _write_wheel(tmp_path, version=artifact_version),
        _write_sdist(tmp_path, version=artifact_version),
    ]

    with pytest.raises(ReleaseArtifactError, match="non-release version"):
        verify_release_artifacts("v0.1.0", artifacts)


def test_release_artifacts_reject_mismatched_version(tmp_path):
    artifacts = [
        _write_wheel(tmp_path, version="0.1.0"),
        _write_sdist(tmp_path, version="0.1.1"),
    ]

    with pytest.raises(ReleaseArtifactError, match="requires '0.1.0'"):
        verify_release_artifacts("v0.1.0", artifacts)


def test_release_artifacts_require_one_wheel_and_one_sdist(tmp_path):
    wheel = _write_wheel(tmp_path)

    with pytest.raises(ReleaseArtifactError, match="exactly one wheel"):
        verify_release_artifacts("v0.1.0", [wheel])


def test_release_artifacts_reject_other_distribution_name(tmp_path):
    artifacts = [
        _write_wheel(tmp_path, name="OtherProject"),
        _write_sdist(tmp_path, name="OtherProject"),
    ]

    with pytest.raises(ReleaseArtifactError, match="expected 'DeepMTP'"):
        verify_release_artifacts("v0.1.0", artifacts)


def test_release_artifact_cli_reports_success(tmp_path, capsys):
    artifacts = [
        _write_wheel(tmp_path),
        _write_sdist(tmp_path),
    ]

    assert main(["v0.1.0", *(str(path) for path in artifacts)]) == 0
    assert "Verified DeepMTP 0.1.0" in capsys.readouterr().out


def test_release_artifact_cli_reports_failure(tmp_path, capsys):
    artifacts = [
        _write_wheel(tmp_path, version="0.1.1"),
        _write_sdist(tmp_path, version="0.1.1"),
    ]

    assert main(["v0.1.0", *(str(path) for path in artifacts)]) == 1
    assert "verification failed" in capsys.readouterr().err
