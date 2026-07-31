import tarfile
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.verify_release_artifacts import (
    ReleaseArtifactError,
    main,
    release_version_from_tag,
    verify_distribution_contract,
    verify_release_artifacts,
)

_EXTRAS = (
    "all",
    "datasets",
    "foundation",
    "graph",
    "hpo",
    "image",
    "sparse",
    "streamlit",
    "tracking",
    "visualization",
)
_PROJECT_URLS = (
    ("Documentation", "https://deepmtp.readthedocs.io/"),
    ("Issues", "https://github.com/diliadis/DeepMTP/issues"),
    ("Repository", "https://github.com/diliadis/DeepMTP"),
)
_REQUIRED_WHEEL_FILES = {
    "DeepMTP/__init__.py",
    "DeepMTP/__main__.py",
    "DeepMTP/architecture.py",
    "DeepMTP/cli.py",
    "DeepMTP/config.py",
    "DeepMTP/data/__init__.py",
    "DeepMTP/hpo/__init__.py",
    "DeepMTP/integrations/huggingface.py",
    "DeepMTP/integrations/streamlit/__init__.py",
    "DeepMTP/main.py",
    "DeepMTP/models/__init__.py",
    "DeepMTP/runtime/__init__.py",
}
_REQUIRED_SDIST_FILES = {
    ".github/workflows/ci.yml",
    ".github/workflows/publish.yml",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "DEMO/main_example_MLC.ipynb",
    "LICENSE.md",
    "MIGRATION.rst",
    "README.md",
    "RELEASING.md",
    "SECURITY.md",
    "docs/source/tutorial/foundation_models.rst",
    "docs/source/tutorial/architecture_visualization.rst",
    "docs/source/index.rst",
    "pyproject.toml",
    "scripts/smoke_test_installation.py",
    "scripts/smoke_test_optional_extras.py",
    "scripts/verify_release_artifacts.py",
}


def _metadata(
    name: str,
    version: str,
    *,
    requires_python: str = ">=3.10",
    extras: tuple[str, ...] = _EXTRAS,
    project_urls: tuple[tuple[str, str], ...] = _PROJECT_URLS,
    license_expression: str = "MIT",
) -> bytes:
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        f"Requires-Python: {requires_python}",
        f"License-Expression: {license_expression}",
        *(f"Project-URL: {label}, {url}" for label, url in project_urls),
        *(f"Provides-Extra: {extra}" for extra in extras),
        "",
        "",
    ]
    return "\n".join(lines).encode()


def _write_wheel(
    directory: Path,
    *,
    name: str = "DeepMTP",
    version: str = "0.1.0",
    requires_python: str = ">=3.10",
    extras: tuple[str, ...] = _EXTRAS,
    project_urls: tuple[tuple[str, str], ...] = _PROJECT_URLS,
    omit_file: str | None = None,
    include_tests: bool = False,
) -> Path:
    path = directory / f"deepmtp-{version}-py3-none-any.whl"
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        wheel_files = _REQUIRED_WHEEL_FILES - {omit_file}
        for wheel_file in sorted(wheel_files):
            archive.writestr(wheel_file, b"")
        if include_tests:
            archive.writestr("DeepMTP/tests/test_packaging.py", b"")
        archive.writestr(
            f"deepmtp-{version}.dist-info/METADATA",
            _metadata(
                name,
                version,
                requires_python=requires_python,
                extras=extras,
                project_urls=project_urls,
            ),
        )
    return path


def _write_sdist(
    directory: Path,
    *,
    name: str = "DeepMTP",
    version: str = "0.1.0",
    requires_python: str = ">=3.10",
    extras: tuple[str, ...] = _EXTRAS,
    project_urls: tuple[tuple[str, str], ...] = _PROJECT_URLS,
    omit_file: str | None = None,
) -> Path:
    path = directory / f"deepmtp-{version}.tar.gz"
    metadata = _metadata(
        name,
        version,
        requires_python=requires_python,
        extras=extras,
        project_urls=project_urls,
    )
    metadata_info = tarfile.TarInfo(f"deepmtp-{version}/PKG-INFO")
    metadata_info.size = len(metadata)
    egg_info = tarfile.TarInfo(f"deepmtp-{version}/DeepMTP.egg-info/PKG-INFO")
    egg_info.size = len(metadata)
    with tarfile.open(path, mode="w:gz") as archive:
        archive.addfile(metadata_info, BytesIO(metadata))
        archive.addfile(egg_info, BytesIO(metadata))
        for project_file in sorted(_REQUIRED_SDIST_FILES - {omit_file}):
            file_info = tarfile.TarInfo(f"deepmtp-{version}/{project_file}")
            file_info.size = 0
            archive.addfile(file_info, BytesIO())
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


def test_distribution_contract_accepts_development_versions(tmp_path):
    artifacts = [
        _write_wheel(tmp_path, version="0.1.1.dev3+gabc123"),
        _write_sdist(tmp_path, version="0.1.1.dev3+gabc123"),
    ]

    assert verify_distribution_contract(artifacts) == "0.1.1.dev3+gabc123"


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


def test_distribution_contract_rejects_mismatched_versions(tmp_path):
    artifacts = [
        _write_wheel(tmp_path, version="0.1.0"),
        _write_sdist(tmp_path, version="0.1.1"),
    ]

    with pytest.raises(ReleaseArtifactError, match="same version"):
        verify_release_artifacts("v0.1.0", artifacts)


def test_release_artifacts_reject_version_that_differs_from_tag(tmp_path):
    artifacts = [
        _write_wheel(tmp_path, version="0.1.1"),
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


@pytest.mark.parametrize(
    ("artifact_kind", "metadata_override", "message"),
    [
        ("wheel", {"requires_python": ">=3.11"}, "Requires-Python"),
        ("wheel", {"extras": ("all", "datasets")}, "extras do not match"),
        ("sdist", {"project_urls": ()}, "project URLs"),
    ],
)
def test_distribution_contract_rejects_incomplete_metadata(
    tmp_path,
    artifact_kind,
    metadata_override,
    message,
):
    wheel_options = metadata_override if artifact_kind == "wheel" else {}
    sdist_options = metadata_override if artifact_kind == "sdist" else {}
    artifacts = [
        _write_wheel(tmp_path, **wheel_options),
        _write_sdist(tmp_path, **sdist_options),
    ]

    with pytest.raises(ReleaseArtifactError, match=message):
        verify_distribution_contract(artifacts)


def test_distribution_contract_rejects_missing_runtime_file(tmp_path):
    artifacts = [
        _write_wheel(tmp_path, omit_file="DeepMTP/config.py"),
        _write_sdist(tmp_path),
    ]

    with pytest.raises(ReleaseArtifactError, match="DeepMTP/config.py"):
        verify_distribution_contract(artifacts)


def test_distribution_contract_rejects_tests_in_wheel(tmp_path):
    artifacts = [
        _write_wheel(tmp_path, include_tests=True),
        _write_sdist(tmp_path),
    ]

    with pytest.raises(ReleaseArtifactError, match="must not contain DeepMTP/tests"):
        verify_distribution_contract(artifacts)


def test_distribution_contract_rejects_missing_sdist_policy(tmp_path):
    artifacts = [
        _write_wheel(tmp_path),
        _write_sdist(tmp_path, omit_file="MIGRATION.rst"),
    ]

    with pytest.raises(ReleaseArtifactError, match="MIGRATION.rst"):
        verify_distribution_contract(artifacts)


def test_release_artifact_cli_reports_success(tmp_path, capsys):
    artifacts = [
        _write_wheel(tmp_path),
        _write_sdist(tmp_path),
    ]

    assert main(["v0.1.0", *(str(path) for path in artifacts)]) == 0
    assert "Verified DeepMTP 0.1.0" in capsys.readouterr().out


def test_distribution_contract_cli_reports_success(tmp_path, capsys):
    artifacts = [
        _write_wheel(tmp_path, version="0.1.1.dev3+gabc123"),
        _write_sdist(tmp_path, version="0.1.1.dev3+gabc123"),
    ]

    assert main(["--contract-only", *(str(path) for path in artifacts)]) == 0
    assert "distribution contract" in capsys.readouterr().out


def test_release_artifact_cli_reports_failure(tmp_path, capsys):
    artifacts = [
        _write_wheel(tmp_path, version="0.1.1"),
        _write_sdist(tmp_path, version="0.1.1"),
    ]

    assert main(["v0.1.0", *(str(path) for path in artifacts)]) == 1
    assert "verification failed" in capsys.readouterr().err
