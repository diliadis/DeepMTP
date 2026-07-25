"""Internal helpers for downloading and extracting external dataset archives."""

from __future__ import annotations

import io
import os
import stat
import tempfile
import zipfile
from pathlib import Path

import requests

DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30.0
DOWNLOAD_CHUNK_SIZE_BYTES = 64 * 1024


class DatasetDownloadError(RuntimeError):
    """Raised when an external dataset archive cannot be downloaded or extracted."""


def _validate_archive_members(
    archive: zipfile.ZipFile,
    destination: Path,
) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        member_path = (destination / member.filename).resolve()
        try:
            member_path.relative_to(destination)
        except ValueError as exc:
            raise DatasetDownloadError(
                f"Dataset archive contains an unsafe path: {member.filename!r}."
            ) from exc

        member_mode = member.external_attr >> 16
        if stat.S_ISLNK(member_mode):
            raise DatasetDownloadError(
                f"Dataset archive contains an unsupported symbolic link: "
                f"{member.filename!r}."
            )


def _extract_archive(
    archive: zipfile.ZipFile,
    destination: Path,
) -> None:
    _validate_archive_members(archive, destination)
    corrupt_member = archive.testzip()
    if corrupt_member is not None:
        raise DatasetDownloadError(
            f"Dataset archive contains a corrupt file: {corrupt_member!r}."
        )
    archive.extractall(destination)


def download_file(
    url: str,
    destination: str | os.PathLike[str],
    *,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> None:
    """Stream a URL to a file without exposing a partial destination."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination_path.parent,
                prefix=f".{destination_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                for chunk in response.iter_content(
                    chunk_size=DOWNLOAD_CHUNK_SIZE_BYTES
                ):
                    if chunk:
                        temporary_file.write(chunk)
        temporary_path.replace(destination_path)
    except requests.RequestException as exc:
        raise DatasetDownloadError(f"Failed to download file from {url!r}.") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def extract_zip_file(
    archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> None:
    """Extract a local ZIP archive after validating all of its members."""
    archive_file = Path(archive_path)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_file) as archive:
            _extract_archive(archive, destination_path)
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError(
            f"Dataset archive is not a valid ZIP file: {archive_file}."
        ) from exc


def download_and_extract_zip(
    url: str,
    destination: str | os.PathLike[str],
    *,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> None:
    """Download a ZIP archive and extract it after validating its contents."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DatasetDownloadError(
            f"Failed to download dataset archive from {url!r}."
        ) from exc

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            _extract_archive(archive, destination_path)
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError(
            f"Downloaded content from {url!r} is not a valid ZIP archive."
        ) from exc
