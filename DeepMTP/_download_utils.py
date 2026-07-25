"""Internal helpers for downloading and extracting external dataset archives."""

from __future__ import annotations

import io
import os
import stat
import zipfile
from pathlib import Path

import requests

DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30.0


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
            _validate_archive_members(archive, destination_path)
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise DatasetDownloadError(
                    f"Dataset archive contains a corrupt file: {corrupt_member!r}."
                )
            archive.extractall(destination_path)
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError(
            f"Downloaded content from {url!r} is not a valid ZIP archive."
        ) from exc
