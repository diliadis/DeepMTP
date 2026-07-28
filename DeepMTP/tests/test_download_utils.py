import hashlib
import io
import zipfile

import pytest
import requests

from DeepMTP import _download_utils
from DeepMTP._download_utils import (
    DatasetDownloadError,
    download_and_extract_zip,
    download_file,
    extract_zip_file,
)


class StubResponse:
    def __init__(self, content, error=None, stream_error=None):
        self.content = content
        self.error = error
        self.stream_error = stream_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def iter_content(self, chunk_size):
        del chunk_size
        midpoint = len(self.content) // 2
        yield self.content[:midpoint]
        if self.stream_error is not None:
            raise self.stream_error
        yield b""
        yield self.content[midpoint:]


def make_zip_payload(files):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)
    return payload.getvalue()


def test_download_and_extract_zip_uses_timeout_and_extracts(monkeypatch, tmp_path):
    calls = []
    payload = make_zip_payload({"dataset/data.txt": "values"})

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return StubResponse(payload)

    monkeypatch.setattr(_download_utils.requests, "get", get)

    download_and_extract_zip(
        "https://example.test/dataset.zip",
        tmp_path,
        timeout=7.5,
    )

    assert calls == [("https://example.test/dataset.zip", {"timeout": 7.5})]
    assert (tmp_path / "dataset" / "data.txt").read_text() == "values"


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout("timed out"),
        requests.HTTPError("bad status"),
    ],
)
def test_download_and_extract_zip_wraps_request_errors(monkeypatch, tmp_path, error):
    def get(url, **kwargs):
        if isinstance(error, requests.Timeout):
            raise error
        return StubResponse(b"", error=error)

    monkeypatch.setattr(_download_utils.requests, "get", get)

    with pytest.raises(DatasetDownloadError, match="Failed to download"):
        download_and_extract_zip("https://example.test/dataset.zip", tmp_path)


def test_download_and_extract_zip_rejects_invalid_archives(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _download_utils.requests,
        "get",
        lambda url, **kwargs: StubResponse(b"not a zip"),
    )

    with pytest.raises(DatasetDownloadError, match="not a valid ZIP"):
        download_and_extract_zip("https://example.test/dataset.zip", tmp_path)


def test_download_and_extract_zip_rejects_path_traversal(monkeypatch, tmp_path):
    payload = make_zip_payload({"../escaped.txt": "unsafe"})
    monkeypatch.setattr(
        _download_utils.requests,
        "get",
        lambda url, **kwargs: StubResponse(payload),
    )

    with pytest.raises(DatasetDownloadError, match="unsafe path"):
        download_and_extract_zip("https://example.test/dataset.zip", tmp_path / "data")

    assert not (tmp_path / "escaped.txt").exists()


def test_download_file_streams_and_atomically_replaces(monkeypatch, tmp_path):
    destination = tmp_path / "dataset.zip"
    destination.write_bytes(b"old content")
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return StubResponse(b"new content")

    monkeypatch.setattr(_download_utils.requests, "get", get)

    download_file(
        "https://example.test/dataset.zip",
        destination,
        timeout=7.5,
        expected_md5=hashlib.md5(
            b"new content",
            usedforsecurity=False,
        )
        .hexdigest()
        .upper(),
    )

    assert calls == [
        (
            "https://example.test/dataset.zip",
            {"stream": True, "timeout": 7.5},
        )
    ]
    assert destination.read_bytes() == b"new content"
    assert list(tmp_path.glob(".dataset.zip.*.tmp")) == []


def test_download_file_preserves_destination_on_stream_failure(
    monkeypatch,
    tmp_path,
):
    destination = tmp_path / "dataset.zip"
    destination.write_bytes(b"old content")
    monkeypatch.setattr(
        _download_utils.requests,
        "get",
        lambda url, **kwargs: StubResponse(
            b"partial content",
            stream_error=requests.ConnectionError("connection lost"),
        ),
    )

    with pytest.raises(DatasetDownloadError, match="Failed to download"):
        download_file("https://example.test/dataset.zip", destination)

    assert destination.read_bytes() == b"old content"
    assert list(tmp_path.glob(".dataset.zip.*.tmp")) == []


def test_download_file_validates_checksum_before_replacing(monkeypatch, tmp_path):
    destination = tmp_path / "dataset.zip"
    destination.write_bytes(b"old content")
    monkeypatch.setattr(
        _download_utils.requests,
        "get",
        lambda url, **kwargs: StubResponse(b"new content"),
    )

    with pytest.raises(DatasetDownloadError, match="checksum"):
        download_file(
            "https://example.test/dataset.zip",
            destination,
            expected_md5=hashlib.md5(
                b"different content",
                usedforsecurity=False,
            ).hexdigest(),
        )

    assert destination.read_bytes() == b"old content"
    assert list(tmp_path.glob(".dataset.zip.*.tmp")) == []


def test_download_file_rejects_invalid_checksum_before_request(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        _download_utils.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("invalid checksum must fail first"),
    )

    with pytest.raises(ValueError, match="expected_md5"):
        download_file(
            "https://example.test/dataset.zip",
            tmp_path / "dataset.zip",
            expected_md5="invalid",
        )


def test_extract_zip_file_extracts_valid_archive(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    archive_path.write_bytes(make_zip_payload({"dataset/data.txt": "values"}))

    extract_zip_file(archive_path, tmp_path / "extracted")

    assert (tmp_path / "extracted" / "dataset" / "data.txt").read_text() == "values"


def test_extract_zip_file_rejects_unsafe_cached_archive(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    archive_path.write_bytes(make_zip_payload({"../escaped.txt": "unsafe"}))

    with pytest.raises(DatasetDownloadError, match="unsafe path"):
        extract_zip_file(archive_path, tmp_path / "extracted")

    assert not (tmp_path / "escaped.txt").exists()
