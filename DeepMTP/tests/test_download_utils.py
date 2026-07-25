import io
import zipfile

import pytest
import requests

from DeepMTP import _download_utils
from DeepMTP._download_utils import DatasetDownloadError, download_and_extract_zip


class StubResponse:
    def __init__(self, content, error=None):
        self.content = content
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error


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
