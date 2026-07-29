import ast
import hashlib
import inspect
import json
import pickle
import zipfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from skmultilearn.dataset import save_dataset_dump

import DeepMTP.data.datasets as dataset_module
from DeepMTP import DataProgressEvent, NullDataProgressObserver
from DeepMTP._download_utils import DatasetDownloadError
from DeepMTP.data.datasets import (
    format_mtr_datasets,
    generate_dummy_dataset,
    generate_interaction_matrix,
    generate_MTP_dataset,
    load_process_DP,
    load_process_MC,
    load_process_MLC,
    load_process_MTL,
    load_process_MTR,
    print_MTR_datasets,
    process_dummy_DP,
    process_dummy_MLC,
    process_dummy_MTR,
)


class RecordingDataProgress:
    def __init__(self) -> None:
        self.events: list[DataProgressEvent] = []

    def __bool__(self) -> bool:
        return False

    def on_event(self, event: DataProgressEvent) -> None:
        self.events.append(event)


def write_mtr_dataset(root, content=None):
    dataset_dir = root / "mtr-datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "enb.arff"
    dataset_path.write_text(
        content
        or """
@RELATION enb
@ATTRIBUTE feature_1 NUMERIC
@ATTRIBUTE feature_2 NUMERIC
@ATTRIBUTE target_1 NUMERIC
@ATTRIBUTE target_2 NUMERIC
@DATA
1,2,3,4
5,6,7,8
""".strip(),
        encoding="utf-8",
    )
    return dataset_path


def write_mtl_dataset(
    root,
    *,
    dataset_name="dog",
    flatten_worker_labels=True,
    include_images=True,
):
    dataset_dir = root / "multi-task_learning-datasets" / dataset_name
    image_dir = dataset_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    worker_labels = [
        [1, 2, -1],
        [0, 1, 0],
        [2, -1, 1],
        [0, 1, 1],
    ]
    if flatten_worker_labels:
        worker_labels = [label for row in worker_labels for label in row]
    (dataset_dir / "data.json").write_text(
        json.dumps(
            {
                "true_labels": [1, 0, 2, 1],
                "WorkerLabels": worker_labels,
            }
        ),
        encoding="utf-8",
    )
    if include_images:
        for index in range(1, 5):
            (image_dir / f"{index}.jpg").touch()
    return dataset_dir


def write_mlc_dump(
    root,
    *,
    dataset_name="emotions",
    variant="undivided",
    X=None,
    y=None,
    sparse_matrices=True,
):
    X = np.arange(24, dtype=float).reshape(8, 3) if X is None else X
    y = (
        np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0],
            ]
        )
        if y is None
        else y
    )
    dataset_dir = root / "multi-label_classification"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / f"{dataset_name}-{variant}.bz2"
    saved_features = sparse.csr_matrix(X) if sparse_matrices else X
    saved_labels = sparse.csr_matrix(y) if sparse_matrices else y
    save_dataset_dump(
        saved_features,
        saved_labels,
        [f"feature_{index}" for index in range(np.asarray(X).shape[1])],
        [f"label_{index}" for index in range(np.asarray(y).shape[1])],
        filename=str(dataset_path),
    )
    return dataset_path


def prepare_cached_dataset(loader_name, root):
    if loader_name == "mlc":
        write_mlc_dump(root)
        return load_process_MLC, {"dataset_name": "emotions"}, "emotions"
    if loader_name == "mtr":
        write_mtr_dataset(root)
        return load_process_MTR, {"dataset_name": "enb"}, "enb"
    write_dp_dataset(root)
    return load_process_DP, {"dataset_name": "ern"}, "ern"


@pytest.mark.parametrize("loader_name", ["mlc", "mtr", "dp"])
def test_cached_dataset_loaders_preserve_developer_console_output(
    loader_name,
    tmp_path,
    capsys,
):
    loader, arguments, _ = prepare_cached_dataset(loader_name, tmp_path)

    loader(
        tmp_path,
        print_mode="dev",
        **arguments,
    )

    assert capsys.readouterr().out == "info: Processing...\ninfo: Done\n"


@pytest.mark.parametrize("loader_name", ["mlc", "mtr", "dp"])
def test_cached_dataset_loaders_emit_structured_progress(
    loader_name,
    tmp_path,
    capsys,
):
    loader, arguments, dataset_name = prepare_cached_dataset(loader_name, tmp_path)
    progress = RecordingDataProgress()

    loader(
        tmp_path,
        print_mode="dev",
        progress=progress,
        **arguments,
    )

    subject = f"{loader_name}_dataset_loading"
    assert capsys.readouterr().out == ""
    assert [
        (event.message, event.kind, event.subject, event.value, event.end)
        for event in progress.events
    ] == [
        (
            "Processing...",
            "operation_started",
            subject,
            dataset_name,
            "\n",
        ),
        (
            "Done",
            "operation_completed",
            subject,
            dataset_name,
            "\n",
        ),
    ]


@pytest.mark.parametrize("loader_name", ["mlc", "mtr", "dp"])
def test_cached_dataset_loaders_respect_null_progress(
    loader_name,
    tmp_path,
    capsys,
):
    loader, arguments, _ = prepare_cached_dataset(loader_name, tmp_path)

    loader(
        tmp_path,
        print_mode="dev",
        progress=NullDataProgressObserver(),
        **arguments,
    )

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "target",
    [
        dataset_module.load_process_MLC,
        dataset_module.load_process_MTR,
        dataset_module.load_process_DP,
        dataset_module._download_movielens_archive,
        dataset_module.load_process_MC,
        dataset_module.load_process_MTL,
        dataset_module.print_MTR_datasets,
    ],
)
def test_dataset_progress_functions_do_not_print_directly(target):
    tree = ast.parse(inspect.getsource(target))

    direct_prints = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

    assert direct_prints == []


def test_mtr_catalog_preserves_console_output(capsys):
    catalog = format_mtr_datasets()

    result = print_MTR_datasets()

    assert result is None
    assert capsys.readouterr().out == catalog + "\n"
    assert catalog.count("\n") == 21
    assert "| atp1d  |    337     |        411         |    6     |" in catalog
    assert "|  scpf  |    1137    |         23         |    3     |" in catalog


def test_mtr_catalog_emits_structured_progress(capsys):
    progress = RecordingDataProgress()

    print_MTR_datasets(progress=progress)

    assert capsys.readouterr().out == ""
    assert len(progress.events) == 1
    event = progress.events[0]
    assert (
        event.kind,
        event.subject,
        event.value,
        event.message,
    ) == (
        "message",
        "mtr_dataset_catalog",
        (
            "atp1d",
            "atp7d",
            "oes97",
            "oes10",
            "rf1",
            "rf2",
            "scm1d",
            "scm20d",
            "edm",
            "sf1",
            "sf2",
            "jura",
            "wq",
            "enb",
            "slump",
            "andro",
            "osales",
            "scpf",
        ),
        format_mtr_datasets(),
    )


def test_mtr_catalog_respects_null_progress(capsys):
    print_MTR_datasets(progress=NullDataProgressObserver())

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("features_type", ["numpy", "dataframe"])
def test_load_process_mlc_reads_cached_dump_without_network(
    monkeypatch,
    tmp_path,
    features_type,
):
    write_mlc_dump(tmp_path)
    monkeypatch.setattr(
        dataset_module,
        "download_file",
        lambda *args, **kwargs: pytest.fail("complete cache should not be downloaded"),
    )

    data = load_process_MLC(
        tmp_path,
        dataset_name="emotions",
        features_type=features_type,
    )

    np.testing.assert_array_equal(
        data["train"]["y"],
        np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0],
            ]
        ),
    )
    if features_type == "numpy":
        assert data["train"]["X_instance"].shape == (8, 3)
    else:
        assert list(data["train"]["X_instance"]["id"]) == list(range(8))
        assert all(
            feature.shape == (3,) for feature in data["train"]["X_instance"]["features"]
        )
    assert data["train"]["X_target"] is None
    assert data["val"] == {"y": None, "X_instance": None, "X_target": None}
    assert data["test"] == {"y": None, "X_instance": None, "X_target": None}


def test_load_process_mlc_reads_cached_published_split(tmp_path):
    train_X = np.arange(18, dtype=float).reshape(6, 3)
    train_y = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
    test_X = np.arange(12, dtype=float).reshape(4, 3)
    test_y = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    write_mlc_dump(tmp_path, variant="train", X=train_X, y=train_y)
    write_mlc_dump(tmp_path, variant="test", X=test_X, y=test_y)

    data = load_process_MLC(
        tmp_path,
        dataset_name="emotions",
        variant="divided",
    )

    np.testing.assert_array_equal(data["train"]["X_instance"], train_X)
    np.testing.assert_array_equal(data["train"]["y"], train_y)
    np.testing.assert_array_equal(data["test"]["X_instance"], test_X)
    np.testing.assert_array_equal(data["test"]["y"], test_y)


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"dataset_name": "unknown"}, ValueError, "dataset_name"),
        ({"variant": "unknown"}, ValueError, "variant"),
        ({"variant": []}, ValueError, "variant"),
        ({"features_type": "unknown"}, ValueError, "features_type"),
        ({"print_mode": "unknown"}, ValueError, "print_mode"),
        ({"print_mode": []}, ValueError, "print_mode"),
        ({"path": object()}, TypeError, "path"),
    ],
)
def test_load_process_mlc_validates_arguments_before_download(
    monkeypatch,
    tmp_path,
    overrides,
    exception,
    message,
):
    monkeypatch.setattr(
        dataset_module,
        "download_file",
        lambda *args, **kwargs: pytest.fail(
            "invalid arguments must not trigger a download"
        ),
    )
    arguments = {"path": tmp_path, "dataset_name": "emotions"}
    arguments.update(overrides)

    with pytest.raises(exception, match=message):
        load_process_MLC(**arguments)


def test_load_process_mlc_downloads_missing_dump_from_manifest(
    monkeypatch,
    tmp_path,
    capsys,
):
    source_path = write_mlc_dump(tmp_path / "source")
    payload = source_path.read_bytes()
    checksum = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    calls = []

    def download(url, destination, **kwargs):
        calls.append((url, destination, kwargs))
        if url.endswith("data.list"):
            destination.write_text(
                f"{checksum};emotions-undivided.bz2\n",
                encoding="utf-8",
            )
        else:
            destination.write_bytes(payload)

    monkeypatch.setattr(dataset_module, "download_file", download)
    progress = RecordingDataProgress()

    data = load_process_MLC(
        tmp_path,
        dataset_name="emotions",
        print_mode="dev",
        progress=progress,
    )

    dataset_dir = tmp_path / "multi-label_classification"
    assert calls == [
        (
            "https://scikit.ml/datasets/data.list",
            dataset_dir / "data.list",
            {},
        ),
        (
            "https://scikit.ml/datasets/emotions-undivided.bz2",
            dataset_dir / "emotions-undivided.bz2",
            {"expected_md5": checksum},
        ),
    ]
    assert data["train"]["y"].shape == (8, 2)
    assert capsys.readouterr().out == ""
    assert [
        (event.message, event.kind, event.subject, event.value)
        for event in progress.events
    ] == [
        (
            "Processing...",
            "operation_started",
            "mlc_dataset_loading",
            "emotions",
        ),
        (
            "Downloaded missing dataset files",
            "message",
            "mlc_dataset_download",
            "emotions",
        ),
        ("Done", "operation_completed", "mlc_dataset_loading", "emotions"),
    ]


def test_load_process_mlc_downloads_only_missing_split_dump(
    monkeypatch,
    tmp_path,
):
    train_X = np.arange(18, dtype=float).reshape(6, 3)
    train_y = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
    write_mlc_dump(tmp_path, variant="train", X=train_X, y=train_y)
    source_path = write_mlc_dump(
        tmp_path / "source",
        variant="test",
        X=np.arange(12, dtype=float).reshape(4, 3),
        y=np.array([[1, 1], [0, 0], [1, 0], [0, 1]]),
    )
    payload = source_path.read_bytes()
    checksum = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    dataset_dir = tmp_path / "multi-label_classification"
    (dataset_dir / "data.list").write_text(
        f"{checksum};emotions-test.bz2\n",
        encoding="utf-8",
    )
    calls = []

    def download(url, destination, **kwargs):
        calls.append((url, destination, kwargs))
        destination.write_bytes(payload)

    monkeypatch.setattr(dataset_module, "download_file", download)

    data = load_process_MLC(
        tmp_path,
        dataset_name="emotions",
        variant="divided",
    )

    assert calls == [
        (
            "https://scikit.ml/datasets/emotions-test.bz2",
            dataset_dir / "emotions-test.bz2",
            {"expected_md5": checksum},
        )
    ]
    assert data["test"]["y"].shape == (4, 2)


def test_load_process_mlc_reports_unavailable_upstream_catalog(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        dataset_module,
        "download_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            DatasetDownloadError("upstream unavailable")
        ),
    )

    with pytest.raises(DatasetDownloadError, match="not cached.*upstream"):
        load_process_MLC(tmp_path, dataset_name="emotions")


def test_load_process_mlc_rejects_directory_at_cache_file_path(
    monkeypatch,
    tmp_path,
):
    cache_path = tmp_path / "multi-label_classification" / "emotions-undivided.bz2"
    cache_path.mkdir(parents=True)
    monkeypatch.setattr(
        dataset_module,
        "download_file",
        lambda *args, **kwargs: pytest.fail("invalid cache must not be downloaded"),
    )

    with pytest.raises(IsADirectoryError, match="cache paths"):
        load_process_MLC(tmp_path, dataset_name="emotions")


def test_load_process_mlc_reports_corrupt_cached_dump(tmp_path):
    dataset_dir = tmp_path / "multi-label_classification"
    dataset_dir.mkdir()
    (dataset_dir / "emotions-undivided.bz2").write_bytes(b"not a dataset dump")

    with pytest.raises(ValueError, match="cache could not be loaded"):
        load_process_MLC(tmp_path, dataset_name="emotions")


@pytest.mark.parametrize(
    "row",
    [
        "invalid;emotions-undivided.bz2",
        "d41d8cd98f00b204e9800998ecf8427e;../emotions-undivided.bz2",
        "d41d8cd98f00b204e9800998ecf8427e;..\\emotions-undivided.bz2",
        "d41d8cd98f00b204e9800998ecf8427e;emotions-unknown.bz2",
    ],
)
def test_parse_mlc_manifest_rejects_invalid_rows(tmp_path, row):
    manifest_path = tmp_path / "data.list"
    manifest_path.write_text(f"{row}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        dataset_module._parse_mlc_manifest(manifest_path)


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (
            np.array([[1.0], [np.inf]]),
            np.array([[1], [0]]),
            "finite values",
        ),
        (
            np.array([[1.0], [2.0]]),
            np.array([[1], [2]]),
            "binary values",
        ),
        (
            np.array([[1.0], [2.0]]),
            np.array([[1]]),
            "same number of rows",
        ),
        (
            np.empty((0, 1)),
            np.empty((0, 1)),
            "non-empty two-dimensional",
        ),
    ],
)
def test_load_process_mlc_rejects_invalid_cached_matrices(
    tmp_path,
    X,
    y,
    message,
):
    write_mlc_dump(
        tmp_path,
        X=X,
        y=y,
        sparse_matrices=False,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        load_process_MLC(tmp_path, dataset_name="emotions")


@pytest.mark.parametrize(
    ("test_X", "test_y", "message"),
    [
        (
            np.arange(16, dtype=float).reshape(4, 4),
            np.array([[1, 0], [0, 1], [1, 1], [0, 0]]),
            "instance features.*column count",
        ),
        (
            np.arange(12, dtype=float).reshape(4, 3),
            np.array([[1], [0], [1], [0]]),
            "labels.*column count",
        ),
    ],
)
def test_load_process_mlc_rejects_inconsistent_published_split(
    tmp_path,
    test_X,
    test_y,
    message,
):
    write_mlc_dump(
        tmp_path,
        variant="train",
        X=np.arange(18, dtype=float).reshape(6, 3),
        y=np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]]),
    )
    write_mlc_dump(
        tmp_path,
        variant="test",
        X=test_X,
        y=test_y,
    )

    with pytest.raises(ValueError, match=message):
        load_process_MLC(
            tmp_path,
            dataset_name="emotions",
            variant="divided",
        )


def write_dp_dataset(
    root,
    *,
    dataset_name="ern",
    y=None,
    X_instance=None,
    X_target=None,
):
    y = np.arange(96, dtype=float).reshape(12, 8) if y is None else y
    X_instance = (
        np.arange(36, dtype=float).reshape(12, 3) if X_instance is None else X_instance
    )
    X_target = (
        np.arange(16, dtype=float).reshape(8, 2) if X_target is None else X_target
    )
    dataset_dir = root / "dyadic_prediction-datasets" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = f"{dataset_name}_" if dataset_name.startswith("dp") else ""
    delimiter = "," if dataset_name in {"ern", "srn"} else " "
    paths = {
        "Y": dataset_dir / f"{filename_prefix}Y.txt",
        "X1": dataset_dir / f"{filename_prefix}X1.txt",
        "X2": dataset_dir / f"{filename_prefix}X2.txt",
    }
    for stem, matrix in {
        "Y": y,
        "X1": X_instance,
        "X2": X_target,
    }.items():
        np.savetxt(paths[stem], matrix, delimiter=delimiter)
    return paths


@pytest.mark.parametrize("dataset_name", ["ern", "dpii"])
def test_load_process_dp_reads_cached_files_without_network(
    monkeypatch,
    tmp_path,
    dataset_name,
):
    paths = write_dp_dataset(tmp_path, dataset_name=dataset_name)

    monkeypatch.setattr(
        dataset_module,
        "download_file",
        lambda *args, **kwargs: pytest.fail("complete cache should not be downloaded"),
    )

    data = load_process_DP(tmp_path, dataset_name=dataset_name)

    delimiter = "," if dataset_name == "ern" else None
    np.testing.assert_array_equal(
        data["train"]["y"],
        np.loadtxt(paths["Y"], delimiter=delimiter, ndmin=2),
    )
    assert data["train"]["X_instance"].shape == (12, 3)
    assert data["train"]["X_target"].shape == (8, 2)
    assert data["val"] == {"y": None, "X_instance": None, "X_target": None}
    assert data["test"] == {"y": None, "X_instance": None, "X_target": None}


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"dataset_name": "unknown"}, ValueError, "dataset_name"),
        ({"variant": "unknown"}, ValueError, "variant"),
        ({"validation_setting": "A"}, ValueError, "validation_setting"),
        ({"validation_setting": 1}, ValueError, "validation_setting"),
        ({"split_instance_features": 1}, TypeError, "split_instance_features"),
        ({"split_target_features": None}, TypeError, "split_target_features"),
        ({"print_mode": "unknown"}, ValueError, "print_mode"),
        ({"print_mode": []}, ValueError, "print_mode"),
        ({"random_state": True}, ValueError, "random_state"),
        ({"split_ratio": {"test": 1.0}}, ValueError, "split ratio"),
        ({"path": object()}, TypeError, "path"),
    ],
)
def test_load_process_dp_validates_arguments_before_download(
    monkeypatch,
    tmp_path,
    overrides,
    exception,
    message,
):
    monkeypatch.setattr(
        dataset_module,
        "download_file",
        lambda *args, **kwargs: pytest.fail(
            "invalid arguments must not trigger a download"
        ),
    )
    arguments = {"path": tmp_path, "dataset_name": "ern"}
    arguments.update(overrides)

    with pytest.raises(exception, match=message):
        load_process_DP(**arguments)


def test_load_process_dp_downloads_only_missing_cache_files(
    monkeypatch,
    tmp_path,
    capsys,
):
    paths = write_dp_dataset(tmp_path)
    paths["X2"].unlink()
    calls = []
    target_features = np.arange(16, dtype=float).reshape(8, 2)

    def download(url, destination):
        calls.append((url, destination))
        np.savetxt(destination, target_features, delimiter=",")

    monkeypatch.setattr(dataset_module, "download_file", download)
    progress = RecordingDataProgress()

    data = load_process_DP(
        tmp_path,
        print_mode="dev",
        progress=progress,
    )

    assert calls == [
        (
            "https://people.montefiore.uliege.be/schrynemackers/ern/X2.txt",
            paths["X2"],
        )
    ]
    np.testing.assert_array_equal(data["train"]["X_target"], target_features)
    assert capsys.readouterr().out == ""
    assert [
        (event.kind, event.subject, event.value, event.end) for event in progress.events
    ] == [
        ("operation_started", "dp_dataset_loading", "ern", "\n"),
        ("operation_started", "dp_dataset_download", "ern", ""),
        ("operation_completed", "dp_dataset_download", "ern", "\n"),
        ("operation_completed", "dp_dataset_loading", "ern", "\n"),
    ]


def test_load_process_dp_reports_missing_downloaded_files(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dataset_module,
        "download_file",
        lambda url, destination: None,
    )

    with pytest.raises(FileNotFoundError, match="expected files"):
        load_process_DP(tmp_path)


def test_load_process_dp_rejects_directory_at_cache_file_path(
    monkeypatch,
    tmp_path,
):
    paths = write_dp_dataset(tmp_path)
    paths["X2"].unlink()
    paths["X2"].mkdir()
    monkeypatch.setattr(
        dataset_module,
        "download_file",
        lambda *args, **kwargs: pytest.fail("invalid cache must not be downloaded"),
    )

    with pytest.raises(IsADirectoryError, match="cache paths"):
        load_process_DP(tmp_path)


@pytest.mark.parametrize(
    ("matrix_name", "matrix", "message"),
    [
        ("Y", np.arange(88, dtype=float).reshape(11, 8), "instance features"),
        ("X2", np.arange(14, dtype=float).reshape(7, 2), "target features"),
    ],
)
def test_load_process_dp_rejects_misaligned_matrices(
    tmp_path,
    matrix_name,
    matrix,
    message,
):
    paths = write_dp_dataset(tmp_path)
    np.savetxt(paths[matrix_name], matrix, delimiter=",")

    with pytest.raises(ValueError, match=message):
        load_process_DP(tmp_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not,numeric\n", "numeric matrix"),
        ("1,inf\n", "finite numeric"),
        ("", "non-empty"),
    ],
)
def test_load_process_dp_rejects_invalid_matrix_values(tmp_path, content, message):
    paths = write_dp_dataset(tmp_path)
    paths["X1"].write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_process_DP(tmp_path)


@pytest.mark.parametrize(
    ("validation_setting", "split_instance_features", "split_target_features"),
    [
        ("B", True, False),
        ("C", False, True),
        ("D", True, True),
    ],
)
def test_load_process_dp_aligns_split_features(
    tmp_path,
    validation_setting,
    split_instance_features,
    split_target_features,
):
    write_dp_dataset(tmp_path)

    data = load_process_DP(
        tmp_path,
        variant="divided",
        split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
        split_instance_features=split_instance_features,
        split_target_features=split_target_features,
        validation_setting=validation_setting,
    )

    for split_name in ("train", "val", "test"):
        interactions = data[split_name]["y"]
        assert interactions is not None
        if split_instance_features:
            assert data[split_name]["X_instance"].shape[0] == interactions.shape[0]
        if split_target_features:
            assert data[split_name]["X_target"].shape[0] == interactions.shape[1]


def test_load_process_dp_keeps_unsplit_features_in_training_bundle(tmp_path):
    write_dp_dataset(tmp_path)

    data = load_process_DP(
        tmp_path,
        variant="divided",
        split_ratio={"train": 0.6, "val": 0.2, "test": 0.2},
        validation_setting="D",
    )

    assert data["train"]["X_instance"].shape == (12, 3)
    assert data["train"]["X_target"].shape == (8, 2)
    assert data["val"]["X_instance"] is None
    assert data["val"]["X_target"] is None
    assert data["test"]["X_instance"] is None
    assert data["test"]["X_target"] is None
    assert sum(data[split]["y"].shape[0] for split in ("train", "val", "test")) == 12
    assert sum(data[split]["y"].shape[1] for split in ("train", "val", "test")) == 8


@pytest.mark.parametrize(
    ("validation_setting", "y", "X_instance", "X_target", "message"),
    [
        (
            "B",
            np.arange(16, dtype=float).reshape(2, 8),
            np.arange(6, dtype=float).reshape(2, 3),
            np.arange(16, dtype=float).reshape(8, 2),
            "training instance",
        ),
        (
            "C",
            np.arange(24, dtype=float).reshape(12, 2),
            np.arange(36, dtype=float).reshape(12, 3),
            np.arange(4, dtype=float).reshape(2, 2),
            "training target",
        ),
    ],
)
def test_load_process_dp_rejects_splits_without_training_entities(
    tmp_path,
    validation_setting,
    y,
    X_instance,
    X_target,
    message,
):
    write_dp_dataset(
        tmp_path,
        y=y,
        X_instance=X_instance,
        X_target=X_target,
    )

    with pytest.raises(ValueError, match=message):
        load_process_DP(
            tmp_path,
            variant="divided",
            validation_setting=validation_setting,
        )


def write_movielens_dataset(
    root,
    content="10\t100\t5\t900\n20\t100\t3\t901\n10\t200\t4\t902\n",
):
    dataset_dir = root / "matrix-completion-datasets" / "ml-100k"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "u.data").write_text(content, encoding="utf-8")
    return dataset_dir


def write_movielens_archive(
    root,
    content="10\t100\t5\t900\n20\t100\t3\t901\n10\t200\t4\t902\n",
):
    archive_path = root / "matrix-completion-datasets" / "ml-100k.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("ml-100k/u.data", content)
    return archive_path


def test_load_process_mc_reads_cached_ratings_without_network(
    monkeypatch,
    tmp_path,
):
    write_movielens_dataset(tmp_path)

    def unexpected_operation(*args, **kwargs):
        raise AssertionError("a complete cache must not be downloaded or extracted")

    monkeypatch.setattr(dataset_module, "download_file", unexpected_operation)
    monkeypatch.setattr(dataset_module, "extract_zip_file", unexpected_operation)

    data = load_process_MC(tmp_path)

    pd.testing.assert_frame_equal(
        data["train"]["y"],
        pd.DataFrame(
            {
                "instance_id": [0, 1, 0],
                "target_id": [0, 0, 1],
                "value": [5, 3, 4],
            }
        ),
    )
    assert data["train"]["X_instance"] is None
    assert data["train"]["X_target"] is None
    assert data["val"] == {"y": None, "X_instance": None, "X_target": None}
    assert data["test"] == {"y": None, "X_instance": None, "X_target": None}


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"dataset_name": "unknown"}, ValueError, "dataset_name"),
        ({"print_mode": "unknown"}, ValueError, "print_mode"),
        ({"print_mode": []}, ValueError, "print_mode"),
        ({"path": object()}, TypeError, "path"),
    ],
)
def test_load_process_mc_validates_arguments_before_download(
    monkeypatch,
    tmp_path,
    overrides,
    exception,
    message,
):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("invalid arguments must not trigger a download")

    monkeypatch.setattr(dataset_module, "download_file", unexpected_download)
    arguments = {"path": tmp_path, "dataset_name": "ml-100k"}
    arguments.update(overrides)

    with pytest.raises(exception, match=message):
        load_process_MC(**arguments)


def test_load_process_mc_repairs_incomplete_cache_from_archive(
    monkeypatch,
    tmp_path,
):
    write_movielens_archive(tmp_path)

    monkeypatch.setattr(
        dataset_module,
        "download_file",
        lambda *args, **kwargs: pytest.fail("cached archive should be reused"),
    )

    data = load_process_MC(tmp_path)

    assert len(data["train"]["y"]) == 3
    assert (tmp_path / "matrix-completion-datasets" / "ml-100k" / "u.data").is_file()


def test_load_process_mc_replaces_corrupt_cached_archive(
    monkeypatch,
    tmp_path,
    capsys,
):
    archive_path = tmp_path / "matrix-completion-datasets" / "ml-100k.zip"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"not a zip")
    calls = []

    def download(url, destination):
        calls.append((url, destination))
        write_movielens_archive(tmp_path)

    monkeypatch.setattr(dataset_module, "download_file", download)
    progress = RecordingDataProgress()

    data = load_process_MC(
        tmp_path,
        print_mode="dev",
        progress=progress,
    )

    assert len(calls) == 1
    assert len(data["train"]["y"]) == 3
    assert capsys.readouterr().out == ""
    assert [
        (event.message, event.kind, event.subject, event.value, event.end)
        for event in progress.events
    ] == [
        (
            "Downloading dataset ml-100k from scratch... ",
            "operation_started",
            "mc_dataset_download",
            "ml-100k",
            "",
        ),
        (
            "Done",
            "operation_completed",
            "mc_dataset_download",
            "ml-100k",
            "\n",
        ),
    ]


def test_load_process_mc_reports_missing_downloaded_dataset(
    monkeypatch,
    tmp_path,
):
    def download(url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("ml-100k/README", "no ratings")

    monkeypatch.setattr(dataset_module, "download_file", download)

    with pytest.raises(FileNotFoundError, match="expected file"):
        load_process_MC(tmp_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "empty"),
        ("10\t100\t5\n", "missing fields"),
        ("10\t100\t5\t900\textra\n", "extra fields"),
        ("ten\t100\t5\t900\n", "must be numeric"),
        ("10\t100\t6\t900\n", "between 1 and 5"),
        ("10\t100\t3.5\t900\n", "must be integers"),
        (
            "10\t100\t5\t900\n10\t100\t4\t901\n",
            "duplicate instance-target",
        ),
    ],
)
def test_load_process_mc_rejects_invalid_ratings(tmp_path, content, message):
    write_movielens_dataset(tmp_path, content)

    with pytest.raises(ValueError, match=message):
        load_process_MC(tmp_path)


@pytest.mark.parametrize(
    ("dataset_name", "flatten_worker_labels"),
    [("dog", True), ("bird", False)],
)
def test_load_process_mtl_reads_cached_json_without_network(
    tmp_path,
    dataset_name,
    flatten_worker_labels,
):
    write_mtl_dataset(
        tmp_path,
        dataset_name=dataset_name,
        flatten_worker_labels=flatten_worker_labels,
    )

    data = load_process_MTL(tmp_path, dataset_name=dataset_name)

    interactions = pd.concat(
        [data["train"]["y"], data["test"]["y"]],
        ignore_index=True,
    )
    instance_features = pd.concat(
        [data["train"]["X_instance"], data["test"]["X_instance"]],
        ignore_index=True,
    )
    assert len(interactions) == 10
    assert set(interactions["value"]) == {0.0, 1.0}
    assert set(interactions["instance_id"]) == set(instance_features["id"])
    assert all(Path(image_path).is_file() for image_path in instance_features["dir"])


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"dataset_name": "unknown"}, ValueError, "dataset_name"),
        ({"print_mode": "unknown"}, ValueError, "print_mode"),
        ({"print_mode": []}, ValueError, "print_mode"),
        ({"path": object()}, TypeError, "path"),
    ],
)
def test_load_process_mtl_validates_arguments_before_download(
    monkeypatch,
    tmp_path,
    overrides,
    exception,
    message,
):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("invalid arguments must not trigger a download")

    monkeypatch.setattr(
        dataset_module,
        "download_and_extract_zip",
        unexpected_download,
    )
    arguments = {"path": tmp_path, "dataset_name": "dog"}
    arguments.update(overrides)

    with pytest.raises(exception, match=message):
        load_process_MTL(**arguments)


def test_load_process_mtl_repairs_incomplete_cache(monkeypatch, tmp_path, capsys):
    (tmp_path / "multi-task_learning-datasets" / "dog").mkdir(parents=True)
    calls = []

    def download(url, destination):
        calls.append((url, destination))
        write_mtl_dataset(tmp_path)

    monkeypatch.setattr(dataset_module, "download_and_extract_zip", download)
    progress = RecordingDataProgress()

    data = load_process_MTL(
        tmp_path,
        print_mode="dev",
        progress=progress,
    )

    assert len(calls) == 1
    assert len(data["train"]["y"]) + len(data["test"]["y"]) == 10
    assert capsys.readouterr().out == ""
    assert [
        (event.message, event.kind, event.subject, event.value, event.end)
        for event in progress.events
    ] == [
        (
            "Downloading dataset dog from scratch... ",
            "operation_started",
            "mtl_dataset_download",
            "dog",
            "",
        ),
        (
            "Done",
            "operation_completed",
            "mtl_dataset_download",
            "dog",
            "\n",
        ),
    ]


def test_load_process_mtl_reports_missing_downloaded_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dataset_module,
        "download_and_extract_zip",
        lambda url, destination: None,
    )

    with pytest.raises(FileNotFoundError, match="expected file"):
        load_process_MTL(tmp_path)


def test_load_process_mtl_supports_legacy_score_cache(tmp_path):
    dataset_dir = tmp_path / "multi-task_learning-datasets" / "dog"
    image_dir = dataset_dir / "image"
    image_dir.mkdir(parents=True)
    for index in range(1, 5):
        (image_dir / f"{index}.jpg").touch()
    with (dataset_dir / "y_dog.pkl").open("wb") as score_file:
        pickle.dump(np.ones((4, 2)), score_file)

    data = load_process_MTL(tmp_path)

    assert len(data["train"]["y"]) + len(data["test"]["y"]) == 8


def test_load_process_mtl_reports_missing_images(tmp_path):
    write_mtl_dataset(tmp_path, include_images=False)

    with pytest.raises(FileNotFoundError, match="image is missing"):
        load_process_MTL(tmp_path)


def test_load_process_mtr_reads_cached_arff_without_network(tmp_path):
    write_mtr_dataset(tmp_path)

    data = load_process_MTR(tmp_path, dataset_name="enb")

    np.testing.assert_array_equal(
        data["train"]["X_instance"],
        np.array([[1.0, 2.0], [5.0, 6.0]]),
    )
    np.testing.assert_array_equal(
        data["train"]["y"],
        np.array([[3.0, 4.0], [7.0, 8.0]]),
    )


def test_load_process_mtr_formats_dataframe_features_and_dev_status(
    tmp_path,
    capsys,
):
    write_mtr_dataset(tmp_path)

    data = load_process_MTR(
        tmp_path,
        dataset_name="enb",
        features_type="dataframe",
        print_mode="dev",
    )

    assert data["train"]["X_instance"].columns.tolist() == ["id", "features"]
    np.testing.assert_array_equal(
        np.stack(data["train"]["X_instance"]["features"]),
        np.array([[1.0, 2.0], [5.0, 6.0]]),
    )
    assert capsys.readouterr().out == "info: Processing...\ninfo: Done\n"


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"dataset_name": "unknown"}, ValueError, "dataset_name"),
        ({"features_type": "unknown"}, ValueError, "features_type"),
        ({"print_mode": "unknown"}, ValueError, "print_mode"),
        ({"print_mode": []}, ValueError, "print_mode"),
        ({"path": object()}, TypeError, "path"),
    ],
)
def test_load_process_mtr_validates_arguments_before_download(
    monkeypatch,
    tmp_path,
    overrides,
    exception,
    message,
):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("invalid arguments must not trigger a download")

    monkeypatch.setattr(
        dataset_module,
        "download_and_extract_zip",
        unexpected_download,
    )
    arguments = {"path": tmp_path, "dataset_name": "enb"}
    arguments.update(overrides)

    with pytest.raises(exception, match=message):
        load_process_MTR(**arguments)


def test_load_process_mtr_rejects_file_as_dataset_root(tmp_path):
    dataset_root = tmp_path / "not-a-directory"
    dataset_root.write_text("content", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        load_process_MTR(dataset_root)


def test_load_process_mtr_repairs_incomplete_cache(monkeypatch, tmp_path, capsys):
    (tmp_path / "mtr-datasets").mkdir()
    calls = []

    def download(url, destination):
        calls.append((url, destination))
        write_mtr_dataset(destination)

    monkeypatch.setattr(dataset_module, "download_and_extract_zip", download)
    progress = RecordingDataProgress()

    data = load_process_MTR(
        tmp_path,
        print_mode="dev",
        progress=progress,
    )

    assert len(calls) == 1
    assert data["train"]["y"].shape == (2, 2)
    assert capsys.readouterr().out == ""
    assert [
        (event.kind, event.subject, event.value, event.end) for event in progress.events
    ] == [
        ("operation_started", "mtr_dataset_loading", "enb", "\n"),
        ("operation_started", "mtr_dataset_download", "enb", ""),
        ("operation_completed", "mtr_dataset_download", "enb", "\n"),
        ("operation_completed", "mtr_dataset_loading", "enb", "\n"),
    ]


def test_failed_mtr_download_does_not_emit_completion(
    monkeypatch,
    tmp_path,
    capsys,
):
    progress = RecordingDataProgress()
    monkeypatch.setattr(
        dataset_module,
        "download_and_extract_zip",
        lambda url, destination: (_ for _ in ()).throw(
            DatasetDownloadError("download failed")
        ),
    )

    with pytest.raises(DatasetDownloadError, match="download failed"):
        load_process_MTR(
            tmp_path,
            print_mode="dev",
            progress=progress,
        )

    assert capsys.readouterr().out == ""
    assert [(event.kind, event.subject) for event in progress.events] == [
        ("operation_started", "mtr_dataset_loading"),
        ("operation_started", "mtr_dataset_download"),
    ]


def test_load_process_mtr_reports_missing_downloaded_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dataset_module,
        "download_and_extract_zip",
        lambda url, destination: None,
    )

    with pytest.raises(FileNotFoundError, match="expected file"):
        load_process_MTR(tmp_path)


def test_load_process_mtr_rejects_invalid_dataset_shape(tmp_path):
    write_mtr_dataset(
        tmp_path,
        """
@RELATION enb
@ATTRIBUTE target_1 NUMERIC
@ATTRIBUTE target_2 NUMERIC
@DATA
1,2
3,4
""".strip(),
    )

    with pytest.raises(ValueError, match="invalid shape"):
        load_process_MTR(tmp_path)


def test_generate_mtp_dataset_splits_without_side_features():
    data = generate_MTP_dataset(
        20,
        10,
        split_instances={"train": 0.6, "val": 0.1, "test": 0.3},
        split_targets={"train": 0.6, "val": 0.1, "test": 0.3},
    )

    for split_name in ("train", "val", "test"):
        assert data[split_name]["y"] is not None
        assert data[split_name]["X_instance"] is None
        assert data[split_name]["X_target"] is None


def test_generate_mtp_dataset_splits_both_axes_consistently():
    data = generate_MTP_dataset(
        20,
        10,
        num_instance_features=3,
        num_target_features=2,
        split_instances={"val": 0.2, "test": 0.2},
        split_targets={"train": 0.5, "val": 0.2, "test": 0.3},
    )
    expected_sizes = {
        "train": (12, 5),
        "val": (4, 2),
        "test": (4, 3),
    }

    for split_name, (num_split_instances, num_split_targets) in expected_sizes.items():
        interactions = data[split_name]["y"]["data"]
        instance_features = data[split_name]["X_instance"]["data"]
        target_features = data[split_name]["X_target"]["data"]

        assert len(interactions) == num_split_instances * num_split_targets
        assert len(instance_features) == num_split_instances
        assert len(target_features) == num_split_targets
        assert set(interactions["instance_id"]) == set(instance_features["id"])
        assert set(interactions["target_id"]) == set(target_features["id"])


@pytest.mark.parametrize(
    ("split_instances", "split_targets"),
    [
        (
            {"test": 0.2},
            {"val": 0.2, "test": 0.2},
        ),
        (
            {"val": 0.2, "test": 0.2},
            {"test": 0.2},
        ),
    ],
)
def test_generate_mtp_dataset_keeps_mixed_axis_splits_aligned(
    split_instances,
    split_targets,
):
    data = generate_MTP_dataset(
        10,
        10,
        num_instance_features=2,
        num_target_features=2,
        split_instances=split_instances,
        split_targets=split_targets,
        return_static_features_data=True,
    )

    for split_name in ("train", "val", "test"):
        interactions = data[split_name]["y"]["data"]
        instance_features = data[split_name]["X_instance"]["data"]
        target_features = data[split_name]["X_target"]["data"]

        assert set(interactions["instance_id"]) == set(instance_features["id"])
        assert set(interactions["target_id"]) == set(target_features["id"])
        assert len(interactions) == len(instance_features) * len(target_features)


def test_generate_mtp_dataset_supports_test_only_split():
    data = generate_MTP_dataset(
        20,
        10,
        num_instance_features=3,
        split_instances={"test": 0.2},
    )

    assert len(data["train"]["y"]["data"]) == 160
    assert len(data["train"]["X_instance"]["data"]) == 16
    assert data["val"]["y"] is None
    assert data["val"]["X_instance"] is None
    assert len(data["test"]["y"]["data"]) == 40
    assert len(data["test"]["X_instance"]["data"]) == 4


def test_generate_mtp_dataset_rejects_splits_without_training_ids():
    with pytest.raises(ValueError, match="split_instances.*training"):
        generate_MTP_dataset(
            2,
            10,
            split_instances={"val": 0.2, "test": 0.2},
        )


@pytest.mark.parametrize(
    ("parameter_name", "value"),
    [
        ("num_instance_features", True),
        ("num_instance_features", 1.5),
        ("num_instance_features", -1),
        ("num_target_features", False),
        ("num_target_features", 1.5),
        ("num_target_features", -1),
    ],
)
def test_generate_mtp_dataset_validates_optional_feature_dimensions(
    parameter_name,
    value,
):
    with pytest.raises(ValueError, match=parameter_name):
        generate_MTP_dataset(20, 10, **{parameter_name: value})


def test_generate_mtp_dataset_treats_zero_features_as_absent():
    data = generate_MTP_dataset(
        20,
        10,
        num_instance_features=0,
        num_target_features=None,
        return_static_features_data=True,
    )

    assert len(data["train"]["y"]["data"]) == 200
    for split_name in ("train", "val", "test"):
        assert data[split_name]["X_instance"] is None
        assert data[split_name]["X_target"] is None


def test_generate_mtp_dataset_preserves_triplet_order_and_values():
    interactions = generate_MTP_dataset(3, 2)["train"]["y"]["data"]

    np.testing.assert_array_equal(
        interactions[["instance_id", "target_id"]].to_numpy(),
        np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [2, 0],
                [2, 1],
            ]
        ),
    )
    np.testing.assert_array_equal(
        interactions["value"].to_numpy(),
        np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
    )


def test_generate_mtp_dataset_validates_static_feature_flag():
    with pytest.raises(TypeError, match="return_static_features_data"):
        generate_MTP_dataset(
            20,
            10,
            return_static_features_data=1,
        )


@pytest.mark.parametrize("feature_key", ["X_instance", "X_target"])
def test_generate_mtp_dataset_returns_independent_static_feature_frames(feature_key):
    data = generate_MTP_dataset(
        6,
        4,
        num_instance_features=2,
        num_target_features=2,
        return_static_features_data=True,
    )
    frames = [
        data[split_name][feature_key]["data"] for split_name in ("train", "val", "test")
    ]

    assert len({id(frame) for frame in frames}) == 3
    assert not np.shares_memory(
        frames[0].iloc[0]["features"],
        frames[1].iloc[0]["features"],
    )
    assert not np.shares_memory(
        frames[0].iloc[0]["features"],
        frames[2].iloc[0]["features"],
    )

    frames[1].at[0, "id"] = 999
    frames[1].iloc[0]["features"][0] = 999

    assert frames[0].at[0, "id"] == 0
    assert frames[2].at[0, "id"] == 0
    assert frames[0].iloc[0]["features"][0] == 0
    assert frames[2].iloc[0]["features"][0] == 0


def test_generate_dummy_dataset_default_mode_is_valid():
    data = generate_dummy_dataset(
        num_instances=20,
        num_targets=5,
        num_instance_features=3,
        num_target_features=3,
        error_mu=0,
        error_sigma=0,
    )

    assert data["train"]["y"].shape == (14, 5)
    assert data["val"]["y"].shape == (2, 5)
    assert data["test"]["y"].shape == (4, 5)
    assert np.isfinite(data["train"]["y"]).all()


def test_generate_dummy_dataset_split_counts_use_original_population():
    data = generate_dummy_dataset(
        num_instances=11,
        num_targets=4,
        num_instance_features=3,
        num_target_features=3,
        error_mu=0,
        error_sigma=0,
        split_ratio={"train": 0.7, "val": 0.1, "test": 0.2},
    )

    assert data["train"]["y"].shape[0] == 6
    assert data["val"]["y"].shape[0] == 2
    assert data["test"]["y"].shape[0] == 3


@pytest.mark.parametrize("sklearn_version", [False, True])
@pytest.mark.parametrize(
    "mode",
    ["u+logv", "u*logv", "u^v", "u^v_reversed", "max(u,v)", "u*v"],
)
def test_generate_dummy_dataset_scores_match_selected_relationship(
    mode,
    sklearn_version,
):
    data = generate_dummy_dataset(
        num_instances=20,
        num_targets=4,
        num_instance_features=3,
        num_target_features=3,
        error_mu=0,
        error_sigma=0,
        mode=mode,
        sklearn_version=sklearn_version,
        seed=17,
    )
    target_features = data["train"]["X_target"]

    for split_name in ("train", "val", "test"):
        instance_features = data[split_name]["X_instance"]
        instance_values = instance_features[:, np.newaxis, :]
        target_values = target_features[np.newaxis, :, :]
        if mode == "u+logv":
            expected = np.sum(instance_values + np.log(target_values), axis=2)
        elif mode == "u*logv":
            expected = np.sum(instance_values * np.log(target_values), axis=2)
        elif mode == "u^v":
            expected = np.sum(instance_values**target_values, axis=2)
        elif mode == "u^v_reversed":
            expected = np.sum(
                instance_values ** target_values[:, :, ::-1],
                axis=2,
            )
        elif mode == "max(u,v)":
            expected = np.sum(np.maximum(instance_values, target_values), axis=2)
        else:
            expected = np.sum(instance_values * target_values, axis=2)

        np.testing.assert_allclose(
            data[split_name]["y"],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("mode", "expected_offset"),
    [
        ("u+logv", 2.5),
        ("max(u,v)", 0.0),
    ],
)
def test_generate_dummy_dataset_preserves_legacy_noise_modes(
    mode,
    expected_offset,
):
    without_error = generate_dummy_dataset(
        num_instances=20,
        num_targets=4,
        num_instance_features=3,
        num_target_features=3,
        error_mu=0,
        error_sigma=0,
        mode=mode,
        seed=17,
    )
    with_error = generate_dummy_dataset(
        num_instances=20,
        num_targets=4,
        num_instance_features=3,
        num_target_features=3,
        error_mu=2.5,
        error_sigma=0,
        mode=mode,
        seed=17,
    )

    for split_name in ("train", "val", "test"):
        np.testing.assert_allclose(
            with_error[split_name]["y"] - without_error[split_name]["y"],
            expected_offset,
        )


def test_generate_interaction_matrix_writes_expected_scores(tmp_path):
    input_path = tmp_path / "labels.json"
    output_path = tmp_path / "generated" / "scores.pkl"
    input_path.write_text(
        json.dumps(
            {
                "true_labels": [1, 0],
                "WorkerLabels": [[1, 2, -1], [0, 1, -1]],
            }
        ),
        encoding="utf-8",
    )

    generate_interaction_matrix(input_path, output_path)

    with output_path.open("rb") as output_file:
        scores = pickle.load(output_file)

    np.testing.assert_equal(
        scores,
        np.array([[1.0, 0.0, np.nan], [1.0, 0.0, np.nan]]),
    )


def test_generate_interaction_matrix_supports_single_flattened_row(tmp_path):
    input_path = tmp_path / "labels.json"
    output_path = tmp_path / "scores.pkl"
    input_path.write_text(
        json.dumps(
            {
                "true_labels": [1],
                "WorkerLabels": [1, 0, -1],
            }
        ),
        encoding="utf-8",
    )

    generate_interaction_matrix(input_path, output_path)

    with output_path.open("rb") as output_file:
        scores = pickle.load(output_file)
    np.testing.assert_equal(scores, np.array([[1.0, 0.0, np.nan]]))


def test_generate_interaction_matrix_reports_invalid_json(tmp_path):
    input_path = tmp_path / "labels.json"
    input_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        generate_interaction_matrix(input_path, tmp_path / "scores.pkl")


def test_generate_interaction_matrix_preserves_existing_output_on_write_failure(
    monkeypatch,
    tmp_path,
):
    input_path = tmp_path / "labels.json"
    output_path = tmp_path / "scores.pkl"
    input_path.write_text(
        json.dumps({"true_labels": [1], "WorkerLabels": [[1, 0]]}),
        encoding="utf-8",
    )
    output_path.write_bytes(b"existing output")

    def fail_after_partial_write(scores, output_file, *, protocol):
        output_file.write(b"partial output")
        raise pickle.PicklingError("serialization failed")

    monkeypatch.setattr(dataset_module.pickle, "dump", fail_after_partial_write)

    with pytest.raises(pickle.PicklingError, match="serialization failed"):
        generate_interaction_matrix(input_path, output_path)

    assert output_path.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".scores.pkl.*.tmp"))


def test_generate_interaction_matrix_rejects_directory_output(tmp_path):
    input_path = tmp_path / "labels.json"
    input_path.write_text(
        json.dumps({"true_labels": [1], "WorkerLabels": [[1]]}),
        encoding="utf-8",
    )
    output_path = tmp_path / "output"
    output_path.mkdir()

    with pytest.raises(IsADirectoryError, match="not a file"):
        generate_interaction_matrix(input_path, output_path)


@pytest.mark.parametrize(
    ("factory", "keyword"),
    [
        (process_dummy_MLC, "features_format"),
        (process_dummy_MTR, "features_format"),
        (process_dummy_DP, "instance_features_format"),
        (process_dummy_DP, "target_features_format"),
    ],
)
def test_dummy_datasets_reject_unknown_feature_formats(factory, keyword):
    with pytest.raises(ValueError, match=keyword):
        factory(**{keyword: "unknown"})


@pytest.mark.parametrize(
    "factory",
    [process_dummy_MLC, process_dummy_MTR, process_dummy_DP],
)
@pytest.mark.parametrize("invalid_format", ["unknown", None, []])
def test_dummy_datasets_reject_unknown_interaction_formats(
    factory,
    invalid_format,
):
    with pytest.raises(ValueError, match="interaction_matrix_format"):
        factory(interaction_matrix_format=invalid_format)


@pytest.mark.parametrize("interaction_format", ["numpy", "dataframe"])
@pytest.mark.parametrize("features_format", ["numpy", "dataframe"])
def test_dummy_mlc_has_stable_format_contract(
    interaction_format,
    features_format,
):
    data = process_dummy_MLC(
        num_features=3,
        num_instances=8,
        num_targets=2,
        interaction_matrix_format=interaction_format,
        features_format=features_format,
        random_state=17,
    )

    assert list(data) == ["train", "test", "val"]
    assert data["train"]["X_target"] is None
    assert data["val"] == {"y": None, "X_instance": None, "X_target": None}
    assert data["test"] == {"y": None, "X_instance": None, "X_target": None}

    features = data["train"]["X_instance"]
    if features_format == "numpy":
        assert isinstance(features, np.ndarray)
        assert features.shape == (8, 3)
        np.testing.assert_array_equal(features[:, 0], np.arange(8))
    else:
        assert isinstance(features, pd.DataFrame)
        assert list(features.columns) == ["id", "features"]
        assert list(features["id"]) == list(range(8))
        feature_vectors = features["features"].tolist()
        assert all(vector.shape == (3,) for vector in feature_vectors)
        assert all(
            not np.shares_memory(left, right)
            for left, right in pairwise(feature_vectors)
        )

    interactions = data["train"]["y"]
    if interaction_format == "numpy":
        assert isinstance(interactions, np.ndarray)
        assert interactions.shape == (8, 2)
        assert set(np.unique(interactions)) <= {0, 1}
    else:
        assert isinstance(interactions, pd.DataFrame)
        assert list(interactions.columns) == [
            "instance_id",
            "target_id",
            "value",
        ]
        assert len(interactions) == 16
        assert interactions.groupby("instance_id").size().eq(2).all()
        assert set(interactions["value"].unique()) <= {0, 1}


def test_dummy_dp_dataframe_contract():
    data = process_dummy_DP(
        num_instance_features=3,
        num_target_features=2,
        num_instances=6,
        num_targets=4,
        interaction_matrix_format="dataframe",
        instance_features_format="dataframe",
        target_features_format="dataframe",
    )

    assert list(data["train"]["X_instance"]["id"]) == list(range(6))
    assert list(data["train"]["X_target"]["id"]) == list(range(4))
    assert list(data["train"]["y"].columns) == [
        "instance_id",
        "target_id",
        "value",
    ]
    assert len(data["train"]["y"]) == 24
    assert data["val"] == {"y": None, "X_instance": None, "X_target": None}
    assert data["test"] == {"y": None, "X_instance": None, "X_target": None}


def test_dummy_mtr_dataframe_splits_keep_features_and_interactions_aligned():
    data = process_dummy_MTR(
        num_instances=20,
        num_targets=3,
        interaction_matrix_format="dataframe",
        features_format="dataframe",
        variant="divided",
        split_ratio={"train": 0.7, "val": 0.1, "test": 0.2},
        random_state=17,
    )

    for split_name, expected_instances in (
        ("train", 14),
        ("val", 2),
        ("test", 4),
    ):
        features = data[split_name]["X_instance"]
        interactions = data[split_name]["y"]
        assert len(features) == expected_instances
        assert len(interactions) == expected_instances * 3
        assert set(features["id"]) == set(interactions["instance_id"])
        assert interactions.groupby("instance_id").size().eq(3).all()
        assert data[split_name]["X_target"] is None


def test_dummy_mtr_rejects_split_that_cannot_leave_training_instances():
    with pytest.raises(ValueError, match="leave at least one training instance"):
        process_dummy_MTR(
            num_instances=2,
            variant="divided",
        )


@pytest.mark.parametrize("variant", ["unknown", None, []])
def test_dummy_mtr_rejects_invalid_variant(variant):
    with pytest.raises(ValueError, match="variant"):
        process_dummy_MTR(variant=variant)


def test_dummy_mtr_split_ratio_is_relative_to_original_dataset():
    data = process_dummy_MTR(
        num_instances=100,
        variant="divided",
        split_ratio={"train": 0.7, "val": 0.1, "test": 0.2},
    )

    assert data["train"]["y"].shape[0] == 70
    assert data["val"]["y"].shape[0] == 10
    assert data["test"]["y"].shape[0] == 20


def test_dummy_mtr_derives_omitted_train_split_ratio():
    data = process_dummy_MTR(
        num_instances=100,
        variant="divided",
        split_ratio={"val": 0.2, "test": 0.2},
    )

    assert data["train"]["y"].shape[0] == 60
    assert data["val"]["y"].shape[0] == 20
    assert data["test"]["y"].shape[0] == 20


def test_generated_dummy_dataset_requires_matching_feature_dimensions():
    with pytest.raises(ValueError, match="must match"):
        generate_dummy_dataset(
            num_instances=20,
            num_targets=5,
            num_instance_features=3,
            num_target_features=2,
            error_mu=0,
            error_sigma=0,
        )


def test_generated_dummy_dataset_rejects_split_without_training_instances():
    with pytest.raises(ValueError, match="leave at least one training instance"):
        generate_dummy_dataset(
            num_instances=2,
            num_targets=5,
            num_instance_features=3,
            num_target_features=3,
            error_mu=0,
            error_sigma=0,
        )


@pytest.mark.parametrize("split_name", ["split_instances", "split_targets"])
def test_generate_mtp_dataset_requires_test_ratio_for_validation(split_name):
    arguments = {split_name: {"val": 0.1}}

    with pytest.raises(ValueError, match="requires a test ratio"):
        generate_MTP_dataset(20, 10, **arguments)


@pytest.mark.parametrize("split_name", ["split_instances", "split_targets"])
@pytest.mark.parametrize(
    ("ratio", "exception"),
    [
        ({"train": 0.7}, ValueError),
        ({"train": 0.8, "test": 0.3}, ValueError),
        ({"val": 0.8, "test": 0.3}, ValueError),
        ({"test": True}, ValueError),
        ({"test": 0.2, "holdout": 0.1}, ValueError),
        ([("test", 0.2)], TypeError),
    ],
)
def test_generate_mtp_dataset_rejects_invalid_split_ratios(
    split_name,
    ratio,
    exception,
):
    with pytest.raises(exception, match=split_name):
        generate_MTP_dataset(20, 10, **{split_name: ratio})


def test_dummy_mlc_generates_both_binary_classes():
    data = process_dummy_MLC(num_instances=100, num_targets=10)

    assert set(np.unique(data["train"]["y"])) == {0, 1}


def test_generated_dummy_dataset_is_reproducible_from_seed():
    arguments = {
        "num_instances": 20,
        "num_targets": 5,
        "num_instance_features": 3,
        "num_target_features": 3,
        "error_mu": 0,
        "error_sigma": 0.1,
        "seed": 17,
    }

    first = generate_dummy_dataset(**arguments)
    second = generate_dummy_dataset(**arguments)

    for split_name in ("train", "val", "test"):
        np.testing.assert_array_equal(
            first[split_name]["y"],
            second[split_name]["y"],
        )
        np.testing.assert_array_equal(
            first[split_name]["X_instance"],
            second[split_name]["X_instance"],
        )


@pytest.mark.parametrize("sklearn_version", [False, True])
@pytest.mark.parametrize(
    "mode",
    ["u+logv", "u*logv", "u^v", "u^v_reversed", "max(u,v)", "u*v"],
)
def test_generated_dummy_dataset_modes_produce_finite_values(
    mode,
    sklearn_version,
):
    data = generate_dummy_dataset(
        num_instances=20,
        num_targets=5,
        num_instance_features=3,
        num_target_features=3,
        error_mu=0,
        error_sigma=0.1,
        mode=mode,
        sklearn_version=sklearn_version,
    )

    for split_name in ("train", "val", "test"):
        assert np.isfinite(data[split_name]["y"]).all()
        assert np.isfinite(data[split_name]["X_instance"]).all()
    assert np.isfinite(data["train"]["X_target"]).all()
    assert data["val"]["X_target"] is None
    assert data["test"]["X_target"] is None


@pytest.mark.parametrize(
    "ratio",
    [
        {"val": 0.1},
        {"val": True, "test": 0.2},
        {"val": "invalid", "test": 0.2},
        {"val": 0.5, "test": 0.5},
        {"train": True, "val": 0.1, "test": 0.2},
        {"train": 0.8, "val": 0.1, "test": 0.2},
        {"val": 0.1, "test": 0.2, "holdout": 0.1},
    ],
)
def test_dummy_mtr_rejects_invalid_split_ratios(ratio):
    with pytest.raises(ValueError, match="split ratio|sum"):
        process_dummy_MTR(split_ratio=ratio)


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"mode": "unsupported"}, ValueError, "mode"),
        ({"sklearn_version": 1}, TypeError, "sklearn_version"),
        ({"error_mu": np.nan}, ValueError, "error_mu"),
        ({"error_sigma": np.inf}, ValueError, "error_sigma"),
        ({"error_sigma": -0.1}, ValueError, "error_sigma"),
    ],
)
def test_generated_dummy_dataset_validates_generation_parameters(
    overrides,
    exception,
    message,
):
    arguments = {
        "num_instances": 20,
        "num_targets": 5,
        "num_instance_features": 3,
        "num_target_features": 3,
        "error_mu": 0,
        "error_sigma": 0,
    }
    arguments.update(overrides)

    with pytest.raises(exception, match=message):
        generate_dummy_dataset(**arguments)


@pytest.mark.parametrize(
    ("factory", "overrides", "message"),
    [
        (process_dummy_MLC, {"num_features": 0}, "num_features"),
        (process_dummy_MTR, {"num_instances": 0}, "num_instances"),
        (process_dummy_DP, {"num_targets": 0}, "num_targets"),
        (
            generate_MTP_dataset,
            {"num_instances": 0, "num_targets": 5},
            "num_instances",
        ),
    ],
)
def test_dummy_datasets_reject_non_positive_dimensions(factory, overrides, message):
    with pytest.raises(ValueError, match=message):
        factory(**overrides)


@pytest.mark.parametrize(
    ("factory", "arguments"),
    [
        (process_dummy_MLC, {"random_state": 17}),
        (
            process_dummy_MTR,
            {
                "variant": "divided",
                "random_state": 17,
            },
        ),
        (process_dummy_DP, {"random_state": 17}),
    ],
)
def test_dummy_dataset_factories_are_reproducible(factory, arguments):
    first = factory(**arguments)
    second = factory(**arguments)

    for split_name in ("train", "val", "test"):
        for source_name in ("y", "X_instance", "X_target"):
            first_source = first[split_name][source_name]
            second_source = second[split_name][source_name]
            if first_source is None:
                assert second_source is None
            else:
                np.testing.assert_array_equal(first_source, second_source)


@pytest.mark.parametrize(
    ("factory", "arguments"),
    [
        (process_dummy_MLC, {"random_state": 17}),
        (process_dummy_MTR, {"random_state": None}),
        (process_dummy_DP, {"random_state": 17}),
        (
            generate_dummy_dataset,
            {
                "num_instances": 20,
                "num_targets": 5,
                "num_instance_features": 3,
                "num_target_features": 3,
                "error_mu": 0,
                "error_sigma": 0,
                "seed": 17,
            },
        ),
    ],
)
def test_dummy_dataset_factories_do_not_mutate_global_rng(factory, arguments):
    original_state = np.random.get_state()
    try:
        np.random.seed(1234)
        expected_values = np.random.random(3)
        np.random.seed(1234)

        factory(**arguments)

        actual_values = np.random.random(3)
    finally:
        np.random.set_state(original_state)

    np.testing.assert_array_equal(actual_values, expected_values)


def test_dummy_dataset_random_state_changes_generated_values():
    first = process_dummy_MLC(random_state=1)
    second = process_dummy_MLC(random_state=2)

    assert not np.array_equal(
        first["train"]["X_instance"],
        second["train"]["X_instance"],
    )


@pytest.mark.parametrize(
    ("factory", "arguments", "parameter_name"),
    [
        (process_dummy_MLC, {"random_state": True}, "random_state"),
        (process_dummy_MTR, {"random_state": -1}, "random_state"),
        (process_dummy_DP, {"random_state": 2**32}, "random_state"),
        (
            generate_dummy_dataset,
            {
                "num_instances": 20,
                "num_targets": 5,
                "num_instance_features": 3,
                "num_target_features": 3,
                "error_mu": 0,
                "error_sigma": 0,
                "seed": True,
            },
            "seed",
        ),
        (
            generate_dummy_dataset,
            {
                "num_instances": 20,
                "num_targets": 5,
                "num_instance_features": 3,
                "num_target_features": 3,
                "error_mu": 0,
                "error_sigma": 0,
                "seed": None,
            },
            "seed",
        ),
    ],
)
def test_dummy_dataset_factories_validate_random_state(
    factory,
    arguments,
    parameter_name,
):
    with pytest.raises(ValueError, match=parameter_name):
        factory(**arguments)
