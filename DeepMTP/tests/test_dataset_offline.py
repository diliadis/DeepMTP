import json
import pickle

import numpy as np
import pytest

from DeepMTP.dataset import (
    generate_dummy_dataset,
    generate_interaction_matrix,
    generate_MTP_dataset,
    process_dummy_DP,
    process_dummy_MLC,
    process_dummy_MTR,
)


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


def test_generate_interaction_matrix_writes_expected_scores(tmp_path):
    input_path = tmp_path / "labels.json"
    output_path = tmp_path / "scores.pkl"
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


def test_dummy_mtr_split_ratio_is_relative_to_original_dataset():
    data = process_dummy_MTR(
        num_instances=100,
        variant="divided",
        split_ratio={"train": 0.7, "val": 0.1, "test": 0.2},
    )

    assert data["train"]["y"].shape[0] == 70
    assert data["val"]["y"].shape[0] == 10
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


@pytest.mark.parametrize("split_name", ["split_instances", "split_targets"])
def test_generate_mtp_dataset_requires_test_ratio_for_validation(split_name):
    arguments = {split_name: {"val": 0.1}}

    with pytest.raises(ValueError, match="requires a test ratio"):
        generate_MTP_dataset(20, 10, **arguments)


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


@pytest.mark.parametrize(
    "ratio",
    [
        {"val": 0.1},
        {"val": True, "test": 0.2},
        {"val": "invalid", "test": 0.2},
        {"val": 0.5, "test": 0.5},
    ],
)
def test_dummy_mtr_rejects_invalid_split_ratios(ratio):
    with pytest.raises(ValueError, match="split ratio|sum"):
        process_dummy_MTR(split_ratio=ratio)


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
