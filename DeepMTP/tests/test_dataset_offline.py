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
    ],
)
def test_dummy_dataset_factories_validate_random_state(
    factory,
    arguments,
    parameter_name,
):
    with pytest.raises(ValueError, match=parameter_name):
        factory(**arguments)
