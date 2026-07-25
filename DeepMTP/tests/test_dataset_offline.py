import json
import pickle

import numpy as np

from DeepMTP.dataset import (
    generate_dummy_dataset,
    generate_interaction_matrix,
    generate_MTP_dataset,
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
