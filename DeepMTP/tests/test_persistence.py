import json
import pickle

import pytest
import torch

from DeepMTP.runtime.persistence import ExperimentStore


def test_create_experiment_directory_is_nested_and_collision_safe(tmp_path):
    first = ExperimentStore.create(tmp_path / "nested", "experiment")
    second = ExperimentStore.create(tmp_path / "nested", "experiment")

    assert first.experiment_dir == tmp_path / "nested" / "experiment"
    assert second.experiment_dir == tmp_path / "nested" / "experiment_1"


@pytest.mark.parametrize("experiment_name", ["", ".", "..", "nested/run"])
def test_create_rejects_invalid_experiment_names(tmp_path, experiment_name):
    with pytest.raises(ValueError, match="experiment_name"):
        ExperimentStore.create(tmp_path, experiment_name)


def test_summary_and_json_config_are_persisted(tmp_path):
    store = ExperimentStore.create(tmp_path, "run")

    summary_path = store.save_summary("train", "validation", "test")
    config_path = store.save_config({"learning_rate": 0.001})

    assert summary_path.read_text() == "train\nvalidation\ntest"
    assert json.loads(config_path.read_text()) == {"learning_rate": 0.001}


def test_non_json_config_falls_back_to_pickle(tmp_path):
    store = ExperimentStore.create(tmp_path, "run")
    store.save_config({"serializable": True})

    config_path = store.save_config({"transform": object()})

    assert config_path == store.experiment_dir / "config.pkl"
    assert not (store.experiment_dir / "config.json").exists()
    with config_path.open("rb") as config_file:
        assert "transform" in pickle.load(config_file)


def test_checkpoint_round_trip_maps_tensors_to_cpu(tmp_path):
    store = ExperimentStore.create(tmp_path, "run")
    checkpoint_path = store.save_checkpoint(
        model_state_dict={"weight": torch.tensor([1.0])},
        optimizer_state_dict={
            "state": {0: {"momentum": torch.tensor([2.0])}},
            "param_groups": [],
        },
        config={"compute_mode": "cuda:0"},
    )

    checkpoint = ExperimentStore.load_checkpoint(checkpoint_path, map_location="cpu")

    assert set(checkpoint) == {
        "model_state_dict",
        "optimizer_state_dict",
        "config",
    }
    assert checkpoint["model_state_dict"]["weight"].device.type == "cpu"
    assert (
        checkpoint["optimizer_state_dict"]["state"][0]["momentum"].device.type == "cpu"
    )


def test_checkpoint_load_validates_path_and_payload(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExperimentStore.load_checkpoint(tmp_path / "missing.pt")

    invalid_path = tmp_path / "invalid.pt"
    torch.save({"model_state_dict": {}}, invalid_path)

    with pytest.raises(ValueError, match="missing required keys"):
        ExperimentStore.load_checkpoint(invalid_path)
