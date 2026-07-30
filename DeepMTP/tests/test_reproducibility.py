import copy
from pathlib import Path

import numpy as np
import pytest

from DeepMTP.data import (
    DATA_PREPARATION_STATE_KEY,
    DataPreparationState,
    DenseScalingState,
    data_process,
    transform_dense_features,
)
from DeepMTP.main import DeepMTP
from DeepMTP.runtime.persistence import ExperimentStore


def make_dense_data():
    return {
        "train": {
            "y": np.tile(np.array([[0, 1]]), (8, 1)),
            "X_instance": np.arange(8, dtype=float).reshape(-1, 1),
        }
    }


def make_dense_config(tmp_path, *, experiment_name, save_model=True):
    return {
        "validation_setting": "B",
        "problem_mode": "classification",
        "general_architecture_version": "dot_product",
        "compute_mode": "cpu",
        "num_workers": 0,
        "train_batchsize": 8,
        "val_batchsize": 8,
        "num_epochs": 1,
        "metrics": [],
        "metrics_average": [],
        "evaluate_train": False,
        "evaluate_val": False,
        "use_early_stopping": False,
        "save_model": save_model,
        "results_path": str(tmp_path),
        "experiment_name": experiment_name,
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": 1,
        "instance_branch_nodes_per_layer": [2],
        "target_branch_architecture": "EMBEDDING",
        "target_branch_input_dim": 2,
        "embedding_size": 2,
    }


def test_data_process_captures_and_replays_dense_scaling_state():
    train, validation, test, data_info = data_process(
        make_dense_data(),
        validation_setting="B",
        ratio={"train": 0.5, "val": 0.25, "test": 0.25},
        shuffle=False,
        scale_instance_features="Standard",
    )

    saved_state = data_info["data_preparation_state"]
    assert all(
        split[DATA_PREPARATION_STATE_KEY] == saved_state
        for split in (train, validation, test)
    )
    restored_state = DataPreparationState.from_config(saved_state)
    assert restored_state.validation_setting == "B"
    assert restored_state.split_method == "random"
    assert restored_state.seed == 42
    assert restored_state.dense.instance is not None
    np.testing.assert_allclose(
        transform_dense_features(
            np.array([[0.0], [3.0], [7.0]]),
            restored_state.dense.instance,
        ).ravel(),
        np.array([-1.34164079, 1.34164079, 4.91934955]),
    )

    replayed = data_process(
        make_dense_data(),
        validation_setting="B",
        ratio={"train": 0.5, "val": 0.25, "test": 0.25},
        shuffle=False,
        dense_preprocessing_state=saved_state,
    )
    for original_split, replayed_split in zip(
        (train, validation, test),
        replayed[:3],
        strict=True,
    ):
        np.testing.assert_allclose(
            np.stack(original_split["X_instance"]["data"]["features"]),
            np.stack(replayed_split["X_instance"]["data"]["features"]),
        )


def test_dense_scaling_state_validates_values_and_feature_width():
    valid_state = {
        "version": 1,
        "method": "Standard",
        "feature_count": 2,
        "offsets": [1.0, 2.0],
        "scales": [2.0, 4.0],
        "data_minimums": [],
        "data_maximums": [],
        "feature_range": [0.0, 1.0],
    }
    state = DenseScalingState.from_config(valid_state)

    np.testing.assert_allclose(
        state.transform([[3.0, 6.0]]),
        [[1.0, 1.0]],
    )
    with pytest.raises(ValueError, match="feature width"):
        state.transform([[1.0]])
    with pytest.raises(ValueError, match="finite"):
        DenseScalingState.from_config(
            {
                **valid_state,
                "offsets": [1.0, float("nan")],
            }
        )
    with pytest.raises(ValueError, match="version"):
        DenseScalingState.from_config({**valid_state, "version": 999})


@pytest.mark.integration
def test_checkpoint_captures_preprocessing_provenance_and_restores_it(tmp_path):
    train, validation, test, data_info = data_process(
        make_dense_data(),
        validation_setting="B",
        ratio={"train": 0.5, "val": 0.25, "test": 0.25},
        shuffle=False,
        scale_instance_features="Standard",
    )
    model = DeepMTP(
        make_dense_config(
            tmp_path,
            experiment_name="reproducible",
        )
    )

    model.train(train, validation, test)

    checkpoint_path = Path(model.experiment_dir) / "model.pt"
    checkpoint = ExperimentStore.load_checkpoint(checkpoint_path)
    assert checkpoint["metadata"]["format_version"] == 1
    assert (
        checkpoint["metadata"]["preprocessing"]["data_preparation_state"]
        == data_info["data_preparation_state"]
    )
    assert checkpoint["metadata"]["training"]["best_epoch"] == 0
    assert checkpoint["metadata"]["environment"]["effective_device"] == "cpu"
    assert (
        checkpoint["config"]["data_preparation_state"]
        == data_info["data_preparation_state"]
    )

    restored = DeepMTP(
        {
            "results_path": str(tmp_path),
            "experiment_name": "restored",
        },
        checkpoint_dir=checkpoint_path,
    )
    assert restored.checkpoint_metadata == checkpoint["metadata"]
    assert restored.preprocessing_state == data_info["data_preparation_state"]
    np.testing.assert_allclose(
        restored.transform_dense_features([[0.0], [7.0]], branch="instance"),
        [[-1.34164079], [4.91934955]],
    )


def test_legacy_checkpoint_remains_loadable_without_metadata(tmp_path):
    source = DeepMTP(
        make_dense_config(
            tmp_path,
            experiment_name="legacy-source",
            save_model=False,
        )
    )
    store = ExperimentStore.create(tmp_path, "legacy-checkpoint")
    checkpoint_path = store.save_checkpoint(
        model_state_dict=source.deepMTP_model.state_dict(),
        optimizer_state_dict=source.optimizer.state_dict(),
        config=source.config,
    )

    restored = DeepMTP(
        {
            "results_path": str(tmp_path),
            "experiment_name": "legacy-restored",
        },
        checkpoint_dir=checkpoint_path,
    )

    assert restored.checkpoint_metadata is None
    assert restored.preprocessing_state is None


def test_trainer_rejects_inconsistent_split_preprocessing_state(tmp_path):
    train, validation, test, _ = data_process(
        make_dense_data(),
        validation_setting="B",
        ratio={"train": 0.5, "val": 0.25, "test": 0.25},
        shuffle=False,
        scale_instance_features="Standard",
    )
    inconsistent_test = copy.deepcopy(test)
    inconsistent_test[DATA_PREPARATION_STATE_KEY]["seed"] = 999
    model = DeepMTP(
        make_dense_config(
            tmp_path,
            experiment_name="inconsistent",
            save_model=False,
        )
    )

    with pytest.raises(ValueError, match="must carry the same"):
        model.train(train, validation, inconsistent_test)


@pytest.mark.integration
def test_training_publishes_reproducible_model_artifact_to_capable_reporter(
    tmp_path,
):
    class ArtifactReporter:
        def __init__(self):
            self.started_config = None
            self.artifacts = []

        def start(self, config, model):
            self.started_config = dict(config)

        def log_metrics(self, metrics, *, step=None):
            return None

        def runtime_metadata(self):
            return {"wandb": {"run_id": "run-id"}}

        def log_model_artifact(self, artifact):
            self.artifacts.append(artifact)
            return {"wandb_model_artifact": {"name": "model:v0"}}

        def close(self):
            return None

    train, validation, test, data_info = data_process(
        make_dense_data(),
        validation_setting="B",
        ratio={"train": 0.5, "val": 0.25, "test": 0.25},
        shuffle=False,
        scale_instance_features="Standard",
    )
    config = make_dense_config(
        tmp_path,
        experiment_name="artifact-run",
    )
    config.update(
        {
            "wandb_project_name": "project",
            "wandb_project_entity": "entity",
            "wandb_log_model_artifact": True,
            "wandb_model_artifact_name": "deepmtp-model",
            "wandb_model_artifact_aliases": ["latest", "candidate"],
        }
    )
    reporter = ArtifactReporter()
    model = DeepMTP(config, reporter=reporter)

    model.train(train, validation, test)

    assert (
        reporter.started_config["data_preparation_state"]
        == data_info["data_preparation_state"]
    )
    assert len(reporter.artifacts) == 1
    artifact = reporter.artifacts[0]
    assert artifact.name == "deepmtp-model"
    assert artifact.aliases == ("latest", "candidate")
    assert artifact.checkpoint_path.name == "model.pt"
    assert {path.name for path in artifact.files} == {
        "model.pt",
        "config.json",
        "summary.txt",
    }
    checkpoint = ExperimentStore.load_checkpoint(artifact.checkpoint_path)
    assert checkpoint["metadata"]["tracking"]["wandb"]["run_id"] == "run-id"
