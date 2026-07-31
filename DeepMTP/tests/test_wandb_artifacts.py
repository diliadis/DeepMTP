from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import DeepMTP.runtime.wandb as wandb_module
from DeepMTP.main import DeepMTP
from DeepMTP.runtime.persistence import ExperimentStore
from DeepMTP.runtime.reporting import OptionalIntegrationError
from DeepMTP.runtime.wandb import WandBCheckpoint, download_wandb_checkpoint


def make_config(tmp_path, experiment_name):
    return {
        "validation_setting": "B",
        "problem_mode": "classification",
        "general_architecture_version": "dot_product",
        "compute_mode": "cpu",
        "num_workers": 0,
        "num_epochs": 1,
        "metrics": [],
        "metrics_average": [],
        "save_model": False,
        "results_path": str(tmp_path),
        "experiment_name": experiment_name,
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": 1,
        "instance_branch_nodes_per_layer": [2],
        "target_branch_architecture": "EMBEDDING",
        "target_branch_input_dim": 2,
        "embedding_size": 2,
    }


def test_download_wandb_checkpoint_uses_public_api(monkeypatch, tmp_path):
    requested_references = []

    class FakeArtifact:
        name = "model"
        version = "v3"
        digest = "digest"

        def download(self, *, root):
            artifact_root = Path(root) / "downloaded"
            artifact_root.mkdir()
            torch.save(
                {
                    "model_state_dict": {},
                    "optimizer_state_dict": {},
                    "config": {},
                },
                artifact_root / "model.pt",
            )
            return str(artifact_root)

    class FakeApi:
        def artifact(self, reference):
            requested_references.append(reference)
            return FakeArtifact()

    monkeypatch.setattr(
        wandb_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(Api=FakeApi),
    )

    downloaded = download_wandb_checkpoint(
        "entity/project/model:production",
        download_dir=tmp_path,
    )

    assert requested_references == ["entity/project/model:production"]
    assert downloaded.checkpoint_path == tmp_path / "downloaded" / "model.pt"
    assert downloaded.artifact_name == "model"
    assert downloaded.artifact_version == "v3"
    assert downloaded.artifact_digest == "digest"
    assert downloaded.checkpoint_format == "full"


def test_download_wandb_checkpoint_accepts_foundation_adapter(
    monkeypatch,
    tmp_path,
):
    class FakeArtifact:
        def download(self, *, root):
            artifact_root = Path(root) / "downloaded"
            artifact_root.mkdir()
            (artifact_root / "foundation_adapter.zip").write_bytes(b"bundle")
            return str(artifact_root)

    monkeypatch.setattr(
        wandb_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(
            Api=lambda: SimpleNamespace(artifact=lambda reference: FakeArtifact())
        ),
    )

    downloaded = download_wandb_checkpoint(
        "entity/project/adapter:latest",
        download_dir=tmp_path,
    )

    assert downloaded.checkpoint_path.name == "foundation_adapter.zip"
    assert downloaded.checkpoint_format == "foundation_adapter"


def test_download_wandb_checkpoint_validates_dependency_and_contents(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        wandb_module.importlib,
        "import_module",
        lambda module_name: (_ for _ in ()).throw(ModuleNotFoundError(module_name)),
    )
    with pytest.raises(OptionalIntegrationError, match=r"DeepMTP\[tracking\]"):
        download_wandb_checkpoint("entity/project/model:latest")

    class EmptyArtifact:
        def download(self, *, root):
            return root

    monkeypatch.setattr(
        wandb_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(
            Api=lambda: SimpleNamespace(artifact=lambda reference: EmptyArtifact())
        ),
    )
    with pytest.raises(ValueError, match="no supported DeepMTP checkpoint"):
        download_wandb_checkpoint(
            "entity/project/model:latest",
            download_dir=tmp_path,
        )


def test_deepmtp_restores_from_downloaded_wandb_artifact(monkeypatch, tmp_path):
    source = DeepMTP(make_config(tmp_path, "source"))
    store = ExperimentStore.create(tmp_path, "artifact")
    checkpoint_path = store.save_checkpoint(
        model_state_dict=source.deepMTP_model.state_dict(),
        optimizer_state_dict=source.optimizer.state_dict(),
        config=source.config,
    )
    downloaded = WandBCheckpoint(
        artifact_reference="entity/project/model:v1",
        checkpoint_path=checkpoint_path,
        download_root=checkpoint_path.parent,
        artifact_name="model",
        artifact_version="v1",
        artifact_digest="digest",
    )
    monkeypatch.setattr(
        wandb_module,
        "download_wandb_checkpoint",
        lambda *args, **kwargs: downloaded,
    )

    restored = DeepMTP.from_wandb_artifact(
        "entity/project/model:v1",
        {
            "results_path": str(tmp_path),
            "experiment_name": "restored",
        },
    )

    assert restored.wandb_artifact is downloaded
    assert restored.checkpoint_dict is not None
    assert restored.configuration.instance_branch_input_dim == 1
