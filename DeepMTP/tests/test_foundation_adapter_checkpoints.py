import copy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import torch

from DeepMTP import DeepMTP
from DeepMTP.data import SequenceInput
from DeepMTP.runtime.foundation_checkpoints import (
    FOUNDATION_ADAPTER_CHECKPOINT_NAME,
    load_foundation_adapter_checkpoint,
    save_foundation_adapter_checkpoint,
)
from DeepMTP.runtime.metadata import build_checkpoint_metadata


class _ArtifactReporter:
    def __init__(self) -> None:
        self.artifact = None

    def runtime_metadata(self):
        return {"provider": "wandb", "run_id": "adapter-run"}

    def log_model_artifact(self, artifact):
        self.artifact = artifact


def _write_local_bert(path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "hello",
        "world",
        "deepmtp",
    ]
    vocabulary_path = path / "vocab.txt"
    vocabulary_path.write_text("\n".join(vocabulary), encoding="utf-8")
    transformers.BertTokenizer(str(vocabulary_path)).save_pretrained(path)
    transformers.BertModel(
        transformers.BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=8,
            pad_token_id=0,
        )
    ).save_pretrained(path)


def _config(tmp_path: Path, model_path: Path, experiment_name: str, **overrides):
    values = {
        "validation_setting": "B",
        "problem_mode": "classification",
        "general_architecture_version": "dot_product",
        "compute_mode": "cpu",
        "num_workers": 0,
        "metrics": [],
        "metrics_average": [],
        "results_path": str(tmp_path / "results"),
        "experiment_name": experiment_name,
        "save_model": True,
        "checkpoint_format": "foundation_adapter",
        "instance_branch_architecture": "SEQUENCE",
        "instance_branch_input_dim": 8,
        "instance_branch_sequence_encoder": "foundation",
        "instance_branch_sequence_output_dim": 3,
        "instance_branch_sequence_padding_idx": 0,
        "instance_branch_foundation_model": {
            "model_name": str(model_path),
            "local_files_only": True,
            "max_length": 8,
            "train_mode": "lora",
            "lora": {
                "rank": 2,
                "alpha": 4,
                "dropout": 0.0,
                "target_modules": ["query", "value"],
            },
        },
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 2,
        "target_branch_nodes_per_layer": [3],
        "embedding_size": 3,
    }
    values.update(overrides)
    return values


def _inputs() -> tuple[SequenceInput, torch.Tensor]:
    return (
        SequenceInput(
            token_ids=torch.tensor([[2, 5, 6, 3], [2, 7, 3, 0]]),
            attention_mask=torch.tensor(
                [[True, True, True, True], [True, True, True, False]]
            ),
            lengths=torch.tensor([4, 3]),
            padding_idx=0,
        ),
        torch.tensor([[1.0, -1.0], [0.5, 2.0]]),
    )


def _complete_training_snapshot(model: DeepMTP) -> None:
    model.early_stopping(
        {"val_loss": 0.5},
        copy.deepcopy(model.deepMTP_model),
        epoch=1,
        optimizer_state_dict=model.optimizer.state_dict(),
    )


@pytest.mark.integration
def test_foundation_adapter_save_model_round_trips_without_base_weights(tmp_path):
    pytest.importorskip("peft")
    model_path = tmp_path / "local-bert"
    model_path.mkdir()
    _write_local_bert(model_path)
    reporter = _ArtifactReporter()
    source = DeepMTP(
        _config(
            tmp_path,
            model_path,
            "source",
            wandb_project_name="deepmtp",
            wandb_project_entity="test-team",
            wandb_log_model_artifact=True,
        ),
        reporter=reporter,
    )
    with torch.no_grad():
        for name, parameter in source.deepMTP_model.named_parameters():
            if parameter.requires_grad:
                parameter.fill_(0.125 if "lora_" in name else 0.25)
    source.deepMTP_model.eval()
    expected = source.deepMTP_model(*_inputs())
    _complete_training_snapshot(source)

    source.save_model()

    checkpoint_path = Path(source.experiment_dir) / FOUNDATION_ADAPTER_CHECKPOINT_NAME
    loaded = load_foundation_adapter_checkpoint(checkpoint_path)
    restored = DeepMTP(
        {
            "results_path": str(tmp_path / "restored-results"),
            "experiment_name": "restored",
        },
        checkpoint_dir=checkpoint_path,
    )
    restored.deepMTP_model.eval()
    actual = restored.deepMTP_model(*_inputs())

    assert checkpoint_path.is_file()
    assert not (Path(source.experiment_dir) / "model.pt").exists()
    assert loaded.manifest["artifact_kind"] == "deepmtp.foundation_adapter"
    assert [component.train_mode for component in loaded.components] == ["lora"]
    assert loaded.components[0].adapter_state_dict
    assert not any(".backbone." in name for name in loaded.task_state_dict)
    assert restored.checkpoint_dict["checkpoint_format"] == "foundation_adapter"
    assert restored.optimizer.state == {}
    assert torch.allclose(actual, expected)
    assert reporter.artifact.checkpoint_path == checkpoint_path
    assert reporter.artifact.files[0] == checkpoint_path
    assert reporter.artifact.metadata["artifact_kind"] == ("deepmtp.foundation_adapter")
    assert (
        reporter.artifact.metadata["foundation_components"][0]["train_mode"] == "lora"
    )


@pytest.mark.integration
def test_foundation_adapter_checkpoint_detects_tampered_state(tmp_path):
    pytest.importorskip("peft")
    model_path = tmp_path / "local-bert"
    model_path.mkdir()
    _write_local_bert(model_path)
    model = DeepMTP(_config(tmp_path, model_path, "tamper-source"))
    checkpoint_path = tmp_path / FOUNDATION_ADAPTER_CHECKPOINT_NAME
    metadata = build_checkpoint_metadata(model.configuration, effective_device="cpu")
    save_foundation_adapter_checkpoint(
        checkpoint_path,
        model=model.deepMTP_model,
        config=model.configuration,
        metadata=metadata,
    )
    with ZipFile(checkpoint_path) as source:
        members = {name: source.read(name) for name in source.namelist()}
    members["task_state.pt"] += b"tampered"
    with ZipFile(checkpoint_path, "w", compression=ZIP_DEFLATED) as destination:
        for name, content in members.items():
            destination.writestr(name, content)

    with pytest.raises(ValueError, match="integrity check failed"):
        load_foundation_adapter_checkpoint(checkpoint_path)


@pytest.mark.integration
def test_foundation_adapter_checkpoint_rejects_full_fine_tuning(tmp_path):
    model_path = tmp_path / "local-bert"
    model_path.mkdir()
    _write_local_bert(model_path)
    config = _config(tmp_path, model_path, "full-source")
    config["instance_branch_foundation_model"] = {
        "model_name": str(model_path),
        "local_files_only": True,
        "max_length": 8,
        "train_mode": "full",
    }
    model = DeepMTP(config)
    metadata = build_checkpoint_metadata(model.configuration, effective_device="cpu")

    with pytest.raises(ValueError, match="fully fine-tuned base weights"):
        save_foundation_adapter_checkpoint(
            tmp_path / FOUNDATION_ADAPTER_CHECKPOINT_NAME,
            model=model.deepMTP_model,
            config=model.configuration,
            metadata=metadata,
        )


def test_checkpoint_format_validates_supported_values(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_format"):
        DeepMTP(
            _config(
                tmp_path,
                tmp_path / "unused",
                "invalid-format",
                checkpoint_format="adapter-ish",
            )
        )
