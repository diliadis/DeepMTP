import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from DeepMTP import (
    DeepMTPConfig,
    FoundationCacheSpec,
    FoundationModelSpec,
    HuggingFaceSequenceEncoder,
)
from DeepMTP.data import SequenceInput
from DeepMTP.integrations import huggingface


class CountingBackbone(nn.Module):
    def __init__(self, *, revision="model-commit"):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=3,
            vocab_size=8,
            pad_token_id=0,
            _commit_hash=revision,
        )
        self.embedding = nn.Embedding(8, 3)
        with torch.no_grad():
            for token_id in range(8):
                self.embedding.weight[token_id] = torch.tensor(
                    [token_id * 10, token_id * 10 + 1, token_id * 10 + 2],
                    dtype=torch.float32,
                )
        self.batch_sizes = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids, attention_mask):
        self.batch_sizes.append(int(input_ids.shape[0]))
        hidden = self.embedding(input_ids)
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=hidden[:, 0],
        )


def make_spec(**overrides):
    values = {
        "model_name": "example/foundation-model",
        "revision": "requested-model-revision",
        "tokenizer_revision": "requested-tokenizer-revision",
        "pooling": "masked_mean",
        "train_mode": "frozen",
        "max_length": 8,
    }
    values.update(overrides)
    return FoundationModelSpec(**values)


def make_values():
    return SequenceInput(
        token_ids=torch.tensor(
            [
                [1, 2, 0],
                [3, 0, 0],
                [1, 2, 0],
            ]
        ),
        attention_mask=torch.tensor(
            [
                [True, True, False],
                [True, False, False],
                [True, True, False],
            ]
        ),
        lengths=torch.tensor([2, 1, 2]),
        padding_idx=0,
    )


def build_encoder(monkeypatch, spec):
    backbone = CountingBackbone()
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: backbone,
    )
    encoder = HuggingFaceSequenceEncoder(
        spec=spec,
        vocabulary_size=8,
        output_dim=3,
        padding_idx=0,
    )
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(3))
        encoder.projection.bias.zero_()
    return encoder, backbone


def make_config(spec):
    return DeepMTPConfig(
        validation_setting="B",
        problem_mode="classification",
        general_architecture_version="dot_product",
        compute_mode="cpu",
        metrics=["accuracy"],
        metrics_average=["macro"],
        instance_branch_architecture="SEQUENCE",
        instance_branch_input_dim=8,
        instance_branch_sequence_encoder="foundation",
        instance_branch_sequence_output_dim=3,
        instance_branch_sequence_padding_idx=0,
        instance_branch_foundation_model=spec.to_dict(),
        target_branch_architecture="MLP",
        target_branch_input_dim=2,
        target_branch_nodes_per_layer=[3],
        embedding_size=3,
    )


def test_cache_spec_normalizes_round_trips_and_validates():
    spec = FoundationCacheSpec.from_config(
        {"directory": "./cache", "memory_capacity": 12}
    )

    assert spec.to_dict() == {"directory": "./cache", "memory_capacity": 12}
    assert FoundationCacheSpec.from_config(spec) is spec
    with pytest.raises(ValueError, match="positive memory_capacity or a directory"):
        FoundationCacheSpec(memory_capacity=0)
    with pytest.raises(ValueError, match="non-negative"):
        FoundationCacheSpec(memory_capacity=-1)
    with pytest.raises(ValueError, match="unknown settings"):
        FoundationCacheSpec.from_config({"enabled": True})


@pytest.mark.parametrize("train_mode", ["full", "lora"])
def test_cache_rejects_trainable_foundation_modes(train_mode):
    values = {"train_mode": train_mode, "cache": FoundationCacheSpec()}
    if train_mode == "lora":
        values["lora"] = {
            "target_modules": ["query"],
            "rank": 2,
            "alpha": 4,
        }

    with pytest.raises(ValueError, match="requires train_mode='frozen'"):
        make_spec(**values)


def test_normalized_config_and_checkpoint_payload_preserve_cache_settings():
    spec = make_spec(cache=FoundationCacheSpec(memory_capacity=64))
    config = make_config(spec)

    restored = DeepMTPConfig.from_mapping(config.to_dict(include_defaults=True))

    assert restored.instance_branch_foundation_model["cache"] == {
        "directory": None,
        "memory_capacity": 64,
    }


def test_memory_cache_deduplicates_entities_and_preserves_projection_gradients(
    monkeypatch,
):
    encoder, backbone = build_encoder(
        monkeypatch,
        make_spec(cache=FoundationCacheSpec(memory_capacity=8)),
    )
    values = make_values()

    first = encoder(values)
    second = encoder(values)
    second.sum().backward()

    assert torch.equal(first, second)
    assert backbone.batch_sizes == [2]
    assert encoder.projection.weight.grad is not None
    assert backbone.embedding.weight.grad is None
    assert encoder.cache_stats == {
        "enabled": True,
        "identity": encoder.cache_stats["identity"],
        "directory": None,
        "memory_capacity": 8,
        "memory_entries": 2,
        "memory_hits": 2,
        "disk_hits": 0,
        "misses": 2,
        "disk_writes": 0,
    }


def test_memory_cache_uses_only_unpadded_entity_tokens(monkeypatch):
    encoder, backbone = build_encoder(
        monkeypatch,
        make_spec(cache=FoundationCacheSpec(memory_capacity=8)),
    )
    first = SequenceInput(
        token_ids=torch.tensor([[1, 2, 0]]),
        attention_mask=torch.tensor([[True, True, False]]),
        lengths=torch.tensor([2]),
        padding_idx=0,
    )
    second = SequenceInput(
        token_ids=torch.tensor([[1, 2, 0, 0, 0]]),
        attention_mask=torch.tensor([[True, True, False, False, False]]),
        lengths=torch.tensor([2]),
        padding_idx=0,
    )

    assert torch.equal(encoder(first), encoder(second))
    assert backbone.batch_sizes == [1]
    assert encoder.cache_stats["memory_hits"] == 1


def test_memory_cache_evicts_least_recently_used_entity(monkeypatch):
    encoder, backbone = build_encoder(
        monkeypatch,
        make_spec(cache=FoundationCacheSpec(memory_capacity=1)),
    )
    encoder(make_values())
    first_entity = SequenceInput(
        token_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.tensor([[True, True]]),
        lengths=torch.tensor([2]),
        padding_idx=0,
    )

    encoder(first_entity)

    assert backbone.batch_sizes == [2, 1]
    assert encoder.cache_stats["memory_entries"] == 1
    assert encoder.cache_stats["misses"] == 3


def test_disk_cache_reuses_entries_across_encoder_instances(tmp_path, monkeypatch):
    cache_spec = FoundationCacheSpec(
        directory=str(tmp_path / "foundation-cache"),
        memory_capacity=0,
    )
    spec = make_spec(
        cache=cache_spec,
        resolved_tokenizer_revision="tokenizer-commit",
    )
    first, first_backbone = build_encoder(monkeypatch, spec)
    expected = first(make_values())

    second, second_backbone = build_encoder(monkeypatch, spec)
    observed = second(make_values())

    assert torch.equal(observed, expected)
    assert first_backbone.batch_sizes == [2]
    assert second_backbone.batch_sizes == []
    assert second.cache_stats["disk_hits"] == 2
    assert second.cache_stats["memory_entries"] == 0
    cache_directory = Path(second.cache_stats["directory"])
    assert (
        json.loads((cache_directory / "manifest.json").read_text())["identity"][
            "model_revision"
        ]
        == "model-commit"
    )
    assert len(list((cache_directory / "entries").glob("*.pt"))) == 2

    second.clear_cache(disk=True)
    assert list((cache_directory / "entries").glob("*.pt")) == []


def test_persistent_cache_requires_resolved_tokenizer_revision(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="resolved_tokenizer_revision"):
        build_encoder(
            monkeypatch,
            make_spec(
                cache=FoundationCacheSpec(
                    directory=str(tmp_path),
                    memory_capacity=0,
                )
            ),
        )


def test_persistent_cache_rejects_a_tampered_manifest(tmp_path, monkeypatch):
    spec = make_spec(
        cache=FoundationCacheSpec(directory=str(tmp_path)),
        resolved_tokenizer_revision="tokenizer-commit",
    )
    encoder, _backbone = build_encoder(monkeypatch, spec)
    cache_directory = Path(encoder.cache_stats["directory"])
    manifest_path = cache_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["pooling"] = "cls"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="stale or incompatible"):
        build_encoder(monkeypatch, spec)


def test_persistent_cache_rejects_a_corrupt_entry(tmp_path, monkeypatch):
    spec = make_spec(
        cache=FoundationCacheSpec(directory=str(tmp_path), memory_capacity=0),
        resolved_tokenizer_revision="tokenizer-commit",
    )
    encoder, _backbone = build_encoder(monkeypatch, spec)
    encoder(make_values())
    cache_directory = Path(encoder.cache_stats["directory"])
    entry_path = next((cache_directory / "entries").glob("*.pt"))
    entry_path.write_bytes(b"not a torch cache entry")

    restored, _backbone = build_encoder(monkeypatch, spec)
    with pytest.raises(ValueError, match="unreadable"):
        restored(make_values())


def test_deepcopy_does_not_duplicate_memory_entries(monkeypatch):
    encoder, _backbone = build_encoder(
        monkeypatch,
        make_spec(cache=FoundationCacheSpec(memory_capacity=8)),
    )
    encoder(make_values())

    copied = copy.deepcopy(encoder)

    assert encoder.cache_stats["memory_entries"] == 2
    assert copied.cache_stats["memory_entries"] == 0
